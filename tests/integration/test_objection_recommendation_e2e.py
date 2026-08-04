"""§16 P4.5 exit criterion — 'end-to-end demo returns unviewed content with
exact evidence and selection reasons.' Builds the fixture through the real
ingestion pipelines (CRM + transcript + content), not hand-crafted graph nodes,
so this is a genuine end-to-end proof: Elena Popescu as Contact/participant, a
seller-owned open Opportunity, an affirmed pricing objection, two ContentAssets
addressing it, one already viewed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.domain.identity import crm_entity_id
from src.domain.knowledge import AssetView, ContentAsset
from src.extraction.fixture_provider import FixtureExtractionProvider
from src.graph.repositories.claim_repository import ClaimRepository
from src.graph.repositories.content_repository import ContentRepository
from src.graph.repositories.conversation_repository import ConversationRepository
from src.graph.repositories.crm_repository import CrmRepository
from src.graph.repositories.source_repository import SourceRepository
from src.ingestion.adapters.gong import GongAdapter
from src.ingestion.adapters.salesforce import SalesforceAdapter
from src.ingestion.pipeline import CrmIngestionPipeline
from src.ingestion.transcript_pipeline import TranscriptIngestionPipeline
from src.usecases.objection_content_recommendation import (
    NoObjectionFoundError,
    NoRelevantCallError,
    ObjectionContentRecommendationUseCase,
)

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc)


async def test_objection_recommendation_end_to_end_excludes_viewed_asset(executor):
    workspace_id = f"ws-demo-{uuid4().hex[:8]}"
    crm_repo = CrmRepository(executor)
    conv_repo = ConversationRepository(executor)
    claim_repo = ClaimRepository(executor)
    content_repo = ContentRepository(executor)
    source_repo = SourceRepository(executor)

    # ── CRM: Account, seller-owned open Opportunity, Elena Popescu Contact ──
    crm_pipeline = CrmIngestionPipeline(crm_repo, source_repo, SalesforceAdapter())
    await crm_pipeline.ingest_accounts(
        workspace_id,
        [{"Id": "001ACME", "Name": "Acme Corp", "Website": "acme.com", "IsDeleted": False, "MasterRecordId": None}],
        ingestion_run_id="run-crm", observed_at=_T0,
    )
    await crm_pipeline.ingest_contacts(
        workspace_id,
        [{"Id": "003ELENA", "AccountId": "001ACME", "Name": "Elena Popescu", "Email": "elena.popescu@acme.com", "IsDeleted": False}],
        ingestion_run_id="run-crm", observed_at=_T0,
    )
    await crm_pipeline.ingest_opportunities(
        workspace_id,
        [{"Id": "006DEAL", "Name": "Acme Renewal", "AccountId": "001ACME", "OwnerId": "005SAM",
          "StageName": "Negotiation", "IsClosed": False, "IsDeleted": False}],
        ingestion_run_id="run-crm", observed_at=_T0,
    )
    opportunity_id = crm_entity_id(workspace_id, "salesforce", "Opportunity", "006DEAL")
    elena_contact_id = crm_entity_id(workspace_id, "salesforce", "Contact", "003ELENA")

    # ── Transcript: buyer raises a pricing objection, references Showpad Genie ──
    transcript_pipeline = TranscriptIngestionPipeline(
        conv_repo, source_repo, claim_repo, GongAdapter(), FixtureExtractionProvider()
    )
    raw_call = {
        "id": "call-demo-1", "started": "2026-06-15T14:00:00Z", "deleted": False,
        "parties": [
            {"speakerId": "spk_1", "name": "Elena Popescu", "emailAddress": "elena.popescu@acme.com"},
            {"speakerId": "spk_2", "name": "Sam Seller", "emailAddress": "sam@ourcompany.com"},
        ],
        "transcript": [
            {"speakerId": "spk_1", "sentences": [
                {"text": "We are concerned about pricing this quarter.", "start": 0, "end": 3000},
            ]},
            {"speakerId": "spk_2", "sentences": [
                {"text": "Let me show you how Showpad Genie and Shared Spaces help justify the ROI.", "start": 3000, "end": 7000},
            ]},
        ],
    }
    transcript_result = await transcript_pipeline.ingest_call(
        workspace_id, raw_call, ingestion_run_id="run-transcript", observed_at=_T0,
        opportunity_id=opportunity_id, account_id="001ACME",
        email_to_contact_id={"elena.popescu@acme.com": elena_contact_id},
        email_to_seller_id={"sam@ourcompany.com": "005SAM"},
    )
    assert transcript_result.claims_created > 0

    # ── Content: two assets addressing "pricing", one already viewed ──
    viewed_asset = ContentAsset(
        content_asset_id="asset-pricing-guide", workspace_id=workspace_id,
        title="Pricing Objection Handling Guide", url="https://showpad.example/pricing-guide",
        tags=["pricing", "objection"],
    )
    unviewed_asset = ContentAsset(
        content_asset_id="asset-roi-calculator", workspace_id=workspace_id,
        title="Enterprise Pricing ROI Calculator", url="https://showpad.example/roi-calculator",
        tags=["pricing", "roi"],
    )
    await content_repo.upsert_content_asset(viewed_asset)
    await content_repo.upsert_content_asset(unviewed_asset)
    await content_repo.upsert_asset_view(AssetView(
        asset_view_id="view-1", workspace_id=workspace_id,
        content_asset_id=viewed_asset.content_asset_id, viewer_contact_id=elena_contact_id,
        viewed_at=_T0,
    ))

    # ── Recommendation ──
    use_case = ObjectionContentRecommendationUseCase(conv_repo, claim_repo, content_repo)
    recommendation = await use_case.recommend(workspace_id, opportunity_id, elena_contact_id)

    assert recommendation.recommended_asset is not None
    assert recommendation.recommended_asset.content_asset_id == unviewed_asset.content_asset_id
    assert viewed_asset.content_asset_id in recommendation.excluded_viewed_asset_ids
    assert recommendation.recommended_asset.content_asset_id not in recommendation.excluded_viewed_asset_ids

    # every factual item cites a served Claim (§15 grounding requirement)
    assert recommendation.objection_claim.predicate == "RAISED_OBJECTION"
    assert recommendation.objection_claim.object_value == "pricing"
    assert "pricing" in recommendation.evidence_text.lower()
    assert recommendation.mapping_source == "content_asset.tags (curated Showpad content taxonomy)"
    assert recommendation.objection_claim.claim_id in recommendation.explanation


async def test_no_conversation_raises_no_relevant_call_error(executor):
    workspace_id = f"ws-demo-empty-{uuid4().hex[:8]}"
    conv_repo = ConversationRepository(executor)
    claim_repo = ClaimRepository(executor)
    content_repo = ContentRepository(executor)
    use_case = ObjectionContentRecommendationUseCase(conv_repo, claim_repo, content_repo)

    with pytest.raises(NoRelevantCallError):
        await use_case.recommend(workspace_id, "opp-does-not-exist", "contact-x")


async def test_conversation_with_no_objection_raises(executor):
    workspace_id = f"ws-demo-noobj-{uuid4().hex[:8]}"
    crm_repo = CrmRepository(executor)
    conv_repo = ConversationRepository(executor)
    claim_repo = ClaimRepository(executor)
    content_repo = ContentRepository(executor)
    source_repo = SourceRepository(executor)

    crm_pipeline = CrmIngestionPipeline(crm_repo, source_repo, SalesforceAdapter())
    await crm_pipeline.ingest_opportunities(
        workspace_id,
        [{"Id": "006NOOBJ", "Name": "Quiet Deal", "AccountId": "001QUIET", "OwnerId": "005QUIET",
          "StageName": "Discovery", "IsClosed": False, "IsDeleted": False}],
        ingestion_run_id="run-crm", observed_at=_T0,
    )
    opportunity_id = crm_entity_id(workspace_id, "salesforce", "Opportunity", "006NOOBJ")

    transcript_pipeline = TranscriptIngestionPipeline(
        conv_repo, source_repo, claim_repo, GongAdapter(), FixtureExtractionProvider()
    )
    await transcript_pipeline.ingest_call(
        workspace_id,
        {"id": "call-quiet", "started": "2026-06-15T14:00:00Z", "deleted": False,
         "parties": [{"speakerId": "spk_1"}],
         "transcript": [{"speakerId": "spk_1", "sentences": [{"text": "Nice weather today.", "start": 0, "end": 1000}]}]},
        ingestion_run_id="run-transcript", observed_at=_T0, opportunity_id=opportunity_id,
    )

    use_case = ObjectionContentRecommendationUseCase(conv_repo, claim_repo, content_repo)
    with pytest.raises(NoObjectionFoundError):
        await use_case.recommend(workspace_id, opportunity_id, "contact-x")
