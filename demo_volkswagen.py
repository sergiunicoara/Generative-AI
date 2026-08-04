#!/usr/bin/env python
"""End-to-end demo: entity resolution (P4) + Context Graph recommendation (P4.5).

Seeds a fresh workspace with the Volkswagen fixture (Volkswagen Group,
Volkswagen Financial Services distractor, a "Volks Wagen" transcript mention,
a seller-owned open Opportunity, an affirmed pricing objection, and two
ContentAssets — one already viewed), then:

1. Resolves "Volks Wagen" via src/resolution/pipeline.py, using a real local
   embedding provider (src/embedding/sentence_transformer_provider.py — no
   API key, no network call beyond the one-time model download) for the
   semantic score, printing every candidate considered, each one's component
   scores (lexical/semantic/base/rel_bonus/final), the named relational
   signals that fired, the top-1/top-2 margin, and the final
   AUTO_LINKED/PENDING_REVIEW/UNRESOLVED status.
2. Runs the objection-to-content recommendation use case (§12) and prints the
   recommended (unviewed) asset with its exact transcript evidence and Claim id.

Requires docker-compose's neo4j service running (`docker compose up -d neo4j`;
see docker-compose.yml's neo4j service comment for the non-default host port).

Usage:
    python demo_volkswagen.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from src.core.neo4j_client import Neo4jClient
from src.embedding.sentence_transformer_provider import SentenceTransformerEmbeddingProvider
from src.domain.identity import crm_entity_id, mention_id, segment_id
from src.domain.conversation import Mention
from src.domain.knowledge import AssetView, ContentAsset
from src.extraction.fixture_provider import FixtureExtractionProvider
from src.graph.execution import GraphExecutor
from src.graph.migrations.migration_001_init_schema import run as run_migration
from src.graph.repositories.claim_repository import ClaimRepository
from src.graph.repositories.content_repository import ContentRepository
from src.graph.repositories.conversation_repository import ConversationRepository
from src.graph.repositories.crm_repository import CrmRepository
from src.graph.repositories.source_repository import SourceRepository
from src.ingestion.adapters.gong import GongAdapter
from src.ingestion.adapters.salesforce import SalesforceAdapter
from src.ingestion.pipeline import CrmIngestionPipeline
from src.ingestion.transcript_pipeline import TranscriptIngestionPipeline
from src.resolution.candidates import CandidateGenerator
from src.resolution.pipeline import gather_relational_signals, resolve_mention
from src.usecases.objection_content_recommendation import ObjectionContentRecommendationUseCase

_T0 = datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc)


def _hr(title: str) -> None:
    # Plain ASCII only — Windows consoles without UTF-8 configured (cp1252)
    # mangle em-dashes and other non-ASCII characters in print() output.
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def main() -> None:
    client = Neo4jClient()
    executor = GraphExecutor(client)
    await run_migration(executor)

    workspace_id = f"demo-vw-{uuid4().hex[:8]}"
    print(f"workspace_id = {workspace_id}")

    crm_repo = CrmRepository(executor)
    conv_repo = ConversationRepository(executor)
    claim_repo = ClaimRepository(executor)
    content_repo = ContentRepository(executor)
    source_repo = SourceRepository(executor)
    candidate_generator = CandidateGenerator(executor)

    # ── Seed: Volkswagen Group + distractor, seller-owned open Opportunity ──
    crm_pipeline = CrmIngestionPipeline(crm_repo, source_repo, SalesforceAdapter())
    await crm_pipeline.ingest_accounts(
        workspace_id,
        [
            {"Id": "001VWGROUP", "Name": "Volkswagen Group", "Website": "vw.com", "IsDeleted": False, "MasterRecordId": None},
            {"Id": "001VWFIN", "Name": "Volkswagen Financial Services", "Website": "vwfs.com", "IsDeleted": False, "MasterRecordId": None},
        ],
        ingestion_run_id="demo-run", observed_at=_T0,
    )
    vw_group_id = crm_entity_id(workspace_id, "salesforce", "Account", "001VWGROUP")
    vw_financial_id = crm_entity_id(workspace_id, "salesforce", "Account", "001VWFIN")

    await crm_pipeline.ingest_contacts(
        workspace_id,
        [{"Id": "003ELENA", "AccountId": "001VWGROUP", "Name": "Elena Popescu", "Email": "elena.popescu@vw.com", "IsDeleted": False}],
        ingestion_run_id="demo-run", observed_at=_T0,
    )
    elena_id = crm_entity_id(workspace_id, "salesforce", "Contact", "003ELENA")

    await crm_pipeline.ingest_opportunities(
        workspace_id,
        [{"Id": "006VWDEAL", "Name": "VW Group Renewal", "AccountId": "001VWGROUP", "OwnerId": "005SAM",
          "StageName": "Negotiation", "IsClosed": False, "IsDeleted": False}],
        ingestion_run_id="demo-run", observed_at=_T0,
    )
    opportunity_id = crm_entity_id(workspace_id, "salesforce", "Opportunity", "006VWDEAL")
    # Opportunity.seller_id is stored as the canonical hashed id (see
    # src/ingestion/adapters/salesforce.py::parse_opportunity) — relational
    # signal lookups must match on that, never the raw external OwnerId.
    seller_id = crm_entity_id(workspace_id, "salesforce", "Seller", "005SAM")

    # ── Transcript: "Volks Wagen" mention + affirmed pricing objection ──
    transcript_pipeline = TranscriptIngestionPipeline(
        conv_repo, source_repo, claim_repo, GongAdapter(), FixtureExtractionProvider()
    )
    raw_call = {
        "id": "call-vw-demo", "started": "2026-06-15T14:00:00Z", "deleted": False,
        "parties": [
            {"speakerId": "spk_1", "name": "Elena Popescu", "emailAddress": "elena.popescu@vw.com"},
            {"speakerId": "spk_2", "name": "Sam Seller", "emailAddress": "sam@ourcompany.com"},
        ],
        "transcript": [
            {"speakerId": "spk_1", "sentences": [
                {"text": "This is Volks Wagen calling, and we are concerned about pricing this quarter.", "start": 0, "end": 4000},
            ]},
            {"speakerId": "spk_2", "sentences": [
                {"text": "Understood — let's review the numbers together.", "start": 4000, "end": 7000},
            ]},
        ],
    }
    transcript_result = await transcript_pipeline.ingest_call(
        workspace_id, raw_call, ingestion_run_id="demo-run", observed_at=_T0,
        opportunity_id=opportunity_id, account_id=vw_group_id,
        email_to_contact_id={"elena.popescu@vw.com": elena_id},
        email_to_seller_id={"sam@ourcompany.com": "005SAM"},
    )
    conversation_id = transcript_result.conversation_id

    # ── Content: two assets addressing "pricing", one already viewed ──
    viewed_asset = ContentAsset(
        content_asset_id="asset-pricing-guide-demo", workspace_id=workspace_id,
        title="Pricing Objection Handling Guide", url="https://showpad.example/pricing-guide",
        tags=["pricing", "objection"],
    )
    unviewed_asset = ContentAsset(
        content_asset_id="asset-roi-calculator-demo", workspace_id=workspace_id,
        title="Enterprise Pricing ROI Calculator", url="https://showpad.example/roi-calculator",
        tags=["pricing", "roi"],
    )
    await content_repo.upsert_content_asset(viewed_asset)
    await content_repo.upsert_content_asset(unviewed_asset)
    await content_repo.upsert_asset_view(AssetView(
        asset_view_id="view-demo-1", workspace_id=workspace_id,
        content_asset_id=viewed_asset.content_asset_id, viewer_contact_id=elena_id, viewed_at=_T0,
    ))

    # ══════════════════════════ Part 1: entity resolution ══════════════════════════
    _hr("PART 1 - Entity resolution: 'Volks Wagen'")

    seg_id = segment_id(conversation_id, 0)
    mention = Mention(
        mention_id=mention_id(seg_id, 15, 26, "volks wagen", "ORG"),
        workspace_id=workspace_id, segment_id=seg_id, char_start=15, char_end=26,
        surface_text="Volks Wagen", normalized_surface="volks wagen", entity_type="ORG",
    )

    signals = await gather_relational_signals(
        candidate_generator, workspace_id=workspace_id,
        conversation_id=conversation_id, seller_id=seller_id, participant_email_domain="vw.com",
    )

    print("\nLoading local embedding model (all-MiniLM-L6-v2, ~80MB, one-time download if not cached)...")
    embedding_provider = SentenceTransformerEmbeddingProvider()

    outcome = await resolve_mention(
        workspace_id=workspace_id, mention=mention, entity_type="Account",
        candidate_generator=candidate_generator, decided_at=_T0,
        relational_signals_by_entity=signals, embedding_provider=embedding_provider,
    )

    print(f"\nMention: {mention.surface_text!r} (mention_id={mention.mention_id[:16]}...)")
    print(f"Candidates shown ({len(outcome.candidates_shown)}):")
    names_by_id = {vw_group_id: "Volkswagen Group", vw_financial_id: "Volkswagen Financial Services"}
    for entity_id in outcome.candidates_shown:
        label = names_by_id.get(entity_id, entity_id[:16] + "...")
        marker = " <-- resolved" if entity_id == outcome.decision.resolved_entity_id else ""
        print(f"  - {label} ({entity_id[:16]}...){marker}")

    print("\nTop candidate component scores:")
    print(f"  lexical         = {outcome.decision.lexical_score}")
    print(f"  semantic        = {outcome.decision.semantic_score}")
    print(f"  base            = {outcome.decision.base_score}")
    print(f"  relational_bonus= {outcome.decision.relational_bonus}")
    print(f"  final           = {outcome.decision.final_score}")
    print(f"  margin (top1-top2) = {outcome.decision.margin}")
    print(f"  relational_signals  = {outcome.decision.relational_signals}")
    print(f"\n>>> STATUS: {outcome.decision.status.value}")
    if outcome.decision.resolved_entity_id:
        print(f">>> Resolved to: {names_by_id.get(outcome.decision.resolved_entity_id)}")

    # ══════════════════════════ Part 2: content recommendation ══════════════════════════
    _hr("PART 2 - Objection-to-content recommendation")

    use_case = ObjectionContentRecommendationUseCase(conv_repo, claim_repo, content_repo)
    recommendation = await use_case.recommend(workspace_id, opportunity_id, elena_id)

    print(f"\nObjection Claim: {recommendation.objection_claim.claim_id}")
    print(f"  predicate = {recommendation.objection_claim.predicate}")
    print(f"  object    = {recommendation.objection_claim.object_value}")
    print(f"  evidence  = {recommendation.evidence_text!r}")
    print(f"  speaker_role = {recommendation.objection_claim.speaker_role.value}")

    print(f"\nMapping source: {recommendation.mapping_source}")
    print(f"Excluded (already viewed): {recommendation.excluded_viewed_asset_ids}")
    print("\nRanked candidates:")
    for ranked in recommendation.ranked_candidates:
        print(f"  - {ranked.asset.title} (score={ranked.rank_score}, tags={ranked.matched_tags})")

    print(f"\n>>> RECOMMENDED ASSET: {recommendation.recommended_asset.title if recommendation.recommended_asset else None}")
    print(f">>> Explanation: {recommendation.explanation}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
