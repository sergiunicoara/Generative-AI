"""§9 — 'ingest -> persist resolved and unresolved results -> complete_with_
review -> reviewer resolves later -> targeted reconciliation.' An unresolved
Mention doesn't block ingestion completion (proven by the fact that ingestion
pipelines already persist PENDING_REVIEW/UNRESOLVED mentions without raising —
see Increment 5's transcript pipeline); this file covers the review-resolves-
later half.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.domain.assertion import Claim
from src.domain.conversation import Mention
from src.domain.enums import AdjudicationStatus, Polarity, ResolutionStatus, SpeakerRole
from src.domain.identity import mention_id, segment_id
from src.graph.repositories.claim_repository import ClaimRepository
from src.graph.repositories.review_repository import ReviewRepository
from src.review.service import ReviewService

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 6, 15, tzinfo=timezone.utc)


def _ws() -> str:
    return f"ws-review-{uuid4().hex[:8]}"


async def test_resolving_a_pending_mention_updates_it_and_records_the_decision(executor):
    workspace_id = _ws()
    review_repo = ReviewRepository(executor)
    service = ReviewService(review_repo)

    seg_id = segment_id("conv-review-1", 0)
    mention = Mention(
        mention_id=mention_id(seg_id, 0, 11, "volks wagen", "ORG"),
        workspace_id=workspace_id, segment_id=seg_id, char_start=0, char_end=11,
        surface_text="Volks Wagen", normalized_surface="volks wagen", entity_type="ORG",
        resolution_status=ResolutionStatus.PENDING_REVIEW,
    )
    await review_repo.upsert_mention(mention)

    decision = await service.resolve(
        workspace_id=workspace_id, mention_id=mention.mention_id, reviewer_id="reviewer@example.com",
        decided_at=_T0, selected_entity_id="account-vw-group", rejected=False,
        candidates_shown=["account-vw-group", "account-vw-financial"],
        original_scores={"account-vw-group": 0.84, "account-vw-financial": 0.50},
        reason="Confirmed via call recording — buyer explicitly named Volkswagen Group.",
    )

    assert decision.selected_entity_id == "account-vw-group"
    assert decision.rejected is False

    updated_mention = await review_repo.get_mention(workspace_id, mention.mention_id)
    assert updated_mention.resolved_entity_id == "account-vw-group"
    assert updated_mention.resolution_status == ResolutionStatus.AUTO_LINKED

    stored_decision = await review_repo.get_review_decision(workspace_id, decision.review_decision_id)
    assert stored_decision is not None
    assert stored_decision.reviewer_id == "reviewer@example.com"
    assert stored_decision.original_scores == {"account-vw-group": 0.84, "account-vw-financial": 0.50}


async def test_rejecting_a_mention_clears_its_resolution(executor):
    workspace_id = _ws()
    review_repo = ReviewRepository(executor)
    service = ReviewService(review_repo)

    seg_id = segment_id("conv-review-2", 0)
    mention = Mention(
        mention_id=mention_id(seg_id, 0, 5, "acme", "ORG"),
        workspace_id=workspace_id, segment_id=seg_id, char_start=0, char_end=5,
        surface_text="Acme", normalized_surface="acme", entity_type="ORG",
        resolution_status=ResolutionStatus.PENDING_REVIEW,
    )
    await review_repo.upsert_mention(mention)

    decision = await service.resolve(
        workspace_id=workspace_id, mention_id=mention.mention_id, reviewer_id="reviewer@example.com",
        decided_at=_T0, selected_entity_id=None, rejected=True,
        candidates_shown=["account-acme-1", "account-acme-2"], original_scores={},
        reason="Neither candidate matches — new account.",
    )

    assert decision.rejected is True
    assert decision.selected_entity_id is None

    updated_mention = await review_repo.get_mention(workspace_id, mention.mention_id)
    assert updated_mention.resolved_entity_id is None
    assert updated_mention.resolution_status == ResolutionStatus.UNRESOLVED


async def test_resolving_a_mention_targets_reconciliation_of_affected_claims_only(executor):
    """§9: 'affected Claims and materialized relationships' — a Claim whose
    subject_id is still the mention's opaque surface gets updated to the
    resolved entity id; claim_id itself never changes."""
    workspace_id = _ws()
    review_repo = ReviewRepository(executor)
    claim_repo = ClaimRepository(executor)
    service = ReviewService(review_repo, claim_repo)

    seg_id = segment_id("conv-review-3", 0)
    mention = Mention(
        mention_id=mention_id(seg_id, 0, 11, "volks wagen", "ORG"),
        workspace_id=workspace_id, segment_id=seg_id, char_start=0, char_end=11,
        surface_text="Volks Wagen", normalized_surface="volks wagen", entity_type="ORG",
        resolution_status=ResolutionStatus.PENDING_REVIEW,
    )
    await review_repo.upsert_mention(mention)

    claim = Claim(
        claim_id="claim-vw-1", workspace_id=workspace_id, subject_id="volks wagen",
        predicate="MENTIONS_ORG", object_value="volkswagen", polarity=Polarity.AFFIRMED,
        source_type="transcript", evidence_char_start=0, evidence_char_end=11,
        source_timestamp=_T0, speaker_role=SpeakerRole.BUYER, confidence=0.8,
        valid_from=_T0, transaction_from=_T0, adjudication_status=AdjudicationStatus.UNREVIEWED,
        retention_class="standard", created_at=_T0,
    )
    await claim_repo.create_claim(claim)

    # an unrelated claim with a different subject must NOT be touched
    other_claim = claim.model_copy(update={
        "claim_id": "claim-other-1", "subject_id": "unrelated-subject",
    })
    await claim_repo.create_claim(other_claim)

    decision = await service.resolve(
        workspace_id=workspace_id, mention_id=mention.mention_id, reviewer_id="reviewer@example.com",
        decided_at=_T0, selected_entity_id="account-vw-group", rejected=False,
        candidates_shown=["account-vw-group"], original_scores={},
    )

    assert decision.affected_claim_ids == ["claim-vw-1"]

    reconciled_claim = await claim_repo.get_claim(workspace_id, "claim-vw-1")
    assert reconciled_claim.subject_id == "account-vw-group"

    untouched_claim = await claim_repo.get_claim(workspace_id, "claim-other-1")
    assert untouched_claim.subject_id == "unrelated-subject"


async def test_resolving_an_unknown_mention_raises():
    from src.graph.repositories.review_repository import ReviewRepository as _ReviewRepository

    service = ReviewService(_ReviewRepository())
    with pytest.raises(ValueError, match="not found"):
        await service.resolve(
            workspace_id="ws-x", mention_id="does-not-exist", reviewer_id="reviewer@example.com",
            decided_at=_T0, selected_entity_id="account-1", rejected=False,
            candidates_shown=[], original_scores={},
        )


async def test_list_pending_returns_only_pending_review_mentions(executor):
    workspace_id = _ws()
    review_repo = ReviewRepository(executor)
    service = ReviewService(review_repo)

    pending = Mention(
        mention_id=mention_id("seg-p", 0, 5, "acme", "ORG"), workspace_id=workspace_id,
        segment_id="seg-p", char_start=0, char_end=5, surface_text="Acme",
        normalized_surface="acme", entity_type="ORG", resolution_status=ResolutionStatus.PENDING_REVIEW,
    )
    resolved = Mention(
        mention_id=mention_id("seg-r", 0, 5, "acme", "ORG"), workspace_id=workspace_id,
        segment_id="seg-r", char_start=0, char_end=5, surface_text="Acme",
        normalized_surface="acme", entity_type="ORG", resolution_status=ResolutionStatus.AUTO_LINKED,
        resolved_entity_id="account-1",
    )
    await review_repo.upsert_mention(pending)
    await review_repo.upsert_mention(resolved)

    pending_list = await service.list_pending(workspace_id)
    ids = {m.mention_id for m in pending_list}
    assert pending.mention_id in ids
    assert resolved.mention_id not in ids
