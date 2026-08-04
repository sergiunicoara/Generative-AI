"""§9 async review — fresh implementation against the P0 Mention/Claim/
ResolutionDecision model, per docs/plan.md's own explicit divergence note: the
legacy src/graph/review_queue.py (kept working in Increment 1) is reference-only
for its enqueue/approve/reject/list_pending/list_all *service shape* — its
ReviewQueueItem (one candidate, no workspace_id) can't hold the full shown
candidate set, component scores, or link to Mention/ResolutionDecision, so this
is a rewrite, not an adaptation.

'ingest -> persist resolved and unresolved results -> complete_with_review ->
reviewer resolves later -> targeted reconciliation' — the ingest/persist/
complete_with_review side is src/resolution/pipeline.py plus the ingestion
pipelines that call it; this module is the 'reviewer resolves later ->
targeted reconciliation' half.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from src.domain.assertion import ReviewDecision
from src.domain.enums import ResolutionStatus
from src.graph.repositories.claim_repository import ClaimRepository
from src.graph.repositories.review_repository import ReviewRepository


def _review_decision_id(mention_id: str, reviewer_id: str, decided_at: datetime) -> str:
    joined = "\x1f".join((mention_id, reviewer_id, decided_at.isoformat()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class ReviewService:
    def __init__(self, review_repo: ReviewRepository, claim_repo: ClaimRepository | None = None):
        self._review_repo = review_repo
        self._claim_repo = claim_repo

    async def list_pending(self, workspace_id: str):
        return await self._review_repo.list_mentions_by_status(workspace_id, ResolutionStatus.PENDING_REVIEW.value)

    async def resolve(
        self,
        *,
        workspace_id: str,
        mention_id: str,
        reviewer_id: str,
        decided_at: datetime,
        selected_entity_id: str | None,
        rejected: bool,
        candidates_shown: list[str],
        original_scores: dict,
        reason: str | None = None,
        previous_review_decision_id: str | None = None,
    ) -> ReviewDecision:
        mention = await self._review_repo.get_mention(workspace_id, mention_id)
        if mention is None:
            raise ValueError(f"mention {mention_id} not found in workspace {workspace_id}")

        new_status = (
            ResolutionStatus.AUTO_LINKED if (not rejected and selected_entity_id) else ResolutionStatus.UNRESOLVED
        )

        # Targeted reconciliation — only this mention's resolved_entity_id and
        # any Claims that used its opaque surface as their subject are touched;
        # nothing else in the graph is re-processed.
        affected_claim_ids: list[str] = []
        if self._claim_repo is not None and not rejected and selected_entity_id:
            affected_claims = await self._claim_repo.list_claims_by_subject(workspace_id, mention.normalized_surface)
            for claim in affected_claims:
                updated_claim = claim.model_copy(update={"subject_id": selected_entity_id})
                await self._claim_repo.create_claim(updated_claim)
                affected_claim_ids.append(claim.claim_id)

        updated_mention = mention.model_copy(update={
            "resolved_entity_id": selected_entity_id if not rejected else None,
            "resolution_status": new_status,
        })
        await self._review_repo.upsert_mention(updated_mention)

        decision = ReviewDecision(
            review_decision_id=_review_decision_id(mention_id, reviewer_id, decided_at),
            workspace_id=workspace_id,
            mention_id=mention_id,
            reviewer_id=reviewer_id,
            decided_at=decided_at,
            selected_entity_id=selected_entity_id if not rejected else None,
            rejected=rejected,
            candidates_shown=candidates_shown,
            original_scores=original_scores,
            reason=reason,
            affected_claim_ids=affected_claim_ids,
            previous_review_decision_id=previous_review_decision_id,
        )
        await self._review_repo.create_review_decision(decision)
        return decision
