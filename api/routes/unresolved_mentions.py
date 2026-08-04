"""§11 required API: GET /api/v1/unresolved-mentions, POST /api/v1/
unresolved-mentions/{id}/resolve. §9: 'The review endpoint is API-only in this
phase. No review UI is required.'
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_workspace_id
from src.graph.execution import GraphExecutor
from src.graph.repositories.claim_repository import ClaimRepository
from src.graph.repositories.review_repository import ReviewRepository
from src.review.service import ReviewService

router = APIRouter(prefix="/api/v1/unresolved-mentions", tags=["review"])


class ResolveMentionRequest(BaseModel):
    reviewer_id: str
    selected_entity_id: str | None = None
    rejected: bool = False
    candidates_shown: list[str] = []
    original_scores: dict = {}
    reason: str | None = None
    previous_review_decision_id: str | None = None


@router.get("")
async def list_unresolved_mentions(workspace_id: str = Depends(get_workspace_id)) -> dict:
    executor = GraphExecutor()
    repo = ReviewRepository(executor)
    mentions = await repo.list_mentions_by_status(workspace_id, "PENDING_REVIEW")
    return {
        "mentions": [
            {
                "mention_id": m.mention_id,
                "segment_id": m.segment_id,
                "surface_text": m.surface_text,
                "normalized_surface": m.normalized_surface,
                "entity_type": m.entity_type,
                "resolution_status": m.resolution_status.value,
            }
            for m in mentions
        ]
    }


@router.post("/{mention_id}/resolve")
async def resolve_mention(mention_id: str, body: ResolveMentionRequest, workspace_id: str = Depends(get_workspace_id)) -> dict:
    if not body.rejected and not body.selected_entity_id:
        raise HTTPException(status_code=422, detail="selected_entity_id is required unless rejected=true")

    executor = GraphExecutor()
    service = ReviewService(ReviewRepository(executor), ClaimRepository(executor))
    try:
        decision = await service.resolve(
            workspace_id=workspace_id,
            mention_id=mention_id,
            reviewer_id=body.reviewer_id,
            decided_at=datetime.now(timezone.utc),
            selected_entity_id=body.selected_entity_id,
            rejected=body.rejected,
            candidates_shown=body.candidates_shown,
            original_scores=body.original_scores,
            reason=body.reason,
            previous_review_decision_id=body.previous_review_decision_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "review_decision_id": decision.review_decision_id,
        "mention_id": decision.mention_id,
        "selected_entity_id": decision.selected_entity_id,
        "rejected": decision.rejected,
        "affected_claim_ids": decision.affected_claim_ids,
    }
