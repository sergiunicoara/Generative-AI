"""Human review queue endpoints — list, approve, and reject ambiguous alias matches."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth.dependencies import require_scope
from graphrag.graph.review_queue import ReviewQueueService

router = APIRouter()


@router.get(
    "/review-queue",
    dependencies=[Depends(require_scope("read"))],
    summary="List pending alias review items",
)
async def list_review_queue(tenant: str = "default", limit: int = 50):
    return {"items": await ReviewQueueService().list_pending(tenant, limit)}


@router.get(
    "/review-queue/all",
    dependencies=[Depends(require_scope("read"))],
    summary="List all alias review items (any status)",
)
async def list_review_queue_all(tenant: str = "default", limit: int = 100):
    return {"items": await ReviewQueueService().list_all(tenant, limit)}


@router.post(
    "/review-queue/{item_id}/approve",
    dependencies=[Depends(require_scope("write"))],
    summary="Approve merge: register raw_name as alias of candidate",
)
async def approve_review_item(
    item_id: str, tenant: str = "default", reviewed_by: str = "human"
):
    return await ReviewQueueService().approve(item_id, reviewed_by, tenant)


@router.post(
    "/review-queue/{item_id}/reject",
    dependencies=[Depends(require_scope("write"))],
    summary="Reject merge: keep entities separate",
)
async def reject_review_item(
    item_id: str, tenant: str = "default", reviewed_by: str = "human"
):
    return await ReviewQueueService().reject(item_id, reviewed_by, tenant)
