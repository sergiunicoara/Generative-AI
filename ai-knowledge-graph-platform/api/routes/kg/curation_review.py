"""Read-only curation view combining pending alias-duplicate candidates and
open contradiction (Conflict) records into one feed.

Roadmap "P1 -- ontology curation and human-in-the-loop workbench", bullet 3:
"Add duplicate and contradiction views backed by the existing entity
resolution, provenance and validation services." Reuses
``ReviewQueueService`` and ``ContradictionDetector`` verbatim -- both
already have working list/read methods over already-persisted Neo4j nodes
(``:ReviewQueueItem``, ``:Conflict``). This module introduces no new
persisted state and no write path of its own: approving/rejecting a
duplicate candidate or resolving a contradiction still happens at their
existing endpoints (``POST /kg/review-queue/{id}/approve|reject``,
``POST /corrections/conflict/resolve``) -- this is a combined view onto
them, not a competing workflow.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from api.auth.dependencies import get_tenant, require_scope
from graphrag.graph.contradiction_detector import ContradictionDetector
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.graph.review_queue import ReviewQueueService

router = APIRouter()


async def build_duplicates_and_contradictions_view(
    *,
    tenant: str,
    limit: int = 50,
    review_service: ReviewQueueService | None = None,
    detector: ContradictionDetector | None = None,
) -> dict:
    """The aggregation logic, with both backing services injectable for
    testing. Runs the two existing, independent read queries concurrently
    and tags each result with which service it came from -- a duplicate
    candidate and a contradiction are different kinds of review item, not
    overlapping records of the same thing, so this deliberately does not
    attempt to merge or deduplicate across the two lists.
    """
    review_service = review_service if review_service is not None else ReviewQueueService()
    detector = detector if detector is not None else ContradictionDetector(get_neo4j())
    duplicates, contradictions = await asyncio.gather(
        review_service.list_pending(tenant, limit),
        detector.get_open_conflicts(limit=limit, tenant=tenant),
    )
    items = (
        [{"kind": "duplicate_candidate", **item} for item in duplicates]
        + [{"kind": "contradiction", **item} for item in contradictions]
    )
    return {
        "tenant": tenant,
        "duplicate_candidate_count": len(duplicates),
        "contradiction_count": len(contradictions),
        "items": items,
    }


@router.get(
    "/curation/duplicates-and-contradictions",
    dependencies=[Depends(require_scope("read"))],
    summary="Unified duplicate-candidate and open-contradiction feed for curation review",
)
async def duplicates_and_contradictions(tenant: str = Depends(get_tenant), limit: int = 50):
    return await build_duplicates_and_contradictions_view(tenant=tenant, limit=limit)
