"""POST /query — publish question to the query queue; GET /query/{id} — poll result.

Results are stored in Redis (via ResultStore) so the API and query worker —
which run as separate containers — share the same result space.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.auth.dependencies import require_scope
from api.limiter import QUERY_LIMIT, limiter
from graphrag.messaging.publishers import publish_query
from graphrag.retrieval.result_store import get_result_store

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"       # local | global | hybrid
    ground_truth: str = ""
    tenant: str = "default"
    session_id: str = ""


class QueryResponse(BaseModel):
    query_id: str
    status: str = "queued"


@router.post("", response_model=QueryResponse, dependencies=[Depends(require_scope("read"))])
@limiter.limit(QUERY_LIMIT)
async def submit_query(request: Request, body: QueryRequest):
    """Submit a question to the async query pipeline.

    Rate-limited to prevent LLM quota exhaustion.
    Default: 60 requests/minute per client IP (override via GRAPHRAG_RATE_LIMIT_QUERY).
    """
    from uuid import uuid4
    query_id = str(uuid4())
    # Write "queued" BEFORE publishing — prevents a fast cache-hit in the worker
    # from writing "completed" before this line, which would then get overwritten.
    await get_result_store().set_status(query_id, "queued")
    try:
        await publish_query(
            question=body.question,
            mode=body.mode,
            ground_truth=body.ground_truth,
            tenant=body.tenant,
            session_id=body.session_id,
            query_id=query_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}")

    return QueryResponse(query_id=query_id)


@router.get("/{query_id}", dependencies=[Depends(require_scope("read"))])
async def get_query_result(query_id: str):
    result = await get_result_store().get(query_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Query not found")
    return result
