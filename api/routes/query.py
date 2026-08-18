"""POST /query — publish question to the query queue; GET /query/{id} — poll result.

Results are stored in Redis (via ResultStore) so the API and query worker —
which run as separate containers — share the same result space.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.auth.dependencies import get_tenant, require_scope
from api.limiter import QUERY_LIMIT, limiter
from graphrag.messaging.publishers import publish_query
from graphrag.retrieval.result_store import ResultStoreUnavailable, get_result_store
from graphrag.retrieval.session_store import SessionContextUnavailable, get_session_store

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"       # local | global | hybrid
    ground_truth: str = ""
    session_id: str = ""
    valid_at: str | None = None
    transaction_at: str | None = None
    # Set by the UI from the second message of a conversation onward — never
    # inferred from session_id server-side, since a first message can carry
    # a freshly-generated session_id too. When true, a follow-up that can't
    # get its history is refused (503) rather than silently answered without
    # context. See tasks/lessons.md A156.
    requires_session_context: bool = False


class QueryResponse(BaseModel):
    query_id: str
    status: str = "queued"


@router.post("", response_model=QueryResponse, dependencies=[Depends(require_scope("read"))])
@limiter.limit(QUERY_LIMIT)
async def submit_query(request: Request, body: QueryRequest, tenant: str = Depends(get_tenant)):
    """Submit a question to the async query pipeline.

    Rate-limited to prevent LLM quota exhaustion.
    Default: 60 requests/minute per client IP (override via GRAPHRAG_RATE_LIMIT_QUERY).
    """
    if body.requires_session_context:
        if not body.session_id:
            raise HTTPException(status_code=400,
                detail="requires_session_context requires a session_id")
        # A real read, not a generic ping — the exact operation the worker
        # would later depend on to enrich this follow-up. required=True
        # overrides the store's own strict/non-strict default for this one
        # call: don't enqueue a real ~13-26s LLM retrieval for a follow-up
        # that's already known to be unable to get correct context.
        try:
            await get_session_store().load_turns(
                body.session_id, tenant=tenant, required=True,
            )
        except SessionContextUnavailable as exc:
            raise HTTPException(status_code=503,
                detail=f"Session context unavailable: {exc}")

    from uuid import uuid4
    query_id = str(uuid4())
    # Write "queued" BEFORE publishing — prevents a fast cache-hit in the worker
    # from writing "completed" before this line, which would then get overwritten.
    # If this can't be persisted, don't publish at all: without it, the worker
    # would do a full (expensive, real LLM cost) retrieval and its result would
    # have nowhere durable to land — the client would poll forever for nothing.
    try:
        await get_result_store().set_status(query_id, "queued", tenant)
    except ResultStoreUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Result store unavailable: {exc}")
    try:
        await publish_query(
            question=body.question,
            mode=body.mode,
            ground_truth=body.ground_truth,
            tenant=tenant,
            session_id=body.session_id,
            valid_at=body.valid_at,
            transaction_at=body.transaction_at,
            query_id=query_id,
            correlation_id=request.state.correlation_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}")

    return QueryResponse(query_id=query_id)


@router.get("/{query_id}", dependencies=[Depends(require_scope("read"))])
async def get_query_result(query_id: str, tenant: str = Depends(get_tenant)):
    try:
        result = await get_result_store().get(query_id)
    except ResultStoreUnavailable as exc:
        # Distinguish "storage is down" from "no such query" — a 404 here
        # would be a lie: we don't actually know whether the query exists.
        raise HTTPException(status_code=503, detail=f"Result store unavailable: {exc}")
    if result is None:
        raise HTTPException(status_code=404, detail="Query not found")
    # Ownership check. The result-store key is the query_id alone, so without
    # this any caller holding a "read" scope could fetch any tenant's stored
    # answer and cited source text by id. Guessing a uuid4 was the only
    # barrier, and /kpis/timeseries used to hand out the ids directly.
    #
    # Fails CLOSED: an entry with no recorded tenant (written by a worker from
    # before this field existed) is treated as not-yours rather than
    # not-checked. During a rolling deploy that turns a stale in-flight poll
    # into a 404 — deliberately preferred over serving it unauthorized.
    # 404 rather than 403 so the endpoint doesn't confirm the id exists.
    if result.get("tenant") != tenant:
        raise HTTPException(status_code=404, detail="Query not found")
    return result
