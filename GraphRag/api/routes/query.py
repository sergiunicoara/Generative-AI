"""POST /query — publish question to the query queue; GET /query/{id} — poll result."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.dependencies import require_scope
from graphrag.messaging.publishers import publish_query

router = APIRouter()

# Simple in-process result store: query_id -> result dict
_results: dict[str, dict] = {}


def store_result(query_id: str, result: dict) -> None:
    _results[query_id] = result


class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"       # local | global | hybrid
    ground_truth: str = ""
    tenant: str = "default"


class QueryResponse(BaseModel):
    query_id: str
    status: str = "queued"


@router.post("", response_model=QueryResponse, dependencies=[Depends(require_scope("read"))])
async def submit_query(request: QueryRequest):
    try:
        query_id = await publish_query(
            question=request.question,
            mode=request.mode,
            ground_truth=request.ground_truth,
            tenant=request.tenant,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}")

    _results[query_id] = {"status": "queued"}
    return QueryResponse(query_id=query_id)


@router.get("/{query_id}", dependencies=[Depends(require_scope("read"))])
async def get_query_result(query_id: str):
    result = _results.get(query_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Query not found")
    return result
