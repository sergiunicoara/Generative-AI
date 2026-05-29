"""Forward-chaining inference engine and counterfactual analysis endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import require_scope
from graphrag.graph.neo4j_client import get_neo4j

router = APIRouter()


# ── Forward-Chaining Inference Engine ────────────────────────────────────────

class InferenceRunRequest(BaseModel):
    tenant: str = "default"
    max_iterations: int = 5
    dry_run: bool = False


class InferenceDocRequest(BaseModel):
    doc_id: str
    tenant: str = "default"


@router.post(
    "/inference/run",
    dependencies=[Depends(require_scope("write"))],
    summary="Run forward-chaining inference rules and materialise inferred edges",
)
async def run_inference(request: InferenceRunRequest):
    from graphrag.graph.inference_engine import ForwardChainingEngine
    engine = ForwardChainingEngine(get_neo4j())
    return await engine.run(
        tenant=request.tenant,
        max_iterations=request.max_iterations,
        dry_run=request.dry_run,
    )


@router.post(
    "/inference/run-for-document",
    dependencies=[Depends(require_scope("write"))],
    summary="Run forward-chaining rules scoped to a single document",
)
async def run_inference_for_doc(request: InferenceDocRequest):
    from graphrag.graph.inference_engine import ForwardChainingEngine
    engine = ForwardChainingEngine(get_neo4j())
    return await engine.run_for_document(
        doc_id=request.doc_id,
        tenant=request.tenant,
    )


# ── Counterfactual Analysis ───────────────────────────────────────────────────

@router.get(
    "/counterfactual/simulate/{doc_id}",
    dependencies=[Depends(require_scope("read"))],
    summary="Simulate the impact of removing a document without modifying data",
)
async def simulate_retraction(doc_id: str, tenant: str = "default"):
    from graphrag.graph.counterfactual import CounterfactualAnalyzer
    analyzer = CounterfactualAnalyzer(get_neo4j())
    return await analyzer.simulate_retraction(doc_id=doc_id, tenant=tenant)
