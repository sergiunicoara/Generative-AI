"""GET /evaluation endpoints."""

from fastapi import APIRouter, Depends

from api.auth.dependencies import get_tenant
from graphrag.business_matrix.kpi_tracker import KPITracker
from graphrag.evaluation.retrieval_trajectory import (
    RetrievalTrajectoryEvaluationRequest,
    evaluate_retrieval_trajectory,
)

router = APIRouter()


@router.get("/summary")
async def evaluation_summary(window_days: int = 7, tenant: str = Depends(get_tenant)):
    tracker = KPITracker()
    return await tracker.get_summary(tenant=tenant, window_days=window_days)


@router.post("/retrieval-trajectory/score")
async def score_retrieval_trajectory(
    body: RetrievalTrajectoryEvaluationRequest,
    _tenant: str = Depends(get_tenant),
):
    """Score a captured route/evidence trace against a golden expectation."""
    return evaluate_retrieval_trajectory(
        body.trajectory,
        body.expected,
        answer_score=body.answer_score,
    )
