"""GET /evaluation endpoints."""

from fastapi import APIRouter, Depends

from api.auth.dependencies import get_tenant
from graphrag.business_matrix.kpi_tracker import KPITracker
from graphrag.evaluation.retrieval_trajectory import (
    RetrievalStageEvaluationRequest,
    RetrievalTrajectoryEvaluationRequest,
    evaluate_retrieval_stages,
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


@router.post("/retrieval-stages/score")
async def score_retrieval_stages(
    body: RetrievalStageEvaluationRequest,
    _tenant: str = Depends(get_tenant),
):
    """Attribute a golden evaluation failure to an observed retrieval stage."""
    return evaluate_retrieval_stages(
        body.trajectory, body.expected, citations=body.citations, answer_score=body.answer_score,
    )
