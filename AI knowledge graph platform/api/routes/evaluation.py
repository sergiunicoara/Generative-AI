"""GET /evaluation endpoints."""

from fastapi import APIRouter
from graphrag.business_matrix.kpi_tracker import KPITracker

router = APIRouter()


@router.get("/summary")
async def evaluation_summary(window_days: int = 7):
    tracker = KPITracker()
    return await tracker.get_summary(window_days=window_days)
