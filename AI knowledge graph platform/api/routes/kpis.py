"""GET /kpis endpoints."""

from fastapi import APIRouter, Depends
from api.auth.dependencies import get_current_user
from graphrag.business_matrix.kpi_tracker import KPITracker

router = APIRouter(dependencies=[Depends(get_current_user)])


@router.get("/summary")
async def kpi_summary(window_days: int = 7):
    tracker = KPITracker()
    return await tracker.get_summary(window_days=window_days)


@router.get("/timeseries")
async def kpi_timeseries(metric: str = "latency_ms", window_days: int = 7):
    tracker = KPITracker()
    return await tracker.get_timeseries(metric=metric, window_days=window_days)
