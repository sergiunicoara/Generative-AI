"""GET /kpis endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from api.auth.dependencies import get_current_user, require_scope
from graphrag.business_matrix.kpi_tracker import KPITracker

# These routes read aggregate platform telemetry, so they require the same
# "read" scope every other read surface does. Authentication alone (the prior
# get_current_user-only gate) let any token holder — including a machine
# client issued a deliberately narrow scope set — pull the KPI table.
router = APIRouter(dependencies=[Depends(get_current_user), Depends(require_scope("read"))])


@router.get("/summary")
async def kpi_summary(window_days: int = 7):
    tracker = KPITracker()
    return await tracker.get_summary(window_days=window_days)


@router.get("/timeseries")
async def kpi_timeseries(metric: str = "latency_ms", window_days: int = 7):
    tracker = KPITracker()
    try:
        return await tracker.get_timeseries(metric=metric, window_days=window_days)
    except ValueError as exc:
        # Unsupported metric name — a client error, not a server fault. The
        # allowlist lives in kpi_tracker so the ORM-column selection is
        # constrained at the point of use rather than only at the edge.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
