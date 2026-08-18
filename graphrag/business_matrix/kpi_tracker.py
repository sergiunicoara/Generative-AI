"""Record KPIEvent rows and compute aggregate metrics.

Fixes applied (round-3 audit):
- datetime.utcnow() → datetime.now(timezone.utc) (aware timestamps, Python 3.12-safe)
- p95 latency now computed as a real percentile (not max()) using in-process sort
  over the windowed result set.  SQLite has no PERCENTILE_CONT, so values are
  loaded and the 95th-percentile index is selected in Python.  Acceptable cost
  for a monitoring dashboard that reads once per page refresh.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import select, func

from graphrag.business_matrix.kpi_store import KPIEventRow, get_session
from graphrag.core.models import KPIEvent

log = structlog.get_logger(__name__)

# Columns a caller may plot. Deliberately an allowlist of NUMERIC measurement
# columns, not "every column except the ones we thought of".
#
# get_timeseries() selects its column via getattr(KPIEventRow, metric), so
# whatever string the caller sends names a real ORM column. KPIEventRow also
# carries `query_id`, and the endpoint that exposes this
# (api/routes/kpis.py) had neither a scope nor a tenant filter — and
# KPIEventRow has no tenant column to filter on. So `?metric=query_id`
# returned every tenant's query IDs, which GET /query/{query_id} would then
# redeem for the full stored answer. Restricting the selectable set removes
# the identifier-enumeration half of that chain.
# See docs/context_graph_gap_plan.md F10.
_ALLOWED_TIMESERIES_METRICS = frozenset({
    "latency_ms",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "cost_usd",
})


def _percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile (0–1) of a sorted list, linear interpolation."""
    if not values:
        return 0.0
    values = sorted(values)
    idx = p * (len(values) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] + frac * (values[hi] - values[lo])


class KPITracker:
    async def record(self, kpi: KPIEvent):
        async with await get_session() as session:
            row = KPIEventRow(
                event_id=kpi.event_id,
                query_id=kpi.query_id,
                recorded_at=kpi.recorded_at,
                latency_ms=kpi.latency_ms,
                faithfulness=kpi.faithfulness,
                answer_relevancy=kpi.answer_relevancy,
                context_precision=kpi.context_precision,
                context_recall=kpi.context_recall,
                cost_usd=kpi.cost_usd,
                retrieval_mode=kpi.retrieval_mode,
                model_version=kpi.model_version,
            )
            session.add(row)
            await session.commit()
        log.info("kpi_tracker.recorded", query_id=kpi.query_id)

    async def get_summary(self, window_days: int = 7) -> dict:
        since = datetime.now(timezone.utc) - timedelta(days=window_days)
        async with await get_session() as session:
            # Aggregate metrics (count, avg, min, max)
            agg = await session.execute(
                select(
                    func.count(KPIEventRow.event_id).label("total_queries"),
                    func.avg(KPIEventRow.latency_ms).label("avg_latency_ms"),
                    func.min(KPIEventRow.latency_ms).label("min_latency_ms"),
                    func.max(KPIEventRow.latency_ms).label("max_latency_ms"),
                    func.avg(KPIEventRow.faithfulness).label("avg_faithfulness"),
                    func.avg(KPIEventRow.answer_relevancy).label("avg_answer_relevancy"),
                    func.avg(KPIEventRow.context_precision).label("avg_context_precision"),
                    func.avg(KPIEventRow.context_recall).label("avg_context_recall"),
                ).where(KPIEventRow.recorded_at >= since)
            )
            row = agg.one()

            # Real p50 / p95 — fetch latency values and compute in Python.
            # SQLite has no PERCENTILE_CONT.  Capped at 10 000 rows (ordered
            # by latency so the cap is stable for percentile computation) to
            # bound memory use at high query volumes.  The recorded_at index
            # ensures the WHERE filter doesn't require a full table scan.
            lat_result = await session.execute(
                select(KPIEventRow.latency_ms)
                .where(KPIEventRow.recorded_at >= since)
                .order_by(KPIEventRow.latency_ms)
                .limit(10_000)
            )
            latencies = [r[0] for r in lat_result.all() if r[0] is not None]
            p50 = _percentile(latencies, 0.50)
            p95 = _percentile(latencies, 0.95)

            return {
                "window_days":           window_days,
                "total_queries":         row.total_queries or 0,
                "avg_latency_ms":        round(row.avg_latency_ms or 0, 1),
                "min_latency_ms":        round(row.min_latency_ms or 0, 1),
                "max_latency_ms":        round(row.max_latency_ms or 0, 1),
                "p50_latency_ms":        round(p50, 1),
                "p95_latency_ms":        round(p95, 1),   # true 95th percentile
                "avg_faithfulness":      round(row.avg_faithfulness or 0, 3),
                "avg_answer_relevancy":  round(row.avg_answer_relevancy or 0, 3),
                "avg_context_precision": round(row.avg_context_precision or 0, 3),
                "avg_context_recall":    round(row.avg_context_recall or 0, 3),
            }

    async def get_timeseries(
        self,
        metric: str = "latency_ms",
        window_days: int = 7,
    ) -> list[dict]:
        # Reject rather than silently falling back to latency_ms: a caller
        # asking for a column that isn't plottable has made a mistake worth
        # surfacing, and the old fallback made an identifier request look like
        # a successful latency query.
        if metric not in _ALLOWED_TIMESERIES_METRICS:
            raise ValueError(
                f"unsupported metric {metric!r}; "
                f"allowed: {', '.join(sorted(_ALLOWED_TIMESERIES_METRICS))}"
            )
        since = datetime.now(timezone.utc) - timedelta(days=window_days)
        col = getattr(KPIEventRow, metric)
        async with await get_session() as session:
            result = await session.execute(
                select(KPIEventRow.recorded_at, col)
                .where(KPIEventRow.recorded_at >= since)
                .order_by(KPIEventRow.recorded_at)
            )
            return [
                {"recorded_at": str(r.recorded_at), metric: getattr(r, metric, 0)}
                for r in result.all()
            ]
