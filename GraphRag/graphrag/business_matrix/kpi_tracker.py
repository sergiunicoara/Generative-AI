"""Record KPIEvent rows and compute aggregate metrics."""

from __future__ import annotations

from datetime import datetime, timedelta

import structlog
from sqlalchemy import select, func

from graphrag.business_matrix.kpi_store import KPIEventRow, get_session
from graphrag.core.models import KPIEvent

log = structlog.get_logger(__name__)


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
        since = datetime.utcnow() - timedelta(days=window_days)
        async with await get_session() as session:
            result = await session.execute(
                select(
                    func.count(KPIEventRow.event_id).label("total_queries"),
                    func.avg(KPIEventRow.latency_ms).label("avg_latency_ms"),
                    func.min(KPIEventRow.latency_ms).label("p50_latency_ms"),
                    func.max(KPIEventRow.latency_ms).label("p95_latency_ms"),
                    func.avg(KPIEventRow.faithfulness).label("avg_faithfulness"),
                    func.avg(KPIEventRow.answer_relevancy).label("avg_answer_relevancy"),
                    func.avg(KPIEventRow.context_precision).label("avg_context_precision"),
                    func.avg(KPIEventRow.context_recall).label("avg_context_recall"),
                ).where(KPIEventRow.recorded_at >= since)
            )
            row = result.one()
            return {
                "window_days": window_days,
                "total_queries": row.total_queries or 0,
                "avg_latency_ms": round(row.avg_latency_ms or 0, 1),
                "min_latency_ms": round(row.p50_latency_ms or 0, 1),
                "max_latency_ms": round(row.p95_latency_ms or 0, 1),
                "avg_faithfulness": round(row.avg_faithfulness or 0, 3),
                "avg_answer_relevancy": round(row.avg_answer_relevancy or 0, 3),
                "avg_context_precision": round(row.avg_context_precision or 0, 3),
                "avg_context_recall": round(row.avg_context_recall or 0, 3),
            }

    async def get_timeseries(
        self,
        metric: str = "latency_ms",
        window_days: int = 7,
    ) -> list[dict]:
        since = datetime.utcnow() - timedelta(days=window_days)
        col = getattr(KPIEventRow, metric, KPIEventRow.latency_ms)
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
