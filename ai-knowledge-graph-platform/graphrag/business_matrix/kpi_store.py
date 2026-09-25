"""Time-series KPI storage — SQLite (portable, zero external dependency)."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, String, inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class KPIEventRow(Base):
    __tablename__ = "kpi_events"

    # Timescale hypertables require every unique constraint to include the
    # partitioning column. Event IDs remain unique per recorded event while
    # allowing the same schema to serve SQLite and TimescaleDB.
    event_id = Column(String, primary_key=True)
    query_id = Column(String, nullable=False, index=True)
    tenant = Column(String, nullable=False, default="default", index=True)
    recorded_at = Column(DateTime(timezone=True), nullable=False, primary_key=True, index=True)
    latency_ms = Column(Float, nullable=False)
    # NULL = not computed; AVG() skips it instead of averaging in a fake 0.0.
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    context_precision = Column(Float, nullable=True)
    context_recall = Column(Float, nullable=True)
    cost_usd = Column(Float, default=0.0)
    retrieval_mode = Column(String, default="hybrid")
    model_version = Column(String, default="")
    judge_decision = Column(String, default="retrieve")
    judge_confidence = Column(Float, default=0.0)
    judge_accept_threshold = Column(Float, default=0.9)
    judge_retrieve_threshold = Column(Float, default=0.55)
    judge_target_fdr = Column(Float, default=0.05)
    retrieval_used = Column(String, default="true")
    abstention_reason = Column(String, default="")
    evaluation_source = Column(String, default="ragas")
    retrieval_cost_usd = Column(Float, default=0.0)


_engine: AsyncEngine | None = None
_session_factory = None


async def ensure_tenant_column(conn) -> None:
    """Backfill the tenant column for pre-isolation KPI databases."""
    has_tenant = await conn.run_sync(
        lambda sync_conn: "tenant" in {
            column["name"] for column in inspect(sync_conn).get_columns("kpi_events")
        }
    )
    if not has_tenant:
        await conn.execute(
            text("ALTER TABLE kpi_events ADD COLUMN tenant VARCHAR NOT NULL DEFAULT 'default'")
        )
    await conn.execute(
        text("CREATE INDEX IF NOT EXISTS ix_kpi_events_tenant_recorded_at "
             "ON kpi_events (tenant, recorded_at)")
    )
    columns = {
        "judge_decision": "VARCHAR NOT NULL DEFAULT 'retrieve'",
        "judge_confidence": "FLOAT NOT NULL DEFAULT 0",
        "judge_accept_threshold": "FLOAT NOT NULL DEFAULT 0.9",
        "judge_retrieve_threshold": "FLOAT NOT NULL DEFAULT 0.55",
        "judge_target_fdr": "FLOAT NOT NULL DEFAULT 0.05",
        "retrieval_used": "VARCHAR NOT NULL DEFAULT 'true'",
        "abstention_reason": "VARCHAR NOT NULL DEFAULT ''",
        "evaluation_source": "VARCHAR NOT NULL DEFAULT 'ragas'",
        "retrieval_cost_usd": "FLOAT NOT NULL DEFAULT 0",
    }
    existing = await conn.run_sync(
        lambda sync_conn: {
            column["name"] for column in inspect(sync_conn).get_columns("kpi_events")
        }
    )
    for name, definition in columns.items():
        if name not in existing:
            await conn.execute(text(f"ALTER TABLE kpi_events ADD COLUMN {name} {definition}"))


def _get_db_url() -> str:
    timescale_url = os.getenv("TIMESCALE_DB_URL")
    if timescale_url and os.getenv("KPI_BACKEND", "sqlite").lower() == "timescale":
        return timescale_url
    db_path = Path(os.getenv("KPI_DB_PATH", "results/kpi_snapshots/kpis.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{db_path}"


async def get_engine() -> AsyncEngine:
    global _engine, _session_factory
    if _engine is None:
        url = _get_db_url()
        _engine = create_async_engine(url, echo=False)
        _session_factory = sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await ensure_tenant_column(conn)
    return _engine


async def get_session() -> AsyncSession:
    await get_engine()
    return _session_factory()
