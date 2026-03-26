"""Time-series KPI storage — SQLite (portable, zero external dependency)."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, String
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class KPIEventRow(Base):
    __tablename__ = "kpi_events"

    event_id = Column(String, primary_key=True)
    query_id = Column(String, nullable=False, index=True)
    recorded_at = Column(DateTime, nullable=False)
    latency_ms = Column(Float, nullable=False)
    faithfulness = Column(Float, default=0.0)
    answer_relevancy = Column(Float, default=0.0)
    context_precision = Column(Float, default=0.0)
    context_recall = Column(Float, default=0.0)
    cost_usd = Column(Float, default=0.0)
    retrieval_mode = Column(String, default="hybrid")
    model_version = Column(String, default="")


_engine: AsyncEngine | None = None
_session_factory = None


def _get_db_url() -> str:
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
    return _engine


async def get_session() -> AsyncSession:
    await get_engine()
    return _session_factory()
