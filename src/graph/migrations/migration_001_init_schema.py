"""Idempotent schema bootstrap — every statement uses IF NOT EXISTS, so running
this migration twice (or against a fresh database) is always safe. Exercised
directly by tests/integration/test_index_readiness.py and, once built, by /ready.
"""

from __future__ import annotations

import structlog

from src.graph.execution import GraphExecutor
from src.graph.schema import ALL_STATEMENTS

log = structlog.get_logger(__name__)


async def run(executor: GraphExecutor) -> None:
    for statement in ALL_STATEMENTS:
        await executor.schema_query(statement)
    log.info("migrations.001_init_schema.applied", statement_count=len(ALL_STATEMENTS))
