# Forked from ai-knowledge-graph-platform (graphrag/graph/neo4j_client.py) — MINIMAL
# slice only. The original is 1553 lines and includes schema init against graphrag's
# own RAG-ingestion shape (Document/Chunk/Community, CONTEXT_GRAPH_SCHEMA, ~35
# MERGE/query helpers) that nothing in sales-context-graph calls. The 8 legacy
# src/graph/*.py modules ported so far only ever call `self._neo4j.run(...)` and
# `get_neo4j()`, so that is the entire surface forked here.
#
# This is a legacy-compat shim for those 8 modules, NOT the P1 tenant-safe execution
# layer (src/graph/execution.py, tenant_query/schema_query/operational_query per
# docs/plan.md §10) — that is new, sales-specific code built on top of this client,
# not a growth of this file.

"""Neo4j connection pool and thin async query runner."""

from __future__ import annotations

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError

from src.core.config import get_settings
from src.core.retry import with_retry

log = structlog.get_logger(__name__)


class Neo4jClient:
    """Thin wrapper around the Neo4j async driver with retry logic."""

    def __init__(self):
        cfg = get_settings()
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
            max_connection_pool_size=50,
            notifications_min_severity="OFF",
        )

    async def close(self):
        await self._driver.close()

    @with_retry(exceptions=(TransientError, ServiceUnavailable), max_attempts=3)
    async def run(self, cypher: str, **params) -> list[dict]:
        async with self._driver.session() as session:
            result = await session.run(cypher, parameters=params)
            return [record.data() async for record in result]


# ── Module-level singleton ───────────────────────────────────────────────────────

_client: Neo4jClient | None = None


def get_neo4j() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client
