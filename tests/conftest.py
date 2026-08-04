"""Integration tests require docker-compose's neo4j service running locally
(host port 7688 — see docker-compose.yml's neo4j service comment for why not the
Neo4j default 7687). `docker compose up -d neo4j` before running this suite.
"""

from __future__ import annotations

import pytest

import src.core.neo4j_client as neo4j_client_module
from src.core.config import get_settings
from src.core.neo4j_client import Neo4jClient
from src.embedding.sentence_transformer_provider import SentenceTransformerEmbeddingProvider
from src.graph.execution import GraphExecutor
from src.graph.migrations.migration_001_init_schema import run as run_migration


@pytest.fixture(scope="session")
def embedding_provider() -> SentenceTransformerEmbeddingProvider:
    """Model loading (~1-2s) happens once per test session, not once per test."""
    return SentenceTransformerEmbeddingProvider()


@pytest.fixture(autouse=True)
def _neo4j_test_env(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7688")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "scg_dev_local")
    get_settings.cache_clear()

    # pytest-asyncio gives each test function its own event loop by default,
    # but src.core.neo4j_client.get_neo4j() is a process-wide singleton — a
    # driver created in one test's loop breaks when reused from a later test's
    # (different, and by then closed) loop. API routes call get_neo4j()
    # directly (not the per-test `executor` fixture below), so drop the
    # singleton here and let each test build its own, bound to its own loop.
    # This is a test-isolation artifact only; a real running process has
    # exactly one event loop, so the singleton is correct there.
    neo4j_client_module._client = None
    yield
    neo4j_client_module._client = None
    get_settings.cache_clear()


@pytest.fixture
async def executor(_neo4j_test_env):
    client = Neo4jClient()
    ex = GraphExecutor(client)
    await run_migration(ex)
    yield ex
    await client.close()
