"""Shared pytest fixtures for all GraphRAG test suites.

These fixtures are auto-discovered by pytest from any test in any subdirectory
under ``tests/``.  They replace the duplicated fixtures previously defined
inline in ``test_safety_paths.py``, ``test_operational_paths.py``, and the
load tests.

Usage::

    # In any test file — just declare the fixture name as a parameter:
    async def test_foo(neo4j_mock, make_entity, make_chunk):
        ...
"""

from __future__ import annotations

import os

# Must run before any graphrag.core.config import (directly or transitively) --
# ENV defaults to "" (fail-closed, see config.py), and .env is gitignored so
# CI has no ENV at all. Without this, the first get_settings() call anywhere
# in the test suite raises ValueError("ENV is not set..."). setdefault() so a
# real ENV in the environment (e.g. a developer testing prod-like config
# locally) is never silently overridden.
os.environ.setdefault("ENV", "test")
# Do not let a developer's local .env signing key leak into the test process.
# The value is deliberately test-only and long enough for HS256/PyJWT.
os.environ["JWT_SECRET_KEY"] = "test-only-jwt-secret-key-with-at-least-32-bytes"

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag.core.models import Chunk, Entity, Relation, SessionTurn, SourceType


# ── Neo4j client mock ─────────────────────────────────────────────────────────

@pytest.fixture
def neo4j_mock() -> MagicMock:
    """Async-compatible Neo4j client mock.

    ``client.run`` is an AsyncMock that returns an empty list by default.
    Tests can override it with ``neo4j_mock.run = AsyncMock(return_value=[...])``.
    """
    client = MagicMock()
    client.run = AsyncMock(return_value=[])
    return client


# ── Entity / Chunk / Relation factory helpers ─────────────────────────────────

@pytest.fixture
def make_entity():
    """Return a factory that creates ``Entity`` instances with sensible defaults."""
    def _make(
        name: str = "SpaceX",
        etype: str = "ORG",
        tenant: str = "default",
        confidence: float = 0.9,
        **kw,
    ) -> Entity:
        return Entity(name=name, type=etype, tenant=tenant,
                      confidence=confidence, **kw)
    return _make


@pytest.fixture
def make_chunk():
    """Return a factory that creates ``Chunk`` instances with sensible defaults."""
    def _make(
        text: str = "Sample text.",
        tenant: str = "default",
        doc_id: str = "doc_1",
        chunk_index: int = 0,
        **kw,
    ) -> Chunk:
        return Chunk(document_id=doc_id, text=text,
                     chunk_index=chunk_index, tenant=tenant, **kw)
    return _make


@pytest.fixture
def make_relation():
    """Return a factory that creates ``Relation`` instances with sensible defaults."""
    def _make(
        source: str = "SpaceX",
        target: str = "Elon Musk",
        relation: str = "FOUNDED_BY",
        tenant: str = "default",
        confidence: float = 0.85,
        **kw,
    ) -> Relation:
        return Relation(
            source=source,
            target=target,
            relation=relation,
            tenant=tenant,
            confidence=confidence,
            source_type=SourceType.document,
            **kw,
        )
    return _make


@pytest.fixture
def make_turn():
    """Return a factory that creates ``SessionTurn`` instances."""
    def _make(question: str = "q", answer: str = "a") -> SessionTurn:
        return SessionTurn(question=question, answer=answer)
    return _make


# ── Session store helpers ─────────────────────────────────────────────────────

@pytest.fixture
def memory_session_store():
    """A ``SessionStore`` backed by in-memory storage (no Redis needed)."""
    from graphrag.retrieval.session_store import SessionStore
    return SessionStore(redis_url=None)



import pytest as _pytest_audit  # noqa: E402


@_pytest_audit.fixture(autouse=True)
def _no_durable_capability_audit(request, monkeypatch):
    """Unit tests never reach a database for the capability audit trail
    (Phase 7). Tests of the audit sink itself patch or call it explicitly;
    e2e tests (live Neo4j) keep the real sink."""
    if "e2e" in str(getattr(request, "fspath", "")):
        return
    from unittest.mock import AsyncMock
    monkeypatch.setattr("mcp_server.audit.record_audit_event", AsyncMock())
