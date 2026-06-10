"""End-to-end tests against live Neo4j and Redis containers.

These tests require ``testcontainers-python`` and Docker.  They are skipped
automatically when either is unavailable so they do not block CI on machines
without Docker.

Install::

    pip install testcontainers

Run::

    pytest tests/e2e/ -v

Or via Make::

    make test-e2e   # (add to Makefile if desired)

Architecture
------------
Each test class spins up isolated containers via testcontainers.  Containers
are started *once per class* (``scope="class"``), so the full setup cost is
paid once regardless of how many tests are in the class.

The Neo4j container uses the official ``neo4j:5`` image.  The Redis container
uses ``redis:7-alpine``.

Isolation
---------
Each test that writes data should use a unique ``tenant`` so tests do not
interfere with each other even when run in the same container instance.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest


# ── Skip guard ─────────────────────────────────────────────────────────────────

def _docker_available() -> bool:
    try:
        import docker
        docker.from_env().ping()
        return True
    except Exception:
        return False


def _testcontainers_available() -> bool:
    try:
        import testcontainers  # noqa: F401
        return True
    except ImportError:
        return False


_SKIP = not (_docker_available() and _testcontainers_available())
_SKIP_REASON = "Docker or testcontainers-python not available"


# ── Neo4j e2e tests ────────────────────────────────────────────────────────────

@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestNeo4jClientE2E:
    """Live Neo4j container — schema init, entity round-trip, retry."""

    @pytest.fixture(scope="class")
    def neo4j_container(self):
        from testcontainers.neo4j import Neo4jContainer
        with Neo4jContainer("neo4j:5") as c:
            yield c

    @pytest.fixture(scope="class")
    def neo4j_url(self, neo4j_container):
        return neo4j_container.get_connection_url()

    async def test_run_simple_query(self, neo4j_url):
        """Basic round-trip: RETURN 1 AS n == 1."""
        from neo4j import AsyncGraphDatabase
        driver = AsyncGraphDatabase.driver(neo4j_url, auth=("neo4j", "password"))
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS n")
            row = await result.single()
            assert row["n"] == 1
        await driver.close()

    async def test_merge_and_match_entity(self, neo4j_url):
        """MERGE an entity node, MATCH it back — data persists within the session."""
        from neo4j import AsyncGraphDatabase
        tenant = f"e2e-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(neo4j_url, auth=("neo4j", "password"))
        async with driver.session() as session:
            await session.run(
                "MERGE (e:Entity {name: $n, type: $t, tenant: $tenant})",
                n="TestCorp", t="ORG", tenant=tenant,
            )
            result = await session.run(
                "MATCH (e:Entity {name: $n, tenant: $tenant}) RETURN e.type AS t",
                n="TestCorp", tenant=tenant,
            )
            row = await result.single()
            assert row["t"] == "ORG"
        await driver.close()


# ── Redis e2e tests ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestSessionStoreE2E:
    """Live Redis container — SessionStore round-trip under real persistence."""

    @pytest.fixture(scope="class")
    def redis_container(self):
        from testcontainers.redis import RedisContainer
        with RedisContainer("redis:7-alpine") as c:
            yield c

    @pytest.fixture(scope="class")
    def redis_url(self, redis_container):
        host = redis_container.get_container_host_ip()
        port = redis_container.get_exposed_port(6379)
        return f"redis://{host}:{port}"

    async def test_save_and_load_turns(self, redis_url):
        """Turns written to Redis are readable back with correct content."""
        from graphrag.retrieval.session_store import SessionStore
        from graphrag.core.models import SessionTurn

        store = SessionStore(redis_url=redis_url, max_turns=10)
        sid = f"e2e-session-{uuid.uuid4().hex[:8]}"

        await store.save_turn(sid, SessionTurn(question="Who?", answer="Alice"))
        await store.save_turn(sid, SessionTurn(question="Where?", answer="Berlin"))

        turns = await store.load_turns(sid)
        assert len(turns) == 2
        questions = {t.question for t in turns}
        assert "Who?" in questions
        assert "Where?" in questions

    async def test_strict_mode_verify_connection(self, redis_url):
        """verify_connection() does not raise when Redis is healthy."""
        from graphrag.retrieval.session_store import SessionStore

        store = SessionStore(redis_url=redis_url, strict=True)
        # Should not raise
        await store.verify_connection()

    async def test_max_turns_sliding_window(self, redis_url):
        """Redis-backed store enforces max_turns via LTRIM."""
        from graphrag.retrieval.session_store import SessionStore
        from graphrag.core.models import SessionTurn

        max_t = 3
        store = SessionStore(redis_url=redis_url, max_turns=max_t)
        sid = f"e2e-window-{uuid.uuid4().hex[:8]}"

        for i in range(max_t + 2):
            await store.save_turn(sid, SessionTurn(question=f"q{i}", answer=f"a{i}"))

        turns = await store.load_turns(sid)
        assert len(turns) == max_t
        questions = {t.question for t in turns}
        assert "q0" not in questions
        assert f"q{max_t + 1}" in questions
