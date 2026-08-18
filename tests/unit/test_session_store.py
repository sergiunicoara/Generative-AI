"""Unit tests for SessionStore.load_turns(required=True) — the per-call
override used by the requires_session_context pre-flight check in
api/routes/query.py. See tasks/lessons.md A156.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.retrieval.session_store import SessionContextUnavailable, SessionStore


@pytest.fixture
def redis_store():
    """SessionStore with a mocked async Redis client."""
    store = SessionStore(redis_url=None)  # don't actually connect
    mock_redis = AsyncMock()
    store._redis = mock_redis
    return store, mock_redis


@pytest.fixture
def memory_store():
    """SessionStore with no Redis configured at all."""
    return SessionStore(redis_url=None)


class TestRequiredTrue:
    async def test_raises_on_redis_failure(self, redis_store):
        store, mock_redis = redis_store
        mock_redis.lrange = AsyncMock(side_effect=ConnectionError("Redis down"))
        with pytest.raises(SessionContextUnavailable):
            await store.load_turns("s1", tenant="acme", required=True)

    async def test_returns_turns_normally_on_success(self, redis_store):
        store, mock_redis = redis_store
        mock_redis.lrange = AsyncMock(return_value=[])
        result = await store.load_turns("s1", tenant="acme", required=True)
        assert list(result) == []

    async def test_does_not_raise_when_redis_never_configured(self, memory_store):
        """required=True only overrides the 'configured but failing' case —
        memory-only mode is a deliberate deployment choice, not a failure."""
        result = await memory_store.load_turns("s1", tenant="acme", required=True)
        assert list(result) == []


class TestRequiredFalseUnchanged:
    """Regression guard: the default (required=False) must preserve every
    existing strict/non-strict behavior exactly as before this change."""

    async def test_non_strict_falls_back_to_memory_on_failure(self, redis_store):
        store, mock_redis = redis_store
        store._strict = False
        mock_redis.lrange = AsyncMock(side_effect=ConnectionError("Redis down"))
        result = await store.load_turns("s1", tenant="acme")  # required defaults to False
        assert list(result) == []  # falls through, does not raise

    async def test_strict_mode_still_raises_via_existing_path(self, redis_store):
        """With required left at its default, strict-mode behavior is
        entirely governed by the module's own _strict flag, unchanged."""
        store, mock_redis = redis_store
        store._strict = True
        mock_redis.lrange = AsyncMock(side_effect=ConnectionError("Redis down"))
        with pytest.raises(ConnectionError):
            await store.load_turns("s1", tenant="acme")


class TestSessionTenantIsolation:
    """Adversarial: session_id is CLIENT-SUPPLIED (api/routes/query.py
    QueryRequest.session_id), so keying session history on it alone let any
    caller read and write another tenant's conversation.

    Read side: SessionContext.enrich_query splices recent entity names from
    stored turns into the outgoing prompt, so tenant B's entity names reached
    tenant A's LLM call. Write side: HybridRetriever records the question AND
    the full answer, depositing A's content into B's history for B to pick up
    on their next follow-up.

    See docs/context_graph_gap_plan.md F11.
    """

    def _turn(self, q, a):
        from graphrag.core.models import SessionTurn
        return SessionTurn(question=q, answer=a)

    async def test_memory_mode_does_not_leak_across_tenants(self, memory_store):
        """Same session_id, two tenants — each sees only its own history."""
        await memory_store.save_turn(
            "shared-id", self._turn("victim q", "victim confidential answer"),
            tenant="victim",
        )
        attacker_turns = await memory_store.load_turns("shared-id", tenant="attacker")
        assert list(attacker_turns) == []

        victim_turns = await memory_store.load_turns("shared-id", tenant="victim")
        assert len(victim_turns) == 1
        assert victim_turns[0].answer == "victim confidential answer"

    async def test_memory_mode_write_does_not_poison_other_tenant(self, memory_store):
        """The write direction: A recording a turn must not append into B's
        history under the same guessed session_id."""
        await memory_store.save_turn("shared-id", self._turn("b q", "b a"), tenant="b")
        await memory_store.save_turn("shared-id", self._turn("a q", "a a"), tenant="a")

        b_turns = await memory_store.load_turns("shared-id", tenant="b")
        assert [t.question for t in b_turns] == ["b q"]

    async def test_clear_is_tenant_scoped(self, memory_store):
        """One tenant clearing a session must not wipe another's."""
        await memory_store.save_turn("shared-id", self._turn("b q", "b a"), tenant="b")
        await memory_store.clear("shared-id", tenant="a")
        assert len(await memory_store.load_turns("shared-id", tenant="b")) == 1

    async def test_redis_key_includes_tenant(self):
        """The Redis key itself must carry the tenant segment — a memory-mode
        pass alone would not prove the deployed (Redis) path is scoped."""
        assert SessionStore._key("s1", "acme") == "graphrag:session:acme:s1"
        assert SessionStore._key("s1", "other") != SessionStore._key("s1", "acme")

    async def test_tenant_containing_colon_cannot_alias_another_pair(self):
        """A tenant named to look like a key boundary must not collide with a
        different (tenant, session) pair."""
        assert SessionStore._key("s1", "a:b") != SessionStore._key("s1", "a")
        assert SessionStore._key("s1", "a:b") != SessionStore._key("b:s1", "a")

    async def test_tenant_is_required_keyword(self):
        """A caller that forgets the tenant must fail loudly, not silently
        read a shared namespace."""
        store = SessionStore(redis_url=None)
        with pytest.raises(TypeError):
            await store.load_turns("s1")
