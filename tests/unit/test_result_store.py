"""Unit tests for ResultStore — Redis-backed cross-process query result store."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.retrieval.result_store import ResultStore, ResultStoreUnavailable


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def memory_store():
    """ResultStore with no Redis — pure in-memory fallback."""
    return ResultStore(redis_url=None)


@pytest.fixture
def redis_store():
    """ResultStore with a mocked async Redis client."""
    store = ResultStore(redis_url=None)  # don't actually connect
    mock_redis = AsyncMock()
    store._redis = mock_redis
    return store, mock_redis


# ── In-memory behaviour ────────────────────────────────────────────────────────

class TestMemoryStore:
    async def test_set_and_get(self, memory_store):
        await memory_store.set("q1", {"status": "completed", "answer": "42"})
        result = await memory_store.get("q1")
        assert result == {"status": "completed", "answer": "42"}

    async def test_get_missing_returns_none(self, memory_store):
        assert await memory_store.get("nonexistent") is None

    async def test_set_status_stores_queued(self, memory_store):
        await memory_store.set_status("q2", "queued", "acme")
        result = await memory_store.get("q2")
        assert result["status"] == "queued"
        assert result["query_id"] == "q2"

    async def test_set_status_records_tenant_for_ownership_check(self, memory_store):
        """GET /query/{id} authorizes by comparing this field to the caller's
        tenant. A status entry without it is unreadable by anyone, so the
        write must always carry it."""
        await memory_store.set_status("q4", "queued", "acme")
        assert (await memory_store.get("q4"))["tenant"] == "acme"

    async def test_delete_removes_entry(self, memory_store):
        await memory_store.set("q3", {"status": "completed"})
        await memory_store.delete("q3")
        assert await memory_store.get("q3") is None

    async def test_delete_nonexistent_no_error(self, memory_store):
        """Deleting a key that does not exist must not raise."""
        await memory_store.delete("ghost")   # should not raise

    async def test_overwrite_replaces_value(self, memory_store):
        await memory_store.set("q4", {"status": "queued"})
        await memory_store.set("q4", {"status": "completed", "answer": "hello"})
        result = await memory_store.get("q4")
        assert result["status"] == "completed"

    def test_is_redis_backed_false_for_memory_store(self, memory_store):
        assert memory_store.is_redis_backed() is False


# ── Redis behaviour ────────────────────────────────────────────────────────────

class TestRedisStore:
    async def test_set_uses_setex(self, redis_store):
        store, mock_redis = redis_store
        mock_redis.setex = AsyncMock(return_value=True)
        await store.set("q1", {"status": "completed"})
        mock_redis.setex.assert_awaited_once()
        # Key should contain the query_id
        call_key = mock_redis.setex.call_args[0][0]
        assert "q1" in call_key

    async def test_get_deserialises_json(self, redis_store):
        import json
        store, mock_redis = redis_store
        mock_redis.get = AsyncMock(return_value=json.dumps({"status": "completed", "answer": "42"}))
        result = await store.get("q1")
        assert result["answer"] == "42"

    async def test_get_returns_none_on_cache_miss(self, redis_store):
        store, mock_redis = redis_store
        mock_redis.get = AsyncMock(return_value=None)
        assert await store.get("missing") is None

    async def test_redis_write_failure_raises_and_does_not_fall_back_to_memory(self, redis_store):
        """When Redis write fails, this must raise ResultStoreUnavailable — not
        silently write to memory, and not silently swallow the error either.
        In a multi-process deployment (API + worker containers), a memory
        fallback here is a split-brain: the writer's process has the value,
        the reader's process never will. The caller (API route, worker retry
        loop) decides how to respond; the store itself must not pretend the
        write succeeded."""
        store, mock_redis = redis_store
        mock_redis.setex = AsyncMock(side_effect=ConnectionError("Redis down"))
        with pytest.raises(ResultStoreUnavailable):
            await store.set("q_fallback", {"status": "completed", "answer": "ok"})
        assert store._memory.get("q_fallback") is None

    async def test_redis_read_failure_raises_not_returns_none(self, redis_store):
        """When Redis read fails, this must raise ResultStoreUnavailable rather
        than returning None — a bare None is indistinguishable from a genuine
        "query not found," which would make GET /query/{id} lie with a 404
        when the real problem is that storage couldn't be checked at all. Must
        also not fall back to this process's memory dict — that memory was
        never populated by whatever process actually wrote the result, so a
        "hit" there would be stale or an unrelated query_id collision."""
        store, mock_redis = redis_store
        # Simulate stale/unrelated local memory state that must NOT be served.
        store._memory["q_mem"] = {"status": "queued", "query_id": "q_mem"}
        mock_redis.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        with pytest.raises(ResultStoreUnavailable):
            await store.get("q_mem")

    async def test_delete_calls_redis_delete(self, redis_store):
        store, mock_redis = redis_store
        mock_redis.delete = AsyncMock(return_value=1)
        await store.delete("q_del")
        mock_redis.delete.assert_awaited_once()

    def test_is_redis_backed_true(self, redis_store):
        store, _ = redis_store
        assert store.is_redis_backed() is True


# ── push_progress: deliberately non-fatal ───────────────────────────────────────

class TestPushProgressNonFatal:
    async def test_push_progress_swallows_unavailable_does_not_raise(self, redis_store):
        """push_progress is a best-effort UI nicety during an in-flight query
        that's already paying real LLM cost — unlike the final result write
        (which the worker retries and ultimately propagates), a transient
        Redis blip here must not abort the retrieval."""
        store, mock_redis = redis_store
        mock_redis.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        await store.push_progress("q_prog", "step one")  # must not raise


# ── Key format ─────────────────────────────────────────────────────────────────

class TestKeyFormat:
    def test_key_includes_prefix_and_id(self):
        key = ResultStore._key("abc123")
        assert key.startswith("graphrag:result:")
        assert "abc123" in key

    def test_different_ids_produce_different_keys(self):
        assert ResultStore._key("a") != ResultStore._key("b")
