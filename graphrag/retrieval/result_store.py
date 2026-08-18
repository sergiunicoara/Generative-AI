"""Cross-process query result store — Redis-backed with in-memory fallback.

Problem solved
--------------
The original query flow wrote results to a module-level dict in the API
process. The query worker runs in a separate container and writes to *its own*
dict — a different process, different memory. Clients polling GET /query/{id}
from the API always saw "queued" because the worker's write never crossed the
process boundary.

Architecture
------------
- Primary: Redis SETEX with configurable TTL (default 1 hour).
  Key pattern: graphrag:result:<query_id>
  Works across any number of API + worker containers sharing the same Redis.

- Fallback: in-memory dict.
  Only used when Redis was never configured at all (no redis_url) — fine
  for single-process development. If Redis *is* configured but a read/write
  fails, `set`/`get` raise `ResultStoreUnavailable` instead of silently
  falling back to memory: in a multi-process deployment a memory fallback
  here is a split-brain (the writer's process has the value, the reader's
  process never will), so callers must decide how to respond (503, retry)
  rather than have this pretend the operation succeeded.

Usage::

    from graphrag.retrieval.result_store import get_result_store

    store = get_result_store()

    # In the worker — write result
    await store.set(query_id, result_dict)

    # In the API — read result
    result = await store.get(query_id)     # None if not found / expired
    await store.delete(query_id)           # optional cleanup
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import structlog

log = structlog.get_logger(__name__)


class ResultStoreUnavailable(Exception):
    """Raised when a configured Redis backend fails to serve a read/write.

    Only raised when Redis was actually configured and the operation itself
    failed — never for the deliberate in-memory-only mode (no redis_url),
    and never for a genuine cache miss (key not found). Callers that need
    to distinguish "storage is down" from "no data" (e.g. GET /query/{id}
    returning 503 vs 404) rely on this distinction.
    """

# TTL is configurable at deploy time without redeploy.
# Precedence: QUERY_RESULT_TTL_SECONDS → GRAPHRAG_RESULT_TTL → YAML retrieval.query_result_ttl_seconds → 3600
_RESULT_TTL = int(
    os.getenv("QUERY_RESULT_TTL_SECONDS")          # ops-friendly name (documented in compose.dev.yaml)
    or os.getenv("GRAPHRAG_RESULT_TTL", "3600")    # legacy env var (kept for backwards compat)
)
_KEY_PREFIX   = "graphrag:result:"


class ResultStore:
    """
    Redis-backed store for async query results.

    Parameters
    ----------
    redis_url : Redis connection URL.  ``None`` → in-memory only.
    ttl       : Key TTL in seconds (applies to Redis only).
    """

    def __init__(self, redis_url: str | None = None, ttl: int = _RESULT_TTL):
        self._ttl     = ttl
        self._redis   = None
        self._memory: dict[str, dict] = {}

        if redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(redis_url, decode_responses=True)
                log.info("result_store.redis_configured", url=redis_url)
            except ImportError:
                log.warning("result_store.redis_not_installed",
                            note="install redis[asyncio] for cross-process results")
            except (OSError, ConnectionError) as exc:
                log.warning("result_store.redis_connect_failed", error=str(exc),
                            note="falling back to in-memory — multi-worker results will not work")
        else:
            log.warning(
                "result_store.no_redis_url",
                note="query results are in-memory — "
                     "GET /query/{id} will not work across multiple workers or restarts",
            )

    # ── Key helper ─────────────────────────────────────────────────────────────

    @staticmethod
    def _key(query_id: str) -> str:
        return f"{_KEY_PREFIX}{query_id}"

    # ── Public API ─────────────────────────────────────────────────────────────

    async def set(self, query_id: str, result: dict) -> None:
        """Persist a query result. Called from the worker after completion.

        Raises ResultStoreUnavailable if Redis is configured but the write
        fails — the caller must decide how to respond (503, retry, etc.)
        rather than this silently pretending the result was saved.
        """
        payload = json.dumps(result)
        if self._redis is not None:
            try:
                await self._redis.setex(self._key(query_id), self._ttl, payload)
            except Exception as exc:  # broad: redis.RedisError hierarchy
                # In a multi-process deployment the result would otherwise live
                # only in this process's memory and the API would never see it
                # — don't fall back to memory, raise so the caller knows.
                log.error("result_store.redis_write_failed",
                          query_id=query_id, error=str(exc),
                          note="result not persisted")
                raise ResultStoreUnavailable(str(exc)) from exc
            return  # regardless of success/fail, don't touch memory
        # In-memory fallback — only active when Redis was never configured
        self._memory[query_id] = result

    async def get(self, query_id: str) -> dict | None:
        """Return the stored result dict, or None if genuinely not found / expired.

        Raises ResultStoreUnavailable if Redis is configured but the read
        fails — distinct from a real cache miss, which returns None.
        """
        if self._redis is not None:
            try:
                raw = await self._redis.get(self._key(query_id))
                if raw is not None:
                    return json.loads(raw)
                return None
            except Exception as exc:  # broad: redis.RedisError hierarchy
                log.error("result_store.redis_read_failed",
                          query_id=query_id, error=str(exc))
                # Don't fall back to memory — wrong process's dict — and don't
                # return None either, since that would look like "not found"
                # when it's really "couldn't check."
                raise ResultStoreUnavailable(str(exc)) from exc
        return self._memory.get(query_id)

    async def set_status(self, query_id: str, status: str, tenant: str) -> None:
        """Write a lightweight status-only entry (used by the API on enqueue).

        ``tenant`` is required, not optional: GET /query/{query_id} authorizes
        the read by comparing the caller's tenant against the one recorded
        here, and a status entry with no tenant would be unreadable by anyone
        (the check fails closed). Making it a required positional parameter
        means a caller that forgets it fails at import/call time rather than
        silently creating an orphaned entry.
        """
        await self.set(query_id, {"status": status, "query_id": query_id, "tenant": tenant})

    async def push_progress(self, query_id: str, step: str) -> None:
        """Append a progress step to an in-flight result (visible to polling clients).

        Deliberately swallows ResultStoreUnavailable: this is a best-effort UI
        nicety mid-retrieval, not the final result. A transient Redis blip here
        must not abort an in-flight query that's already paying real LLM cost —
        unlike the final result write, which the worker does retry (see
        graphrag/messaging/consumers.py).
        """
        try:
            current = await self.get(query_id) or {"status": "processing", "query_id": query_id}
            steps = current.setdefault("steps", [])
            if step not in steps:  # deduplicate — agentic fallback may re-run the retriever
                steps.append(step)
            await self.set(query_id, current)
        except ResultStoreUnavailable as exc:
            log.warning("result_store.push_progress_failed", query_id=query_id, error=str(exc))

    async def delete(self, query_id: str) -> None:
        """Remove a result entry (optional cleanup)."""
        if self._redis is not None:
            try:
                await self._redis.delete(self._key(query_id))
            except Exception as exc:  # broad: redis.RedisError hierarchy
                log.error("result_store.redis_delete_failed",
                          query_id=query_id, error=str(exc))
            return
        self._memory.pop(query_id, None)

    def is_redis_backed(self) -> bool:
        return self._redis is not None


@lru_cache(maxsize=1)
def get_result_store() -> ResultStore:
    """Return the singleton ResultStore, configured from settings."""
    try:
        from graphrag.core.config import get_settings
        cfg = get_settings()
        import os
        redis_url = os.environ.get("REDIS_URL") or cfg.retrieval.get("redis_url", "")
        ttl       = int(cfg.retrieval.get("query_result_ttl_seconds", _RESULT_TTL))
    except Exception:  # noqa: BLE001
        redis_url = ""
        ttl       = _RESULT_TTL
    return ResultStore(redis_url=redis_url or None, ttl=ttl)
