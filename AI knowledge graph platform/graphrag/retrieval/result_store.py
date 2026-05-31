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
  Used automatically when Redis is unavailable. This regresses to the old
  single-process behaviour, which works fine for development but will
  fail in multi-worker production — the fallback logs a WARNING to make
  this visible.

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

_RESULT_TTL   = int(os.getenv("GRAPHRAG_RESULT_TTL", "3600"))   # 1 hour default
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
        """Persist a query result. Called from the worker after completion."""
        payload = json.dumps(result)
        if self._redis is not None:
            try:
                await self._redis.setex(self._key(query_id), self._ttl, payload)
                return
            except Exception as exc:  # broad: redis.RedisError hierarchy
                log.warning("result_store.redis_write_failed",
                            query_id=query_id, error=str(exc))
        # In-memory fallback
        self._memory[query_id] = result

    async def get(self, query_id: str) -> dict | None:
        """Return the stored result dict, or None if not found / expired."""
        if self._redis is not None:
            try:
                raw = await self._redis.get(self._key(query_id))
                if raw is not None:
                    return json.loads(raw)
                return None
            except Exception as exc:  # broad: redis.RedisError hierarchy
                log.warning("result_store.redis_read_failed",
                            query_id=query_id, error=str(exc))
        return self._memory.get(query_id)

    async def set_status(self, query_id: str, status: str) -> None:
        """Write a lightweight status-only entry (used by the API on enqueue)."""
        await self.set(query_id, {"status": status, "query_id": query_id})

    async def delete(self, query_id: str) -> None:
        """Remove a result entry (optional cleanup)."""
        if self._redis is not None:
            try:
                await self._redis.delete(self._key(query_id))
                return
            except Exception as exc:  # broad: redis.RedisError hierarchy
                log.warning("result_store.redis_delete_failed",
                            query_id=query_id, error=str(exc))
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
