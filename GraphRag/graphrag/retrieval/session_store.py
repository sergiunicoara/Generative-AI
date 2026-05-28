"""Redis-backed session store with in-memory fallback.

Problem solved
--------------
SessionContext previously held all session history in a Python dict.
Any process restart — worker recycle, deployment update, crash — wiped
every active session.  Users mid-conversation lost all context, causing
the next question to be answered as if it were the first.

Architecture
------------
- Primary: Redis list per session (key = graphrag:session:<session_id>)
  - Each list element is a JSON-serialised SessionTurn
  - List is capped to SESSION_MAX_TURNS via LTRIM after every RPUSH
  - Keys expire after SESSION_TTL_SECONDS (default 24 h)
- Fallback: in-memory deque (identical to the pre-Redis behaviour)
  Used automatically when Redis is unavailable or not configured.

Configuration (config/settings.yml → retrieval):
  session_store: redis        # "memory" | "redis"
  redis_url: redis://localhost:6379/0
  session_ttl_seconds: 86400

Usage::

    store = get_session_store()
    turns = await store.load_turns(session_id)
    await store.save_turn(session_id, new_turn)
    await store.clear(session_id)
"""

from __future__ import annotations

import json
import logging
from collections import deque

import structlog

from graphrag.core.models import SessionTurn

log = structlog.get_logger(__name__)

SESSION_MAX_TURNS  = 10
SESSION_TTL_SECONDS = 86_400   # 24 hours


class SessionStore:
    """
    Persistent session turn store backed by Redis with in-memory fallback.

    All public methods are async to accommodate the Redis I/O path without
    changing the interface for the in-memory path.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        max_turns: int = SESSION_MAX_TURNS,
        ttl_seconds: int = SESSION_TTL_SECONDS,
    ):
        self._max_turns  = max_turns
        self._ttl        = ttl_seconds
        self._redis      = None
        self._memory: dict[str, deque[SessionTurn]] = {}

        if redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(redis_url, decode_responses=True)
                log.info("session_store.redis_connected", url=redis_url)
            except ImportError:
                log.warning(
                    "session_store.redis_not_installed",
                    note="install redis[asyncio] for persistent sessions",
                )
            except Exception as exc:
                log.warning("session_store.redis_connect_failed", error=str(exc))

    # ── Key helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _key(session_id: str) -> str:
        return f"graphrag:session:{session_id}"

    # ── Core operations ────────────────────────────────────────────────────────

    async def load_turns(self, session_id: str) -> deque[SessionTurn]:
        """Return the turn history for a session (most recent last)."""
        if self._redis is not None:
            try:
                raw_list = await self._redis.lrange(self._key(session_id), 0, -1)
                turns: deque[SessionTurn] = deque(maxlen=self._max_turns)
                for raw in raw_list:
                    try:
                        turns.append(SessionTurn(**json.loads(raw)))
                    except Exception:
                        pass   # skip corrupted entries
                return turns
            except Exception as exc:
                log.warning("session_store.redis_load_failed", error=str(exc))
                # Fall through to memory

        return self._memory.get(session_id, deque(maxlen=self._max_turns))

    async def save_turn(self, session_id: str, turn: SessionTurn) -> None:
        """Append a turn to the session history, capping at max_turns."""
        if self._redis is not None:
            try:
                key = self._key(session_id)
                await self._redis.rpush(key, turn.model_dump_json())
                await self._redis.ltrim(key, -self._max_turns, -1)
                await self._redis.expire(key, self._ttl)
                return
            except Exception as exc:
                log.warning("session_store.redis_save_failed", error=str(exc))
                # Fall through to memory

        # Memory fallback
        if session_id not in self._memory:
            self._memory[session_id] = deque(maxlen=self._max_turns)
        self._memory[session_id].append(turn)

    async def clear(self, session_id: str) -> None:
        """Delete all history for a session."""
        if self._redis is not None:
            try:
                await self._redis.delete(self._key(session_id))
                return
            except Exception as exc:
                log.warning("session_store.redis_clear_failed", error=str(exc))

        self._memory.pop(session_id, None)

    def is_redis_backed(self) -> bool:
        return self._redis is not None

    async def ping(self) -> bool:
        """Health-check: returns True if Redis is reachable."""
        if self._redis is None:
            return False
        try:
            return await self._redis.ping()
        except Exception:
            return False


# ── Module-level singleton ─────────────────────────────────────────────────────

_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Return the process-level SessionStore instance (created lazily)."""
    global _store
    if _store is None:
        from graphrag.core.config import get_settings
        cfg  = get_settings()
        ret  = cfg.retrieval
        mode = ret.get("session_store", "memory")
        url  = ret.get("redis_url", "") if mode == "redis" else ""
        ttl  = int(ret.get("session_ttl_seconds", SESSION_TTL_SECONDS))
        max_t = int(ret.get("session_max_turns", SESSION_MAX_TURNS))
        _store = SessionStore(
            redis_url=url or None,
            max_turns=max_t,
            ttl_seconds=ttl,
        )
    return _store
