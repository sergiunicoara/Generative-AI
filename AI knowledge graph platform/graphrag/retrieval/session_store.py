"""Redis-backed session store with in-memory fallback and optional strict mode.

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
  Used automatically when Redis is unavailable or not configured,
  UNLESS strict mode is enabled (see below).

Strict mode
-----------
Set ``session_store_strict: true`` in config (retrieval section) to
switch from graceful degradation to fail-fast behaviour:

  - Import failure  (``redis[asyncio]`` package missing)
    → raises ``ImportError`` at process start (caught by lifespan hook)
  - Connection failure (Redis unreachable at startup)
    → ``verify_connection()`` raises ``ConnectionError``
    → FastAPI lifespan hook aborts startup so the issue is visible
  - Per-operation failure (transient blip during a live request)
    → logged at ERROR level (not WARNING) but still falls back to memory
      so in-flight requests are not dropped mid-answer

Rationale: killing a live request mid-stream on a transient Redis blip
causes worse user experience than degrading to memory for that one turn.
The strict/non-strict distinction therefore applies at startup time only.

Configuration (config/settings.yml → retrieval):
  session_store: redis             # "memory" | "redis"
  redis_url: redis://localhost:6379/0
  session_ttl_seconds: 86400
  session_store_strict: true       # fail-fast if Redis unavailable at startup

Usage::

    store = get_session_store()
    await store.verify_connection()   # call once at startup (FastAPI lifespan)
    turns = await store.load_turns(session_id)
    await store.save_turn(session_id, new_turn)
    await store.clear(session_id)
"""

from __future__ import annotations

import json
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

    Parameters
    ----------
    redis_url:
        Connection URL for Redis.  ``None`` → memory-only mode.
    max_turns:
        Maximum turns kept per session (sliding window).
    ttl_seconds:
        Redis key TTL in seconds.
    strict:
        When ``True``, a missing ``redis[asyncio]`` package or a failed
        connection attempt at init raises immediately instead of falling
        back silently.  Per-operation failures during live requests still
        fall back to memory but are logged at ERROR level.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        max_turns: int = SESSION_MAX_TURNS,
        ttl_seconds: int = SESSION_TTL_SECONDS,
        strict: bool = False,
    ):
        self._max_turns  = max_turns
        self._ttl        = ttl_seconds
        self._strict     = strict
        self._redis      = None
        self._memory: dict[str, deque[SessionTurn]] = {}

        if redis_url:
            try:
                import redis.asyncio as aioredis
            except ImportError as exc:
                msg = (
                    "redis[asyncio] is not installed but session_store=redis. "
                    "Install it with: pip install redis[asyncio]"
                )
                if strict:
                    raise ImportError(msg) from exc
                log.warning(
                    "session_store.redis_not_installed",
                    note="install redis[asyncio] for persistent sessions",
                )
                return

            try:
                self._redis = aioredis.from_url(redis_url, decode_responses=True)
                log.info("session_store.redis_configured", url=redis_url)
            except Exception as exc:
                if strict:
                    raise ConnectionError(
                        f"Failed to create Redis client for {redis_url}: {exc}"
                    ) from exc
                log.warning("session_store.redis_connect_failed", error=str(exc))

    # ── Startup verification ───────────────────────────────────────────────────

    async def verify_connection(self) -> None:
        """
        Verify that the Redis connection is live.

        Call this once during application startup (e.g. FastAPI lifespan).
        - Non-strict mode: logs a warning on failure, continues.
        - Strict mode: raises ``ConnectionError`` on failure so the process
          exits with a clear error rather than serving sessions from memory
          without the operator knowing.
        """
        if self._redis is None:
            # Either memory-only mode or import failed — nothing to ping
            return

        try:
            ok = await self._redis.ping()
            if ok:
                log.info("session_store.redis_ping_ok")
                return
            # ping() returned False
            raise ConnectionError("Redis PING returned False")
        except ConnectionError:
            raise
        except Exception as exc:
            err = ConnectionError(
                f"Redis is unreachable at startup: {exc}"
            )
            if self._strict:
                raise err from exc
            log.warning(
                "session_store.redis_unreachable_at_startup",
                error=str(exc),
                note="sessions will use in-memory fallback",
            )

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
                self._log_op_failure("load", exc)
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
                self._log_op_failure("save", exc)
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
                self._log_op_failure("clear", exc)

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

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _log_op_failure(self, op: str, exc: Exception) -> None:
        """
        Log a per-operation Redis failure.

        Strict mode → ERROR (operator attention required; data may drift to memory).
        Non-strict mode → WARNING (expected fallback).
        """
        fields = {
            "op": op,
            "error": str(exc),
            "fallback": "memory",
        }
        if self._strict:
            log.error("session_store.redis_op_failed_strict", **fields)
        else:
            log.warning(f"session_store.redis_{op}_failed", **fields)


# ── Module-level singleton ─────────────────────────────────────────────────────

_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Return the process-level SessionStore instance (created lazily).

    Reads ``session_store``, ``redis_url``, ``session_ttl_seconds``,
    ``session_max_turns``, and ``session_store_strict`` from
    ``config/settings.yml`` (retrieval section).
    """
    global _store
    if _store is None:
        from graphrag.core.config import get_settings
        cfg    = get_settings()
        ret    = cfg.retrieval
        mode   = ret.get("session_store", "memory")
        url    = ret.get("redis_url", "") if mode == "redis" else ""
        ttl    = int(ret.get("session_ttl_seconds", SESSION_TTL_SECONDS))
        max_t  = int(ret.get("session_max_turns", SESSION_MAX_TURNS))
        strict = bool(ret.get("session_store_strict", False))
        _store = SessionStore(
            redis_url=url or None,
            max_turns=max_t,
            ttl_seconds=ttl,
            strict=strict,
        )
    return _store
