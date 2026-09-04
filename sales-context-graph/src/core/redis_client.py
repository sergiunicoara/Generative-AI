"""Async Redis connection singleton — mirrors src/core/neo4j_client.py's pattern.

Returns None (not a client) when REDIS_URL is unset, so callers (api/state.py's
get_ingestion_store()) fall back to the in-memory store for local dev without
docker-compose's redis service, rather than failing to start.
"""

from __future__ import annotations

import asyncio
import weakref

import redis.asyncio as aioredis

from src.core.config import get_settings

_client: aioredis.Redis | None = None
# A weak reference to the loop the cached client's connections are bound to.
# NOT id(loop): CPython reuses memory addresses, so a freshly created loop can
# land on the same id() as a closed one, and an identity check against that id
# would wrongly reuse a client whose transport is already dead (the symptom is
# "'NoneType' object has no attribute 'send'" from proactor_events, because the
# closed loop's _proactor is None). A weakref compares the actual object and
# also goes empty once the old loop is collected.
_client_loop: weakref.ReferenceType | None = None


def get_redis() -> aioredis.Redis | None:
    global _client, _client_loop
    # Async Redis connections are loop-bound. Test harnesses create a loop per
    # test; server processes normally keep one loop for their whole life.
    # Never reuse a client whose loop is gone, closed, or simply different.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if _client is not None:
        bound_loop = _client_loop() if _client_loop is not None else None
        # `.is_closed()` deliberately comes before `is not loop` in the `or`
        # below (order doesn't matter for either check -- both are cheap and
        # side-effect-free). Reversed, mypy's identity-comparison narrowing
        # treats the `bound_loop is not loop` check as unifying bound_loop's
        # type with loop's (AbstractEventLoop | None) for the rest of the
        # expression, and flags `.is_closed()` as possibly called on None --
        # even though the `if bound_loop is None` above already ruled that out.
        if bound_loop is None:
            stale = True
        else:
            stale = bound_loop.is_closed() or bound_loop is not loop
        if stale:
            _client = None
            _client_loop = None

    if _client is None:
        cfg = get_settings()
        if not cfg.redis_url:
            return None
        _client = aioredis.from_url(cfg.redis_url, decode_responses=True)
        _client_loop = weakref.ref(loop) if loop is not None else None
    return _client
