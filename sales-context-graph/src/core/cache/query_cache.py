"""Exact-match query-result cache (Phase 5, docs/evaluation.md's semantic/
result-cache item). Deliberately exact-match, not vector-similarity --
docs/evaluation.md's own recommendation: semantic caching risks serving a
near-miss answer as if it were exact, and this vertical slice has no
measured need for that extra complexity yet. Start simple; a caller asking
the identical normalized question twice inside the TTL is the actual,
observed pattern this closes.

Uses the shared src/core/redis_client.py::get_redis() singleton, not a
second, independent connection -- unlike src/graph/alias_registry.py's own
separate client, which docs/evaluation.md's own external-review cross-check
already flagged as legacy/inconsistent with the rest of this codebase.

Every key is workspace-scoped by construction: get_cached_result()/
cache_result() both require workspace_id as a real parameter folded into
the key server-side, not left to a caller-assembled key string that could
omit it -- the same "structural, not conventional" tenant-isolation
discipline src/graph/execution.py's tenant_query() already enforces for
Cypher. Callers own building `cache_key` (a caller-defined string) from
whatever actually determines the answer -- see src/usecases/nlq/ask.py's
call site for why that must include more than just the question text.

Fails open like every other optional-Redis path in this codebase
(src/graph/alias_registry.py, api/state.py's job store): disabled or an
unreachable Redis both degrade to "no cache," never a hard failure -- a
caller always has a correct fallback (run the real computation).
"""

from __future__ import annotations

import hashlib
from typing import cast

import structlog

from src.core.config import get_settings
from src.core.redis_client import get_redis

log = structlog.get_logger(__name__)

_KEY_PREFIX = "scg:query_cache:"


def _redis_key(workspace_id: str, cache_key: str) -> str:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return f"{_KEY_PREFIX}{workspace_id}:{digest}"


async def get_cached_result(workspace_id: str, cache_key: str) -> str | None:
    """Returns the cached JSON string, or None on a miss, a disabled cache,
    or an unreachable Redis -- these are deliberately indistinguishable to
    the caller, which should react to all three the same way: compute the
    real answer."""
    if not get_settings().query_cache_enabled:
        return None
    client = get_redis()
    if client is None:
        return None
    # redis-py's own stub types Redis.get() as bytes | str | None unconditionally
    # -- it has no generic Redis[str] to reflect decode_responses at the type
    # level -- but src/core/redis_client.py::get_redis() is the sole factory for
    # every client in this codebase and always passes decode_responses=True, so
    # a real bytes value here would mean that invariant broke, not a type worth
    # silently forwarding as bytes | str | None to every caller.
    return cast("str | None", await client.get(_redis_key(workspace_id, cache_key)))


async def cache_result(workspace_id: str, cache_key: str, value: str, *, ttl: int | None = None) -> None:
    if not get_settings().query_cache_enabled:
        return
    client = get_redis()
    if client is None:
        return
    settings = get_settings()
    await client.set(_redis_key(workspace_id, cache_key), value, ex=ttl or settings.query_cache_ttl_seconds)


async def invalidate_workspace_cache(workspace_id: str) -> int:
    """Deletes every cached entry for one workspace. Coarse (workspace-wide,
    not entity-scoped) but correct: over-invalidating costs a few extra
    cache misses; under-invalidating risks serving erased data back out.

    Ready to be called wherever erasure execution actually happens --
    src/domain/assertion.py's ErasureEvent.erasure_scope already lists
    "cache" as an example value the way a real erasure would name this
    cache among what it clears. As of this writing there is no erasure
    *execution* pathway anywhere in src/ to call this from yet (confirmed
    by search -- ErasureEvent itself is only ever referenced by its own
    domain module and the generic model-roundtrip test); this function
    exists so that gap is one call site away from closed, not so it's
    already wired to something that runs. Not silently glossed over --
    see docs/evaluation.md.
    """
    client = get_redis()
    if client is None:
        return 0
    deleted = 0
    async for key in client.scan_iter(match=f"{_KEY_PREFIX}{workspace_id}:*"):
        await client.delete(key)
        deleted += 1
    return deleted
