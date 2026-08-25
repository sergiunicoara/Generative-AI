"""Content-addressed cache for completed, governed query answers.

The cache key deliberately differs from ``ContextManifest.integrity_hash``.
The manifest hash proves that one historical trace has not changed; this key
identifies whether the inputs that can change a *new* answer are unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import unicodedata
from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import structlog

from graphrag.observability.operational_metrics import set_store_degraded

log = structlog.get_logger(__name__)

_DEFAULT_TTL = 3600
_KEY_PREFIX = "graphrag:answer-cache:v2:"
_PROVENANCE_TTL = 86400

# The in-process fallback is a development convenience, not a second cache
# tier. Bounding it keeps a Redis outage from turning an unbounded stream of
# distinct queries into unbounded process memory: entries were only ever
# expired lazily, on a get() for that exact key, so a key written once and
# never read again was retained until the process died.
_DEFAULT_MEMORY_MAX_ENTRIES = 2048


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def normalize_query(query: str) -> str:
    """Conservatively normalize formatting without changing query meaning."""
    return " ".join(unicodedata.normalize("NFKC", query).split()).casefold()


@dataclass(frozen=True)
class QueryCacheContext:
    """Versioned inputs whose change must force a cache miss."""

    corpus_revision: int
    requested_mode: str
    effective_mode: str
    model_route: dict[str, str]
    prompt_version: str
    retrieval_config: dict[str, Any]
    ontology_version: str
    valid_at: str | None = None
    transaction_at: str | None = None
    # Entitlements are part of an answer's identity: a tenant-wide cache entry
    # must never be reused by a caller with a different document ACL.
    access_fingerprint: str = "tenant-default"
    cache_schema_version: str = "3"

    def canonical_content(self) -> dict[str, Any]:
        return asdict(self)


def build_cache_key(
    query: str,
    tenant: str,
    context: QueryCacheContext,
) -> str:
    """Return a stable SHA-256 key independent of dictionary ordering."""
    payload = {
        "tenant": tenant,
        "normalized_query": normalize_query(query),
        "context": context.canonical_content(),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    tenant_digest = hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:16]
    return f"{_KEY_PREFIX}{tenant_digest}:{digest}"


class QueryCacheUnavailable(RuntimeError):
    """Raised in strict mode when the shared cache backend cannot be reached."""


class QueryCache:
    """Redis-backed answer cache with a bounded process-local fallback.

    The fallback is per-process, so in any multi-replica deployment it also
    makes ``invalidate_for_entities`` local: a correction applied on one worker
    cannot evict another worker copy of the affected answer, and that worker
    keeps serving the superseded answer until its TTL expires. ``strict=True``
    refuses to start in that state instead of degrading into it silently --
    the same trade-off ``session_store_strict`` already makes for sessions.
    """

    def __init__(
        self,
        ttl: int = _DEFAULT_TTL,
        redis_url: str | None = None,
        *,
        strict: bool = False,
        max_memory_entries: int = _DEFAULT_MEMORY_MAX_ENTRIES,
    ):
        self._ttl = ttl
        self._redis_url = redis_url
        self._redis = None
        self._strict = strict
        self._max_memory_entries = max(1, max_memory_entries)
        self._memory: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._prov_index: dict[tuple[str, str], set[str]] = {}
        # Reverse of _prov_index. Without it, removing one entry has to scan
        # every index bucket -- which makes eviction quadratic in exactly the
        # degraded mode this bound exists to protect.
        self._prov_reverse: dict[str, set[tuple[str, str]]] = {}
        self._evictions = 0
        self._last_sweep = 0.0

    async def connect(self) -> None:
        if not self._redis_url:
            if self._strict:
                raise QueryCacheUnavailable(
                    "semantic_answer_cache_strict is set but no redis_url is configured"
                )
            log.warning("query_cache.no_redis", fallback="in-memory")
            set_store_degraded("query_cache", True)
            return
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            log.info("query_cache.redis_connected")
            set_store_degraded("query_cache", False)
        except Exception as exc:  # Redis errors differ between redis-py versions.
            self._redis = None
            if self._strict:
                raise QueryCacheUnavailable(
                    f"answer cache requires Redis but it is unreachable: {exc}"
                ) from exc
            log.warning("query_cache.redis_unavailable", error=str(exc), fallback="in-memory")
            # Invalidation cannot reach sibling replicas in this state, so a
            # corrected answer keeps being served elsewhere for a full TTL.
            set_store_degraded("query_cache", True)

    async def get(
        self,
        query: str,
        tenant: str,
        context: QueryCacheContext,
    ) -> dict[str, Any] | None:
        key = build_cache_key(query, tenant, context)
        if self._redis is not None:
            try:
                raw = await self._redis.get(key)
                if raw:
                    log.info("query_cache.hit", key=key[-12:], tenant=tenant)
                    return json.loads(raw)
                return None
            except Exception as exc:
                log.warning("query_cache.get_error", error=str(exc), tenant=tenant)
                return None

        self._expire_memory()
        entry = self._memory.get(key)
        if entry is None:
            return None
        # The throttled sweep above may not have run this call, so the entry
        # being returned is always checked against its own timestamp.
        if time.time() - float(entry.get("cached_at", 0)) >= self._ttl:
            self._forget(key)
            return None
        self._memory.move_to_end(key)
        return dict(entry)

    async def set(
        self,
        query: str,
        tenant: str,
        context: QueryCacheContext,
        result: dict[str, Any],
        *,
        source_query_id: str,
        source_trace_id: str,
        entities_used: list[str] | None = None,
    ) -> str:
        key = build_cache_key(query, tenant, context)
        payload: dict[str, Any] = {
            "cache_key": key,
            "cached_at": time.time(),
            "tenant": tenant,
            "context": context.canonical_content(),
            "result": result,
            "source_query_id": source_query_id,
            "source_trace_id": source_trace_id,
            "entities_used": entities_used or [],
        }
        if self._redis is not None:
            try:
                await self._redis.setex(key, self._ttl, _canonical_json(payload))
                for entity in entities_used or []:
                    provenance_key = self._provenance_key(entity, tenant)
                    await self._redis.sadd(provenance_key, key)
                    await self._redis.expire(provenance_key, _PROVENANCE_TTL)
                log.info("query_cache.set", key=key[-12:], tenant=tenant)
            except Exception as exc:
                log.warning("query_cache.set_error", error=str(exc), tenant=tenant)
            return key

        self._remember(key, payload)
        for entity in entities_used or []:
            self._index_provenance(key, (tenant, entity.casefold()))
        return key

    async def invalidate_for_entities(
        self,
        entity_names: list[str],
        tenant: str = "default",
    ) -> int:
        """Eagerly evict affected keys; corpus revision remains the hard guard."""
        keys_to_delete: set[str] = set()
        if self._redis is not None:
            try:
                for entity in entity_names:
                    provenance_key = self._provenance_key(entity, tenant)
                    keys_to_delete.update(await self._redis.smembers(provenance_key))
                    await self._redis.delete(provenance_key)
                if keys_to_delete:
                    await self._redis.delete(*keys_to_delete)
                return len(keys_to_delete)
            except Exception as exc:
                log.warning("query_cache.invalidate_error", error=str(exc), tenant=tenant)
                return 0

        for entity in entity_names:
            index_key = (tenant, entity.casefold())
            for key in self._prov_index.get(index_key, set()):
                keys_to_delete.add(key)
        for key in keys_to_delete:
            self._forget(key)
        return len(keys_to_delete)

    async def flush_tenant(self, tenant: str) -> int:
        tenant_digest = hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:16]
        pattern = f"{_KEY_PREFIX}{tenant_digest}:*"
        if self._redis is not None:
            try:
                keys = [key async for key in self._redis.scan_iter(pattern)]
                if keys:
                    await self._redis.delete(*keys)
                return len(keys)
            except Exception as exc:
                log.warning("query_cache.flush_error", tenant=tenant, error=str(exc))
                return 0
        keys = [key for key, value in self._memory.items() if value.get("tenant") == tenant]
        for key in keys:
            self._forget(key)
        return len(keys)

    async def stats(self) -> dict[str, Any]:
        if self._redis is not None:
            try:
                return {"backend": "redis", "keyspace": await self._redis.info("keyspace")}
            except Exception:
                return {"backend": "redis", "error": "unavailable"}
        self._expire_memory(force=True)
        return {
            "backend": "memory",
            "entries": len(self._memory),
            "max_entries": self._max_memory_entries,
            "evictions": self._evictions,
        }

    def _expire_memory(self, *, force: bool = False) -> None:
        """Drop every TTL-expired fallback entry, not only the one being read.

        Throttled: a full scan on every single operation would put an O(n) walk
        on the query path for no benefit, since nothing can expire in the
        microseconds since the last one. Callers that need a precise count
        (``stats``) pass ``force``; ``get`` independently re-checks the one
        entry it is about to return, so throttling can never serve stale data.
        """
        if not self._memory:
            return
        now = time.time()
        if not force and now - self._last_sweep < self._sweep_interval():
            return
        self._last_sweep = now
        cutoff = now - self._ttl
        expired = [
            key for key, entry in self._memory.items()
            if float(entry.get("cached_at", 0)) <= cutoff
        ]
        for key in expired:
            self._forget(key)

    def _sweep_interval(self) -> float:
        """How often a full expiry scan is worth doing, relative to the TTL."""
        return max(1.0, self._ttl / 10)

    def _remember(self, key: str, payload: dict[str, Any]) -> None:
        """Insert an entry, evicting the least recently used one when full."""
        self._expire_memory()
        self._memory[key] = payload
        self._memory.move_to_end(key)
        while len(self._memory) > self._max_memory_entries:
            evicted, _ = self._memory.popitem(last=False)
            self._drop_from_provenance(evicted)
            self._evictions += 1

    def _forget(self, key: str) -> None:
        self._memory.pop(key, None)
        self._drop_from_provenance(key)

    def _index_provenance(self, key: str, index_key: tuple[str, str]) -> None:
        self._prov_index.setdefault(index_key, set()).add(key)
        self._prov_reverse.setdefault(key, set()).add(index_key)

    def _drop_from_provenance(self, key: str) -> None:
        """Keep the provenance index from outliving the entries it points at.

        Without this the index is a second unbounded structure: it accumulated
        one entry per (tenant, entity) forever, holding cache keys that had
        already expired or been evicted. The reverse index makes this O(number
        of entities that entry cited) rather than O(whole index).
        """
        for index_key in self._prov_reverse.pop(key, ()):
            keys = self._prov_index.get(index_key)
            if keys is None:
                continue
            keys.discard(key)
            if not keys:
                self._prov_index.pop(index_key, None)

    @staticmethod
    def _provenance_key(entity_name: str, tenant: str) -> str:
        raw = f"{tenant}\0{entity_name.casefold()}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"graphrag:answer-cache-provenance:v2:{digest}"


@lru_cache(maxsize=1)
def _cache_settings() -> tuple[str | None, int, bool, int]:
    from graphrag.core.config import get_settings

    cfg = get_settings()
    redis_url = os.getenv("REDIS_URL") or cfg.retrieval.get("redis_url", "") or None
    ttl = int(cfg.retrieval.get("semantic_answer_cache_ttl_seconds", _DEFAULT_TTL))
    strict = bool(cfg.retrieval.get("semantic_answer_cache_strict", False))
    max_entries = int(cfg.retrieval.get(
        "semantic_answer_cache_max_memory_entries", _DEFAULT_MEMORY_MAX_ENTRIES,
    ))
    return redis_url, ttl, strict, max_entries


_cache: QueryCache | None = None
_cache_lock: asyncio.Lock | None = None


async def get_query_cache() -> QueryCache:
    """Return the process singleton, safe against concurrent cold-start.

    ``connect()`` awaits, so without a lock two coroutines racing on the first
    query both pass the ``None`` check, each open a Redis connection pool, and
    one pool leaks for the life of the process. This mirrors the same fix
    already applied to ``get_rabbitmq()``. A failed connect is not cached, so a
    transient Redis outage at startup does not permanently pin the process to
    the in-memory fallback.
    """
    global _cache, _cache_lock
    # asyncio.Lock() must be constructed inside a running loop, and there is no
    # await between this check and the assignment, so this is not itself a race.
    if _cache_lock is None:
        _cache_lock = asyncio.Lock()
    async with _cache_lock:
        if _cache is None:
            redis_url, ttl, strict, max_entries = _cache_settings()
            candidate = QueryCache(
                ttl=ttl, redis_url=redis_url,
                strict=strict, max_memory_entries=max_entries,
            )
            await candidate.connect()
            _cache = candidate
    return _cache


async def close_query_cache() -> None:
    """Close and reset the process singleton when it was initialized."""
    global _cache, _cache_lock
    cache, _cache = _cache, None
    _cache_lock = None
    redis = getattr(cache, "_redis", None) if cache is not None else None
    if redis is not None:
        try:
            await redis.aclose()
        except Exception as exc:  # noqa: BLE001 - shutdown must never raise
            log.warning("query_cache.close_error", error=str(exc))
