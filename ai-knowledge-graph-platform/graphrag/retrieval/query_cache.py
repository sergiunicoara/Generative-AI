"""Query result cache — Redis-backed with provenance-aware invalidation.

Problem solved
--------------
The retrieval pipeline runs 6 stages (ANN, BM25, rerank, multihop, GNN, LLM)
on every query.  For identical questions in the same tenant and session context,
this is pure waste.  A cache that returns the stored answer in O(1) reduces
latency from ~2–5 s to ~5 ms and cuts LLM token spend.

Invalidation challenge
----------------------
A naive TTL cache becomes stale when new documents are ingested: a cached
answer to "Who is the CEO of SpaceX?" is invalid the moment a new document
changes that fact.  But invalidating ALL cached answers on every ingestion
is too aggressive — most ingestions don't affect most cached queries.

Provenance-aware invalidation
------------------------------
Each cached result stores the entity names used in generating the answer.
When a document is ingested, its entity names are extracted and compared
against the provenance set of each cached result.  Only results that mention
at least one entity affected by the new ingestion are invalidated.

This requires storing:
  cache_key  → {answer, contexts, citations, entities_used, created_at}

Cache key
---------
    SHA-256( lower(query) + "|" + tenant + "|" + session_id )
    session_id="" for non-session queries.

Backends
--------
1. Redis — primary.  TTL-based expiry with provenance-set storage.
2. In-memory dict — fallback when Redis is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_DEFAULT_TTL = 3600        # 1 hour
_PROVENANCE_TTL = 86400    # 1 day — keep provenance index longer than results


class QueryCache:
    """
    Redis-backed query result cache with provenance-aware invalidation.

    Usage::

        cache = QueryCache(ttl=3600)
        await cache.connect()

        # Store a result
        await cache.set(
            query="Who is CEO of SpaceX?",
            tenant="acme",
            result={"answer": "Elon Musk", "contexts": [...], "citations": [...]},
            entities_used=["SpaceX", "Elon Musk"],
            session_id="",
        )

        # Retrieve
        hit = await cache.get("Who is CEO of SpaceX?", tenant="acme")
        if hit:
            return hit["answer"]

        # Invalidate on new document ingest
        invalidated = await cache.invalidate_for_entities(
            entity_names=["SpaceX", "Tesla"],
            tenant="acme",
        )
    """

    def __init__(self, ttl: int = _DEFAULT_TTL):
        self._ttl = ttl
        self._redis = None
        self._memory: dict[str, dict] = {}    # in-memory fallback
        self._prov_index: dict[str, set[str]] = {}  # entity → set of cache keys

    async def connect(self) -> None:
        """Attempt to connect to Redis; fall back to in-memory silently."""
        try:
            import redis.asyncio as aioredis
            from graphrag.core.config import get_settings
            cfg = get_settings()
            redis_url = getattr(cfg, "redis_url", "redis://localhost:6379/0")
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
            await self._redis.ping()
            log.info("query_cache.redis_connected")
        except (ImportError, OSError, ConnectionError) as exc:
            log.warning("query_cache.redis_unavailable", error=str(exc),
                        fallback="in-memory")
            self._redis = None

    # ── Public API ─────────────────────────────────────────────────────────────

    async def get(
        self,
        query: str,
        tenant: str = "default",
        session_id: str = "",
    ) -> dict | None:
        """
        Return cached result or None on miss.
        """
        key = self._cache_key(query, tenant, session_id)
        if self._redis:
            try:
                raw = await self._redis.get(key)
                if raw:
                    log.info("query_cache.hit", key=key[:16])
                    return json.loads(raw)
            except Exception as exc:
                log.warning("query_cache.get_error", error=str(exc))
        else:
            entry = self._memory.get(key)
            if entry and time.time() - entry.get("_cached_at", 0) < self._ttl:
                log.info("query_cache.hit_memory", key=key[:16])
                return entry
        return None

    async def set(
        self,
        query: str,
        tenant: str = "default",
        result: dict | None = None,
        entities_used: list[str] | None = None,
        session_id: str = "",
    ) -> None:
        """
        Store a query result with its provenance entities for later invalidation.
        """
        key = self._cache_key(query, tenant, session_id)
        payload = {
            **(result or {}),
            "_cache_key":     key,
            "_cached_at":     time.time(),
            "_entities_used": entities_used or [],
            "_tenant":        tenant,
        }

        if self._redis:
            try:
                await self._redis.setex(key, self._ttl, json.dumps(payload))
                # Build provenance index: entity → set of cache keys
                for entity in (entities_used or []):
                    prov_key = self._prov_key(entity, tenant)
                    await self._redis.sadd(prov_key, key)
                    await self._redis.expire(prov_key, _PROVENANCE_TTL)
                log.info("query_cache.set", key=key[:16], entities=len(entities_used or []))
            except Exception as exc:
                log.warning("query_cache.set_error", error=str(exc))
        else:
            self._memory[key] = payload
            for entity in (entities_used or []):
                self._prov_index.setdefault(entity, set()).add(key)

    async def invalidate_for_entities(
        self,
        entity_names: list[str],
        tenant: str = "default",
    ) -> int:
        """
        Invalidate all cached results that used any of the given entities.

        Call this after ingesting a new document with extracted entity names.
        Returns count of invalidated entries.
        """
        invalidated = 0
        if self._redis:
            try:
                keys_to_delete: set[str] = set()
                for entity in entity_names:
                    prov_key = self._prov_key(entity, tenant)
                    members = await self._redis.smembers(prov_key)
                    keys_to_delete.update(members)
                    await self._redis.delete(prov_key)
                if keys_to_delete:
                    await self._redis.delete(*keys_to_delete)
                    invalidated = len(keys_to_delete)
            except Exception as exc:
                log.warning("query_cache.invalidate_error", error=str(exc))
        else:
            keys_to_delete = set()
            for entity in entity_names:
                keys_to_delete.update(self._prov_index.pop(entity, set()))
            for key in keys_to_delete:
                self._memory.pop(key, None)
            invalidated = len(keys_to_delete)

        if invalidated:
            log.info(
                "query_cache.invalidated",
                count=invalidated,
                entities=len(entity_names),
                tenant=tenant,
            )
        return invalidated

    async def flush_tenant(self, tenant: str) -> int:
        """Remove all cached results for a tenant (use after bulk re-ingestion)."""
        if self._redis:
            try:
                # Scan for all keys with tenant in them
                pattern = f"qcache:{tenant}:*"
                count = 0
                async for key in self._redis.scan_iter(pattern):
                    await self._redis.delete(key)
                    count += 1
                return count
            except Exception as exc:  # broad: redis.RedisError hierarchy; redis pkg may not be installed at module scope
                log.warning("query_cache.flush_error", tenant=tenant, error=str(exc))
                return 0
        else:
            before = len(self._memory)
            self._memory = {
                k: v for k, v in self._memory.items()
                if v.get("_tenant") != tenant
            }
            return before - len(self._memory)

    async def stats(self) -> dict:
        """Return cache statistics."""
        if self._redis:
            try:
                info = await self._redis.info("keyspace")
                return {"backend": "redis", "keyspace": info}
            except Exception:
                return {"backend": "redis", "error": "unavailable"}
        return {"backend": "memory", "entries": len(self._memory)}

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(query: str, tenant: str, session_id: str) -> str:
        raw = f"{query.strip().lower()}|{tenant}|{session_id}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:24]
        return f"qcache:{tenant}:{digest}"

    @staticmethod
    def _prov_key(entity_name: str, tenant: str) -> str:
        safe = entity_name.lower().replace(" ", "_")[:40]
        return f"qcache_prov:{tenant}:{safe}"


# ── Module-level singleton ─────────────────────────────────────────────────────

_cache: QueryCache | None = None


async def get_query_cache() -> QueryCache:
    global _cache
    if _cache is None:
        from graphrag.core.config import get_settings
        cfg = get_settings()
        ttl = getattr(cfg, "query_cache_ttl", _DEFAULT_TTL)
        _cache = QueryCache(ttl=ttl)
        await _cache.connect()
    return _cache
