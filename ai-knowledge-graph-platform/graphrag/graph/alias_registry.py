"""Alias registry — resolves alternative entity names to canonical ones.

Problem solved
--------------
The same entity appears under different names across documents:
  "SpaceX" / "Space Exploration Technologies" / "Space Exploration Corp."
Without resolution, each variant creates a separate node, breaking
retrieval, GNN propagation, and community detection.

Strategy
--------
1. Exact match on stored aliases (instant)
2. Normalized match — strip punctuation, lowercase, collapse whitespace
3. Embedding similarity against known entity embeddings (cosine > 0.92)
4. Queue ambiguous cases for human review rather than auto-creating

The registry is loaded once per process and refreshed after every
ingestion batch.
"""

from __future__ import annotations

import json
import os
import re
import structlog
from functools import lru_cache

log = structlog.get_logger(__name__)

_EMBEDDING_SIMILARITY_THRESHOLD = 0.92   # cosine similarity for dedup
_FUZZY_SCORE_THRESHOLD = 85              # rapidfuzz ratio for soft match
_REDIS_TTL = 86400                       # 24h — alias table refreshed each ingestion batch

# Redis key pattern:  graphrag:aliases:{tenant}   → Hash { normalized_alias: "canonical|type" }
_REDIS_KEY = "graphrag:aliases:{tenant}"


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Regulatory agency prefixes that are stripped when normalizing AD / regulation names.
# "EASA AD 2022-0201" and "AD 2022-0201" refer to the same document —
# the prefix is a source attribution, not a distinguishing identifier.
_REGULATORY_PREFIXES = re.compile(
    r"^(easa|faa|icao|dot|tc|stc|pma|tso)\s+",
    re.IGNORECASE,
)


def _normalize_regulatory(text: str) -> str:
    """Like _normalize but also strips known regulatory agency prefixes.

    Applied to AIRWORTHINESS_DIRECTIVE and REGULATION entity names so that
    'EASA AD 2022-0201' and 'AD 2022-0201' resolve to the same canonical key,
    enabling forward-chaining transitivity to fire across the supersession chain.
    """
    stripped = _REGULATORY_PREFIXES.sub("", text.strip())
    return _normalize(stripped)


async def _get_redis():
    """Return an async Redis client or None if unavailable."""
    try:
        import redis.asyncio as aioredis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = aioredis.from_url(redis_url, decode_responses=True)
        await client.ping()
        return client
    except Exception:
        return None


class AliasRegistry:
    """
    In-memory alias registry backed by Neo4j.

    Usage::

        registry = AliasRegistry(neo4j_client)
        await registry.load()
        canonical = registry.resolve("Space Exploration Technologies")
        # → ("SpaceX", "ORG") or None if unknown
    """

    def __init__(self, neo4j_client, tenant: str = "default"):
        self._neo4j = neo4j_client
        self._tenant = tenant
        # normalized_alias → (canonical_name, canonical_type)
        # Keyed per-tenant so the registry is safe for single-tenant usage
        # (multi-tenant deployments should use one registry instance per tenant)
        self._exact: dict[str, tuple[str, str]] = {}
        self._loaded = False

    async def load(self) -> None:
        """Refresh alias table from Neo4j and push to Redis for cross-worker sharing."""
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})
            OPTIONAL MATCH (e)<-[:ALIAS_OF]-(a:Alias {tenant: $tenant})
            RETURN e.name AS canonical_name,
                   e.type AS canonical_type,
                   collect(a.value) AS aliases
            """,
            tenant=self._tenant,
        )
        _REGULATION_TYPES = {"AIRWORTHINESS_DIRECTIVE", "REGULATION", "TYPE_CERTIFICATE",
                              "SUPPLEMENTAL_TYPE_CERTIFICATE", "AIRWORTHINESS_APPROVAL"}

        self._exact.clear()
        for row in rows:
            cname = row["canonical_name"]
            ctype = row["canonical_type"]
            self._exact[_normalize(cname)] = (cname, ctype)
            # Also index under prefix-stripped key for regulatory entities so
            # "EASA AD 2022-0201" and "AD 2022-0201" resolve to the same node.
            if ctype in _REGULATION_TYPES:
                self._exact[_normalize_regulatory(cname)] = (cname, ctype)
            for alias in row.get("aliases") or []:
                if alias:
                    self._exact[_normalize(alias)] = (cname, ctype)
                    if ctype in _REGULATION_TYPES:
                        self._exact[_normalize_regulatory(alias)] = (cname, ctype)
        self._loaded = True
        log.info("alias_registry.loaded", tenant=self._tenant, entries=len(self._exact))

        # Push to Redis so other workers share this alias table without each
        # doing a full Neo4j round-trip on startup.
        redis = await _get_redis()
        if redis:
            rkey = _REDIS_KEY.format(tenant=self._tenant)
            try:
                pipe = redis.pipeline()
                pipe.delete(rkey)
                if self._exact:
                    # Store as hash: normalized_alias → "canonical_name|canonical_type"
                    mapping = {k: f"{v[0]}|{v[1]}" for k, v in self._exact.items()}
                    pipe.hset(rkey, mapping=mapping)
                    pipe.expire(rkey, _REDIS_TTL)
                await pipe.execute()
                log.info("alias_registry.redis_pushed", tenant=self._tenant,
                         entries=len(self._exact))
            except Exception as exc:
                log.warning("alias_registry.redis_push_failed", error=str(exc))
            finally:
                await redis.aclose()

    def resolve(self, raw_name: str) -> tuple[str, str] | None:
        """
        Resolve a raw name to (canonical_name, canonical_type).
        Checks in-memory cache first (O(1)), then falls back to None.
        Redis is used only during load() — resolve() stays sync and fast.
        Returns None if not found — caller should treat as new entity.
        """
        key = _normalize(raw_name)

        # 1. Exact / normalized match (in-memory — loaded from Redis or Neo4j)
        if key in self._exact:
            return self._exact[key]

        # 1b. Regulatory prefix-stripped match
        # "EASA AD 2022-0201" → "AD 2022-0201" canonical key
        reg_key = _normalize_regulatory(raw_name)
        if reg_key != key and reg_key in self._exact:
            log.debug("alias_registry.regulatory_prefix_match", raw=raw_name,
                      canonical=self._exact[reg_key][0])
            return self._exact[reg_key]

        # 2. Fuzzy match (optional, requires rapidfuzz)
        try:
            from rapidfuzz import fuzz
            best_score = 0
            best_match = None
            for stored_key, canonical in self._exact.items():
                score = fuzz.ratio(key, stored_key)
                if score > best_score:
                    best_score = score
                    best_match = canonical
            if best_score >= _FUZZY_SCORE_THRESHOLD and best_match:
                log.debug(
                    "alias_registry.fuzzy_match",
                    raw=raw_name,
                    canonical=best_match[0],
                    score=best_score,
                )
                return best_match
        except ImportError:
            pass  # rapidfuzz not installed — skip fuzzy step

        return None

    async def register_alias(
        self,
        raw_value: str,
        canonical_name: str,
        canonical_type: str,
        source_doc_id: str = "",
        confidence: float = 1.0,
    ) -> None:
        """Persist a new alias to Neo4j (tenant-scoped) and update in-memory cache."""
        from uuid import uuid4
        await self._neo4j.run(
            """
            MATCH (e:Entity {name: $canonical_name, type: $canonical_type, tenant: $tenant})
            MERGE (a:Alias {value: $raw_value, tenant: $tenant})
            ON CREATE SET a.id           = $alias_id,
                          a.normalized   = $normalized,
                          a.source_doc   = $source_doc,
                          a.confidence   = $confidence,
                          a.created_at   = datetime()
            MERGE (a)-[:ALIAS_OF]->(e)
            """,
            canonical_name=canonical_name,
            canonical_type=canonical_type,
            tenant=self._tenant,
            raw_value=raw_value,
            alias_id=str(uuid4()),
            normalized=_normalize(raw_value),
            source_doc=source_doc_id,
            confidence=confidence,
        )
        self._exact[_normalize(raw_value)] = (canonical_name, canonical_type)
        log.info(
            "alias_registry.alias_added",
            raw=raw_value,
            canonical=canonical_name,
        )

    async def find_duplicate_by_embedding(
        self,
        embedding: list[float],
        entity_type: str,
        exclude_name: str = "",
    ) -> tuple[str, str, float] | None:
        """
        Search for an existing entity whose embedding is very close
        to the given one — tenant-scoped.  Returns (name, type, similarity) or None.
        """
        rows = await self._neo4j.run(
            """
            CALL db.index.vector.queryNodes('entity_embeddings', 5, $embedding)
            YIELD node AS e, score
            WHERE e.type = $entity_type
              AND e.name <> $exclude_name
              AND e.tenant = $tenant
              AND score >= $threshold
            RETURN e.name AS name, e.type AS type, score
            ORDER BY score DESC
            LIMIT 1
            """,
            embedding=embedding,
            entity_type=entity_type,
            exclude_name=exclude_name,
            tenant=self._tenant,
            threshold=_EMBEDDING_SIMILARITY_THRESHOLD,
        )
        if rows:
            r = rows[0]
            return r["name"], r["type"], float(r["score"])
        return None


# ── Per-tenant registry pool ──────────────────────────────────────────────────
# One AliasRegistry instance per tenant — entity identity is tenant-scoped.

_registries: dict[str, AliasRegistry] = {}


def get_alias_registry(neo4j_client=None, tenant: str = "default") -> AliasRegistry:
    global _registries
    if tenant not in _registries:
        if neo4j_client is None:
            from graphrag.graph.neo4j_client import get_neo4j
            neo4j_client = get_neo4j()
        _registries[tenant] = AliasRegistry(neo4j_client, tenant=tenant)
    return _registries[tenant]


async def load_alias_registry(neo4j_client=None, tenant: str = "default") -> AliasRegistry:
    """
    Load (or warm) the alias registry for a tenant.

    Tries Redis first — if another worker already pushed the alias table,
    this skips the Neo4j MATCH entirely.  Falls back to Neo4j on Redis miss
    and then pushes the result to Redis for the next worker.
    """
    registry = get_alias_registry(neo4j_client=neo4j_client, tenant=tenant)
    if registry._loaded:
        return registry

    # Try Redis warm-load — avoids full Neo4j scan on worker startup
    redis = await _get_redis()
    if redis:
        rkey = _REDIS_KEY.format(tenant=tenant)
        try:
            mapping = await redis.hgetall(rkey)
            if mapping:
                registry._exact = {
                    k: tuple(v.split("|", 1))   # type: ignore[assignment]
                    for k, v in mapping.items()
                }
                registry._loaded = True
                log.info("alias_registry.redis_warm_load", tenant=tenant,
                         entries=len(registry._exact))
                await redis.aclose()
                return registry
        except Exception as exc:
            log.warning("alias_registry.redis_load_failed", error=str(exc))
        finally:
            try:
                await redis.aclose()
            except Exception:
                pass

    # Redis miss or unavailable — load from Neo4j (also pushes to Redis)
    await registry.load()
    return registry
