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

import re
import structlog
from functools import lru_cache

log = structlog.get_logger(__name__)

_EMBEDDING_SIMILARITY_THRESHOLD = 0.92   # cosine similarity for dedup
_FUZZY_SCORE_THRESHOLD = 85              # rapidfuzz ratio for soft match


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class AliasRegistry:
    """
    In-memory alias registry backed by Neo4j.

    Usage::

        registry = AliasRegistry(neo4j_client)
        await registry.load()
        canonical = registry.resolve("Space Exploration Technologies")
        # → ("SpaceX", "ORG") or None if unknown
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client
        # normalized_alias → (canonical_name, canonical_type)
        self._exact: dict[str, tuple[str, str]] = {}
        self._loaded = False

    async def load(self) -> None:
        """Refresh alias table from Neo4j."""
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)<-[:ALIAS_OF]-(a:Alias)
            RETURN e.name AS canonical_name,
                   e.type AS canonical_type,
                   collect(a.value) AS aliases
            """
        )
        self._exact.clear()
        for row in rows:
            cname = row["canonical_name"]
            ctype = row["canonical_type"]
            # Register the canonical name itself
            self._exact[_normalize(cname)] = (cname, ctype)
            # Register every alias
            for alias in row.get("aliases") or []:
                if alias:
                    self._exact[_normalize(alias)] = (cname, ctype)
        self._loaded = True
        log.info("alias_registry.loaded", entries=len(self._exact))

    def resolve(self, raw_name: str) -> tuple[str, str] | None:
        """
        Resolve a raw name to (canonical_name, canonical_type).
        Returns None if not found — caller should treat as new entity.
        """
        key = _normalize(raw_name)

        # 1. Exact / normalized match
        if key in self._exact:
            return self._exact[key]

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
        """Persist a new alias to Neo4j and update in-memory cache."""
        from uuid import uuid4
        await self._neo4j.run(
            """
            MATCH (e:Entity {name: $canonical_name, type: $canonical_type})
            MERGE (a:Alias {value: $raw_value})
            ON CREATE SET a.id           = $alias_id,
                          a.normalized   = $normalized,
                          a.source_doc   = $source_doc,
                          a.confidence   = $confidence,
                          a.created_at   = datetime()
            MERGE (a)-[:ALIAS_OF]->(e)
            """,
            canonical_name=canonical_name,
            canonical_type=canonical_type,
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
        to the given one.  Returns (name, type, similarity) or None.

        Used before creating a new entity to prevent embedding-level
        duplicates that escaped name-based resolution.
        """
        rows = await self._neo4j.run(
            """
            CALL db.index.vector.queryNodes('entity_embeddings', 5, $embedding)
            YIELD node AS e, score
            WHERE e.type = $entity_type
              AND e.name <> $exclude_name
              AND score >= $threshold
            RETURN e.name AS name, e.type AS type, score
            ORDER BY score DESC
            LIMIT 1
            """,
            embedding=embedding,
            entity_type=entity_type,
            exclude_name=exclude_name,
            threshold=_EMBEDDING_SIMILARITY_THRESHOLD,
        )
        if rows:
            r = rows[0]
            return r["name"], r["type"], float(r["score"])
        return None


# ── Module-level singleton ─────────────────────────────────────────────────────

_registry: AliasRegistry | None = None


def get_alias_registry(neo4j_client=None) -> AliasRegistry:
    global _registry
    if _registry is None:
        if neo4j_client is None:
            from graphrag.graph.neo4j_client import get_neo4j
            neo4j_client = get_neo4j()
        _registry = AliasRegistry(neo4j_client)
    return _registry
