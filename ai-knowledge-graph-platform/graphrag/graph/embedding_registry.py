"""Embedding model versioning and re-embedding pipeline.

Problem solved
--------------
Entity embeddings are stored vectors.  When the embedding model changes
(new sentence-transformer, dimension shift, OpenAI revision) every stored
vector becomes geometrically incomparable with vectors from the new model.
Cosine similarity between a new-model query and an old-model entity returns
a meaningless number.  ANN recall collapses silently — no error, just wrong
results.

Two distinct failure modes
--------------------------
1. Dimension mismatch  — old model outputs 768d, new model outputs 1536d.
   Neo4j vector index immediately rejects the new vectors.  Detectable.

2. Same dimension, different space — old model = paraphrase-MiniLM,
   new model = text-embedding-3-small, both 768d.  Vectors are numerically
   compatible but semantically in different spaces.  Cosine scores look
   plausible but retrieval quality silently degrades.  Invisible without a
   golden-set recall check.

Architecture
------------
- Every embedding write stamps `embedding_model` and `embedding_version`
  on the Entity node (via merge_entity in neo4j_client.py).
- EmbeddingRegistry tracks which model versions are in the graph and
  how many entities are on each version.
- `check_compatibility()` returns a report: which entities are on the
  current model vs. stale models, blocking vs. non-blocking mismatches.
- `queue_re_embed()` marks stale entities with `embedding_stale=true` so
  the background re-embed job can process them without a full graph scan.
- `apply_re_embedding()` updates a batch of entities with fresh vectors
  and clears the stale flag.

Integration with neo4j_client.py
---------------------------------
merge_entity now sets:
    e.embedding_model   = $embedding_model   (model name string)
    e.embedding_version = $embedding_version (semver string)
Both default to "" when not provided so existing entities are not broken.
"""

from __future__ import annotations

from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)


class EmbeddingRegistry:
    """
    Track embedding model versions across the graph and manage re-embedding.

    Usage::

        registry = EmbeddingRegistry(neo4j_client)

        # Inspect what's in the graph
        report = await registry.check_compatibility(
            current_model="text-embedding-3-small",
            current_version="3.0",
            expected_dim=1536,
        )

        # Mark stale entities for re-embedding
        count = await registry.queue_re_embed(
            current_model="text-embedding-3-small",
            current_version="3.0",
            tenant="acme",
        )

        # Apply fresh vectors from the caller's embedding function
        async def my_embedder(texts):
            ...  # returns list[list[float]]

        updated = await registry.apply_re_embedding(
            embedder=my_embedder,
            current_model="text-embedding-3-small",
            current_version="3.0",
            tenant="acme",
            batch_size=100,
        )
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── Version inventory ──────────────────────────────────────────────────────

    async def inventory(self, tenant: str = "default") -> list[dict]:
        """
        Return all distinct (embedding_model, embedding_version, dim) tuples
        present in the graph, with entity counts per version.

        A healthy graph has exactly one entry.  Multiple entries indicate
        partially-migrated state.
        """
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity)
            WHERE ($tenant = 'default' OR e.tenant = $tenant)
              AND e.embedding IS NOT NULL AND size(e.embedding) > 0
            RETURN coalesce(e.embedding_model,   'unknown') AS model,
                   coalesce(e.embedding_version, 'unknown') AS version,
                   size(e.embedding)                         AS dim,
                   count(e)                                  AS entity_count
            ORDER BY entity_count DESC
            """,
            tenant=tenant,
        )
        return [dict(r) for r in rows]

    async def check_compatibility(
        self,
        current_model: str,
        current_version: str,
        expected_dim: int,
        tenant: str = "default",
    ) -> dict:
        """
        Check how many entities are on the current model vs. stale models.

        Returns::
            {
                "current_model":   str,
                "current_version": str,
                "expected_dim":    int,
                "up_to_date":      int,    # entities on current model
                "stale":           int,    # entities on older model/version
                "dim_mismatch":    int,    # entities with wrong dimension
                "unknown_model":   int,    # no model tag (legacy)
                "blocking":        bool,   # True if dim_mismatch > 0
                "action_required": bool,   # True if stale > 0
                "versions":        list[dict],  # inventory breakdown
            }
        """
        versions = await self.inventory(tenant)

        up_to_date  = 0
        stale       = 0
        dim_mismatch = 0
        unknown     = 0
        for v in versions:
            is_current = (
                v["model"] == current_model
                and v["version"] == current_version
            )
            if v["dim"] != expected_dim:
                dim_mismatch += v["entity_count"]
            elif v["model"] == "unknown":
                unknown += v["entity_count"]
            elif is_current:
                up_to_date += v["entity_count"]
            else:
                stale += v["entity_count"]

        report = {
            "current_model":   current_model,
            "current_version": current_version,
            "expected_dim":    expected_dim,
            "up_to_date":      up_to_date,
            "stale":           stale,
            "dim_mismatch":    dim_mismatch,
            "unknown_model":   unknown,
            "blocking":        dim_mismatch > 0,
            "action_required": stale > 0 or unknown > 0,
            "versions":        versions,
        }
        if report["blocking"]:
            log.error(
                "embedding_registry.dim_mismatch",
                dim_mismatch=dim_mismatch,
                expected_dim=expected_dim,
                impact="ANN index will reject vectors; retrieval broken",
                fix="run re_embed.py --model current_model before restarting",
            )
        elif report["action_required"]:
            log.warning(
                "embedding_registry.stale_embeddings",
                stale=stale,
                unknown=unknown,
                impact="retrieval quality degraded for stale entities",
                fix="run re_embed.py --model current_model",
            )
        return report

    # ── Stale marking ─────────────────────────────────────────────────────────

    async def queue_re_embed(
        self,
        current_model: str,
        current_version: str,
        tenant: str = "default",
        limit: int = 0,
    ) -> int:
        """
        Mark entities whose embedding_model or embedding_version does not match
        the current model as `embedding_stale=true`.

        Returns the number of entities marked.
        A subsequent call to `apply_re_embedding()` processes the queue.
        """
        limit_clause = f"LIMIT {limit}" if limit > 0 else ""
        rows = await self._neo4j.run(
            f"""
            MATCH (e:Entity)
            WHERE ($tenant = 'default' OR e.tenant = $tenant)
              AND e.embedding IS NOT NULL
              AND (
                  coalesce(e.embedding_model,   '') <> $model
               OR coalesce(e.embedding_version, '') <> $version
              )
            SET e.embedding_stale = true
            {limit_clause}
            RETURN count(e) AS marked
            """,
            tenant=tenant,
            model=current_model,
            version=current_version,
        )
        count = rows[0]["marked"] if rows else 0
        log.info(
            "embedding_registry.stale_marked",
            count=count,
            tenant=tenant,
            target_model=current_model,
        )
        return count

    # ── Re-embedding ───────────────────────────────────────────────────────────

    async def apply_re_embedding(
        self,
        embedder,                   # async callable: list[str] -> list[list[float]]
        current_model: str,
        current_version: str,
        tenant: str = "default",
        batch_size: int = 100,
    ) -> int:
        """
        Fetch stale entities in batches, re-embed their descriptions, and
        write the new vectors back to Neo4j with updated model metadata.

        ``embedder`` must be an async function::
            async def embedder(texts: list[str]) -> list[list[float]]: ...

        Returns the total number of entities successfully re-embedded.
        """
        total = 0
        while True:
            rows = await self._neo4j.run(
                """
                MATCH (e:Entity)
                WHERE ($tenant = 'default' OR e.tenant = $tenant)
                  AND e.embedding_stale = true
                RETURN e.name AS name, e.type AS type,
                       e.description AS description, e.tenant AS tenant
                LIMIT $batch
                """,
                tenant=tenant,
                batch=batch_size,
            )
            if not rows:
                break

            texts = [r.get("description") or r["name"] for r in rows]
            try:
                new_embeddings = await embedder(texts)
            except Exception as exc:
                log.error("embedding_registry.embedder_failed", error=str(exc))
                break

            for row, emb in zip(rows, new_embeddings):
                await self._neo4j.run(
                    """
                    MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
                    SET e.embedding         = $embedding,
                        e.embedding_model   = $model,
                        e.embedding_version = $version,
                        e.embedding_stale   = false,
                        e.re_embedded_at    = datetime()
                    """,
                    name=row["name"],
                    type=row["type"],
                    tenant=row["tenant"],
                    embedding=emb,
                    model=current_model,
                    version=current_version,
                )
            total += len(rows)
            log.info(
                "embedding_registry.batch_complete",
                batch=len(rows),
                total_so_far=total,
                tenant=tenant,
            )

        log.info("embedding_registry.re_embed_done", total=total, tenant=tenant)
        return total

    # ── Version record ─────────────────────────────────────────────────────────

    async def record_version(
        self,
        model: str,
        version: str,
        dim: int,
        notes: str = "",
    ) -> None:
        """
        Persist an EmbeddingModelVersion audit node so the history of model
        changes is queryable even after all entities are migrated.
        """
        await self._neo4j.run(
            """
            MERGE (v:EmbeddingModelVersion {model: $model, version: $version})
            ON CREATE SET v.id         = $id,
                          v.dim        = $dim,
                          v.notes      = $notes,
                          v.registered_at = datetime()
            """,
            id=str(uuid4()),
            model=model,
            version=version,
            dim=dim,
            notes=notes,
        )
        log.info("embedding_registry.version_recorded", model=model, version=version, dim=dim)
