"""Edge embeddings — TransE-style triple embeddings for link prediction.

Problems solved
---------------
1. No structural link prediction — the GNN scores existing paths but cannot
   suggest plausible missing edges.  A TransE model can score unseen triples
   and surface the most likely missing relations.

2. Relation-type blindness in GNN — the current GNN aggregation treats all
   edge types equally.  Relation-type embeddings allow the model to distinguish
   "MANUFACTURES" from "CEO_OF" in the scoring layer.

3. No embedding for edges — only entities and chunks have stored embeddings.
   Edges participate in the adjacency matrix by weight/confidence only.
   Adding triple embeddings enables edge-aware retrieval.

TransE model
------------
TransE models each relation r as a translation in entity embedding space:
    score(h, r, t) = ‖ h + r - t ‖₂

A plausible triple has a low score.  Learning: push observed triples to low
scores while random triples score high.

In this implementation:
  - Entity embeddings come from the existing Entity.embedding field.
  - Relation embeddings are deterministically derived from the relation name
    (seeded from a hash) if no learned embedding is stored.  Learned embeddings
    are stored in RelationEmbedding nodes and take precedence.
  - Triple embeddings = concatenation of (h, r, t) for storage and retrieval.
  - Link prediction: score all candidate (entity, relation) pairs against a
    query entity and return the highest-scoring unseen relations.

Architecture
------------
- EdgeEmbeddingService computes and stores relation embeddings.
- embed_triple() derives the composite (h + r - t) vector for a triple.
- store_triple_embedding() persists it on the RELATES_TO edge.
- predict_missing_links() returns scored candidate (target, relation) pairs.
- get_relation_embedding() returns or derives the embedding for a relation name.
- train() delegates to TransXTrainer (graphrag.graph.transx_trainer) to keep
  the 190-line SGD loop out of this file.

Relation embedding derivation (deterministic fallback)
------------------------------------------------------
Without training, we derive a stable pseudo-random vector from the relation
name using a seeded RNG (seeded by hash of the name).  This preserves the
identity property: same relation always maps to the same vector, different
relations map to different vectors.  The vectors are L2-normalized.
"""

from __future__ import annotations

import hashlib
import math
import random

import structlog

log = structlog.get_logger(__name__)

# Embedding dimension — must match Entity.embedding dimension.
# Using 768 to match typical sentence-transformer outputs; configurable.
DEFAULT_EMBED_DIM = 768
TRANSX_MARGIN = 1.0   # margin for scoring; lower = more permissive

# Scope marker for relation embeddings that are NOT tenant-specific.
#
# RelationEmbedding nodes have two sources with different ownership:
#   - 'derived'  — _derive_relation_embedding(), a pure function of the
#                  relation NAME (SHA-256 seed -> fixed RNG draw). Identical
#                  for every tenant, so storing one shared copy is a cache,
#                  not a leak.
#   - 'trained'  — TransE fitted to ONE tenant's edges. Tenant-specific.
#
# Both used to MERGE on {relation} alone, so a tenant that trained embeddings
# silently overwrote the vector every other tenant's link prediction reads.
# Tenant is now part of node identity in both cases; derived nodes carry this
# sentinel so a MERGE for the shared copy can never accidentally match (and
# overwrite) some tenant's trained node that happens to share the name.
# See docs/context_graph_gap_plan.md F13.
DERIVED_SCOPE = "__derived__"


class EdgeEmbeddingService:
    """
    Compute, store, and query TransE-style triple embeddings.

    Usage::

        svc = EdgeEmbeddingService(neo4j_client, embed_dim=768)

        # Store relation embeddings (call once or after model update)
        await svc.seed_relation_embeddings(["CEO_OF", "MANUFACTURES", "USES"])

        # Compute and store triple embedding for a specific edge
        await svc.store_triple_embedding(
            src_name="Elon Musk",  src_type="PERSON",
            relation="CEO_OF",
            tgt_name="SpaceX",     tgt_type="ORG",
            tenant="default",
        )

        # Batch-embed all edges for a tenant
        count = await svc.embed_all_edges(tenant="acme")

        # Link prediction: which entities is "SpaceX" most likely CEO_OF?
        candidates = await svc.predict_missing_links(
            entity_name="SpaceX", entity_type="ORG",
            relation="FOUNDED_BY", tenant="acme", top_k=5,
        )
    """

    def __init__(self, neo4j_client, embed_dim: int = DEFAULT_EMBED_DIM):
        self._neo4j     = neo4j_client
        self._embed_dim = embed_dim
        # In-memory cache for relation embeddings, keyed (tenant, relation).
        # Derived (tenant-independent) entries are cached under DERIVED_SCOPE.
        self._rel_emb: dict[tuple[str, str], list[float]] = {}

    # ── Relation embeddings ────────────────────────────────────────────────────

    def _derive_relation_embedding(self, relation: str) -> list[float]:
        """
        Deterministically derive a relation embedding from its name.

        Uses a seeded RNG (seed = SHA-256 hash of uppercased relation name)
        to produce a stable, reproducible unit vector.  Two relation names
        that are identical produce the same vector; distinct names produce
        different vectors.

        This is the fallback for relations with no learned embedding.
        """
        seed = int(hashlib.sha256(relation.upper().encode()).hexdigest(), 16) % (2 ** 32)
        rng  = random.Random(seed)
        raw  = [rng.gauss(0.0, 1.0) for _ in range(self._embed_dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    async def get_relation_embedding(self, relation: str, *, tenant: str) -> list[float]:
        """
        Return the embedding for a relation name, as seen by one tenant.

        Priority:
        1. In-memory cache, keyed by (tenant, relation)
        2. This tenant's own TRAINED RelationEmbedding node
        3. The shared DERIVED node (identical for every tenant by construction)
        4. Deterministically derived in-process (fallback)

        `tenant` is keyword-only and required: step 2 is the whole point of
        the fix, and a caller that omits it would silently read whatever
        happened to be stored under the bare relation name — which is exactly
        the cross-tenant read this replaced.
        """
        relation = relation.upper()
        cache_key = (tenant, relation)
        if cache_key in self._rel_emb:
            return self._rel_emb[cache_key]

        # A tenant's own trained vector wins over the shared derived one.
        # ORDER BY puts the tenant-scoped row first when both exist, so this
        # stays a single round-trip rather than two sequential queries.
        rows = await self._neo4j.run(
            """
            MATCH (re:RelationEmbedding {relation: $rel})
            WHERE re.tenant IN [$tenant, $derived]
            RETURN re.embedding AS embedding, re.tenant AS tenant
            ORDER BY CASE WHEN re.tenant = $tenant THEN 0 ELSE 1 END
            LIMIT 1
            """,
            rel=relation,
            tenant=tenant,
            derived=DERIVED_SCOPE,
        )
        if rows and rows[0].get("embedding"):
            emb = list(rows[0]["embedding"])
            self._rel_emb[cache_key] = emb
            return emb

        # Fallback: derive deterministically
        emb = self._derive_relation_embedding(relation)
        self._rel_emb[cache_key] = emb
        return emb

    async def seed_relation_embeddings(
        self,
        relations: list[str],
        overwrite: bool = False,
    ) -> dict[str, bool]:
        """
        Pre-compute and store relation embeddings for a list of relation names.

        If ``overwrite=False`` (default), existing stored embeddings are not
        replaced.  Returns a dict of {relation: was_seeded}.
        """
        # Derived embeddings are a pure function of the relation name, so one
        # shared node per relation is correct — this is a global cache, not a
        # per-tenant store. It is written under DERIVED_SCOPE, never a real
        # tenant name, so it can never be mistaken for (or overwritten by) a
        # tenant's trained node with the same relation name.
        results: dict[str, bool] = {}
        for rel in relations:
            rel_upper = rel.upper()
            cache_key = (DERIVED_SCOPE, rel_upper)
            if not overwrite:
                rows = await self._neo4j.run(
                    "MATCH (re:RelationEmbedding {relation: $rel, tenant: $tenant}) "
                    "RETURN count(re) AS n",
                    rel=rel_upper,
                    tenant=DERIVED_SCOPE,
                )
                if rows and rows[0].get("n", 0) > 0:
                    results[rel_upper] = False
                    continue

            emb = self._derive_relation_embedding(rel_upper)
            self._rel_emb[cache_key] = emb
            await self._neo4j.run(
                """
                MERGE (re:RelationEmbedding {relation: $rel, tenant: $tenant})
                SET re.embedding   = $embedding,
                    re.embed_dim   = $dim,
                    re.source      = 'derived',
                    re.updated_at  = datetime()
                """,
                rel=rel_upper,
                tenant=DERIVED_SCOPE,
                embedding=emb,
                dim=self._embed_dim,
            )
            results[rel_upper] = True

        log.info("edge_embeddings.relations_seeded", seeded=sum(results.values()))
        return results

    # ── Triple embeddings ──────────────────────────────────────────────────────

    def _transx_score(
        self,
        head: list[float],
        relation: list[float],
        tail: list[float],
    ) -> float:
        """
        TransE scoring function: ‖ h + r - t ‖₂

        Lower score = more plausible triple.
        """
        dim = min(len(head), len(relation), len(tail))
        diff = [head[i] + relation[i] - tail[i] for i in range(dim)]
        return math.sqrt(sum(x * x for x in diff))

    def _combine_triple(
        self,
        head: list[float],
        relation: list[float],
        tail: list[float],
    ) -> list[float]:
        """
        Return h + r - t as the composite triple embedding.

        The resulting vector can be stored and used for nearest-neighbour
        search over the triple space.
        """
        dim = min(len(head), len(relation), len(tail))
        return [head[i] + relation[i] - tail[i] for i in range(dim)]

    async def store_triple_embedding(
        self,
        src_name: str,
        src_type: str,
        relation: str,
        tgt_name: str,
        tgt_type: str,
        tenant: str = "default",
    ) -> bool:
        """
        Compute and persist the TransE triple embedding on the RELATES_TO edge.

        The embedding is stored as `triple_embedding` on the edge.
        Returns False if either entity is missing an embedding (skipped).
        """
        rows = await self._neo4j.run(
            """
            MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
            MATCH (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
            RETURN s.embedding AS head_emb, t.embedding AS tail_emb
            """,
            src_name=src_name,
            src_type=src_type,
            tgt_name=tgt_name,
            tgt_type=tgt_type,
            tenant=tenant,
        )
        if not rows:
            return False
        row = rows[0]
        head_emb = row.get("head_emb")
        tail_emb = row.get("tail_emb")
        if not head_emb or not tail_emb:
            return False

        rel_emb      = await self.get_relation_embedding(relation, tenant=tenant)
        triple_emb   = self._combine_triple(head_emb, rel_emb, tail_emb)
        transx_score = self._transx_score(head_emb, rel_emb, tail_emb)

        await self._neo4j.run(
            """
            MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
                  -[r:RELATES_TO {relation: $relation}]->
                  (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
            SET r.triple_embedding = $triple_emb,
                r.transx_score     = $score,
                r.embed_updated_at = datetime()
            """,
            src_name=src_name,
            src_type=src_type,
            tgt_name=tgt_name,
            tgt_type=tgt_type,
            relation=relation,
            tenant=tenant,
            triple_emb=triple_emb,
            score=transx_score,
        )
        return True

    async def embed_all_edges(
        self,
        tenant: str = "default",
        limit: int = 5000,
    ) -> int:
        """
        Batch-compute triple embeddings for all RELATES_TO edges in a tenant.

        Skips edges where either endpoint has no embedding.
        Returns the count of edges that were embedded.
        """
        rows = await self._neo4j.run(
            """
            MATCH (s:Entity {tenant: $tenant})-[r:RELATES_TO]->(t:Entity {tenant: $tenant})
            WHERE r.triple_embedding IS NULL
              AND s.embedding IS NOT NULL AND size(s.embedding) > 0
              AND t.embedding IS NOT NULL AND size(t.embedding) > 0
            RETURN s.name AS src_name, s.type AS src_type,
                   t.name AS tgt_name, t.type AS tgt_type,
                   r.relation AS relation
            LIMIT $limit
            """,
            tenant=tenant,
            limit=limit,
        )

        count = 0
        for row in rows:
            ok = await self.store_triple_embedding(
                src_name=row["src_name"],
                src_type=row["src_type"],
                relation=row["relation"],
                tgt_name=row["tgt_name"],
                tgt_type=row["tgt_type"],
                tenant=tenant,
            )
            if ok:
                count += 1

        log.info("edge_embeddings.batch_complete", embedded=count, tenant=tenant)
        return count

    # ── Link prediction ────────────────────────────────────────────────────────

    async def predict_missing_links(
        self,
        entity_name: str,
        entity_type: str,
        relation: str,
        tenant: str = "default",
        top_k: int = 10,
    ) -> list[dict]:
        """
        Predict the most plausible target entities for:
            (entity_name)-[relation]->(?)

        Scores all candidate entities using TransE: ‖ h + r - t ‖₂
        Lower score = more plausible.

        Only returns candidates that do NOT already have this edge, i.e.
        genuinely predicted *missing* links.

        Returns top_k results sorted by score ascending (most plausible first).
        """
        src_rows = await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
            RETURN e.embedding AS embedding
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
        )
        if not src_rows or not src_rows[0].get("embedding"):
            return []
        head_emb = list(src_rows[0]["embedding"])
        rel_emb  = await self.get_relation_embedding(relation, tenant=tenant)

        candidates = await self._neo4j.run(
            """
            MATCH (t:Entity {tenant: $tenant})
            WHERE t.embedding IS NOT NULL AND size(t.embedding) > 0
              AND NOT t.quarantined = true
              AND NOT EXISTS {
                  MATCH (src:Entity {name: $src_name, type: $src_type, tenant: $tenant})
                        -[:RELATES_TO {relation: $relation}]->(t)
              }
              AND NOT (t.name = $src_name AND t.type = $src_type)
            RETURN t.name AS name, t.type AS type, t.embedding AS embedding
            LIMIT 2000
            """,
            tenant=tenant,
            src_name=entity_name,
            src_type=entity_type,
            relation=relation,
        )

        scored: list[dict] = []
        for cand in candidates:
            tail_emb = list(cand["embedding"])
            score = self._transx_score(head_emb, rel_emb, tail_emb)
            scored.append({
                "target":   cand["name"],
                "type":     cand["type"],
                "relation": relation,
                "score":    round(score, 4),
            })

        scored.sort(key=lambda x: x["score"])
        return scored[:top_k]

    async def score_triple(
        self,
        src_name: str,
        src_type: str,
        relation: str,
        tgt_name: str,
        tgt_type: str,
        tenant: str = "default",
    ) -> float | None:
        """
        Score a specific (h, r, t) triple using TransE.

        Returns the L2 norm ‖ h + r - t ‖₂ or None if either entity
        is missing an embedding.  Lower score = more plausible.
        """
        rows = await self._neo4j.run(
            """
            MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
            MATCH (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
            RETURN s.embedding AS head_emb, t.embedding AS tail_emb
            """,
            src_name=src_name,
            src_type=src_type,
            tgt_name=tgt_name,
            tgt_type=tgt_type,
            tenant=tenant,
        )
        if not rows:
            return None
        head_emb = rows[0].get("head_emb")
        tail_emb = rows[0].get("tail_emb")
        if not head_emb or not tail_emb:
            return None

        rel_emb = await self.get_relation_embedding(relation, tenant=tenant)
        return round(self._transx_score(head_emb, rel_emb, tail_emb), 4)

    # ── TransE training (delegates to TransXTrainer) ───────────────────────────

    async def train(
        self,
        tenant: str = "default",
        epochs: int = 100,
        lr: float = 0.01,
        margin: float = 1.0,
        neg_samples: int = 5,
        batch_size: int = 256,
        seed: int = 42,
    ) -> dict:
        """
        Train TransE relation embeddings using negative-sampling SGD.

        Delegates to TransXTrainer (graphrag.graph.transx_trainer) which
        contains the full training loop.  The shared ``_rel_emb`` dict is
        passed by reference so learned embeddings are immediately reflected
        in this service's cache after training.

        Returns a summary dict: triples, entities, epochs, final_loss,
        loss_curve, relations_updated.
        """
        from graphrag.graph.transx_trainer import TransXTrainer
        trainer = TransXTrainer(
            neo4j_client=self._neo4j,
            rel_emb=self._rel_emb,   # shared reference — mutations visible here
            embed_dim=self._embed_dim,
        )
        return await trainer.train(
            tenant=tenant,
            epochs=epochs,
            lr=lr,
            margin=margin,
            neg_samples=neg_samples,
            batch_size=batch_size,
            seed=seed,
        )
