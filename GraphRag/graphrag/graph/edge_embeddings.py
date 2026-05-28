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
from typing import Iterator

import structlog

log = structlog.get_logger(__name__)

# Embedding dimension — must match Entity.embedding dimension.
# Using 768 to match typical sentence-transformer outputs; configurable.
DEFAULT_EMBED_DIM = 768
TRANSX_MARGIN = 1.0   # margin for scoring; lower = more permissive


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
        self._neo4j    = neo4j_client
        self._embed_dim = embed_dim
        # In-memory cache for relation embeddings (populated by seed / load)
        self._rel_emb: dict[str, list[float]] = {}

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

    async def get_relation_embedding(self, relation: str) -> list[float]:
        """
        Return the embedding for a relation name.

        Priority:
        1. In-memory cache (populated at startup or via seed_relation_embeddings)
        2. Stored RelationEmbedding node in Neo4j
        3. Deterministically derived from relation name (fallback)
        """
        relation = relation.upper()
        if relation in self._rel_emb:
            return self._rel_emb[relation]

        rows = await self._neo4j.run(
            """
            MATCH (re:RelationEmbedding {relation: $rel})
            RETURN re.embedding AS embedding
            """,
            rel=relation,
        )
        if rows and rows[0].get("embedding"):
            emb = list(rows[0]["embedding"])
            self._rel_emb[relation] = emb
            return emb

        # Fallback: derive deterministically
        emb = self._derive_relation_embedding(relation)
        self._rel_emb[relation] = emb
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
        results: dict[str, bool] = {}
        for rel in relations:
            rel_upper = rel.upper()
            if not overwrite:
                rows = await self._neo4j.run(
                    "MATCH (re:RelationEmbedding {relation: $rel}) RETURN count(re) AS n",
                    rel=rel_upper,
                )
                if rows and rows[0].get("n", 0) > 0:
                    results[rel_upper] = False
                    continue

            emb = self._derive_relation_embedding(rel_upper)
            self._rel_emb[rel_upper] = emb
            await self._neo4j.run(
                """
                MERGE (re:RelationEmbedding {relation: $rel})
                SET re.embedding   = $embedding,
                    re.embed_dim   = $dim,
                    re.source      = 'derived',
                    re.updated_at  = datetime()
                """,
                rel=rel_upper,
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
        # Fetch entity embeddings
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

        rel_emb     = await self.get_relation_embedding(relation)
        triple_emb  = self._combine_triple(head_emb, rel_emb, tail_emb)
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
        # Get source entity embedding
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
        rel_emb  = await self.get_relation_embedding(relation)

        # Get all candidate target entities with embeddings
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

        # Score all candidates
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

        # Sort by score ascending (lower = more plausible)
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

        rel_emb = await self.get_relation_embedding(relation)
        return round(self._transx_score(head_emb, rel_emb, tail_emb), 4)

    # ── TransE training loop ───────────────────────────────────────────────────

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

        Positive triples come from RELATES_TO edges in Neo4j.
        Negative triples are generated by corrupting the head or tail entity
        with a randomly sampled entity from the same tenant.

        Only relation embeddings are updated — entity embeddings remain fixed
        (they are pre-computed by the sentence transformer).  Updated embeddings
        are L2-normalised after each epoch and stored back to Neo4j.

        Returns a training summary with final loss per relation.

        Parameters
        ----------
        epochs      : Number of full passes over positive triples.
        lr          : SGD learning rate.
        margin      : Margin γ in: L = max(0, γ + d_pos − d_neg).
        neg_samples : How many negative triples to generate per positive.
        batch_size  : Mini-batch size for memory efficiency.
        seed        : Random seed for reproducibility.
        """
        rng = random.Random(seed)

        # ── Load positive triples ────────────────────────────────────────────
        log.info("transx_train.loading_triples", tenant=tenant)
        triple_rows = await self._neo4j.run(
            """
            MATCH (s:Entity {tenant: $tenant})-[r:RELATES_TO]->(t:Entity {tenant: $tenant})
            WHERE s.embedding IS NOT NULL AND size(s.embedding) > 0
              AND t.embedding IS NOT NULL AND size(t.embedding) > 0
            RETURN s.embedding AS head_emb,
                   r.relation  AS relation,
                   t.embedding AS tail_emb
            LIMIT 50000
            """,
            tenant=tenant,
        )
        if not triple_rows:
            log.warning("transx_train.no_triples", tenant=tenant)
            return {"epochs": 0, "triples": 0, "error": "no_triples_found"}

        # Collect unique entities for negative sampling
        entity_rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})
            WHERE e.embedding IS NOT NULL AND size(e.embedding) > 0
            RETURN e.embedding AS embedding
            LIMIT 10000
            """,
            tenant=tenant,
        )
        entity_embs = [list(r["embedding"]) for r in entity_rows]
        if len(entity_embs) < 2:
            return {"epochs": 0, "triples": 0, "error": "too_few_entities"}

        triples: list[tuple[list[float], str, list[float]]] = [
            (list(r["head_emb"]), r["relation"], list(r["tail_emb"]))
            for r in triple_rows
        ]
        n_triples = len(triples)
        log.info("transx_train.start",
                 triples=n_triples, entities=len(entity_embs),
                 epochs=epochs, lr=lr, margin=margin)

        # ── Ensure relation embeddings exist in memory ───────────────────────
        unique_rels = {rel for _, rel, _ in triples}
        for rel in unique_rels:
            await self.get_relation_embedding(rel)  # loads / derives into cache

        # ── Training loop ────────────────────────────────────────────────────
        epoch_losses: list[float] = []

        for epoch in range(epochs):
            rng.shuffle(triples)
            epoch_loss = 0.0
            n_updates  = 0

            for batch_start in range(0, n_triples, batch_size):
                batch = triples[batch_start: batch_start + batch_size]

                # Accumulate gradients per relation in this batch
                rel_grad: dict[str, list[float]] = {}
                rel_count: dict[str, int] = {}

                for head, rel, tail in batch:
                    r_vec = self._rel_emb[rel.upper()]
                    d_pos = self._transx_score(head, r_vec, tail) + 1e-8

                    for _ in range(neg_samples):
                        # Corrupt head or tail (50/50)
                        if rng.random() < 0.5:
                            neg_head = rng.choice(entity_embs)
                            neg_tail = tail
                        else:
                            neg_head = head
                            neg_tail = rng.choice(entity_embs)

                        d_neg = self._transx_score(neg_head, r_vec, neg_tail) + 1e-8
                        loss = max(0.0, margin + d_pos - d_neg)
                        if loss <= 0.0:
                            continue

                        epoch_loss += loss
                        n_updates  += 1

                        # Gradient of d_pos w.r.t. r: (h + r - t) / d_pos
                        dim = len(r_vec)
                        pos_diff = [
                            head[i] + r_vec[i] - tail[i]
                            for i in range(dim)
                        ]
                        grad_pos = [pos_diff[i] / d_pos for i in range(dim)]

                        # Gradient of d_neg w.r.t. r: (h' + r - t') / d_neg
                        neg_diff = [
                            neg_head[i] + r_vec[i] - neg_tail[i]
                            for i in range(dim)
                        ]
                        grad_neg = [neg_diff[i] / d_neg for i in range(dim)]

                        # Combined gradient: grad_pos - grad_neg
                        grad = [grad_pos[i] - grad_neg[i] for i in range(dim)]

                        rel_key = rel.upper()
                        if rel_key not in rel_grad:
                            rel_grad[rel_key]  = [0.0] * dim
                            rel_count[rel_key] = 0
                        for i in range(dim):
                            rel_grad[rel_key][i] += grad[i]
                        rel_count[rel_key] += 1

                # Apply averaged gradient update + L2 normalise
                for rel_key, grad_acc in rel_grad.items():
                    n = rel_count[rel_key] or 1
                    r_vec = self._rel_emb[rel_key]
                    dim   = len(r_vec)
                    updated = [r_vec[i] - lr * (grad_acc[i] / n) for i in range(dim)]
                    # L2 normalise
                    norm = math.sqrt(sum(x * x for x in updated)) or 1.0
                    self._rel_emb[rel_key] = [x / norm for x in updated]

            avg_loss = epoch_loss / max(n_updates, 1)
            epoch_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                log.info(
                    "transx_train.epoch",
                    epoch=epoch + 1,
                    avg_loss=round(avg_loss, 5),
                    n_updates=n_updates,
                )

        # ── Persist learned embeddings to Neo4j ──────────────────────────────
        stored = 0
        for rel_key, emb in self._rel_emb.items():
            await self._neo4j.run(
                """
                MERGE (re:RelationEmbedding {relation: $rel})
                SET re.embedding   = $embedding,
                    re.embed_dim   = $dim,
                    re.source      = 'trained',
                    re.updated_at  = datetime()
                """,
                rel=rel_key,
                embedding=emb,
                dim=self._embed_dim,
            )
            stored += 1

        summary = {
            "triples":      n_triples,
            "entities":     len(entity_embs),
            "epochs":       epochs,
            "final_loss":   round(epoch_losses[-1], 5) if epoch_losses else 0,
            "loss_curve":   [round(l, 5) for l in epoch_losses[::max(1, epochs // 20)]],
            "relations_updated": stored,
        }
        log.info("transx_train.complete", **summary)
        return summary
