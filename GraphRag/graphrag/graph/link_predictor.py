"""Link prediction inference interface using trained TransE embeddings.

``TransXTrainer`` (``graphrag.graph.transx_trainer``) handles the SGD training
loop and stores learned relation vectors in the shared ``_rel_emb`` dict.
``LinkPredictor`` wraps that trained model to answer three completion queries:

  - ``predict_tail(head, relation)``     — (h, r, ?) rank candidate tails
  - ``predict_relation(head, tail)``     — (h, ?, t) rank candidate relations
  - ``find_missing_links(entity_ids)``   — discover high-confidence absent edges

The TransE score for a triple is ``‖h + r − t‖₂`` (lower = more plausible).
This is inverted to a similarity score in [0, 1] for consistent ranking:
``sim = 1 / (1 + dist)``.

When no trained relation embeddings exist for a requested relation the
predictor degrades gracefully to a raw cosine ANN from the head embedding.

Usage::

    from graphrag.graph.link_predictor import LinkPredictor
    from graphrag.graph.transx_trainer import TransXTrainer
    from graphrag.graph.neo4j_client import get_neo4j

    rel_emb = {}
    trainer = TransXTrainer(get_neo4j(), rel_emb=rel_emb, embed_dim=768)
    await trainer.train(tenant="aerospace", epochs=50)

    predictor = LinkPredictor(get_neo4j(), trainer)
    candidates = await predictor.predict_tail(
        "FAA-AD-2024-01-02", "SUPERSEDES", top_k=5, tenant="aerospace"
    )
"""

from __future__ import annotations

import math

import structlog

log = structlog.get_logger(__name__)


# ── Pure-Python vector math (no numpy dep) ─────────────────────────────────────

def _l2_norm(v: list[float]) -> list[float]:
    """Normalise a vector to unit length."""
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _translate(head: list[float], rel: list[float]) -> list[float]:
    """TransE query vector: L2-normalise(h + r)."""
    dim = min(len(head), len(rel))
    return _l2_norm([head[i] + rel[i] for i in range(dim)])


def _transe_similarity(
    head: list[float],
    rel: list[float],
    tail: list[float],
) -> float:
    """Convert TransE distance ‖h + r − t‖₂ to a [0, 1] similarity score."""
    dim  = min(len(head), len(rel), len(tail))
    dist = math.sqrt(
        sum((head[i] + rel[i] - tail[i]) ** 2 for i in range(dim))
    ) + 1e-8
    return 1.0 / (1.0 + dist)


# ── LinkPredictor ──────────────────────────────────────────────────────────────

class LinkPredictor:
    """Predict missing knowledge graph links using TransE relation embeddings.

    Parameters
    ----------
    neo4j_client :
        Async Neo4j client for embedding look-ups and ANN queries.
    trainer :
        A ``TransXTrainer`` instance whose ``_rel_emb`` dict contains
        trained relation vectors.  Call ``trainer.train()`` before using
        this class.
    """

    def __init__(self, neo4j_client, trainer) -> None:
        self._neo4j   = neo4j_client
        self._trainer = trainer

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _get_entity_embedding(
        self, entity_id: str, tenant: str
    ) -> list[float]:
        """Fetch entity embedding from Neo4j; raise ValueError if absent."""
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})
            WHERE e.id = $eid OR e.name = $eid
            RETURN e.embedding AS emb
            LIMIT 1
            """,
            tenant=tenant,
            eid=entity_id,
        )
        if not rows or rows[0].get("emb") is None:
            raise ValueError(
                f"Entity not found or missing embedding: {entity_id!r}"
            )
        return list(rows[0]["emb"])

    async def _ann_search(
        self,
        query_vector: list[float],
        top_k: int,
        tenant: str,
        exclude_ids: list[str] | None = None,
    ) -> list[dict]:
        """ANN search over entity embeddings using the Neo4j vector index."""
        rows = await self._neo4j.run(
            """
            CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $qv)
            YIELD node AS e, score
            WHERE ($tenant = 'default' OR e.tenant = $tenant)
              AND NOT e.quarantined = true
            RETURN e.id        AS entity_id,
                   e.name      AS name,
                   e.type      AS type,
                   e.embedding AS embedding,
                   score
            LIMIT $top_k
            """,
            top_k=top_k * 2,   # over-fetch so filtering excluded IDs still returns top_k
            qv=query_vector,
            tenant=tenant,
        )
        excluded = set(exclude_ids or [])
        return [r for r in rows if r.get("entity_id") not in excluded][:top_k]

    # ── Public API ─────────────────────────────────────────────────────────────

    async def predict_tail(
        self,
        head_id: str,
        relation: str,
        top_k: int = 10,
        tenant: str = "default",
    ) -> list[dict]:
        """Given (head, relation, ?), rank candidate tail entities.

        Uses the TransE projection ``h + r`` as the ANN query vector, then
        re-scores results with the full TransE score using each candidate's
        stored embedding.

        If no trained embedding exists for ``relation``, falls back to a
        raw cosine ANN from the head embedding (no relation translation).

        Parameters
        ----------
        head_id :
            Entity ``id`` or ``name`` as stored in Neo4j.
        relation :
            Relation name (case-insensitive, e.g. ``"SUPERSEDES"``).
        top_k :
            Number of candidates to return.
        tenant :
            Tenant scope for the query.

        Returns
        -------
        list[dict]
            ``[{entity_id, name, type, score}]`` sorted by score descending.
            Score is in [0, 1] — higher means more plausible completion.

        Raises
        ------
        ValueError
            If the head entity is not found or has no embedding.
        """
        head_emb = await self._get_entity_embedding(head_id, tenant)
        rel_key  = relation.upper()
        rel_emb  = self._trainer._rel_emb.get(rel_key)

        if rel_emb is not None:
            query_vec = _translate(head_emb, list(rel_emb))
        else:
            log.warning("link_predictor.no_rel_emb",
                        relation=rel_key, fallback="cosine_heuristic")
            query_vec = _l2_norm(head_emb)

        candidates = await self._ann_search(
            query_vector=query_vec,
            top_k=top_k,
            tenant=tenant,
            exclude_ids=[head_id],
        )

        results: list[dict] = []
        for c in candidates:
            tail_emb = list(c.get("embedding") or [])
            if tail_emb and rel_emb:
                score = _transe_similarity(head_emb, list(rel_emb), tail_emb)
            else:
                score = float(c.get("score", 0.0))
            results.append(
                {
                    "entity_id": c.get("entity_id", ""),
                    "name":      c.get("name", ""),
                    "type":      c.get("type", ""),
                    "score":     round(score, 4),
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        log.info("link_predictor.predict_tail",
                 head=head_id, relation=rel_key, candidates=len(results))
        return results[:top_k]

    async def predict_relation(
        self,
        head_id: str,
        tail_id: str,
        top_k: int = 10,
        tenant: str = "default",
    ) -> list[dict]:
        """Given (head, ?, tail), rank candidate relations by TransE score.

        Scores every known relation in ``trainer._rel_emb`` and returns the
        top-k by plausibility.

        Parameters
        ----------
        head_id :
            Entity ``id`` or ``name`` for the source entity.
        tail_id :
            Entity ``id`` or ``name`` for the target entity.
        top_k :
            Number of relations to return.
        tenant :
            Tenant scope for embedding look-ups.

        Returns
        -------
        list[dict]
            ``[{relation, score}]`` sorted by score descending.

        Raises
        ------
        ValueError
            If either entity is missing or has no embedding.
        """
        head_emb = await self._get_entity_embedding(head_id, tenant)
        tail_emb = await self._get_entity_embedding(tail_id, tenant)

        if not self._trainer._rel_emb:
            log.warning("link_predictor.no_relations_trained",
                        hint="call TransXTrainer.train() first")
            return []

        results: list[dict] = []
        for rel_key, rel_vec in self._trainer._rel_emb.items():
            score = _transe_similarity(head_emb, list(rel_vec), tail_emb)
            results.append({"relation": rel_key, "score": round(score, 4)})

        results.sort(key=lambda x: x["score"], reverse=True)
        log.info("link_predictor.predict_relation",
                 head=head_id, tail=tail_id, relations=len(results))
        return results[:top_k]

    async def find_missing_links(
        self,
        entity_ids: list[str],
        threshold: float = 0.7,
        tenant: str = "default",
    ) -> list[dict]:
        """Find high-confidence missing links within a set of entities.

        For each (entity, relation) pair, predicts the top-1 tail.  Triples
        whose predicted score exceeds ``threshold`` and which do not already
        exist in Neo4j are returned as candidate missing links.

        Parameters
        ----------
        entity_ids :
            List of entity ids or names to scan.
        threshold :
            Minimum TransE similarity score for a triple to be reported.
        tenant :
            Tenant scope.

        Returns
        -------
        list[dict]
            ``[{head, relation, tail, score}]`` sorted by score descending.
        """
        if not entity_ids:
            return []

        # Fetch existing edges to avoid re-predicting asserted facts
        existing_rows = await self._neo4j.run(
            """
            MATCH (s:Entity {tenant: $tenant})-[r:RELATES_TO]->(t:Entity {tenant: $tenant})
            WHERE s.id IN $ids OR s.name IN $ids
            RETURN s.id AS src, r.relation AS rel, t.id AS tgt
            """,
            tenant=tenant,
            ids=entity_ids,
        )
        known = {(r["src"], r["rel"], r["tgt"]) for r in existing_rows}

        missing: list[dict] = []
        for head_id in entity_ids:
            for rel_key in list(self._trainer._rel_emb.keys()):
                try:
                    candidates = await self.predict_tail(
                        head_id, rel_key, top_k=1, tenant=tenant
                    )
                except ValueError:
                    continue   # entity not found or no embedding

                for c in candidates:
                    if c["score"] < threshold:
                        continue
                    triple = (head_id, rel_key, c["entity_id"])
                    if triple not in known:
                        missing.append(
                            {
                                "head":     head_id,
                                "relation": rel_key,
                                "tail":     c["entity_id"],
                                "score":    c["score"],
                            }
                        )

        missing.sort(key=lambda x: x["score"], reverse=True)
        log.info("link_predictor.find_missing",
                 checked=len(entity_ids), found=len(missing),
                 threshold=threshold)
        return missing
