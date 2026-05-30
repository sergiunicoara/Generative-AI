"""GNN-based chunk re-scorer using GCN or GAT message passing.

Architecture overview
---------------------
After cross-encoder reranking and multi-hop graph traversal, the GNN scorer
computes *graph-aware* entity representations by propagating information across
the RELATES_TO entity subgraph, then blends those structural scores with the
cross-encoder scores to produce a final ranking.

Pipeline position
-----------------
    Vector ANN + BM25 → RRF → Cross-encoder → Multi-hop → GNNScorer → LLM
                                                              ↑ HERE

Why GNN here?
-------------
The cross-encoder scores (query, chunk_text) pairs in isolation — it ignores
graph topology.  The GNN adds structural signal: an entity that is *centrally
connected* to query-relevant entities receives higher weight even when its direct
text match is weak.  This is especially valuable for multi-document reasoning
where the answer is spread across a chain of entities.

GCN layer  (Kipf & Welling 2017)
---------------------------------
    H'  = ReLU( D̃⁻¹/² Ã D̃⁻¹/²  H )
    Ã   = A + I    (adjacency with added self-loops)
    D̃   = degree matrix of Ã

GAT layer  (Veličković et al. 2018) — attention via cosine similarity
----------------------------------------------------------------------
    α_ij = softmax_j( cos(h_i, h_j) )  for j ∈ N(i) ∪ {i}
    h'_i = Σ_j  α_ij · h_j

Both variants are zero-shot (no training) — they leverage the existing
Gemini text-embedding-004 entity embeddings stored in Neo4j.

Scoring
-------
For each chunk c, the GNN score is:
    gnn_score(c) = max_{e ∈ mentions(c)}  cos( q_gnn, e_gnn )

where q_gnn is the query vector and e_gnn is the post-propagation entity vector.

Final blend:
    final_score = α · sigmoid(cross_encoder_score / 5) + β · gnn_score
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import structlog

log = structlog.get_logger(__name__)


# ── GCN ───────────────────────────────────────────────────────────────────────

def _normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Symmetric GCN normalization: Â = D̃⁻¹/² Ã D̃⁻¹/²  where Ã = A + I."""
    A_hat = A + np.eye(A.shape[0], dtype=np.float32)
    row_sum = A_hat.sum(axis=1)
    d_inv_sqrt = np.where(row_sum > 0, 1.0 / np.sqrt(row_sum), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return (D_inv_sqrt @ A_hat @ D_inv_sqrt).astype(np.float32)


def _gcn_layer(A_norm: np.ndarray, H: np.ndarray, last: bool = False) -> np.ndarray:
    """Single GCN message-passing layer.

    H' = ReLU( Â H )   — no learnable W; uses identity projection.
    Last layer skips ReLU to preserve signed cosine similarity range.
    """
    out = A_norm @ H
    return out if last else np.maximum(0.0, out)


# ── GAT ───────────────────────────────────────────────────────────────────────

def _gat_layer(A: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Single GAT message-passing layer with cosine-similarity attention.

    For each node i:
        α_ij = softmax_j( cos(h_i, h_j) )  for j in N(i) ∪ {i}
        h'_i = Σ_j  α_ij · h_j

    Nodes with no edges fall back to their own embedding (self-loop).
    """
    N = H.shape[0]

    # L2-normalise for cosine similarity
    norms = np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
    H_hat = H / norms                         # (N, D)

    # Raw attention scores: cos(h_i, h_j) for all pairs
    cos_sim = H_hat @ H_hat.T                 # (N, N)

    # Build neighbour mask (include self-loops)
    mask = (A > 0).astype(np.float32)
    np.fill_diagonal(mask, 1.0)

    # Mask non-neighbours with large negative so softmax → 0
    cos_sim_masked = np.where(mask > 0, cos_sim, -1e9)

    # Numerically stable softmax along axis=1
    cos_max = cos_sim_masked.max(axis=1, keepdims=True)
    exp_vals = np.exp(cos_sim_masked - cos_max) * mask
    alpha = exp_vals / (exp_vals.sum(axis=1, keepdims=True) + 1e-8)   # (N, N)

    return alpha @ H   # weighted aggregation — (N, D)


# ── Scorer ────────────────────────────────────────────────────────────────────

class GNNScorer:
    """Graph-aware chunk scorer using GCN or GAT message passing.

    Usage (called from LocalSearch after multi-hop graph traversal)::

        scorer = GNNScorer(gnn_type="gat", num_layers=2)
        chunks = scorer.score(
            query_vec       = embedding,       # list[float] — 768d query embedding
            chunks          = all_chunks,      # list[dict]  — from retrieval
            chunk_entities  = chunk_entities,  # list[dict]  — chunk_id + entity + emb
            entity_edges    = entity_edges,    # list[dict]  — src + tgt + weight
        )

    Each chunk dict gains two new fields after scoring:
        ``gnn_score``   — [0, 1] structural score from GNN
        ``final_score`` — α·norm(rerank) + β·gnn_score
    """

    def __init__(
        self,
        gnn_type: str = "gat",   # "gcn" | "gat"
        num_layers: int = 2,
        alpha: float = 0.9,       # weight for cross-encoder / vector score
        beta: float = 0.1,        # weight for GNN structural score
        edge_confidence_threshold: float = 0.7,  # drop edges below this confidence
        confidence_half_life_days: int = 0,       # 0 = no decay
    ):
        if gnn_type not in ("gcn", "gat"):
            raise ValueError(f"gnn_type must be 'gcn' or 'gat', got {gnn_type!r}")
        self._type = gnn_type
        self._num_layers = num_layers
        self._alpha = alpha
        self._beta = beta
        self._edge_conf_threshold = edge_confidence_threshold
        self._half_life_days = confidence_half_life_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        query_vec: list[float],
        chunks: list[dict],
        chunk_entities: list[dict],   # [{chunk_id, entity_name, embedding}]
        entity_edges: list[dict],     # [{src, tgt, weight, confidence, extracted_at}]
        alpha: float | None = None,   # override instance default (query-adaptive)
        beta: float | None = None,
    ) -> list[dict]:
        """Compute GNN-blended scores and re-sort chunks in-place.

        Args:
            query_vec:       768-d query embedding (raw, unnormalised).
            chunks:          Retrieved chunks with optional ``rerank_score`` /
                             ``score`` fields.
            chunk_entities:  Rows mapping chunk_id → entity_name + embedding.
                             Entities without embeddings should be pre-filtered.
            entity_edges:    RELATES_TO edges between entities in this subgraph.

        Returns:
            Same chunk list, re-sorted by ``final_score`` (descending).
        """
        alpha = alpha if alpha is not None else self._alpha
        beta  = beta  if beta  is not None else self._beta

        if not chunk_entities:
            log.info("gnn_scorer.skip", reason="no_entity_embeddings")
            for c in chunks:
                text_score       = self._text_score(c)
                c["gnn_score"]   = 0.0
                # No graph evidence — weight falls entirely on text score.
                # Applying alpha preserves the documented formula when gnn=0:
                #   final = α·text + β·0 = α·text
                c["final_score"] = alpha * text_score
                c["explanation"] = f"GNN skipped: no entities (text={text_score:.3f})"
            chunks.sort(key=lambda c: c["final_score"], reverse=True)
            return chunks

        q = np.array(query_vec, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-8                  # unit vector

        # ── Build entity index ────────────────────────────────────────
        entity_names = list({r["entity_name"] for r in chunk_entities})
        entity_idx   = {name: i for i, name in enumerate(entity_names)}
        N            = len(entity_names)
        emb_dim      = len(chunk_entities[0]["embedding"])

        H = np.zeros((N, emb_dim), dtype=np.float32)
        for r in chunk_entities:
            H[entity_idx[r["entity_name"]]] = np.array(r["embedding"], dtype=np.float32)

        # ── Build adjacency matrix ────────────────────────────────────
        A = np.zeros((N, N), dtype=np.float32)
        edge_count = 0
        skipped_edges = 0
        now = datetime.now(timezone.utc)
        for edge in entity_edges:
            conf = float(edge.get("confidence", 1.0))
            # Timestamp decay: conf *= exp(-ln2 / half_life * age_days)
            if self._half_life_days > 0:
                raw_ts = edge.get("extracted_at")
                if raw_ts:
                    try:
                        ts = datetime.fromisoformat(str(raw_ts))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        age_days = (now - ts).days
                        conf *= math.exp(-math.log(2) / self._half_life_days * age_days)
                    except (ValueError, TypeError):
                        pass
            if conf < self._edge_conf_threshold:
                skipped_edges += 1
                continue
            i = entity_idx.get(edge["src"])
            j = entity_idx.get(edge["tgt"])
            if i is not None and j is not None and i != j:
                w = float(edge.get("weight") or 1.0)
                A[i, j] = w
                A[j, i] = w   # treat as undirected
                edge_count += 1
        if skipped_edges:
            log.info("gnn_scorer.edges_filtered",
                     skipped=skipped_edges,
                     threshold=self._edge_conf_threshold)

        # ── Hub dampening ─────────────────────────────────────────────
        # High-degree hubs dominate propagation even when only weakly relevant.
        # Apply per-node penalty: 1 / log(1 + degree).
        # Use graph-level degree from Neo4j (passed via chunk_entity_embeddings
        # 'degree' field) when available; fall back to local adjacency sum.
        degree_from_graph: dict[str, int] = {}
        for r in chunk_entities:
            deg = r.get("degree")
            if deg is not None:
                degree_from_graph[r["entity_name"]] = int(deg)

        hub_penalties = np.ones(N, dtype=np.float32)
        for name, idx in entity_idx.items():
            deg = degree_from_graph.get(name, int(A[idx].sum()))
            hub_penalties[idx] = float(1.0 / math.log1p(max(deg, 1)))

        # A_hub[i,j] = A[i,j] * min(penalty_i, penalty_j)
        hub_mat = np.minimum(hub_penalties[:, None], hub_penalties[None, :])
        A = A * hub_mat

        # ── GNN propagation ───────────────────────────────────────────
        H_out = self._propagate(A, H)

        # ── L2-normalise output embeddings for cosine scoring ─────────
        norms = np.linalg.norm(H_out, axis=1, keepdims=True) + 1e-8
        H_norm = H_out / norms                          # (N, D), unit vectors

        # ── Build chunk → entity indices map ─────────────────────────
        idx_to_name: dict[int, str] = {i: n for n, i in entity_idx.items()}
        idx_to_type: dict[int, str] = {}
        for r in chunk_entities:
            idx_to_type[entity_idx[r["entity_name"]]] = r.get("entity_type", "")
        chunk_to_eidxs: dict[str, list[int]] = {}
        for r in chunk_entities:
            chunk_to_eidxs.setdefault(r["chunk_id"], []).append(
                entity_idx[r["entity_name"]]
            )

        _SKIP_TYPES = {"LOCATION"}   # don't use these as explanation labels

        # ── Score & blend ─────────────────────────────────────────────
        for chunk in chunks:
            cid   = chunk["chunk_id"]
            eidxs = chunk_to_eidxs.get(cid, [])
            if eidxs:
                entity_scores = H_norm[eidxs] @ q          # cosine per entity
                best_pos      = int(entity_scores.argmax())
                gnn_score     = float(max(0.0, entity_scores[best_pos]))
                # For the label, prefer non-LOCATION entities
                label_pos = best_pos
                for rank_pos in entity_scores.argsort()[::-1]:
                    if idx_to_type.get(eidxs[rank_pos]) not in _SKIP_TYPES:
                        label_pos = rank_pos
                        break
                best_entity = idx_to_name[eidxs[label_pos]]
                via           = chunk.get("via_entity", "")
                if via and via != best_entity:
                    explanation = (
                        f"Via {via} → {best_entity} "
                        f"(gnn={gnn_score:.3f}, rerank={chunk.get('rerank_score', 0):.2f})"
                    )
                else:
                    explanation = (
                        f"Top entity: {best_entity} "
                        f"(gnn={gnn_score:.3f}, rerank={chunk.get('rerank_score', 0):.2f})"
                    )
            else:
                gnn_score   = 0.0
                explanation = f"No linked entities (rerank={chunk.get('rerank_score', 0):.2f})"

            text_score           = self._text_score(chunk)
            chunk["gnn_score"]   = gnn_score
            chunk["text_score"]  = text_score
            chunk["explanation"] = explanation
            # Documented formula: final = α·text_score + β·gnn_score
            chunk["final_score"] = alpha * text_score + beta * gnn_score

        chunks.sort(key=lambda c: c["final_score"], reverse=True)

        log.info(
            "gnn_scorer.done",
            gnn_type=self._type,
            layers=self._num_layers,
            entities=N,
            edges=edge_count,
            chunks=len(chunks),
            alpha=self._alpha,
            beta=self._beta,
            top_gnn=round(chunks[0]["gnn_score"],   4) if chunks else 0,
            top_final=round(chunks[0]["final_score"], 4) if chunks else 0,
        )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propagate(self, A: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Run num_layers of GCN or GAT message passing."""
        if self._type == "gcn":
            A_norm = _normalize_adjacency(A)
            for layer in range(self._num_layers):
                last = layer == self._num_layers - 1
                H = _gcn_layer(A_norm, H, last=last)
        else:  # gat
            for _ in range(self._num_layers):
                H = _gat_layer(A, H)
        return H

    @staticmethod
    def _text_score(chunk: dict) -> float:
        """Normalise the text-based score to [0, 1] for blending.

        Cross-encoder logits are unbounded; apply sigmoid(x/5) to bring
        them into a range comparable to cosine similarity scores.
        Vector scores are already in [0, 1] and are returned as-is.

        This is the *text* component of the final blend formula:
            final = α · text_score + β · gnn_score
        """
        if "rerank_score" in chunk:
            x = float(chunk["rerank_score"])
            return float(1.0 / (1.0 + np.exp(-x / 5.0)))   # sigmoid
        return float(chunk.get("score", 0.0))

    # ------------------------------------------------------------------
    # Confidence propagation
    # ------------------------------------------------------------------

    @staticmethod
    def propagate_confidence(
        entity_edges: list[dict],
        seed_entities: list[str],
        max_hops: int = 3,
        min_confidence: float = 0.05,
    ) -> dict[str, float]:
        """
        Propagate confidence outward from seed entities through the edge graph.

        Confidence decays multiplicatively along each path:
            conf(path h→r→t) = conf(edge h→t) * conf(h)

        At each entity, we keep the *maximum* confidence across all paths that
        reach it.  This is equivalent to finding the most-confident path from any
        seed to each reachable entity.

        Use this to annotate chunk results with the highest-confidence path
        connecting any seed entity to a chunk's mention entities.

        Parameters
        ----------
        entity_edges   : RELATES_TO edge dicts — each must have ``src``, ``tgt``,
                         and ``confidence`` fields.
        seed_entities  : Entity names assigned confidence 1.0 at the start.
        max_hops       : Maximum number of hops to propagate (prevents runaway).
        min_confidence : Prune paths whose confidence drops below this threshold.

        Returns
        -------
        dict[entity_name → max_path_confidence]
            Includes seeds (confidence 1.0) and all reachable entities.
        """
        # Build adjacency: entity_name → [(neighbour, edge_confidence)]
        adj: dict[str, list[tuple[str, float]]] = {}
        for edge in entity_edges:
            src  = edge.get("src") or edge.get("source") or ""
            tgt  = edge.get("tgt") or edge.get("target") or ""
            conf = float(edge.get("confidence", 1.0))
            if not src or not tgt:
                continue
            adj.setdefault(src, []).append((tgt, conf))
            adj.setdefault(tgt, []).append((src, conf))   # undirected propagation

        # BFS / Bellman-Ford with max-product semantics
        best: dict[str, float] = {e: 1.0 for e in seed_entities}
        frontier = list(seed_entities)

        for _hop in range(max_hops):
            next_frontier: list[str] = []
            for entity in frontier:
                current_conf = best[entity]
                for neighbour, edge_conf in adj.get(entity, []):
                    propagated = current_conf * edge_conf
                    if propagated < min_confidence:
                        continue
                    if propagated > best.get(neighbour, 0.0):
                        best[neighbour] = propagated
                        next_frontier.append(neighbour)
            if not next_frontier:
                break
            frontier = next_frontier

        return best

    @staticmethod
    def annotate_path_confidence(
        chunks: list[dict],
        chunk_entities: list[dict],
        path_confidences: dict[str, float],
    ) -> list[dict]:
        """
        Stamp each chunk with the maximum path confidence across its mention entities.

        Mutates chunks in-place — adds ``path_confidence`` field.

        Parameters
        ----------
        chunks            : Chunk dicts (already scored by ``score()``).
        chunk_entities    : [{chunk_id, entity_name, ...}, ...]
        path_confidences  : Output of ``propagate_confidence()``.

        Returns the same chunks list for chaining.
        """
        # Build chunk_id → max entity confidence
        chunk_max: dict[str, float] = {}
        for row in chunk_entities:
            cid  = row["chunk_id"]
            ename = row["entity_name"]
            conf = path_confidences.get(ename, 0.0)
            if conf > chunk_max.get(cid, 0.0):
                chunk_max[cid] = conf

        for chunk in chunks:
            chunk["path_confidence"] = round(chunk_max.get(chunk["chunk_id"], 0.0), 4)

        return chunks
