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
        alpha: float = 0.6,       # weight for cross-encoder / vector score
        beta: float = 0.4,        # weight for GNN structural score
    ):
        if gnn_type not in ("gcn", "gat"):
            raise ValueError(f"gnn_type must be 'gcn' or 'gat', got {gnn_type!r}")
        self._type = gnn_type
        self._num_layers = num_layers
        self._alpha = alpha
        self._beta = beta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        query_vec: list[float],
        chunks: list[dict],
        chunk_entities: list[dict],   # [{chunk_id, entity_name, embedding}]
        entity_edges: list[dict],     # [{src, tgt, weight}]
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
        if not chunk_entities:
            log.info("gnn_scorer.skip", reason="no_entity_embeddings")
            for c in chunks:
                c["gnn_score"] = 0.0
                c["final_score"] = self._fallback_score(c)
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
        for edge in entity_edges:
            i = entity_idx.get(edge["src"])
            j = entity_idx.get(edge["tgt"])
            if i is not None and j is not None and i != j:
                w = float(edge.get("weight") or 1.0)
                A[i, j] = w
                A[j, i] = w   # treat as undirected
                edge_count += 1

        # ── GNN propagation ───────────────────────────────────────────
        H_out = self._propagate(A, H)

        # ── L2-normalise output embeddings for cosine scoring ─────────
        norms = np.linalg.norm(H_out, axis=1, keepdims=True) + 1e-8
        H_norm = H_out / norms                          # (N, D), unit vectors

        # ── Build chunk → entity indices map ─────────────────────────
        chunk_to_eidxs: dict[str, list[int]] = {}
        for r in chunk_entities:
            chunk_to_eidxs.setdefault(r["chunk_id"], []).append(
                entity_idx[r["entity_name"]]
            )

        # ── Score & blend ─────────────────────────────────────────────
        for chunk in chunks:
            cid   = chunk["chunk_id"]
            eidxs = chunk_to_eidxs.get(cid, [])
            if eidxs:
                # Max cosine similarity over all mentioned entities
                gnn_score = float((H_norm[eidxs] @ q).max())
                # Clip to [0, 1] (cosine can be negative)
                gnn_score = max(0.0, gnn_score)
            else:
                gnn_score = 0.0

            chunk["gnn_score"]   = gnn_score
            chunk["final_score"] = (
                self._alpha * self._fallback_score(chunk)
                + self._beta * gnn_score
            )

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
    def _fallback_score(chunk: dict) -> float:
        """Normalise existing score to [0, 1] for blending.

        Cross-encoder logits are unbounded; apply sigmoid(x/5) to map
        them to a comparable range.  Vector scores are already in [0, 1].
        """
        if "rerank_score" in chunk:
            x = float(chunk["rerank_score"])
            return float(1.0 / (1.0 + np.exp(-x / 5.0)))   # sigmoid
        return float(chunk.get("score", 0.0))
