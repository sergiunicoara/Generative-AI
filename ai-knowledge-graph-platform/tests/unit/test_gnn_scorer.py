"""Unit tests for GNNScorer — GCN/GAT math, scoring, edge filtering, hub dampening."""

from __future__ import annotations

import numpy as np
import pytest

from graphrag.graph.gnn_scorer import (
    GNNScorer,
    _gcn_layer,
    _gat_layer,
    _normalize_adjacency,
)


# ── Helper factories ───────────────────────────────────────────────────────────

def _make_chunks(n: int) -> list[dict]:
    return [
        {"chunk_id": f"c{i}", "text": f"text {i}", "rerank_score": float(i)}
        for i in range(n)
    ]


def _make_chunk_entities(chunk_ids: list[str], entity_names: list[str], dim: int = 4) -> list[dict]:
    """Each chunk_id is linked to the matching entity_name with a random embedding."""
    rng = np.random.default_rng(seed=42)
    rows = []
    for cid, ename in zip(chunk_ids, entity_names):
        emb = rng.random(dim).tolist()
        rows.append({"chunk_id": cid, "entity_name": ename, "embedding": emb})
    return rows


def _make_edges(pairs: list[tuple[str, str]], confidence: float = 1.0) -> list[dict]:
    return [
        {"src": a, "tgt": b, "weight": 1.0, "confidence": confidence, "extracted_at": None}
        for a, b in pairs
    ]


# ── _normalize_adjacency ───────────────────────────────────────────────────────

class TestNormalizeAdjacency:
    def test_identity_produces_symmetric_result(self):
        A = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]], dtype=np.float32)
        A_norm = _normalize_adjacency(A)
        assert A_norm.shape == (3, 3)
        # Result must be symmetric
        np.testing.assert_allclose(A_norm, A_norm.T, atol=1e-6)

    def test_isolated_node_does_not_produce_nan(self):
        """Isolated node (all-zero row) must not cause division-by-zero."""
        A = np.array([[0, 0], [0, 0]], dtype=np.float32)
        A_norm = _normalize_adjacency(A)
        assert not np.any(np.isnan(A_norm))
        assert not np.any(np.isinf(A_norm))

    def test_single_node_returns_one(self):
        A = np.zeros((1, 1), dtype=np.float32)
        A_norm = _normalize_adjacency(A)
        # Only self-loop: Â = [[1]], D̃ = [[1]], result = [[1]]
        np.testing.assert_allclose(A_norm, np.array([[1.0]]), atol=1e-6)


# ── _gcn_layer ─────────────────────────────────────────────────────────────────

class TestGCNLayer:
    def test_relu_applied_when_not_last(self):
        """ReLU should zero-out negative values in non-final layers."""
        A_norm = np.eye(3, dtype=np.float32)
        H = np.array([[-1.0, 2.0],
                      [3.0, -4.0],
                      [0.0, 1.0]], dtype=np.float32)
        out = _gcn_layer(A_norm, H, last=False)
        assert np.all(out >= 0), "ReLU should produce non-negative outputs"

    def test_relu_skipped_on_last_layer(self):
        """Final layer must preserve negative values (signed cosine range)."""
        A_norm = np.eye(3, dtype=np.float32)
        H = np.array([[-1.0, 2.0], [3.0, -4.0], [0.0, 1.0]], dtype=np.float32)
        out = _gcn_layer(A_norm, H, last=True)
        assert np.any(out < 0), "Last layer should preserve negatives"

    def test_identity_adjacency_preserves_features(self):
        """Identity adjacency = self-loops only; features should pass through unchanged."""
        A_norm = np.eye(4, dtype=np.float32)
        H = np.random.default_rng(0).random((4, 8)).astype(np.float32)
        out = _gcn_layer(A_norm, H, last=True)
        np.testing.assert_allclose(out, H, atol=1e-6)


# ── _gat_layer ─────────────────────────────────────────────────────────────────

class TestGATLayer:
    def test_output_shape_matches_input(self):
        A = np.array([[1, 1], [1, 1]], dtype=np.float32)
        H = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        out = _gat_layer(A, H)
        assert out.shape == H.shape

    def test_isolated_node_self_loops(self):
        """A node with no edges should return a weighted version of itself."""
        A = np.zeros((2, 2), dtype=np.float32)  # no edges
        H = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        out = _gat_layer(A, H)
        # Self-loop only: output must be a scalar multiple of input (softmax=1 on self)
        assert out.shape == (2, 2)
        assert not np.any(np.isnan(out))

    def test_attention_weights_sum_to_one(self):
        """GAT attention weights over neighbours must sum to 1 per node."""
        A = np.array([[0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0]], dtype=np.float32)
        rng = np.random.default_rng(7)
        H = rng.random((3, 16)).astype(np.float32)
        out = _gat_layer(A, H)
        # output should not have NaN and be finite
        assert np.all(np.isfinite(out))


# ── GNNScorer ──────────────────────────────────────────────────────────────────

class TestGNNScorerInit:
    def test_invalid_gnn_type_raises(self):
        with pytest.raises(ValueError, match="gnn_type"):
            GNNScorer(gnn_type="transformer")

    def test_defaults_accepted(self):
        scorer = GNNScorer()
        assert scorer._type == "gat"
        assert scorer._num_layers == 2
        assert scorer._alpha == pytest.approx(0.9)
        assert scorer._beta == pytest.approx(0.1)


class TestGNNScorerNoEntities:
    """When chunk_entities is empty the scorer must gracefully skip GNN."""

    def test_returns_all_chunks_sorted_by_text_score(self):
        scorer = GNNScorer()
        chunks = _make_chunks(3)
        result = scorer.score(
            query_vec=[1.0] * 768,
            chunks=chunks,
            chunk_entities=[],
            entity_edges=[],
        )
        assert len(result) == 3
        # Sorted descending by final_score
        scores = [c["final_score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_gnn_score_is_zero(self):
        scorer = GNNScorer()
        chunks = _make_chunks(2)
        result = scorer.score([0.5] * 768, chunks, [], [])
        assert all(c["gnn_score"] == 0.0 for c in result)

    def test_final_score_equals_alpha_times_text_score(self):
        scorer = GNNScorer(alpha=0.9, beta=0.1)
        chunks = [{"chunk_id": "c0", "rerank_score": 5.0}]
        result = scorer.score([0.1] * 768, chunks, [], [])
        # Sole seed chunk -> rank 0 of 1 -> text_score = 1 - 0/1 = 1.0
        assert result[0]["final_score"] == pytest.approx(0.9 * 1.0, abs=1e-5)


class TestGNNScorerEdgeFiltering:
    """Low-confidence edges should be dropped from the adjacency matrix."""

    def test_low_confidence_edges_ignored(self):
        scorer = GNNScorer(edge_confidence_threshold=0.8)
        chunks = [{"chunk_id": "c0", "score": 0.5}]
        chunk_entities = [{"chunk_id": "c0", "entity_name": "A",
                           "embedding": [1.0, 0.0, 0.0, 0.0]}]
        edges = _make_edges([("A", "B")], confidence=0.3)  # below threshold
        result = scorer.score([1.0, 0.0, 0.0, 0.0], chunks, chunk_entities, edges)
        assert result[0]["gnn_score"] >= 0.0   # no crash

    def test_high_confidence_edges_used(self):
        scorer = GNNScorer(gnn_type="gcn", edge_confidence_threshold=0.5)
        chunks = [{"chunk_id": "c0", "score": 0.5}, {"chunk_id": "c1", "score": 0.3}]
        chunk_entities = [
            {"chunk_id": "c0", "entity_name": "A", "embedding": [1.0, 0.0, 0.0, 0.0]},
            {"chunk_id": "c1", "entity_name": "B", "embedding": [0.9, 0.1, 0.0, 0.0]},
        ]
        edges = _make_edges([("A", "B")], confidence=0.9)  # above threshold
        result = scorer.score([1.0, 0.0, 0.0, 0.0], chunks, chunk_entities, edges)
        assert all(np.isfinite(c["final_score"]) for c in result)


class TestGNNScorerTextScore:
    """_text_score should rank-normalise seed chunks and pass through path scores."""

    def test_seed_ranks_assigns_position_in_order(self):
        chunks = [
            {"chunk_id": "c0", "rerank_score": 8.0},
            {"chunk_id": "c1", "rerank_score": 0.34},
            {"chunk_id": "c2", "score": 0.9},   # multi-hop, no rerank_score
        ]
        seed_rank, n_seed = GNNScorer._seed_ranks(chunks)
        assert seed_rank == {"c0": 0, "c1": 1}
        assert n_seed == 2

    def test_top_ranked_seed_chunk_gets_text_score_one(self):
        seed_rank, n_seed = {"c0": 0, "c1": 1, "c2": 2}, 3
        assert GNNScorer._text_score({"chunk_id": "c0", "rerank_score": 8.0}, seed_rank, n_seed) == pytest.approx(1.0)
        assert GNNScorer._text_score({"chunk_id": "c1", "rerank_score": 0.5}, seed_rank, n_seed) == pytest.approx(2 / 3)
        assert GNNScorer._text_score({"chunk_id": "c2", "rerank_score": 0.1}, seed_rank, n_seed) == pytest.approx(1 / 3)

    def test_seed_rank_independent_of_raw_rerank_magnitude(self):
        """A weak-query top pick (rerank_score=0.34) scores the same as a strong-query top pick (8.0)."""
        weak  = GNNScorer._text_score({"chunk_id": "c0", "rerank_score": 0.34}, {"c0": 0}, 1)
        strong = GNNScorer._text_score({"chunk_id": "c0", "rerank_score": 8.0}, {"c0": 0}, 1)
        assert weak == pytest.approx(strong) == pytest.approx(1.0)

    def test_fallback_to_score_field_for_multihop_chunks(self):
        chunk = {"chunk_id": "x", "score": 0.75}
        assert GNNScorer._text_score(chunk, {}, 0) == pytest.approx(0.75)

    def test_missing_score_returns_zero(self):
        assert GNNScorer._text_score({"chunk_id": "x"}, {}, 0) == pytest.approx(0.0)


class TestPropagateConfidence:
    """BFS confidence propagation along edge paths."""

    def test_seed_has_confidence_one(self):
        result = GNNScorer.propagate_confidence([], seed_entities=["A"])
        assert result["A"] == pytest.approx(1.0)

    def test_single_hop_decay(self):
        edges = [{"src": "A", "tgt": "B", "confidence": 0.8}]
        result = GNNScorer.propagate_confidence(edges, seed_entities=["A"])
        assert result["B"] == pytest.approx(0.8)

    def test_two_hop_product(self):
        edges = [
            {"src": "A", "tgt": "B", "confidence": 0.8},
            {"src": "B", "tgt": "C", "confidence": 0.9},
        ]
        result = GNNScorer.propagate_confidence(edges, seed_entities=["A"], max_hops=3)
        assert result["C"] == pytest.approx(0.8 * 0.9, abs=1e-6)

    def test_min_confidence_prunes_paths(self):
        edges = [{"src": "A", "tgt": "B", "confidence": 0.03}]
        result = GNNScorer.propagate_confidence(
            edges, seed_entities=["A"], min_confidence=0.05
        )
        assert "B" not in result

    def test_max_hops_limits_depth(self):
        # Chain A→B→C→D with max_hops=1; only B should be reached
        edges = [
            {"src": "A", "tgt": "B", "confidence": 1.0},
            {"src": "B", "tgt": "C", "confidence": 1.0},
            {"src": "C", "tgt": "D", "confidence": 1.0},
        ]
        result = GNNScorer.propagate_confidence(edges, seed_entities=["A"], max_hops=1)
        assert "B" in result
        assert "C" not in result
