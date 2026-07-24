"""Unit tests for LocalSearch — pipeline feature flag control flow.

These tests verify that each feature flag (bm25_enabled, reranker_enabled,
gnn_enabled) actually gates its stage and that the final result structure
is always correct.  Neo4j, BM25, and reranker are mocked so no live services
are required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.retrieval.local_search import LocalSearch, _adaptive_weights


# ── Adaptive weights ───────────────────────────────────────────────────────────

class TestAdaptiveWeights:
    def test_relational_query_returns_balanced(self):
        a, b = _adaptive_weights("how is A related to B", 0.9, 0.1)
        assert a == pytest.approx(0.5)
        assert b == pytest.approx(0.5)

    def test_factoid_query_returns_defaults(self):
        a, b = _adaptive_weights("what is the melting point of steel", 0.9, 0.1)
        assert a == pytest.approx(0.9)
        assert b == pytest.approx(0.1)

    def test_connected_signal_triggers_balanced(self):
        a, b = _adaptive_weights("which entities are connected", 0.9, 0.1)
        assert a == pytest.approx(0.5)

    def test_empty_question_returns_defaults(self):
        a, b = _adaptive_weights("", 0.9, 0.1)
        assert a == pytest.approx(0.9)


# ── LocalSearch.search — feature flag control ──────────────────────────────────

def _make_local_search(cfg_overrides: dict | None = None) -> LocalSearch:
    """Build a LocalSearch with all heavy dependencies mocked."""
    base_cfg = {
        "local_top_k": 5,
        "multihop_depth": 2,
        "bm25_enabled": True,
        "reranker_enabled": True,
        "gnn_enabled": True,
        "rerank_top_k": 3,
        "gnn_type": "gat",
        "gnn_layers": 2,
        "gnn_alpha": 0.9,
        "gnn_beta": 0.1,
        "gnn_edge_confidence_threshold": 0.7,
        "gnn_confidence_half_life_days": 0,
        "gnn_adaptive_weights": False,
        "authority_weighting_enabled": False,
        "session_context_enabled": False,
    }
    if cfg_overrides:
        base_cfg.update(cfg_overrides)

    with (
        patch("graphrag.retrieval.local_search.get_settings") as mock_settings,
        patch("graphrag.retrieval.local_search.get_neo4j") as mock_neo4j,
        patch("graphrag.retrieval.local_search.Embedder") as mock_embedder_cls,
        patch("graphrag.retrieval.local_search.HybridBM25Search") as mock_bm25_cls,
        patch("graphrag.retrieval.local_search.CrossEncoderReranker") as mock_reranker_cls,
        patch("graphrag.retrieval.local_search.GNNScorer") as mock_gnn_cls,
        patch("graphrag.retrieval.local_search.DocumentAuthorityService"),
        patch("graphrag.retrieval.local_search.get_session_context"),
    ):
        mock_settings.return_value.retrieval = base_cfg

        ls = LocalSearch.__new__(LocalSearch)
        # search() resolves per-tenant config from self._cfg via
        # resolve_tenant_config(); base_cfg has no tenant_overrides key, so it
        # passes through unchanged and these tests exercise the intended knobs.
        ls._cfg = base_cfg
        ls._neo4j = AsyncMock()
        ls._neo4j.get_chunk_filenames = AsyncMock(return_value={})
        ls._embedder = AsyncMock()
        ls._bm25 = AsyncMock()
        ls._reranker = AsyncMock()
        ls._gnn = MagicMock()
        ls._adaptive_weights = False
        ls._use_authority_weights = False
        ls._authority_svc = AsyncMock()
        ls._use_session_ctx = False
        ls._session_ctx = None

    return ls


def _chunk(cid: str) -> dict:
    return {"chunk_id": cid, "text": f"text of {cid}", "score": 0.5}


class TestLocalSearchPipelineFlags:
    async def test_returns_required_keys(self):
        ls = _make_local_search()
        ls._embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        ls._neo4j.vector_search_chunks = AsyncMock(return_value=[_chunk("c1")])
        ls._bm25.search = AsyncMock(return_value=[_chunk("c1")])
        ls._reranker.rerank = AsyncMock(return_value=[_chunk("c1")])
        ls._neo4j.get_multihop_chunks = AsyncMock(return_value=[])
        ls._neo4j.get_chunk_entity_embeddings = AsyncMock(return_value=[])
        ls._neo4j.get_entity_relations_subgraph = AsyncMock(return_value=[])
        ls._neo4j.get_entity_neighbors = AsyncMock(return_value=[])
        ls._gnn.score = MagicMock(return_value=[_chunk("c1")])

        result = await ls.search("test question")
        assert "chunks" in result
        assert "entities" in result
        assert "referenced_entities" in result
        assert "referenced_chunks" in result

    async def test_chunk_entity_embeddings_fetched_once_not_twice(self):
        """Regression test: get_chunk_entity_embeddings must be called
        exactly once per search() call. It used to be called twice — once
        directly, once again inside the old _fetch_subgraph_edges, which
        re-fetched the same chunk_ids under the same asyncio.gather — see
        tasks/lessons.md (2026-07-24)."""
        ls = _make_local_search()
        ls._embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        ls._neo4j.vector_search_chunks = AsyncMock(return_value=[_chunk("c1")])
        ls._bm25.search = AsyncMock(return_value=[_chunk("c1")])
        ls._reranker.rerank = AsyncMock(return_value=[_chunk("c1")])
        ls._neo4j.get_multihop_chunks = AsyncMock(return_value=[])
        ls._neo4j.get_chunk_entity_embeddings = AsyncMock(return_value=[])
        ls._neo4j.get_entity_relations_subgraph = AsyncMock(return_value=[])
        ls._neo4j.get_entity_neighbors = AsyncMock(return_value=[])
        ls._gnn.score = MagicMock(return_value=[_chunk("c1")])

        await ls.search("test question")
        ls._neo4j.get_chunk_entity_embeddings.assert_called_once()

    async def test_bm25_disabled_skips_bm25_call(self):
        ls = _make_local_search({"bm25_enabled": False})
        ls._embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        ls._neo4j.vector_search_chunks = AsyncMock(return_value=[_chunk("c1")])
        ls._reranker.rerank = AsyncMock(return_value=[_chunk("c1")])
        ls._neo4j.get_multihop_chunks = AsyncMock(return_value=[])
        ls._neo4j.get_chunk_entity_embeddings = AsyncMock(return_value=[])
        ls._neo4j.get_entity_relations_subgraph = AsyncMock(return_value=[])
        ls._neo4j.get_entity_neighbors = AsyncMock(return_value=[])
        ls._gnn.score = MagicMock(return_value=[_chunk("c1")])

        await ls.search("test question")
        ls._bm25.search.assert_not_called()

    async def test_reranker_disabled_skips_rerank_call(self):
        ls = _make_local_search({"reranker_enabled": False})
        ls._embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        ls._neo4j.vector_search_chunks = AsyncMock(return_value=[_chunk("c1")])
        ls._bm25.search = AsyncMock(return_value=[_chunk("c1")])
        ls._neo4j.get_multihop_chunks = AsyncMock(return_value=[])
        ls._neo4j.get_chunk_entity_embeddings = AsyncMock(return_value=[])
        ls._neo4j.get_entity_relations_subgraph = AsyncMock(return_value=[])
        ls._neo4j.get_entity_neighbors = AsyncMock(return_value=[])
        ls._gnn.score = MagicMock(return_value=[_chunk("c1")])

        await ls.search("test question")
        ls._reranker.rerank.assert_not_called()

    async def test_gnn_disabled_skips_gnn_call(self):
        ls = _make_local_search({"gnn_enabled": False})
        ls._embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        ls._neo4j.vector_search_chunks = AsyncMock(return_value=[_chunk("c1")])
        ls._bm25.search = AsyncMock(return_value=[_chunk("c1")])
        ls._reranker.rerank = AsyncMock(return_value=[_chunk("c1")])
        ls._neo4j.get_multihop_chunks = AsyncMock(return_value=[])
        ls._neo4j.get_entity_neighbors = AsyncMock(return_value=[])

        await ls.search("test question")
        ls._gnn.score.assert_not_called()

    async def test_multihop_chunks_deduped_from_seed(self):
        """Multihop chunks already in seed set must not be added twice."""
        ls = _make_local_search({"gnn_enabled": False})
        ls._embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        ls._neo4j.vector_search_chunks = AsyncMock(return_value=[_chunk("c1"), _chunk("c2")])
        ls._bm25.search = AsyncMock(return_value=[_chunk("c1"), _chunk("c2")])
        ls._reranker.rerank = AsyncMock(return_value=[_chunk("c1")])
        # Multihop returns c1 (duplicate) + c3 (new)
        ls._neo4j.get_multihop_chunks = AsyncMock(
            return_value=[_chunk("c1"), _chunk("c3")]
        )
        ls._neo4j.get_entity_neighbors = AsyncMock(return_value=[])

        result = await ls.search("test")
        chunk_ids = [c["chunk_id"] for c in result["chunks"]]
        assert chunk_ids.count("c1") == 1   # no duplicates
        assert "c3" in chunk_ids

    async def test_empty_vector_results_still_returns_valid_structure(self):
        ls = _make_local_search({"gnn_enabled": False})
        ls._embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        ls._neo4j.vector_search_chunks = AsyncMock(return_value=[])
        ls._bm25.search = AsyncMock(return_value=[])
        ls._reranker.rerank = AsyncMock(return_value=[])
        ls._neo4j.get_multihop_chunks = AsyncMock(return_value=[])
        ls._neo4j.get_entity_neighbors = AsyncMock(return_value=[])

        result = await ls.search("test")
        assert result["chunks"] == []
        assert result["referenced_chunks"] == []

    async def test_lexical_seed_floor_preserves_distinct_documents(self):
        ls = _make_local_search({
            "gnn_enabled": False,
            "rerank_top_k": 3,
            "lexical_seed_min_documents": 3,
        })
        c1, c2, c3, c4 = (_chunk(cid) for cid in ("c1", "c2", "c3", "c4"))
        ls._embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        ls._neo4j.vector_search_chunks = AsyncMock(return_value=[c1, c2, c3, c4])
        ls._bm25.search = AsyncMock(return_value=[c1, c2, c3, c4])
        # The reranker crowds out document B with a second chunk from A.
        ls._reranker.rerank = AsyncMock(return_value=[c1, c2, c4])
        ls._neo4j.get_chunk_filenames = AsyncMock(return_value={
            "c1": "A.txt", "c2": "A.txt", "c3": "B.txt", "c4": "C.txt",
        })
        ls._neo4j.get_multihop_chunks = AsyncMock(return_value=[])
        ls._neo4j.get_entity_neighbors = AsyncMock(return_value=[])

        result = await ls.search("test")

        assert [c["chunk_id"] for c in result["chunks"]] == ["c1", "c3", "c4"]
