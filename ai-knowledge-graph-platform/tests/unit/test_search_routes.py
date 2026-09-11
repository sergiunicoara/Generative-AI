"""Unit tests for api/routes/search.py.

POST /search's entire reason to exist is "ranked chunks, no LLM call, no
caller-chosen profile" — those three claims are the load-bearing assertions
below, not incidental coverage. See api/routes/search.py's module docstring
for why the profile is fixed rather than a request field: it is an ACL
boundary, not a performance default.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import search as search_routes
from graphrag.retrieval.hybrid_retriever import retrieval_profile_overrides


def _make_client(scope: str = "read") -> TestClient:
    app = FastAPI()
    app.include_router(search_routes.router, prefix="/search")
    app.dependency_overrides[get_current_user] = lambda: {
        "scope": scope, "sub": "test", "tenant": "test-tenant",
    }
    return TestClient(app)


def _chunk(chunk_id: str, **fields) -> dict:
    return {"chunk_id": chunk_id, "text": f"text for {chunk_id}", **fields}


class TestRankedResultsNoSynthesis:
    def test_results_are_returned_sorted_descending_by_available_score(self):
        # text_hybrid never sets final_score (that only happens inside the
        # GNN branch, which this profile disables), so chunks arrive in
        # fusion order and the route must sort defensively by whichever score
        # key is actually present.
        chunks = [
            _chunk("low", rerank_score=0.2),
            _chunk("high", rerank_score=0.9),
            _chunk("mid", rerank_score=0.5),
        ]
        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(return_value={"chunks": chunks})

        with patch.object(search_routes, "_get_searcher", return_value=mock_searcher):
            resp = _make_client().post("/search", json={"query": "hello"})

        assert resp.status_code == 200
        ids = [hit["chunk_id"] for hit in resp.json()["results"]]
        assert ids == ["high", "mid", "low"]

    def test_falls_back_to_raw_fusion_score_when_no_rerank_score(self):
        chunks = [_chunk("a", score=0.1), _chunk("b", score=0.7)]
        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(return_value={"chunks": chunks})

        with patch.object(search_routes, "_get_searcher", return_value=mock_searcher):
            resp = _make_client().post("/search", json={"query": "hello"})

        ids = [hit["chunk_id"] for hit in resp.json()["results"]]
        assert ids == ["b", "a"]

    def test_no_entities_or_graph_edges_in_the_response(self):
        # entity context is produced by get_entity_neighbors, one of the
        # queries that cannot be ACL-filtered -- it must never reach the
        # client through this route regardless of what LocalSearch returns.
        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(return_value={
            "chunks": [_chunk("c1", score=0.5)],
            "entities": [{"entity": "Boeing", "type": "ORG", "description": "x", "neighbors": []}],
            "entity_edges": [{"src": "A", "relation": "R", "tgt": "B"}],
        })

        with patch.object(search_routes, "_get_searcher", return_value=mock_searcher):
            resp = _make_client().post("/search", json={"query": "hello"})

        body = resp.json()
        assert set(body.keys()) == {"results"}
        assert "entities" not in body["results"][0]


class TestFixedProfileIsNotClientSelectable:
    def test_profile_field_in_the_body_is_ignored(self):
        # No `profile` field exists on SearchRequest at all -- a client
        # attempting to send one must have no effect, not be silently accepted.
        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(return_value={"chunks": []})

        with patch.object(search_routes, "_get_searcher", return_value=mock_searcher):
            resp = _make_client().post(
                "/search", json={"query": "hello", "profile": "full"},
            )

        assert resp.status_code == 200
        call_kwargs = mock_searcher.search.await_args.kwargs
        overrides = call_kwargs["config_overrides"]
        expected = retrieval_profile_overrides("text_hybrid")
        for key, value in expected.items():
            assert overrides[key] == value
        # The un-ACL'd stages must be off no matter what the caller sent.
        assert overrides["multihop_depth"] == 0
        assert overrides["gnn_enabled"] is False
        assert overrides["entity_context_enabled"] is False

    def test_top_k_overrides_candidate_count_not_just_client_slice(self):
        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(return_value={"chunks": []})

        with patch.object(search_routes, "_get_searcher", return_value=mock_searcher):
            _make_client().post("/search", json={"query": "hello", "top_k": 25})

        overrides = mock_searcher.search.await_args.kwargs["config_overrides"]
        assert overrides["local_top_k"] == 25
        assert overrides["rerank_top_k"] == 25


class TestAccessControlFromToken:
    def test_tenant_comes_from_the_token_not_the_body(self):
        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(return_value={"chunks": []})

        with patch.object(search_routes, "_get_searcher", return_value=mock_searcher):
            _make_client().post("/search", json={"query": "hello"})

        assert mock_searcher.search.await_args.kwargs["tenant"] == "test-tenant"

    def test_access_context_is_built_from_claims_and_threaded_through(self):
        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(return_value={"chunks": []})

        with patch.object(search_routes, "_get_searcher", return_value=mock_searcher):
            _make_client().post("/search", json={"query": "hello"})

        access_context = mock_searcher.search.await_args.kwargs["access_context"]
        assert access_context is not None
        assert "user:test" in access_context.principals

    def test_403_without_read_scope(self):
        client = _make_client(scope="write")
        resp = client.post("/search", json={"query": "hello"})
        assert resp.status_code == 403


class TestNoLLMIsEverInvoked:
    def test_llm_client_is_never_constructed_or_called(self):
        # The entire point of this route: LocalSearch.search() never touches
        # an LLM (see local_search.py's module docstring), and this route
        # must never reach HybridRetriever, the layer where synthesis lives.
        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(return_value={"chunks": [_chunk("c1", score=0.9)]})

        with (
            patch.object(search_routes, "_get_searcher", return_value=mock_searcher),
            patch("graphrag.core.llm_client.get_llm") as mock_get_llm,
        ):
            resp = _make_client().post("/search", json={"query": "hello"})

        assert resp.status_code == 200
        mock_get_llm.assert_not_called()


class TestInputLimits:
    @pytest.mark.parametrize(
        "payload",
        [
            {"query": ""},
            {"query": "x" * 8_001},
            {"query": "hello", "top_k": 0},
            {"query": "hello", "top_k": 51},
        ],
    )
    def test_invalid_requests_are_rejected_without_reaching_retrieval(self, payload):
        mock_searcher = MagicMock()
        with patch.object(search_routes, "_get_searcher", return_value=mock_searcher):
            resp = _make_client().post("/search", json=payload)

        assert resp.status_code == 422
        mock_searcher.search.assert_not_awaited()


class TestTimeout:
    def test_504_when_retrieval_exceeds_the_budget(self):
        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(side_effect=asyncio.TimeoutError)

        with (
            patch.object(search_routes, "_get_searcher", return_value=mock_searcher),
            patch("api.routes.search.asyncio.wait_for", side_effect=asyncio.TimeoutError),
        ):
            resp = _make_client().post("/search", json={"query": "hello"})

        assert resp.status_code == 504

    def test_503_when_retrieval_raises(self):
        mock_searcher = MagicMock()
        mock_searcher.search = AsyncMock(side_effect=RuntimeError("neo4j down"))

        with patch.object(search_routes, "_get_searcher", return_value=mock_searcher):
            resp = _make_client().post("/search", json={"query": "hello"})

        assert resp.status_code == 503


class TestSearcherIsASingleton:
    def test_get_searcher_returns_the_same_instance_across_calls(self):
        search_routes._searcher = None
        with patch.object(search_routes, "LocalSearch") as mock_cls:
            mock_cls.return_value = MagicMock()
            first = search_routes._get_searcher()
            second = search_routes._get_searcher()

        assert first is second
        mock_cls.assert_called_once()
        search_routes._searcher = None
