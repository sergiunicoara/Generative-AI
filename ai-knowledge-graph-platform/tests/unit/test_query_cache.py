from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from graphrag.core.models import QueryResult
from graphrag.retrieval.hybrid_retriever import HybridRetriever
from graphrag.retrieval.query_cache import (
    QueryCache,
    QueryCacheContext,
    build_cache_key,
    normalize_query,
)


def _context(**overrides) -> QueryCacheContext:
    values = {
        "corpus_revision": 7,
        "requested_mode": "hybrid",
        "effective_mode": "local",
        "model_route": {
            "primary": "deepseek:deepseek-v4-pro",
            "fallback": "groq:llama-3.3-70b-versatile",
            "agentic_fast": "groq:llama-3.1-8b-instant",
        },
        "prompt_version": "hybrid-answer-v1",
        "retrieval_config": {"rerank_top_k": 5, "gnn_enabled": True},
        "ontology_version": "platform/v1",
    }
    values.update(overrides)
    return QueryCacheContext(**values)


def test_query_normalization_is_conservative_and_stable() -> None:
    assert normalize_query("  Is   placement ALLOWED?\n") == "is placement allowed?"


def test_cache_key_ignores_dictionary_order() -> None:
    left = _context(retrieval_config={"a": 1, "nested": {"x": 2, "y": 3}})
    right = _context(retrieval_config={"nested": {"y": 3, "x": 2}, "a": 1})
    assert build_cache_key("Question", "marketing", left) == build_cache_key(
        " question ", "marketing", right
    )


def test_material_inputs_change_cache_key() -> None:
    baseline = build_cache_key("Question", "marketing", _context())
    variants = [
        build_cache_key("Different", "marketing", _context()),
        build_cache_key("Question", "other", _context()),
        build_cache_key("Question", "marketing", _context(corpus_revision=8)),
        build_cache_key("Question", "marketing", _context(prompt_version="v2")),
        build_cache_key(
            "Question",
            "marketing",
            _context(model_route={"primary": "groq:new", "fallback": "none"}),
        ),
        build_cache_key(
            "Question",
            "marketing",
            _context(retrieval_config={"rerank_top_k": 10}),
        ),
        build_cache_key(
            "Question",
            "marketing",
            _context(valid_at="2025-01-01T00:00:00+00:00"),
        ),
        build_cache_key(
            "Question",
            "marketing",
            _context(transaction_at="2025-02-01T00:00:00+00:00"),
        ),
    ]
    assert all(candidate != baseline for candidate in variants)


async def test_memory_cache_returns_governed_source_metadata() -> None:
    cache = QueryCache(ttl=60)
    context = _context()
    result = QueryResult(question="Question", answer="Answer", citations=["doc-1"])
    key = await cache.set(
        "Question",
        "marketing",
        context,
        result.model_dump(mode="json"),
        source_query_id="query-original",
        source_trace_id="decision-original",
    )

    cached = await cache.get(" question ", "marketing", context)

    assert cached is not None
    assert cached["cache_key"] == key
    assert cached["source_query_id"] == "query-original"
    assert cached["source_trace_id"] == "decision-original"


async def test_memory_cache_expires(monkeypatch) -> None:
    cache = QueryCache(ttl=10)
    context = _context()
    monkeypatch.setattr("graphrag.retrieval.query_cache.time.time", lambda: 100.0)
    await cache.set(
        "Question", "marketing", context, {"answer": "old"},
        source_query_id="query-original", source_trace_id="decision-original",
    )
    monkeypatch.setattr("graphrag.retrieval.query_cache.time.time", lambda: 111.0)
    assert await cache.get("Question", "marketing", context) is None


def _retriever() -> HybridRetriever:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._cfg = {
        "semantic_answer_cache_enabled": True,
        "query_planner_enabled": False,
        "query_rewrite_enabled": False,
        "conflict_annotation_enabled": False,
        "feedback_ranking_enabled": False,
        "claim_verification": False,
        "agentic_fallback": False,
        "rerank_top_k": 5,
        "hybrid_weight_local": 0.6,
        "hybrid_weight_global": 0.4,
    }
    retriever._model_name = "test-model"
    retriever._model_version = "test-model"
    retriever._local = AsyncMock()
    retriever._global = AsyncMock()
    retriever._context_builder = AsyncMock()
    retriever._contradiction = AsyncMock()
    retriever._agentic = AsyncMock()
    retriever._verifier = AsyncMock()
    retriever._rewriter = AsyncMock()
    retriever._use_session_ctx = False
    retriever._session_ctx = None
    retriever._feedback = AsyncMock()
    retriever._context_graph = AsyncMock()
    return retriever


async def test_hybrid_retriever_cache_hit_keeps_new_query_id() -> None:
    retriever = _retriever()
    cached_result = QueryResult(
        query_id="query-original",
        question="Is placement allowed?",
        answer="Placement is allowed.",
        citations=["policy"],
        model_version="test-model",
    )
    cache = AsyncMock()
    cache.get.return_value = {
        "result": cached_result.model_dump(mode="json"),
        "cache_key": "cache-key",
        "source_query_id": "query-original",
        "source_trace_id": "decision-original",
    }
    neo4j = AsyncMock()
    neo4j.get_corpus_state.return_value = {"revision": 3, "updating": False}

    with (
        patch("graphrag.retrieval.hybrid_retriever.get_neo4j", return_value=neo4j),
        patch("graphrag.retrieval.hybrid_retriever.get_query_cache", AsyncMock(return_value=cache)),
        patch(
            "graphrag.retrieval.hybrid_retriever.get_generation_route",
            return_value={"primary": "test-model", "fallback": "test-fallback"},
        ),
        patch("graphrag.retrieval.result_store.get_result_store") as result_store,
    ):
        result_store.return_value.push_progress = AsyncMock()
        result = await retriever.retrieve_and_answer(
            "IS   PLACEMENT ALLOWED?", tenant="marketing", query_id="query-new"
        )

    assert result.query_id == "query-new"
    assert result.cache_hit is True
    assert result.source_query_id == "query-original"
    assert result.source_trace_id == "decision-original"
    retriever._local.search.assert_not_awaited()
    retriever._global.search.assert_not_awaited()


async def test_hybrid_retriever_bypasses_cache_for_session_context() -> None:
    retriever = _retriever()
    retriever._local.search.return_value = {}
    retriever._global.search.return_value = {}
    retriever._context_builder = MagicMock()
    retriever._context_builder.build.return_value = ("context", [], [])
    neo4j = AsyncMock()

    with (
        patch("graphrag.retrieval.hybrid_retriever.get_neo4j", return_value=neo4j),
        patch("graphrag.retrieval.hybrid_retriever.get_llm") as llm,
        patch("graphrag.retrieval.result_store.get_result_store") as result_store,
    ):
        llm.return_value.generate = AsyncMock(return_value="Answer")
        result_store.return_value.push_progress = AsyncMock()
        await retriever.retrieve_and_answer(
            "Follow-up", tenant="marketing", session_id="session-1", query_id="query-1"
        )

    neo4j.get_corpus_state.assert_not_awaited()


async def test_hybrid_retriever_stores_only_after_governed_trace() -> None:
    retriever = _retriever()
    retriever._local.search.return_value = {
        "chunks": [{"chunk_id": "chunk-1", "text": "Policy allows it.", "score": 1.0}],
        "referenced_chunks": ["chunk-1"],
    }
    retriever._context_builder = MagicMock()
    retriever._context_builder.build.return_value = ("Policy allows it.", ["policy-doc"], [])
    retriever._record_context_trace = AsyncMock(return_value="decision-1")
    cache = AsyncMock()
    cache.get.return_value = None
    cache.set.return_value = "cache-key"
    neo4j = AsyncMock()
    neo4j.get_corpus_state.return_value = {"revision": 4, "updating": False}

    with (
        patch("graphrag.retrieval.hybrid_retriever.get_neo4j", return_value=neo4j),
        patch("graphrag.retrieval.hybrid_retriever.get_query_cache", AsyncMock(return_value=cache)),
        patch(
            "graphrag.retrieval.hybrid_retriever.get_generation_route",
            return_value={"primary": "test-model", "fallback": "test-fallback"},
        ),
        patch("graphrag.retrieval.hybrid_retriever.get_llm") as llm,
        patch("graphrag.retrieval.result_store.get_result_store") as result_store,
    ):
        llm.return_value.generate = AsyncMock(return_value="Placement is allowed.")
        result_store.return_value.push_progress = AsyncMock()
        result = await retriever.retrieve_and_answer(
            "Is placement allowed?", mode="local", tenant="marketing", query_id="query-1"
        )

    retriever._record_context_trace.assert_awaited_once()
    cache.set.assert_awaited_once()
    assert result.cache_hit is False
    assert result.cache_key == "cache-key"
    assert result.source_query_id == "query-1"
    assert result.source_trace_id == "decision-1"


async def test_hybrid_retriever_does_not_store_without_trace() -> None:
    retriever = _retriever()
    retriever._local.search.return_value = {
        "chunks": [{"chunk_id": "chunk-1", "text": "Policy allows it.", "score": 1.0}],
        "referenced_chunks": ["chunk-1"],
    }
    retriever._context_builder = MagicMock()
    retriever._context_builder.build.return_value = ("Policy allows it.", ["policy-doc"], [])
    retriever._record_context_trace = AsyncMock(return_value=None)
    cache = AsyncMock()
    cache.get.return_value = None
    neo4j = AsyncMock()
    neo4j.get_corpus_state.return_value = {"revision": 4, "updating": False}

    with (
        patch("graphrag.retrieval.hybrid_retriever.get_neo4j", return_value=neo4j),
        patch("graphrag.retrieval.hybrid_retriever.get_query_cache", AsyncMock(return_value=cache)),
        patch(
            "graphrag.retrieval.hybrid_retriever.get_generation_route",
            return_value={"primary": "test-model", "fallback": "test-fallback"},
        ),
        patch("graphrag.retrieval.hybrid_retriever.get_llm") as llm,
        patch("graphrag.retrieval.result_store.get_result_store") as result_store,
    ):
        llm.return_value.generate = AsyncMock(return_value="Placement is allowed.")
        result_store.return_value.push_progress = AsyncMock()
        await retriever.retrieve_and_answer(
            "Is placement allowed?", mode="local", tenant="marketing", query_id="query-1"
        )

    cache.set.assert_not_awaited()


async def test_hybrid_retriever_bypasses_cache_during_ingestion() -> None:
    retriever = _retriever()
    retriever._local.search.return_value = {}
    retriever._context_builder = MagicMock()
    retriever._context_builder.build.return_value = ("context", [], [])
    neo4j = AsyncMock()
    neo4j.get_corpus_state.return_value = {"revision": 4, "updating": True}

    with (
        patch("graphrag.retrieval.hybrid_retriever.get_neo4j", return_value=neo4j),
        patch("graphrag.retrieval.hybrid_retriever.get_query_cache") as get_cache,
        patch("graphrag.retrieval.hybrid_retriever.get_llm") as llm,
        patch("graphrag.retrieval.result_store.get_result_store") as result_store,
    ):
        llm.return_value.generate = AsyncMock(return_value="Answer")
        result_store.return_value.push_progress = AsyncMock()
        await retriever.retrieve_and_answer(
            "Question", mode="local", tenant="marketing", query_id="query-1"
        )

    get_cache.assert_not_called()
    retriever._local.search.assert_awaited_once()
