"""Unit tests for HybridRetriever.retrieve_and_answer() — in particular the
mode="hybrid" concurrency fix (see tasks/lessons.md A145): local_search and
global_search have no data dependency, so they now run under an
asyncio.TaskGroup instead of back-to-back sequential awaits.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.retrieval.hybrid_retriever import HybridRetriever
from graphrag.core.models import QueryResult


def _make_hybrid_retriever(cfg_overrides: dict | None = None) -> HybridRetriever:
    base_cfg = {
        "query_rewrite_enabled": False,
        "conflict_annotation_enabled": False,
        "claim_verification": False,
        "agentic_fallback": False,
        "session_context_enabled": False,
        "rerank_top_k": 5,
        "hybrid_weight_local": 0.6,
        "hybrid_weight_global": 0.4,
    }
    if cfg_overrides:
        base_cfg.update(cfg_overrides)

    hr = HybridRetriever.__new__(HybridRetriever)
    hr._model_name = "test-model"
    hr._cfg = base_cfg
    hr._local = AsyncMock()
    hr._global = AsyncMock()
    hr._context_builder = MagicMock()
    hr._context_builder.build.return_value = ("some context", ["e1"], [])
    hr._contradiction = AsyncMock()
    hr._model_version = "test-model"
    hr._agentic = AsyncMock()
    hr._verifier = AsyncMock()
    hr._rewriter = AsyncMock()
    hr._use_session_ctx = False
    hr._session_ctx = None
    return hr


@pytest.fixture(autouse=True)
def _patch_llm_and_config():
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="A confident answer.")
    with (
        patch("graphrag.retrieval.hybrid_retriever.get_llm", return_value=mock_llm),
        patch(
            "graphrag.retrieval.hybrid_retriever.resolve_tenant_config",
            side_effect=lambda base, tenant: base,
        ),
    ):
        yield mock_llm


class TestConcurrency:
    async def test_local_and_global_search_run_concurrently(self) -> None:
        hr = _make_hybrid_retriever()

        # Overlap is asserted structurally rather than by wall-clock: the old
        # version slept 50ms per branch and asserted elapsed < 90ms, a 40ms
        # margin that flakes on a loaded CI runner or under coverage
        # instrumentation. Tracking concurrent occupancy proves the same
        # property — both searches in flight at once — with no timing margin.
        in_flight = 0
        max_in_flight = 0

        async def _tracked(*args, **kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0)   # yield so a concurrent peer can start
            in_flight -= 1
            return {}

        hr._local.search = AsyncMock(side_effect=_tracked)
        hr._global.search = AsyncMock(side_effect=_tracked)

        await hr.retrieve_and_answer("question", mode="hybrid")

        assert max_in_flight == 2, (
            f"local and global search ran sequentially "
            f"(max concurrent occupancy was {max_in_flight}, expected 2)"
        )


class TestModeGating:
    async def test_hybrid_calls_both(self) -> None:
        hr = _make_hybrid_retriever()
        hr._local.search = AsyncMock(return_value={})
        hr._global.search = AsyncMock(return_value={})

        await hr.retrieve_and_answer("question", mode="hybrid")

        hr._local.search.assert_awaited_once()
        hr._global.search.assert_awaited_once()

    async def test_local_mode_skips_global(self) -> None:
        hr = _make_hybrid_retriever()
        hr._local.search = AsyncMock(return_value={})
        hr._global.search = AsyncMock(return_value={})

        await hr.retrieve_and_answer("question", mode="local")

        hr._local.search.assert_awaited_once()
        hr._global.search.assert_not_awaited()

    async def test_global_mode_skips_local(self) -> None:
        hr = _make_hybrid_retriever()
        hr._local.search = AsyncMock(return_value={})
        hr._global.search = AsyncMock(return_value={})

        await hr.retrieve_and_answer("question", mode="global")

        hr._global.search.assert_awaited_once()
        hr._local.search.assert_not_awaited()


class TestRetrievalTrajectory:
    async def test_standard_query_returns_structural_route_and_evidence_trace(self) -> None:
        hr = _make_hybrid_retriever({"trajectory_capture_enabled": True})
        hr._local.search = AsyncMock(return_value={
            "chunks": [{"chunk_id": "chunk-a", "text": "evidence"}],
            "referenced_chunks": ["chunk-a"],
            "referenced_entities": ["SpaceX"],
            "entity_edges": [{"src": "SpaceX", "relation": "LAUNCHED", "tgt": "Falcon 9"}],
        })
        hr._global.search = AsyncMock(return_value={})

        result = await hr.retrieve_and_answer("What did SpaceX launch?", mode="local")

        trajectory = result.retrieval_trajectory
        assert trajectory is not None
        assert trajectory.planned_mode == "local"
        assert trajectory.selected_surfaces == ["text", "vector", "graph"]
        assert trajectory.evidence_ids == ["chunk-a"]
        assert trajectory.graph_edges == ["SpaceX|LAUNCHED|Falcon 9"]
        assert trajectory.tool_calls == 1

    async def test_vector_only_profile_reports_only_the_vector_surface(self) -> None:
        hr = _make_hybrid_retriever({"trajectory_capture_enabled": True})
        hr._local.search = AsyncMock(return_value={
            "chunks": [{"chunk_id": "chunk-a", "text": "evidence"}],
            "referenced_chunks": ["chunk-a"],
        })

        result = await hr.retrieve_and_answer(
            "What did SpaceX launch?", retrieval_profile="vector_only",
        )

        assert result.retrieval_trajectory is not None
        assert result.retrieval_trajectory.selected_surfaces == ["vector"]


class TestExceptionTypePreservation:
    """TaskGroup wraps every failure in an ExceptionGroup, even a single one.
    rabbitmq_client.py's DLQ handler logs type(exc).__name__ — without the
    unwrap, a real APIStatusError would show up in logs/DLQ as the opaque
    "ExceptionGroup" instead of the actual failure type."""

    async def test_local_search_failure_preserves_exception_type(self) -> None:
        hr = _make_hybrid_retriever()
        hr._local.search = AsyncMock(side_effect=ValueError("boom"))
        hr._global.search = AsyncMock(return_value={})

        with pytest.raises(ValueError, match="boom"):
            await hr.retrieve_and_answer("question", mode="hybrid")

    async def test_global_search_failure_preserves_exception_type(self) -> None:
        hr = _make_hybrid_retriever()
        hr._local.search = AsyncMock(return_value={})
        hr._global.search = AsyncMock(side_effect=ValueError("boom"))

        with pytest.raises(ValueError, match="boom"):
            await hr.retrieve_and_answer("question", mode="hybrid")


class TestNegativeClassTopKWiring:
    """Regression test for the bug found diagnosing NEG-03 (2026-08-17, see
    docs/audit-2026-08-13.md "What's left"): query_planner_enabled computed a
    per-query-class top_k but every self._local.search()/self._global.search()
    call passed config_overrides=profile_overrides — the pre-planner dict —
    instead of the updated cfg, so the computed top_k never actually reached
    LocalSearch, and separately never touched rerank_top_k either. Scoped
    narrowly to query_class == "negative" only (see hybrid_retriever.py's
    inline comment for why the other classes aren't touched here — this exact
    lever has a documented regression history, A124/A125/local_top_k=15)."""

    async def test_negative_class_question_gets_widened_top_k(self) -> None:
        hr = _make_hybrid_retriever({
            "query_planner_enabled": True,
            "adaptive_router_enabled": False,  # keyword_planner path — no adaptive_router mock needed
        })
        hr._local.search = AsyncMock(return_value={})
        hr._global.search = AsyncMock(return_value={})

        await hr.retrieve_and_answer(
            "Is there a FAA airworthiness directive governing Airbus aircraft in this corpus?",
            mode="hybrid",
        )

        _, kwargs = hr._local.search.call_args
        assert kwargs["config_overrides"]["local_top_k"] == 10
        assert kwargs["config_overrides"]["rerank_top_k"] == 10

    async def test_factoid_class_question_is_unaffected(self) -> None:
        # Regression guard: a plain factoid question must NOT pick up the
        # negative-class widening — config_overrides should stay exactly the
        # pre-fix profile_overrides dict (empty here), same as before this
        # session's change.
        hr = _make_hybrid_retriever({
            "query_planner_enabled": True,
            "adaptive_router_enabled": False,
        })
        hr._local.search = AsyncMock(return_value={})
        hr._global.search = AsyncMock(return_value={})

        await hr.retrieve_and_answer("Who manufactures the Boeing 737 MAX?", mode="hybrid")

        _, kwargs = hr._local.search.call_args
        assert "local_top_k" not in kwargs["config_overrides"]
        assert "rerank_top_k" not in kwargs["config_overrides"]


class TestPlannedAgenticFallback:
    async def test_multi_hop_plan_falls_back_when_global_has_no_chunk_evidence(self) -> None:
        """An incidental global citation must not suppress the planned IRCoT path."""
        hr = _make_hybrid_retriever({
            "query_planner_enabled": True,
            "adaptive_router_enabled": False,
            "agentic_fallback": True,
        })
        hr._global.search = AsyncMock(return_value={})
        hr._agentic.retrieve_and_answer = AsyncMock(return_value=QueryResult(
            question="Explain the full compliance chain across steps.",
            answer="The FAA directive applies to Southwest's 737 MAX fleet.",
            citations=["FAA-AD-2024-01-02", "SWA_fleet_registry_2024"],
            retrieval_mode="agentic",
        ))
        hr._record_context_trace = AsyncMock(return_value=None)

        neo4j = MagicMock()
        neo4j.get_document_filenames = AsyncMock(return_value=[])
        with patch("graphrag.retrieval.hybrid_retriever.get_neo4j", return_value=neo4j):
            result = await hr.retrieve_and_answer(
                "Explain the full compliance chain across steps.", mode="hybrid",
            )

        hr._agentic.retrieve_and_answer.assert_awaited_once()
        assert result.retrieval_mode == "agentic"

    async def test_cold_start_route_honors_the_planned_agentic_fallback(self) -> None:
        """The ADAPTIVE ROUTER's cold-start route must still reach IRCoT.

        Complement of the test above: that one disables the adaptive router to
        exercise the keyword planner, this one leaves it enabled to exercise
        the router. Both need `query_planner_enabled` — hybrid_retriever gates
        the whole routing block on it, so without it the router is never
        consulted and a test that stubs `_adaptive_router` is asserting
        nothing about routing at all.
        """
        hr = _make_hybrid_retriever({
            "query_planner_enabled": True,
            "adaptive_router_enabled": True,
            "agentic_fallback": True,
        })
        hr._global.search = AsyncMock(return_value={})
        hr._agentic.retrieve_and_answer = AsyncMock(return_value=QueryResult(
            question="Explain the full compliance chain across steps.",
            answer="The FAA directive applies to Southwest's 737 MAX fleet.",
            citations=["FAA-AD-2024-01-02", "SWA_fleet_registry_2024"],
            retrieval_mode="agentic",
        ))
        hr._record_context_trace = AsyncMock(return_value=None)
        hr._adaptive_router = AsyncMock()
        hr._adaptive_router.choose = AsyncMock(return_value=MagicMock(
            mode="global", top_k=10, reason="planner_cold_start",
        ))

        neo4j = MagicMock()
        neo4j.get_document_filenames = AsyncMock(return_value=[])
        with patch("graphrag.retrieval.hybrid_retriever.get_neo4j", return_value=neo4j):
            result = await hr.retrieve_and_answer(
                "Explain the full compliance chain across steps.", mode="hybrid",
            )

        # Guard against this test silently going vacuous again: if the routing
        # block stops running, `choose` is never awaited and the cold-start
        # reason never reaches the result, even though the fallback assertion
        # below would still pass for unrelated reasons.
        hr._adaptive_router.choose.assert_awaited_once()
        assert result.routing_reason == "planner_cold_start"
        hr._agentic.retrieve_and_answer.assert_awaited_once()
        assert result.retrieval_mode == "agentic"


class TestStructuredEvidence:
    """`QueryResult.evidence` is populated alongside `citations`, and both
    empty together on abstention — additive, never replacing `citations`."""

    async def test_evidence_is_populated_alongside_citations(self) -> None:
        from graphrag.core.models import CitationEvidence

        hr = _make_hybrid_retriever()
        hr._local.search = AsyncMock(return_value={"chunks": [{"chunk_id": "c1", "text": "t"}]})
        hr._global.search = AsyncMock(return_value={})
        hr._context_builder.build.return_value = (
            "context", ["DocA"],
            [CitationEvidence(source_id="DocA", source_label="DocA", path="[DocA]", confidence=0.9)],
        )

        result = await hr.retrieve_and_answer("question", mode="local")

        assert result.citations == ["DocA"]
        assert len(result.evidence) == 1
        assert result.evidence[0].source_id == "DocA"
        assert result.evidence[0].confidence == 0.9

    async def test_abstention_empties_evidence_alongside_citations(self) -> None:
        from graphrag.core.models import CitationEvidence

        hr = _make_hybrid_retriever({
            "retrieval_sufficiency_abstain_enabled": True,
            "retrieval_sufficiency_min_evidence": 5,
        })
        hr._local.search = AsyncMock(return_value={"chunks": []})
        hr._global.search = AsyncMock(return_value={})
        hr._context_builder.build.return_value = (
            "context", ["DocA"],
            [CitationEvidence(source_id="DocA", source_label="DocA")],
        )

        result = await hr.retrieve_and_answer("question", mode="local")

        assert result.citations == []
        assert result.evidence == []


class TestSemanticAnswerCacheKeyStability:
    """Regression coverage for the 2026-09-23 audit finding: a missing
    valid_at was defaulted to datetime.now(timezone.utc).isoformat() BEFORE
    the QueryCacheContext was built, so the cache key embedded a fresh
    microsecond-precision timestamp on every call. Two back-to-back
    identical queries with no explicit valid_at built two different keys,
    so the semantic answer cache never returned a hit for any caller that
    didn't pass valid_at explicitly -- while still paying for the
    corpus-state read and the Redis write on every single call."""

    async def test_two_calls_with_no_valid_at_build_the_same_cache_context(self) -> None:
        from graphrag.core.models import CitationEvidence

        hr = _make_hybrid_retriever({"semantic_answer_cache_enabled": True})
        hr._local.search = AsyncMock(return_value={"chunks": []})
        hr._global.search = AsyncMock(return_value={})
        hr._context_builder.build.return_value = (
            "context", ["DocA"],
            [CitationEvidence(source_id="DocA", source_label="DocA")],
        )

        fake_neo4j = MagicMock()
        fake_neo4j.get_corpus_state = AsyncMock(return_value={"revision": 1, "updating": False})
        fake_cache = AsyncMock()
        fake_cache.get = AsyncMock(return_value=None)  # every call is a cache miss
        fake_cache.set = AsyncMock(return_value="cache-key")
        seen_contexts = []

        async def _get(question, tenant, context):
            seen_contexts.append(context)
            return None

        fake_cache.get.side_effect = _get

        with (
            patch("graphrag.retrieval.hybrid_retriever.get_neo4j", return_value=fake_neo4j),
            patch("graphrag.retrieval.hybrid_retriever.get_query_cache", AsyncMock(return_value=fake_cache)),
        ):
            await hr.retrieve_and_answer("question", mode="local", query_id="q1")
            await hr.retrieve_and_answer("question", mode="local", query_id="q2")

        assert len(seen_contexts) == 2
        # The bug: these used to differ (each a fresh isoformat() timestamp),
        # which meant build_cache_key(...) never produced the same key twice
        # for a caller that never passes valid_at explicitly.
        assert seen_contexts[0].valid_at is None
        assert seen_contexts[1].valid_at is None
        assert seen_contexts[0] == seen_contexts[1]

    async def test_an_explicit_valid_at_still_participates_in_the_cache_key(self) -> None:
        """The fix must not flatten a genuine historical query into the same
        key as an "as of now" one."""
        from graphrag.core.models import CitationEvidence

        hr = _make_hybrid_retriever({"semantic_answer_cache_enabled": True})
        hr._local.search = AsyncMock(return_value={"chunks": []})
        hr._global.search = AsyncMock(return_value={})
        hr._context_builder.build.return_value = (
            "context", ["DocA"],
            [CitationEvidence(source_id="DocA", source_label="DocA")],
        )

        fake_neo4j = MagicMock()
        fake_neo4j.get_corpus_state = AsyncMock(return_value={"revision": 1, "updating": False})
        fake_cache = AsyncMock()
        fake_cache.set = AsyncMock(return_value="cache-key")
        seen_contexts = []

        async def _get(question, tenant, context):
            seen_contexts.append(context)
            return None

        fake_cache.get = AsyncMock(side_effect=_get)

        with (
            patch("graphrag.retrieval.hybrid_retriever.get_neo4j", return_value=fake_neo4j),
            patch("graphrag.retrieval.hybrid_retriever.get_query_cache", AsyncMock(return_value=fake_cache)),
        ):
            await hr.retrieve_and_answer(
                "question", mode="local", query_id="q1", valid_at="2020-01-01T00:00:00+00:00",
            )
            await hr.retrieve_and_answer("question", mode="local", query_id="q2")

        assert seen_contexts[0].valid_at == "2020-01-01T00:00:00+00:00"
        assert seen_contexts[1].valid_at is None
        assert seen_contexts[0] != seen_contexts[1]


class TestAnswerCacheIndexesEntityNames:
    """/kg/cache/invalidate takes entity names, so cached answers must be
    indexed by the entities retrieval used -- not by citation labels, which
    never matched and made every invalidation report 0."""

    async def test_cache_set_receives_referenced_entity_names(self) -> None:
        from graphrag.core.models import CitationEvidence

        hr = _make_hybrid_retriever({"semantic_answer_cache_enabled": True, "agentic_fallback": False})
        hr._local.search = AsyncMock(return_value={
            "chunks": [], "referenced_entities": ["Boeing 737", "FAA"],
        })
        hr._global.search = AsyncMock(return_value={})
        hr._context_builder.build.return_value = (
            "context", ["DocA"],
            [CitationEvidence(source_id="DocA", source_label="DocA")],
        )
        hr._record_context_trace = AsyncMock(return_value="trace-1")

        fake_neo4j = MagicMock()
        fake_neo4j.get_corpus_state = AsyncMock(return_value={"revision": 1, "updating": False})
        fake_cache = AsyncMock()
        fake_cache.get = AsyncMock(return_value=None)
        fake_cache.set = AsyncMock(return_value="cache-key")

        with (
            patch("graphrag.retrieval.hybrid_retriever.get_neo4j", return_value=fake_neo4j),
            patch("graphrag.retrieval.hybrid_retriever.get_query_cache", AsyncMock(return_value=fake_cache)),
        ):
            await hr.retrieve_and_answer("question", mode="local", query_id="q1")

        fake_cache.set.assert_awaited_once()
        assert fake_cache.set.call_args.kwargs["entities_used"] == ["Boeing 737", "FAA"]
