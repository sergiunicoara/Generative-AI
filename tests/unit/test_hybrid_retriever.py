"""Unit tests for HybridRetriever.retrieve_and_answer() — in particular the
mode="hybrid" concurrency fix (see tasks/lessons.md A145): local_search and
global_search have no data dependency, so they now run under an
asyncio.TaskGroup instead of back-to-back sequential awaits.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.retrieval.hybrid_retriever import HybridRetriever


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
    hr._context_builder.build.return_value = ("some context", ["e1"])
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
