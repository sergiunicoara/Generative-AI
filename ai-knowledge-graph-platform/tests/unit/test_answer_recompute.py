"""Opt-in re-answering of invalidated query decisions."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.graph.invalidation import answer_recompute as ar
from graphrag.graph.invalidation.models import ArtifactState


def _settings(enabled=True, acl=False):
    return SimpleNamespace(invalidation={"reanswer_decisions": enabled}, access_control={"enabled": acl})


def _neo(rows):
    n = MagicMock()
    n.run = AsyncMock(return_value=rows)
    n.run_read = AsyncMock(return_value=rows)
    return n


def _retriever(answer="new", citations=("c1",), sufficient=True):
    r = MagicMock()
    r.retrieve_and_answer = AsyncMock(return_value=SimpleNamespace(
        answer=answer, citations=list(citations), confidence=0.8,
        retrieval_sufficiency={"sufficient": sufficient}))
    return r


def test_off_by_default():
    with patch("graphrag.core.config.get_settings", return_value=_settings(enabled=False)):
        assert ar.default_answer_recomputer(_neo([])) is None


async def test_reanswers_a_query_decision_and_keeps_the_recorded_one():
    neo = _neo([{"case_type": "retrieval_query", "question": "who owns Bolt?"}])
    retriever = _retriever()
    with patch("graphrag.core.config.get_settings", return_value=_settings()):
        fn = ar.default_answer_recomputer(neo, retriever_factory=lambda: retriever)
        state = await fn("t", "dec-1")
    assert state is ArtifactState.VALID
    retriever.retrieve_and_answer.assert_awaited_once_with("who owns Bolt?", tenant="t")
    cypher = neo.run.await_args.args[0]
    assert "dec.reanswered_answer" in cypher and "dec.answer =" not in cypher    # recorded decision untouched


async def test_insufficient_answer_is_not_marked_valid():
    neo = _neo([{"case_type": "retrieval_query", "question": "q"}])
    with patch("graphrag.core.config.get_settings", return_value=_settings()):
        fn = ar.default_answer_recomputer(neo, retriever_factory=lambda: _retriever(sufficient=False))
        assert await fn("t", "d") is ArtifactState.INSUFFICIENT_EVIDENCE
        fn = ar.default_answer_recomputer(neo, retriever_factory=lambda: _retriever(citations=()))
        assert await fn("t", "d") is ArtifactState.INSUFFICIENT_EVIDENCE
    neo.run.assert_not_awaited()


async def test_refused_when_access_control_is_enabled():
    neo = _neo([{"case_type": "retrieval_query", "question": "q"}])
    retriever = _retriever()
    with patch("graphrag.core.config.get_settings", return_value=_settings(acl=True)):
        fn = ar.default_answer_recomputer(neo, retriever_factory=lambda: retriever)
        assert await fn("t", "d") is ArtifactState.NEEDS_REVIEW
    retriever.retrieve_and_answer.assert_not_awaited()


@pytest.mark.parametrize("rows", [[], [{"case_type": "work_order", "question": "x"}],
                                  [{"case_type": "retrieval_query", "question": ""}]])
async def test_non_query_decisions_stay_with_human_review(rows):
    retriever = _retriever()
    with patch("graphrag.core.config.get_settings", return_value=_settings()):
        fn = ar.default_answer_recomputer(_neo(rows), retriever_factory=lambda: retriever)
        assert await fn("t", "d") is ArtifactState.NEEDS_REVIEW
    retriever.retrieve_and_answer.assert_not_awaited()
