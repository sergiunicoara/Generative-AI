"""Unit tests for QueryRewriter — malformed-output and fail-open guards."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from graphrag.retrieval.query_rewriter import QueryRewriter


async def _rewrite_with_mocked_llm(question: str, llm_output: str) -> str:
    rw = QueryRewriter()
    with patch("graphrag.retrieval.query_rewriter.get_fast_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=llm_output)
        mock_get_llm.return_value = mock_llm
        return await rw.rewrite(question)


class TestQueryRewriterPassthrough:
    @pytest.mark.asyncio
    async def test_good_rewrite_is_used(self):
        result = await _rewrite_with_mocked_llm(
            "What is the target ROAS for Nova Beverages Global campaigns per the Statement of Work?",
            "target ROAS Nova Beverages Statement of Work",
        )
        assert result == "target ROAS Nova Beverages Statement of Work"

    @pytest.mark.asyncio
    async def test_takes_first_line_only(self):
        result = await _rewrite_with_mocked_llm(
            "original question",
            "rewritten query\nsome trailing explanation",
        )
        assert result == "rewritten query"


class TestQueryRewriterEmptyOrEchoGuard:
    @pytest.mark.asyncio
    async def test_empty_response_falls_back(self):
        result = await _rewrite_with_mocked_llm("original question", "")
        assert result == "original question"

    @pytest.mark.asyncio
    async def test_none_response_falls_back(self):
        result = await _rewrite_with_mocked_llm("original question", None)
        assert result == "original question"

    @pytest.mark.asyncio
    async def test_echoed_question_falls_back(self):
        result = await _rewrite_with_mocked_llm("Original Question", "original question")
        assert result == "Original Question"

    @pytest.mark.asyncio
    async def test_runaway_expansion_falls_back(self):
        question = "short question"
        runaway = "a " * 200  # far more than 6x the original length
        result = await _rewrite_with_mocked_llm(question, runaway)
        assert result == question


class TestQueryRewriterMalformedOutputGuard:
    """The prompt occasionally leaks boolean syntax or literal placeholders —
    both actively hurt BM25 retrieval rather than helping it."""

    @pytest.mark.asyncio
    async def test_boolean_and_rejected(self):
        result = await _rewrite_with_mocked_llm(
            "original question",
            "term one AND term two AND (something OR other)",
        )
        assert result == "original question"

    @pytest.mark.asyncio
    async def test_boolean_or_rejected(self):
        result = await _rewrite_with_mocked_llm(
            "original question",
            "term one OR term two",
        )
        assert result == "original question"

    @pytest.mark.asyncio
    async def test_literal_placeholder_rejected(self):
        result = await _rewrite_with_mocked_llm(
            "original question",
            "campaign policy document ID: XXXX",
        )
        assert result == "original question"

    @pytest.mark.asyncio
    async def test_code_fence_rejected(self):
        result = await _rewrite_with_mocked_llm(
            "original question",
            "```rewritten query```",
        )
        assert result == "original question"

    @pytest.mark.asyncio
    async def test_case_insensitive_marker_match(self):
        result = await _rewrite_with_mocked_llm(
            "original question",
            "term one and term two",  # lowercase "and" as a real substring test
        )
        # lowercase "and" as a normal English word is expected to trigger the
        # guard too, since the check is intentionally case-insensitive and
        # substring-based rather than trying to parse real boolean syntax.
        assert result == "original question"


class TestQueryRewriterFailOpen:
    @pytest.mark.asyncio
    async def test_llm_exception_falls_back(self):
        rw = QueryRewriter()
        with patch("graphrag.retrieval.query_rewriter.get_fast_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
            mock_get_llm.return_value = mock_llm
            result = await rw.rewrite("original question")
        assert result == "original question"


class TestQueryRewriterDomainNeutrality:
    """Regression test for the leaked few-shot example bug: the old prompt's
    hardcoded 'APQP' / 'rev2 rev4' example bled into unrelated-domain
    rewrites. The prompt itself can't be unit-tested against a real LLM here,
    but this locks in that the code applies no domain-specific injection of
    its own — any acronym/synonym content is fully attributable to the LLM
    response the guards already cover above."""

    def test_prompt_has_no_hardcoded_domain_examples(self):
        from graphrag.retrieval.query_rewriter import _REWRITE_PROMPT
        # APQP was the automotive-specific acronym hardcoded into the old
        # prompt's few-shot example; it leaked into unrelated-domain
        # rewrites regardless of the actual question. The fixed prompt
        # describes the transformation generically instead of via a
        # concrete domain example.
        assert "APQP" not in _REWRITE_PROMPT
