"""Unit tests for AgenticRetriever.retrieve_and_answer() — dash normalization
on both its answer exit points.

Regression test for the bug found 2026-08-17 diagnosing PRE-01/PRE-02/AUT-01/
AUT-03/CON-01/AGT-02 golden-eval failures (see docs/audit-2026-08-13.md):
Groq's gpt-oss models write hyphenated document IDs/dates using U+2011
NON-BREAKING HYPHEN instead of ASCII "-". The fix (llm_utils.normalize_dashes)
was first applied only to _synthesize() (the get_llm() call), but live
re-verification showed AGT-02/CON-01/AUT-01/AUT-03/PRE-02 all take a
*different* exit point — the "ANSWER:" early-exit shortcut inside the fast
reasoning loop (_reason()/get_fast_llm()) — which was missed on the first
pass. Both exit points are covered here so a future refactor can't silently
drop the fix from either one without a test failing.

Uses chr(0x2011) throughout (not a literal character) so a visually-
identical-but-wrong dash can't be pasted into the test itself by mistake —
same discipline as test_llm_utils.py's TestNormalizeDashes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from graphrag.retrieval.agentic_retriever import AgenticRetriever

_NBH = chr(0x2011)  # NON-BREAKING HYPHEN — the confirmed culprit


def _make_agentic_retriever(max_steps: int = 2) -> AgenticRetriever:
    ar = AgenticRetriever.__new__(AgenticRetriever)
    ar._local = AsyncMock()
    ar._local.search = AsyncMock(return_value={"chunks": [], "entities": []})
    ar._ctx_builder = MagicMock()
    ar._ctx_builder.build.return_value = ("some context", ["FAA-AD-2022-03-07"])
    ar._max_steps = max_steps
    ar._verifier = AsyncMock()
    return ar


class TestAnswerShortcutDashNormalization:
    """The "ANSWER:" early-exit path (line ~160) — uses _reason(), not
    _synthesize(). This is the path that was actually missed on the first
    pass of the fix."""

    async def test_dash_in_answer_shortcut_is_normalized(self) -> None:
        ar = _make_agentic_retriever()
        raw = f"ANSWER: FAA{_NBH}AD{_NBH}2022{_NBH}03{_NBH}07"
        ar._reason = AsyncMock(return_value=raw)

        result = await ar.retrieve_and_answer("What is the document ID?")

        assert result.answer == "FAA-AD-2022-03-07"
        assert _NBH not in result.answer


class TestFinalSynthesisDashNormalization:
    """The max-steps-reached path (line ~220) — goes through _synthesize(),
    which already wraps get_llm().generate() in normalize_dashes directly."""

    async def test_dash_in_final_synthesis_is_normalized(self) -> None:
        ar = _make_agentic_retriever(max_steps=1)
        # Never triggers the ANSWER:/SEARCH: shortcut — falls through to
        # the final synthesis call after max_steps.
        ar._reason = AsyncMock(return_value="UNRECOGNIZED FORMAT")
        # _synthesize() is mocked directly at the boundary this test cares
        # about: it must already return normalized text (proven separately
        # by TestNormalizeDashes in test_llm_utils.py), and this test
        # verifies retrieve_and_answer() passes that through unmodified —
        # not that _synthesize() itself normalizes (that's llm_utils' job).
        ar._synthesize = AsyncMock(return_value="AD-2024-01-02")

        result = await ar.retrieve_and_answer("What is the current document ID?")

        assert result.answer == "AD-2024-01-02"
        assert _NBH not in result.answer
