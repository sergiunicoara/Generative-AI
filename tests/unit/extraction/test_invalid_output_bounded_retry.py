"""§16 — 'invalid LLM output fails explicitly after bounded retries.' chat_fn is
injected, so no real API key/network is needed — this tests the retry/repair
loop and permanent-failure behavior in isolation.
"""

from __future__ import annotations

import pytest

from src.domain.conversation import ExtractionWindow
from src.extraction.llm_provider import ExtractionFailedPermanently, LlmExtractionProvider
from src.extraction.provider import ExtractionInput, WindowSegmentText

pytestmark = pytest.mark.asyncio


def _input() -> ExtractionInput:
    window = ExtractionWindow(
        window_id="win-1", workspace_id="ws-1", conversation_id="conv-1",
        segment_ids=["seg-1"], start_segment_index=0, end_segment_index=0,
    )
    return ExtractionInput(
        window=window,
        segments=[WindowSegmentText(segment_id="seg-1", speaker_label="spk_1", text="We are concerned about pricing.")],
    )


async def test_valid_output_on_first_attempt_succeeds():
    async def chat_fn(prompt: str) -> str:
        return '{"assertions": []}'

    provider = LlmExtractionProvider(chat_fn, max_attempts=3)
    results = await provider.extract([_input()])
    assert results[0].window_id == "win-1"
    assert results[0].assertions == []


async def test_malformed_json_retries_then_succeeds():
    calls = []

    async def chat_fn(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) < 3:
            return "not json at all"
        return '{"assertions": []}'

    provider = LlmExtractionProvider(chat_fn, max_attempts=3)
    results = await provider.extract([_input()])
    assert len(calls) == 3
    assert results[0].assertions == []
    # the repair prompt on retry includes the previous error
    assert "invalid" in calls[1].lower()


async def test_schema_invalid_json_retries_then_succeeds():
    calls = []

    async def chat_fn(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) < 2:
            return '{"assertions": [{"predicate": "X"}]}'  # missing required fields
        return '{"assertions": []}'

    provider = LlmExtractionProvider(chat_fn, max_attempts=3)
    results = await provider.extract([_input()])
    assert len(calls) == 2
    assert results[0].assertions == []


async def test_permanently_invalid_output_fails_explicitly_after_bounded_retries():
    calls = []

    async def chat_fn(prompt: str) -> str:
        calls.append(prompt)
        return "still not json"

    provider = LlmExtractionProvider(chat_fn, max_attempts=3)

    with pytest.raises(ExtractionFailedPermanently) as exc_info:
        await provider.extract([_input()])

    assert len(calls) == 3  # bounded — not unbounded retry
    assert exc_info.value.window_id == "win-1"
    assert exc_info.value.attempts == 3


async def test_max_attempts_must_be_positive():
    async def chat_fn(prompt: str) -> str:
        return "{}"

    with pytest.raises(ValueError):
        LlmExtractionProvider(chat_fn, max_attempts=0)
