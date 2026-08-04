"""§16 — 'deterministic fake extraction is byte-stable.' FixtureExtractionProvider
has no randomness or hidden state, so repeated calls on identical input must
produce byte-identical JSON output.
"""

from __future__ import annotations

import pytest

from src.domain.conversation import ExtractionWindow
from src.extraction.fixture_provider import FixtureExtractionProvider
from src.extraction.provider import ExtractionInput, WindowSegmentText

pytestmark = pytest.mark.asyncio


def _input() -> ExtractionInput:
    window = ExtractionWindow(
        window_id="win-1", workspace_id="ws-1", conversation_id="conv-1",
        segment_ids=["seg-1", "seg-2"], start_segment_index=0, end_segment_index=1,
    )
    segments = [
        WindowSegmentText(segment_id="seg-1", speaker_label="spk_1", text="We are concerned about the pricing."),
        WindowSegmentText(segment_id="seg-2", speaker_label="spk_2", text="I understand, let's discuss security too."),
    ]
    return ExtractionInput(window=window, segments=segments)


async def test_fixture_extraction_is_byte_stable_across_repeated_runs():
    provider = FixtureExtractionProvider()
    item = _input()

    first = await provider.extract([item])
    second = await provider.extract([item])

    assert first[0].model_dump_json() == second[0].model_dump_json()


async def test_fixture_extraction_finds_expected_assertions():
    provider = FixtureExtractionProvider()
    results = await provider.extract([_input()])
    predicates = {a.predicate for a in results[0].assertions}
    assert "RAISED_OBJECTION" in predicates
    assert "HAS_BLOCKER" in predicates
