"""§16 — 'negated and hypothetical variants remain distinct Claims.' Proven at
two levels: the fixture extractor's polarity detection, and (already proven in
tests/unit/domain/test_claim_identity_split.py) that polarity is part of
assertion_id — different polarity always means a different Claim.
"""

from __future__ import annotations

import pytest

from src.domain.conversation import ExtractionWindow
from src.domain.enums import Polarity
from src.extraction.fixture_provider import FixtureExtractionProvider
from src.extraction.provider import ExtractionInput, WindowSegmentText

pytestmark = pytest.mark.asyncio


def _window() -> ExtractionWindow:
    return ExtractionWindow(
        window_id="win-1", workspace_id="ws-1", conversation_id="conv-1",
        segment_ids=["seg-1"], start_segment_index=0, end_segment_index=0,
    )


async def test_affirmed_pricing_objection():
    provider = FixtureExtractionProvider()
    item = ExtractionInput(
        window=_window(),
        segments=[WindowSegmentText(segment_id="seg-1", speaker_label="spk_1", text="We are concerned about the pricing.")],
    )
    results = await provider.extract([item])
    assertion = next(a for a in results[0].assertions if a.predicate == "RAISED_OBJECTION")
    assert assertion.polarity == Polarity.AFFIRMED


async def test_negated_pricing_objection():
    provider = FixtureExtractionProvider()
    item = ExtractionInput(
        window=_window(),
        segments=[WindowSegmentText(segment_id="seg-1", speaker_label="spk_1", text="We are not concerned about the pricing.")],
    )
    results = await provider.extract([item])
    assertion = next(a for a in results[0].assertions if a.predicate == "RAISED_OBJECTION")
    assert assertion.polarity == Polarity.NEGATED


async def test_hypothetical_pricing_objection():
    provider = FixtureExtractionProvider()
    item = ExtractionInput(
        window=_window(),
        segments=[WindowSegmentText(segment_id="seg-1", speaker_label="spk_1", text="If the pricing changed, that could be an issue.")],
    )
    results = await provider.extract([item])
    assertion = next(a for a in results[0].assertions if a.predicate == "RAISED_OBJECTION")
    assert assertion.polarity == Polarity.HYPOTHETICAL


async def test_all_three_polarity_variants_produce_distinct_assertion_ids():
    from src.domain.identity import assertion_id

    common = dict(
        workspace="ws-1", source_segment_id="seg-1", evidence_char_start=0, evidence_char_end=10,
        canonical_subject="spk_1", predicate="RAISED_OBJECTION", normalized_object="pricing",
    )
    ids = {
        assertion_id(**common, polarity=p.value)
        for p in (Polarity.AFFIRMED, Polarity.NEGATED, Polarity.HYPOTHETICAL)
    }
    assert len(ids) == 3
