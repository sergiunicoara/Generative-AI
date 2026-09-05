"""Pure-function scoring tests for src/context_graph/builder.py -- no live
Neo4j needed since _score_claim and _blend_relevance are pure functions over
already-in-memory values.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.context_graph.builder import _blend_relevance, _normalize_relevance, _score_claim
from src.domain.assertion import Claim
from src.domain.enums import AdjudicationStatus, Polarity, SpeakerRole

_T0 = datetime(2026, 6, 15, tzinfo=timezone.utc)


def _claim(*, source_system: str | None = None, confidence: float = 0.5) -> Claim:
    return Claim(
        claim_id="claim-1", workspace_id="ws-1", subject_id="contact-1",
        predicate="RAISED_OBJECTION", object_value="pricing", polarity=Polarity.AFFIRMED,
        source_type="transcript", evidence_char_start=0, evidence_char_end=5,
        source_timestamp=_T0, speaker_role=SpeakerRole.BUYER, confidence=confidence,
        valid_from=_T0, transaction_from=_T0, adjudication_status=AdjudicationStatus.UNREVIEWED,
        retention_class="standard", created_at=_T0, source_system=source_system,
    )


def test_score_claim_weighs_known_source_system_higher():
    salesforce_score = _score_claim(_claim(source_system="salesforce"), now=_T0)
    gong_score = _score_claim(_claim(source_system="gong"), now=_T0)
    assert salesforce_score > gong_score


def test_score_claim_defaults_unknown_or_missing_source_system_to_neutral_weight():
    missing_score = _score_claim(_claim(source_system=None), now=_T0)
    unknown_score = _score_claim(_claim(source_system="some_future_adapter"), now=_T0)
    assert missing_score == unknown_score


def test_blend_relevance_at_logit_zero_is_the_midpoint():
    # sigmoid(0) == 0.5, the exact midpoint of the normalized relevance range.
    assert _normalize_relevance(0.0) == 0.5


def test_blend_relevance_is_monotonic_in_relevance():
    base_score = 0.6
    low = _blend_relevance(base_score, raw_relevance=-5.0)
    mid = _blend_relevance(base_score, raw_relevance=0.0)
    high = _blend_relevance(base_score, raw_relevance=5.0)
    assert low < mid < high


def test_blend_relevance_never_fully_replaces_the_base_score():
    # A very confident base score should still pull the blended result up
    # even against a maximally negative relevance signal -- the whole point
    # of blending instead of replacing.
    high_base_low_relevance = _blend_relevance(1.0, raw_relevance=-10.0)
    low_base_low_relevance = _blend_relevance(0.0, raw_relevance=-10.0)
    assert high_base_low_relevance > low_base_low_relevance
