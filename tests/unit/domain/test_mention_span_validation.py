"""§15 — 'evidence spans map to exact source segments.' Mention enforces its own
well-formedness (0 <= char_start < char_end) via a model_validator; whether a
Mention's span actually falls inside its referenced TranscriptSegment's text is a
separate, cross-entity check (mention_within_segment) since a Mention alone
doesn't carry its segment's length.
"""

import pytest
from pydantic import ValidationError

from src.domain.conversation import Mention, TranscriptSegment, mention_within_segment


def _segment(text: str = "We are concerned about pricing this quarter.") -> TranscriptSegment:
    return TranscriptSegment(
        segment_id="seg-1",
        workspace_id="ws-1",
        conversation_id="conv-1",
        source_segment_index=0,
        speaker_label="spk_1",
        text=text,
    )


def _mention(**overrides) -> Mention:
    base = dict(
        mention_id="mention-1",
        workspace_id="ws-1",
        segment_id="seg-1",
        char_start=24,
        char_end=32,
        surface_text="pricing",
        normalized_surface="pricing",
        entity_type="OBJECTION",
    )
    base.update(overrides)
    return Mention(**base)


def test_valid_span_constructs_successfully():
    mention = _mention(char_start=0, char_end=5)
    assert mention.char_start == 0
    assert mention.char_end == 5


def test_negative_char_start_is_rejected():
    with pytest.raises(ValidationError, match="char_start"):
        _mention(char_start=-1, char_end=5)


def test_char_end_not_greater_than_char_start_is_rejected():
    with pytest.raises(ValidationError, match="char_end"):
        _mention(char_start=10, char_end=10)
    with pytest.raises(ValidationError, match="char_end"):
        _mention(char_start=10, char_end=5)


def test_mention_within_segment_true_for_span_inside_text():
    segment = _segment()
    mention = _mention(segment_id=segment.segment_id, char_start=24, char_end=32)
    assert mention_within_segment(mention, segment) is True


def test_mention_within_segment_false_when_span_exceeds_segment_length():
    segment = _segment(text="short")  # length 5
    mention = _mention(segment_id=segment.segment_id, char_start=0, char_end=10)
    assert mention_within_segment(mention, segment) is False


def test_mention_within_segment_false_for_mismatched_segment_id():
    segment = _segment()
    mention = _mention(segment_id="a-different-segment", char_start=0, char_end=5)
    assert mention_within_segment(mention, segment) is False
