"""§7 window construction rules 2-4: duration/token/topic-boundary triggers,
configurable overlap, and never discarding the closing portion of a call.
"""

from __future__ import annotations

from src.domain.conversation import TranscriptSegment
from src.extraction.windowing import build_windows


def _segment(index: int, text: str, start_ms: int, end_ms: int, speaker: str = "spk_1") -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=f"seg-{index}", workspace_id="ws-1", conversation_id="conv-1",
        source_segment_index=index, speaker_label=speaker, text=text, start_ms=start_ms, end_ms=end_ms,
    )


def test_empty_segments_produce_no_windows():
    assert build_windows([], workspace_id="ws-1", conversation_id="conv-1") == []


def test_short_call_is_a_single_window():
    segments = [
        _segment(0, "Hello there.", 0, 1000),
        _segment(1, "How are you?", 1000, 2000),
    ]
    windows = build_windows(segments, workspace_id="ws-1", conversation_id="conv-1")
    assert len(windows) == 1
    assert windows[0].segment_ids == ["seg-0", "seg-1"]


def test_duration_budget_splits_into_multiple_windows():
    segments = [
        _segment(0, "First part of the call.", 0, 10_000),
        _segment(1, "Still going strong here.", 10_000, 100_000),  # exceeds default 90s budget
        _segment(2, "Wrapping up now.", 100_000, 110_000),
    ]
    windows = build_windows(segments, workspace_id="ws-1", conversation_id="conv-1", max_duration_ms=90_000)
    assert len(windows) >= 2
    # rule 4 — never discard the closing portion
    all_segment_ids = {sid for w in windows for sid in w.segment_ids}
    assert "seg-2" in all_segment_ids


def test_never_discards_the_closing_portion():
    segments = [_segment(i, f"Segment number {i}.", i * 1000, (i + 1) * 1000) for i in range(30)]
    windows = build_windows(segments, workspace_id="ws-1", conversation_id="conv-1", max_tokens=10)
    last_window = windows[-1]
    assert last_window.segment_ids[-1] == "seg-29"


def test_configurable_overlap_between_adjacent_windows():
    segments = [_segment(i, f"Segment number {i} here now.", i * 1000, (i + 1) * 1000) for i in range(10)]
    windows = build_windows(
        segments, workspace_id="ws-1", conversation_id="conv-1", max_tokens=15, overlap_segments=2
    )
    assert len(windows) >= 2
    first_window_ids = windows[0].segment_ids
    second_window_ids = windows[1].segment_ids
    overlap = set(first_window_ids) & set(second_window_ids)
    assert len(overlap) == 2


def test_topic_gap_triggers_a_new_window():
    segments = [
        _segment(0, "Talking about pricing now.", 0, 1000),
        _segment(1, "Now a totally different topic after a long pause.", 20_000, 21_000),  # 19s silence gap
    ]
    windows = build_windows(segments, workspace_id="ws-1", conversation_id="conv-1", topic_gap_ms=3_000)
    assert len(windows) == 2


def test_window_ids_are_deterministic_for_identical_input():
    segments = [_segment(0, "Hello there.", 0, 1000)]
    first = build_windows(segments, workspace_id="ws-1", conversation_id="conv-1")
    second = build_windows(segments, workspace_id="ws-1", conversation_id="conv-1")
    assert first[0].window_id == second[0].window_id
