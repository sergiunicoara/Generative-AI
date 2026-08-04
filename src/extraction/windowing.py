"""§7 window construction — the 5 rules:
1. Persist every source segment before extraction (enforced by the ingestion
   pipeline's call order, not this module — this module only groups segments
   that are assumed already persisted).
2. Accumulate consecutive speaker turns until a topic boundary, 60-90s, or a
   configurable token budget is reached.
3. Add a small configurable overlap between adjacent windows.
4. Never discard the closing portion of a call.
5. Deduplicate assertions from overlapping windows by stable assertion_id (this
   falls out of src/graph/repositories/claim_repository.py's MERGE-on-claim_id
   write, given assertion_id is itself content-deterministic — nothing extra is
   needed here beyond producing overlapping windows correctly).

'Topic boundary' has no real topic-segmentation model in this vertical slice —
a configurable silence gap between consecutive segments (`topic_gap_ms`) stands
in as a documented heuristic, not a claim of real topic detection.
"""

from __future__ import annotations

import hashlib

from src.domain.conversation import ExtractionWindow, TranscriptSegment

_SEP = "\x1f"


def _window_id(conversation_id: str, start_segment_index: int, end_segment_index: int) -> str:
    # Deterministic but deliberately NOT in src/domain/identity.py's §6 list —
    # ExtractionWindow isn't a §6-mandated stable identity, just a convenience
    # for idempotent re-windowing under the same config.
    joined = _SEP.join(str(p) for p in (conversation_id, start_segment_index, end_segment_index))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_windows(
    segments: list[TranscriptSegment],
    *,
    workspace_id: str,
    conversation_id: str,
    max_duration_ms: int = 90_000,
    max_tokens: int = 200,
    overlap_segments: int = 1,
    topic_gap_ms: int = 3_000,
) -> list[ExtractionWindow]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s.source_segment_index)
    windows: list[ExtractionWindow] = []
    current: list[TranscriptSegment] = []
    window_start_ms: int | None = None
    prev_end_ms: int | None = None

    def flush() -> None:
        if not current:
            return
        windows.append(ExtractionWindow(
            window_id=_window_id(conversation_id, current[0].source_segment_index, current[-1].source_segment_index),
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            segment_ids=[s.segment_id for s in current],
            start_segment_index=current[0].source_segment_index,
            end_segment_index=current[-1].source_segment_index,
        ))

    def current_tokens() -> int:
        return sum(len(s.text.split()) for s in current)

    for seg in ordered:
        topic_boundary = (
            bool(current) and seg.start_ms is not None and prev_end_ms is not None
            and (seg.start_ms - prev_end_ms) >= topic_gap_ms
        )
        duration_exceeded = (
            bool(current) and seg.end_ms is not None and window_start_ms is not None
            and (seg.end_ms - window_start_ms) > max_duration_ms
        )
        tokens_exceeded = bool(current) and (current_tokens() + len(seg.text.split())) > max_tokens

        if current and (topic_boundary or duration_exceeded or tokens_exceeded):
            flush()
            overlap = current[-overlap_segments:] if overlap_segments > 0 else []
            current = list(overlap)
            window_start_ms = current[0].start_ms if current else seg.start_ms

        if not current:
            window_start_ms = seg.start_ms
        current.append(seg)
        prev_end_ms = seg.end_ms

    flush()  # rule 4 — never discard the closing portion of the call
    return windows
