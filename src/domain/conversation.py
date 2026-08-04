"""Gong-shaped conversation entities (docs/plan.md §5 conversation entities, §7-8).

TranscriptSegment is the immutable source-level sentence/utterance (frozen model).
ExtractionWindow is a derived grouping used only to reduce extraction cost —
evidence offsets always remain relative to the immutable source segment, never to
a window's own (possibly overlapping) offsets (§7).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator

from src.domain.enums import ResolutionStatus, SpeakerRole


class Conversation(BaseModel):
    conversation_id: str
    workspace_id: str
    source_record_id: str
    source_system: str
    external_call_id: str
    occurred_at: datetime
    opportunity_id: str | None = None
    account_id: str | None = None


class Participant(BaseModel):
    participant_id: str
    workspace_id: str
    conversation_id: str
    speaker_label: str  # opaque raw speaker id/label from the source (§8 "opaque speaker IDs")
    contact_id: str | None = None
    seller_id: str | None = None
    role: SpeakerRole = SpeakerRole.UNKNOWN


class TranscriptSegment(BaseModel):
    """§5 — the immutable source-level sentence/utterance. Every source segment is
    persisted even when extraction is skipped (Codex prompt non-negotiable rules)."""
    model_config = ConfigDict(frozen=True)

    segment_id: str
    workspace_id: str
    conversation_id: str
    source_segment_index: int
    speaker_label: str
    text: str
    start_ms: int | None = None
    end_ms: int | None = None

    @property
    def char_length(self) -> int:
        return len(self.text)


class ExtractionWindow(BaseModel):
    """§7 — derived grouping over consecutive segments, used only to reduce
    extraction cost. Never authoritative for evidence offsets."""
    window_id: str
    workspace_id: str
    conversation_id: str
    segment_ids: list[str]
    start_segment_index: int
    end_segment_index: int

    @model_validator(mode="after")
    def _segment_range_is_well_formed(self) -> "ExtractionWindow":
        if self.end_segment_index < self.start_segment_index:
            raise ValueError("end_segment_index must be >= start_segment_index")
        if not self.segment_ids:
            raise ValueError("segment_ids must not be empty")
        return self


class Mention(BaseModel):
    """§5-6 — an entity mention detected inside a TranscriptSegment.

    Only self-contained well-formedness (0 <= char_start < char_end) is enforced
    here as a model_validator, since a Mention alone doesn't carry its segment's
    length. Cross-entity bound checking against the actual segment ("evidence
    spans map to exact source segments", §15) is `mention_within_segment` below —
    it needs both objects and so cannot live as a validator on Mention alone.
    """
    mention_id: str
    workspace_id: str
    segment_id: str
    char_start: int
    char_end: int
    surface_text: str
    normalized_surface: str
    entity_type: str
    resolved_entity_id: str | None = None
    resolution_status: ResolutionStatus = ResolutionStatus.UNRESOLVED

    @model_validator(mode="after")
    def _span_is_well_formed(self) -> "Mention":
        if self.char_start < 0:
            raise ValueError("char_start must be >= 0")
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be > char_start")
        return self


def mention_within_segment(mention: Mention, segment: TranscriptSegment) -> bool:
    """§15 — 'evidence spans map to exact source segments'. True iff `mention`'s
    span falls entirely within `segment`'s text and it is in fact a mention of
    that segment (segment_id match)."""
    if mention.segment_id != segment.segment_id:
        return False
    return mention.char_end <= segment.char_length


class SpeakerResolution(BaseModel):
    """§8 — resolved authority of an opaque transcript speaker label. Failed
    resolution still produces a record (role=UNKNOWN); it never discards Claims."""
    resolution_id: str
    workspace_id: str
    conversation_id: str
    speaker_label: str
    resolved_contact_id: str | None = None
    resolved_seller_id: str | None = None
    role: SpeakerRole = SpeakerRole.UNKNOWN
    evidence: str | None = None
