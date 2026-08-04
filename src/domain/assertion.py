"""Assertion and audit entities (docs/plan.md §5 assertion/audit entities, §6 Claim
fields, §8-9 resolution/review, §13 erasure).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, model_validator

from src.domain.enums import (
    AdjudicationStatus,
    ConflictStatus,
    ErasureStatus,
    Polarity,
    ResolutionStatus,
    SpeakerRole,
)


class Claim(BaseModel):
    """§6 — the exact field list. Multiple non-superseded contradictory Claims may
    coexist; is_superseded=False means "not replaced", not "accepted truth" — see
    adjudication_status for that judgment.
    """
    claim_id: str
    workspace_id: str
    subject_id: str
    predicate: str
    object_id: str | None = None
    object_value: str | None = None
    polarity: Polarity
    source_type: str
    source_record_id: str | None = None
    source_segment_id: str | None = None
    evidence_char_start: int
    evidence_char_end: int
    source_timestamp: datetime
    speaker_id: str | None = None
    speaker_role: SpeakerRole = SpeakerRole.UNKNOWN
    confidence: float
    valid_from: datetime
    valid_to: datetime | None = None
    transaction_from: datetime
    transaction_to: datetime | None = None
    is_superseded: bool = False
    adjudication_status: AdjudicationStatus = AdjudicationStatus.UNREVIEWED
    retention_class: str
    erasure_status: ErasureStatus = ErasureStatus.ACTIVE
    created_at: datetime

    @model_validator(mode="after")
    def _object_is_exactly_one_of_id_or_value(self) -> "Claim":
        if bool(self.object_id) == bool(self.object_value):
            raise ValueError("exactly one of object_id or object_value must be set")
        return self

    @model_validator(mode="after")
    def _evidence_span_is_well_formed(self) -> "Claim":
        if self.evidence_char_start < 0:
            raise ValueError("evidence_char_start must be >= 0")
        if self.evidence_char_end <= self.evidence_char_start:
            raise ValueError("evidence_char_end must be > evidence_char_start")
        return self

    @model_validator(mode="after")
    def _confidence_in_unit_interval(self) -> "Claim":
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        return self


class ExtractionRun(BaseModel):
    """§5 — one extraction EXECUTION (see src/domain/identity.py::extraction_run_id
    for why this is deliberately distinct from Claim/assertion identity)."""
    extraction_run_id: str
    workspace_id: str
    provider: str
    model: str
    prompt_version: str
    extractor_version: str
    run_nonce: str
    started_at: datetime
    completed_at: datetime | None = None


class ResolutionDecision(BaseModel):
    """§8 — the automated entity-resolution outcome for one Mention, with the full
    component-score breakdown required for explainability."""
    resolution_decision_id: str
    workspace_id: str
    mention_id: str
    resolved_entity_id: str | None = None
    status: ResolutionStatus
    lexical_score: float | None = None
    semantic_score: float | None = None
    base_score: float | None = None
    relational_bonus: float | None = None
    final_score: float | None = None
    margin: float | None = None
    relational_signals: list[str] = []
    decided_at: datetime

    @model_validator(mode="after")
    def _auto_linked_requires_resolved_entity(self) -> "ResolutionDecision":
        if self.status == ResolutionStatus.AUTO_LINKED and not self.resolved_entity_id:
            raise ValueError("AUTO_LINKED status requires resolved_entity_id")
        return self


class ReviewDecision(BaseModel):
    """§9 — a human reviewer's manual decision on a PENDING_REVIEW Mention. Carries
    everything §9 requires: reviewer identity, timestamp, selection/rejection, the
    candidate set actually shown, original scores, optional reason, affected
    Claims, and the overridden decision if any."""
    review_decision_id: str
    workspace_id: str
    mention_id: str
    reviewer_id: str
    decided_at: datetime
    selected_entity_id: str | None = None
    rejected: bool = False
    candidates_shown: list[str] = []
    original_scores: dict = {}
    reason: str | None = None
    affected_claim_ids: list[str] = []
    previous_review_decision_id: str | None = None

    @model_validator(mode="after")
    def _selection_and_rejection_are_mutually_exclusive(self) -> "ReviewDecision":
        if self.rejected and self.selected_entity_id:
            raise ValueError("a rejected review cannot also carry a selected_entity_id")
        if not self.rejected and not self.selected_entity_id:
            raise ValueError("a non-rejected review must set selected_entity_id")
        return self


class Conflict(BaseModel):
    """First-class conflict between two coexisting, contradictory Claims."""
    conflict_id: str
    workspace_id: str
    claim_id_a: str
    claim_id_b: str
    conflict_type: str
    status: ConflictStatus = ConflictStatus.OPEN
    detected_at: datetime
    resolved_at: datetime | None = None


class ErasureEvent(BaseModel):
    """§13 — audit record of an erasure request/completion, without retaining the
    erased personal content itself."""
    erasure_event_id: str
    workspace_id: str
    subject_type: str
    subject_id: str
    requested_at: datetime
    completed_at: datetime | None = None
    erasure_scope: list[str] = []  # e.g. ["text", "embeddings", "search_index", "cache"]
