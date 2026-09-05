"""Assertion and audit entities (docs/plan.md §5 assertion/audit entities, §6 Claim
fields, §8-9 resolution/review, §13 erasure).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, model_validator

from src.domain.enums import (
    AdjudicationStatus,
    ConflictStatus,
    ConflictType,
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

    # --- Resolved subject identity (§8/§9) ---------------------------------
    # `subject_id` above is the *surface* the claim was made about (a speaker's
    # normalized name, or an opaque speaker_label when the transcript gives no
    # name). These four record what entity resolution concluded that surface
    # refers to, and are deliberately properties rather than a
    # (:Claim)-[:ABOUT]->(:Contact) edge: §10 (quoted at the top of
    # claim_repository.py) forbids Claim fan-out directly from Account/Contact,
    # routing evidence through Conversation/TranscriptSegment instead.
    #
    # All four are optional and default to "not resolved", so every existing
    # construction site and every stored Claim written before this existed
    # stays valid -- resolution is additive evidence, never a precondition for
    # recording that something was said.
    resolved_entity_id: str | None = None
    resolved_entity_type: str | None = None  # "Account" | "Contact"
    resolution_status: ResolutionStatus | None = None
    resolution_score: float | None = None

    # --- Extraction provenance (P3.2) --------------------------------------
    # Which ExtractionRun produced this Claim -- distinct from assertion_id
    # (§6), which deliberately excludes extraction-execution details so a
    # re-extraction by a newer model links to the same Claim. This field is
    # the other half: additive provenance that lets "which model/prompt
    # version produced this specific assertion" actually be queried.
    extraction_run_id: str | None = None

    # --- Adjudication audit trail ------------------------------------------
    # adjudication_status itself (line 48) stays a plain current-value
    # property -- these three carry *why*/*who*/*when* for the most recent
    # transition. History beyond "most recent" lives in ClaimRevision nodes
    # via ClaimRepository.transition_adjudication_status(), not here; these
    # three mirror only the live node's current transition for cheap reads
    # that don't need the full history.
    adjudication_reason: str | None = None
    adjudication_decided_by: str | None = None
    adjudication_decided_at: datetime | None = None

    # --- Source authority ---------------------------------------------------
    # Denormalized from Conversation.source_system at Claim-construction time
    # (src/ingestion/transcript_pipeline.py) so ContextGraphBuilder._score_claim
    # can weigh source authority without an extra repository fetch per Claim
    # (would otherwise be an N+1 join through source_record_id -> SourceRecord).
    source_system: str | None = None

    @model_validator(mode="after")
    def _resolved_entity_requires_auto_link(self) -> "Claim":
        """Mirror of ResolutionDecision's own invariant.

        A resolved entity id may only be attached to a claim whose resolution
        actually auto-linked. PENDING_REVIEW and UNRESOLVED must leave the id
        empty -- otherwise a low-confidence guess becomes indistinguishable
        from a confirmed link once it is read back out of the graph.
        """
        if self.resolved_entity_id and self.resolution_status not in (
            ResolutionStatus.AUTO_LINKED,
            None,  # None = set directly by a reviewer's confirmation path
        ):
            raise ValueError(
                "resolved_entity_id may only be set when resolution_status is AUTO_LINKED"
            )
        if self.resolution_score is not None and not (0.0 <= self.resolution_score <= 1.0):
            raise ValueError("resolution_score must be in [0.0, 1.0]")
        return self

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


class CandidateScore(BaseModel):
    """P4.2/P4.3 — one scored candidate as considered during resolution, kept
    alongside the winner so a genuine multi-way ambiguity is recoverable after
    the fact, not collapsed to a single top-1 score breakdown."""
    entity_id: str
    entity_type: str
    lexical_score: float
    semantic_score: float | None = None
    base_score: float
    relational_bonus: float
    final_score: float
    rank: int


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
    # P4.2/P4.3 — the full ranked candidate set (bounded, see
    # src/resolution/pipeline.py's _CANDIDATE_SCORES_CAP), not just the
    # winner. Empty for a Stage A deterministic match, which has no runner-up
    # to show. Deliberately a property, not a (:ResolutionDecision)-
    # [:POSSIBLY_REFERS_TO]->(:Account|:Contact) edge fan-out — see §10 as
    # quoted in claim_repository.py.
    candidates: list[CandidateScore] = []
    # P4.4 — which threshold set produced this decision, so re-tuning
    # src/resolution/policy.py's defaults can be traced back to affected
    # records. "deterministic" for Stage A matches, which never go through
    # decide()/PolicyThresholds at all.
    policy_version: str

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
    conflict_type: ConflictType
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
