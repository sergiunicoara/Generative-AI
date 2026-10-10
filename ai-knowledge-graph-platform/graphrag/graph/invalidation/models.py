"""Invalidation events, derived-artifact states and their allowed transitions."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class EventKind(str, Enum):
    FACT_CORRECTED = "FACT_CORRECTED"            # an entity was quarantined / corrected
    RELATION_CHANGED = "RELATION_CHANGED"        # an edge was rejected, retracted, overridden
    SUPERSEDED = "SUPERSEDED"                    # a newer document superseded older ones
    DOCUMENT_REINGESTED = "DOCUMENT_REINGESTED"  # a document's evidence was replaced
    EVIDENCE_EXPIRED = "EVIDENCE_EXPIRED"        # valid_to passed on a document or edge
    ER_REVISED = "ER_REVISED"                    # an entity-resolution decision changed
    SCHEMA_CHANGED = "SCHEMA_CHANGED"            # the active schema version changed


class ArtifactKind(str, Enum):
    DECISION = "decision"                  # a recorded answer (CGDecision)
    COMMUNITY_SNAPSHOT = "community_snapshot"
    INFERRED_EDGE = "inferred_edge"


class ArtifactState(str, Enum):
    VALID = "VALID"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    RECOMPUTING = "RECOMPUTING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    RECOMPUTE_FAILED = "RECOMPUTE_FAILED"


S = ArtifactState
# Any state may be invalidated again (a new event can arrive while recomputing
# or after a terminal outcome). Only a claimed (RECOMPUTING) artifact can be
# completed, and a recompute may hand an artifact back to human review.
ALLOWED_TRANSITIONS: dict[ArtifactState, frozenset[ArtifactState]] = {
    S.VALID: frozenset({S.NEEDS_REVIEW}),
    S.NEEDS_REVIEW: frozenset({S.NEEDS_REVIEW, S.RECOMPUTING}),
    S.RECOMPUTING: frozenset({S.VALID, S.NEEDS_REVIEW, S.INSUFFICIENT_EVIDENCE,
                              S.VALIDATION_FAILED, S.RECOMPUTE_FAILED}),
    S.INSUFFICIENT_EVIDENCE: frozenset({S.NEEDS_REVIEW}),
    S.VALIDATION_FAILED: frozenset({S.NEEDS_REVIEW}),
    S.RECOMPUTE_FAILED: frozenset({S.NEEDS_REVIEW}),
}
COMPLETION_STATES = ALLOWED_TRANSITIONS[S.RECOMPUTING]


class InvalidTransition(ValueError):
    pass


def check_transition(current: ArtifactState, target: ArtifactState) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransition(f"{current.value} -> {target.value} is not allowed")


class EntityRef(BaseModel):
    name: str
    type: str

    @property
    def token(self) -> str:
        return f"{self.type}:{self.name}"


class RelationRef(BaseModel):
    src_name: str
    src_type: str
    relation: str
    tgt_name: str
    tgt_type: str

    @property
    def key(self) -> str:
        from graphrag.graph.inference_engine import relation_key
        return relation_key(self.src_type, self.src_name, self.relation, self.tgt_type, self.tgt_name)

    @property
    def endpoints(self) -> list[EntityRef]:
        return [EntityRef(name=self.src_name, type=self.src_type),
                EntityRef(name=self.tgt_name, type=self.tgt_type)]


class InvalidationEvent(BaseModel):
    """A change to an underlying fact. Its id is deterministic in (tenant, kind,
    subjects, cause), so a redelivered or repeated event is a no-op."""

    tenant: str
    kind: EventKind
    reason: str = ""
    actor: str = "system"
    cause: str = ""  # stable id of what caused it (audit id, doc id + valid_to, ...)
    entities: list[EntityRef] = Field(default_factory=list)
    relations: list[RelationRef] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    schema_version: str | None = None  # SCHEMA_CHANGED: the newly active label
    # True when the change can make answers that cited nothing related wrong
    # (a new or re-enabled fact). Those still need the tenant-wide revision bump.
    additive: bool = False
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def id(self) -> str:
        subjects = {
            "entities": sorted(e.token for e in self.entities),
            "relations": sorted(r.key for r in self.relations),
            "documents": sorted(self.document_ids),
            "chunks": sorted(self.chunk_ids),
            "schema": self.schema_version,
        }
        raw = json.dumps([self.tenant, self.kind.value, subjects, self.cause], sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def subjects_json(self) -> str:
        return self.model_dump_json(include={"entities", "relations", "document_ids",
                                             "chunk_ids", "schema_version"})


@dataclass
class Closure:
    """Everything an event reaches, after following dependencies to a fixpoint."""
    entities: dict[str, EntityRef] = field(default_factory=dict)  # token -> ref
    entity_ids: set[str] = field(default_factory=set)
    document_ids: set[str] = field(default_factory=set)
    chunk_ids: set[str] = field(default_factory=set)
    relation_keys: set[str] = field(default_factory=set)
    inferred_edges: dict[str, dict] = field(default_factory=dict)  # key -> {src.., rule}
    snapshot_ids: set[str] = field(default_factory=set)
    decision_ids: set[str] = field(default_factory=set)
    depth_reached: int = 0
    truncated: bool = False

    def counts(self) -> dict[str, int]:
        return {
            "decisions": len(self.decision_ids),
            "community_snapshots": len(self.snapshot_ids),
            "inferred_edges": len(self.inferred_edges),
            "chunks": len(self.chunk_ids),
            "entities": len(self.entities),
        }
