"""Shared Pydantic dataclasses used across the entire pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class SourceType(str, Enum):
    """Origin of a fact — separates authoritative from inferred knowledge."""
    DOCUMENT = "document"   # extracted directly from a source document
    INFERRED = "inferred"   # derived by reasoning across documents
    LLM      = "llm"        # LLM-generated without direct document grounding
    MANUAL   = "manual"     # human-entered override


class ConstraintType(str, Enum):
    """How strictly a relation constraint must be respected."""
    HARD       = "hard"        # must — violating blocks assembly / process
    SOFT       = "soft"        # should — deviation requires justification
    REGULATORY = "regulatory"  # legally mandated (ITAR, FAA, EASA, etc.)
    ADVISORY   = "advisory"    # best practice, no hard enforcement


class AuthorityLevel(int, Enum):
    """Document authority hierarchy — lower number = higher authority."""
    REGULATORY         = 1   # airworthiness directives, regulations
    MANUFACTURER_SPEC  = 2   # OEM design specifications
    INTERNAL_PROCEDURE = 3   # company SOPs and work instructions
    INFORMAL           = 4   # emails, meeting notes, wiki pages


# ── Ingestion models ───────────────────────────────────────────────────────────

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    source_path: str
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"   # pending | processing | done | failed
    authority_level: int = AuthorityLevel.INFORMAL
    supersedes: list[str] = Field(default_factory=list)   # doc IDs this replaces
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    chunk_index: int
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: str   # PERSON | ORG | PRODUCT | CONCEPT | LOCATION | EVENT
    description: str = ""
    embedding: list[float] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_type: SourceType = SourceType.DOCUMENT
    canonical_id: str | None = None   # set when this is a duplicate of another entity


class Relation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity_id: str
    target_entity_id: str
    relation: str
    weight: float = 1.0
    confidence: float = 1.0
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    source_chunk_id: str = ""
    source_doc_id: str = ""
    source_type: SourceType = SourceType.DOCUMENT
    constraint_type: ConstraintType = ConstraintType.SOFT
    valid_from: datetime | None = None
    valid_to: datetime | None = None   # None = currently valid


# ── Graph models ───────────────────────────────────────────────────────────────

class Community(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    level: int
    member_entity_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    embedding: list[float] = Field(default_factory=list)
    member_count: int = 0


class CanonicalPart(BaseModel):
    """Single source of truth for a shared component used in multiple places."""
    part_number: str
    name: str
    description: str = ""
    spec_revision: str = ""
    material: str = ""
    supplier: str = ""
    embedding: list[float] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AliasEntry(BaseModel):
    """Raw name → canonical entity mapping for the alias registry."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    raw_value: str       # alternative name as seen in documents
    normalized: str      # lowercased, stripped, punctuation removed
    canonical_name: str
    canonical_type: str
    source_doc_id: str = ""
    confidence: float = 1.0


class ChangeLog(BaseModel):
    """Audit trail entry for every graph mutation."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    target_id: str
    target_label: str   # Entity | Relation | Document | etc.
    changed_by: str = "system"
    changed_at: datetime = Field(default_factory=datetime.utcnow)
    operation: str = "update"   # create | update | delete | merge
    old_values: dict[str, Any] = Field(default_factory=dict)
    new_values: dict[str, Any] = Field(default_factory=dict)
    source_doc_id: str = ""


# ── Query / retrieval models ───────────────────────────────────────────────────

class QueryResult(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    retrieval_mode: str = "hybrid"
    model_version: str = ""


class SessionTurn(BaseModel):
    """One exchange in a multi-turn conversational session."""
    turn_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer: str
    referenced_entities: list[str] = Field(default_factory=list)
    referenced_chunks: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Evaluation models ──────────────────────────────────────────────────────────

class EvalJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    query_result: QueryResult
    ground_truth: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvalResult(BaseModel):
    job_id: str
    query_id: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    scored_at: datetime = Field(default_factory=datetime.utcnow)


# ── Business Matrix models ─────────────────────────────────────────────────────

class KPIEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: float
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    cost_usd: float = 0.0
    retrieval_mode: str = "hybrid"
    model_version: str = ""


# ── Message queue payloads ─────────────────────────────────────────────────────

class IngestMessage(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    document: Document
    priority: str = "normal"


class QueryMessage(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    mode: str = "hybrid"
    ground_truth: str = ""
    tenant: str = "default"
    session_id: str = ""   # for multi-turn conversational context
