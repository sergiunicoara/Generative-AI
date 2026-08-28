"""Shared Pydantic dataclasses used across the entire pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from graphrag.enterprise.models import (
    AccessContext,
    DocumentLink,
    DocumentAccessPolicy,
    LineageAssertion,
    MetadataEnvelope,
    ObligationDraft,
)


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
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"   # pending | processing | done | failed
    authority_level: int = AuthorityLevel.INFORMAL
    supersedes: list[str] = Field(default_factory=list)   # doc IDs this replaces
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    tenant: str = "default"
    source_id: str | None = None   # optional KGSource catalog reference
    # Governed three-tier metadata: universal envelope, per-collection schema
    # fields, and bounded open-discovery metadata.  Kept separate from legacy
    # ``metadata`` so older API callers remain compatible while new sources get
    # a stable, versioned contract.
    metadata_envelope: MetadataEnvelope = Field(default_factory=MetadataEnvelope)
    # Document ACL is written with the document and inherited by every chunk at
    # query time.  It is never accepted from a query request.
    access_policy: DocumentAccessPolicy = Field(default_factory=DocumentAccessPolicy)
    # Explicit outgoing document references observed in the source.  They are
    # never inferred from semantic similarity; ingestion resolves them to
    # tenant-scoped Document-[:LINKS_TO]->Document edges when a target exists.
    outbound_links: list[DocumentLink] = Field(default_factory=list, max_length=10_000)
    # Only explicit, source-backed claims enter these lists.  The ingestion
    # writer turns them into pending human-review items by default.
    lineage_assertions: list[LineageAssertion] = Field(default_factory=list)
    obligation_drafts: list[ObligationDraft] = Field(default_factory=list)
    # sha256 of raw_text (graphrag/core/content_hash.py). Lets ingestion tell
    # whether a re-ingested file actually changed instead of re-chunking,
    # re-embedding and re-extracting an unchanged document every run. Empty
    # string means "not computed" (data predating this field) and is treated
    # as "assume changed", never as a real hash.
    content_hash: str = ""
    # Soft-delete markers, set when a source document disappears from the
    # corpus. Deliberately NOT a physical delete — that is GDPR erasure's job
    # (graphrag/graph/gdpr.py). A tombstoned document's chunks are excluded
    # from retrieval but stay recoverable, matching how `quarantined` already
    # works for entities.
    is_deleted: bool = False
    deleted_at: datetime | None = None


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    chunk_index: int
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tenant: str = "default"


class IntelligenceArtifact(BaseModel):
    """A source-grounded assertion extracted during ingestion.

    This is deliberately distinct from :class:`ClaimNode` in
    ``graphrag.evidence.claim_graph``: the latter represents a sentence in an
    answer, while this model represents an assertion the *source document*
    made.  ``evidence_quote`` must be a verbatim span from ``source_chunk_id``.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    artifact_type: Literal["CLAIM", "OBSERVATION", "EVENT", "FINDING"]
    text: str
    evidence_quote: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunk_id: str
    source_doc_id: str
    entity_names: list[str] = Field(default_factory=list)
    event_start: datetime | None = None
    event_end: datetime | None = None
    extraction_model: str = ""
    prompt_version: str = "intelligence-v1"
    tenant: str = "default"


class StructuredTable(BaseModel):
    """A source table retained as a queryable JSON-LD-shaped graph artifact."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    table_index: int
    caption: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    source_page: int | None = None
    extraction_method: str = "structured_source"
    source_chunk_id: str = ""
    tenant: str = "default"

    def as_jsonld(self) -> dict[str, Any]:
        return {
            "@context": {"schema": "https://schema.org/"},
            "@type": "schema:Table",
            "@id": f"urn:graphrag:table:{self.id}",
            "schema:name": self.caption,
            "columns": self.columns,
            "rows": self.rows,
            "sourcePage": self.source_page,
            "extractionMethod": self.extraction_method,
        }


class IngestionRunManifest(BaseModel):
    """Durable, integrity-protected receipt for one document ingestion run."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    job_id: str
    tenant: str
    document_id: str = ""
    filename: str
    content_hash: str = ""
    correlation_id: str = ""
    model_provider: str = ""
    model_version: str = ""
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    stage_metrics: dict[str, dict[str, float | int | str | None]] = Field(default_factory=dict)
    status: Literal["running", "completed", "failed"] = "running"
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error: str = ""
    integrity_hash: str = ""

    def compute_integrity_hash(self) -> str:
        import hashlib
        import json

        payload = self.model_dump(mode="json", exclude={"integrity_hash", "completed_at", "status", "error"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()


class Entity(BaseModel):
    """An extraction-local entity mention.

    ``id`` binds relations produced in the same extraction response; it is not
    a durable graph identifier.  The canonical graph identity is the scoped
    ``(tenant, canonical_name, canonical_type)`` natural key.  This replaces
    the unused ``canonical_id`` field, whose meaning conflicted with document
    canonical IDs and Neo4j node IDs.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: str   # PERSON | ORG | PRODUCT | CONCEPT | LOCATION | EVENT
    description: str = ""
    # Extraction confidence is used by optional post-ingestion enrichment
    # (for example Wikidata linking).  Keeping it on the model prevents
    # Pydantic from silently discarding the extractor's value.
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    embedding: list[float] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_type: SourceType = SourceType.DOCUMENT
    tenant: str = "default"           # tenant scope — entities are isolated per tenant
    canonical_name: str = ""          # resolver-approved canonical entity name
    canonical_type: str = ""          # resolver-approved canonical entity type
    # ── Deep provenance ────────────────────────────────────────────────────────
    source_doc_id: str = ""           # first document to introduce this entity
    extraction_model: str = ""        # LLM model that extracted this entity
    prompt_version: str = "v1"        # prompt template version at extraction time

    @property
    def canonical_identity(self) -> tuple[str, str] | None:
        """Resolved `(name, type)` used for graph writes, if this mention redirected."""
        if not self.canonical_name:
            return None
        return self.canonical_name, self.canonical_type or self.type

    def redirect_to(self, name: str, entity_type: str) -> None:
        """Record the resolver's canonical natural key on this transient mention."""
        self.canonical_name = name
        self.canonical_type = entity_type


class Relation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity_id: str
    target_entity_id: str
    relation: str
    weight: float = 1.0
    confidence: float = 1.0
    confidence_state: str = "ASSERTED"  # ASSERTED | INFERRED | DISPUTED | RETRACTED | APPROVED
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_chunk_id: str = ""
    source_doc_id: str = ""
    source_type: SourceType = SourceType.DOCUMENT
    constraint_type: ConstraintType = ConstraintType.SOFT
    valid_from: datetime | None = None
    valid_to: datetime | None = None   # None = currently valid
    # ── Deep provenance ────────────────────────────────────────────────────────
    chunk_span_start: int | None = None   # character offset where relation was found
    chunk_span_end: int | None = None
    extraction_model: str = ""
    prompt_version: str = "v1"


# ── Graph models ───────────────────────────────────────────────────────────────

class Community(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    level: int
    member_entity_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    embedding: list[float] = Field(default_factory=list)
    member_count: int = 0
    tenant: str = "default"


class CanonicalPart(BaseModel):
    """Single source of truth for a shared component used in multiple places."""
    part_number: str
    name: str
    description: str = ""
    spec_revision: str = ""
    material: str = ""
    supplier: str = ""
    embedding: list[float] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
    changed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    operation: str = "update"   # create | update | delete | merge
    old_values: dict[str, Any] = Field(default_factory=dict)
    new_values: dict[str, Any] = Field(default_factory=dict)
    source_doc_id: str = ""


# ── Knowledge graph extension models ──────────────────────────────────────────

class NegativeRelation(BaseModel):
    """
    Asserts that a relation does NOT hold between two entities.

    Stored as a NEGATIVE_RELATES_TO edge with the same provenance model as
    Relation (source_doc_ids accumulation, confidence, valid_from/valid_to).
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity_name: str
    source_entity_type: str
    target_entity_name: str
    target_entity_type: str
    relation: str
    confidence: float = 1.0
    source_doc_id: str = ""
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    tenant: str = "default"
    asserted_by: str = "system"


class Statement(BaseModel):
    """
    A reified relation — a triple (subject, relation, object) promoted to a
    first-class node so that meta-statements can be made about it.

    Stored as a Statement node with SUBJECT_OF and OBJECT_OF edges back to
    the entity endpoints.  The originating RELATES_TO edge is preserved.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    src_name: str
    src_type: str
    tgt_name: str
    tgt_type: str
    relation: str
    confidence: float = 1.0
    source_doc_ids: list[str] = Field(default_factory=list)
    tenant: str = "default"
    reified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CalibrationSample(BaseModel):
    """A single (predicted_confidence, actual_outcome) data point."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    predicted_confidence: float
    actual_outcome: float           # 1.0 = correct, 0.0 = incorrect
    relation: str = ""
    source_doc_id: str = ""
    prompt_version: str = ""
    tenant: str = "default"
    verified_by: str = "system"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GraphSnapshot(BaseModel):
    """Lightweight checkpoint of graph statistics at a point in time."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    tenant: str = "default"
    entity_count: int = 0
    edge_count: int = 0
    negative_count: int = 0
    conflict_count: int = 0
    community_count: int = 0
    orphan_count: int = 0
    avg_confidence: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Query / retrieval models ───────────────────────────────────────────────────

class RetrievalStep(BaseModel):
    """One observable retrieval action and the evidence it produced."""

    step: int = Field(ge=1)
    action: str
    query: str
    surfaces: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    new_evidence_ids: list[str] = Field(default_factory=list)
    graph_edges: list[str] = Field(default_factory=list)
    outcome: str = "completed"
    latency_ms: float = Field(default=0.0, ge=0.0)


class RetrievalTrajectory(BaseModel):
    """Machine-readable route/evidence trace for one answer."""

    query_class: str = "factoid"
    planned_mode: str = "hybrid"
    routing_reason: str = ""
    steps: list[RetrievalStep] = Field(default_factory=list)
    selected_surfaces: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    graph_edges: list[str] = Field(default_factory=list)
    tool_calls: int = Field(default=0, ge=0)
    completed_by: str = "synthesis"


class QueryResult(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    retrieval_mode: str = "hybrid"
    model_version: str = ""
    cache_hit: bool = False
    cache_key: str = ""
    source_query_id: str = ""
    source_trace_id: str = ""
    valid_at: str | None = None
    transaction_at: str | None = None
    correlation_id: str = ""
    routing_reason: str = ""
    policy_result: str = ""
    policy_reason_code: str = ""
    retrieval_sufficiency: dict[str, Any] = Field(default_factory=dict)
    evidence_bundle: dict[str, Any] = Field(default_factory=dict)
    retrieval_trajectory: RetrievalTrajectory | None = None


class SessionTurn(BaseModel):
    """One exchange in a multi-turn conversational session."""
    turn_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer: str
    referenced_entities: list[str] = Field(default_factory=list)
    referenced_chunks: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Evaluation models ──────────────────────────────────────────────────────────

class EvalJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    query_result: QueryResult
    ground_truth: str = ""
    # Kept defaulted for backwards-compatible consumption of already-queued
    # messages. New publishers always supply the query's trusted tenant and
    # correlation ID; neither is inferred from user-provided evaluation data.
    tenant: str = "default"
    correlation_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvalResult(BaseModel):
    job_id: str
    query_id: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    judge_decision: str = "retrieve"
    judge_confidence: float = 0.0
    judge_accept_threshold: float = 0.9
    judge_retrieve_threshold: float = 0.55
    judge_target_fdr: float = 0.05
    retrieval_used: bool = True
    abstention_reason: str = ""
    evaluation_source: str = "ragas"
    rubric_score: float = 0.0
    rubric_passed: bool = False
    rubric_hard_failed: bool = False
    rubric_results: list[dict[str, Any]] = Field(default_factory=list)
    rubric_config: dict[str, Any] = Field(default_factory=dict)
    stage_metrics: list[dict[str, Any]] = Field(default_factory=list)
    failure_category: str = ""
    failure_reason: str = ""
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Business Matrix models ─────────────────────────────────────────────────────

class KPIEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str
    tenant: str
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    cost_usd: float = 0.0
    retrieval_mode: str = "hybrid"
    model_version: str = ""
    judge_decision: str = "retrieve"
    judge_confidence: float = 0.0
    judge_accept_threshold: float = 0.9
    judge_retrieve_threshold: float = 0.55
    judge_target_fdr: float = 0.05
    retrieval_used: bool = True
    abstention_reason: str = ""
    evaluation_source: str = "ragas"
    retrieval_cost_usd: float = 0.0


# ── Message queue payloads ─────────────────────────────────────────────────────

class IngestMessage(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    document: Document
    priority: Literal["normal", "high"] = "normal"


class QueryMessage(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str = Field(min_length=1, max_length=8_000)
    mode: Literal["local", "global", "hybrid"] = "hybrid"
    ground_truth: str = Field(default="", max_length=16_000)
    tenant: str = Field(default="default", min_length=1, max_length=256)
    session_id: str = Field(default="", max_length=256)  # multi-turn context
    valid_at: str | None = Field(default=None, max_length=64)
    transaction_at: str | None = Field(default=None, max_length=64)
    correlation_id: str = Field(default="", max_length=128)
    # Constructed from authenticated claims by the API before queueing.  It is
    # part of the durable message because workers must not reconstruct identity
    # from untrusted request input or ambient process state.
    access_context: AccessContext = Field(default_factory=AccessContext)
