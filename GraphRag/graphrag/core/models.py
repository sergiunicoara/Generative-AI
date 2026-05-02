"""Shared Pydantic dataclasses used across the entire pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── Ingestion models ───────────────────────────────────────────────────────────

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    source_path: str
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending | processing | done | failed


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
    type: str  # PERSON | ORG | PRODUCT | CONCEPT | LOCATION | EVENT
    description: str = ""
    embedding: list[float] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity_id: str
    target_entity_id: str
    relation: str           # e.g. "ACQUIRED", "BUILT", "WORKS_AT"
    weight: float = 1.0
    confidence: float = 1.0  # LLM extraction confidence [0, 1]
    source_chunk_id: str = ""


# ── Graph models ───────────────────────────────────────────────────────────────

class Community(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    level: int              # 0 = fine-grained, 1 = medium, 2 = coarse
    member_entity_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    embedding: list[float] = Field(default_factory=list)
    member_count: int = 0


# ── Query / retrieval models ───────────────────────────────────────────────────

class QueryResult(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)   # retrieved text chunks
    citations: list[str] = Field(default_factory=list)  # source doc/chunk IDs
    latency_ms: float = 0.0
    retrieval_mode: str = "hybrid"  # local | global | hybrid
    model_version: str = ""


# ── Evaluation models ──────────────────────────────────────────────────────────

class EvalJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    query_result: QueryResult
    ground_truth: str = ""   # expected answer for context_recall scoring
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
    priority: str = "normal"   # normal | high


class QueryMessage(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    mode: str = "hybrid"       # local | global | hybrid
    ground_truth: str = ""     # optional; used for eval
    tenant: str = "default"
