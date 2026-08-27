"""Typed contracts shared by access control, sync, metadata and lineage flows.

The platform deliberately keeps these contracts provider-neutral.  A connector
such as SharePoint adapts its change and permission payloads into them; retrieval
and graph code never need to trust provider-specific JSON at query time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def normalise_document_url(value: str) -> str:
    """Canonicalise an http(s) document identity while leaving local paths alone."""
    parsed = urlsplit(value.strip())
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, parsed.query, ""))
    return value.strip()


class ACLState(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class DocumentAccessPolicy(BaseModel):
    """Document ACL material normalised before it reaches the graph.

    Principal values are namespaced (``user:<id>`` or ``group:<id>``), avoiding
    accidental overlap between an identifier in two identity domains.  A
    restricted document with unknown ACL state is intentionally not readable
    while enforcement is enabled.
    """

    mode: Literal["tenant", "restricted"] = "tenant"
    state: ACLState = ACLState.NOT_APPLICABLE
    allow_principals: list[str] = Field(default_factory=list, max_length=5_000)
    deny_principals: list[str] = Field(default_factory=list, max_length=5_000)
    requires_group_resolution: bool = False

    @field_validator("allow_principals", "deny_principals")
    @classmethod
    def _normalise_principals(cls, principals: list[str]) -> list[str]:
        normalised = sorted({p.strip() for p in principals if p and p.strip()})
        if any(len(p) > 512 for p in normalised):
            raise ValueError("principal identifiers must be at most 512 characters")
        return normalised

    @model_validator(mode="after")
    def _validate_restricted_policy(self):
        if self.mode == "restricted" and self.state != ACLState.KNOWN:
            # Ingestion may store an upstream ACL failure, but cannot label it
            # restricted-and-usable. Retrieval treats it as denied.
            return self
        if self.mode == "restricted" and not self.allow_principals:
            raise ValueError("restricted access requires at least one allowed principal")
        return self


class AccessContext(BaseModel):
    """Trusted query-time identity resolution, serialisable across RabbitMQ."""

    subject_id: str = ""
    principals: list[str] = Field(default_factory=list, max_length=5_000)
    groups_resolved: bool = False
    resolution_source: str = "token_claims"

    @field_validator("principals")
    @classmethod
    def _normalise_principals(cls, principals: list[str]) -> list[str]:
        return sorted({p.strip() for p in principals if p and p.strip()})

    @property
    def fingerprint(self) -> str:
        import hashlib
        material = "\0".join([self.subject_id, str(self.groups_resolved), *self.principals])
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @classmethod
    def from_claims(cls, claims: dict[str, Any]) -> "AccessContext":
        subject = str(claims.get("sub") or "").strip()
        raw_groups = claims.get("groups")
        # A missing groups claim is materially different from an empty, trusted
        # group resolution.  The former must not unlock group-protected docs.
        groups_resolved = isinstance(raw_groups, list)
        groups = raw_groups if groups_resolved else []
        principals = ([f"user:{subject}"] if subject else [])
        principals.extend(f"group:{str(group).strip()}" for group in groups if str(group).strip())
        return cls(
            subject_id=subject,
            principals=principals,
            groups_resolved=groups_resolved,
            resolution_source="token_claims" if groups_resolved else "token_without_groups",
        )


class MetadataEnvelope(BaseModel):
    """Universal metadata envelope plus bounded collection and discovery tiers."""

    collection: str = Field(default="default", min_length=1, max_length=128)
    schema_version: str = Field(default="v1", min_length=1, max_length=64)
    source_system: str = Field(default="manual", min_length=1, max_length=128)
    external_id: str = Field(default="", max_length=512)
    source_url: str = Field(default="", max_length=4_096)
    source_version: str = Field(default="", max_length=256)
    content_type: str = Field(default="text/plain", max_length=256)
    classification: str = Field(default="", max_length=128)
    # Effective time belongs to the universal envelope; transaction time is
    # supplied by Neo4j's recorded_at when the document revision is written.
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    collection_metadata: dict[str, Any] = Field(default_factory=dict, max_length=200)
    discovery_metadata: dict[str, str] = Field(default_factory=dict, max_length=100)

    @field_validator("collection_metadata")
    @classmethod
    def _bounded_collection_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        for key, item in value.items():
            if len(str(key)) > 128 or len(str(item)) > 4_096:
                raise ValueError("collection metadata keys/values exceed bounds")
        return value

    @field_validator("discovery_metadata")
    @classmethod
    def _bounded_discovery_values(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if len(str(key)) > 128 or len(str(item)) > 1_024:
                raise ValueError("discovery metadata keys/values exceed bounds")
        return {str(k): str(v) for k, v in value.items()}

    @model_validator(mode="after")
    def _validate_effective_interval(self):
        if self.effective_from and self.effective_to and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be later than effective_from")
        return self


class DocumentLink(BaseModel):
    """An explicit, source-observed reference from one document to another.

    This is intentionally a transport contract rather than an inferred graph
    edge.  The source document supplies the provenance and ACL snapshot when it
    is persisted; a target document is connected only after its URL resolves in
    the same tenant.
    """

    target_url: str = Field(min_length=1, max_length=4_096)
    anchor_text: str = Field(default="", max_length=1_024)
    source_locator: str = Field(default="", max_length=512)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_system: str = Field(default="manual", min_length=1, max_length=128)
    source_version: str = Field(default="", max_length=256)

    @field_validator("target_url")
    @classmethod
    def _normalise_target_url(cls, value: str) -> str:
        parsed = urlsplit(normalise_document_url(value))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("document link target_url must be an absolute http(s) URL")
        return normalise_document_url(value)


class CollectionSchema(BaseModel):
    """Versioned, collection-specific governed metadata contract."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    collection: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    required_fields: list[str] = Field(default_factory=list, max_length=100)
    allowed_fields: list[str] = Field(default_factory=list, max_length=200)
    status: Literal["draft", "active", "retired"] = "draft"
    tenant: str = "default"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("required_fields", "allowed_fields")
    @classmethod
    def _normalise_field_names(cls, fields: list[str]) -> list[str]:
        result = sorted({field.strip() for field in fields if field and field.strip()})
        if any(len(field) > 128 for field in result):
            raise ValueError("metadata field names must be at most 128 characters")
        return result


class LineageRelation(str, Enum):
    SUPERSEDES = "SUPERSEDES"
    AMENDS = "AMENDS"


class LineageAssertion(BaseModel):
    """An explicit text-backed document lineage claim awaiting review if needed."""

    relation: LineageRelation
    target_document_id: str = Field(min_length=1, max_length=256)
    evidence_chunk_id: str = Field(min_length=1, max_length=256)
    evidence_quote: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0.0, le=1.0)
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    requires_human_review: bool = True


class ObligationDraft(BaseModel):
    """Source-backed obligation, never a free-standing generated assertion."""

    obligation: str = Field(min_length=1, max_length=4_000)
    subject: str = Field(default="", max_length=512)
    beneficiary: str = Field(default="", max_length=512)
    due_at: datetime | None = None
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    evidence_chunk_id: str = Field(min_length=1, max_length=256)
    evidence_quote: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool = True


class SyncChangeType(str, Enum):
    UPSERT = "upsert"
    DELETE = "delete"


class SyncChange(BaseModel):
    """Provider-neutral external change record, usable by webhook or polling."""

    change_type: SyncChangeType
    external_id: str = Field(min_length=1, max_length=512)
    filename: str = Field(default="", max_length=255)
    text: str = Field(default="", max_length=8_000_000)
    cursor: str = Field(default="", max_length=2_048)
    metadata: MetadataEnvelope = Field(default_factory=MetadataEnvelope)
    access_policy: DocumentAccessPolicy = Field(default_factory=DocumentAccessPolicy)
    document_links: list[DocumentLink] = Field(default_factory=list, max_length=10_000)
    lineage: list[LineageAssertion] = Field(default_factory=list, max_length=100)
    obligations: list[ObligationDraft] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def _require_content_for_upsert(self):
        if self.change_type == SyncChangeType.UPSERT and (not self.filename or not self.text):
            raise ValueError("upsert changes require filename and text")
        return self
