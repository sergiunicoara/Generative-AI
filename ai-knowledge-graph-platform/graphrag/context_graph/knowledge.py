"""Typed contracts for contextual knowledge: what a requirement says, where it
applies, how strongly it binds, and how it can be checked.

These contracts are deliberately separate axes:

* **source authority** (``SourceAuthority``) -- who issued the knowledge;
* **normative modality** (``NormativeModality``) -- how strongly it binds;
* **specificity** -- how narrowly ``Applicability`` scopes it (never a licence
  to override anything on its own);
* **confidence** -- how sure an extractor was (informational only; it never
  promotes text into an executable rule);
* **temporal validity** -- ``valid_from``/``valid_to`` (valid time) and
  ``recorded_at`` (transaction time).

Authoritative representation: a published :class:`KnowledgeItem` /
:class:`KnowledgeException` inside a knowledge release (see
``knowledge_release.py``).  Anything else -- a manifest entry, an RDF
projection, an answer citation -- is a derived view of it.

``ObligationDraft`` (``graphrag.enterprise.models``) stays the extraction-side
contract.  :meth:`KnowledgeItem.from_obligation_draft` turns one into a *draft*
item with no executable constraint and unknown applicability, which a human
must curate before publication.  Nothing here evaluates arbitrary code: a
constraint is a bounded list of predicates over named facts, using the same
``ConditionOperator`` vocabulary as the Context Graph policy engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from graphrag.context_graph.models import ConditionOperator
from graphrag.enterprise.models import DocumentAccessPolicy, ObligationDraft

KNOWLEDGE_SCHEMA_VERSION = "contextual-knowledge/v1"

#: Business-context dimensions.  They are orthogonal: a market is not a child
#: of a customer, and a brand may span several product families.  No implicit
#: hierarchy is assumed anywhere in the evaluator.
DIMENSIONS: tuple[str, ...] = (
    "customer", "brand", "market", "product", "product_family", "workflow",
)

_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,127}$"
_FACT_PATTERN = r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+){0,5}$"
_SCALAR = (str, int, float, bool)


class NormativeModality(StrEnum):
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    PERMITTED = "permitted"
    PROHIBITED = "prohibited"


#: Modalities whose violation makes a decision non-compliant.  A recommendation
#: is surfaced, never blocking; a permission has no compliance outcome.
BLOCKING_MODALITIES = frozenset({NormativeModality.MANDATORY, NormativeModality.PROHIBITED})


class SourceAuthority(StrEnum):
    REGULATORY = "regulatory"
    CUSTOMER = "customer"
    BRAND = "brand"
    INTERNAL = "internal"
    EXTRACTED = "extracted"


class PublicationStatus(StrEnum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    WITHDRAWN = "withdrawn"


class KnowledgeOrigin(StrEnum):
    CURATED = "curated"
    EXTRACTED = "extracted"


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceRef(_Strict):
    """A citation: where a requirement or a fact comes from."""

    source_id: str = Field(min_length=1, max_length=256)
    source_type: str = Field(default="document", min_length=1, max_length=64)
    locator: str = Field(default="", max_length=512)
    quote: str = Field(default="", max_length=2_000)
    recorded_at: datetime | None = None

    @field_validator("recorded_at")
    @classmethod
    def _tz(cls, value):
        return _utc(value)


class DimensionScope(_Strict):
    """How one dimension restricts applicability.

    ``any``      unrestricted: the item applies whatever the context value is,
                 including when the request does not state it.
    ``in``       applies only to the listed values.
    ``not_in``   applies to every value except the listed ones.

    An *absent* scope (``None`` on :class:`Applicability`) means the metadata
    is unknown; that is never treated as unrestricted.
    """

    mode: Literal["any", "in", "not_in"]
    values: tuple[str, ...] = ()

    @field_validator("values", mode="before")
    @classmethod
    def _normalise(cls, values):
        if values is None:
            return ()
        cleaned = sorted({str(v).strip() for v in values if str(v).strip()})
        if any(len(v) > 128 for v in cleaned):
            raise ValueError("dimension values must be at most 128 characters")
        if len(cleaned) > 200:
            raise ValueError("at most 200 values per dimension")
        return tuple(cleaned)

    @model_validator(mode="after")
    def _consistent(self):
        if self.mode == "any" and self.values:
            raise ValueError("an unrestricted ('any') scope cannot list values")
        if self.mode != "any" and not self.values:
            raise ValueError(f"scope mode {self.mode!r} requires at least one value")
        return self

    @classmethod
    def unrestricted(cls) -> "DimensionScope":
        return cls(mode="any")


class Applicability(_Strict):
    """Where and when an item applies.  ``None`` = unknown metadata."""

    customer: DimensionScope | None = None
    brand: DimensionScope | None = None
    market: DimensionScope | None = None
    product: DimensionScope | None = None
    product_family: DimensionScope | None = None
    workflow: DimensionScope | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    @field_validator("valid_from", "valid_to")
    @classmethod
    def _tz(cls, value):
        return _utc(value)

    @model_validator(mode="after")
    def _window(self):
        if self.valid_from and self.valid_to and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be after valid_from")
        return self

    def scope(self, dimension: str) -> DimensionScope | None:
        return getattr(self, dimension)


def _require_finite(value: Any) -> None:
    """NaN and infinities compare inconsistently, so they are never accepted."""
    for element in value if isinstance(value, (list, tuple)) else (value,):
        if isinstance(element, float) and not math.isfinite(element):
            raise ValueError("numeric values must be finite (NaN and infinity are rejected)")


class Predicate(_Strict):
    """One bounded check over a named fact.

    ``when_missing`` declares the semantics of an absent fact: ``unknown``
    (default) keeps the outcome as insufficient evidence; ``violation`` is
    only for requirements whose meaning is that absence itself is a breach.
    """

    fact: str = Field(pattern=_FACT_PATTERN, max_length=128)
    operator: ConditionOperator
    value: Any = None
    when_missing: Literal["unknown", "violation"] = "unknown"

    @model_validator(mode="after")
    def _bounded_value(self):
        _require_finite(self.value)
        op = self.operator
        if op in (ConditionOperator.IS_EMPTY, ConditionOperator.NOT_EMPTY):
            if self.value is not None:
                raise ValueError(f"operator {op.value} takes no value")
            return self
        if op == ConditionOperator.IN:
            if not isinstance(self.value, (list, tuple)) or not self.value or len(self.value) > 100:
                raise ValueError("operator 'in' needs a non-empty list of at most 100 scalars")
            if not all(isinstance(v, _SCALAR) for v in self.value):
                raise ValueError("operator 'in' accepts scalar list items only")
            return self
        if not isinstance(self.value, _SCALAR):
            raise ValueError(f"operator {op.value} needs a scalar value (str, int, float, bool)")
        if op in (ConditionOperator.GT, ConditionOperator.GTE, ConditionOperator.LT, ConditionOperator.LTE):
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError(f"operator {op.value} needs a numeric value")
        if isinstance(self.value, str) and len(self.value) > 512:
            raise ValueError("string values must be at most 512 characters")
        return self


class Constraint(_Strict):
    """A conjunction of predicates.  Disjunction and expressions are out of
    scope on purpose; a requirement needing them is left unexecutable and goes
    to human review instead of being approximated."""

    all_of: tuple[Predicate, ...] = Field(min_length=1, max_length=20)


class ConstraintProvenance(_Strict):
    """Who turned the statement into an executable constraint, and on what
    approval.  Required before any constraint is served."""

    authored_by: str = Field(min_length=1, max_length=256)
    approval_ref: str = Field(min_length=1, max_length=256)


class KnowledgeItem(_Strict):
    """One versioned, tenant-owned unit of contextual knowledge."""

    id: str = Field(pattern=_ID_PATTERN)
    version: int = Field(ge=1)
    tenant: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=256)
    statement: str = Field(min_length=1, max_length=4_000)
    subject_key: str = Field(pattern=r"^[a-z][a-z0-9_.]{0,127}$")
    modality: NormativeModality
    authority: SourceAuthority
    status: PublicationStatus = PublicationStatus.DRAFT
    origin: KnowledgeOrigin = KnowledgeOrigin.CURATED
    applicability: Applicability = Field(default_factory=Applicability)
    constraint: Constraint | None = None
    constraint_provenance: ConstraintProvenance | None = None
    sources: tuple[EvidenceRef, ...] = ()
    access: DocumentAccessPolicy = Field(default_factory=DocumentAccessPolicy)
    recorded_at: datetime
    supersedes: tuple[str, ...] = ()
    #: Revision lineage.  Items in the same series are successive revisions of
    #: one requirement; two of them in force at once without a supersession
    #: edge ordering them is an ambiguity, reported as a conflict.
    series: str = Field(default="", pattern=r"^$|^[a-z][a-z0-9_.\-]{0,127}$")
    #: Declared intent to take precedence over other items.  It is recorded
    #: and reported, but authorises nothing: only a published
    #: :class:`KnowledgeException` can waive or order requirements.
    overrides: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reviewed_by: str = Field(default="", max_length=256)
    review_ref: str = Field(default="", max_length=256)

    @field_validator("recorded_at")
    @classmethod
    def _tz(cls, value):
        return _utc(value)

    @model_validator(mode="after")
    def _governance(self):
        served = self.status in (PublicationStatus.APPROVED, PublicationStatus.PUBLISHED)
        if self.constraint is not None and served and self.constraint_provenance is None:
            raise ValueError(
                f"{self.id}: an approved or published constraint needs constraint_provenance "
                "(a human author and an approval reference)"
            )
        if (
            self.origin == KnowledgeOrigin.EXTRACTED and self.constraint is not None
            and self.constraint_provenance is None
        ):
            raise ValueError(
                f"{self.id}: extracted text cannot carry an executable constraint without "
                "human-authored constraint_provenance, whatever its confidence"
            )
        if self.status == PublicationStatus.PUBLISHED:
            if not self.sources:
                raise ValueError(f"{self.id}: a published item must cite at least one source")
            if not self.reviewed_by or not self.review_ref:
                raise ValueError(f"{self.id}: a published item needs reviewed_by and review_ref")
        if self.id in self.supersedes or self.id in self.overrides:
            raise ValueError(f"{self.id}: an item cannot supersede or override itself")
        return self

    @property
    def key(self) -> str:
        return f"{self.id}@v{self.version}"

    @classmethod
    def from_obligation_draft(
        cls,
        draft: ObligationDraft,
        *,
        item_id: str,
        tenant: str,
        subject_key: str,
        title: str,
        recorded_at: datetime,
        access: DocumentAccessPolicy | None = None,
    ) -> "KnowledgeItem":
        """Adapt an extracted obligation into a reviewable draft.

        The result never has an executable constraint and declares no
        applicability (every dimension unknown), regardless of the draft's
        confidence.  Valid time is carried over; everything else needs a
        curator.
        """
        return cls(
            id=item_id, version=1, tenant=tenant, title=title,
            statement=draft.obligation, subject_key=subject_key,
            modality=NormativeModality.MANDATORY, authority=SourceAuthority.EXTRACTED,
            status=PublicationStatus.DRAFT, origin=KnowledgeOrigin.EXTRACTED,
            applicability=Applicability(valid_from=draft.effective_from, valid_to=draft.effective_to),
            sources=(EvidenceRef(
                source_id=draft.evidence_chunk_id, source_type="chunk", quote=draft.evidence_quote,
            ),),
            access=access or DocumentAccessPolicy(), recorded_at=recorded_at,
            confidence=draft.confidence,
        )


class KnowledgeException(_Strict):
    """An explicitly authorised resolution between requirements.

    ``waiver``      the targets do not bind within this exception's scope.
    ``precedence``  ``winner_item_id`` prevails over the other targets within
                    this scope (an approved conflict resolution).

    Both need a grant, an approval reference and a citation; both are scoped
    and time-bounded exactly like knowledge items.  Greater specificity of an
    item never substitutes for one of these.
    """

    id: str = Field(pattern=_ID_PATTERN)
    version: int = Field(ge=1)
    tenant: str = Field(min_length=1, max_length=128)
    kind: Literal["waiver", "precedence"]
    target_item_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    winner_item_id: str = ""
    rationale: str = Field(min_length=1, max_length=2_000)
    applicability: Applicability
    granted_by: str = Field(default="", max_length=256)
    approval_ref: str = Field(default="", max_length=256)
    sources: tuple[EvidenceRef, ...] = ()
    status: PublicationStatus = PublicationStatus.DRAFT
    access: DocumentAccessPolicy = Field(default_factory=DocumentAccessPolicy)
    recorded_at: datetime

    @field_validator("recorded_at")
    @classmethod
    def _tz(cls, value):
        return _utc(value)

    @model_validator(mode="after")
    def _authorised(self):
        if self.kind == "precedence":
            if not self.winner_item_id or self.winner_item_id in self.target_item_ids:
                raise ValueError(f"{self.id}: precedence needs a winner that is not among the targets")
        elif self.winner_item_id:
            raise ValueError(f"{self.id}: a waiver has no winner")
        if self.status == PublicationStatus.PUBLISHED:
            if not (self.granted_by and self.approval_ref and self.sources):
                raise ValueError(
                    f"{self.id}: a published exception needs granted_by, approval_ref and a source"
                )
        return self

    @property
    def key(self) -> str:
        return f"{self.id}@v{self.version}"


class BusinessContext(_Strict):
    """The situation being evaluated.  ``None`` on a dimension means the caller
    does not know it; it never widens what applies."""

    customer: tuple[str, ...] | None = None
    brand: tuple[str, ...] | None = None
    market: tuple[str, ...] | None = None
    product: tuple[str, ...] | None = None
    product_family: tuple[str, ...] | None = None
    workflow: tuple[str, ...] | None = None
    as_of: datetime | None = None
    known_as: datetime | None = None

    @field_validator(*DIMENSIONS, mode="before")
    @classmethod
    def _values(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        cleaned = sorted({str(v).strip() for v in value if str(v).strip()})
        if not cleaned:
            # An empty list carries no information; treat it as unknown rather
            # than as "matches nothing" or "matches everything".
            return None
        if len(cleaned) > 50 or any(len(v) > 128 for v in cleaned):
            raise ValueError("at most 50 values of at most 128 characters per dimension")
        return tuple(cleaned)

    @field_validator("as_of", "known_as")
    @classmethod
    def _tz(cls, value):
        return _utc(value)

    def values(self, dimension: str) -> tuple[str, ...] | None:
        return getattr(self, dimension)

    def fingerprint(self) -> str:
        return _digest(self.model_dump(mode="json"))


class Fact(_Strict):
    """An observed value about the subject under evaluation, with evidence."""

    value: Any = None
    evidence: tuple[EvidenceRef, ...] = ()

    @field_validator("value")
    @classmethod
    def _bounded(cls, value):
        _require_finite(value)
        if value is None or isinstance(value, _SCALAR):
            if isinstance(value, str) and len(value) > 2_000:
                raise ValueError("fact strings must be at most 2000 characters")
            return value
        if isinstance(value, (list, tuple)) and len(value) <= 200 and all(isinstance(v, _SCALAR) for v in value):
            return list(value)
        raise ValueError("a fact value must be a scalar or a list of at most 200 scalars")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def content_digest(value: Any) -> str:
    """Stable sha256 over canonical JSON (same pattern as manifests/receipts)."""
    return _digest(value)


__all__ = [
    "Applicability", "BLOCKING_MODALITIES", "BusinessContext", "Constraint", "ConstraintProvenance",
    "DIMENSIONS", "DimensionScope", "EvidenceRef", "Fact", "KNOWLEDGE_SCHEMA_VERSION", "KnowledgeException",
    "KnowledgeItem", "KnowledgeOrigin", "NormativeModality", "Predicate", "PublicationStatus",
    "SourceAuthority", "content_digest",
]
