"""Registry of graph-publication rules.

Every rule has a stable ``rule_id`` that appears in reports, quarantine records
and metrics. Where an ingestion SHACL shape expresses the same constraint,
``shacl_ref`` names it (``ontology/shapes/ingestion.shapes.ttl``), so a rule
id maps back to the shape a reviewer would look at.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    BLOCKING = "BLOCKING"            # record is quarantined, never published
    WARNING = "WARNING"              # published, reported
    INFORMATIONAL = "INFORMATIONAL"  # published, reported


class PublicationState(str, Enum):
    """Lifecycle of one ingestion batch (kept on the run manifest)."""
    EXTRACTED = "EXTRACTED"
    STAGED = "STAGED"
    VALIDATED = "VALIDATED"
    PUBLISHED = "PUBLISHED"
    REJECTED = "REJECTED"  # a document-level BLOCKING rule failed; nothing written


@dataclass(frozen=True)
class Rule:
    rule_id: str
    severity: Severity
    target: str  # document | entity | relation
    description: str
    shacl_ref: str | None = None


_ENT = "ing:MappedEntityShape"
_REL = "ing:MappedRelationShape"

_ALL = [
    # ── document / batch ───────────────────────────────────────────────────
    Rule("DOC-TENANT-001", Severity.BLOCKING, "document", "document tenant is empty"),
    Rule("DOC-TENANT-002", Severity.BLOCKING, "document",
         "a chunk's tenant differs from its document's tenant"),
    Rule("DOC-PROV-001", Severity.BLOCKING, "document",
         "document has no source identifier (filename, source_path or content_hash)"),
    Rule("DOC-TEMPORAL-001", Severity.BLOCKING, "document", "document valid_from is after valid_to"),
    Rule("DOC-SUPERSEDES-001", Severity.WARNING, "document",
         "SUPERSEDES references a document that does not exist in this tenant"),
    # ── entities ───────────────────────────────────────────────────────────
    Rule("ENT-REQ-001", Severity.BLOCKING, "entity", "entity name is empty", f"{_ENT}/rdfs:label"),
    Rule("ENT-REQ-002", Severity.BLOCKING, "entity", "entity type is empty", f"{_ENT}/ing:entityType"),
    Rule("ENT-TENANT-001", Severity.BLOCKING, "entity",
         "entity tenant differs from the batch tenant", f"{_ENT}/ing:tenant"),
    Rule("ENT-CONF-001", Severity.BLOCKING, "entity", "entity confidence is not a number in [0, 1]"),
    Rule("ENT-ID-001", Severity.BLOCKING, "entity",
         "the same entity id is used for different (name, type) identities in one batch"),
    Rule("ENT-ER-001", Severity.BLOCKING, "entity",
         "malformed entity-resolution output (unknown status or half-set canonical identity)"),
    Rule("ENT-TYPE-001", Severity.WARNING, "entity", "entity type is not in the tenant ontology"),
    Rule("ENT-PROP-001", Severity.WARNING, "entity", "required semantic property is missing"),
    Rule("ENT-PROP-002", Severity.WARNING, "entity", "semantic property value is not allowed"),
    Rule("ENT-ORPHAN-001", Severity.INFORMATIONAL, "entity",
         "entity takes part in no relation in this batch"),
    # ── relations ──────────────────────────────────────────────────────────
    Rule("REL-REQ-001", Severity.BLOCKING, "relation", "relation type is empty", f"{_REL}/ing:relation"),
    Rule("REL-REQ-002", Severity.WARNING, "relation",
         "relation type is not UPPER_SNAKE_CASE (the writer will normalise it)", f"{_REL}/ing:relation"),
    Rule("REL-REF-001", Severity.BLOCKING, "relation",
         "relation endpoint is not an entity of this batch (dangling reference)",
         f"{_REL}/ing:source|ing:target"),
    Rule("REL-REF-002", Severity.BLOCKING, "relation", "relation endpoint was itself rejected"),
    Rule("REL-SELF-001", Severity.BLOCKING, "relation", "relation is a self-loop"),
    Rule("REL-CONF-001", Severity.BLOCKING, "relation",
         "relation confidence is not in [0, 1] or weight is not finite", f"{_REL}/ing:confidence"),
    Rule("REL-TEMPORAL-001", Severity.BLOCKING, "relation", "relation valid_from is after valid_to"),
    Rule("REL-STATE-001", Severity.BLOCKING, "relation", "unknown confidence_state"),
    Rule("REL-DOMAIN-001", Severity.WARNING, "relation",
         "extracted endpoint types are outside the relation's domain/range"),
    Rule("REL-DOMAIN-002", Severity.BLOCKING, "relation",
         "resolved endpoint types are outside the relation's domain/range"),
    Rule("SEM-VIOLATION-001", Severity.BLOCKING, "entity",
         "semantic-model mutation validator rejected the record (cardinality, datatype, ...)"),
]

RULES: dict[str, Rule] = {r.rule_id: r for r in _ALL}


def rule(rule_id: str) -> Rule:
    return RULES[rule_id]
