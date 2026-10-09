"""Pure, pre-write validation of an extracted batch.

No I/O: given the in-memory entities/relations of one chunk (or one relational
batch), return what may be published, what must be quarantined and why. Graph
context checks that need Neo4j live in ``graph_checks.py`` and are read-only.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from graphrag.core.models import Chunk, Document, Entity, Relation
from graphrag.graph.validation.report import ValidationReport, Violation
from graphrag.graph.validation.rules import Severity

VALID_RESOLUTION_STATUSES = frozenset({"", "auto_resolved", "needs_review", "created_new"})
VALID_CONFIDENCE_STATES = frozenset({"ASSERTED", "INFERRED", "DISPUTED", "RETRACTED", "APPROVED"})
_RELATION_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,49}$")
# Entity.tenant defaults to "default" when an extractor did not set it; the
# writer stamps the chunk tenant, so "default" is treated as unset here.
_UNSET_TENANTS = frozenset({"", "default"})

TripletValidator = Callable[[str, str, str], tuple[bool, str]]
PropertyChecker = Callable[[str, str, dict], list[dict]]


@dataclass
class RejectedRecord:
    record_kind: str
    record_key: str
    rule_ids: list[str]
    messages: list[str]
    payload: dict
    chunk_id: str = ""


@dataclass
class BatchResult:
    entities: list[Entity]
    relations: list[Relation]
    rejected: list[RejectedRecord] = field(default_factory=list)
    report: ValidationReport | None = None


def entity_key(e: Entity) -> str:
    return f"{e.type}:{e.name}"


def relation_key(r: Relation, by_id: dict[str, Entity]) -> str:
    src = by_id.get(r.source_entity_id)
    tgt = by_id.get(r.target_entity_id)
    s = src.name if src else f"?{r.source_entity_id}"
    t = tgt.name if tgt else f"?{r.target_entity_id}"
    return f"{s}-{r.relation}->{t}"


def _bad_unit(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(v) or v < 0.0 or v > 1.0


def _bad_interval(start, end) -> bool:
    return start is not None and end is not None and start > end


def entity_payload(e: Entity) -> dict:
    return e.model_dump(mode="json", exclude={"embedding"})


def relation_payload(r: Relation, by_id: dict[str, Entity]) -> dict:
    payload = r.model_dump(mode="json")
    for side, eid in (("source_endpoint", r.source_entity_id), ("target_endpoint", r.target_entity_id)):
        ent = by_id.get(eid)
        payload[side] = {"name": ent.name, "type": ent.type} if ent else None
    return payload


def validate_document(doc: Document, chunks: Iterable[Chunk], *, source: str) -> ValidationReport:
    report = ValidationReport(tenant=doc.tenant or "", source=source, records_checked=1)
    key = doc.filename or doc.id

    def add(rule_id: str, msg: str = "") -> None:
        report.violations.append(Violation(rule_id, "document", key, msg))

    if not (doc.tenant or "").strip():
        add("DOC-TENANT-001")
    if not (doc.filename or doc.source_path or doc.content_hash):
        add("DOC-PROV-001")
    if _bad_interval(doc.valid_from, doc.valid_to):
        add("DOC-TEMPORAL-001", f"{doc.valid_from} > {doc.valid_to}")
    for chunk in chunks:
        if chunk.tenant != doc.tenant:
            add("DOC-TENANT-002", f"chunk {chunk.chunk_index} tenant={chunk.tenant!r}")
    return report


def _entity_violations(
    e: Entity,
    *,
    tenant: str,
    allowed_entity_types: set[str] | None,
    property_checker: PropertyChecker | None,
    semantic_validator,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not (e.name or "").strip():
        out.append(("ENT-REQ-001", ""))
    if not (e.type or "").strip():
        out.append(("ENT-REQ-002", ""))
    if e.tenant not in _UNSET_TENANTS and e.tenant != tenant:
        out.append(("ENT-TENANT-001", f"entity tenant={e.tenant!r}, batch tenant={tenant!r}"))
    if _bad_unit(e.confidence):
        out.append(("ENT-CONF-001", f"confidence={e.confidence!r}"))
    if e.resolution_status not in VALID_RESOLUTION_STATUSES or (
        bool(e.canonical_name) != bool(e.canonical_type)
    ):
        out.append(("ENT-ER-001", f"status={e.resolution_status!r}"))
    if allowed_entity_types and e.type and e.type not in allowed_entity_types:
        out.append(("ENT-TYPE-001", f"type={e.type!r}"))
    if property_checker is not None and e.name and e.type:
        for issue in property_checker(e.name, e.type, dict(e.semantic_properties or {})):
            rid = "ENT-PROP-001" if issue.get("type") == "missing_required" else "ENT-PROP-002"
            out.append((rid, str(issue.get("constraint") or issue.get("property") or "")))
    if semantic_validator is not None and e.type:
        for v in semantic_validator.validate_node(e.type, dict(e.semantic_properties or {}), tenant=tenant):
            out.append(("SEM-VIOLATION-001", f"{v.code}: {v.message}"))
    return out


def validate_batch(
    entities: list[Entity],
    relations: list[Relation],
    *,
    tenant: str,
    source: str,
    chunk_id: str = "",
    allowed_entity_types: set[str] | None = None,
    triplet_validator: TripletValidator | None = None,
    property_checker: PropertyChecker | None = None,
    semantic_validator=None,
) -> BatchResult:
    report = ValidationReport(tenant=tenant, source=source,
                              records_checked=len(entities) + len(relations))
    rejected: list[RejectedRecord] = []
    by_id: dict[str, Entity] = {}
    identity_by_id: dict[str, tuple[str, str]] = {}
    conflicting_ids: set[str] = set()
    for e in entities:
        ident = (e.name, e.type)
        if e.id in identity_by_id and identity_by_id[e.id] != ident:
            conflicting_ids.add(e.id)
        identity_by_id.setdefault(e.id, ident)
        by_id.setdefault(e.id, e)

    def record(kind: str, key: str, found: list[tuple[str, str]], payload: dict) -> bool:
        """Report every finding; return True when the record must be quarantined."""
        for rid, msg in found:
            report.violations.append(Violation(rid, kind, key, msg))
        blocking = [(rid, msg) for rid, msg in found if Violation(rid, kind, key).severity is Severity.BLOCKING]
        if blocking:
            rejected.append(RejectedRecord(
                record_kind=kind, record_key=key,
                rule_ids=sorted({rid for rid, _ in blocking}),
                messages=[m for _, m in blocking if m],
                payload=payload, chunk_id=chunk_id,
            ))
        return bool(blocking)

    kept_entities: list[Entity] = []
    rejected_ids: set[str] = set()
    for e in entities:
        found = _entity_violations(
            e, tenant=tenant, allowed_entity_types=allowed_entity_types,
            property_checker=property_checker, semantic_validator=semantic_validator,
        )
        if e.id in conflicting_ids:
            found.append(("ENT-ID-001", f"id={e.id}"))
        if record("entity", entity_key(e), found, entity_payload(e)):
            rejected_ids.add(e.id)
        else:
            kept_entities.append(e)

    kept_relations: list[Relation] = []
    used_ids: set[str] = set()
    for r in relations:
        found: list[tuple[str, str]] = []
        name = (r.relation or "").strip()
        if not name:
            found.append(("REL-REQ-001", ""))
        elif not _RELATION_RE.match(name):
            found.append(("REL-REQ-002", f"relation={name!r}"))
        for eid in (r.source_entity_id, r.target_entity_id):
            if eid not in by_id:
                found.append(("REL-REF-001", f"endpoint id={eid}"))
            elif eid in rejected_ids:
                found.append(("REL-REF-002", f"endpoint {entity_key(by_id[eid])} was rejected"))
        src, tgt = by_id.get(r.source_entity_id), by_id.get(r.target_entity_id)
        if r.source_entity_id == r.target_entity_id or (
            src and tgt and (src.name, src.type) == (tgt.name, tgt.type)
        ):
            found.append(("REL-SELF-001", ""))
        if _bad_unit(r.confidence) or not math.isfinite(float(r.weight)):
            found.append(("REL-CONF-001", f"confidence={r.confidence!r} weight={r.weight!r}"))
        if _bad_interval(r.valid_from, r.valid_to):
            found.append(("REL-TEMPORAL-001", f"{r.valid_from} > {r.valid_to}"))
        if r.confidence_state not in VALID_CONFIDENCE_STATES:
            found.append(("REL-STATE-001", f"state={r.confidence_state!r}"))
        if triplet_validator is not None and src and tgt and name:
            ok, _ = triplet_validator(src.type, name, tgt.type)
            if not ok:
                found.append(("REL-DOMAIN-001", f"{src.type}-{name}->{tgt.type}"))
        if semantic_validator is not None and src and tgt and name:
            for v in semantic_validator.validate_relation(name, src.type, tgt.type, tenant=tenant):
                found.append(("SEM-VIOLATION-001", f"{v.code}: {v.message}"))
        if not record("relation", relation_key(r, by_id), found, relation_payload(r, by_id)):
            kept_relations.append(r)
            used_ids.update((r.source_entity_id, r.target_entity_id))

    for e in kept_entities:
        if e.id not in used_ids:
            report.violations.append(Violation("ENT-ORPHAN-001", "entity", entity_key(e)))

    return BatchResult(kept_entities, kept_relations, rejected, report)
