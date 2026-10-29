"""RDF projection of published knowledge, validated with SHACL generated from
the canonical semantic model.

``ontology/models/contextual-knowledge.yaml`` is the canonical model of a
published knowledge release.  The semantic-model compiler turns it into OWL,
SHACL and Neo4j artefacts exactly as it does for Energy and automotive
(``make semantic-model-check`` drift-checks them).  This module

* projects the published items and exceptions of a release into an RDF graph
  that uses only terms declared in that model, and
* validates the graph against the shapes compiled from the model in memory,
  so a model change takes effect without a separate generation step.

The release gate (``knowledge_release.check_pack``) runs this before the
regression cases: a candidate whose projection does not conform is refused.

The Pydantic contracts remain the stricter check (vocabularies, conditional
publication rules, the bounded-predicate language); SHACL here checks that
the projected release has the structure the canonical model declares.  A
reference to an entry outside the release (an item superseding a withdrawn
one) is not projected as an edge, because the shapes require every edge to
end at a released item; it is returned in ``omitted_references`` instead.
The gate's referential checks cover those references.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from urllib.parse import quote

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from graphrag.context_graph.knowledge import (
    DIMENSIONS, Applicability, EvidenceRef, KnowledgeException, KnowledgeItem,
)
from graphrag.semantic_model import SemanticModel, compile_model, load_model

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_MODEL_PATH = ROOT / "ontology" / "models" / "contextual-knowledge.yaml"


def knowledge_model(path: str | Path | None = None) -> SemanticModel:
    return _load(str(path or KNOWLEDGE_MODEL_PATH))


@lru_cache(maxsize=4)
def _load(path: str) -> SemanticModel:
    return load_model(path)


@lru_cache(maxsize=4)
def _shapes(path: str) -> str:
    return compile_model(_load(path)).shacl


@dataclass(frozen=True)
class KnowledgeProjection:
    graph: Graph
    omitted_references: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShapeReport:
    conforms: bool
    violations: tuple[str, ...] = ()
    omitted_references: tuple[str, ...] = ()


class _Builder:
    def __init__(self, model: SemanticModel) -> None:
        self.ns = Namespace(model.namespace)
        self.graph = Graph()
        self.graph.bind("ck", self.ns)
        self.omitted: list[str] = []

    def iri(self, tenant: str, kind: str, entry_id: str, *parts: str) -> URIRef:
        path = "/".join(quote(p, safe="") for p in (tenant, kind, entry_id, *parts))
        return URIRef(f"{self.ns}entry/{path}")

    def add(self, subject, name: str, value, datatype=None) -> None:
        if value is None or value == "":
            return
        self.graph.add((subject, self.ns[name], Literal(value, datatype=datatype)))

    def entry(self, subject: URIRef, entry: KnowledgeItem | KnowledgeException) -> None:
        self.add(subject, "entryId", entry.id)
        self.add(subject, "version", entry.version, XSD.integer)
        self.add(subject, "tenant", entry.tenant)
        self.add(subject, "status", entry.status.value)
        self.add(subject, "recordedAt", entry.recorded_at, XSD.dateTime)
        self.add(subject, "accessMode", entry.access.mode)
        self.add(subject, "accessState", entry.access.state.value)
        self.add(subject, "requiresGroupResolution", entry.access.requires_group_resolution, XSD.boolean)
        for principal in entry.access.allow_principals:
            self.add(subject, "allowPrincipal", principal)
        for principal in entry.access.deny_principals:
            self.add(subject, "denyPrincipal", principal)
        self.applicability(subject, entry.applicability)
        for position, source in enumerate(entry.sources):
            self.citation(URIRef(f"{subject}/source/{position}"), subject, source)

    def applicability(self, subject: URIRef, applicability: Applicability) -> None:
        self.add(subject, "validFrom", applicability.valid_from, XSD.dateTime)
        self.add(subject, "validTo", applicability.valid_to, XSD.dateTime)
        for dimension in DIMENSIONS:
            scope = applicability.scope(dimension)
            if scope is None:
                continue  # unknown metadata: nothing is asserted
            node = URIRef(f"{subject}/scope/{dimension}")
            self.graph.add((subject, self.ns.hasScope, node))
            self.graph.add((node, RDF.type, self.ns.ApplicabilityScope))
            self.add(node, "dimension", dimension)
            self.add(node, "scopeMode", scope.mode)
            for value in scope.values:
                self.add(node, "scopeValue", value)

    def citation(self, node: URIRef, subject: URIRef, source: EvidenceRef) -> None:
        self.graph.add((subject, self.ns.citesSource, node))
        self.graph.add((node, RDF.type, self.ns.SourceCitation))
        self.add(node, "sourceId", source.source_id)
        self.add(node, "sourceType", source.source_type)
        self.add(node, "locator", source.locator)
        self.add(node, "quote", source.quote)
        self.add(node, "citedRecordedAt", source.recorded_at, XSD.dateTime)

    def reference(self, subject: URIRef, relation: str, target: str, released: dict[str, URIRef],
                  owner_key: str) -> None:
        if target in released:
            self.graph.add((subject, self.ns[relation], released[target]))
        else:
            self.omitted.append(f"{owner_key} {relation} {target}")


def project_release(
    items: Iterable[KnowledgeItem], exceptions: Iterable[KnowledgeException] = (), *,
    model: SemanticModel | None = None,
) -> KnowledgeProjection:
    """Project a release's published entries (callers pass only published
    ones, as the release does)."""
    model = model or knowledge_model()
    items = sorted(items, key=lambda i: (i.tenant, i.id))
    exceptions = sorted(exceptions, key=lambda e: (e.tenant, e.id))
    b = _Builder(model)
    released = {item.id: b.iri(item.tenant, "item", item.id) for item in items}
    for item in items:
        subject = released[item.id]
        b.graph.add((subject, RDF.type, b.ns.KnowledgeItem))
        b.entry(subject, item)
        for name, value in (("title", item.title), ("statement", item.statement),
                            ("subjectKey", item.subject_key), ("modality", item.modality.value),
                            ("authority", item.authority.value), ("origin", item.origin.value),
                            ("series", item.series), ("reviewedBy", item.reviewed_by),
                            ("reviewRef", item.review_ref)):
            b.add(subject, name, value)
        if item.confidence is not None:
            b.add(subject, "confidence", item.confidence, XSD.decimal)
        if item.constraint_provenance is not None:
            b.add(subject, "constraintAuthoredBy", item.constraint_provenance.authored_by)
            b.add(subject, "constraintApprovalRef", item.constraint_provenance.approval_ref)
        if item.constraint is not None:
            for position, predicate in enumerate(item.constraint.all_of):
                node = URIRef(f"{subject}/predicate/{position}")
                b.graph.add((subject, b.ns.hasPredicate, node))
                b.graph.add((node, RDF.type, b.ns.ConstraintPredicate))
                b.add(node, "position", position, XSD.integer)
                b.add(node, "factName", predicate.fact)
                b.add(node, "operator", predicate.operator.value)
                if predicate.value is not None:
                    b.add(node, "expectedValueJson", json.dumps(predicate.value, sort_keys=True))
                b.add(node, "whenMissing", predicate.when_missing)
        for target in item.supersedes:
            b.reference(subject, "supersedes", target, released, item.key)
        for target in item.overrides:
            b.reference(subject, "declaresOverrideOf", target, released, item.key)
    for exc in exceptions:
        subject = b.iri(exc.tenant, "exception", exc.id)
        b.graph.add((subject, RDF.type, b.ns.KnowledgeException))
        b.entry(subject, exc)
        for name, value in (("exceptionKind", exc.kind), ("rationale", exc.rationale),
                            ("grantedBy", exc.granted_by), ("approvalRef", exc.approval_ref)):
            b.add(subject, name, value)
        for target in exc.target_item_ids:
            b.reference(subject, "exceptionTarget", target, released, exc.key)
        if exc.winner_item_id:
            b.reference(subject, "precedenceWinner", exc.winner_item_id, released, exc.key)
    return KnowledgeProjection(graph=b.graph, omitted_references=tuple(b.omitted))


def validate_projection(projection: KnowledgeProjection, *, model_path: str | Path | None = None) -> ShapeReport:
    """Validate a projection against the shapes compiled from the canonical
    model.  Violations are reported as ``<focus> <path>: <message>``."""
    import pyshacl

    shapes = Graph().parse(data=_shapes(str(model_path or KNOWLEDGE_MODEL_PATH)), format="turtle")
    conforms, results, _ = pyshacl.validate(
        projection.graph, shacl_graph=shapes, inference="none", allow_warnings=True, advanced=False,
    )
    violations: list[str] = []
    sh = Namespace("http://www.w3.org/ns/shacl#")
    for result in results.subjects(RDF.type, sh.ValidationResult):
        focus = results.value(result, sh.focusNode)
        path = results.value(result, sh.resultPath)
        message = results.value(result, sh.resultMessage)
        focus_text = str(focus) if not isinstance(focus, BNode) else "_:blank"
        violations.append(f"{focus_text} {path}: {message}")
    return ShapeReport(conforms=bool(conforms), violations=tuple(sorted(violations)),
                       omitted_references=projection.omitted_references)


def shape_failures(items: Iterable[KnowledgeItem], exceptions: Iterable[KnowledgeException] = ()) -> list[str]:
    """Release-gate helper: SHACL violations of a candidate release (empty = conforms)."""
    report = validate_projection(project_release(items, exceptions))
    return [f"shacl: {violation}" for violation in report.violations]


__all__ = [
    "KNOWLEDGE_MODEL_PATH", "KnowledgeProjection", "ShapeReport", "knowledge_model", "project_release",
    "shape_failures", "validate_projection",
]
