"""SHACL validation for the platform's RDF representations.

Validates two things:

  1. The graph produced by ``scripts/export_rdf.py`` — structural sanity
     checks that encode invariants the ingestion pipeline is already
     supposed to guarantee (every entity has a label and a type; every
     reified relation carries its full triple; confidence is a valid float
     in [0, 1]).
  2. A candidate relational-ingestion batch (``validate_relational_batch``) —
     a real pre-write mutation gate for ``graphrag/ingestion/relational.py``.

This is a structural sanity check that runs independent of the OWL-RL
reasoner (``owl_reasoner.py``), which handles entailment rather than shape
conformance. SHACL and OWL-RL answer different questions: OWL-RL asks "what
else is entailed by this graph?"; SHACL asks "does this graph conform to the
shape my pipeline promises?"

Shapes live in version-controlled Turtle files under ``ontology/shapes/``
(the source of truth), loaded by path at validation time. The inline string
fallbacks below exist only for the case where the file is missing (e.g. a
packaging/deployment issue that stripped non-.py assets) — they are kept
byte-identical to the shipped .ttl files, not a second design surface, so
loading from disk vs. the fallback can never silently change behavior.

Severity — ``sh:Violation`` vs ``sh:Warning``
----------------------------------------------
Every constraint declares its severity explicitly. ``conforms`` (both from
``validate()`` and ``validate_report()``) reflects only ``sh:Violation``
results — ``sh:Warning`` results are visible in the report and counts but do
not fail validation, via pyshacl's ``allow_warnings=True``. Without that
flag, pyshacl's ``conforms`` flips to ``False`` for a bare Warning too
(verified empirically — this is not documented anywhere obvious), which
would silently turn every advisory warning into a rejection.

Usage::

    from graphrag.graph.shacl_validator import SHACLValidator

    validator = SHACLValidator.from_turtle("exports/graph_export.ttl")
    conforms, report = validator.validate()          # unchanged, back-compat
    if not conforms:
        print(report)

    # Machine-readable report, additive — new capability, old API untouched:
    detailed = validator.validate_report()
    print(detailed.counts)          # {"validated": N, "violations": V, "warnings": W}
    print(detailed.failures_by_shape)  # {"EntityLabelProperty": 3, ...}

Integration with export_rdf.py::

    python scripts/export_rdf.py --tenant default --validate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, SH, XSD

from graphrag.core.config import ROOT

log = structlog.get_logger(__name__)

try:
    from prometheus_client import Counter
except ImportError:  # pragma: no cover - optional at local import time
    Counter = None

_records_validated = Counter(
    "graphrag_shacl_records_validated_total",
    "SHACL validation runs by target (export | relational_batch)",
    ["target"],
) if Counter else None
_validation_failures = Counter(
    "graphrag_shacl_validation_failures_total",
    "SHACL validation failures (non-conformant runs) by target and severity",
    ["target", "severity"],
) if Counter else None
_failures_by_shape = Counter(
    "graphrag_shacl_failures_by_shape_total",
    "SHACL result count by target and source shape name",
    ["target", "shape"],
) if Counter else None

_SHAPES_DIR = ROOT / "ontology" / "shapes"
_EXPORT_SHAPES_PATH = _SHAPES_DIR / "export.shapes.ttl"
_INGESTION_SHAPES_PATH = _SHAPES_DIR / "ingestion.shapes.ttl"

# ── Fallback shapes (Turtle) ─────────────────────────────────────────────────
# Kept byte-identical to ontology/shapes/*.ttl. Used ONLY if the file is
# missing at runtime (e.g. a deployment that stripped non-.py assets) — the
# .ttl files are the source of truth; this is a last-resort, not a second
# place to edit shapes.

_SHAPES_TTL_FALLBACK = """
@prefix sh:    <http://www.w3.org/ns/shacl#> .
@prefix owl:   <http://www.w3.org/2002/07/owl#> .
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix annot: <https://graphrag.example.com/annotation#> .
@prefix shapes: <https://graphrag.example.com/shapes#> .

shapes:EntityShape a sh:NodeShape ;
    sh:targetClass owl:NamedIndividual ;
    sh:property shapes:EntityLabelProperty ;
    sh:property shapes:EntityTypeProperty .

shapes:EntityLabelProperty a sh:PropertyShape ;
    sh:path rdfs:label ;
    sh:minCount 1 ;
    sh:datatype xsd:string ;
    sh:severity sh:Violation ;
    sh:message "Entity is missing an rdfs:label." .

shapes:EntityTypeProperty a sh:PropertyShape ;
    sh:path rdf:type ;
    sh:minCount 2 ;
    sh:severity sh:Violation ;
    sh:message "Entity must have a domain type beyond owl:NamedIndividual." .

shapes:AxiomShape a sh:NodeShape ;
    sh:targetClass owl:Axiom ;
    sh:property shapes:AxiomSourceProperty ;
    sh:property shapes:AxiomPropertyProperty ;
    sh:property shapes:AxiomTargetProperty ;
    sh:property shapes:AxiomConfidencePresenceProperty ;
    sh:property shapes:AxiomConfidenceRangeProperty .

shapes:AxiomSourceProperty a sh:PropertyShape ;
    sh:path owl:annotatedSource ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:severity sh:Violation ;
    sh:message "Axiom is missing annotatedSource." .

shapes:AxiomPropertyProperty a sh:PropertyShape ;
    sh:path owl:annotatedProperty ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:severity sh:Violation ;
    sh:message "Axiom is missing annotatedProperty." .

shapes:AxiomTargetProperty a sh:PropertyShape ;
    sh:path owl:annotatedTarget ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:severity sh:Violation ;
    sh:message "Axiom is missing annotatedTarget." .

shapes:AxiomConfidencePresenceProperty a sh:PropertyShape ;
    sh:path annot:confidence ;
    sh:minCount 1 ;
    sh:severity sh:Warning ;
    sh:message "Axiom has no confidence annotation (may be legitimate for older data)." .

shapes:AxiomConfidenceRangeProperty a sh:PropertyShape ;
    sh:path annot:confidence ;
    sh:maxCount 1 ;
    sh:datatype xsd:float ;
    sh:minInclusive 0.0 ;
    sh:maxInclusive 1.0 ;
    sh:severity sh:Violation ;
    sh:message "confidence must be an xsd:float in [0, 1]." .
"""

_INGESTION_SHAPES_TTL_FALLBACK = """
@prefix sh:  <http://www.w3.org/ns/shacl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:<http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ing: <https://graphrag.example.com/ingestion#> .

ing:MappedEntityShape a sh:NodeShape ;
    sh:targetClass ing:MappedEntity ;
    sh:property [ sh:path rdfs:label ; sh:minCount 1 ; sh:datatype xsd:string ;
                  sh:severity sh:Violation ; sh:message "Mapped entity is missing a label." ] ;
    sh:property [ sh:path ing:entityType ; sh:minCount 1 ; sh:datatype xsd:string ;
                  sh:severity sh:Violation ; sh:message "Mapped entity is missing an entityType." ] ;
    sh:property [ sh:path ing:tenant ; sh:minCount 1 ; sh:datatype xsd:string ;
                  sh:severity sh:Violation ; sh:message "Mapped entity is missing a tenant." ] .

ing:MappedRelationShape a sh:NodeShape ;
    sh:targetClass ing:MappedRelation ;
    sh:property [ sh:path ing:source ; sh:minCount 1 ; sh:class ing:MappedEntity ;
                  sh:severity sh:Violation ; sh:message "Mapped relation has no valid source entity." ] ;
    sh:property [ sh:path ing:target ; sh:minCount 1 ; sh:class ing:MappedEntity ;
                  sh:severity sh:Violation ; sh:message "Mapped relation has no valid target entity." ] ;
    sh:property [ sh:path ing:relation ; sh:minCount 1 ; sh:datatype xsd:string ;
                  sh:severity sh:Violation ; sh:message "Mapped relation is missing a relation name." ] ;
    sh:property [ sh:path ing:confidence ; sh:minCount 1 ; sh:maxCount 1 ;
                  sh:datatype xsd:float ; sh:minInclusive 0.0 ; sh:maxInclusive 1.0 ;
                  sh:severity sh:Violation ; sh:message "Mapped relation confidence must be an xsd:float in [0, 1]." ] ;
    sh:property [ sh:path ing:tenant ; sh:minCount 1 ; sh:datatype xsd:string ;
                  sh:severity sh:Violation ; sh:message "Mapped relation is missing a tenant." ] .
"""

_INGEST = Namespace("https://graphrag.example.com/ingestion#")


def _load_shapes_graph(path: Path, fallback_ttl: str) -> Graph:
    """Load a shapes graph from `path`; fall back to the inline Turtle string
    if the file is missing. Never silently returns an empty shapes graph —
    a parse error in either source is a real bug and should raise."""
    if path.exists():
        return Graph().parse(str(path), format="turtle")
    log.warning("shacl_validator.shapes_file_missing", path=str(path),
               note="using inline fallback shapes")
    return Graph().parse(data=fallback_ttl, format="turtle")


@dataclass
class ShaclResult:
    """One SHACL validation result, as a plain structure — not the raw RDF.

    Meant to be consumed by callers that want per-violation detail (a UI, a
    log line, a test assertion) without walking the pyshacl results graph
    themselves.
    """
    severity: str          # "Violation" | "Warning" | "Info"
    focus_node: str
    message: str
    shape: str              # local name of the sh:sourceShape, e.g. "EntityLabelProperty"
    result_path: str = ""


@dataclass
class ShaclReport:
    """Machine-readable SHACL validation report — additive alongside the
    existing (bool, str) `validate()` contract, not a replacement for it."""
    conforms: bool
    text: str
    results: list[ShaclResult] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        violations = sum(1 for r in self.results if r.severity == "Violation")
        warnings = sum(1 for r in self.results if r.severity == "Warning")
        return {
            "total_results": len(self.results),
            "violations": violations,
            "warnings": warnings,
        }

    @property
    def failures_by_shape(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.shape] = counts.get(r.shape, 0) + 1
        return counts


def _local_name(uri: str) -> str:
    """The fragment/last-segment of a URI, for grouping by shape without the
    full namespace noise. Blank-node shapes (bNNN) pass through as-is —
    they only occur if a shapes file still declares an anonymous [] shape."""
    for sep in ("#", "/"):
        if sep in uri:
            return uri.rsplit(sep, 1)[-1]
    return uri


def _parse_results(results_graph: Graph) -> list[ShaclResult]:
    parsed: list[ShaclResult] = []
    for result in results_graph.subjects(RDF.type, SH.ValidationResult):
        severity_uri = results_graph.value(result, SH.resultSeverity)
        message = results_graph.value(result, SH.resultMessage)
        focus = results_graph.value(result, SH.focusNode)
        path = results_graph.value(result, SH.resultPath)
        shape = results_graph.value(result, SH.sourceShape)
        parsed.append(ShaclResult(
            severity=_local_name(str(severity_uri)) if severity_uri else "Violation",
            focus_node=str(focus) if focus else "",
            message=str(message) if message else "",
            shape=_local_name(str(shape)) if shape else "",
            result_path=str(path) if path else "",
        ))
    return parsed


def _run_pyshacl(data_graph: Graph, shapes_graph: Graph) -> tuple[bool, str, Graph]:
    """Shared pyshacl invocation. `allow_warnings=True` is load-bearing: without
    it, a single sh:Warning-severity result flips `conforms` to False just like
    a sh:Violation would (verified empirically against pyshacl 0.40 — this
    isn't documented anywhere obvious), which would make every advisory
    warning behave as a silent rejection."""
    try:
        from pyshacl import validate  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyshacl is required for SHACL validation: "
            "pip install pyshacl>=0.29.0"
        ) from exc

    return validate(
        data_graph,
        shacl_graph=shapes_graph,
        data_graph_format="turtle",
        shacl_graph_format="turtle",
        inference="none",
        abort_on_first=False,
        allow_warnings=True,
    )


def _record_metrics(target: str, report: ShaclReport) -> None:
    if _records_validated is not None:
        _records_validated.labels(target=target).inc()
    if not report.conforms and _validation_failures is not None:
        _validation_failures.labels(target=target, severity="Violation").inc()
    if report.counts["warnings"] and _validation_failures is not None:
        _validation_failures.labels(target=target, severity="Warning").inc()
    if _failures_by_shape is not None:
        for shape, count in report.failures_by_shape.items():
            _failures_by_shape.labels(target=target, shape=shape).inc(count)


class SHACLValidator:
    """Validate an rdflib Graph against the platform's SHACL shapes.

    Parameters
    ----------
    graph :
        rdflib Graph to validate (typically the output of ``export_rdf.py``).
    """

    def __init__(self, graph: Graph) -> None:
        self._g = graph

    # ── Constructors ───────────────────────────────────────────────────────────

    @classmethod
    def from_turtle(cls, ttl_path: Path | str) -> "SHACLValidator":
        """Parse a Turtle file and return a validator instance.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        """
        path = Path(ttl_path)
        if not path.exists():
            raise FileNotFoundError(f"Turtle file not found: {path}")
        g = Graph()
        g.parse(str(path), format="turtle")
        log.info("shacl_validator.loaded", path=str(path), triples=len(g))
        return cls(g)

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate(self) -> tuple[bool, str]:
        """Run SHACL validation and return ``(conforms, text_report)``.

        Unchanged signature — see ``validate_report()`` for the
        machine-readable form (counts, per-result detail, Prometheus
        metrics), added alongside this rather than replacing it.

        Raises
        ------
        ImportError
            If the ``pyshacl`` package is not installed.
        """
        report = self.validate_report()
        return report.conforms, report.text

    def validate_report(self) -> ShaclReport:
        """Run SHACL validation and return a machine-readable ``ShaclReport``.

        Also records Prometheus counters (validated runs, failures by
        severity, failures by shape) when ``prometheus_client`` is installed
        — a no-op otherwise, same optional-dependency pattern used elsewhere
        in ``graphrag/observability``.
        """
        shapes_graph = _load_shapes_graph(_EXPORT_SHAPES_PATH, _SHAPES_TTL_FALLBACK)
        conforms, results_graph, results_text = _run_pyshacl(self._g, shapes_graph)
        report = ShaclReport(
            conforms=conforms,
            text=results_text,
            results=_parse_results(results_graph),
        )
        log.info("shacl_validator.validated", conforms=conforms, triples=len(self._g),
                 **report.counts)
        _record_metrics("export", report)
        return report

    @staticmethod
    def validate_relational_batch(
        entities: list[object], relations: list[object], *, tenant: str,
    ) -> tuple[bool, str]:
        """Validate a candidate relational import before Neo4j writes.

        The temporary RDF graph is intentionally small and uses dedicated
        ingestion shapes: mapped entities require identity/type/tenant and
        relations require valid endpoint nodes, relation name, confidence and
        tenant. This complements the post-write export validation by making
        SHACL a real mutation gate for the relational ingestion path.

        Unchanged signature — see ``validate_relational_batch_report()`` for
        the machine-readable form.
        """
        report = SHACLValidator.validate_relational_batch_report(
            entities, relations, tenant=tenant,
        )
        return report.conforms, report.text

    @staticmethod
    def validate_relational_batch_report(
        entities: list[object], relations: list[object], *, tenant: str,
    ) -> ShaclReport:
        """Machine-readable form of ``validate_relational_batch`` — additive,
        same content, plus per-result detail and Prometheus counters."""
        graph = Graph()
        entity_uris: dict[str, URIRef] = {}
        for entity in entities:
            entity_id = str(getattr(entity, "id", ""))
            uri = URIRef(f"https://graphrag.example.com/ingestion/entity/{entity_id}")
            entity_uris[entity_id] = uri
            graph.add((uri, RDF.type, _INGEST.MappedEntity))
            graph.add((uri, RDFS.label, Literal(str(getattr(entity, "name", "")))))
            graph.add((uri, _INGEST.entityType, Literal(str(getattr(entity, "type", "")))))
            graph.add((uri, _INGEST.tenant, Literal(str(getattr(entity, "tenant", "")))))

        for relation in relations:
            relation_id = str(getattr(relation, "id", ""))
            uri = URIRef(f"https://graphrag.example.com/ingestion/relation/{relation_id}")
            source_id = str(getattr(relation, "source_entity_id", ""))
            target_id = str(getattr(relation, "target_entity_id", ""))
            graph.add((uri, RDF.type, _INGEST.MappedRelation))
            graph.add((uri, _INGEST.source, entity_uris.get(source_id, URIRef("urn:missing:source"))))
            graph.add((uri, _INGEST.target, entity_uris.get(target_id, URIRef("urn:missing:target"))))
            graph.add((uri, _INGEST.relation, Literal(str(getattr(relation, "relation", "")))))
            graph.add((uri, _INGEST.confidence, Literal(float(getattr(relation, "confidence", -1)), datatype=XSD.float)))
            graph.add((uri, _INGEST.tenant, Literal(tenant)))

        shapes_graph = _load_shapes_graph(_INGESTION_SHAPES_PATH, _INGESTION_SHAPES_TTL_FALLBACK)
        conforms, results_graph, results_text = _run_pyshacl(graph, shapes_graph)
        report = ShaclReport(
            conforms=conforms,
            text=results_text,
            results=_parse_results(results_graph),
        )
        log.info(
            "shacl_validator.relational_batch_validated",
            conforms=conforms,
            entities=len(entities),
            relations=len(relations),
            tenant=tenant,
            **report.counts,
        )
        _record_metrics("relational_batch", report)
        return report
