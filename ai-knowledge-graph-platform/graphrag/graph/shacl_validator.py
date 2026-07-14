"""SHACL validation for the exported RDF knowledge graph.

Validates the graph produced by ``scripts/export_rdf.py`` against a set of
SHACL shapes that encode the invariants the ingestion pipeline is already
supposed to guarantee:

  - Every ``owl:NamedIndividual`` has an ``rdfs:label`` and at least one
    ``rdf:type`` beyond ``owl:NamedIndividual`` itself.
  - Every ``owl:Axiom`` (reified relation) has ``annotatedSource``,
    ``annotatedProperty``, and ``annotatedTarget``.
  - ``:confidence`` annotations are ``xsd:float`` in the closed range [0, 1].

This is a structural sanity check that runs *after* export, independent of
the OWL-RL reasoner (``owl_reasoner.py``), which handles entailment rather
than shape conformance. SHACL and OWL-RL answer different questions:
OWL-RL asks "what else is entailed by this graph?"; SHACL asks "does this
graph conform to the shape my pipeline promises?"

Usage::

    from graphrag.graph.shacl_validator import SHACLValidator

    validator = SHACLValidator.from_turtle("exports/graph_export.ttl")
    conforms, report = validator.validate()
    if not conforms:
        print(report)

Integration with export_rdf.py::

    python scripts/export_rdf.py --tenant default --validate
"""

from __future__ import annotations

from pathlib import Path

import structlog
from rdflib import Graph

log = structlog.get_logger(__name__)

# ── SHACL shapes (Turtle) ────────────────────────────────────────────────────
# Kept inline rather than as a separate .ttl asset — the shapes are small,
# tightly coupled to export_rdf.py's data model, and easier to keep in sync
# living next to the code that produces the graph they validate.

_SHAPES_TTL = """
@prefix sh:    <http://www.w3.org/ns/shacl#> .
@prefix owl:   <http://www.w3.org/2002/07/owl#> .
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix annot: <https://graphrag.example.com/annotation#> .

# Every NamedIndividual (entity) must carry a human-readable label and a
# domain type beyond owl:NamedIndividual itself.
[] a sh:NodeShape ;
    sh:targetClass owl:NamedIndividual ;
    sh:property [
        sh:path rdfs:label ;
        sh:minCount 1 ;
        sh:datatype xsd:string ;
        sh:message "Entity is missing an rdfs:label." ;
    ] ;
    sh:property [
        sh:path rdf:type ;
        sh:minCount 2 ;
        sh:message "Entity must have a domain type beyond owl:NamedIndividual." ;
    ] .

# Every reified relation (owl:Axiom) must carry its full subject/predicate/
# object triple as annotations — export_rdf.py always sets all three.
[] a sh:NodeShape ;
    sh:targetClass owl:Axiom ;
    sh:property [
        sh:path owl:annotatedSource ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:message "Axiom is missing annotatedSource." ;
    ] ;
    sh:property [
        sh:path owl:annotatedProperty ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:message "Axiom is missing annotatedProperty." ;
    ] ;
    sh:property [
        sh:path owl:annotatedTarget ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:message "Axiom is missing annotatedTarget." ;
    ] ;
    sh:property [
        sh:path annot:confidence ;
        sh:maxCount 1 ;
        sh:datatype xsd:float ;
        sh:minInclusive 0.0 ;
        sh:maxInclusive 1.0 ;
        sh:message "confidence must be an xsd:float in [0, 1]." ;
    ] .
"""


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

        Raises
        ------
        ImportError
            If the ``pyshacl`` package is not installed.
        """
        try:
            from pyshacl import validate  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pyshacl is required for SHACL validation: "
                "pip install pyshacl>=0.29.0"
            ) from exc

        shapes_graph = Graph().parse(data=_SHAPES_TTL, format="turtle")
        conforms, _results_graph, results_text = validate(
            self._g,
            shacl_graph=shapes_graph,
            data_graph_format="turtle",
            shacl_graph_format="turtle",
            inference="none",
            abort_on_first=False,
        )
        log.info("shacl_validator.validated", conforms=conforms, triples=len(self._g))
        return conforms, results_text
