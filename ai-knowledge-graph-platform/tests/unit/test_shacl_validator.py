"""Tests for graphrag.graph.shacl_validator.SHACLValidator."""

from __future__ import annotations

import pytest
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from graphrag.graph.shacl_validator import SHACLValidator

EX = Namespace("https://example.com/test#")
ANNOT = Namespace("https://graphrag.example.com/annotation#")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _valid_entity_graph() -> Graph:
    """A NamedIndividual with a label and a domain type — should conform."""
    g = Graph()
    g.add((EX.doc1, RDF.type, OWL.NamedIndividual))
    g.add((EX.doc1, RDF.type, EX.DOCUMENT))
    g.add((EX.doc1, RDFS.label, Literal("Test Doc")))
    return g


def _valid_axiom_graph() -> Graph:
    """A fully reified owl:Axiom with source/property/target and valid confidence."""
    g = Graph()
    ax = EX.axiom1
    g.add((ax, RDF.type, OWL.Axiom))
    g.add((ax, OWL.annotatedSource, EX.a))
    g.add((ax, OWL.annotatedProperty, EX.RELATES_TO))
    g.add((ax, OWL.annotatedTarget, EX.b))
    g.add((ax, ANNOT.confidence, Literal(0.85, datatype=XSD.float)))
    return g


# ── Construction tests ─────────────────────────────────────────────────────────

class TestSHACLValidatorConstruction:
    def test_wraps_graph(self) -> None:
        g = Graph()
        v = SHACLValidator(g)
        assert v._g is g

    def test_from_turtle_round_trip(self, tmp_path) -> None:
        g = _valid_entity_graph()
        ttl = tmp_path / "test.ttl"
        g.serialize(destination=str(ttl), format="turtle")

        v = SHACLValidator.from_turtle(ttl)
        assert len(v._g) == len(g)

    def test_from_turtle_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            SHACLValidator.from_turtle(tmp_path / "missing.ttl")


# ── Entity shape tests ─────────────────────────────────────────────────────────

class TestEntityShape:
    def test_valid_entity_conforms(self) -> None:
        conforms, _ = SHACLValidator(_valid_entity_graph()).validate()
        assert conforms

    def test_missing_label_violates(self) -> None:
        g = Graph()
        g.add((EX.doc1, RDF.type, OWL.NamedIndividual))
        g.add((EX.doc1, RDF.type, EX.DOCUMENT))
        conforms, report = SHACLValidator(g).validate()
        assert not conforms
        assert "label" in report.lower()

    def test_missing_domain_type_violates(self) -> None:
        g = Graph()
        g.add((EX.doc1, RDF.type, OWL.NamedIndividual))
        g.add((EX.doc1, RDFS.label, Literal("Test Doc")))
        conforms, report = SHACLValidator(g).validate()
        assert not conforms


# ── Axiom shape tests ───────────────────────────────────────────────────────────

class TestAxiomShape:
    def test_valid_axiom_conforms(self) -> None:
        conforms, _ = SHACLValidator(_valid_axiom_graph()).validate()
        assert conforms

    def test_axiom_missing_target_violates(self) -> None:
        g = Graph()
        ax = EX.axiom1
        g.add((ax, RDF.type, OWL.Axiom))
        g.add((ax, OWL.annotatedSource, EX.a))
        g.add((ax, OWL.annotatedProperty, EX.RELATES_TO))
        conforms, report = SHACLValidator(g).validate()
        assert not conforms
        assert "annotatedTarget" in report

    def test_confidence_out_of_range_violates(self) -> None:
        g = _valid_axiom_graph()
        g.remove((EX.axiom1, ANNOT.confidence, None))
        g.add((EX.axiom1, ANNOT.confidence, Literal(1.5, datatype=XSD.float)))
        conforms, _ = SHACLValidator(g).validate()
        assert not conforms

    def test_confidence_wrong_datatype_violates(self) -> None:
        g = _valid_axiom_graph()
        g.remove((EX.axiom1, ANNOT.confidence, None))
        g.add((EX.axiom1, ANNOT.confidence, Literal("high")))
        conforms, _ = SHACLValidator(g).validate()
        assert not conforms


# ── Empty graph ─────────────────────────────────────────────────────────────────

class TestEmptyGraph:
    def test_empty_graph_conforms(self) -> None:
        """No targeted nodes → vacuously conforms."""
        conforms, _ = SHACLValidator(Graph()).validate()
        assert conforms
