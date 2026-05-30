"""Tests for graphrag.graph.owl_reasoner.OWLRLReasoner."""

from __future__ import annotations

import copy
import inspect

import pytest
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from graphrag.graph.owl_reasoner import OWLRLReasoner

EX = Namespace("https://example.com/test#")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _subclass_chain_graph() -> Graph:
    """Graph encoding A ⊂ B ⊂ C (OWL-RL should infer A ⊂ C)."""
    g = Graph()
    for cls in (EX.A, EX.B, EX.C):
        g.add((cls, RDF.type, OWL.Class))
    g.add((EX.A, RDFS.subClassOf, EX.B))
    g.add((EX.B, RDFS.subClassOf, EX.C))
    return g


def _symmetric_graph() -> Graph:
    """Graph with a symmetric property: Alice knows Bob → Bob knows Alice."""
    g = Graph()
    g.add((EX.knows,   RDF.type,              OWL.ObjectProperty))
    g.add((EX.knows,   RDF.type,              OWL.SymmetricProperty))
    g.add((EX.Alice,   RDF.type,              OWL.NamedIndividual))
    g.add((EX.Bob,     RDF.type,              OWL.NamedIndividual))
    g.add((EX.Alice,   EX.knows,              EX.Bob))
    return g


def _inconsistent_graph() -> Graph:
    """A class that is declared as a subclass of owl:Nothing — inconsistency."""
    g = Graph()
    g.add((EX.BadClass, RDF.type,        OWL.Class))
    g.add((EX.BadClass, RDFS.subClassOf, OWL.Nothing))
    g.add((EX.bad,      RDF.type,        EX.BadClass))
    return g


# ── Construction tests ─────────────────────────────────────────────────────────

class TestOWLRLReasonerConstruction:
    def test_from_graph_wraps_correctly(self) -> None:
        g = Graph()
        r = OWLRLReasoner(g)
        assert r._g is g
        assert not r._closure_applied

    def test_from_turtle_round_trip(self, tmp_path) -> None:
        g = _subclass_chain_graph()
        ttl = tmp_path / "test.ttl"
        g.serialize(destination=str(ttl), format="turtle")

        reasoner = OWLRLReasoner.from_turtle(ttl)
        assert len(reasoner._g) == len(g)

    def test_from_turtle_missing_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            OWLRLReasoner.from_turtle(tmp_path / "nope.ttl")


# ── apply_closure tests ────────────────────────────────────────────────────────

class TestApplyClosure:
    def test_empty_graph_returns_non_negative_int(self) -> None:
        # owlrl adds OWL built-in tautologies (owl:Thing, owl:Nothing etc.)
        # even to empty graphs, so n >= 0 is the correct expectation.
        reasoner = OWLRLReasoner(Graph())
        n = reasoner.apply_closure()
        assert isinstance(n, int)
        assert n >= 0

    def test_subclass_chain_infers_transitive(self) -> None:
        """A ⊂ B ⊂ C → OWL-RL should materialise A ⊂ C."""
        reasoner = OWLRLReasoner(_subclass_chain_graph())
        n = reasoner.apply_closure()
        assert n > 0
        # Check the inferred triple is present
        assert (EX.A, RDFS.subClassOf, EX.C) in reasoner._g

    def test_symmetric_property_inferred(self) -> None:
        """Alice knows Bob → Bob knows Alice under owl:SymmetricProperty."""
        reasoner = OWLRLReasoner(_symmetric_graph())
        reasoner.apply_closure()
        assert (EX.Bob, EX.knows, EX.Alice) in reasoner._g

    def test_idempotent_second_call_returns_zero(self) -> None:
        reasoner = OWLRLReasoner(_subclass_chain_graph())
        n1 = reasoner.apply_closure()
        n2 = reasoner.apply_closure()
        assert n1 > 0
        assert n2 == 0


# ── is_consistent tests ────────────────────────────────────────────────────────

class TestIsConsistent:
    def test_valid_graph_is_consistent(self) -> None:
        # Symmetric graph: no individual is typed as owl:Nothing → consistent
        reasoner = OWLRLReasoner(_symmetric_graph())
        reasoner.apply_closure()
        assert reasoner.is_consistent() is True

    def test_inconsistent_graph_detected(self) -> None:
        """An individual typed as a subclass of owl:Nothing is inconsistent."""
        reasoner = OWLRLReasoner(_inconsistent_graph())
        reasoner.apply_closure()
        assert reasoner.is_consistent() is False


# ── inferred_triples tests ─────────────────────────────────────────────────────

class TestInferredTriples:
    def test_inferred_triples_subset_of_closed_graph(self) -> None:
        g        = _subclass_chain_graph()
        original = copy.deepcopy(g)
        reasoner = OWLRLReasoner(g)
        reasoner.apply_closure()

        inferred = reasoner.inferred_triples(original)
        assert isinstance(inferred, list)
        # All inferred triples are in the closed graph
        for triple in inferred:
            assert triple in reasoner._g
        # None of the inferred triples were in the original
        original_set = set(original)
        for triple in inferred:
            assert triple not in original_set

    def test_no_inferred_triples_before_closure(self) -> None:
        g        = _subclass_chain_graph()
        original = copy.deepcopy(g)
        reasoner = OWLRLReasoner(g)
        # Don't call apply_closure
        inferred = reasoner.inferred_triples(original)
        assert inferred == []


# ── serialize tests ────────────────────────────────────────────────────────────

class TestSerialize:
    def test_serialize_writes_valid_turtle(self, tmp_path) -> None:
        reasoner = OWLRLReasoner(_subclass_chain_graph())
        reasoner.apply_closure()
        out = tmp_path / "expanded.ttl"
        reasoner.serialize(out)

        assert out.exists()
        g2 = Graph()
        g2.parse(str(out), format="turtle")
        assert len(g2) == len(reasoner._g)

    def test_serialize_creates_parent_dirs(self, tmp_path) -> None:
        reasoner = OWLRLReasoner(Graph())
        out = tmp_path / "deep" / "nested" / "out.ttl"
        reasoner.serialize(out)
        assert out.exists()


# ── Source code quality checks ─────────────────────────────────────────────────

class TestSourceCodeQuality:
    def test_no_utcnow_in_owl_reasoner(self) -> None:
        import graphrag.graph.owl_reasoner as mod
        src = inspect.getsource(mod)
        assert "utcnow" not in src

    def test_no_hand_rolled_turtle(self) -> None:
        import graphrag.graph.owl_reasoner as mod
        src = inspect.getsource(mod)
        # Should not contain raw string concatenation like "@prefix" + ...
        assert '+ "@prefix"' not in src
        assert '+ "rdf:"' not in src
