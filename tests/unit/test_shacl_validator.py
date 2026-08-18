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


# ── Shape loading: file vs. fallback ─────────────────────────────────────────

class TestShapesLoadFromVersionControlledFiles:
    """Priority 1 hardening: shapes now live in ontology/shapes/*.ttl (the
    source of truth), not only as inline Python strings. See
    docs/context_graph_gap_plan.md."""

    def test_export_shapes_file_exists_and_loads(self) -> None:
        from graphrag.graph.shacl_validator import _EXPORT_SHAPES_PATH
        assert _EXPORT_SHAPES_PATH.exists()
        g = Graph().parse(str(_EXPORT_SHAPES_PATH), format="turtle")
        assert len(g) > 0

    def test_ingestion_shapes_file_exists_and_loads(self) -> None:
        from graphrag.graph.shacl_validator import _INGESTION_SHAPES_PATH
        assert _INGESTION_SHAPES_PATH.exists()
        g = Graph().parse(str(_INGESTION_SHAPES_PATH), format="turtle")
        assert len(g) > 0

    def test_fallback_is_byte_identical_to_file_in_triple_count(self) -> None:
        """The inline fallback must never silently drift from the .ttl file
        it's a last resort for — a triple-count mismatch means someone edited
        one without the other."""
        from graphrag.graph.shacl_validator import (
            _EXPORT_SHAPES_PATH,
            _SHAPES_TTL_FALLBACK,
            _load_shapes_graph,
        )
        from_file = _load_shapes_graph(_EXPORT_SHAPES_PATH, _SHAPES_TTL_FALLBACK)
        from_fallback = Graph().parse(data=_SHAPES_TTL_FALLBACK, format="turtle")
        assert len(from_file) == len(from_fallback)

    def test_missing_file_falls_back_without_raising(self, tmp_path) -> None:
        from graphrag.graph.shacl_validator import _load_shapes_graph
        g = _load_shapes_graph(tmp_path / "does-not-exist.ttl", """
            @prefix sh: <http://www.w3.org/ns/shacl#> .
            [] a sh:NodeShape .
        """)
        assert len(g) == 1


# ── Severity: sh:Violation vs sh:Warning ─────────────────────────────────────

class TestSeverityDistinction:
    """A missing confidence annotation is a WARNING (export_rdf.py only sets
    it when the source edge's confidence is non-null, so absence can be
    legitimate); a malformed one, when present, is still a VIOLATION."""

    def test_missing_confidence_is_warning_not_violation(self) -> None:
        g = _valid_axiom_graph()
        g.remove((EX.axiom1, ANNOT.confidence, None))  # drop confidence entirely
        report = SHACLValidator(g).validate_report()
        assert report.conforms is True   # warning alone must not fail validation
        assert report.counts["warnings"] == 1
        assert report.counts["violations"] == 0

    def test_out_of_range_confidence_is_still_a_violation(self) -> None:
        g = _valid_axiom_graph()
        g.remove((EX.axiom1, ANNOT.confidence, None))
        g.add((EX.axiom1, ANNOT.confidence, Literal(1.5, datatype=XSD.float)))
        report = SHACLValidator(g).validate_report()
        assert report.conforms is False
        assert report.counts["violations"] >= 1

    def test_warning_and_violation_together_still_fails_conforms(self) -> None:
        """A warning must never mask a real violation present in the same run."""
        g = _valid_axiom_graph()
        g.remove((EX.axiom1, ANNOT.confidence, None))  # -> warning
        g.add((EX.doc_no_label, RDF.type, OWL.NamedIndividual))
        g.add((EX.doc_no_label, RDF.type, EX.DOCUMENT))  # missing label -> violation
        report = SHACLValidator(g).validate_report()
        assert report.conforms is False
        assert report.counts["violations"] == 1
        assert report.counts["warnings"] == 1


# ── Machine-readable report (ShaclReport) ────────────────────────────────────

class TestValidateReport:
    def test_old_and_new_api_agree_on_conforms(self) -> None:
        g = _valid_entity_graph()
        conforms_old, _ = SHACLValidator(g).validate()
        report = SHACLValidator(g).validate_report()
        assert conforms_old == report.conforms

    def test_failures_by_shape_uses_named_shapes_not_blank_nodes(self) -> None:
        """Shapes were renamed from anonymous [] blocks to stable IRIs
        specifically so failures-by-shape grouping is meaningful — a blank
        node's local id is not stable across runs."""
        g = Graph()
        g.add((EX.doc1, RDF.type, OWL.NamedIndividual))
        g.add((EX.doc1, RDF.type, EX.DOCUMENT))
        # missing label -> violates EntityLabelProperty
        report = SHACLValidator(g).validate_report()
        assert "EntityLabelProperty" in report.failures_by_shape
        assert not report.failures_by_shape["EntityLabelProperty"] == 0

    def test_results_list_has_structured_detail(self) -> None:
        g = Graph()
        g.add((EX.doc1, RDF.type, OWL.NamedIndividual))
        g.add((EX.doc1, RDF.type, EX.DOCUMENT))
        report = SHACLValidator(g).validate_report()
        assert len(report.results) == 1
        result = report.results[0]
        assert result.severity == "Violation"
        assert result.focus_node == str(EX.doc1)
        assert "label" in result.message.lower()
        assert result.shape == "EntityLabelProperty"

    def test_empty_graph_has_zero_counts(self) -> None:
        report = SHACLValidator(Graph()).validate_report()
        assert report.conforms is True
        assert report.counts == {"total_results": 0, "violations": 0, "warnings": 0}
        assert report.failures_by_shape == {}


# ── validate_relational_batch_report (machine-readable ingestion gate) ───────

class TestValidateRelationalBatchReport:
    def _entity(self, id_, name, etype, tenant):
        e = type("E", (), {})()
        e.id, e.name, e.type, e.tenant = id_, name, etype, tenant
        return e

    def _relation(self, id_, src, tgt, relation, confidence):
        r = type("R", (), {})()
        r.id, r.source_entity_id, r.target_entity_id = id_, src, tgt
        r.relation, r.confidence = relation, confidence
        return r

    def test_valid_batch_conforms_with_no_results(self) -> None:
        entities = [self._entity("e1", "Acme", "ORG", "acme")]
        relations: list = []
        report = SHACLValidator.validate_relational_batch_report(entities, relations, tenant="acme")
        assert report.conforms is True
        assert report.counts["total_results"] == 0

    def test_out_of_range_confidence_violates_and_names_the_shape(self) -> None:
        """getattr(relation, "confidence", -1) always adds SOME literal, so
        minCount can never fail through this code path — the reachable
        violation is an out-of-range value, e.g. the -1 default itself."""
        src = self._entity("e1", "Acme", "ORG", "acme")
        tgt = self._entity("e2", "Beta", "ORG", "acme")
        rel = self._relation("r1", "e1", "e2", "PARTNERS_WITH", -1.0)  # out of [0,1]
        report = SHACLValidator.validate_relational_batch_report([src, tgt], [rel], tenant="acme")
        assert report.conforms is False
        assert any(r.shape for r in report.results)

    def test_legacy_tuple_api_still_works_for_this_path_too(self) -> None:
        e = self._entity("e1", "Acme", "ORG", "acme")
        conforms, text = SHACLValidator.validate_relational_batch([e], [], tenant="acme")
        assert conforms is True
        assert isinstance(text, str)
