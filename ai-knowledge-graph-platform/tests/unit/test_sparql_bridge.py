"""Tests for graphrag.graph.sparql_bridge.SPARQLBridge."""

from __future__ import annotations

import pytest
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS, XSD

from graphrag.graph.sparql_bridge import SPARQLBridge

BASE  = Namespace("https://graphrag.example.com/ontology#")
INST  = Namespace("https://graphrag.example.com/entity/")
ANNOT = Namespace("https://graphrag.example.com/annotation#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _minimal_graph() -> Graph:
    """Build a small rdflib Graph with two entities and one relation."""
    g = Graph()
    # Classes
    g.add((BASE.PERSON,       RDF.type,        OWL.Class))
    g.add((BASE.EMPLOYEE,     RDF.type,        OWL.Class))
    g.add((BASE.EMPLOYEE,     RDFS.subClassOf, BASE.PERSON))
    # Properties
    g.add((BASE.WORKS_AT,     RDF.type,        OWL.ObjectProperty))
    # Instances
    alice = INST["default/PERSON/Alice"]
    acme  = INST["default/ORG/Acme_Corp"]
    g.add((alice, RDF.type,        OWL.NamedIndividual))
    g.add((alice, RDF.type,        BASE.PERSON))
    g.add((alice, RDFS.label,      Literal("Alice")))
    g.add((acme,  RDF.type,        OWL.NamedIndividual))
    g.add((acme,  RDFS.label,      Literal("Acme Corp")))
    g.add((alice, BASE.WORKS_AT,   acme))
    # Reified confidence
    ax = INST["axiom/alice_works_acme"]
    g.add((ax, RDF.type,              OWL.Axiom))
    g.add((ax, OWL.annotatedSource,   alice))
    g.add((ax, OWL.annotatedProperty, BASE.WORKS_AT))
    g.add((ax, OWL.annotatedTarget,   acme))
    g.add((ax, ANNOT.confidence,      Literal(0.9, datatype=XSD.float)))
    return g


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestSPARQLBridgeConstruction:
    def test_from_graph(self) -> None:
        bridge = SPARQLBridge(_minimal_graph())
        assert bridge._g is not None

    def test_from_turtle_round_trip(self, tmp_path) -> None:
        g = _minimal_graph()
        ttl = tmp_path / "test.ttl"
        g.serialize(destination=str(ttl), format="turtle")

        bridge = SPARQLBridge.from_turtle(ttl)
        assert len(bridge._g) == len(g)

    def test_from_turtle_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            SPARQLBridge.from_turtle(tmp_path / "nonexistent.ttl")


class TestSPARQLBridgeQuery:
    def setup_method(self) -> None:
        self.bridge = SPARQLBridge(_minimal_graph())

    def test_select_returns_rows(self) -> None:
        rows = self.bridge.query(
            "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 3"
        )
        assert isinstance(rows, list)
        assert len(rows) <= 3
        assert all(isinstance(r, dict) for r in rows)

    def test_select_with_filter(self) -> None:
        rows = self.bridge.query(
            """
            SELECT ?label WHERE {
                ?e rdfs:label ?label .
                FILTER(str(?label) = "Alice")
            }
            """
        )
        assert len(rows) == 1
        assert rows[0]["label"] == "Alice"

    def test_empty_graph_returns_empty_list(self) -> None:
        bridge = SPARQLBridge(Graph())
        rows = bridge.query("SELECT ?s WHERE { ?s ?p ?o }")
        assert rows == []

    def test_invalid_sparql_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="SPARQL"):
            self.bridge.query("NOT VALID SPARQL AT ALL !!!!")

    def test_namespaces_injected_in_query(self) -> None:
        # Query using a custom namespace prefix
        rows = self.bridge.query(
            "SELECT ?x WHERE { ?x rdf:type owl:Class }",
            init_ns={"rdf": str(RDF), "owl": str(OWL)},
        )
        # BASE.PERSON, BASE.EMPLOYEE are owl:Class
        assert len(rows) >= 2

    def test_skos_prefix_is_available_without_manual_binding(self) -> None:
        self.bridge._g.add((INST["default/PERSON/Alice"], SKOS.prefLabel, Literal("Alice")))

        rows = self.bridge.query("SELECT ?label WHERE { ?entity skos:prefLabel ?label }")

        assert rows == [{"label": "Alice"}]

    def test_result_keys_are_variable_names(self) -> None:
        rows = self.bridge.query("SELECT ?label WHERE { ?e rdfs:label ?label }")
        for row in rows:
            assert "label" in row


class TestSPARQLBridgeDescribe:
    def test_describe_returns_turtle_string(self) -> None:
        bridge = SPARQLBridge(_minimal_graph())
        alice_uri = str(INST["default/PERSON/Alice"])
        ttl = bridge.describe(alice_uri)
        assert isinstance(ttl, str)
        assert len(ttl) > 0

    def test_describe_unknown_uri_returns_empty_doc(self) -> None:
        bridge = SPARQLBridge(_minimal_graph())
        ttl = bridge.describe("https://example.com/unknown/Entity")
        # Empty graph serialises as a valid (but minimal) Turtle doc
        assert isinstance(ttl, str)


class TestSPARQLBridgeUpdate:
    def setup_method(self) -> None:
        self.bridge = SPARQLBridge(_minimal_graph())

    def test_insert_data_adds_triple(self) -> None:
        before = len(self.bridge._g)
        self.bridge.update(
            """
            INSERT DATA {
                <https://graphrag.example.com/entity/default/PERSON/Bob>
                    a <https://graphrag.example.com/ontology#PERSON> .
            }
            """
        )
        after = len(self.bridge._g)
        assert after == before + 1

    def test_delete_data_removes_triple(self) -> None:
        alice = INST["default/PERSON/Alice"]
        assert (alice, RDFS.label, Literal("Alice")) in self.bridge._g

        self.bridge.update(
            """
            DELETE DATA {
                <https://graphrag.example.com/entity/default/PERSON/Alice>
                    <http://www.w3.org/2000/01/rdf-schema#label> "Alice" .
            }
            """
        )
        assert (alice, RDFS.label, Literal("Alice")) not in self.bridge._g

    def test_update_returns_new_triple_count(self) -> None:
        n = self.bridge.update(
            """
            INSERT DATA {
                <https://graphrag.example.com/entity/default/PERSON/Bob>
                    a <https://graphrag.example.com/ontology#PERSON> .
            }
            """
        )
        assert n == len(self.bridge._g)

    def test_service_rejected(self) -> None:
        with pytest.raises(ValueError, match="not permitted"):
            self.bridge.update(
                "INSERT { ?s ?p ?o } WHERE { SERVICE <http://evil.example.com/sparql> { ?s ?p ?o } }"
            )

    def test_load_rejected(self) -> None:
        # LOAD isn't a recognised update form at all (it's not in
        # _ALLOWED_UPDATE_FORMS), so this is rejected by the "no allowed
        # form found" branch rather than the explicit forbidden-keyword
        # check -- either way, LOAD never executes.
        with pytest.raises(ValueError, match="permitted"):
            self.bridge.update("LOAD <http://evil.example.com/data.ttl>")

    def test_load_rejected_even_when_wrapped_in_insert(self) -> None:
        # A LOAD smuggled alongside a permitted keyword must still be
        # rejected by the explicit _FORBIDDEN_UPDATE check.
        with pytest.raises(ValueError, match="not permitted"):
            self.bridge.update(
                "INSERT { ?s ?p ?o } WHERE { LOAD <http://evil.example.com/data.ttl> }"
            )

    def test_non_update_form_rejected(self) -> None:
        with pytest.raises(ValueError, match="permitted"):
            self.bridge.update("SELECT ?s WHERE { ?s ?p ?o }")

    def test_save_and_reload_round_trip(self, tmp_path) -> None:
        self.bridge.update(
            """
            INSERT DATA {
                <https://graphrag.example.com/entity/default/PERSON/Bob>
                    a <https://graphrag.example.com/ontology#PERSON> .
            }
            """
        )
        out = tmp_path / "updated.ttl"
        self.bridge.save(out)

        reloaded = SPARQLBridge.from_turtle(out)
        assert len(reloaded._g) == len(self.bridge._g)
        bob = INST["default/PERSON/Bob"]
        assert (bob, RDF.type, BASE.PERSON) in reloaded._g


class TestSPARQLBridgeConvenienceQueries:
    def setup_method(self) -> None:
        self.bridge = SPARQLBridge(_minimal_graph())

    def test_entity_relations_returns_outgoing(self) -> None:
        alice_uri = str(INST["default/PERSON/Alice"])
        rows = self.bridge.entity_relations(alice_uri)
        relations = [r["relation"] for r in rows]
        # WORKS_AT should appear; rdf:type and rdfs:label should be excluded
        assert any("WORKS_AT" in rel for rel in relations)
        assert not any(str(RDF.type) in rel for rel in relations)

    def test_subclass_hierarchy_all(self) -> None:
        rows = self.bridge.subclass_hierarchy()
        # EMPLOYEE ⊂ PERSON
        pairs = {(r["child"], r["parent"]) for r in rows}
        assert (str(BASE.EMPLOYEE), str(BASE.PERSON)) in pairs

    def test_subclass_hierarchy_root_filter(self) -> None:
        rows = self.bridge.subclass_hierarchy(root_type="PERSON")
        # Should return EMPLOYEE → PERSON pair
        assert any("EMPLOYEE" in r["child"] for r in rows)

    def test_confidence_summary_filters_by_min_conf(self) -> None:
        rows_all  = self.bridge.confidence_summary(min_conf=0.0)
        rows_high = self.bridge.confidence_summary(min_conf=0.95)
        assert len(rows_all) >= 1      # Alice → Acme at 0.9 shows up
        assert len(rows_high) == 0     # 0.9 < 0.95 → filtered out

    def test_confidence_summary_returns_correct_fields(self) -> None:
        rows = self.bridge.confidence_summary(min_conf=0.0)
        assert len(rows) >= 1
        row = rows[0]
        assert "source" in row
        assert "property" in row
        assert "target" in row
        assert "confidence" in row
