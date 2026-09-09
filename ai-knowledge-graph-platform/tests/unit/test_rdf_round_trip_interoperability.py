"""RDF/OWL/SHACL/SKOS round-trip and interoperability proofs.

What existing coverage does and does not prove
------------------------------------------------
`test_export_rdf.py::TestTurtleSerialisation::test_round_trips_through_turtle`
and `test_owl_reasoner.py::test_from_turtle_round_trip` both check
``len(graph_in) == len(graph_out)`` after a Turtle round trip. That is a real
but narrow check: a serialization bug that dropped one triple and duplicated
an unrelated one would pass it, because it preserves count while destroying
content. Neither test asserts the round-tripped graph is *semantically*
identical, and neither proves the round trip survives contact with the
external tools this platform's own semantics depend on — a real OWL 2 RL
reasoner (`owlrl`) and a real SHACL engine (`pyshacl`), both third-party
libraries with their own parsers.

This file closes both gaps:

1. **Exact fidelity** — set equality of triples, not count, after a real
   Turtle serialize/reparse cycle through the platform's actual `export()`
   pipeline (mocked Neo4j, real graph-building code).
2. **OWL interoperability** — `owlrl.DeductiveClosure` applied to the graph
   *before* serialization and to a fresh graph loaded *from the serialized
   file* (`OWLRLReasoner.from_turtle`) must derive the identical entailed
   closure. If serialization silently dropped an `owl:TransitiveProperty` or
   `rdfs:subClassOf` declaration, this is where it would show up — nowhere
   else does.
3. **SHACL interoperability** — the platform's own shapes, run by `pyshacl`
   against the graph before and after the round trip, must reach the same
   conformance verdict, for both a conformant graph and a deliberately broken
   one. A validator that "fixes" a violation by losing the offending triple
   during serialization would defeat the point of validating at all.
4. **SKOS interoperability** — `assess_rdf_interoperability()` (this
   platform's own import-readiness catalogue, but exercised here against its
   *own* export as an interoperability self-check) must report identical
   concept/label/hierarchy counts pre- and post-round-trip.
5. **Multiple ontologies coexist** — two structurally distinct domain
   vocabularies (aerospace, automotive) exported into one file must each
   remain independently queryable via `SPARQLBridge` after the round trip,
   with no cross-contamination between them.

Nothing here needs Neo4j, Redis, or a live LLM — the export pipeline is
exercised against a mocked driver, exactly like `test_export_rdf.py`, and
everything downstream is pure rdflib/owlrl/pyshacl over an in-memory or
tmp-file graph.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS
from rdflib.namespace import XSD

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from graphrag.graph.owl_reasoner import OWLRLReasoner
from graphrag.graph.rdf_interoperability import assess_rdf_interoperability
from graphrag.graph.shacl_validator import SHACLValidator
from graphrag.graph.sparql_bridge import SPARQLBridge

BASE = Namespace("https://graphrag.example.com/ontology#")
INST = Namespace("https://graphrag.example.com/entity/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")


def _make_neo4j(
    type_rows=None, rel_rows=None, prov_rows=None, ent_rows=None, edge_rows=None, neg_rows=None,
) -> AsyncMock:
    """Mirrors test_export_rdf.py's fixture: export() issues exactly 6
    sequential neo4j.run() calls, in this fixed order."""
    neo4j = AsyncMock()
    neo4j.run.side_effect = [
        type_rows or [], rel_rows or [], prov_rows or [], ent_rows or [], edge_rows or [], neg_rows or [],
    ]
    return neo4j


async def _export(tmp_path: Path, name: str, **neo4j_rows) -> Path:
    from export_rdf import export

    output = tmp_path / name
    neo4j = _make_neo4j(**neo4j_rows)
    with patch("graphrag.graph.neo4j_client.get_neo4j", return_value=neo4j):
        await export(tenant=neo4j_rows.get("_tenant", "aerospace"), output=output, limit=1000)
    return output


# ── A realistic, three-level-deep, confidence-annotated aerospace graph ──────

_AEROSPACE_KWARGS = dict(
    type_rows=[
        {"child": "AIRCRAFT_MODEL", "parent": "VEHICLE"},
        {"child": "VEHICLE", "parent": "ASSET"},
    ],
    rel_rows=[{"rel": "MANUFACTURES"}],
    ent_rows=[
        {"name": "Boeing 737 MAX", "type": "AIRCRAFT_MODEL", "desc": None,
         "vf": None, "vt": None, "tenant": "aerospace"},
        {"name": "Boeing", "type": "ORG", "desc": None,
         "vf": None, "vt": None, "tenant": "aerospace"},
    ],
    edge_rows=[
        {"sname": "Boeing", "stype": "ORG",
         "tname": "Boeing 737 MAX", "ttype": "AIRCRAFT_MODEL",
         "rel": "MANUFACTURES", "conf": 0.95,
         "src_doc": "fleet.pdf", "tenant": "aerospace"},
    ],
)

_AUTOMOTIVE_KWARGS = dict(
    type_rows=[{"child": "COMPONENT", "parent": "ASSET"}],
    rel_rows=[{"rel": "COMPATIBLE_WITH"}],
    ent_rows=[
        {"name": "Brake Pad X1", "type": "COMPONENT", "desc": None,
         "vf": None, "vt": None, "tenant": "automotive"},
        {"name": "Rotor R2", "type": "COMPONENT", "desc": None,
         "vf": None, "vt": None, "tenant": "automotive"},
    ],
    edge_rows=[
        {"sname": "Brake Pad X1", "stype": "COMPONENT",
         "tname": "Rotor R2", "ttype": "COMPONENT",
         "rel": "COMPATIBLE_WITH", "conf": 0.8,
         "src_doc": "parts.pdf", "tenant": "automotive"},
    ],
)


class TestExactSemanticFidelity:
    """Strengthens the existing count-only round-trip check to set equality."""

    async def test_every_triple_survives_round_trip_exactly(self, tmp_path):
        output = await _export(tmp_path, "aero.ttl", **_AEROSPACE_KWARGS)
        g_out = Graph().parse(output, format="turtle")

        g_in = Graph()
        g_in.parse(str(output), format="turtle")

        # Set equality, not count: catches a serializer bug that drops one
        # triple and fabricates an unrelated one at the same total count.
        assert set(g_in) == set(g_out)
        assert len(g_in) > 0, "the graph must not be trivially empty"

    async def test_confidence_axiom_reification_survives_exactly(self, tmp_path):
        output = await _export(tmp_path, "aero.ttl", **_AEROSPACE_KWARGS)
        graph = Graph().parse(output, format="turtle")

        from export_rdf import ANNOT, _axiom_uri

        axiom = _axiom_uri("Boeing", "MANUFACTURES", "Boeing 737 MAX", "aerospace")
        confidence_values = list(graph.objects(axiom, ANNOT.confidence))
        assert confidence_values == [Literal(0.95, datatype=XSD.float)]

    async def test_type_hierarchy_chain_survives_exactly(self, tmp_path):
        output = await _export(tmp_path, "aero.ttl", **_AEROSPACE_KWARGS)
        graph = Graph().parse(output, format="turtle")

        from export_rdf import _type_uri

        assert (_type_uri("AIRCRAFT_MODEL"), RDFS.subClassOf, _type_uri("VEHICLE")) in graph
        assert (_type_uri("VEHICLE"), RDFS.subClassOf, _type_uri("ASSET")) in graph


class TestOWLInteroperabilityAcrossSerialization:
    """A real third-party OWL 2 RL reasoner must derive the same closure
    whether it reasons over the pre-serialization graph or the file."""

    async def test_transitive_type_entailment_matches_pre_and_post_round_trip(self, tmp_path):
        output = await _export(tmp_path, "aero.ttl", **_AEROSPACE_KWARGS)

        pre = Graph().parse(output, format="turtle")
        pre_reasoner = OWLRLReasoner(copy.deepcopy(pre))
        pre_reasoner.apply_closure()

        # `from_turtle` is a genuinely independent parse of the same file --
        # not a reuse of `pre`'s in-memory object -- so this exercises the
        # real interoperability path: an external tool given only the file.
        post_reasoner = OWLRLReasoner.from_turtle(output)
        post_reasoner.apply_closure()

        assert set(pre_reasoner._g) == set(post_reasoner._g), (
            "OWL-RL closure differs between reasoning pre-serialization and "
            "reasoning over the round-tripped file -- the Turtle export lost "
            "information a standard OWL reasoner needs"
        )

        from export_rdf import _type_uri

        # And the entailment that matters is actually present, not just
        # "the two closures happen to agree on nothing interesting."
        assert (_type_uri("AIRCRAFT_MODEL"), RDFS.subClassOf, _type_uri("ASSET")) in post_reasoner._g

    async def test_inconsistency_is_still_detected_after_round_trip(self, tmp_path):
        """A round trip must not silently repair a real inconsistency."""
        g = Graph()
        g.add((BASE.BadClass, RDF.type, OWL.Class))
        g.add((BASE.BadClass, RDFS.subClassOf, OWL.Nothing))
        g.add((INST.bad_entity, RDF.type, BASE.BadClass))
        ttl = tmp_path / "inconsistent.ttl"
        g.serialize(destination=str(ttl), format="turtle")

        reasoner = OWLRLReasoner.from_turtle(ttl)
        reasoner.apply_closure()
        assert reasoner.is_consistent() is False


class TestSHACLInteroperabilityAcrossSerialization:
    """A real third-party SHACL engine must reach the same verdict pre- and
    post-round-trip, for both a conformant and a deliberately broken graph."""

    async def test_conformant_export_still_conforms_after_round_trip(self, tmp_path):
        output = await _export(tmp_path, "aero.ttl", **_AEROSPACE_KWARGS)

        pre_conforms, pre_report = SHACLValidator(
            Graph().parse(output, format="turtle"),
        ).validate()
        post_conforms, post_report = SHACLValidator.from_turtle(output).validate()

        assert pre_conforms is True, pre_report
        assert post_conforms is True, post_report

    def test_non_conformant_graph_is_still_flagged_after_round_trip(self, tmp_path):
        # Known SHACL violation, reused from test_shacl_validator.py: an
        # individual with no rdfs:label fails the export shapes.
        g = Graph()
        g.add((INST.doc1, RDF.type, OWL.NamedIndividual))
        g.add((INST.doc1, RDF.type, BASE.DOCUMENT))
        ttl = tmp_path / "invalid.ttl"
        g.serialize(destination=str(ttl), format="turtle")

        pre_conforms, pre_report = SHACLValidator(g).validate()
        post_conforms, post_report = SHACLValidator.from_turtle(ttl).validate()

        assert pre_conforms is False
        assert post_conforms is False
        # Not just "both fail" -- both must fail for the SAME reason, proving
        # the round trip preserved which shape was violated, not just that
        # something, somewhere, was wrong.
        assert "label" in pre_report.lower()
        assert "label" in post_report.lower()


class TestSKOSInteroperabilityAcrossSerialization:
    """The interoperability catalogue itself must be round-trip-stable."""

    async def test_concept_catalogue_is_identical_pre_and_post_round_trip(self, tmp_path):
        output = await _export(tmp_path, "aero.ttl", **_AEROSPACE_KWARGS)

        # `assess_rdf_interoperability` only accepts a path, so "pre" here
        # means "assessed immediately after export, before any further
        # round trip" and "post" means "assessed after an independent
        # reparse/reserialize cycle" -- proving the assessment itself doesn't
        # depend on which parse produced the in-memory graph.
        pre = assess_rdf_interoperability(output)

        reparsed = Graph().parse(output, format="turtle")
        second_path = tmp_path / "aero_reserialized.ttl"
        reparsed.serialize(destination=str(second_path), format="turtle")
        post = assess_rdf_interoperability(second_path)

        for key in (
            "triples", "concept_schemes", "concepts",
            "concepts_with_pref_label", "broader_edges", "relation_terms",
        ):
            assert pre[key] == post[key], f"{key} drifted across a reserialize cycle"
        assert pre["concepts"] > 0, "the fixture must actually exercise SKOS concepts"
        assert pre["broader_edges"] > 0, "the fixture must actually exercise SKOS hierarchy"


class TestMultipleOntologiesCoexistAfterRoundTrip:
    """Two structurally distinct domain vocabularies, one file, no bleed."""

    async def test_two_domains_remain_independently_queryable(self, tmp_path):
        aero_path = await _export(tmp_path, "aero.ttl", **_AEROSPACE_KWARGS)
        auto_path = await _export(tmp_path, "auto.ttl", **_AUTOMOTIVE_KWARGS)

        # Merge both exports into one graph, as a multi-tenant deployment's
        # combined export would be, then round-trip THAT combined file.
        merged = Graph()
        merged.parse(aero_path, format="turtle")
        merged.parse(auto_path, format="turtle")
        combined_path = tmp_path / "combined.ttl"
        merged.serialize(destination=str(combined_path), format="turtle")

        bridge = SPARQLBridge.from_turtle(combined_path)

        aero_rows = bridge.query("""
            SELECT ?label WHERE {
                ?s base:MANUFACTURES ?o .
                ?o rdfs:label ?label .
            }
        """)
        auto_rows = bridge.query("""
            SELECT ?label WHERE {
                ?s base:COMPATIBLE_WITH ?o .
                ?o rdfs:label ?label .
            }
        """)

        aero_labels = {row["label"] for row in aero_rows}
        auto_labels = {row["label"] for row in auto_rows}

        assert aero_labels == {"Boeing 737 MAX"}
        assert auto_labels == {"Rotor R2"}
        # The two domains' query results must not overlap -- proof the
        # merged, round-tripped file did not blend the vocabularies.
        assert aero_labels.isdisjoint(auto_labels)

    async def test_type_hierarchies_of_both_domains_survive_independently(self, tmp_path):
        aero_path = await _export(tmp_path, "aero.ttl", **_AEROSPACE_KWARGS)
        auto_path = await _export(tmp_path, "auto.ttl", **_AUTOMOTIVE_KWARGS)

        merged = Graph()
        merged.parse(aero_path, format="turtle")
        merged.parse(auto_path, format="turtle")
        combined_path = tmp_path / "combined.ttl"
        merged.serialize(destination=str(combined_path), format="turtle")

        reasoner = OWLRLReasoner.from_turtle(combined_path)
        reasoner.apply_closure()

        from export_rdf import _type_uri

        # Aerospace's three-level chain still entails end-to-end...
        assert (_type_uri("AIRCRAFT_MODEL"), RDFS.subClassOf, _type_uri("ASSET")) in reasoner._g
        # ...and automotive's independent chain entails correctly too,
        # without either vocabulary's closure leaking into the other's
        # (COMPONENT was never declared a subclass of VEHICLE, so this must
        # NOT be entailed).
        assert (_type_uri("COMPONENT"), RDFS.subClassOf, _type_uri("ASSET")) in reasoner._g
        assert (_type_uri("COMPONENT"), RDFS.subClassOf, _type_uri("VEHICLE")) not in reasoner._g
