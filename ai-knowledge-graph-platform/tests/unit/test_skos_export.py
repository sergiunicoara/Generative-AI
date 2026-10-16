"""SKOS concept-scheme export from a domain ontology YAML."""
from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Graph, URIRef
from rdflib.namespace import SKOS

from graphrag.graph.skos_export import export_skos, ontology_to_skos

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://graphrag.example.com/taxonomy/demo#"
SCHEME = URIRef("https://graphrag.example.com/taxonomy/demo")


def _c(name: str) -> URIRef:
    return URIRef(BASE + name)


ONTOLOGY = {
    "ontology": {"id": "demo", "version": "2.0"},
    "type_hierarchy": [["SUPPLIER", "ORG"], ["CLIENT_ORG", "ORG"], ["ORG", "THING"]],
    "vocabulary": {"SUPPLIER": {"definition": "A company that supplies parts.", "synonyms": ["vendor", "source"]}},
}


def test_hierarchy_becomes_broader_and_narrower_and_only_the_root_is_a_top_concept():
    g = ontology_to_skos(ONTOLOGY)
    assert (_c("SUPPLIER"), SKOS.broader, _c("ORG")) in g and (_c("ORG"), SKOS.narrower, _c("SUPPLIER")) in g
    assert (_c("ORG"), SKOS.broader, _c("THING")) in g
    assert set(g.objects(SCHEME, SKOS.hasTopConcept)) == {_c("THING")}
    assert (_c("THING"), SKOS.topConceptOf, SCHEME) in g and (_c("ORG"), SKOS.topConceptOf, SCHEME) not in g


def test_every_concept_is_in_the_scheme_and_has_a_readable_label():
    g = ontology_to_skos(ONTOLOGY)
    concepts = set(g.subjects(rdf_type := URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), SKOS.Concept))
    assert concepts == {_c(n) for n in ("SUPPLIER", "CLIENT_ORG", "ORG", "THING")}
    assert all((c, SKOS.inScheme, SCHEME) in g for c in concepts)
    assert {str(o) for o in g.objects(_c("CLIENT_ORG"), SKOS.prefLabel)} == {"Client org"}
    assert rdf_type


def test_vocabulary_adds_definition_and_synonyms_as_alt_labels():
    g = ontology_to_skos(ONTOLOGY)
    assert {str(o) for o in g.objects(_c("SUPPLIER"), SKOS.definition)} == {"A company that supplies parts."}
    assert {str(o) for o in g.objects(_c("SUPPLIER"), SKOS.altLabel)} == {"vendor", "source"}
    assert not list(g.objects(_c("ORG"), SKOS.altLabel))


def test_equivalence_mappings_are_added_only_when_given_and_must_name_a_known_type():
    assert not list(ontology_to_skos(ONTOLOGY).subject_objects(SKOS.exactMatch))
    g = ontology_to_skos(ONTOLOGY, mappings={"exactMatch": [("supplier", "http://other.example/Vendor")],
                                             "closeMatch": [("ORG", "http://other.example/Company")]})
    assert (_c("SUPPLIER"), SKOS.exactMatch, URIRef("http://other.example/Vendor")) in g
    assert (_c("ORG"), SKOS.closeMatch, URIRef("http://other.example/Company")) in g
    with pytest.raises(ValueError, match="not a type"):
        ontology_to_skos(ONTOLOGY, mappings={"exactMatch": [("NOPE", "http://x/y")]})
    with pytest.raises(ValueError, match="unknown SKOS mapping kind"):
        ontology_to_skos(ONTOLOGY, mappings={"sameAs": [("ORG", "http://x/y")]})


def test_the_shipped_automotive_ontology_exports_and_round_trips(tmp_path):
    out = tmp_path / "automotive.ttl"
    g = export_skos(ROOT / "config" / "ontologies" / "automotive_iatf.yml", out)
    again = Graph().parse(str(out), format="turtle")
    assert len(g) == len(again) > 20
    scheme = URIRef("https://graphrag.example.com/taxonomy/automotive-iatf")
    base = "https://graphrag.example.com/taxonomy/automotive-iatf#"
    assert (URIRef(base + "SUPPLIER"), SKOS.broader, URIRef(base + "ORG")) in again
    assert (scheme, URIRef("http://www.w3.org/2002/07/owl#versionInfo"), None) in again


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_skos(tmp_path / "missing.yml")
