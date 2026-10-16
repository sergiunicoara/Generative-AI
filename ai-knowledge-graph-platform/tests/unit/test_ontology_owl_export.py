"""OWL and SHACL generated from the domain ontology YAMLs, run with real rdflib and pyshacl (no mocks)."""
from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import BNode, Graph, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS

from graphrag.graph.domain_ontology import (
    get_functional_relations,
    get_relation_rules,
    get_type_hierarchy_pairs,
    load_domain_ontology,
)
from graphrag.graph.ontology_owl_export import (
    OntologyNamespaces,
    export_ontology,
    ontology_to_owl,
    ontology_to_shacl,
)
from graphrag.graph.shacl_validator import SHACLValidator

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "ontology" / "from_config"
DOMAINS = ["aerospace_regulatory", "legal_contracts", "marketing_adtech", "pharma_commercial",
           "sustainability_supply_chain", "telecom_oss"]


def _load(name: str) -> dict:
    return load_domain_ontology(ROOT / "config" / "ontologies" / f"{name}.yml")


def _rules_with_both_sides(ontology: dict):
    return {k.upper(): v for k, v in get_relation_rules(ontology).items()
            if v and v.get("domain") and v.get("target")}


def _validate(data: Graph, shapes: Graph, tmp_path: Path):
    path = tmp_path / "shapes.ttl"
    path.write_text(shapes.serialize(format="turtle"), encoding="utf-8")
    return SHACLValidator(data, shapes_path=path).validate_report(target="test")


@pytest.mark.parametrize("name", DOMAINS)
def test_owl_has_every_class_hierarchy_edge_and_relation(name):
    ontology = _load(name)
    ns = OntologyNamespaces(ontology)
    g = ontology_to_owl(ontology)
    assert (ns.iri, RDF.type, OWL.Ontology) in g
    for child, parent in get_type_hierarchy_pairs(ontology):
        assert (ns.cls[child], RDFS.subClassOf, ns.cls[parent]) in g
        assert (ns.cls[parent], RDF.type, OWL.Class) in g
    for relation in get_relation_rules(ontology):
        assert (ns.rel[relation.upper()], RDF.type, OWL.ObjectProperty) in g
    for relation in get_functional_relations(ontology):
        assert (ns.rel[relation], RDF.type, OWL.FunctionalProperty) in g


@pytest.mark.parametrize("name", DOMAINS)
def test_domain_and_range_are_a_class_or_a_union_of_exactly_the_listed_classes(name):
    ontology = _load(name)
    ns = OntologyNamespaces(ontology)
    g = ontology_to_owl(ontology)
    for relation, rule in _rules_with_both_sides(ontology).items():
        for predicate, key in ((RDFS.domain, "domain"), (RDFS.range, "target")):
            expected = {ns.cls[str(n).upper()] for n in ([rule[key]] if isinstance(rule[key], str) else rule[key])}
            value = g.value(ns.rel[relation], predicate)
            if isinstance(value, BNode):
                members = set(Collection(g, g.value(value, OWL.unionOf)))
            else:
                members = {value}
            assert members == expected, (name, relation, key)


def test_inverse_and_transitive_relations_are_declared_from_the_yaml():
    ontology = _load("telecom_oss")
    g = ontology_to_owl(ontology)
    inverses = {(s, o) for s, o in g.subject_objects(OWL.inverseOf)}
    assert inverses, "telecom_oss declares inverse relations"
    assert all(isinstance(s, URIRef) and isinstance(o, URIRef) for s, o in inverses)
    transitive = [s for n in DOMAINS for s in ontology_to_owl(_load(n)).subjects(RDF.type, OWL.TransitiveProperty)]
    assert transitive, "at least one domain declares a transitive relation"


@pytest.mark.parametrize("name", DOMAINS)
def test_a_conforming_instance_passes_and_a_wrong_subject_or_object_type_fails(name, tmp_path):
    ontology = _load(name)
    ns = OntologyNamespaces(ontology)
    owl_graph, shapes = ontology_to_owl(ontology), ontology_to_shacl(ontology)
    functional = set(get_functional_relations(ontology))
    relation, rule = next((r, v) for r, v in _rules_with_both_sides(ontology).items() if r not in functional)
    domain = [rule["domain"]] if isinstance(rule["domain"], str) else rule["domain"]
    target = [rule["target"]] if isinstance(rule["target"], str) else rule["target"]
    stranger = ns.cls["NOT_A_TYPE_OF_THIS_ONTOLOGY"]

    def instance(subject_type, object_type) -> Graph:
        data = Graph()
        data += owl_graph
        s, o = URIRef("http://e/s"), URIRef("http://e/o")
        data.add((s, RDF.type, subject_type))
        data.add((o, RDF.type, object_type))
        data.add((s, ns.rel[relation], o))
        return data

    assert _validate(instance(ns.cls[domain[0].upper()], ns.cls[target[0].upper()]), shapes, tmp_path).conforms
    bad_subject = _validate(instance(stranger, ns.cls[target[0].upper()]), shapes, tmp_path)
    bad_object = _validate(instance(ns.cls[domain[0].upper()], stranger), shapes, tmp_path)
    assert not bad_subject.conforms and not bad_object.conforms
    assert any("subject" in r.message for r in bad_subject.results)
    assert any("object" in r.message for r in bad_object.results)


def test_a_functional_relation_allows_one_value_and_rejects_two(tmp_path):
    ontology = _load("telecom_oss")
    ns = OntologyNamespaces(ontology)
    owl_graph, shapes = ontology_to_owl(ontology), ontology_to_shacl(ontology)
    relation = next(r for r in get_functional_relations(ontology) if r in _rules_with_both_sides(ontology))
    rule = _rules_with_both_sides(ontology)[relation]
    domain = rule["domain"][0].upper() if isinstance(rule["domain"], list) else rule["domain"].upper()
    target = rule["target"][0].upper() if isinstance(rule["target"], list) else rule["target"].upper()

    def with_values(count: int) -> Graph:
        data = Graph()
        data += owl_graph
        s = URIRef("http://e/s")
        data.add((s, RDF.type, ns.cls[domain]))
        for i in range(count):
            o = URIRef(f"http://e/o{i}")
            data.add((o, RDF.type, ns.cls[target]))
            data.add((s, ns.rel[relation], o))
        return data

    assert _validate(with_values(1), shapes, tmp_path).conforms
    two = _validate(with_values(2), shapes, tmp_path)
    assert not two.conforms and two.counts["violations"] >= 1


@pytest.mark.parametrize("name", DOMAINS)
def test_committed_files_match_what_the_generator_produces_today(name, tmp_path):
    ontology = _load(name)
    owl_path, shapes_path = export_ontology(ontology, tmp_path)
    for fresh in (owl_path, shapes_path):
        committed = GENERATED / fresh.name
        assert committed.exists(), f"run scripts/export_ontology_owl.py --all ontology/from_config ({fresh.name})"
        assert Graph().parse(str(fresh), format="turtle").isomorphic(Graph().parse(str(committed), format="turtle")), (
            f"{committed.name} is stale: regenerate with scripts/export_ontology_owl.py --all ontology/from_config")


def test_vocabulary_terms_in_the_shipped_ontologies_are_types_of_that_ontology():
    for name in ("automotive_iatf", "aerospace_regulatory"):
        ontology = _load(name)
        types = {t for pair in get_type_hierarchy_pairs(ontology) for t in pair}
        vocabulary = {k.upper() for k in ontology["vocabulary"]}
        assert vocabulary <= types, (name, sorted(vocabulary - types))
        assert all(v["definition"].strip() for v in ontology["vocabulary"].values())
