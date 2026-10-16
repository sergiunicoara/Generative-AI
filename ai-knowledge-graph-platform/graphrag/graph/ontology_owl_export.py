"""OWL and SHACL generated from a domain ontology YAML (``config/ontologies/*.yml``).

``ontology/models/*.yaml`` has its own compiler for two domains; the other domains only had the lighter YAML
(type hierarchy, relation rules, functional relations, inference rules). This module turns that YAML into:

* an OWL ontology: classes with ``rdfs:subClassOf``; one ``owl:ObjectProperty`` per relation with ``rdfs:domain`` and
  ``rdfs:range`` (a union when the rule lists several types); ``owl:inverseOf``; ``owl:FunctionalProperty`` for the
  functional relations; ``owl:TransitiveProperty`` for inference rules of type ``transitivity``;
* SHACL shapes: the subject of a relation must be an instance of one of its domain types, every object an instance
  of one of its target types, and a functional relation has at most one value per subject.

Only constructs with a clear OWL/SHACL meaning are emitted. ``exclusive_state_pairs`` and the other inference rule
types have no faithful OWL equivalent and are left out rather than approximated.

    python scripts/export_ontology_owl.py config/ontologies/legal_contracts.yml ontology/from_config
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS, XSD

from graphrag.graph.domain_ontology import (
    _as_upper_list,
    get_functional_relations,
    get_inference_rules,
    get_relation_rules,
    get_type_hierarchy_pairs,
)

SH = Namespace("http://www.w3.org/ns/shacl#")
BASE = "https://graphrag.example.com/ontology/"


class OntologyNamespaces:
    def __init__(self, ontology: dict) -> None:
        self.id = str((ontology.get("ontology") or {}).get("id") or "ontology")
        root = f"{BASE}{quote(self.id, safe='')}"
        self.iri = URIRef(root)
        self.cls = Namespace(f"{root}/class#")
        self.rel = Namespace(f"{root}/relation#")
        self.shape = Namespace(f"{root}/shape#")


def _term(ns: Namespace, name: str) -> URIRef:
    return ns[quote(name, safe="")]


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _class_expression(g: Graph, ns: OntologyNamespaces, names: list[str]):
    """A single class, or an anonymous ``owl:unionOf`` class when several are listed."""
    if len(names) == 1:
        return _term(ns.cls, names[0])
    node = BNode()
    g.add((node, RDF.type, OWL.Class))
    g.add((node, OWL.unionOf, Collection(g, BNode(), [_term(ns.cls, n) for n in names]).uri))
    return node


def ontology_to_owl(ontology: dict) -> Graph:
    ns = OntologyNamespaces(ontology)
    g = Graph()
    g.bind("owl", OWL)
    g.bind("cls", ns.cls)
    g.bind("rel", ns.rel)
    g.add((ns.iri, RDF.type, OWL.Ontology))
    version = (ontology.get("ontology") or {}).get("version")
    if version:
        g.add((ns.iri, OWL.versionInfo, Literal(str(version))))

    rules = get_relation_rules(ontology)
    classes: set[str] = set()
    for child, parent in get_type_hierarchy_pairs(ontology):
        classes |= {child, parent}
        g.add((_term(ns.cls, child), RDFS.subClassOf, _term(ns.cls, parent)))
    relations: set[str] = set(map(str.upper, rules)) | set(get_functional_relations(ontology))
    for name, rule in rules.items():
        for side in ("domain", "target"):
            classes |= set(_as_upper_list((rule or {}).get(side)))
        inverse = (rule or {}).get("inverse")
        if inverse:
            relations.add(str(inverse).upper())
    for name in sorted(classes):
        g.add((_term(ns.cls, name), RDF.type, OWL.Class))
        g.add((_term(ns.cls, name), RDFS.label, Literal(_label(name), lang="en")))

    for name, rule in rules.items():
        prop = _term(ns.rel, name.upper())
        rule = rule or {}
        domain, target = _as_upper_list(rule.get("domain")), _as_upper_list(rule.get("target"))
        if domain:
            g.add((prop, RDFS.domain, _class_expression(g, ns, domain)))
        if target:
            g.add((prop, RDFS.range, _class_expression(g, ns, target)))
        if rule.get("inverse"):
            g.add((prop, OWL.inverseOf, _term(ns.rel, str(rule["inverse"]).upper())))
    for name in sorted(relations):
        g.add((_term(ns.rel, name), RDF.type, OWL.ObjectProperty))
        g.add((_term(ns.rel, name), RDFS.label, Literal(_label(name), lang="en")))
    for name in get_functional_relations(ontology):
        g.add((_term(ns.rel, name), RDF.type, OWL.FunctionalProperty))
    for rule in get_inference_rules(ontology):
        if rule.get("rule_type") == "transitivity" and rule.get("relation"):
            relation = str(rule["relation"]).upper()
            g.add((_term(ns.rel, relation), RDF.type, OWL.ObjectProperty))
            g.add((_term(ns.rel, relation), RDF.type, OWL.TransitiveProperty))
    return g


def _or_of_classes(g: Graph, ns: OntologyNamespaces, names: list[str]) -> BNode:
    members = []
    for name in names:
        member = BNode()
        g.add((member, SH["class"], _term(ns.cls, name)))
        members.append(member)
    head = BNode()
    Collection(g, head, members)
    return head


def ontology_to_shacl(ontology: dict) -> Graph:
    ns = OntologyNamespaces(ontology)
    g = Graph()
    g.bind("sh", SH)
    g.bind("shape", ns.shape)
    rules = get_relation_rules(ontology)
    functional = set(get_functional_relations(ontology))
    for name in sorted(set(map(str.upper, rules)) | functional):
        rule = next((r for k, r in rules.items() if k.upper() == name), None) or {}
        domain, target = _as_upper_list(rule.get("domain")), _as_upper_list(rule.get("target"))
        if not (domain or target or name in functional):
            continue
        shape = _term(ns.shape, name)
        prop = _term(ns.rel, name)
        g.add((shape, RDF.type, SH.NodeShape))
        g.add((shape, SH.targetSubjectsOf, prop))
        if domain:
            g.add((shape, SH["or"], _or_of_classes(g, ns, domain)))
            g.add((shape, SH.message, Literal(f"The subject of {name} must be one of: {', '.join(domain)}.")))
        if target or name in functional:
            pshape = BNode()
            g.add((shape, SH.property, pshape))
            g.add((pshape, SH.path, prop))
            if target:
                g.add((pshape, SH["or"], _or_of_classes(g, ns, target)))
                g.add((pshape, SH.message, Literal(f"Every object of {name} must be one of: {', '.join(target)}.")))
            if name in functional:
                g.add((pshape, SH.maxCount, Literal(1, datatype=XSD.integer)))
    return g


def export_ontology(ontology: dict, out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``<id>.owl.ttl`` and ``<id>.shapes.ttl`` under ``out_dir`` and return both paths."""
    ns = OntologyNamespaces(ontology)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = "# DO NOT EDIT: generated from the domain ontology YAML by graphrag/graph/ontology_owl_export.py\n"
    owl_path, shapes_path = out / f"{ns.id}.owl.ttl", out / f"{ns.id}.shapes.ttl"
    owl_path.write_text(header + ontology_to_owl(ontology).serialize(format="turtle"), encoding="utf-8")
    shapes_path.write_text(header + ontology_to_shacl(ontology).serialize(format="turtle"), encoding="utf-8")
    return owl_path, shapes_path


__all__ = ["OntologyNamespaces", "export_ontology", "ontology_to_owl", "ontology_to_shacl"]
