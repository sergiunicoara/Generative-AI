"""SPARQL competency questions with asserted result sets, run against the real ontology files.

The cases live in evals/competency/cases.json. Schema-level questions run on the shipped ontology Turtle files;
instance-level questions run on the ontology merged with a small hand-written instance file. Each expected set is
exact, so a wrong query, a changed ontology or changed data fails the case. This proves the queries and the ontology
agree; it does not measure any model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import Graph, Literal, URIRef

from graphrag.graph.sparql_bridge import SPARQLBridge

ROOT = Path(__file__).resolve().parents[2]
SUITE = json.loads((ROOT / "evals" / "competency" / "cases.json").read_text(encoding="utf-8"))
ONTOLOGY = {
    "automotive": ROOT / "ontology" / "automotive" / "automotive-iatf-quality.ttl",
    "energy": ROOT / "ontology" / "energy" / "energy-asset-intelligence.ttl",
}
INSTANCES = ROOT / "evals" / "competency" / "automotive_iatf_instances.ttl"


def _graph(domain: str, level: str) -> Graph:
    g = Graph()
    g.parse(str(ONTOLOGY[domain]), format="turtle")
    if level == "instance":
        g.parse(str(INSTANCES), format="turtle")
    return g


def _answer(g: Graph, sparql: str) -> list[str]:
    rows = SPARQLBridge(g).query(sparql, init_ns=SUITE["prefixes"])
    return sorted(r["v"] for r in rows)


CASES = SUITE["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_competency_question_returns_exactly_the_expected_set(case):
    assert _answer(_graph(case["domain"], case["level"]), case["sparql"]) == sorted(case["expected"])


def test_suite_covers_both_domains_and_both_levels_and_has_unique_ids():
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert {c["domain"] for c in CASES} == {"automotive", "energy"}
    assert {c["level"] for c in CASES} == {"schema", "instance"}


def _case(case_id):
    return next(c for c in CASES if c["id"] == case_id)


def test_current_documents_question_notices_when_a_supersession_link_is_removed():
    case = _case("AUTO-CQ5")
    g = _graph("automotive", "instance")
    supersedes = URIRef("https://example.automotive.demo/ontology#supersedes")
    for triple in list(g.triples((None, supersedes, URIRef("https://example.automotive.demo/instance#R0")))):
        g.remove(triple)
    assert _answer(g, case["sparql"]) != sorted(case["expected"])


def test_open_nonconformity_question_notices_when_the_nonconformity_is_closed():
    case = _case("AUTO-CQ4")
    g = _graph("automotive", "instance")
    closed = URIRef("https://example.automotive.demo/ontology#closedAt")
    g.add((URIRef("https://example.automotive.demo/instance#N1"), closed, Literal("2026-04-01T00:00:00Z")))
    assert _answer(g, case["sparql"]) == []


def test_schema_question_notices_when_a_class_loses_its_parent():
    case = _case("AUTO-CQ1")
    g = _graph("automotive", "schema")
    sub = URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    supplier = URIRef("https://example.automotive.demo/ontology#Supplier")
    for triple in list(g.triples((supplier, sub, None))):
        g.remove(triple)
    assert _answer(g, case["sparql"]) == ["ClientOrg"]
