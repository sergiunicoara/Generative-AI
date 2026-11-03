"""Read-only SPARQL conformance adapters, exercised offline.

The competency suite is the contract; a target is anything that answers SPARQL 1.1 queries. The local rdflib
baseline must pass every case, and a Stardog-shaped endpoint (an httpx MockTransport that executes the query on
a local rdflib graph and answers in SPARQL 1.1 results JSON) must pass through RemoteSPARQLEndpoint with the same
verdicts. No network and no container is involved, so none of this proves a real Stardog behaves the same way.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import httpx
import pytest
from rdflib import Graph, URIRef

from graphrag.graph.sparql_conformance import (
    LocalSPARQLSource,
    build_graph,
    dataset_fingerprint,
    load_suite,
    run_conformance,
    run_local_baseline,
)
from graphrag.graph.triplestore import RemoteSPARQLEndpoint, TripleStoreTarget

ROOT = Path(__file__).resolve().parents[2]
SUITE = load_suite()
_QUERY_FORM = re.compile(r"^\s*(?:PREFIX\s+\S*\s*<[^>]*>\s*)*(SELECT|ASK)\b", re.IGNORECASE)


class FakeStardog:
    """A Stardog-shaped SPARQL 1.1 endpoint over a local graph, recording every request it receives."""

    def __init__(self, graph: Graph, database: str = "kg") -> None:
        self.graph = graph
        self.requests: list[httpx.Request] = []
        self.target = TripleStoreTarget(
            "stardog", "http://stardog.test:5820", database=database,
            client=httpx.AsyncClient(transport=httpx.MockTransport(self._handle)),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        # A non-empty initNs stops rdflib falling back to the prefixes the Turtle files declared, so a query that
        # relies on a prefix it never declared fails here the way it would on a real store.
        result = self.graph.query(request.content.decode("utf-8"), initNs={"zz": "urn:unused#"})
        return httpx.Response(200, json=json.loads(result.serialize(format="json")))


def _failed_ids(report) -> list[str]:
    return sorted(c.case_id for c in report.cases if not c.passed)


@pytest.mark.asyncio
async def test_local_baseline_passes_every_case_in_the_suite():
    report = await run_local_baseline(SUITE)
    assert (report.total, report.passed) == (len(SUITE["cases"]), len(SUITE["cases"]))


@pytest.mark.asyncio
async def test_local_baseline_covers_all_ten_cases():
    assert (await run_local_baseline(SUITE)).total == 10


@pytest.mark.asyncio
async def test_report_names_the_dataset_the_target_is_expected_to_hold():
    report = await run_conformance(
        LocalSPARQLSource(build_graph("automotive", "schema")), suite=SUITE, domain="automotive", level="schema",
    )
    assert report.expected_datasets == [{"domain": "automotive", "level": "schema"}]


@pytest.mark.asyncio
async def test_report_carries_target_label_and_fingerprint_of_the_selected_dataset():
    report = await run_conformance(
        LocalSPARQLSource(build_graph("energy", "schema")), suite=SUITE, domain="energy", level="schema",
        target="local-rdflib",
    )
    assert (report.target, report.dataset_fingerprint) == (
        "local-rdflib", dataset_fingerprint(domain="energy", level="schema"),
    )


@pytest.mark.asyncio
async def test_domain_and_level_filters_select_only_matching_cases():
    report = await run_conformance(
        LocalSPARQLSource(build_graph("energy", "schema")), suite=SUITE, domain="energy", level="schema",
    )
    assert {c.case_id for c in report.cases} == {
        c["id"] for c in SUITE["cases"] if c["domain"] == "energy" and c["level"] == "schema"
    }


def test_fingerprint_is_stable_across_calls():
    assert dataset_fingerprint() == dataset_fingerprint()


def test_fingerprint_differs_between_datasets():
    assert dataset_fingerprint(domain="automotive", level="schema") != dataset_fingerprint(
        domain="automotive", level="instance"
    )


def _copy_inputs(tmp_path: Path) -> Path:
    for rel in ("evals/competency", "ontology/automotive", "ontology/energy"):
        shutil.copytree(ROOT / rel, tmp_path / rel)
    return tmp_path


def test_fingerprint_is_the_same_for_a_byte_identical_copy_of_the_inputs(tmp_path):
    assert dataset_fingerprint(root=_copy_inputs(tmp_path)) == dataset_fingerprint()


def test_fingerprint_changes_when_an_ontology_file_changes(tmp_path):
    root = _copy_inputs(tmp_path)
    ttl = root / "ontology" / "automotive" / "automotive-iatf-quality.ttl"
    ttl.write_text(ttl.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert dataset_fingerprint(root=root) != dataset_fingerprint()


def test_fingerprint_changes_when_the_instance_file_changes(tmp_path):
    root = _copy_inputs(tmp_path)
    ttl = root / "evals" / "competency" / "automotive_iatf_instances.ttl"
    ttl.write_text(ttl.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert dataset_fingerprint(domain="automotive", level="instance", root=root) != dataset_fingerprint(
        domain="automotive", level="instance"
    )


def test_fingerprint_of_a_dataset_without_the_instance_file_ignores_its_changes(tmp_path):
    root = _copy_inputs(tmp_path)
    before = dataset_fingerprint(domain="energy", level="schema", root=root)
    ttl = root / "evals" / "competency" / "automotive_iatf_instances.ttl"
    ttl.write_text(ttl.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert dataset_fingerprint(domain="energy", level="schema", root=root) == before


def test_fingerprint_changes_when_a_case_changes(tmp_path):
    root = _copy_inputs(tmp_path)
    cases = root / "evals" / "competency" / "cases.json"
    suite = json.loads(cases.read_text(encoding="utf-8"))
    suite["cases"][0]["expected"] = ["Nothing"]
    cases.write_text(json.dumps(suite), encoding="utf-8")
    assert dataset_fingerprint(root=root) != dataset_fingerprint()


def test_fingerprint_ignores_line_ending_differences(tmp_path):
    root = _copy_inputs(tmp_path)
    ttl = root / "ontology" / "energy" / "energy-asset-intelligence.ttl"
    ttl.write_bytes(ttl.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert dataset_fingerprint(root=root) == dataset_fingerprint()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "domain, level",
    [("automotive", "schema"), ("automotive", "instance"), ("energy", "schema")],
)
async def test_fake_stardog_holding_the_right_data_passes_through_the_remote_adapter(domain, level):
    fake = FakeStardog(build_graph(domain, level))
    report = await run_conformance(fake.target, suite=SUITE, domain=domain, level=level, target="stardog:kg")
    assert report.failed == 0 and report.total > 0


@pytest.mark.asyncio
async def test_remote_adapter_also_works_through_a_bare_remote_sparql_endpoint():
    fake = FakeStardog(build_graph("energy", "schema"))
    endpoint = RemoteSPARQLEndpoint(fake.target.query_url, client=fake.target._client)
    report = await run_conformance(endpoint, suite=SUITE, domain="energy", level="schema")
    assert report.failed == 0 and report.total > 0


@pytest.mark.asyncio
async def test_fake_stardog_holding_wrong_data_fails_exactly_the_affected_case():
    g = build_graph("automotive", "schema")
    sub = URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    supplier = URIRef("https://example.automotive.demo/ontology#Supplier")
    for triple in list(g.triples((supplier, sub, None))):
        g.remove(triple)
    report = await run_conformance(FakeStardog(g).target, suite=SUITE, domain="automotive", level="schema")
    assert _failed_ids(report) == ["AUTO-CQ1"]


@pytest.mark.asyncio
async def test_failing_case_reports_expected_and_actual_sets():
    g = build_graph("automotive", "schema")
    sub = URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    supplier = URIRef("https://example.automotive.demo/ontology#Supplier")
    for triple in list(g.triples((supplier, sub, None))):
        g.remove(triple)
    report = await run_conformance(FakeStardog(g).target, suite=SUITE, domain="automotive", level="schema")
    failing = next(c for c in report.cases if c.case_id == "AUTO-CQ1")
    assert (failing.expected, failing.actual) == (["ClientOrg", "Supplier"], ["ClientOrg"])


@pytest.mark.asyncio
async def test_an_empty_store_fails_every_case_with_data():
    fake = FakeStardog(Graph())
    report = await run_conformance(fake.target, suite=SUITE, domain="automotive", level="schema")
    assert report.passed == 0


@pytest.mark.asyncio
async def test_only_select_or_ask_queries_ever_reach_the_wire():
    fake = FakeStardog(build_graph("automotive", "instance"))
    await run_conformance(fake.target, suite=SUITE, domain="automotive", level="instance")
    forms = [_QUERY_FORM.match(r.content.decode("utf-8")) for r in fake.requests]
    assert fake.requests and all(forms)


@pytest.mark.asyncio
async def test_every_request_is_a_post_to_the_stardog_query_path():
    fake = FakeStardog(build_graph("energy", "schema"), database="kg")
    await run_conformance(fake.target, suite=SUITE, domain="energy", level="schema")
    assert {(r.method, r.url.path) for r in fake.requests} == {("POST", "/kg/query")}


@pytest.mark.asyncio
async def test_a_write_query_in_a_suite_is_reported_as_an_error_and_never_sent():
    suite = {
        "prefixes": SUITE["prefixes"],
        "cases": [{
            "id": "BAD-1", "domain": "energy", "level": "schema", "question": "write",
            "sparql": "INSERT DATA { <urn:a> <urn:b> <urn:c> }", "expected": [],
        }],
    }
    fake = FakeStardog(Graph())
    report = await run_conformance(fake.target, suite=suite, domain="energy", level="schema")
    assert (fake.requests, report.cases[0].passed, bool(report.cases[0].error)) == ([], False, True)


@pytest.mark.asyncio
async def test_an_unreachable_endpoint_is_reported_per_case_instead_of_crashing():
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    target = TripleStoreTarget(
        "stardog", "http://stardog.test:5820", database="kg",
        client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)),
    )
    report = await run_conformance(target, suite=SUITE, domain="energy", level="schema")
    assert report.failed == report.total > 0 and all(c.error for c in report.cases)


def test_report_as_dict_is_json_serialisable_and_has_no_credentials():
    import asyncio

    report = asyncio.run(run_local_baseline(SUITE))
    payload = json.loads(json.dumps(report.as_dict()))
    assert set(payload) >= {"target", "dataset_fingerprint", "expected_datasets", "total", "passed", "failed", "cases"}
