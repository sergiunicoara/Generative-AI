"""Direct tests for the SPARQL result/request helpers, the PROV-O vocabulary layer, and the new
/validate and /reason routes. These modules were previously covered only through other tests."""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, HTTPException
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes.kg import knowledge as kg_knowledge
from graphrag.graph import sparql_results as sr
from graphrag.provenance import prov_o

# ── sparql_results ────────────────────────────────────────────────────────────


class _FakeRequest:
    def __init__(self, content_type: str | None, body: bytes):
        self.headers = {"content-type": content_type} if content_type is not None else {}
        self._body = body

    async def body(self) -> bytes:
        return self._body


@pytest.mark.parametrize("accept,expected", [
    ("", sr.LEGACY_CONTENT_TYPE),
    ("*/*", sr.LEGACY_CONTENT_TYPE),
    ("application/json", sr.LEGACY_CONTENT_TYPE),
    ("application/sparql-results+json", sr.SPARQL_RESULTS_JSON),
    ("text/html, application/sparql-results+xml;q=0.9", sr.SPARQL_RESULTS_XML),
    ("text/csv", sr.SPARQL_RESULTS_CSV),
])
def test_accept_negotiation_picks_the_standard_type_or_the_legacy_default(accept, expected):
    assert sr.negotiate_accept(accept) == expected


async def test_json_body_carries_query_and_namespaces():
    body = json.dumps({"query": "SELECT * WHERE {?s ?p ?o}", "namespaces": {"ex": "http://e/"}}).encode()
    for ct in ("application/json; charset=utf-8", "", None):
        assert await sr.parse_sparql_query_request(_FakeRequest(ct, body)) == (
            "SELECT * WHERE {?s ?p ?o}", {"ex": "http://e/"})


async def test_sparql_query_and_form_bodies_have_no_namespaces():
    assert await sr.parse_sparql_query_request(
        _FakeRequest("application/sparql-query", b"SELECT 1 WHERE {}")) == ("SELECT 1 WHERE {}", {})
    assert await sr.parse_sparql_query_request(
        _FakeRequest("application/x-www-form-urlencoded", b"query=SELECT+%3Fs+WHERE+%7B%3Fs+%3Fp+%3Fo%7D")
    ) == ("SELECT ?s WHERE {?s ?p ?o}", {})


@pytest.mark.parametrize("ct,body,status", [
    ("application/json", b"{not json", 400),
    ("application/json", b"{}", 400),
    ("application/json", b'{"query": "x", "namespaces": ["a"]}', 400),
    ("application/sparql-query", b"   ", 400),
    ("application/x-www-form-urlencoded", b"other=1", 400),
    ("text/plain", b"SELECT 1", 415),
])
async def test_malformed_or_unsupported_requests_are_rejected_with_the_right_status(ct, body, status):
    with pytest.raises(HTTPException) as err:
        await sr.parse_sparql_query_request(_FakeRequest(ct, body))
    assert err.value.status_code == status


def test_results_serialise_to_each_standard_format_and_reject_others():
    g = Graph()
    g.add((URIRef("http://e/a"), RDFS.label, Literal("A")))
    result = g.query("SELECT ?s ?l WHERE { ?s <http://www.w3.org/2000/01/rdf-schema#label> ?l }")
    as_json = json.loads(sr.serialize_typed_result(result, sr.SPARQL_RESULTS_JSON))
    assert as_json["results"]["bindings"][0]["l"]["value"] == "A"
    assert b"<sparql" in sr.serialize_typed_result(result, sr.SPARQL_RESULTS_XML)
    assert sr.serialize_typed_result(result, sr.SPARQL_RESULTS_CSV).splitlines()[0] == b"s,l"
    with pytest.raises(ValueError):
        sr.serialize_typed_result(result, "text/html")


# ── prov_o ────────────────────────────────────────────────────────────────────


def test_uris_are_tenant_scoped_and_escape_path_separators():
    assert prov_o.document_uri("d1", "acme") != prov_o.document_uri("d1", "other")
    assert prov_o.document_uri("d1", "acme") != prov_o.chunk_uri("d1", "acme")
    tricky = prov_o.document_uri("a/b c", "ac/me")
    assert str(tricky).endswith("document/ac%2Fme/a%2Fb%20c")


def test_activity_records_times_status_tenant_and_normalises_a_trailing_z():
    g = Graph()
    act = prov_o.activity_uri("ingest", "run1", "acme")
    prov_o.add_activity(g, act, label="Ingest", tenant="acme", started_at="2026-01-01T00:00:00Z",
                        ended_at="2026-01-01T01:00:00+00:00", status="done")
    assert (act, RDF.type, prov_o.PROV.Activity) in g
    start = next(g.objects(act, prov_o.PROV.startedAtTime))
    assert start.datatype == XSD.dateTime and str(start) == "2026-01-01T00:00:00+00:00"
    assert str(next(g.objects(act, prov_o.ANNOT.status))) == "done"
    assert str(next(g.objects(act, prov_o.ANNOT.tenant))) == "acme"


def test_activity_without_times_or_status_adds_none_of_them():
    g = Graph()
    act = prov_o.activity_uri("ingest", "run2", "acme")
    prov_o.add_activity(g, act, label="Ingest", tenant="acme")
    assert not list(g.objects(act, prov_o.PROV.startedAtTime))
    assert not list(g.objects(act, prov_o.ANNOT.status))


def test_software_agent_is_also_typed_agent_and_a_plain_agent_is_not_a_software_agent():
    g = Graph()
    bot, person = prov_o.agent_uri("llm", "m1", "acme"), prov_o.agent_uri("user", "u1", "acme")
    prov_o.add_agent(g, bot, label="model")
    prov_o.add_agent(g, person, label="user", software=False)
    assert {(bot, RDF.type, prov_o.PROV.SoftwareAgent), (bot, RDF.type, prov_o.PROV.Agent)} <= set(g)
    assert (person, RDF.type, prov_o.PROV.SoftwareAgent) not in g and (person, RDF.type, prov_o.PROV.Agent) in g


def test_relation_helpers_write_the_expected_prov_triples():
    g = Graph()
    doc, chunk = prov_o.document_uri("d", "t"), prov_o.chunk_uri("c", "t")
    act, agent = prov_o.activity_uri("ingest", "r", "t"), prov_o.agent_uri("llm", "m", "t")
    prov_o.add_used(g, act, doc)
    prov_o.add_generated(g, chunk, act)
    prov_o.add_derived(g, chunk, doc)
    prov_o.add_specialization(g, chunk, doc)
    prov_o.add_association(g, act, agent)
    P = prov_o.PROV
    assert {(act, P.used, doc), (chunk, P.wasGeneratedBy, act), (act, P.generated, chunk),
            (chunk, P.wasDerivedFrom, doc), (chunk, P.specializationOf, doc),
            (act, P.wasAssociatedWith, agent)} <= set(g)


def test_entity_label_type_and_tenant_are_optional():
    g = Graph()
    bare, full = prov_o.entity_uri("e1", "t"), prov_o.entity_uri("e2", "t")
    prov_o.add_entity(g, bare)
    prov_o.add_entity(g, full, label="E", entity_type=prov_o.PROV.Collection, tenant="t")
    assert list(g.predicate_objects(bare)) == [(RDF.type, prov_o.PROV.Entity)]
    assert (full, RDF.type, prov_o.PROV.Collection) in g and (full, RDFS.label, Literal("E")) in g


# ── POST /validate and /reason ────────────────────────────────────────────────


def _client(tenant="acme", scope="read") -> TestClient:
    app = FastAPI()
    app.include_router(kg_knowledge.router)
    app.dependency_overrides[get_current_user] = lambda: {"scope": scope, "sub": "t", "tenant": tenant}
    return TestClient(app)


def _export(tmp_path, monkeypatch, ttl: str, tenant="acme"):
    monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))
    (tmp_path / tenant).mkdir()
    (tmp_path / tenant / "graph_export.ttl").write_text(ttl, encoding="utf-8")


_UNLABELLED = ("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
               "<http://e/x> a owl:NamedIndividual .\n")


def test_validate_reports_a_violation_for_an_unlabelled_entity(tmp_path, monkeypatch):
    _export(tmp_path, monkeypatch, _UNLABELLED)
    body = _client().post("/validate", json={}).json()
    assert body["conforms"] is False
    assert body["counts"]["violations"] >= 1
    assert any("label" in r["message"] for r in body["results"])


def test_validate_conforms_on_an_empty_export_and_truncates_results_on_request(tmp_path, monkeypatch):
    _export(tmp_path, monkeypatch, "@prefix ex: <http://e/> .\n")
    assert _client().post("/validate", json={}).json()["conforms"] is True


def test_validate_truncation_flag_follows_the_limit(tmp_path, monkeypatch):
    _export(tmp_path, monkeypatch, _UNLABELLED)
    body = _client().post("/validate", json={"limit": 0}).json()
    assert body["results"] == [] and body["truncated"] is True and body["counts"]["violations"] >= 1


@pytest.mark.parametrize("shapes", ["../export", "nope", "export.shapes.ttl"])
def test_validate_only_accepts_shapes_from_the_fixed_list(tmp_path, monkeypatch, shapes):
    _export(tmp_path, monkeypatch, _UNLABELLED)
    response = _client().post("/validate", json={"shapes": shapes})
    assert response.status_code == 400 and "available" in response.json()["detail"]


def test_validate_accepts_a_named_shapes_file(tmp_path, monkeypatch):
    _export(tmp_path, monkeypatch, "@prefix ex: <http://e/> .\n")
    assert _client().post("/validate", json={"shapes": "export"}).json()["conforms"] is True


def test_validate_and_reason_return_404_without_an_export_and_403_for_a_bad_tenant(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))
    assert _client().post("/validate", json={}).status_code == 404
    assert _client().post("/reason", json={}).status_code == 404
    assert _client(tenant="../x").post("/validate", json={}).status_code == 403


def test_reason_materialises_the_subclass_closure_without_touching_the_export(tmp_path, monkeypatch):
    ttl = ("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
           "@prefix e: <http://e/> .\n"
           "e:Dog rdfs:subClassOf e:Animal .\ne:Animal rdfs:subClassOf e:Thing .\n"
           "e:rex a e:Dog .\n")
    _export(tmp_path, monkeypatch, ttl)
    before = (tmp_path / "acme" / "graph_export.ttl").read_text(encoding="utf-8")
    body = _client().post("/reason", json={"limit": 5000}).json()
    inferred = {tuple(t) for t in body["inferred"]}
    assert ("<http://e/rex>", "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>", "<http://e/Animal>") in inferred
    assert ("<http://e/Dog>", "<http://www.w3.org/2000/01/rdf-schema#subClassOf>", "<http://e/Thing>") in inferred
    assert body["new_triples"] >= 2 and body["consistent"] is True
    assert (tmp_path / "acme" / "graph_export.ttl").read_text(encoding="utf-8") == before


def test_reason_flags_an_inconsistent_graph(tmp_path, monkeypatch):
    ttl = ("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
           "@prefix e: <http://e/> .\n"
           "e:Impossible rdfs:subClassOf owl:Nothing .\ne:x a e:Impossible .\n")
    _export(tmp_path, monkeypatch, ttl)
    body = _client().post("/reason", json={}).json()
    assert body["consistent"] is False and body["inconsistencies"]


def test_reason_flags_a_disjointness_violation_with_its_message(tmp_path, monkeypatch):
    ttl = ("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n@prefix e: <http://e/> .\n"
           "e:A owl:disjointWith e:B .\ne:x a e:A, e:B .\n")
    _export(tmp_path, monkeypatch, ttl)
    body = _client().post("/reason", json={}).json()
    assert body["consistent"] is False and any("Disjoint classes" in m for m in body["inconsistencies"])


def test_oversized_export_is_refused_with_413(tmp_path, monkeypatch):
    _export(tmp_path, monkeypatch, "@prefix e: <http://e/> .\ne:a e:p e:b .\ne:c e:p e:d .\n")
    monkeypatch.setattr(kg_knowledge, "_MAX_RDF_CHECK_TRIPLES", 1)
    assert _client().post("/validate", json={}).status_code == 413
    assert _client().post("/reason", json={}).status_code == 413
