"""IATF evidence view: document ACLs on every read path, effective dates in the as-of view."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import iatf_audit as routes
from graphrag.domains.automotive import audit
from graphrag.domains.automotive import signoff as so
from graphrag.graph.validation.graph_checks import assert_read_only

NOW = datetime(2026, 10, 13, tzinfo=timezone.utc)
CLAUSE = audit.Clause("4.3.2", "CSR", ("CSR-CLIENT",), ("kpi",))


def _doc(doc_id, *, recorded_days_ago, effective_days_ago=None, superseded_by=None):
    return audit.CorpusDocument(
        doc_id, "graph", NOW - timedelta(days=recorded_days_ago), (), ["kpi"], superseded_by, None,
        NOW - timedelta(days=effective_days_ago) if effective_days_ago is not None else None)


def test_effective_date_decides_presence_in_an_as_of_view_and_falls_back_to_recorded():
    # recorded long ago but only in force 10 days ago -> not in force 30 days ago
    doc = _doc("CSR-CLIENT-2023", recorded_days_ago=100, effective_days_ago=10)
    as_of = NOW - timedelta(days=30)
    assert audit.audit_view(catalogue={"standard": "", "mapping_status": "proposed", "max_document_age_days": 999,
                                       "clauses": [CLAUSE]}, docs=[doc], as_of=as_of)["documents_considered"] == 0
    no_effective = _doc("CSR-CLIENT-2023", recorded_days_ago=100)
    assert no_effective.available_from == no_effective.recorded_at


def test_successor_not_yet_in_force_does_not_supersede():
    old = _doc("CSR-CLIENT-2021", recorded_days_ago=400, effective_days_ago=400, superseded_by="CSR-CLIENT-2023")
    new = _doc("CSR-CLIENT-2023", recorded_days_ago=100, effective_days_ago=10)   # loaded early, in force later
    cat = {"standard": "", "mapping_status": "proposed", "max_document_age_days": 999, "clauses": [CLAUSE]}
    item = audit.audit_view(catalogue=cat, docs=[old, new], as_of=NOW - timedelta(days=30))["clauses"][0]
    assert [e["doc_id"] for e in item["evidence"]] == ["CSR-CLIENT-2021"]
    assert item["evidence"][0]["effective_from"] is not None


def test_graph_query_applies_the_shared_fail_closed_acl_predicate():
    assert_read_only(audit.GRAPH_DOCUMENTS)
    for fragment in ("$acl_enabled", "deny_principals", "allow_principals", "acl_state", "requires_group_resolution"):
        assert fragment in audit.GRAPH_DOCUMENTS
    assert "d.valid_from" in audit.GRAPH_DOCUMENTS


async def test_graph_loader_passes_the_callers_principals_and_the_acl_switch():
    from graphrag.enterprise.models import AccessContext

    captured = {}

    class Neo:
        async def run_read(self, cypher, **params):
            captured.update(params)
            return []

    await audit.load_graph_corpus(Neo(), "t", access_context=AccessContext.from_claims({"sub": "u1", "groups": ["qa"]}),
                                  acl_enabled=True)
    assert captured["acl_enabled"] is True and captured["tenant"] == "t"
    assert sorted(captured["acl_principals"]) == ["group:qa", "user:u1"] and captured["acl_groups_resolved"] is True


class _FakeNeo4j:
    async def run(self, cypher, **p):
        if cypher.startswith("CREATE CONSTRAINT"):
            return []
        if cypher is so.HEAD:
            return [{"hash": so.GENESIS, "seq": 0}]
        if cypher is so.LIST:
            return []
        if cypher is so.APPEND:
            return [{"id": p["id"]}]
        raise AssertionError(cypher)


def _client(sub="aud-1"):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"scope": "read write", "sub": sub,
                                                         "tenant": "automotive", "groups": ["quality"]}
    return TestClient(app)


@pytest.fixture
def acl_on():
    with patch.object(routes, "_acl_enforced", return_value=True):
        yield


@pytest.mark.parametrize("method,path,body", [
    ("get", "/audit/iatf/clauses", None),
    ("get", "/audit/iatf/clauses/8.4", None),
    ("post", "/audit/iatf/clauses/8.4/signoff", {"decision": "accepted"}),
    ("get", "/audit/iatf/export", None),
])
def test_file_source_is_refused_on_every_evidence_route_when_acls_are_enforced(acl_on, method, path, body):
    with patch.object(routes, "get_neo4j", return_value=_FakeNeo4j()):
        client = _client()
        response = client.post(path, json=body) if method == "post" else client.get(path)
    assert response.status_code == 403 and "ACL" in response.json()["detail"]


def test_graph_source_receives_the_callers_access_context(acl_on):
    seen = {}

    async def fake_loader(neo4j, tenant, metadata_dir, *, access_context=None, acl_enabled=None):
        seen.update(tenant=tenant, principals=access_context.principals)
        return []

    cfg = {"tenants": {"automotive": {"source": "graph", "metadata_dir": "data/metadata/automotive"}}}
    with patch.object(routes, "get_neo4j", return_value=_FakeNeo4j()), \
            patch.object(routes.audit, "load_graph_corpus", fake_loader), \
            patch.object(routes, "get_settings") as settings:
        settings.return_value.iatf_audit = cfg
        settings.return_value.access_control = {"enabled": True}
        response = _client().get("/audit/iatf/clauses")
    assert response.status_code == 200
    assert seen == {"tenant": "automotive", "principals": ["group:quality", "user:aud-1"]}


def test_metrics_are_not_set_from_a_per_user_filtered_view(acl_on):
    cfg = {"tenants": {"automotive": {"source": "graph"}}}

    async def fake_loader(*_a, **_k):
        return []

    with patch.object(routes, "get_neo4j", return_value=_FakeNeo4j()), \
            patch.object(routes.audit, "load_graph_corpus", fake_loader), \
            patch.object(routes, "get_settings") as settings, \
            patch.object(routes.metrics, "record_view") as record:
        settings.return_value.iatf_audit = cfg
        _client().get("/audit/iatf/clauses")
    record.assert_not_called()
