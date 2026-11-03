"""Spreadsheet export/import and Jira routes for ontology proposals (offline).

The real ``OntologyProposalService`` runs against a tenant-filtering fake
Neo4j, so tenant isolation and the reviewer identity actually reach the query
parameters.  Jira is exercised through ``httpx.MockTransport``.
"""
from __future__ import annotations

import csv
import io
import json

import httpx
import openpyxl
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes.kg import ontology_proposals as routes
from graphrag.curation.exchange import EXPORT_COLUMNS, MAX_DECISION_ROWS, MAX_UPLOAD_BYTES
from graphrag.curation.jira import JiraClient, JiraConfig

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
TOKEN = "jira-secret-token"


def _proposal(pid: str, tenant: str = "acme", **extra) -> dict:
    return {"id": pid, "tenant": tenant, "kind": "entity_type", "proposed_value": f"V-{pid}",
            "status": "pending", "confidence": 0.9, "conflicts": [], "affected_use_cases": [],
            "reason": "unknown_entity_type", "urgency": "high", **extra}


class FakeNeo4j:
    """Minimal stand-in: filters by the ``tenant`` kwarg like the real Cypher does."""

    def __init__(self, proposals: list[dict]):
        self.proposals = proposals
        self.calls: list[tuple[str, dict]] = []

    async def run(self, query: str, **params):
        self.calls.append((query, params))
        rows = [p for p in self.proposals if p["tenant"] == params["tenant"]]
        if "SET " in query:  # decide
            hit = [p for p in rows if p["id"] == params["proposal_id"] and p["status"] == "pending"]
            return [{"id": p["id"], "kind": p["kind"], "proposed_value": p["proposed_value"],
                     "status": params["status"]} for p in hit]
        if "IN $ids" in query:  # get_many
            return [p for p in rows if p["id"] in params["ids"]]
        return [p for p in rows if not params.get("status") or p["status"] == params["status"]]

    def decide_calls(self) -> list[dict]:
        return [params for query, params in self.calls if "SET " in query]


@pytest.fixture
def claims():
    return {"scope": "read write", "sub": "curator-1", "tenant": "acme"}


@pytest.fixture
def db(monkeypatch) -> FakeNeo4j:
    fake = FakeNeo4j([_proposal("p-1"), _proposal("p-2"), _proposal("other-1", tenant="rival")])
    monkeypatch.setattr(routes, "get_neo4j", lambda: fake)
    return fake


def _client(claims: dict, jira: JiraClient | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: claims
    app.dependency_overrides[routes.get_jira_client] = lambda: jira
    return TestClient(app)


def _csv(rows: list[list[str]]) -> bytes:
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return buf.getvalue().encode()


def _import(client, rows, *, dry_run=None, extra_query=""):
    query = "?" + (f"dry_run={str(dry_run).lower()}&" if dry_run is not None else "") + extra_query
    return client.post("/ontology/proposals/decisions/import" + query, content=_csv(rows),
                       headers={"content-type": "text/csv"})


class TestExportRoutes:
    def test_csv_export_lists_only_the_token_tenant(self, claims, db):
        response = _client(claims).get("/ontology/proposals/export.csv")
        ids = [r[0] for r in list(csv.reader(io.StringIO(response.text.lstrip("﻿"))))[1:]]
        assert sorted(ids) == ["p-1", "p-2"]

    def test_csv_export_content_type_and_attachment(self, claims, db):
        response = _client(claims).get("/ontology/proposals/export.csv")
        assert (response.headers["content-type"].split(";")[0], "attachment" in response.headers["content-disposition"]) == ("text/csv", True)

    def test_xlsx_export_is_a_workbook_with_stable_header(self, claims, db):
        response = _client(claims).get("/ontology/proposals/export.xlsx")
        ws = openpyxl.load_workbook(io.BytesIO(response.content)).active
        assert [c.value for c in ws[1]] == list(EXPORT_COLUMNS)

    def test_xlsx_export_mime_type(self, claims, db):
        assert _client(claims).get("/ontology/proposals/export.xlsx").headers["content-type"] == XLSX_MIME

    def test_export_requires_read_scope(self, claims, db):
        claims["scope"] = "write"
        assert _client(claims).get("/ontology/proposals/export.csv").status_code == 403

    def test_export_passes_token_tenant_to_the_query(self, claims, db):
        _client(claims).get("/ontology/proposals/export.csv?tenant=rival")
        assert db.calls[0][1]["tenant"] == "acme"

    def test_export_formula_values_are_escaped(self, claims, db):
        db.proposals.append(_proposal("p-3", proposed_value="=cmd|' /C calc'!A0"))
        text = _client(claims).get("/ontology/proposals/export.csv").text
        assert "'=cmd" in text

    def test_export_limit_is_capped(self, claims, db):
        assert _client(claims).get(f"/ontology/proposals/export.csv?limit={MAX_DECISION_ROWS + 1}").status_code == 422

    def test_existing_json_export_route_is_still_registered(self):
        paths = {route.path for route in routes.router.routes}
        assert {"/ontology/proposals/export", "/ontology/proposals/export.csv"} <= paths


class TestImportRoute:
    def test_dry_run_is_the_default_and_decides_nothing(self, claims, db):
        response = _import(_client(claims), [["proposal_id", "action"], ["p-1", "approve"]])
        assert (response.status_code, response.json()["dry_run"], db.decide_calls()) == (200, True, [])

    def test_dry_run_preview_marks_pending_unknown_and_foreign_rows(self, claims, db):
        rows = [["proposal_id", "action"], ["p-1", "approve"], ["nope", "approve"], ["other-1", "approve"]]
        items = _import(_client(claims), rows).json()["items"]
        assert [i["outcome"] for i in items] == ["will_apply", "not_found", "not_found"]

    def test_dry_run_reports_non_pending_status(self, claims, db):
        db.proposals.append(_proposal("done", status="approved"))
        items = _import(_client(claims), [["proposal_id", "action"], ["done", "defer"]]).json()["items"]
        assert items[0]["outcome"] == "not_pending"

    def test_apply_calls_decide_with_token_subject_not_a_body_reviewer(self, claims, db):
        rows = [["proposal_id", "action", "reviewed_by"], ["p-1", "approve", "mallory"]]
        _import(_client(claims), rows, dry_run=False, extra_query="reviewed_by=mallory")
        assert db.decide_calls()[0]["reviewed_by"] == "curator-1"

    def test_apply_uses_token_tenant(self, claims, db):
        _import(_client(claims), [["proposal_id", "action"], ["p-1", "approve"]], dry_run=False)
        assert db.decide_calls()[0]["tenant"] == "acme"

    def test_apply_cannot_touch_another_tenants_proposal(self, claims, db):
        response = _import(_client(claims), [["proposal_id", "action"], ["other-1", "approve"]], dry_run=False)
        assert response.json()["items"][0]["outcome"] == "error"

    def test_apply_reports_per_row_results(self, claims, db):
        rows = [["proposal_id", "action"], ["p-1", "approve"], ["missing", "approve"]]
        items = _import(_client(claims), rows, dry_run=False).json()["items"]
        assert [i["outcome"] for i in items] == ["applied", "error"]

    def test_apply_passes_edit_value_and_reason(self, claims, db):
        rows = [["proposal_id", "action", "reason", "edited_value"], ["p-1", "edit", "typo", "SUPPLIER"]]
        _import(_client(claims), rows, dry_run=False)
        params = db.decide_calls()[0]
        assert (params["edited_value"], params["reason"], params["status"]) == ("SUPPLIER", "typo", "edited")

    def test_apply_is_refused_when_any_row_is_invalid(self, claims, db):
        rows = [["proposal_id", "action"], ["p-1", "approve"], ["p-2", "explode"]]
        response = _import(_client(claims), rows, dry_run=False)
        assert (response.status_code, db.decide_calls()) == (422, [])

    def test_dry_run_reports_row_errors_with_200(self, claims, db):
        response = _import(_client(claims), [["proposal_id", "action"], ["p-2", "explode"]])
        assert (response.status_code, response.json()["errors"][0]["row"]) == (200, 2)

    def test_import_requires_write_scope(self, claims, db):
        claims["scope"] = "read"
        assert _import(_client(claims), [["proposal_id", "action"]]).status_code == 403

    def test_token_without_subject_is_refused(self, claims, db):
        del claims["sub"]
        assert _import(_client(claims), [["proposal_id", "action"]], dry_run=False).status_code == 403

    def test_xlsx_upload_is_accepted(self, claims, db):
        wb = openpyxl.Workbook()
        wb.active.append(["proposal_id", "action"])
        wb.active.append(["p-1", "defer"])
        out = io.BytesIO()
        wb.save(out)
        response = _client(claims).post("/ontology/proposals/decisions/import?dry_run=false",
                                        content=out.getvalue(), headers={"content-type": XLSX_MIME})
        assert [i["outcome"] for i in response.json()["items"]] == ["applied"]

    def test_unsupported_content_type_is_415(self, claims, db):
        response = _client(claims).post("/ontology/proposals/decisions/import", content=b"{}",
                                        headers={"content-type": "application/json"})
        assert response.status_code == 415

    def test_oversized_upload_is_413(self, claims, db):
        response = _client(claims).post("/ontology/proposals/decisions/import",
                                        content=b"a" * (MAX_UPLOAD_BYTES + 1), headers={"content-type": "text/csv"})
        assert response.status_code == 413

    def test_structural_problem_is_422_with_code(self, claims, db):
        response = _import(_client(claims), [["proposal_id", "reason"], ["p-1", "x"]])
        assert (response.status_code, response.json()["detail"]["code"]) == (422, "missing_columns")

    def test_empty_dry_run_flag_value_is_validated(self, claims, db):
        assert _import(_client(claims), [["proposal_id", "action"]], extra_query="dry_run=maybe").status_code == 422


class TestJiraRoute:
    def _jira(self, handler) -> JiraClient:
        config = JiraConfig(base_url="https://example.atlassian.net", email="bot@example.com",
                            api_token=TOKEN, project_key="ONT")
        return JiraClient(config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    def test_not_configured_is_503_with_code(self, claims, db, monkeypatch):
        for name in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_PROJECT_KEY"):
            monkeypatch.delenv(name, raising=False)
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[get_current_user] = lambda: claims
        response = TestClient(app).post("/ontology/proposals/p-1/jira")
        assert (response.status_code, response.json()["detail"]["code"]) == (503, "jira_not_configured")

    def test_creates_one_issue_and_returns_its_key(self, claims, db):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(201, json={"key": "ONT-9"})

        response = _client(claims, self._jira(handler)).post("/ontology/proposals/p-1/jira")
        assert (response.status_code, response.json()["issue_key"], len(requests)) == (200, "ONT-9", 1)

    def test_issue_payload_describes_the_proposal(self, claims, db):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(201, json={"key": "ONT-9"})

        _client(claims, self._jira(handler)).post("/ontology/proposals/p-1/jira")
        assert "V-p-1" in seen["fields"]["summary"] and seen["fields"]["project"] == {"key": "ONT"}

    def test_other_tenants_proposal_is_404_and_no_ticket_created(self, claims, db):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(1)
            return httpx.Response(201, json={"key": "ONT-9"})

        response = _client(claims, self._jira(handler)).post("/ontology/proposals/other-1/jira")
        assert (response.status_code, requests) == (404, [])

    def test_requires_write_scope(self, claims, db):
        claims["scope"] = "read"
        assert _client(claims, self._jira(lambda r: httpx.Response(201, json={"key": "X-1"}))).post(
            "/ontology/proposals/p-1/jira").status_code == 403

    def test_jira_failure_is_502_without_leaking_token(self, claims, db):
        response = _client(claims, self._jira(lambda r: httpx.Response(401, text=f"nope {TOKEN}"))).post(
            "/ontology/proposals/p-1/jira")
        assert (response.status_code, TOKEN in response.text) == (502, False)

    def test_jira_timeout_is_504(self, claims, db):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        response = _client(claims, self._jira(handler)).post("/ontology/proposals/p-1/jira")
        assert (response.status_code, response.json()["detail"]["code"]) == (504, "jira_timeout")


class TestServiceGetMany:
    async def test_get_many_is_tenant_scoped(self):
        from graphrag.graph.ontology_proposals import OntologyProposalService
        fake = FakeNeo4j([_proposal("p-1"), _proposal("p-1x", tenant="rival")])
        rows = await OntologyProposalService(fake).get_many("acme", ["p-1", "p-1x"])
        assert [r["id"] for r in rows] == ["p-1"]
