"""``kg.knowledge.evaluate@1.0.0`` over the MCP registry and ``POST
/knowledge/evaluate`` over HTTP: tenant binding, entitlements, structured
errors.  Offline: temporary SQLite governance store, synthetic pack."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import knowledge as knowledge_routes
from graphrag.context_graph.knowledge_release import load_cases, load_pack, publish_pack
from graphrag.context_graph.knowledge_service import KnowledgeEvaluationService, set_knowledge_service
from graphrag.domains.energy.governance_store import GovernanceStore
from mcp_server.capabilities import build_registry
from mcp_server.identity import CallerIdentity
from mcp_server.registry import DeniedCapabilityCall

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "config" / "knowledge_packs" / "packaging_labeling.yml"
CASES = ROOT / "evals" / "knowledge_eval" / "packaging_labeling_cases.json"
CONTEXT = {"market": "NORTHLAND", "product_family": "otc_tablets", "workflow": "label_review",
           "customer": "CORNERSHOP", "brand": "OTHERBRAND", "product": "PX-200",
           "as_of": "2026-08-01T00:00:00Z"}
EVIDENCE = [{"source_id": "ARTWORK-PX200-R3"}]
FACTS = {name: {"value": value, "evidence": EVIDENCE} for name, value in {
    "label.batch_code_present": True, "label.min_font_pt": 8, "label.front_claims": [],
    "label.allergen_statement": "Contains lactose", "label.registration_number_present": False,
}.items()}


def _identity(tenant="packaging_demo", groups=None, scopes=("read",)) -> CallerIdentity:
    return CallerIdentity(subject="agent-7", tenant=tenant, scopes=frozenset(scopes), authenticated=True,
                          groups=tuple(groups or ()), groups_resolved=groups is not None)


@pytest.fixture
async def published(tmp_path):
    store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}",
                            blob_root=tmp_path / "blobs")
    await store.open()
    await publish_pack(store, load_pack(PACK), published_by="user:publisher", cases=load_cases(CASES))
    set_knowledge_service(KnowledgeEvaluationService(store))
    yield store
    set_knowledge_service(None)
    await store.close()


class TestCapability:
    async def test_evaluates_for_the_identity_tenant(self, published):
        result = await build_registry().call(
            "kg.knowledge.evaluate@1.0.0", {"domain": "packaging_labeling", "context": CONTEXT, "facts": FACTS},
            _identity())
        assert result["tenant"] == "packaging_demo" and result["overall"] == "compliant"
        assert result["release"]["domain"] == "packaging_labeling"
        assert "provenance_receipt" in result
        assert all(r["item_id"] != "PKG-015" for r in result["results"])

    async def test_resolved_groups_unlock_restricted_knowledge(self, published):
        result = await build_registry().call(
            "kg.knowledge.evaluate@1.0.0", {"domain": "packaging_labeling", "context": CONTEXT, "facts": FACTS},
            _identity(groups=["regulatory-affairs"]))
        restricted = next(r for r in result["results"] if r["item_id"] == "PKG-015")
        assert restricted["outcome"] == "fail" and result["overall"] == "non_compliant"

    async def test_caller_cannot_pick_another_tenant(self, published):
        result = await build_registry().call(
            "kg.knowledge.evaluate@1.0.0",
            {"domain": "packaging_labeling", "context": CONTEXT, "tenant": "other_tenant"}, _identity())
        assert isinstance(result, DeniedCapabilityCall) and result.reason == "tenant_mismatch"

    async def test_scope_and_argument_validation_are_structured(self, published):
        registry = build_registry()
        no_scope = await registry.call("kg.knowledge.evaluate@1.0.0",
                                       {"domain": "packaging_labeling", "context": CONTEXT}, _identity(scopes=()))
        assert isinstance(no_scope, DeniedCapabilityCall) and no_scope.reason == "missing_scope"
        bad_domain = await registry.call("kg.knowledge.evaluate@1.0.0",
                                         {"domain": "../etc", "context": CONTEXT}, _identity())
        assert isinstance(bad_domain, DeniedCapabilityCall) and bad_domain.reason == "invalid_arg"
        bad_context = await registry.call("kg.knowledge.evaluate@1.0.0",
                                          {"domain": "packaging_labeling", "context": {"planet": "mars"}},
                                          _identity())
        assert bad_context["error"]["code"] == "invalid_request"
        bad_fact = await registry.call("kg.knowledge.evaluate@1.0.0",
                                       {"domain": "packaging_labeling", "context": CONTEXT,
                                        "facts": {"x": {"value": {"nested": "object"}}}}, _identity())
        assert bad_fact["error"]["code"] == "invalid_request"

    async def test_other_tenant_has_no_release(self, published):
        result = await build_registry().call(
            "kg.knowledge.evaluate@1.0.0", {"domain": "packaging_labeling", "context": CONTEXT},
            _identity(tenant="other_tenant"))
        assert result["error"]["code"] == "no_published_release"


def _client(tmp_path: Path, *, tenant="packaging_demo", scope="read") -> TestClient:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}",
                                blob_root=tmp_path / "blobs")
        await store.open()
        await publish_pack(store, load_pack(PACK), published_by="user:publisher", cases=load_cases(CASES))
        set_knowledge_service(KnowledgeEvaluationService(store))
        try:
            yield
        finally:
            set_knowledge_service(None)
            await store.close()

    app = FastAPI(lifespan=lifespan)
    app.include_router(knowledge_routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"scope": scope, "sub": "analyst", "tenant": tenant}
    return TestClient(app)


class TestRoute:
    def test_route_returns_the_structured_decision(self, tmp_path):
        with _client(tmp_path) as client:
            response = client.post("/knowledge/evaluate", json={
                "domain": "packaging_labeling", "context": CONTEXT,
                "facts": {**FACTS, "label.min_font_pt": {"value": 7, "evidence": EVIDENCE}}})
        assert response.status_code == 200
        body = response.json()
        assert body["overall"] == "non_compliant"
        failing = [r["item_id"] for r in body["results"] if r["outcome"] == "fail"]
        assert failing == ["PKG-011"]

    def test_route_maps_errors(self, tmp_path):
        with _client(tmp_path, tenant="other_tenant") as client:
            missing = client.post("/knowledge/evaluate", json={"domain": "packaging_labeling", "context": CONTEXT})
            invalid = client.post("/knowledge/evaluate", json={"domain": "packaging_labeling",
                                                               "context": {"planet": "mars"}})
        assert missing.status_code == 404 and missing.json()["detail"]["code"] == "no_published_release"
        assert invalid.status_code == 422

    def test_route_requires_read_scope(self, tmp_path):
        with _client(tmp_path, scope="") as client:
            response = client.post("/knowledge/evaluate", json={"domain": "packaging_labeling", "context": CONTEXT})
        assert response.status_code == 403


class TestStructuredFailures:
    """Audit 2026-10-24: bad fact types and non-finite numbers come back as
    structured results through the registry and the HTTP route, never as
    crashes or silent passes."""

    STRING_FONT = {**FACTS, "label.min_font_pt": {"value": "8", "evidence": EVIDENCE}}
    NAN_FONT = {**FACTS, "label.min_font_pt": {"value": float("nan"), "evidence": EVIDENCE}}

    async def test_registry_reports_type_mismatch_as_insufficient_evidence(self, published):
        result = await build_registry().call(
            "kg.knowledge.evaluate@1.0.0",
            {"domain": "packaging_labeling", "context": CONTEXT, "facts": self.STRING_FONT}, _identity())
        font = next(r for r in result["results"] if r["item_id"] == "PKG-011")
        assert font["outcome"] == "insufficient_evidence"
        assert font["predicates"][0]["reason"] == "fact_type_mismatch"
        assert result["overall"] == "needs_review" and result["human_review_required"]

    async def test_registry_rejects_non_finite_facts(self, published):
        result = await build_registry().call(
            "kg.knowledge.evaluate@1.0.0",
            {"domain": "packaging_labeling", "context": CONTEXT, "facts": self.NAN_FONT}, _identity())
        assert result["error"]["code"] == "invalid_request"

    def test_route_maps_type_mismatch_and_non_finite_values(self, tmp_path):
        with _client(tmp_path) as client:
            mismatch = client.post("/knowledge/evaluate", json={
                "domain": "packaging_labeling", "context": CONTEXT, "facts": self.STRING_FONT})
            # a non-compliant JSON client may still send the bare NaN token
            non_finite = client.post("/knowledge/evaluate", headers={"content-type": "application/json"},
                                     content=json.dumps({"domain": "packaging_labeling", "context": CONTEXT,
                                                         "facts": self.NAN_FONT}))
        assert mismatch.status_code == 200 and mismatch.json()["overall"] == "needs_review"
        assert non_finite.status_code == 422
        assert non_finite.json()["detail"]["code"] == "invalid_request"
