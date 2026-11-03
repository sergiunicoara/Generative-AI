"""Authenticated pack review / publish / rollback / history routes.

Reviewer and publisher identity come from the token ``sub`` (never the body);
the tenant comes from the token and must match the pack's tenant; a release
is gated by the same checks as ``publish_pack``.  Offline: temporary SQLite
governance store, synthetic pack.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import knowledge as knowledge_routes
from graphrag.context_graph.knowledge_release import load_cases
from graphrag.context_graph.knowledge_service import KnowledgeEvaluationService, set_knowledge_service
from graphrag.domains.energy.governance_store import GovernanceStore

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "evals" / "knowledge_eval" / "packaging_labeling_cases.json"
DOMAIN = "packaging_labeling"
CONTEXT = {"market": "NORTHLAND", "product_family": "otc_tablets", "workflow": "label_review",
           "customer": "CORNERSHOP", "brand": "OTHERBRAND", "product": "PX-200",
           "as_of": "2026-08-01T00:00:00Z"}


def _client(tmp_path: Path, claims: dict) -> TestClient:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}",
                                blob_root=tmp_path / "blobs")
        await store.open()
        set_knowledge_service(KnowledgeEvaluationService(store))
        try:
            yield
        finally:
            set_knowledge_service(None)
            await store.close()

    app = FastAPI(lifespan=lifespan)
    app.include_router(knowledge_routes.router)
    app.dependency_overrides[get_current_user] = lambda: claims
    return TestClient(app)


@pytest.fixture
def claims():
    return {"scope": "read write admin", "sub": "curator-1", "tenant": "packaging_demo"}


def _publish(client, domain=DOMAIN):
    return client.post(f"/knowledge/packs/{domain}/publish")


class TestPublish:
    def test_requires_admin_scope(self, tmp_path, claims):
        claims["scope"] = "read write"
        with _client(tmp_path, claims) as client:
            assert _publish(client).status_code == 403

    def test_publishes_gated_release_as_the_token_subject(self, tmp_path, claims):
        with _client(tmp_path, claims) as client:
            response = _publish(client)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["published_by"] == "curator-1" and body["version"] == 1
            assert body["checks"]["regression"] == "passed"
            assert body["checks"]["regression_cases"] == len(load_cases(CASES))
            assert body["item_count"] > 0 and body["exception_count"] > 0
            # a release holds restricted knowledge: its content is never returned here
            assert not {"items", "exceptions", "item_keys", "exception_keys"} & set(body)
            evaluated = client.post("/knowledge/evaluate", json={"domain": DOMAIN, "context": CONTEXT})
            assert evaluated.status_code == 200
            assert evaluated.json()["release"]["release_id"] == body["release_id"]

    def test_publishing_identical_content_is_idempotent(self, tmp_path, claims):
        with _client(tmp_path, claims) as client:
            first, second = _publish(client).json(), _publish(client).json()
        assert first["release_id"] == second["release_id"] and second["version"] == 1

    def test_unknown_domain_and_foreign_tenant_are_not_found(self, tmp_path, claims):
        with _client(tmp_path, claims) as client:
            assert _publish(client, "no_such_pack").status_code == 404
            assert _publish(client, "no_such_pack").json()["detail"]["code"] == "pack_not_found"
            claims["tenant"] = "other_tenant"
            foreign = _publish(client)
        assert foreign.status_code == 404 and foreign.json()["detail"]["code"] == "pack_not_found"

    def test_domain_in_the_path_is_validated(self, tmp_path, claims):
        with _client(tmp_path, claims) as client:
            assert _publish(client, "Bad-Domain").status_code == 422

    def test_gate_failure_is_reported_and_leaves_no_release(self, tmp_path, claims, monkeypatch):
        raw = json.loads(CASES.read_text(encoding="utf-8"))
        first = raw["cases"][0]
        first["expected"]["overall"] = "needs_review" if first["expected"].get("overall") != "needs_review" else "compliant"
        (tmp_path / f"{DOMAIN}_cases.json").write_text(json.dumps(raw), encoding="utf-8")
        monkeypatch.setattr(knowledge_routes, "CASES_DIR", tmp_path)
        with _client(tmp_path, claims) as client:
            rejected = _publish(client)
            assert rejected.status_code == 422
            detail = rejected.json()["detail"]
            assert detail["code"] == "release_rejected" and any(first["id"] in f for f in detail["failures"])
            missing = client.post("/knowledge/evaluate", json={"domain": DOMAIN, "context": CONTEXT})
            assert missing.status_code == 404

    def test_missing_regression_cases_are_refused(self, tmp_path, claims, monkeypatch):
        monkeypatch.setattr(knowledge_routes, "CASES_DIR", tmp_path / "empty")
        with _client(tmp_path, claims) as client:
            response = _publish(client)
        assert response.status_code == 422
        assert any("regression" in failure for failure in response.json()["detail"]["failures"])


class TestReviewQueue:
    def test_lists_drafts_for_curators(self, tmp_path, claims):
        claims["scope"] = "write"
        with _client(tmp_path, claims) as client:
            response = client.get(f"/knowledge/packs/{DOMAIN}/review-queue")
        assert response.status_code == 200
        assert "PKG-013@v1" in [entry["key"] for entry in response.json()["queue"]]

    def test_leaves_out_drafts_the_caller_may_not_read(self, tmp_path, claims, monkeypatch):
        pack_dir = tmp_path / "packs"
        pack_dir.mkdir()
        raw = (ROOT / "config" / "knowledge_packs" / f"{DOMAIN}.yml").read_text(encoding="utf-8")
        marker = "    id: PKG-013\n"
        restricted = marker + ('    access: {mode: restricted, state: known, allow_principals: ["group:secret"],'
                               ' requires_group_resolution: true}\n')
        assert raw.count(marker) == 1
        (pack_dir / f"{DOMAIN}.yml").write_text(raw.replace(marker, restricted), encoding="utf-8")
        monkeypatch.setattr(knowledge_routes, "PACK_DIR", pack_dir)
        claims["scope"] = "write"
        with _client(tmp_path, claims) as client:
            response = client.get(f"/knowledge/packs/{DOMAIN}/review-queue")
        assert response.status_code == 200
        assert response.json()["queue"] == [] and "PKG-013" not in response.text

    def test_requires_write_scope(self, tmp_path, claims):
        claims["scope"] = "read"
        with _client(tmp_path, claims) as client:
            assert client.get(f"/knowledge/packs/{DOMAIN}/review-queue").status_code == 403

    def test_unknown_pack_is_not_found(self, tmp_path, claims):
        with _client(tmp_path, claims) as client:
            assert client.get("/knowledge/packs/no_such_pack/review-queue").status_code == 404


class TestHistoryAndRollback:
    def test_rollback_appends_a_release_and_history_shows_both(self, tmp_path, claims):
        with _client(tmp_path, claims) as client:
            first = _publish(client).json()
            rolled = client.post(f"/knowledge/releases/{DOMAIN}/rollback", json={"release_id": first["release_id"]})
            assert rolled.status_code == 200, rolled.text
            body = rolled.json()
            assert body["version"] == 2 and body["rolled_back_from"] == first["release_id"]
            assert body["published_by"] == "curator-1" and "items" not in body
            history = client.get(f"/knowledge/releases/{DOMAIN}").json()
        assert [r["version"] for r in history["releases"]] == [1, 2]
        assert history["active_release_id"] == body["release_id"]
        assert not any({"items", "exceptions", "item_keys"} & set(r) for r in history["releases"])

    def test_rollback_to_an_unknown_release_is_not_found(self, tmp_path, claims):
        with _client(tmp_path, claims) as client:
            _publish(client)
            missing = client.post(f"/knowledge/releases/{DOMAIN}/rollback", json={"release_id": "0" * 32})
        assert missing.status_code == 404 and missing.json()["detail"]["code"] == "release_not_found"

    def test_rollback_requires_admin_and_a_well_formed_id(self, tmp_path, claims):
        with _client(tmp_path, claims) as client:
            first = _publish(client).json()
            claims["scope"] = "read write"
            assert client.post(f"/knowledge/releases/{DOMAIN}/rollback",
                               json={"release_id": first["release_id"]}).status_code == 403
            claims["scope"] = "admin"
            assert client.post(f"/knowledge/releases/{DOMAIN}/rollback",
                               json={"release_id": "not-a-release"}).status_code == 422

    def test_history_is_tenant_scoped_and_read_only(self, tmp_path, claims):
        with _client(tmp_path, claims) as client:
            _publish(client)
            claims.update(scope="read", tenant="other_tenant")
            other = client.get(f"/knowledge/releases/{DOMAIN}")
        assert other.status_code == 200 and other.json() == {"releases": [], "active_release_id": None}
