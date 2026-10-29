"""``/knowledge/packs`` review and publication routes: identity from the token
``sub``, tenant from the token, scope per operation, compare-and-swap on the
working revision, and the unchanged release gate.  Offline: temporary SQLite
governance store, synthetic pack."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import knowledge_curation as curation_routes
from graphrag.context_graph.knowledge_service import KnowledgeEvaluationService, set_knowledge_service
from graphrag.domains.energy.governance_store import GovernanceStore

DOMAIN = "/knowledge/packs/packaging_labeling"


def _client(tmp_path: Path, who: dict) -> TestClient:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}",
                                blob_root=tmp_path / "blobs")
        await store.open()
        app.state.store = store
        set_knowledge_service(KnowledgeEvaluationService(store))
        try:
            yield
        finally:
            set_knowledge_service(None)
            await store.close()

    app = FastAPI(lifespan=lifespan)
    app.include_router(curation_routes.router)
    app.dependency_overrides[get_current_user] = lambda: dict(who)
    return TestClient(app)


def _user(sub="curator-1", scope="read write admin", tenant="packaging_demo") -> dict:
    return {"sub": sub, "scope": scope, "tenant": tenant}


def _imported(client) -> int:
    response = client.post(f"{DOMAIN}/import", json={"expected_revision": 0})
    assert response.status_code == 200, response.text
    return response.json()["revision"]


def _entry(client, item_id: str) -> dict:
    body = client.get(DOMAIN).json()
    return next(e for e in body["items"] + body["exceptions"] if e["id"] == item_id)


class TestImport:
    def test_import_needs_admin_and_records_the_subject(self, tmp_path):
        who = _user(scope="read write")
        with _client(tmp_path, who) as client:
            assert client.post(f"{DOMAIN}/import", json={"expected_revision": 0}).status_code == 403
            who.update(_user(sub="admin-9"))
            assert _imported(client) == 1
            history = client.get(f"{DOMAIN}/history").json()["revisions"]
        assert [(r["revision"], r["changed_by"], r["change"]["action"]) for r in history] == [(1, "admin-9", "import")]

    def test_another_tenants_pack_is_not_found(self, tmp_path):
        with _client(tmp_path, _user(tenant="other_tenant")) as client:
            imported = client.post(f"{DOMAIN}/import", json={"expected_revision": 0})
            working = client.get(DOMAIN)
        assert imported.status_code == 404 and working.status_code == 404

    def test_unknown_or_malformed_domain(self, tmp_path):
        with _client(tmp_path, _user()) as client:
            assert client.post("/knowledge/packs/no_such_pack/import",
                               json={"expected_revision": 0}).status_code == 404
            assert client.post("/knowledge/packs/..%2Fetc/import", json={"expected_revision": 0}).status_code in (404, 422)


class TestReview:
    def test_reviewer_is_the_token_subject_not_a_body_field(self, tmp_path):
        with _client(tmp_path, _user(sub="curator-1")) as client:
            revision = _imported(client)
            spoofed = client.post(f"{DOMAIN}/items/PKG-013/approve", json={
                "expected_revision": revision, "review_ref": "R-1", "publish": False,
                "reviewed_by": "someone-else"})
            approved = client.post(f"{DOMAIN}/items/PKG-013/approve", json={
                "expected_revision": revision, "review_ref": "R-1", "publish": False})
            entry = _entry(client, "PKG-013")
            queue = client.get(DOMAIN).json()["review_queue"]
        assert spoofed.status_code == 422
        assert approved.status_code == 200 and approved.json()["changed_by"] == "curator-1"
        assert approved.json()["change"] == {"action": "approve", "kind": "items", "from": "PKG-013@v1",
                                             "to": "PKG-013@v2", "status": "approved", "reference": "R-1"}
        assert (entry["version"], entry["status"], entry["reviewed_by"]) == (2, "approved", "curator-1")
        assert all(q["key"] != "PKG-013@v2" for q in queue)

    def test_review_needs_write_scope(self, tmp_path):
        who = _user()
        with _client(tmp_path, who) as client:
            revision = _imported(client)
            who.update(_user(scope="read"))
            response = client.post(f"{DOMAIN}/items/PKG-013/approve",
                                   json={"expected_revision": revision, "review_ref": "R-1"})
        assert response.status_code == 403

    def test_missing_subject_is_refused(self, tmp_path):
        who = _user()
        with _client(tmp_path, who) as client:
            revision = _imported(client)
            who.update(_user(sub=""))
            response = client.post(f"{DOMAIN}/items/PKG-013/withdraw",
                                   json={"expected_revision": revision, "review_ref": "R-1"})
        assert response.status_code == 403

    def test_stale_revision_is_a_conflict(self, tmp_path):
        with _client(tmp_path, _user()) as client:
            revision = _imported(client)
            first = client.post(f"{DOMAIN}/items/PKG-013/withdraw",
                                json={"expected_revision": revision, "review_ref": "R-1"})
            stale = client.post(f"{DOMAIN}/items/PKG-001/withdraw",
                                json={"expected_revision": revision, "review_ref": "R-2"})
        assert first.status_code == 200 and stale.status_code == 409

    def test_invalid_transitions_and_unknown_entries(self, tmp_path):
        with _client(tmp_path, _user()) as client:
            revision = _imported(client)
            withdrawn_again = client.post(f"{DOMAIN}/items/PKG-014/withdraw",
                                          json={"expected_revision": revision, "review_ref": "R"})
            reapprove_published = client.post(f"{DOMAIN}/items/PKG-001/approve",
                                              json={"expected_revision": revision, "review_ref": "R"})
            unknown = client.post(f"{DOMAIN}/items/PKG-404/approve",
                                  json={"expected_revision": revision, "review_ref": "R"})
            bad_kind = client.post(f"{DOMAIN}/rules/PKG-001/approve",
                                   json={"expected_revision": revision, "review_ref": "R"})
        assert withdrawn_again.status_code == 409
        assert reapprove_published.status_code == 409
        assert unknown.status_code == 404
        assert bad_kind.status_code == 422

    def test_exception_grant_comes_from_the_token_subject(self, tmp_path):
        with _client(tmp_path, _user(sub="approver-3")) as client:
            revision = _imported(client)
            withdrawn = client.post(f"{DOMAIN}/exceptions/EXC-001/withdraw",
                                    json={"expected_revision": revision, "review_ref": "W-1"})
            reapproved = client.post(f"{DOMAIN}/exceptions/EXC-001/approve",
                                     json={"expected_revision": revision + 1, "review_ref": "W-2"})
        assert withdrawn.status_code == 200
        assert withdrawn.json()["change"]["to"] == "EXC-001@v2"
        assert reapproved.status_code == 409  # a withdrawn exception needs a new draft


class TestPublish:
    def test_publish_needs_admin_and_names_the_subject_as_publisher(self, tmp_path):
        who = _user(sub="publisher-5")
        with _client(tmp_path, who) as client:
            revision = _imported(client)
            who.update(_user(sub="publisher-5", scope="read write"))
            refused = client.post(f"{DOMAIN}/publish", json={"expected_revision": revision})
            who.update(_user(sub="publisher-5"))
            published = client.post(f"{DOMAIN}/publish", json={"expected_revision": revision})
            summary = client.get(DOMAIN).json()
        assert refused.status_code == 403
        assert published.status_code == 200, published.text
        body = published.json()
        assert body["published_by"] == "publisher-5"
        assert body["checks"]["regression"] == "passed" and body["checks"]["regression_cases"] == 26
        assert summary["active_release"]["release_id"] == body["release_id"]

    def test_failed_gate_is_422_and_keeps_the_active_release(self, tmp_path):
        with _client(tmp_path, _user()) as client:
            revision = _imported(client)
            first = client.post(f"{DOMAIN}/publish", json={"expected_revision": revision}).json()
            # publishing an uncurated extracted draft: applicability unknown everywhere
            approved = client.post(f"{DOMAIN}/items/PKG-013/approve",
                                   json={"expected_revision": revision, "review_ref": "R-13"})
            assert approved.status_code == 200
            refused = client.post(f"{DOMAIN}/publish", json={"expected_revision": revision + 1})
            active = client.get(DOMAIN).json()["active_release"]
        assert refused.status_code == 422
        detail = refused.json()["detail"]
        assert detail["code"] == "gate_failed" and detail["failures"]
        assert active["release_id"] == first["release_id"]

    def test_publish_of_a_stale_revision_is_a_conflict(self, tmp_path):
        with _client(tmp_path, _user()) as client:
            revision = _imported(client)
            client.post(f"{DOMAIN}/items/PKG-013/withdraw", json={"expected_revision": revision, "review_ref": "R"})
            stale = client.post(f"{DOMAIN}/publish", json={"expected_revision": revision})
        assert stale.status_code == 409

    def test_publish_without_working_pack_is_not_found(self, tmp_path):
        with _client(tmp_path, _user()) as client:
            response = client.post(f"{DOMAIN}/publish", json={"expected_revision": 0})
        assert response.status_code == 404
