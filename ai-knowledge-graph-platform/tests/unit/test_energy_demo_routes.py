"""Route-level tests for the Energy demo's new SHACL-publication-gate
endpoints: GET /publication, GET /quarantine, POST /rollback.

Mirrors tests/unit/test_enterprise_routes.py's pattern (mount the real
router, override get_current_user). api/routes/energy_demo.py holds a
module-level singleton `_service` built at import time -- each test
replaces it with a fresh EnergyDemoService() via monkeypatch so state never
leaks between tests (a rollback in one test must not be visible in another).
"""

from __future__ import annotations

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import energy_demo as energy_demo_routes
from graphrag.domains.energy.demo import EnergyDemoService


def _client(*, scope: str = "read") -> TestClient:
    app = FastAPI()
    # Real mount point: api/main.py's app.include_router(energy_demo.router,
    # prefix="/energy-demo", ...) -- the router's own "" (empty-path)
    # demo_page route requires a non-empty prefix to combine with.
    app.include_router(energy_demo_routes.router, prefix="/energy-demo")
    app.dependency_overrides[get_current_user] = lambda: {
        "scope": scope, "sub": "reviewer-1", "tenant": "energy-demo",
    }
    return TestClient(app)


def _fresh_service(monkeypatch) -> None:
    monkeypatch.setattr(energy_demo_routes, "_service", EnergyDemoService())


class TestGetPublication:
    def test_returns_the_current_published_version(self, monkeypatch):
        _fresh_service(monkeypatch)
        response = _client().get("/energy-demo/publication")
        assert response.status_code == 200
        body = response.json()
        assert body["quarantined_records"] == []
        assert body["published_triple_count"] > 0
        assert "version_id" in body

    def test_wrong_tenant_is_hidden_as_404(self, monkeypatch):
        _fresh_service(monkeypatch)
        app = FastAPI()
        app.include_router(energy_demo_routes.router, prefix="/energy-demo")
        app.dependency_overrides[get_current_user] = lambda: {
            "scope": "read", "sub": "x", "tenant": "another-tenant",
        }
        response = TestClient(app).get("/energy-demo/publication")
        assert response.status_code == 404


class TestGetQuarantine:
    def test_returns_empty_list_when_nothing_is_quarantined(self, monkeypatch):
        _fresh_service(monkeypatch)
        response = _client().get("/energy-demo/quarantine")
        assert response.status_code == 200
        assert response.json()["quarantined_records"] == []


class TestPostRollback:
    def test_requires_write_scope(self, monkeypatch):
        _fresh_service(monkeypatch)
        response = _client(scope="read").post("/energy-demo/rollback")
        assert response.status_code == 403

    def test_rollback_with_nothing_to_roll_back_to_is_a_conflict(self, monkeypatch):
        _fresh_service(monkeypatch)
        response = _client(scope="write").post("/energy-demo/rollback")
        assert response.status_code == 409

    def test_rollback_after_a_second_publish_restores_the_first_version(self, monkeypatch):
        _fresh_service(monkeypatch)
        service = energy_demo_routes._service
        first_version_id = service.publication_report().version_id

        # Stage a second version directly (same shape the real demo would
        # use if it ever re-published), so there's something to roll back to.
        second_candidate = type(service.graph)()
        for triple in service.graph:
            second_candidate.add(triple)
        service._publisher.stage_and_publish(second_candidate)
        service.graph = service._publisher.current

        response = _client(scope="write").post("/energy-demo/rollback")
        assert response.status_code == 200
        body = response.json()
        assert body["rolled_back_from"] == first_version_id
