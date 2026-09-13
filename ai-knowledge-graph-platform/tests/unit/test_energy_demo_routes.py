"""Energy routes use lifespan state, not per-process module singletons."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import energy_demo as energy_demo_routes
from graphrag.domains.energy.demo import EnergyDemoService
from graphrag.domains.energy.governance_store import GovernanceStore
from graphrag.domains.energy.workflow import MaintenanceWorkflow


def _client(tmp_path: Path, *, scope: str = "read") -> TestClient:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = GovernanceStore(
            f"sqlite+aiosqlite:///{(tmp_path / 'governance.sqlite').as_posix()}",
            blob_root=tmp_path / "published",
        )
        await store.open()
        service = await EnergyDemoService.create(governance_store=store)
        app.state.energy_demo_service = service
        app.state.energy_demo_workflow = MaintenanceWorkflow(store, tenant=service.tenant)
        try:
            yield
        finally:
            await store.close()

    app = FastAPI(lifespan=lifespan)
    app.include_router(energy_demo_routes.router, prefix="/energy-demo")
    app.dependency_overrides[get_current_user] = lambda: {
        "scope": scope, "sub": "reviewer-1", "tenant": "energy-demo",
    }
    return TestClient(app)


def test_publication_and_quarantine_read_the_lifespan_published_version(tmp_path: Path):
    with _client(tmp_path) as client:
        publication = client.get("/energy-demo/publication")
        quarantine = client.get("/energy-demo/quarantine")

    assert publication.status_code == 200
    assert publication.json()["published_triple_count"] > 0
    assert quarantine.json()["quarantined_records"] == []


def test_workflow_endpoint_requires_cas_and_replays_an_idempotent_command_verbatim(tmp_path: Path):
    with _client(tmp_path, scope="write") as client:
        initial = client.get("/energy-demo/work-orders/WO-9001/lifecycle")
        assert initial.status_code == 200
        assert initial.json()["object_version"] == 0
        payload = {
            "to_state": "approved", "reason": "Reviewed mapped telemetry and bulletin",
            "expected_version": 0, "command_id": "browser-retry-1",
        }
        first = client.post("/energy-demo/work-orders/WO-9001/transition", json=payload)
        replay = client.post("/energy-demo/work-orders/WO-9001/transition", json=payload)
        stale = client.post("/energy-demo/work-orders/WO-9001/transition", json={
            "to_state": "completed", "reason": "stale browser tab", "expected_version": 0,
            "command_id": "browser-stale-2",
        })
        lifecycle = client.get("/energy-demo/work-orders/WO-9001/lifecycle")

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.content == first.content
    assert stale.status_code == 409
    assert lifecycle.json()["current_state"] == "approved"
    assert lifecycle.json()["object_version"] == 1


def test_read_scope_cannot_mutate_a_workflow(tmp_path: Path):
    with _client(tmp_path, scope="read") as client:
        response = client.post("/energy-demo/work-orders/WO-9001/transition", json={
            "to_state": "approved", "reason": "Not authorized", "expected_version": 0,
            "command_id": "forbidden",
        })
    assert response.status_code == 403
