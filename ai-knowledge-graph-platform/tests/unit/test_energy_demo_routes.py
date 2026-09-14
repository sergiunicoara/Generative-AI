"""Energy routes use lifespan state, not per-process module singletons."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import energy_demo as energy_demo_routes
from graphrag.domains.energy.demo import EnergyDemoService
from graphrag.domains.energy.evidence_requests import EvidenceRequestService
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
        app.state.energy_demo_evidence_requests = EvidenceRequestService(store, tenant=service.tenant)
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


def test_dashboard_html_closes_its_style_block_before_the_workspace_markup(tmp_path: Path):
    with _client(tmp_path) as client:
        response = client.get("/energy-demo")

    assert response.status_code == 200
    assert "</style><header>" in response.text
    assert "<h1>Energy Asset Intelligence</h1>" in response.text


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


def _evidence_request_payload(**overrides: object) -> dict:
    payload = {
        "asset_id": "WT-02", "missing_field": "temperature_c", "source_system": "Snowflake",
        "owner": "ops-team", "priority": "high", "reason": "Confirm gearbox temperature",
        "command_id": "browser-evidence-1",
    }
    payload.update(overrides)
    return payload


def test_evidence_request_endpoint_requires_cas_and_replays_an_idempotent_command_verbatim(tmp_path: Path):
    with _client(tmp_path, scope="write") as client:
        created = client.post("/energy-demo/evidence-requests", json=_evidence_request_payload())
        assert created.status_code == 201
        request_id = created.json()["request_id"]
        assert created.json()["object_version"] == 0

        payload = {
            "to_state": "in_progress", "reason": "Snowflake export requested",
            "expected_version": 0, "command_id": "browser-evidence-transition-1",
        }
        first = client.post(f"/energy-demo/evidence-requests/{request_id}/transition", json=payload)
        replay = client.post(f"/energy-demo/evidence-requests/{request_id}/transition", json=payload)
        stale = client.post(f"/energy-demo/evidence-requests/{request_id}/transition", json={
            "to_state": "fulfilled", "reason": "stale browser tab", "expected_version": 0,
            "command_id": "browser-evidence-transition-2",
        })
        detail = client.get(f"/energy-demo/evidence-requests/{request_id}")

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.content == first.content
    assert stale.status_code == 409
    assert detail.json()["request"]["state"] == "in_progress"
    assert detail.json()["request"]["object_version"] == 1


def test_read_scope_cannot_create_or_transition_evidence_requests(tmp_path: Path):
    with _client(tmp_path, scope="write") as writer:
        seed = writer.post("/energy-demo/evidence-requests", json=_evidence_request_payload(
            command_id="seed-for-read-scope-test",
        ))
        request_id = seed.json()["request_id"]

    with _client(tmp_path, scope="read") as client:
        create_response = client.post("/energy-demo/evidence-requests", json=_evidence_request_payload(
            command_id="forbidden-create",
        ))
        transition_response = client.post(
            f"/energy-demo/evidence-requests/{request_id}/transition",
            json={"to_state": "in_progress", "reason": "Not authorized", "expected_version": 0,
                  "command_id": "forbidden-transition"},
        )
    assert create_response.status_code == 403
    assert transition_response.status_code == 403


def test_insufficient_evidence_answer_is_byte_identical_before_and_after_a_remediation_request(tmp_path: Path):
    """The safety boundary: a remediation request never touches published
    RDF, so it must not change the advisory answer that motivated it."""
    with _client(tmp_path, scope="write") as client:
        before = client.get("/energy-demo/answer/insufficient_evidence")
        created = client.post("/energy-demo/evidence-requests", json=_evidence_request_payload())
        request_id = created.json()["request_id"]
        client.post(f"/energy-demo/evidence-requests/{request_id}/transition", json={
            "to_state": "in_progress", "reason": "Snowflake export requested",
            "expected_version": 0, "command_id": "safety-boundary-in-progress",
        })
        client.post(f"/energy-demo/evidence-requests/{request_id}/transition", json={
            "to_state": "fulfilled", "reason": "Telemetry backfilled",
            "expected_version": 1, "command_id": "safety-boundary-fulfilled",
        })
        after = client.get("/energy-demo/answer/insufficient_evidence")

    assert before.status_code == 200
    assert after.status_code == 200
    assert after.content == before.content


def test_dashboard_html_includes_evidence_remediation_panel_and_disclaimer(tmp_path: Path):
    with _client(tmp_path) as client:
        response = client.get("/energy-demo")

    assert response.status_code == 200
    assert 'id="evidence-remediation"' in response.text
    assert (
        "This creates a governed follow-up task. "
        "It does not infer or create the missing operational evidence."
    ) in response.text
    # The capture script targets the 2nd <button> on the rendered page (the
    # "incomplete" priorities button) via a fragile positional selector; the
    # remediation panel's own markup must stay purely additive at the end of
    # the existing workspace, not introduce any *static* button earlier in
    # the HTML than the existing workflow's approve/reject buttons.
    workspace_end = response.text.index("</details>")
    remediation_start = response.text.index('id="evidence-remediation"')
    assert workspace_end < remediation_start
