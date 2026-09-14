"""Energy routes use lifespan state, not per-process module singletons."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from rdflib import RDF, Graph, URIRef
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import energy_demo as energy_demo_routes
from graphrag.domains.energy.demo import EnergyDemoService
from graphrag.domains.energy.evidence_requests import EvidenceRequestService
from graphrag.domains.energy.governance_store import GovernanceStore
from graphrag.domains.energy.publication import PublicationReport
from graphrag.domains.energy.workflow import MaintenanceWorkflow


def _client(
    tmp_path: Path, *, scope: str = "read", include_invalid_fixture: bool = False,
    tenant: str = "energy-demo",
) -> TestClient:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = GovernanceStore(
            f"sqlite+aiosqlite:///{(tmp_path / 'governance.sqlite').as_posix()}",
            blob_root=tmp_path / "published",
        )
        await store.open()
        service = await EnergyDemoService.create(
            governance_store=store, include_invalid_fixture=include_invalid_fixture,
        )
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
        "scope": scope, "sub": "reviewer-1", "tenant": tenant,
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


def test_quarantine_endpoint_is_empty_by_default(tmp_path: Path):
    """The standard demo fixture is fully conformant -- no invalid records
    ship by default. Matches every other zero-quarantine assertion in
    tests/unit/test_energy_demo.py and the E2E acceptance gate."""
    with _client(tmp_path) as client:
        response = client.get("/energy-demo/quarantine")

    assert response.status_code == 200
    assert response.json()["quarantined_records"] == []


def test_quarantine_endpoint_reports_an_invalid_record_when_the_fixture_is_enabled(tmp_path: Path):
    with _client(tmp_path, include_invalid_fixture=True) as client:
        response = client.get("/energy-demo/quarantine")

    assert response.status_code == 200
    records = response.json()["quarantined_records"]
    assert len(records) == 1
    record = records[0]
    assert record["subject"].endswith("obs-WT-10-temperature_c-invalid")
    assert record["reasons"]
    assert record["source_type"] == "snowflake_telemetry_invalid_sample"
    assert record["provenance"] == "urn:synthetic:snowflake:telemetry-invalid-sample"


def test_publication_report_reflects_the_invalid_fixture(tmp_path: Path):
    with _client(tmp_path, include_invalid_fixture=True) as client:
        response = client.get("/energy-demo/publication")

    assert response.status_code == 200
    report = response.json()
    assert len(report["quarantined_records"]) == 1
    # The *published* subset still conforms -- only the quarantined record
    # was invalid, and it was excluded before publication.
    assert report["conforms"] is True


def test_dashboard_html_includes_publication_audit_panel_and_safety_disclaimer(tmp_path: Path):
    with _client(tmp_path) as client:
        response = client.get("/energy-demo")

    assert response.status_code == 200
    assert 'id="publication-audit"' in response.text
    assert (
        "Operators can inspect why a record was excluded. "
        "They cannot silently release invalid RDF from the dashboard."
    ) in response.text


def test_dashboard_html_does_not_contain_a_quarantine_release_control(tmp_path: Path):
    with _client(tmp_path) as client:
        response = client.get("/energy-demo")

    assert response.status_code == 200
    start = response.text.index('id="publication-audit"')
    end = response.text.index("</section>", start)
    section_html = response.text[start:end]
    assert "<button" not in section_html


def test_publication_audit_panel_fetches_client_side_not_embedded_server_side(tmp_path: Path):
    """Requirement: don't duplicate publication logic in the HTML route --
    the panel must fetch from the existing GET routes client-side, not have
    `_dashboard_html()` embed `service.publication_report()` into the
    server-rendered `data` blob the way `current`/`incomplete`/`validation`
    already are."""
    with _client(tmp_path) as client:
        response = client.get("/energy-demo")

    assert response.status_code == 200
    data_start = response.text.index("const data=")
    data_end = response.text.index(";\n", data_start)
    embedded_data = response.text[data_start:data_end]
    assert '"publication"' not in embedded_data
    assert '"quarantine"' not in embedded_data
    assert "fetch('/energy-demo/publication')" in response.text
    assert "fetch('/energy-demo/quarantine')" in response.text


def test_publication_audit_script_has_an_error_handling_path(tmp_path: Path):
    """Code-presence check only: asserts the client-side error-handling path
    exists (mirroring loadWorkflow()'s existing catch pattern), writing a
    readable message into the panel on fetch failure. This does not exercise
    a live API failure -- simulating one through TestClient (which runs no
    JS and talks to the real route) isn't practical here."""
    with _client(tmp_path) as client:
        response = client.get("/energy-demo")

    assert response.status_code == 200
    script_start = response.text.index("loadPublicationAudit")
    script_slice = response.text[script_start:script_start + 600]
    assert "catch(error)" in script_slice
    assert "summaryBox.textContent=error.message" in script_slice


def test_read_scope_can_view_publication_and_quarantine(tmp_path: Path):
    with _client(tmp_path, scope="read") as client:
        publication = client.get("/energy-demo/publication")
        quarantine = client.get("/energy-demo/quarantine")
        validation = client.get("/energy-demo/validation")
        history = client.get("/energy-demo/publication/history")

    assert publication.status_code == 200
    assert quarantine.status_code == 200
    assert validation.status_code == 200
    assert history.status_code == 200


def test_publication_quarantine_validation_404_for_wrong_tenant(tmp_path: Path):
    with _client(tmp_path, tenant="other-tenant") as client:
        publication = client.get("/energy-demo/publication")
        quarantine = client.get("/energy-demo/quarantine")
        validation = client.get("/energy-demo/validation")
        history = client.get("/energy-demo/publication/history")

    assert publication.status_code == 404
    assert quarantine.status_code == 404
    assert validation.status_code == 404
    assert history.status_code == 404


def test_energy_demo_routes_expose_no_new_mutation_endpoints():
    """Regression guard: the publication/quarantine audit panel must stay
    read-only. Enumerates every POST/PUT/DELETE route under /energy-demo and
    asserts nothing beyond the four known, pre-existing write operations was
    added."""
    mutating_paths = {
        route.path
        for route in energy_demo_routes.router.routes
        if getattr(route, "methods", None) and route.methods & {"POST", "PUT", "DELETE"}
    }
    assert mutating_paths == {
        "/work-orders/{work_order_id}/transition",
        "/rollback",
        "/evidence-requests",
        "/evidence-requests/{request_id}/transition",
    }


def test_publication_history_endpoint_is_read_only():
    """The new history route must never accept a mutating method -- a
    narrower, path-specific companion to the router-wide regression guard
    above."""
    (route,) = [r for r in energy_demo_routes.router.routes if r.path == "/publication/history"]
    assert route.methods == {"GET"}


def test_publication_history_reports_the_lone_lifespan_published_version(tmp_path: Path):
    """With no rollback, history has exactly the one version the lifespan
    published, it is the active version, and it carries no rollback
    provenance."""
    with _client(tmp_path) as client:
        history = client.get("/energy-demo/publication/history")
        publication = client.get("/energy-demo/publication")

    assert history.status_code == 200
    body = history.json()
    assert body["active_version_id"] == publication.json()["version_id"]
    assert [p["version_id"] for p in body["publications"]] == [body["active_version_id"]]
    entry = body["publications"][0]
    assert entry["rolled_back_from"] is None
    assert entry["conforms"] is True
    assert entry["published_triple_count"] > 0
    # No filesystem paths or blob hashes -- PublicationReport carries none of
    # that, but assert it directly so a future field addition can't regress
    # the "no filesystem paths, no secrets" contract.
    assert set(entry) == {
        "version_id", "published_at", "conforms",
        "published_triple_count", "quarantined_count", "rolled_back_from",
    }


async def _seed_a_rollback(tmp_path: Path, *, tenant: str = "energy-demo") -> None:
    """Seed a durable governance store at `tmp_path` with two distinct
    published versions and one rollback to the first -- directly through
    `GovernanceStore`, since `EnergyDemoService` publishes deterministic,
    content-identical graphs on every construction (idempotent by content
    hash) and so cannot produce a second version on its own. `_client()`'s
    own lifespan then republishes the identical first-version content,
    which `GovernanceStore.publish()` recognises as already-active and
    returns unchanged -- so the three seeded rows here are exactly what the
    history endpoint later sees."""
    store = GovernanceStore(
        f"sqlite+aiosqlite:///{(tmp_path / 'governance.sqlite').as_posix()}",
        blob_root=tmp_path / "published",
    )
    await store.open()
    try:
        service = await EnergyDemoService.create(governance_store=store)
        first_version_id = service.publication_report().version_id

        second_graph = Graph()
        second_graph += service.graph
        second_graph.add((URIRef("urn:test:marker"), RDF.type, URIRef("urn:test:Marker")))
        await store.publish(
            tenant,
            PublicationReport(
                version_id="seed-second-version",
                # A real, later wall-clock timestamp -- not a hardcoded
                # literal -- so this sorts after the original publish (real
                # "now" at service-creation time) regardless of when the
                # test runs, matching the same `_now_iso()`-style format
                # the store itself uses.
                published_at=datetime.now(timezone.utc).isoformat(),
                published_triple_count=len(second_graph),
                candidate_record_count=len(second_graph),
            ),
            second_graph,
        )
        restored = await store.rollback(tenant)
        assert restored.rolled_back_from == first_version_id
    finally:
        await store.close()


def test_publication_history_orders_versions_chronologically_and_marks_a_rollback(tmp_path: Path):
    asyncio.run(_seed_a_rollback(tmp_path))

    with _client(tmp_path) as client:
        response = client.get("/energy-demo/publication/history")

    assert response.status_code == 200
    body = response.json()
    publications = body["publications"]
    assert len(publications) == 3
    # Chronological (oldest first) -- the seeded second version sits between
    # the original publish and the rollback that restores the original.
    assert publications[1]["version_id"] == "seed-second-version"
    first_version_id = publications[0]["version_id"]
    rollback_entry = publications[2]
    assert publications[0]["rolled_back_from"] is None
    assert publications[1]["rolled_back_from"] is None
    assert rollback_entry["rolled_back_from"] == first_version_id
    # The rollback is what's actually serving now.
    assert body["active_version_id"] == rollback_entry["version_id"]
    assert rollback_entry["version_id"] != first_version_id


def test_dashboard_html_includes_publication_history_section_fetched_client_side(tmp_path: Path):
    with _client(tmp_path) as client:
        response = client.get("/energy-demo")

    assert response.status_code == 200
    assert 'id="publication-history"' in response.text
    assert "fetch('/energy-demo/publication/history')" in response.text
    data_start = response.text.index("const data=")
    data_end = response.text.index(";\n", data_start)
    embedded_data = response.text[data_start:data_end]
    assert '"publications"' not in embedded_data
