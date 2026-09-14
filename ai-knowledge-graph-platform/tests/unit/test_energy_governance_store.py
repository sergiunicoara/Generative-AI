"""Durability and concurrency gates for Energy publication/workflow state."""

from __future__ import annotations

import asyncio
import multiprocessing
from pathlib import Path

import pytest

from graphrag.domains.energy.demo import EnergyDemoService
from graphrag.domains.energy.evidence_requests import EvidenceRequestError, EvidenceRequestService
from graphrag.domains.energy.governance_store import GovernanceStore, WorkflowConflictError
from graphrag.domains.energy.workflow import (
    APPROVED, CANCELLED, COMPLETED, MaintenanceWorkflow, REVIEW_REQUIRED,
    WorkflowTransitionError,
)


def _db_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


async def _runtime(tmp_path: Path):
    store = GovernanceStore(_db_url(tmp_path / "governance.sqlite"), blob_root=tmp_path / "published")
    await store.open()
    service = await EnergyDemoService.create(governance_store=store)
    return store, service, MaintenanceWorkflow(store, tenant=service.tenant)


async def test_publication_and_workflow_survive_a_store_restart(tmp_path: Path):
    store, service, workflow = await _runtime(tmp_path)
    initial_report = service.publication_report()
    transition, _wire, _replayed = await workflow.transition(
        "WO-9001", to_state=APPROVED, changed_by="operator", reason="Evidence reviewed",
        expected_version=0, command_id="restart-command",
    )
    await store.close()

    restarted = GovernanceStore(_db_url(tmp_path / "governance.sqlite"), blob_root=tmp_path / "published")
    await restarted.open()
    report, graph = await restarted.current("energy-demo")
    restored = MaintenanceWorkflow(restarted, tenant="energy-demo")

    assert report.version_id == initial_report.version_id
    assert len(graph) == report.published_triple_count
    assert await restored.current("WO-9001") == (APPROVED, transition.object_version)
    await restarted.close()


async def test_idempotency_replays_original_response_and_rejects_argument_reuse(tmp_path: Path):
    store, _service, workflow = await _runtime(tmp_path)
    first, first_wire, first_replayed = await workflow.transition(
        "WO-9001", to_state=APPROVED, changed_by="operator", reason="Evidence reviewed",
        expected_version=0, command_id="same-command",
    )
    replay, replay_wire, replayed = await workflow.transition(
        "WO-9001", to_state=APPROVED, changed_by="operator", reason="Evidence reviewed",
        expected_version=0, command_id="same-command",
    )

    assert first_replayed is False
    assert replayed is True
    assert replay == first
    assert replay_wire == first_wire
    with pytest.raises(WorkflowTransitionError, match="different arguments"):
        await workflow.transition(
            "WO-9001", to_state=APPROVED, changed_by="operator", reason="different reason",
            expected_version=0, command_id="same-command",
        )
    await store.close()


async def test_transition_log_replays_to_the_materialized_head_and_terminal_states_block_writes(tmp_path: Path):
    store, _service, workflow = await _runtime(tmp_path)
    approved, _, _ = await workflow.transition(
        "WO-9001", to_state=APPROVED, changed_by="reviewer", reason="approved",
        expected_version=0, command_id="approve",
    )
    completed, _, _ = await workflow.transition(
        "WO-9001", to_state=COMPLETED, changed_by="operator", reason="completed",
        expected_version=approved.object_version, command_id="complete",
    )
    history = await workflow.history("WO-9001")

    replayed_state = REVIEW_REQUIRED
    for event in history:
        assert event.from_state == replayed_state
        replayed_state = event.to_state
    assert (replayed_state, completed.object_version) == await workflow.current("WO-9001")
    with pytest.raises(WorkflowTransitionError, match="cannot transition"):
        await workflow.transition(
            "WO-9001", to_state=CANCELLED, changed_by="operator", reason="too late",
            expected_version=completed.object_version, command_id="cancel-after-complete",
        )
    await store.close()


def _process_transition(db_url: str, blob_root: str, command_id: str, queue) -> None:
    """Separate interpreter/process: proves SQLite WAL/CAS beyond one event loop."""
    async def run() -> None:
        store = GovernanceStore(db_url, blob_root=Path(blob_root))
        await store.open()
        try:
            await store.transition(
                "energy-demo", "WO-9001", to_state=APPROVED, changed_by="parallel-operator",
                reason="concurrent review", expected_version=0, command_id=command_id,
            )
        except WorkflowConflictError:
            queue.put("conflict")
        except Exception as exc:  # surfaced to the parent instead of disappearing in child stderr
            queue.put(f"error:{type(exc).__name__}:{exc}")
        else:
            queue.put("ok")
        finally:
            await store.close()
    asyncio.run(run())


def test_two_os_processes_conflict_and_exactly_one_cas_transition_wins(tmp_path: Path):
    """The production gate: a stale write produces one conflict, never two approvals."""
    async def publish() -> tuple[str, str]:
        store, _service, _workflow = await _runtime(tmp_path)
        db_url, blob_root = store.db_url, str(store.blob_root)
        await store.close()
        return db_url, blob_root

    db_url, blob_root = asyncio.run(publish())
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_process_transition, args=(db_url, blob_root, f"concurrent-{index}", queue))
        for index in range(2)
    ]
    for process in processes:
        process.start()
    outcomes = [queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    assert sorted(outcomes) == ["conflict", "ok"]


async def _evidence_runtime(tmp_path: Path):
    store, service, _workflow = await _runtime(tmp_path)
    return store, service, EvidenceRequestService(store, tenant=service.tenant)


def _blocked_rows(service: EnergyDemoService) -> list[dict[str, str]]:
    return service.answer("insufficient_evidence", tenant=service.tenant)["query_rows"]


async def test_evidence_request_survives_a_store_restart(tmp_path: Path):
    store, service, evidence = await _evidence_runtime(tmp_path)
    created, _wire, _replayed = await evidence.create(
        asset_id="WT-02", missing_field="temperature_c", source_system="Snowflake",
        owner="ops-team", priority="high", reason="Confirm gearbox temperature",
        created_by="operator-1", command_id="restart-evidence", graph=service.graph,
        query_rows=_blocked_rows(service),
    )
    await store.close()

    restarted = GovernanceStore(store.db_url, blob_root=store.blob_root)
    await restarted.open()
    restored = EvidenceRequestService(restarted, tenant="energy-demo")
    record, history = await restored.get(created.request_id)

    assert record.asset_id == "WT-02"
    assert record.state == "open"
    assert history == []
    await restarted.close()


async def test_evidence_request_idempotency_replays_original_response_and_rejects_argument_reuse(tmp_path: Path):
    store, service, evidence = await _evidence_runtime(tmp_path)
    rows = _blocked_rows(service)
    first, first_wire, first_replayed = await evidence.create(
        asset_id="WT-02", missing_field="temperature_c", source_system="Snowflake",
        owner="ops-team", priority="high", reason="Confirm gearbox temperature",
        created_by="operator-1", command_id="same-evidence-command", graph=service.graph, query_rows=rows,
    )
    replay, replay_wire, replayed = await evidence.create(
        asset_id="WT-02", missing_field="temperature_c", source_system="Snowflake",
        owner="ops-team", priority="high", reason="Confirm gearbox temperature",
        created_by="operator-1", command_id="same-evidence-command", graph=service.graph, query_rows=rows,
    )

    assert first_replayed is False
    assert replayed is True
    assert replay == first
    assert replay_wire == first_wire
    with pytest.raises(EvidenceRequestError, match="different arguments"):
        await evidence.create(
            asset_id="WT-02", missing_field="temperature_c", source_system="Snowflake",
            owner="ops-team", priority="low", reason="different priority this time",
            created_by="operator-1", command_id="same-evidence-command", graph=service.graph, query_rows=rows,
        )
    await store.close()


async def test_evidence_request_transition_log_and_terminal_states_block_writes(tmp_path: Path):
    store, service, evidence = await _evidence_runtime(tmp_path)
    created, _wire, _replayed = await evidence.create(
        asset_id="WT-02", missing_field="temperature_c", source_system="Snowflake",
        owner="ops-team", priority="high", reason="Confirm gearbox temperature",
        created_by="operator-1", command_id="evidence-lifecycle", graph=service.graph,
        query_rows=_blocked_rows(service),
    )
    in_progress, _wire, _replayed = await evidence.transition(
        created.request_id, to_state="in_progress", changed_by="snowflake-liaison",
        reason="Export requested from Snowflake", expected_version=created.object_version,
        command_id="evidence-in-progress",
    )
    fulfilled, _wire, _replayed = await evidence.transition(
        created.request_id, to_state="fulfilled", changed_by="snowflake-liaison",
        reason="Telemetry backfilled", expected_version=in_progress.object_version,
        command_id="evidence-fulfilled",
    )
    record, history = await evidence.get(created.request_id)

    assert record.state == "fulfilled"
    assert [item.to_state for item in history] == ["in_progress", "fulfilled"]
    with pytest.raises(EvidenceRequestError, match="cannot transition"):
        await evidence.transition(
            created.request_id, to_state="cancelled", changed_by="operator-1",
            reason="too late", expected_version=fulfilled.object_version,
            command_id="evidence-cancel-after-fulfilled",
        )
    await store.close()


async def test_evidence_requests_are_tenant_isolated(tmp_path: Path):
    """The API always gates on the single demo tenant, so this exercises the
    store's own `(tenant, request_id)` scoping directly -- the mechanism that
    actually enforces isolation, rather than only the coarser API-level gate."""
    store, service, evidence = await _evidence_runtime(tmp_path)
    created, _wire, _replayed = await evidence.create(
        asset_id="WT-02", missing_field="temperature_c", source_system="Snowflake",
        owner="ops-team", priority="high", reason="Confirm gearbox temperature",
        created_by="operator-1", command_id="evidence-tenant-a", graph=service.graph,
        query_rows=_blocked_rows(service),
    )
    other_tenant = EvidenceRequestService(store, tenant="other-tenant")

    with pytest.raises(EvidenceRequestError, match="unknown evidence request"):
        await other_tenant.get(created.request_id)
    assert await other_tenant.list() == []
    same_tenant_list = await evidence.list()
    assert [item.request_id for item in same_tenant_list] == [created.request_id]
    await store.close()


async def test_evidence_request_stale_object_version_conflict(tmp_path: Path):
    store, service, evidence = await _evidence_runtime(tmp_path)
    created, _wire, _replayed = await evidence.create(
        asset_id="WT-02", missing_field="temperature_c", source_system="Snowflake",
        owner="ops-team", priority="high", reason="Confirm gearbox temperature",
        created_by="operator-1", command_id="evidence-cas", graph=service.graph,
        query_rows=_blocked_rows(service),
    )
    await evidence.transition(
        created.request_id, to_state="in_progress", changed_by="snowflake-liaison",
        reason="Export requested", expected_version=created.object_version,
        command_id="evidence-cas-advance",
    )
    with pytest.raises(EvidenceRequestError, match="expected object_version"):
        await evidence.transition(
            created.request_id, to_state="in_progress", changed_by="operator-1",
            reason="stale retry", expected_version=created.object_version,
            command_id="evidence-cas-stale",
        )
    await store.close()
