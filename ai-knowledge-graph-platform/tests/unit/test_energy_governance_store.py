"""Durability and concurrency gates for Energy publication/workflow state."""

from __future__ import annotations

import asyncio
import multiprocessing
from pathlib import Path

import pytest

from graphrag.domains.energy.demo import EnergyDemoService
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
