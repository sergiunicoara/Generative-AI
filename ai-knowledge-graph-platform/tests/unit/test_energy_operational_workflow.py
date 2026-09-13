from __future__ import annotations

from pathlib import Path

import pytest
from rdflib.namespace import RDF

from graphrag.domains.energy.demo import ENERGY, REC, EnergyDemoService
from graphrag.domains.energy.governance_store import GovernanceStore
from graphrag.domains.energy.workflow import (
    APPROVED, COMPLETED, MaintenanceWorkflow, REVIEW_REQUIRED, WorkflowTransitionError,
)


async def _workflow(tmp_path: Path) -> tuple[GovernanceStore, MaintenanceWorkflow]:
    store = GovernanceStore(
        f"sqlite+aiosqlite:///{(tmp_path / 'workflow.sqlite').as_posix()}",
        blob_root=tmp_path / "published",
    )
    await store.open()
    service = await EnergyDemoService.create(governance_store=store)
    return store, MaintenanceWorkflow(store, tenant=service.tenant)


async def test_transitions_are_append_only_and_current_state_is_derived(tmp_path: Path):
    store, workflow = await _workflow(tmp_path)
    assert await workflow.current_state("WO-9001") == REVIEW_REQUIRED
    approved, _, _ = await workflow.transition(
        "WO-9001", to_state=APPROVED, changed_by="operator-1", reason="Evidence reviewed",
        expected_version=0, command_id="approve",
    )
    completed, _, _ = await workflow.transition(
        "WO-9001", to_state=COMPLETED, changed_by="operator-2", reason="Work completed",
        expected_version=approved.object_version, command_id="complete",
    )

    assert await workflow.current_state("WO-9001") == COMPLETED
    assert [item.transition_id for item in await workflow.history("WO-9001")] == [
        approved.transition_id, completed.transition_id,
    ]
    await store.close()


async def test_illegal_transition_and_unknown_active_work_order_are_rejected(tmp_path: Path):
    store, workflow = await _workflow(tmp_path)
    with pytest.raises(WorkflowTransitionError, match="cannot transition"):
        await workflow.transition(
            "WO-9001", to_state=COMPLETED, changed_by="operator-1", reason="Skip approval",
            expected_version=0, command_id="skip",
        )
    with pytest.raises(WorkflowTransitionError, match="unknown work order"):
        await workflow.current_state("WO-unknown")
    await store.close()


async def test_rdf_projection_keeps_each_transition_as_an_independent_resource(tmp_path: Path):
    store, workflow = await _workflow(tmp_path)
    await workflow.transition(
        "WO-9001", to_state=APPROVED, changed_by="operator-1", reason="Evidence reviewed",
        expected_version=0, command_id="approved",
    )
    graph = await workflow.rdf_projection("WO-9001")

    transitions = set(graph.subjects(RDF.type, ENERGY.OperationalTransition))
    assert len(transitions) == 1
    transition = transitions.pop()
    assert (transition, ENERGY.changesWorkOrder, REC["WO-9001"]) in graph
    assert (transition, ENERGY.toState, ENERGY[APPROVED]) in graph
    await store.close()
