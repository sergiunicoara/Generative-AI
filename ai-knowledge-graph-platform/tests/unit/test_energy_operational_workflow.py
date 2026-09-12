from __future__ import annotations

import pytest
from rdflib.namespace import RDF

from graphrag.domains.energy.demo import ENERGY, REC
from graphrag.domains.energy.workflow import (
    APPROVED,
    COMPLETED,
    MaintenanceWorkflow,
    REVIEW_REQUIRED,
    WorkflowTransitionError,
)


def test_transitions_are_append_only_and_current_state_is_derived():
    workflow = MaintenanceWorkflow()

    assert workflow.current_state("WO-9001") == REVIEW_REQUIRED
    approved = workflow.transition("WO-9001", to_state=APPROVED, changed_by="operator-1", reason="Evidence reviewed")
    completed = workflow.transition("WO-9001", to_state=COMPLETED, changed_by="operator-2", reason="Work completed")

    assert workflow.current_state("WO-9001") == COMPLETED
    assert [item.transition_id for item in workflow.history("WO-9001")] == [
        approved.transition_id, completed.transition_id,
    ]


def test_illegal_transition_and_unknown_work_order_are_rejected():
    workflow = MaintenanceWorkflow()
    with pytest.raises(WorkflowTransitionError, match="cannot transition"):
        workflow.transition("WO-9001", to_state=COMPLETED, changed_by="operator-1", reason="Skip approval")
    with pytest.raises(WorkflowTransitionError, match="unknown"):
        workflow.current_state("WO-unknown")


def test_rdf_projection_keeps_each_transition_as_an_independent_resource():
    workflow = MaintenanceWorkflow()
    workflow.transition("WO-9001", to_state=APPROVED, changed_by="operator-1", reason="Evidence reviewed")

    graph = workflow.rdf_projection("WO-9001")

    transitions = set(graph.subjects(RDF.type, ENERGY.OperationalTransition))
    assert len(transitions) == 1
    transition = transitions.pop()
    assert (transition, ENERGY.changesWorkOrder, REC["WO-9001"]) in graph
    assert (transition, ENERGY.toState, ENERGY[APPROVED]) in graph
