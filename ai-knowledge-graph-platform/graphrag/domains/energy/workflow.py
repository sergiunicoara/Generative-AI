"""Append-only, tenant-scoped maintenance workflow for the Energy POC.

The source RDF graph retains SAP-shaped facts such as a work order's reported
``energy:status``. This module adds a distinct operational layer: the current
workflow state is derived from immutable transition records, rather than
overwriting source evidence. It is intentionally in-memory for the local POC;
it demonstrates the contract and is not a replacement for an enterprise work
order system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from uuid import uuid4

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from graphrag.domains.energy.demo import ENERGY, REC

REVIEW_REQUIRED = "review_required"
APPROVED = "approved"
COMPLETED = "completed"
_KNOWN_WORK_ORDERS = frozenset({"WO-9001"})
_ALLOWED_TRANSITIONS = {
    REVIEW_REQUIRED: frozenset({APPROVED}),
    APPROVED: frozenset({COMPLETED}),
    COMPLETED: frozenset(),
}


class WorkflowTransitionError(ValueError):
    """Raised when a requested maintenance transition is not legal."""


@dataclass(frozen=True)
class WorkflowTransition:
    transition_id: str
    work_order_id: str
    from_state: str
    to_state: str
    changed_at: str
    changed_by: str
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


class MaintenanceWorkflow:
    """Controlled transitions for the one Energy POC work order.

    Every state change appends one immutable ``WorkflowTransition``. The
    current state is computed from the last transition, so neither a prior
    transition nor source-system evidence is rewritten or deleted.
    """

    def __init__(self) -> None:
        self._transitions: list[WorkflowTransition] = []

    @staticmethod
    def _require_work_order(work_order_id: str) -> None:
        if work_order_id not in _KNOWN_WORK_ORDERS:
            raise WorkflowTransitionError(f"unknown Energy POC work order {work_order_id!r}")

    def current_state(self, work_order_id: str) -> str:
        self._require_work_order(work_order_id)
        for transition in reversed(self._transitions):
            if transition.work_order_id == work_order_id:
                return transition.to_state
        return REVIEW_REQUIRED

    def history(self, work_order_id: str) -> list[WorkflowTransition]:
        self._require_work_order(work_order_id)
        return [item for item in self._transitions if item.work_order_id == work_order_id]

    def transition(self, work_order_id: str, *, to_state: str, changed_by: str, reason: str) -> WorkflowTransition:
        self._require_work_order(work_order_id)
        from_state = self.current_state(work_order_id)
        if to_state not in _ALLOWED_TRANSITIONS[from_state]:
            raise WorkflowTransitionError(
                f"cannot transition {work_order_id} from {from_state!r} to {to_state!r}"
            )
        if not changed_by.strip():
            raise WorkflowTransitionError("changed_by is required")
        if not reason.strip():
            raise WorkflowTransitionError("reason is required")

        record = WorkflowTransition(
            transition_id=uuid4().hex,
            work_order_id=work_order_id,
            from_state=from_state,
            to_state=to_state,
            changed_at=datetime.now(timezone.utc).isoformat(),
            changed_by=changed_by,
            reason=reason,
        )
        self._transitions.append(record)
        return record

    def rdf_projection(self, work_order_id: str) -> Graph:
        """Return a standards-aligned RDF projection of the lifecycle.

        Each state change has its own IRI and remains in the graph. The
        current state is derived by consumers from the final transition; this
        method does not assert a mutable ``status`` property.
        """
        self._require_work_order(work_order_id)
        graph = Graph()
        graph.bind("energy", ENERGY)
        work_order = REC[work_order_id]
        graph.add((work_order, RDF.type, ENERGY.WorkOrder))
        for item in self.history(work_order_id):
            transition = URIRef(f"https://example.energy.demo/workflow/{item.transition_id}")
            graph.add((transition, RDF.type, ENERGY.OperationalTransition))
            graph.add((transition, ENERGY.changesWorkOrder, work_order))
            graph.add((transition, ENERGY.fromState, ENERGY[item.from_state]))
            graph.add((transition, ENERGY.toState, ENERGY[item.to_state]))
            graph.add((transition, ENERGY.changedAt, Literal(item.changed_at, datatype=XSD.dateTime)))
            graph.add((transition, ENERGY.changedBy, Literal(item.changed_by)))
            graph.add((transition, ENERGY.transitionReason, Literal(item.reason)))
        current = ENERGY[self.current_state(work_order_id)]
        graph.add((current, RDF.type, ENERGY.MaintenanceState))
        graph.add((current, RDFS.label, Literal(self.current_state(work_order_id).replace("_", " "))))
        return graph


__all__ = [
    "APPROVED",
    "COMPLETED",
    "MaintenanceWorkflow",
    "REVIEW_REQUIRED",
    "WorkflowTransition",
    "WorkflowTransitionError",
]
