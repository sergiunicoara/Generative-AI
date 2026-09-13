"""Tenant-scoped, durable maintenance workflow facade.

The workflow never mutates source-system RDF. It projects the append-only
transition log held by ``GovernanceStore``; work-order existence is looked up
from the active published RDF version, not a Python allow-list.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from graphrag.domains.energy.governance_store import (
    APPROVED, CANCELLED, COMPLETED, REJECTED, REVIEW_REQUIRED,
    CommandReuseError, GovernanceStore, UnknownWorkOrderError, WorkflowConflictError,
)
from graphrag.domains.energy.vocabulary import ENERGY, REC


class WorkflowTransitionError(ValueError):
    """Public workflow error, mapped to a conflict or not-found API response."""


@dataclass(frozen=True)
class WorkflowTransition:
    transition_id: str
    work_order_id: str
    from_state: str
    to_state: str
    changed_at: str
    changed_by: str
    reason: str
    object_version: int

    def as_dict(self) -> dict:
        return asdict(self)


class MaintenanceWorkflow:
    """Async facade around the durable governance store."""

    def __init__(self, store: GovernanceStore, *, tenant: str) -> None:
        self._store = store
        self._tenant = tenant

    @staticmethod
    def _record(data: dict[str, object]) -> WorkflowTransition:
        return WorkflowTransition(
            transition_id=str(data["transition_id"]), work_order_id=str(data["work_order_id"]),
            from_state=str(data["from_state"]), to_state=str(data["to_state"]),
            changed_at=str(data["changed_at"]), changed_by=str(data["changed_by"]),
            reason=str(data["reason"]), object_version=int(data["object_version"]),
        )

    async def current(self, work_order_id: str) -> tuple[str, int]:
        try:
            return await self._store.current_state(self._tenant, work_order_id)
        except (UnknownWorkOrderError, WorkflowConflictError) as exc:
            raise WorkflowTransitionError(str(exc)) from exc

    async def current_state(self, work_order_id: str) -> str:
        return (await self.current(work_order_id))[0]

    async def history(self, work_order_id: str) -> list[WorkflowTransition]:
        try:
            items = await self._store.transition_history(self._tenant, work_order_id)
        except UnknownWorkOrderError as exc:
            raise WorkflowTransitionError(str(exc)) from exc
        return [self._record(item) for item in items]

    async def transition(
        self, work_order_id: str, *, to_state: str, changed_by: str, reason: str,
        expected_version: int, command_id: str,
    ) -> tuple[WorkflowTransition, str, bool]:
        try:
            item, response_json, replayed = await self._store.transition(
                self._tenant, work_order_id, to_state=to_state, changed_by=changed_by,
                reason=reason, expected_version=expected_version, command_id=command_id,
            )
        except (UnknownWorkOrderError, WorkflowConflictError, CommandReuseError) as exc:
            raise WorkflowTransitionError(str(exc)) from exc
        return self._record(item), response_json, replayed

    async def rdf_projection(self, work_order_id: str) -> Graph:
        """Standards-aligned RDF view; previous transitions are never overwritten."""
        history = await self.history(work_order_id)
        state, _version = await self.current(work_order_id)
        graph = Graph()
        graph.bind("energy", ENERGY)
        work_order = REC[work_order_id]
        graph.add((work_order, RDF.type, ENERGY.WorkOrder))
        for item in history:
            transition = URIRef(f"https://example.energy.demo/workflow/{item.transition_id}")
            graph.add((transition, RDF.type, ENERGY.OperationalTransition))
            graph.add((transition, ENERGY.changesWorkOrder, work_order))
            graph.add((transition, ENERGY.fromState, ENERGY[item.from_state]))
            graph.add((transition, ENERGY.toState, ENERGY[item.to_state]))
            graph.add((transition, ENERGY.changedAt, Literal(item.changed_at, datatype=XSD.dateTime)))
            graph.add((transition, ENERGY.changedBy, Literal(item.changed_by)))
            graph.add((transition, ENERGY.transitionReason, Literal(item.reason)))
        current = ENERGY[state]
        graph.add((current, RDF.type, ENERGY.MaintenanceState))
        graph.add((current, RDFS.label, Literal(state.replace("_", " "))))
        return graph


__all__ = [
    "APPROVED", "CANCELLED", "COMPLETED", "MaintenanceWorkflow", "REJECTED",
    "REVIEW_REQUIRED", "WorkflowTransition", "WorkflowTransitionError",
]
