"""Tenant-scoped, durable evidence-remediation workflow facade.

When ``insufficient_evidence`` reports an asset blocked on a missing
observation or work-order status, this module lets an authorised operator
open a governed follow-up task tracking a request for that evidence from its
owning source system. Creating, progressing, or fulfilling a request never
writes to published RDF, source-system fixtures, or the active-version
pointer -- it is purely governance metadata layered over the same durable
store the maintenance workflow uses. By construction, nothing here can turn
an ``insufficient_evidence`` answer into a maintenance conclusion: the
answer is derived solely from published RDF (see ``answers.py``), and this
module never touches that graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from graphrag.domains.energy.governance_store import (
    EVIDENCE_CANCELLED, EVIDENCE_FULFILLED, EVIDENCE_IN_PROGRESS, EVIDENCE_OPEN,
    CommandReuseError, EvidenceRequestConflictError, GovernanceStore,
    UnknownEvidenceRequestError,
)
from graphrag.domains.energy.vocabulary import ASSET, ENERGY

# The only two evidence gaps `insufficient_evidence` can currently detect
# (see evals/energy_demo/sparql/insufficient_evidence.rq), each tied to
# exactly one owning source system. A request naming any other field, or
# pairing a known field with the wrong system, is rejected -- this table is
# the single source of truth for both checks.
MISSING_FIELD_SOURCE_SYSTEMS = {
    "temperature_c": "Snowflake",
    "work_order_status": "SAP",
}


class EvidenceRequestError(ValueError):
    """Public evidence-request error, mapped to a 4xx API response."""


class UnknownAssetError(EvidenceRequestError):
    """The named asset does not exist in the active published dataset."""


class EvidenceRequestConflict(EvidenceRequestError):
    """A stale object_version, reused command_id, or illegal transition was rejected."""


def _local_name(iri: str) -> str:
    return iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def known_asset_ids(graph: Graph) -> set[str]:
    """Local-named ids of every asset in the active published graph."""
    return {_local_name(str(subject)) for subject in graph.subjects(RDF.type, ENERGY.WindTurbine)}


def validate_missing_evidence(query_rows: list[dict[str, str]], *, asset_id: str, missing_field: str) -> None:
    """Re-derive "is this genuinely missing" from `insufficient_evidence`'s own rows.

    Never trusts client input: `query_rows` must come from a fresh call to
    `EnergyDemoService.answer("insufficient_evidence", ...)`. Raises
    `EvidenceRequestError` if the asset is not currently blocked on the
    requested field.
    """
    row_key = {"temperature_c": "hasTemperature", "work_order_status": "hasWorkOrder"}.get(missing_field)
    if row_key is None:
        raise EvidenceRequestError(f"unsupported missing_field {missing_field!r}")
    for row in query_rows:
        if _local_name(str(row.get("asset", ""))) != asset_id:
            continue
        if str(row.get(row_key, "")).lower() != "true":
            return
        raise EvidenceRequestError(
            f"{asset_id} already has {missing_field}; it is not currently missing evidence"
        )
    raise EvidenceRequestError(
        f"{asset_id} is not currently blocked on missing {missing_field}"
    )


@dataclass(frozen=True)
class EvidenceRequestRecord:
    request_id: str
    tenant: str
    asset_id: str
    missing_field: str
    target_source_system: str
    owner: str
    priority: str
    state: str
    reason: str
    created_at: str
    updated_at: str
    created_by: str
    object_version: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRequestTransitionRecord:
    transition_id: str
    request_id: str
    from_state: str
    to_state: str
    changed_at: str
    changed_by: str
    reason: str
    object_version: int

    def as_dict(self) -> dict:
        return asdict(self)


class EvidenceRequestService:
    """Async facade around the durable governance store's evidence-request tables."""

    def __init__(self, store: GovernanceStore, *, tenant: str) -> None:
        self._store = store
        self._tenant = tenant

    @staticmethod
    def _record(data: dict[str, object]) -> EvidenceRequestRecord:
        return EvidenceRequestRecord(
            request_id=str(data["request_id"]), tenant=str(data["tenant"]),
            asset_id=str(data["asset_id"]), missing_field=str(data["missing_field"]),
            target_source_system=str(data["target_source_system"]), owner=str(data["owner"]),
            priority=str(data["priority"]), state=str(data["state"]), reason=str(data["reason"]),
            created_at=str(data["created_at"]), updated_at=str(data["updated_at"]),
            created_by=str(data["created_by"]), object_version=int(data["object_version"]),
        )

    @staticmethod
    def _transition_record(data: dict[str, object]) -> EvidenceRequestTransitionRecord:
        return EvidenceRequestTransitionRecord(
            transition_id=str(data["transition_id"]), request_id=str(data["request_id"]),
            from_state=str(data["from_state"]), to_state=str(data["to_state"]),
            changed_at=str(data["changed_at"]), changed_by=str(data["changed_by"]),
            reason=str(data["reason"]), object_version=int(data["object_version"]),
        )

    async def create(
        self, *, asset_id: str, missing_field: str, source_system: str, owner: str,
        priority: str, reason: str, created_by: str, command_id: str,
        graph: Graph, query_rows: list[dict[str, str]],
    ) -> tuple[EvidenceRequestRecord, str, bool]:
        """Validate against the live dataset, then persist a governed request.

        Validation order: asset existence, field/source-system mapping, then
        that the field is genuinely missing right now -- each a distinct 4xx
        cause. None of this touches `graph`; it is read-only evidence for the
        decision to accept or reject the request.
        """
        if asset_id not in known_asset_ids(graph):
            raise UnknownAssetError(f"unknown asset {asset_id!r} in the active published dataset")
        expected_system = MISSING_FIELD_SOURCE_SYSTEMS.get(missing_field)
        if expected_system is None:
            raise EvidenceRequestError(f"unsupported missing_field {missing_field!r}")
        if source_system != expected_system:
            raise EvidenceRequestError(
                f"{missing_field} is owned by {expected_system}, not {source_system!r}"
            )
        validate_missing_evidence(query_rows, asset_id=asset_id, missing_field=missing_field)
        try:
            item, response_json, replayed = await self._store.create_evidence_request(
                self._tenant, asset_id=asset_id, missing_field=missing_field,
                target_source_system=source_system, owner=owner, priority=priority,
                reason=reason, created_by=created_by, command_id=command_id,
            )
        except (EvidenceRequestConflictError, CommandReuseError) as exc:
            raise EvidenceRequestConflict(str(exc)) from exc
        return self._record(item), response_json, replayed

    async def list(self) -> list[EvidenceRequestRecord]:
        items = await self._store.evidence_requests(self._tenant)
        return [self._record(item) for item in items]

    async def get(self, request_id: str) -> tuple[EvidenceRequestRecord, list[EvidenceRequestTransitionRecord]]:
        try:
            item = await self._store.evidence_request(self._tenant, request_id)
            history = await self._store.evidence_request_transition_history(self._tenant, request_id)
        except UnknownEvidenceRequestError as exc:
            raise EvidenceRequestError(str(exc)) from exc
        return self._record(item), [self._transition_record(entry) for entry in history]

    async def transition(
        self, request_id: str, *, to_state: str, changed_by: str, reason: str,
        expected_version: int, command_id: str,
    ) -> tuple[EvidenceRequestTransitionRecord, str, bool]:
        try:
            item, response_json, replayed = await self._store.transition_evidence_request(
                self._tenant, request_id, to_state=to_state, changed_by=changed_by,
                reason=reason, expected_version=expected_version, command_id=command_id,
            )
        except (UnknownEvidenceRequestError, EvidenceRequestConflictError, CommandReuseError) as exc:
            raise EvidenceRequestError(str(exc)) from exc
        return self._transition_record(item), response_json, replayed

    async def rdf_projection(self, request_id: str) -> Graph:
        """Governance metadata as RDF, kept separate from source RDF.

        Never merged into the published/validated dataset -- mirrors
        `MaintenanceWorkflow.rdf_projection`.
        """
        record, history = await self.get(request_id)
        graph = Graph()
        graph.bind("energy", ENERGY)
        request_node = URIRef(f"https://example.energy.demo/evidence-request/{record.request_id}")
        graph.add((request_node, RDF.type, ENERGY.EvidenceRequest))
        graph.add((request_node, ENERGY.forAsset, ASSET[record.asset_id]))
        graph.add((request_node, ENERGY.missingField, Literal(record.missing_field)))
        graph.add((request_node, ENERGY.targetSourceSystem, Literal(record.target_source_system)))
        graph.add((request_node, ENERGY.assignedTo, Literal(record.owner)))
        graph.add((request_node, ENERGY.priority, Literal(record.priority)))
        graph.add((request_node, ENERGY.state, Literal(record.state)))
        graph.add((request_node, ENERGY.reason, Literal(record.reason)))
        for item in history:
            transition = URIRef(f"https://example.energy.demo/evidence-request-transition/{item.transition_id}")
            graph.add((transition, RDF.type, ENERGY.EvidenceRequestTransition))
            graph.add((transition, ENERGY.changesRequest, request_node))
            graph.add((transition, ENERGY.fromState, Literal(item.from_state)))
            graph.add((transition, ENERGY.toState, Literal(item.to_state)))
            graph.add((transition, ENERGY.changedAt, Literal(item.changed_at, datatype=XSD.dateTime)))
            graph.add((transition, ENERGY.changedBy, Literal(item.changed_by)))
            graph.add((transition, ENERGY.transitionReason, Literal(item.reason)))
        current = ENERGY[record.state]
        graph.add((current, RDF.type, ENERGY.MaintenanceState))
        graph.add((current, RDFS.label, Literal(record.state.replace("_", " "))))
        return graph


__all__ = [
    "EVIDENCE_CANCELLED", "EVIDENCE_FULFILLED", "EVIDENCE_IN_PROGRESS", "EVIDENCE_OPEN",
    "EvidenceRequestConflict", "EvidenceRequestError", "EvidenceRequestRecord",
    "EvidenceRequestService", "EvidenceRequestTransitionRecord", "MISSING_FIELD_SOURCE_SYSTEMS",
    "UnknownAssetError", "known_asset_ids", "validate_missing_evidence",
]
