"""Propose, approve and apply ontology actions; call ontology functions (ADR-0015).

Flow for one action:

1. ``propose`` reads the target object from the graph (its model label and key
   property, scoped by tenant), runs ``check_action`` and records the proposal
   with the snapshot, parameters, verdict, model version and action time.
   A failed check is recorded as ``rejected``. A passing action that does not
   need approval is applied at once; otherwise it waits as ``pending_approval``.
2. ``decide`` lets a different person approve or reject (four-eyes). Approval
   re-reads the object: if a guarded property changed, or the action no longer
   passes, the proposal becomes ``stale`` and nothing is written.
3. The write is one Cypher statement guarded on the properties the decision
   depended on, inside a ``CorpusMutation``; zero matched rows means ``stale``.

Every transition is appended to the proposal record and to an append-only
event table, so the log shows who proposed, what was checked, who approved
and what was written. ``replay`` re-runs the recorded check.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Awaitable, Callable, Mapping

import structlog

from graphrag.context_graph.decision_tables import DecisionTable
from graphrag.core.tenancy import require_tenant
from graphrag.semantic_model.actions import (
    ObjectSnapshot,
    UnknownFunctionError,
    check_action,
    counted_relations,
    function_value,
)
from graphrag.semantic_model.models import SemanticModel, key_property

log = structlog.get_logger(__name__)

PROPOSAL_SCHEMA = "ontology-action-proposal/v1"
_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

TablesProvider = Callable[[str, str], Awaitable[Mapping[str, DecisionTable]]]


class ActionError(ValueError):
    """The request cannot be served (unknown object, action or proposal)."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _plain(value: Any) -> Any:
    """Neo4j temporal values to ISO strings, so snapshots are JSON and replayable."""
    for attribute in ("iso_format", "isoformat"):
        method = getattr(value, attribute, None)
        if callable(method) and not isinstance(value, (str, bytes)):
            return method()
    return value


def model_digest(model: SemanticModel) -> str:
    payload = model.model_dump(mode="json", exclude={"source_path", "source_lines"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _safe(name: str) -> str:
    if not _SAFE.fullmatch(name):
        raise ActionError(f"unsafe identifier {name!r}")
    return name


class ActionService:
    def __init__(self, model: SemanticModel, store, neo4j, *, tables_provider: TablesProvider | None = None,
                 clock: Callable[[], str] = _utcnow):
        self.model = model
        self._store = store
        self._neo4j = neo4j
        self._tables_provider = tables_provider
        self._clock = clock

    # -- reading ---------------------------------------------------------------------

    def _label(self, type_name: str) -> str:
        return _safe(self.model.lpg_label(type_name, self.model.types[type_name]))

    async def snapshot(self, tenant: str, type_name: str, object_key: str) -> ObjectSnapshot | None:
        key = key_property(self.model, type_name)
        if key is None:
            raise ActionError(f"{type_name} has no single key property")
        counts = []
        for index, (relation, direction) in enumerate(counted_relations(self.model, type_name)):
            spec = self.model.relations[relation]
            rel = _safe(self.model.lpg_relation(relation, spec))
            pattern = f"(n)-[:{rel}]->()" if direction == "out" else f"(n)<-[:{rel}]-()"
            counts.append(f"'{relation}:{direction}': COUNT {{ {pattern} }}")
        rows = await self._neo4j.run(
            f"MATCH (n:{self._label(type_name)} {{tenant: $tenant, {_safe(key)}: $key}}) "
            f"RETURN properties(n) AS properties, {{{', '.join(counts)}}} AS counts LIMIT 2",
            tenant=tenant, key=object_key,
        )
        rows = list(rows or [])
        if not rows:
            return None
        if len(rows) > 1:
            raise ActionError(f"{type_name} {object_key!r} is not unique for the tenant")
        properties = {k: _plain(v) for k, v in dict(rows[0].get("properties") or {}).items() if k != "tenant"}
        return ObjectSnapshot(type_name, object_key, properties,
                              {k: int(v) for k, v in dict(rows[0].get("counts") or {}).items()})

    async def call_function(self, tenant: str, function: str, object_key: str) -> dict[str, Any]:
        tenant = require_tenant(tenant)
        spec = self.model.functions.get(function)
        if spec is None:
            raise ActionError(f"unknown function {function!r}")
        snapshot = await self.snapshot(tenant, spec.applies_to, object_key)
        if snapshot is None:
            raise ActionError(f"no {spec.applies_to} {object_key!r}")
        try:
            value = function_value(self.model, function, snapshot)
        except UnknownFunctionError as exc:
            raise ActionError(f"unknown function {function!r}") from exc
        return {"function": function, "type": spec.applies_to, "object_key": object_key, "value": value}

    async def _tables(self, tenant: str, action: str) -> Mapping[str, DecisionTable]:
        gate = self.model.actions[action].decision_gate
        if gate is None or self._tables_provider is None:
            return {}
        try:
            return await self._tables_provider(tenant, gate.domain)
        except Exception as exc:  # no active release: the gate fails closed in check_action
            log.warning("ontology_action.tables_unavailable", tenant=tenant, domain=gate.domain, error=type(exc).__name__)
            return {}

    # -- proposing and deciding -----------------------------------------------------------

    async def propose(self, tenant: str, action: str, object_key: str, parameters: Mapping[str, Any], *,
                      proposed_by: str, actor_type: str = "human", scopes: frozenset[str] = frozenset({"write"})) -> dict:
        tenant = require_tenant(tenant)
        spec = self.model.actions.get(action)
        if spec is None:
            raise ActionError(f"unknown action {action!r}")
        if spec.required_scope not in scopes:
            raise PermissionError(f"action {action} needs scope {spec.required_scope!r}")
        snapshot = await self.snapshot(tenant, spec.applies_to, object_key)
        if snapshot is None:
            raise ActionError(f"no {spec.applies_to} {object_key!r}")
        now = self._clock()
        tables = await self._tables(tenant, action)
        check = check_action(self.model, action, snapshot, dict(parameters), tenant=tenant, now=now, tables=tables)
        status = "rejected" if not check.allowed else ("pending_approval" if spec.requires_approval else "approved")
        body = {
            "schema_version": PROPOSAL_SCHEMA, "tenant": tenant, "model_id": self.model.id,
            "model_version": self.model.version, "model_sha256": model_digest(self.model),
            "action": action, "type": spec.applies_to, "object_key": object_key,
            "parameters": dict(parameters), "proposed_by": proposed_by, "actor_type": actor_type,
            "checked_at": now,
            "snapshot": {"properties": snapshot.properties, "relation_counts": snapshot.relation_counts},
            "tables": {key: table.model_dump(mode="json") for key, table in tables.items()},
            "check": check.as_dict(),
        }
        record = body | {
            "proposal_id": "AP-" + hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:24],
            "status": status, "requires_approval": spec.requires_approval,
            "events": [{"to": status, "actor": proposed_by, "at": now,
                        "detail": {"violations": [v["code"] for v in check.as_dict()["violations"]]}}],
        }
        await self._store.record_action_proposal(record)
        log.info("ontology_action.proposed", tenant=tenant, action=action, status=status,
                 proposal_id=record["proposal_id"])
        if status == "approved":
            return await self._apply(tenant, record, actor=proposed_by)
        return record

    async def decide(self, tenant: str, proposal_id: str, *, approve: bool, decided_by: str, reason: str = "",
                     scopes: frozenset[str] = frozenset({"write"})) -> dict:
        tenant = require_tenant(tenant)
        record = await self._store.action_proposal(tenant, proposal_id)
        if record is None:
            raise ActionError(f"no proposal {proposal_id!r}")
        spec = self.model.actions.get(record["action"])
        if spec is None or record["model_id"] != self.model.id:
            raise ActionError("the proposal belongs to another model or a removed action")
        if spec.required_scope not in scopes:
            raise PermissionError(f"action {record['action']} needs scope {spec.required_scope!r}")
        if decided_by == record["proposed_by"]:
            raise PermissionError("a proposer cannot decide their own proposal")
        now = self._clock()
        if not approve:
            return await self._store.transition_action_proposal(
                tenant, proposal_id, from_status={"pending_approval"}, to_status="denied", actor=decided_by, at=now,
                detail={"reason": reason})
        record = await self._store.transition_action_proposal(
            tenant, proposal_id, from_status={"pending_approval"}, to_status="approved", actor=decided_by, at=now,
            detail={"reason": reason})
        return await self._apply(tenant, record, actor=decided_by)

    async def _apply(self, tenant: str, record: dict, *, actor: str) -> dict:
        from graphrag.graph.corpus_revision import CorpusMutation

        action, object_key = record["action"], record["object_key"]
        spec = self.model.actions[action]
        now = self._clock()
        fresh = await self.snapshot(tenant, spec.applies_to, object_key)
        recorded_guard = record["check"]["guard"]
        reason = ""
        if fresh is None:
            reason = "object no longer exists"
        else:
            if {k: fresh.properties.get(k) for k in recorded_guard} != recorded_guard:
                reason = "a property the decision depended on changed"
            else:
                tables = await self._tables(tenant, action)
                recheck = check_action(self.model, action, fresh, record["parameters"], tenant=tenant,
                                       now=record["checked_at"], tables=tables)
                if not recheck.allowed:
                    reason = "the action no longer passes its checks"
        if reason:
            return await self._store.transition_action_proposal(
                tenant, record["proposal_id"], from_status={"approved"}, to_status="stale", actor=actor, at=now,
                detail={"reason": reason})

        changes = dict(record["check"]["changes"])
        key = key_property(self.model, spec.applies_to)
        params: dict[str, Any] = {"tenant": tenant, "key": object_key}
        conditions, assignments = [], []
        for index, (name, expected) in enumerate(sorted(recorded_guard.items())):
            if expected is None:
                conditions.append(f"n.{_safe(name)} IS NULL")
            else:
                params[f"g{index}"] = expected
                conditions.append(f"n.{_safe(name)} = $g{index}")
        for index, (name, value) in enumerate(sorted(changes.items())):
            params[f"v{index}"] = value
            datatype = self.model.effective_properties(spec.applies_to)[name].datatype
            target = f"datetime($v{index})" if datatype == "dateTime" else (
                f"date($v{index})" if datatype == "date" else f"$v{index}")
            assignments.append(f"n.{_safe(name)} = {target}")
        cypher = (f"MATCH (n:{self._label(spec.applies_to)} {{tenant: $tenant, {_safe(key or '')}: $key}}) "
                  + (f"WHERE {' AND '.join(conditions)} " if conditions else "")
                  + f"SET {', '.join(assignments)} RETURN count(n) AS updated")
        async with CorpusMutation(self._neo4j, tenant, reason=f"ontology_action.{action}"):
            rows = await self._neo4j.run(cypher, **params)
        updated = int((list(rows or [{}]) or [{}])[0].get("updated") or 0)
        if updated != 1:
            return await self._store.transition_action_proposal(
                tenant, record["proposal_id"], from_status={"approved"}, to_status="stale", actor=actor, at=now,
                detail={"reason": "the object changed before the write"})
        log.info("ontology_action.applied", tenant=tenant, action=action, proposal_id=record["proposal_id"])
        return await self._store.transition_action_proposal(
            tenant, record["proposal_id"], from_status={"approved"}, to_status="applied", actor=actor, at=now,
            detail={"changes": changes})

    async def proposals(self, tenant: str, *, status: str | None = None, limit: int = 100) -> list[dict]:
        return await self._store.action_proposals(require_tenant(tenant), status=status, limit=limit)


def replay(model: SemanticModel, record: dict) -> bool:
    """Re-run the recorded check: same model content, snapshot, parameters,
    tables and action time must give the same verdict, violations and changes."""
    if record.get("model_sha256") != model_digest(model):
        return False
    snapshot = ObjectSnapshot(record["type"], record["object_key"], record["snapshot"]["properties"],
                              record["snapshot"]["relation_counts"])
    tables = {key: DecisionTable.model_validate(value) for key, value in (record.get("tables") or {}).items()}
    check = check_action(model, record["action"], snapshot, record["parameters"], tenant=record["tenant"],
                         now=record["checked_at"], tables=tables)
    return json.loads(json.dumps(check.as_dict(), default=str)) == record["check"]


__all__ = ["ActionError", "ActionService", "PROPOSAL_SCHEMA", "model_digest", "replay"]
