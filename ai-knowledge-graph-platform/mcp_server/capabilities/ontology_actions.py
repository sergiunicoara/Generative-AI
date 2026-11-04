"""Ontology actions and functions as governed capabilities (ADR-0015).

* ``kg.action.propose`` -- propose a declared action on one object; the
  result is the recorded proposal (rejected, pending approval, applied or
  stale).
* ``kg.action.decide`` -- approve or reject a pending proposal (never the
  proposer's own).
* ``kg.action.list`` -- the tenant's proposals, optionally by status.
* ``kg.function.call`` -- evaluate a read-only function on one object.

Approval is enforced by ``ActionService`` (declared per action, four-eyes),
so the registry-level approval hook is not used for these write capabilities.
The tenant is the caller's identity tenant; the action's own required scope is
checked against the identity's scopes in addition to the capability scope.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from graphrag.semantic_model import load_model
from graphrag.semantic_model.action_service import ActionError, ActionService
from mcp_server.registry import CapabilityRegistry, CapabilitySpec

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "ontology" / "models"
STATUSES = ["rejected", "pending_approval", "approved", "applied", "denied", "stale"]


@lru_cache(maxsize=8)
def _model(model_id: str):
    path = MODELS_DIR / f"{model_id}.yaml"
    if not path.is_file():
        raise ActionError(f"unknown model {model_id!r}")
    return load_model(path)


def action_models() -> list[str]:
    return sorted(p.stem for p in MODELS_DIR.glob("*.yaml") if _model(p.stem).actions or _model(p.stem).functions)


async def _tables(tenant: str, domain: str):
    from graphrag.context_graph.knowledge_service import get_knowledge_service

    catalog = await (await get_knowledge_service()).catalog(tenant, domain)
    return {table.id: table for table in catalog.decision_tables}


async def _service(model_id: str) -> ActionService:
    from graphrag.context_graph.knowledge_service import get_knowledge_service
    from graphrag.graph.neo4j_client import get_neo4j

    store = (await get_knowledge_service())._store
    return ActionService(_model(model_id), store, get_neo4j(), tables_provider=_tables)


def _error(exc: Exception) -> dict[str, Any]:
    kind = "forbidden" if isinstance(exc, PermissionError) else "invalid_request"
    return {"error": kind, "detail": str(exc)}


async def _propose(tenant: str, model: str, action: str, object_key: str, parameters: dict | None = None,
                   identity=None) -> dict[str, Any]:
    try:
        service = await _service(model)
        return await service.propose(
            tenant, action, object_key, parameters or {}, proposed_by=identity.subject,
            actor_type="agent" if identity.token_type == "m2m" else "human", scopes=frozenset(identity.scopes))
    except (ActionError, PermissionError) as exc:
        return _error(exc)


async def _decide(tenant: str, model: str, proposal_id: str, decision: str, reason: str = "",
                  identity=None) -> dict[str, Any]:
    try:
        service = await _service(model)
        return await service.decide(tenant, proposal_id, approve=decision == "approve", decided_by=identity.subject,
                                    reason=reason, scopes=frozenset(identity.scopes))
    except (ActionError, PermissionError) as exc:
        return _error(exc)


async def _list(tenant: str, model: str, status: str | None = None, limit: int = 50) -> dict[str, Any]:
    try:
        service = await _service(model)
    except ActionError as exc:
        return _error(exc)
    return {"proposals": [p for p in await service.proposals(tenant, status=status, limit=limit)
                          if p.get("model_id") == model]}


async def _function(tenant: str, model: str, function: str, object_key: str) -> dict[str, Any]:
    try:
        return await (await _service(model)).call_function(tenant, function, object_key)
    except ActionError as exc:
        return _error(exc)


_IDENT = {"type": str, "required": True, "max_length": 64, "pattern": r"^[A-Za-z_][A-Za-z0-9_]*$"}
_MODEL = {"type": str, "required": True, "max_length": 128, "pattern": r"^[a-z0-9][a-z0-9-]*$"}
_KEY = {"type": str, "required": True, "max_length": 512}
_APPROVAL = "ActionService (declared per action, four-eyes, guarded write)"


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        capability_id="kg.action.propose", version="1.0.0",
        title="Propose a declared ontology action on one object (checked, approved when required, logged)",
        kind="write", risk="moderate", fn=_propose, required_scopes=("write",),
        arg_schema={"tenant": {"type": str}, "model": _MODEL, "action": _IDENT, "object_key": _KEY,
                    "parameters": {"type": dict}},
        requires_approval=True, approval_enforced_by=_APPROVAL, pass_identity=True,
        read_only_session=False))
    registry.register(CapabilitySpec(
        capability_id="kg.action.decide", version="1.0.0",
        title="Approve or reject a pending ontology action proposed by someone else",
        kind="write", risk="moderate", fn=_decide, required_scopes=("write",),
        arg_schema={"tenant": {"type": str}, "model": _MODEL,
                    "proposal_id": {"type": str, "required": True, "pattern": r"^AP-[0-9a-f]{24}$"},
                    "decision": {"type": str, "required": True, "allowed": ["approve", "reject"]},
                    "reason": {"type": str, "max_length": 2000}},
        requires_approval=True, approval_enforced_by=_APPROVAL, pass_identity=True,
        read_only_session=False))
    registry.register(CapabilitySpec(
        capability_id="kg.action.list", version="1.0.0",
        title="List ontology action proposals of the tenant",
        kind="read", risk="safe", fn=_list, required_scopes=("read",),
        arg_schema={"tenant": {"type": str}, "model": _MODEL,
                    "status": {"type": str, "allowed": STATUSES}, "limit": {"type": int, "min": 1, "max": 200}}))
    registry.register(CapabilitySpec(
        capability_id="kg.function.call", version="1.0.0",
        title="Evaluate a read-only ontology function on one object",
        kind="read", risk="safe", fn=_function, required_scopes=("read",),
        arg_schema={"tenant": {"type": str}, "model": _MODEL, "function": _IDENT, "object_key": _KEY}))
