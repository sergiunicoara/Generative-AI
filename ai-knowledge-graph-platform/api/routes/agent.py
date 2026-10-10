"""Agent tool dispatch — the HTTP surface of :class:`ToolPolicy`.

``ToolPolicy`` implements the allowlist, per-tool risk levels, scope
enforcement, argument validation, cross-tenant guard, dry-run mode, timeout
and audit log that the README describes as the agent safety story. It was
fully tested but had no production caller, so none of those guarantees applied
to anything a request could reach. This router is that caller.

Scopes and tenant both come from the caller's signed token, never from the
request body: the policy's scope checks are only meaningful if the caller
cannot choose its own scopes.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.auth.dependencies import get_current_user, get_tenant, require_scope
from graphrag.agents.tool_policy import DeniedAction, ToolPolicy
from graphrag.core.config import get_settings

router = APIRouter()

# Mutating tools need an explicit confirmation on top of their scopes
# (docs/mcp-security.md); read tools that do not carry the caller's ACL context
# are unavailable while document access control is enforced.
_MUTATING_RISKS = {"high", "restricted"}
_ACL_UNSAFE_TOOLS = {"local_search", "global_search", "get_neighbors", "search_graph", "get_community"}


class ToolCallRequest(BaseModel):
    tool: str = Field(min_length=1, max_length=128)
    args: dict = Field(default_factory=dict, max_length=100)
    dry_run: bool = False
    # Explicit human/caller confirmation for mutating tools. A call without it
    # is refused with a preview, so an agent cannot mutate in a single step.
    confirm: bool = False


def _denied(tool: str, reason: str, detail: str, tenant: str, operation_id: str) -> dict:
    return {"outcome": "denied", "tool": tool, "reason": reason, "detail": detail, "tenant": tenant,
            "operation_id": operation_id}


async def _audit(operation_id: str, tool: str, spec, user: dict, tenant: str, args: dict, outcome: str,
                 started: float) -> None:
    import structlog

    from mcp_server.audit import record_audit_event
    event = {
        "operation_id": operation_id, "operation": f"agent.{tool}" if spec else None,
        "requested_name": tool[:200], "kind": getattr(spec, "risk", None), "tenant": tenant,
        "subject": str(user.get("sub") or ""), "outcome": outcome,
        "args_sha256": hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest(),
        "duration_ms": round((time.monotonic() - started) * 1000, 1),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import asyncio
        from mcp_server.audit import AUDIT_TIMEOUT_S
        await asyncio.wait_for(record_audit_event(event, transport="rest"), timeout=AUDIT_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - audit persistence never masks the result
        structlog.get_logger(__name__).error("agent_tool.audit_persist_failed", error=str(exc)[:200])


@router.post(
    "/tool",
    summary="Invoke a registered agent tool through the ToolPolicy gate",
    dependencies=[Depends(require_scope("read"))],
)
async def call_tool(
    request: ToolCallRequest,
    user: dict = Depends(get_current_user),
    tenant: str = Depends(get_tenant),
):
    """Execute one tool call under policy.

    Returns ``{"outcome": "executed", "result": ...}`` on success, or
    ``{"outcome": "denied", ...}`` with the policy's reason. A denial is a
    200 with an explicit outcome rather than an error status: the refusal is
    the product here, and the caller needs the structured reason.
    """
    started = time.monotonic()
    operation_id = str(uuid.uuid4())
    policy = ToolPolicy.from_defaults(
        caller_scopes=user.get("scope", "").split(),
        dry_run=request.dry_run,
    )
    spec = policy._tools.get(request.tool)
    args = dict(request.args)
    outcome = "error"
    try:
        if spec is not None:
            mutating = spec.risk in _MUTATING_RISKS
            # A mutating tool may only act on the caller's own tenant, whatever
            # tenant:<x> scopes the token carries.
            if mutating and args.get("tenant") not in (None, tenant):
                outcome = "tenant_mismatch"
                return _denied(request.tool, outcome, "mutating tools act only on the caller's tenant",
                               tenant, operation_id)
            if mutating:
                args["tenant"] = tenant
            # The actor is the authenticated caller, never a body field.
            if "requested_by" in spec.arg_schema:
                args["requested_by"] = str(user.get("sub") or "unknown")
            if (request.tool in _ACL_UNSAFE_TOOLS
                    and get_settings().access_control.get("enabled", False)):
                outcome = "acl_unsupported"
                return _denied(request.tool, outcome,
                               "this tool does not apply document access control; use /query", tenant,
                               operation_id)
            if mutating and not request.confirm and not request.dry_run:
                outcome = "confirmation_required"
                return _denied(request.tool, outcome,
                               "mutating tool: repeat the call with confirm=true after reviewing it",
                               tenant, operation_id)
        result = await policy.call(request.tool, args, tenant=tenant)
        if isinstance(result, DeniedAction):
            outcome = result.reason
            return {
                "outcome": "denied",
                "tool":    result.tool,
                "reason":  result.reason,
                "detail":  result.detail,
                "tenant":  result.tenant,
                "operation_id": operation_id,
            }
        outcome = "executed"
        return {"outcome": "executed", "tool": request.tool, "result": result, "operation_id": operation_id}
    finally:
        await _audit(operation_id, request.tool, spec, user, tenant, request.args, outcome, started)


@router.get(
    "/tools",
    summary="List the tools this caller is permitted to invoke",
    dependencies=[Depends(require_scope("read"))],
)
async def list_tools(user: dict = Depends(get_current_user)):
    scopes = set(user.get("scope", "").split())
    policy = ToolPolicy.from_defaults(caller_scopes=sorted(scopes))
    return {
        "tools": [
            {
                "name":      spec.name,
                "risk":      spec.risk,
                "scopes":    spec.scopes,
                "permitted": all(s in scopes for s in spec.scopes),
            }
            for spec in policy._tools.values()
        ]
    }


@router.get(
    "/audit",
    summary="Structured audit log of tool calls made by this policy instance",
    dependencies=[Depends(require_scope("read"))],
)
async def tool_audit(limit: int = 100, tenant: str = Depends(get_tenant)):
    """Durable audit of this tenant's agent-tool and MCP capability calls
    (``:CapabilityAuditEvent``; arguments are stored as a hash only)."""
    from mcp_server.audit import list_audit_events
    return {"entries": await list_audit_events(tenant, limit=limit)}
