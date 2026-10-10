"""Persistent audit of guarded graph operations (docs/mcp-security.md).

Every capability call (MCP) and agent tool call (REST) produces one
``(:CapabilityAuditEvent)`` with the operation id, operation, outcome, subject,
tenant, a hash of the arguments (never the raw arguments, which may contain
user text) and the duration. Denied calls are audited too, including calls to
fabricated operation names. Persistence is best effort: a failure is logged
and never changes the call's result.
"""
from __future__ import annotations

AUDIT_TIMEOUT_S = 2.0

_CREATE = """
CREATE (:CapabilityAuditEvent {
    tenant: $tenant, operation_id: $operation_id, operation: $operation,
    requested_name: $requested_name, kind: $kind, subject: $subject, outcome: $outcome,
    args_sha256: $args_sha256, duration_ms: $duration_ms, transport: $transport,
    recorded_at: datetime($recorded_at)
})
"""

_LIST = """
MATCH (a:CapabilityAuditEvent {tenant: $tenant})
RETURN a.operation_id AS operation_id, a.operation AS operation, a.requested_name AS requested_name,
       a.kind AS kind, a.subject AS subject, a.outcome AS outcome, a.duration_ms AS duration_ms,
       a.transport AS transport, toString(a.recorded_at) AS recorded_at
ORDER BY a.recorded_at DESC LIMIT $limit
"""


async def record_audit_event(event: dict, *, transport: str, neo4j=None) -> None:
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j = neo4j or get_neo4j()
    await neo4j.run(
        _CREATE,
        tenant=event.get("tenant") or "_unauthenticated",
        operation_id=event.get("operation_id"), operation=event.get("operation"),
        requested_name=event.get("requested_name"), kind=event.get("kind"),
        subject=event.get("subject") or "", outcome=event.get("outcome"),
        args_sha256=event.get("args_sha256"), duration_ms=event.get("duration_ms"),
        transport=transport, recorded_at=event.get("recorded_at"),
    )


async def neo4j_audit_sink(event: dict) -> None:
    # Time-boxed: an unreachable database must not stall every tool call.
    import asyncio
    await asyncio.wait_for(record_audit_event(event, transport="mcp"), timeout=AUDIT_TIMEOUT_S)


async def list_audit_events(tenant: str, *, limit: int = 100, neo4j=None) -> list[dict]:
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j = neo4j or get_neo4j()
    run = getattr(neo4j, "run_read", None) or neo4j.run
    return await run(_LIST, tenant=tenant, limit=max(1, min(int(limit), 1000)))
