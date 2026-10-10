"""biz.workorder.create@1.0.0 -- create a remediation WorkOrder from a
ComplianceFinding, over MCP.

The only mutating capability exposed by this server. Unlike the read
capabilities in this package, it needs the caller's full identity (not just
an asserted `tenant`) to bind `actor_id`/`actor_type` on the
`CommandEnvelope` it builds -- so it opts into `CapabilitySpec.pass_identity`,
the one registry extension Wave 6 needed on top of Wave 4's foundation.

All of the actual safety machinery (schema validation, entitlement, policy
escalation, optimistic concurrency, atomic execution, idempotent receipts)
lives in `graphrag.business.service.WorkOrderService` and is exercised
identically whether the caller arrived over HTTP
(`api/routes/business.py`) or here over MCP -- this module is a thin
envelope-building adapter, not a second implementation of the write path.
"""

from __future__ import annotations

from graphrag.business.commands import CommandEnvelope
from graphrag.business.service import WORKORDER_CREATE_CAPABILITY, WorkOrderService
from graphrag.graph.neo4j_client import get_neo4j
from mcp_server.registry import CapabilityRegistry, CapabilitySpec


async def _create_work_order(
    tenant: str,
    reason_code: str,
    originating_finding_id: str,
    title: str,
    identity,
    *,
    description: str = "",
    assignee: str = "",
    expected_version: int | None = None,
    dry_run: bool = False,
    approval_id: str | None = None,
    command_id: str | None = None,
    correlation_id: str = "",
) -> dict:
    envelope = CommandEnvelope(
        **({"command_id": command_id} if command_id else {}),
        capability=WORKORDER_CREATE_CAPABILITY,
        tenant=tenant,
        actor_id=identity.subject,
        actor_type="agent" if identity.token_type == "m2m" else "human",
        reason_code=reason_code,
        args={
            "originating_finding_id": originating_finding_id,
            "title": title,
            "description": description,
            "assignee": assignee,
        },
        expected_version=expected_version,
        dry_run=dry_run,
        approval_id=approval_id,
        correlation_id=correlation_id,
    )
    service = WorkOrderService(get_neo4j())
    receipt = await service.create_from_finding(envelope)
    return receipt.model_dump(mode="json")


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        capability_id="biz.workorder.create",
        version="1.0.0",
        title="Create a remediation work order from a compliance finding",
        kind="write",
        risk="moderate",
        fn=_create_work_order,
        required_scopes=("biz:write",),
        arg_schema={
            "reason_code": {"type": str, "required": True},
            "originating_finding_id": {"type": str, "required": True},
            "title": {"type": str, "required": True},
            "description": {"type": str},
            "assignee": {"type": str},
            "expected_version": {"type": int, "min": 1},
            "dry_run": {"type": bool},
            "approval_id": {"type": str},
            "command_id": {"type": str},
            "correlation_id": {"type": str},
        },
        dry_run_ok=True,
        requires_approval=True,
        approval_enforced_by="WorkOrderService (governed command with approval)",
        pass_identity=True,
    ))
