"""Approval-gated compensation for ``biz.workorder.create@1.0.0``.

This adapter intentionally contains no business mutation logic.  It binds
the MCP caller identity to a command envelope and delegates all idempotency,
approval, version checks, atomic state changes, and receipts to the shared
business service used by the HTTP API.
"""

from __future__ import annotations

from graphrag.business.commands import CommandEnvelope
from graphrag.business.service import WORKORDER_COMPENSATE_CAPABILITY, WorkOrderService
from graphrag.graph.neo4j_client import get_neo4j
from mcp_server.registry import CapabilityRegistry, CapabilitySpec


async def _compensate_work_order(
    tenant: str,
    reason_code: str,
    work_order_id: str,
    original_command_id: str,
    expected_version: int,
    expected_finding_version: int,
    identity,
    *,
    dry_run: bool = False,
    approval_id: str | None = None,
    command_id: str | None = None,
    correlation_id: str = "",
) -> dict:
    envelope = CommandEnvelope(
        **({"command_id": command_id} if command_id else {}),
        capability=WORKORDER_COMPENSATE_CAPABILITY,
        tenant=tenant, actor_id=identity.subject,
        actor_type="agent" if identity.token_type == "m2m" else "human",
        reason_code=reason_code,
        args={
            "work_order_id": work_order_id,
            "original_command_id": original_command_id,
            "expected_finding_version": expected_finding_version,
        },
        expected_version=expected_version, dry_run=dry_run,
        approval_id=approval_id, correlation_id=correlation_id,
    )
    receipt = await WorkOrderService(get_neo4j()).compensate_work_order(envelope)
    return receipt.model_dump(mode="json")


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        capability_id="biz.workorder.compensate",
        version="1.0.0",
        title="Cancel a work order and reopen its finding through an approved compensation",
        kind="write",
        risk="destructive",
        fn=_compensate_work_order,
        required_scopes=("biz:write",),
        arg_schema={
            "reason_code": {"type": str, "required": True},
            "work_order_id": {"type": str, "required": True},
            "original_command_id": {"type": str, "required": True},
            "expected_version": {"type": int, "required": True, "min": 1},
            "expected_finding_version": {"type": int, "required": True, "min": 1},
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
