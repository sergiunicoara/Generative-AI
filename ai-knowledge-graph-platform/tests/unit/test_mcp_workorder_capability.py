"""Unit tests for the biz.workorder.create MCP capability wiring.

Covers `CapabilitySpec.pass_identity` (the one registry extension Wave 6
added), the envelope-building adapter in
`mcp_server/capabilities/workorder_create.py`, and the `create_work_order`
tool wrapper in `mcp_server/server.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp_server.capabilities import build_registry
from mcp_server.capabilities.workorder_create import _create_work_order
from mcp_server.identity import CallerIdentity
from mcp_server.registry import CapabilityRegistry, CapabilitySpec, DeniedCapabilityCall


def _identity(**kw) -> CallerIdentity:
    defaults = dict(
        subject="agent-1", tenant="aerospace",
        scopes=frozenset({"biz:write"}), token_type="m2m", authenticated=True,
    )
    defaults.update(kw)
    return CallerIdentity(**defaults)


class TestPassIdentityPlumbing:
    async def test_registry_injects_identity_only_when_opted_in(self):
        received = {}

        async def _needs_identity(tenant: str, identity) -> dict:
            received["identity"] = identity
            return {"tenant": tenant}

        registry = CapabilityRegistry()
        registry.register(CapabilitySpec(
            capability_id="test.needs-identity", version="1.0.0", title="t",
            kind="write", risk="safe", fn=_needs_identity, pass_identity=True,
            approval_enforced_by="test service",  # Phase 7: writes need an approval path
        ))
        identity = _identity()
        await registry.call("test.needs-identity@1.0.0", {}, identity)
        assert received["identity"] is identity

    async def test_read_capability_unaffected_by_flag_default(self):
        async def _plain(tenant: str) -> dict:
            return {"tenant": tenant}

        registry = CapabilityRegistry()
        registry.register(CapabilitySpec(
            capability_id="test.plain", version="1.0.0", title="t",
            kind="read", risk="safe", fn=_plain,
        ))
        result = await registry.call("test.plain@1.0.0", {}, _identity())
        assert {k: v for k, v in result.items() if k != "provenance_receipt"} == {"tenant": "aerospace"}


class TestCreateWorkOrderAdapter:
    async def test_builds_envelope_with_identity_bound_actor(self):
        fake_receipt = SimpleNamespace(model_dump=lambda mode="json": {"outcome": "executed"})
        with patch("mcp_server.capabilities.workorder_create.WorkOrderService") as mock_cls:
            mock_service = mock_cls.return_value
            mock_service.create_from_finding = AsyncMock(return_value=fake_receipt)

            result = await _create_work_order(
                tenant="aerospace", reason_code="remediation",
                originating_finding_id="finding-1", title="Rotate key",
                identity=_identity(subject="agent-1", token_type="m2m"),
                expected_version=1,
            )

        assert result == {"outcome": "executed"}
        envelope = mock_service.create_from_finding.call_args.args[0]
        assert envelope.actor_id == "agent-1"
        assert envelope.actor_type == "agent"  # m2m token -> agent actor
        assert envelope.tenant == "aerospace"
        assert envelope.args["originating_finding_id"] == "finding-1"

    async def test_human_token_type_maps_to_human_actor(self):
        fake_receipt = SimpleNamespace(model_dump=lambda mode="json": {})
        with patch("mcp_server.capabilities.workorder_create.WorkOrderService") as mock_cls:
            mock_service = mock_cls.return_value
            mock_service.create_from_finding = AsyncMock(return_value=fake_receipt)

            await _create_work_order(
                tenant="aerospace", reason_code="r", originating_finding_id="f-1",
                title="t", identity=_identity(token_type="browser"),
            )

        envelope = mock_service.create_from_finding.call_args.args[0]
        assert envelope.actor_type == "human"


class TestServerToolWrapper:
    async def test_create_work_order_tool_denies_anonymous_caller(self):
        # An anonymous CallerIdentity has both empty scopes and
        # authenticated=False; resolve()'s entitlement check runs before
        # call()'s authentication check, so this is the reason a real
        # unauthenticated MCP session actually observes.
        registry = build_registry()
        anonymous = CallerIdentity.anonymous()
        result = await registry.call(
            "biz.workorder.create@1.0.0",
            {"reason_code": "r", "originating_finding_id": "f-1", "title": "t"},
            anonymous,
        )
        assert isinstance(result, DeniedCapabilityCall)
        assert result.reason == "missing_scope"
