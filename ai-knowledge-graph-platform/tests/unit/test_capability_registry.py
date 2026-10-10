"""Unit tests for mcp_server.registry.CapabilityRegistry."""

from __future__ import annotations

import pytest

from mcp_server.identity import CallerIdentity
from mcp_server.registry import CapabilityRegistry, CapabilitySpec, DeniedCapabilityCall


def _without_receipt(result):
    """Phase 7 adds a provenance receipt to dict results (asserted in test_guarded_operations.py)."""
    assert "operation_id" in result["provenance_receipt"]
    return {k: v for k, v in result.items() if k != "provenance_receipt"}


def _identity(**kw) -> CallerIdentity:
    defaults = dict(
        subject="agent-1", tenant="aerospace",
        scopes=frozenset({"read"}), authenticated=True,
    )
    defaults.update(kw)
    return CallerIdentity(**defaults)


async def _echo(tenant: str, value: str = "") -> dict:
    return {"tenant": tenant, "value": value}


def _sync_echo(tenant: str) -> dict:
    return {"tenant": tenant}


def _spec(**kw) -> CapabilitySpec:
    defaults = dict(
        capability_id="kg.echo", version="1.0.0", title="Echo",
        kind="read", risk="safe", fn=_echo,
        arg_schema={"tenant": {"type": str}, "value": {"type": str}},
    )
    defaults.update(kw)
    return CapabilitySpec(**defaults)


class TestRegister:
    def test_duplicate_qualified_name_rejected(self):
        registry = CapabilityRegistry()
        registry.register(_spec())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_spec())

    def test_duplicate_legacy_alias_rejected(self):
        registry = CapabilityRegistry()
        registry.register(_spec(legacy_aliases=("echo",)))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_spec(version="2.0.0", legacy_aliases=("echo",)))


class TestResolve:
    def test_resolves_by_qualified_name(self):
        registry = CapabilityRegistry()
        registry.register(_spec())
        resolved = registry.resolve("kg.echo@1.0.0", _identity())
        assert isinstance(resolved, CapabilitySpec)

    def test_resolves_by_legacy_alias(self):
        registry = CapabilityRegistry()
        registry.register(_spec(legacy_aliases=("echo",)))
        resolved = registry.resolve("echo", _identity())
        assert isinstance(resolved, CapabilitySpec)
        assert resolved.qualified_name == "kg.echo@1.0.0"

    def test_resolves_bare_id_to_latest_version(self):
        registry = CapabilityRegistry()
        registry.register(_spec(version="1.0.0"))
        registry.register(_spec(version="2.1.0"))
        resolved = registry.resolve("kg.echo", _identity())
        assert resolved.version == "2.1.0"

    def test_unknown_name_denied_not_found(self):
        registry = CapabilityRegistry()
        result = registry.resolve("nope", _identity())
        assert isinstance(result, DeniedCapabilityCall)
        assert result.reason == "not_found"

    def test_missing_required_scope_denied(self):
        registry = CapabilityRegistry()
        registry.register(_spec(required_scopes=("biz:write",)))
        result = registry.resolve("kg.echo@1.0.0", _identity(scopes=frozenset({"read"})))
        assert isinstance(result, DeniedCapabilityCall)
        assert result.reason == "missing_scope"


class TestDiscover:
    def test_filters_out_capabilities_caller_lacks_scope_for(self):
        registry = CapabilityRegistry()
        registry.register(_spec())
        registry.register(_spec(capability_id="kg.write", required_scopes=("biz:write",)))
        listing = registry.discover(_identity(scopes=frozenset({"read"})))
        ids = {item["capability_id"] for item in listing}
        assert ids == {"kg.echo"}

    def test_includes_capability_when_scope_granted(self):
        registry = CapabilityRegistry()
        registry.register(_spec(capability_id="kg.write", required_scopes=("biz:write",)))
        listing = registry.discover(_identity(scopes=frozenset({"biz:write"})))
        assert {item["capability_id"] for item in listing} == {"kg.write"}

    async def test_platform_discovery_capability_returns_only_entitled_specs(self):
        from mcp_server.capabilities import build_registry

        registry = build_registry()
        identity = _identity(scopes=frozenset({"read"}))
        result = await registry.call("platform.capabilities.discover@1.0.0", {}, identity)
        assert result["tenant"] == "aerospace"
        ids = {item["capability_id"] for item in result["capabilities"]}
        assert "biz.workorder.create" not in ids
        assert "platform.capabilities.discover" in ids


class TestContractSnapshot:
    def test_is_entitlement_independent_and_sorted(self):
        registry = CapabilityRegistry()
        registry.register(_spec(capability_id="kg.b", required_scopes=("admin",)))
        registry.register(_spec(capability_id="kg.a"))
        snapshot = registry.contract_snapshot()
        assert [item["capability_id"] for item in snapshot] == ["kg.a", "kg.b"]

    def test_snapshot_shape_has_stable_keys(self):
        registry = CapabilityRegistry()
        registry.register(_spec(legacy_aliases=("echo",)))
        snapshot = registry.contract_snapshot()
        assert snapshot[0] == {
            "capability_id": "kg.echo", "version": "1.0.0",
            "qualified_name": "kg.echo@1.0.0", "title": "Echo",
            "kind": "read", "risk": "safe", "required_scopes": [],
            "arg_schema_keys": ["tenant", "value"], "dry_run_ok": True,
            "requires_approval": False, "deprecated": False,
            "replacement": None, "legacy_aliases": ["echo"],
        }


class TestCall:
    async def test_successful_call_injects_identity_tenant(self):
        registry = CapabilityRegistry()
        registry.register(_spec())
        result = await registry.call("kg.echo@1.0.0", {"value": "x"}, _identity(tenant="aerospace"))
        assert _without_receipt(result) == {"tenant": "aerospace", "value": "x"}

    async def test_caller_supplied_tenant_matching_identity_is_allowed(self):
        # validate_args' own cross-tenant guard (shared with ToolPolicy) also
        # requires an explicit tenant:<name> scope whenever 'tenant' is a
        # caller-supplied argument -- a real token carries this after the
        # Wave 1 scope-issuance fix, so the identity needs it here too.
        registry = CapabilityRegistry()
        registry.register(_spec())
        result = await registry.call(
            "kg.echo@1.0.0", {"tenant": "aerospace", "value": "x"},
            _identity(tenant="aerospace", scopes=frozenset({"read", "tenant:aerospace"})),
        )
        assert _without_receipt(result) == {"tenant": "aerospace", "value": "x"}

    async def test_caller_supplied_tenant_mismatch_denied(self):
        registry = CapabilityRegistry()
        registry.register(_spec())
        result = await registry.call(
            "kg.echo@1.0.0", {"tenant": "other-tenant"}, _identity(tenant="aerospace"),
        )
        assert isinstance(result, DeniedCapabilityCall)
        assert result.reason == "tenant_mismatch"

    async def test_unauthenticated_identity_denied(self):
        registry = CapabilityRegistry()
        registry.register(_spec())
        result = await registry.call("kg.echo@1.0.0", {}, CallerIdentity.anonymous())
        assert isinstance(result, DeniedCapabilityCall)
        assert result.reason == "unauthenticated"

    async def test_invalid_argument_denied(self):
        registry = CapabilityRegistry()
        registry.register(_spec())
        result = await registry.call("kg.echo@1.0.0", {"value": 123}, _identity())
        assert isinstance(result, DeniedCapabilityCall)
        assert result.reason == "invalid_arg"

    async def test_dry_run_never_executes(self):
        registry = CapabilityRegistry()
        called = False

        async def _tracked(tenant: str) -> dict:
            nonlocal called
            called = True
            return {"tenant": tenant}

        registry.register(_spec(fn=_tracked))
        result = await registry.call("kg.echo@1.0.0", {}, _identity(), dry_run=True)
        assert isinstance(result, DeniedCapabilityCall)
        assert result.reason == "dry_run"
        assert called is False

    async def test_dry_run_not_allowed_when_spec_disallows_it(self):
        registry = CapabilityRegistry()
        registry.register(_spec(dry_run_ok=False))
        result = await registry.call("kg.echo@1.0.0", {}, _identity(), dry_run=True)
        assert result.reason == "dry_run_not_allowed"

    async def test_supports_synchronous_capability_functions(self):
        registry = CapabilityRegistry()
        registry.register(_spec(fn=_sync_echo, arg_schema={"tenant": {"type": str}}))
        result = await registry.call("kg.echo@1.0.0", {}, _identity(tenant="aerospace"))
        assert _without_receipt(result) == {"tenant": "aerospace"}

    async def test_not_found_denied(self):
        registry = CapabilityRegistry()
        result = await registry.call("nope", {}, _identity())
        assert result.reason == "not_found"
