"""``platform.capabilities.discover`` -- entitlement-filtered discovery."""

from __future__ import annotations

from mcp_server.registry import CapabilityRegistry, CapabilitySpec


def register(registry: CapabilityRegistry) -> None:
    async def _discover(tenant: str, identity) -> dict:
        # The injected tenant is included for traceability, while the registry
        # itself filters each returned capability by the bound caller scopes.
        return {"tenant": tenant, "capabilities": registry.discover(identity)}

    registry.register(CapabilitySpec(
        capability_id="platform.capabilities.discover",
        version="1.0.0",
        title="Discover MCP capabilities available to the authenticated caller",
        kind="read",
        risk="safe",
        fn=_discover,
        required_scopes=("read",),
        arg_schema={},
        dry_run_ok=True,
        pass_identity=True,
    ))
