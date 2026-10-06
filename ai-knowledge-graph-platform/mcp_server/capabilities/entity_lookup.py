"""``kg.entity.lookup`` -- tenant-bound entity resolution and evidence lookup."""

from __future__ import annotations

from mcp_server.registry import CapabilityRegistry, CapabilitySpec
from mcp_server.tools import lookup_entity


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        capability_id="kg.entity.lookup",
        version="1.0.0",
        title="Resolve an entity and return tenant-scoped graph evidence",
        kind="read",
        risk="safe",
        fn=lookup_entity,
        required_scopes=("read",),
        arg_schema={
            "name": {"type": str, "required": True},
            "tenant": {"type": str},
            "as_of": {"type": str},
            "limit": {"type": int, "min": 1, "max": 100},
        },
    ))
