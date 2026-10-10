"""``cg.precedent.find`` -- retrieve outcome-backed Context Graph memory."""

from __future__ import annotations

from graphrag.context_graph.repository import ContextGraphRepository
from graphrag.graph.neo4j_client import get_neo4j
from mcp_server.registry import CapabilityRegistry, CapabilitySpec


async def _find_precedents(tenant: str, policy_version_id: str, *, limit: int = 10) -> list[dict]:
    """Return tenant-scoped, explainable precedents for one policy version.

    This is deliberately a read capability. Actions, outcomes, and feedback
    continue to enter through their explicit Context Graph write endpoints,
    which preserve human approval/audit semantics rather than letting an agent
    silently rewrite its own memory.
    """
    return await ContextGraphRepository(get_neo4j()).find_precedents(
        tenant, policy_version_id, limit,
    )


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        capability_id="cg.precedent.find",
        version="1.0.0",
        title="Find outcome-backed Context Graph precedents",
        kind="read",
        risk="moderate",
        fn=_find_precedents,
        required_scopes=("read",),
        arg_schema={
            "policy_version_id": {"type": str, "required": True, "pattern": r"^[\w.:\-]{1,128}$"},
            "tenant": {"type": str},
            "limit": {"type": int, "min": 1, "max": 50},
        },
    ))
