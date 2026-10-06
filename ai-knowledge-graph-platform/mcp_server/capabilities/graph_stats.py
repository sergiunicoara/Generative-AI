"""kg.graph.stats@1.0.0 -- entity/edge/community/document counts for a tenant.

The compatibility-adapter proof case for the capability registry: same
implementation and wire behavior as the original bare `graph_stats` MCP
tool, re-registered under a dotted, versioned id with `graph_stats` kept as
a `legacy_alias` -- an existing MCP client registration
(`.claude/settings.json`, `claude mcp add graphrag`) needs no change.

The one real behavior change is `tenant`'s role. It used to be *authority*:
whatever value the caller passed was trusted outright. Now it is an
*assertion*, enforced by `CapabilityRegistry.call()` before this function
ever runs: a caller-supplied `tenant` that disagrees with the identity-bound
tenant is denied, and the value actually passed to `_graph_stats` is always
the identity's own tenant.
"""

from __future__ import annotations

from graphrag.graph.neo4j_client import get_neo4j
from mcp_server.registry import CapabilityRegistry, CapabilitySpec


async def _graph_stats(tenant: str) -> dict:
    neo4j = get_neo4j()
    rows = await neo4j.run(
        """
        CALL {
            MATCH (e:Entity {tenant: $tenant}) RETURN count(e) AS entities
        }
        CALL {
            MATCH ()-[r:RELATES_TO {tenant: $tenant}]->() RETURN count(r) AS edges
        }
        CALL {
            MATCH (c:Community {tenant: $tenant}) RETURN count(c) AS communities
        }
        CALL {
            MATCH (d:Document {tenant: $tenant}) RETURN count(d) AS documents
        }
        RETURN entities, edges, communities, documents
        """,
        tenant=tenant,
    )
    return rows[0] if rows else {"entities": 0, "edges": 0, "communities": 0, "documents": 0}


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        capability_id="kg.graph.stats",
        version="1.0.0",
        title="Knowledge graph tenant statistics",
        kind="read",
        risk="safe",
        fn=_graph_stats,
        required_scopes=("read",),
        arg_schema={"tenant": {"type": str}},
        legacy_aliases=("graph_stats",),
    ))
