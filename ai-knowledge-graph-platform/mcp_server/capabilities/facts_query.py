"""kg.facts.query@1.0.0 -- controlled natural-language graph fact lookups.

Re-admitted from the orphaned `mcp_server/tools.py::query_graph_facts`. It
wraps the already-hardened `graphrag.graph.controlled_query` planner --
tenant-parameterized, accepts no raw Cypher or SPARQL from the caller, and
already has unit coverage -- so admitting it costs no new attack surface.

The remaining previously orphaned read tools (`query_knowledge_graph` and
`lookup_entity`) were later admitted through their own versioned capability
modules after their tenant and invocation contracts were covered by tests.
"""

from __future__ import annotations

from mcp_server.registry import CapabilityRegistry, CapabilitySpec
from mcp_server.tools import query_graph_facts


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        capability_id="kg.facts.query",
        version="1.0.0",
        title="Controlled natural-language graph fact query",
        kind="read",
        risk="safe",
        fn=query_graph_facts,
        required_scopes=("read",),
        arg_schema={
            "question": {"type": str, "required": True, "max_length": 500},
            "tenant": {"type": str},
            "limit": {"type": int, "min": 1, "max": 100},
        },
    ))
