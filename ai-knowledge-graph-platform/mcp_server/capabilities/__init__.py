"""Capability registrations for the GraphRAG MCP server.

Each sibling module registers one or more `CapabilitySpec`s into a shared
`CapabilityRegistry`. `build_registry()` assembles the complete registry --
this package is the single place that knows the full set of capabilities
exposed over MCP, at what version, under what scopes.
"""

from __future__ import annotations

from mcp_server.registry import CapabilityRegistry


def build_registry() -> CapabilityRegistry:
    from mcp_server.audit import neo4j_audit_sink

    registry = CapabilityRegistry(audit_sink=neo4j_audit_sink)
    from mcp_server.capabilities.answer_query import register as register_answer_query
    from mcp_server.capabilities.context_precedent import register as register_context_precedent
    from mcp_server.capabilities.discover import register as register_discover
    from mcp_server.capabilities.entity_lookup import register as register_entity_lookup
    from mcp_server.capabilities.facts_query import register as register_facts_query
    from mcp_server.capabilities.graph_stats import register as register_graph_stats
    from mcp_server.capabilities.workorder_create import register as register_workorder_create
    from mcp_server.capabilities.workorder_compensate import register as register_workorder_compensate

    register_graph_stats(registry)
    register_facts_query(registry)
    register_answer_query(registry)
    register_entity_lookup(registry)
    register_context_precedent(registry)
    register_workorder_create(registry)
    register_workorder_compensate(registry)
    register_discover(registry)
    return registry
