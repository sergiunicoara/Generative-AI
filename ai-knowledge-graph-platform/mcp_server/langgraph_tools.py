"""LangGraph / LangChain tools over the governed capability registry.

A thin adapter for external LangGraph agents: each capability the caller is
entitled to becomes a ``StructuredTool`` whose call goes through
``CapabilityRegistry.call``, so scopes, tenant binding (the tenant comes from
the bound identity and is never a tool argument), argument validation, size
limits, approval rules and audit are exactly those of the MCP and GraphQL
paths. Orchestration inside this platform does not use LangGraph.

``langchain-core`` is imported lazily: it is present in the development
environment (pulled in by ``langchain``) but not in the API image.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mcp_server.identity import CallerIdentity
from mcp_server.registry import CapabilityRegistry, DeniedCapabilityCall
from mcp_server.transport_20260728 import _result_payload, _tool_schema

AUDIT_TRANSPORT = "langgraph"
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def tool_name(qualified_name: str) -> str:
    """Model providers accept ``[A-Za-z0-9_-]`` only: ``kg.graph.stats@1.0.0`` → ``kg_graph_stats_1_0_0``."""
    return _UNSAFE.sub("_", qualified_name)[:64]


def langgraph_registry() -> CapabilityRegistry:
    """The production registry with audit events labelled ``transport=langgraph``."""
    from mcp_server.capabilities import build_registry

    return build_registry(audit_transport=AUDIT_TRANSPORT)


def registry_tools(registry: CapabilityRegistry, identity: CallerIdentity) -> list[Any]:
    """Tools for the capabilities ``identity`` may discover, bound to that identity."""
    from langchain_core.tools import StructuredTool

    tools = []
    seen: dict[str, str] = {}
    for entry in registry.discover(identity):
        spec = registry.resolve(entry["qualified_name"], identity)
        if isinstance(spec, DeniedCapabilityCall):
            continue
        name = tool_name(spec.qualified_name)
        if name in seen:
            raise ValueError(f"tool name collision: {spec.qualified_name} and {seen[name]}")
        seen[name] = spec.qualified_name
        tools.append(StructuredTool.from_function(
            coroutine=_caller(registry, identity, spec.qualified_name),
            name=name,
            description=spec.title,
            args_schema=_tool_schema(spec),
            # The model reads JSON; the unmodified payload stays on ToolMessage.artifact.
            response_format="content_and_artifact",
        ))
    return tools


def _caller(registry: CapabilityRegistry, identity: CallerIdentity, qualified_name: str):
    async def call(**arguments: Any) -> tuple[str, dict[str, Any]]:
        payload, _denied = _result_payload(await registry.call(qualified_name, arguments, identity))
        return json.dumps(payload, default=str, sort_keys=True), payload

    return call


__all__ = ["AUDIT_TRANSPORT", "langgraph_registry", "registry_tools", "tool_name"]
