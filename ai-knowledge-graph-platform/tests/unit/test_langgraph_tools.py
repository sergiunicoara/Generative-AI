"""LangGraph adapter over the capability registry: same result, scopes,
tenant binding and audit as the MCP path."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from mcp_server.langgraph_tools import registry_tools, tool_name  # noqa: E402
from mcp_server.registry import CapabilitySpec  # noqa: E402
from tests.unit.test_guarded_operations import ident, read_spec, reg  # noqa: E402


async def _run_graph(tools, name, args):
    def agent(_state):
        return {"messages": [AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1"}])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=False))
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "tools")
    graph.add_edge("tools", END)
    out = await graph.compile().ainvoke({"messages": []})
    return out["messages"][-1]


def _strip(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in ("provenance_receipt", "scope")}


async def test_a_langgraph_call_matches_the_registry_call_and_is_audited():
    registry, audit = reg(read_spec())
    tools = registry_tools(registry, ident())
    assert [t.name for t in tools] == ["kg_test_read_1_0_0"]
    assert "tenant" not in tools[0].args, "the tenant is bound, never an argument"

    message = await _run_graph(tools, "kg_test_read_1_0_0", {"name": "pump", "limit": 3})
    via_graph = message.artifact
    assert json.loads(message.content)["name"] == "pump", "the model receives JSON"
    direct = await registry.call("kg.test.read@1.0.0", {"name": "pump", "limit": 3}, ident())
    assert _strip(via_graph) == _strip(direct) == {"tenant": "acme", "name": "pump", "limit": 3}
    assert [e["outcome"] for e in audit] == ["executed", "executed"]
    assert audit[0]["operation"] == audit[1]["operation"] == "kg.test.read@1.0.0"


async def test_capabilities_without_the_scope_are_not_offered():
    write = CapabilitySpec(capability_id="kg.test.write", version="1.0.0", title="w", kind="write", risk="moderate",
                           fn=lambda tenant: {}, required_scopes=("write",))
    registry, _ = reg(read_spec(), write)
    assert [t.name for t in registry_tools(registry, ident())] == ["kg_test_read_1_0_0"]


async def test_a_tool_cannot_reach_another_tenant_or_bypass_validation():
    registry, audit = reg(read_spec())
    tool = registry_tools(registry, ident())[0]
    _, crossed = await tool.coroutine(name="pump", tenant="other")
    assert crossed["denied"] is True and crossed["reason"] == "tenant_mismatch"
    _, invalid = await tool.coroutine(limit=1000)
    assert invalid["denied"] is True and invalid["reason"] == "invalid_arg"
    assert len(audit) == 2 and all(event["outcome"] != "executed" for event in audit), audit


def test_tool_names_are_provider_safe():
    assert tool_name("kg.graph.stats@1.0.0") == "kg_graph_stats_1_0_0"
