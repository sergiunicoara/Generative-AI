from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import Mount

from mcp_server.identity import CallerIdentity
from mcp_server.transport_20260728 import HEADER_MISMATCH, MCP_PROTOCOL_VERSION, StatelessMCP20260728App


class _Spec:
    qualified_name = "kg.example@1.0.0"
    title = "Example"
    arg_schema = {"question": {"type": str, "required": True}, "tenant": {"type": str}}


class _Registry:
    def __init__(self):
        self.spec = _Spec()
        self.calls = []

    def discover(self, _identity):
        return [{"qualified_name": self.spec.qualified_name}]

    def resolve(self, _name, _identity):
        return self.spec

    async def call(self, name, arguments, identity):
        self.calls.append((name, arguments, identity.tenant))
        return {"answer": "grounded"}


def _client(monkeypatch):
    app = StatelessMCP20260728App()
    registry = _Registry()
    app.registry = registry
    monkeypatch.setattr(
        CallerIdentity, "current",
        classmethod(lambda cls: CallerIdentity(subject="agent", tenant="acme", authenticated=True)),
    )
    return TestClient(Starlette(routes=[Mount("/mcp", app=app)])), registry


def _headers(**extra):
    return {"MCP-Protocol-Version": MCP_PROTOCOL_VERSION, **extra}


def test_20260728_tools_list_is_stateless_and_hides_tenant_parameter(monkeypatch):
    client, _ = _client(monkeypatch)
    response = client.post("/mcp", headers=_headers(**{"Mcp-Method": "tools/list"}), json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })

    assert response.status_code == 200
    assert response.headers["mcp-protocol-version"] == MCP_PROTOCOL_VERSION
    tool = response.json()["result"]["tools"][0]
    assert tool["name"] == "kg.example@1.0.0"
    assert "tenant" not in tool["inputSchema"]["properties"]


def test_20260728_rejects_mismatched_routing_headers_before_tool_call(monkeypatch):
    client, registry = _client(monkeypatch)
    response = client.post("/mcp", headers=_headers(**{"Mcp-Method": "tools/list"}), json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "kg.example@1.0.0", "arguments": {"question": "q"}},
    })

    assert response.status_code == 400
    assert response.json()["error"]["code"] == HEADER_MISMATCH
    assert registry.calls == []


def test_20260728_tool_call_requires_matching_name_header(monkeypatch):
    client, registry = _client(monkeypatch)
    response = client.post("/mcp", headers=_headers(**{
        "Mcp-Method": "tools/call", "Mcp-Name": "kg.example@1.0.0",
    }), json={
        "jsonrpc": "2.0", "id": "call-1", "method": "tools/call",
        "params": {"name": "kg.example@1.0.0", "arguments": {"question": "q"}},
    })

    assert response.status_code == 200
    assert registry.calls == [("kg.example@1.0.0", {"question": "q"}, "acme")]
    assert response.json()["result"]["structuredContent"] == {"answer": "grounded"}
