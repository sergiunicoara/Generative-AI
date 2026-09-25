import pytest
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import Mount

from mcp_server.identity import CallerIdentity
from mcp_server.registry import DeniedCapabilityCall
from mcp_server.transport_20260728 import (
    HEADER_MISMATCH, MCP_PROTOCOL_VERSION, ProtocolVersionDispatch, StatelessMCP20260728App,
    _header, _json_type, _tool_schema,
)


class _Spec:
    qualified_name = "kg.example@1.0.0"
    title = "Example"
    arg_schema = {"question": {"type": str, "required": True}, "tenant": {"type": str}}


class _Registry:
    def __init__(self, *, call_result=None, resolve_result=None):
        self.spec = _Spec()
        self.calls = []
        self._call_result = call_result if call_result is not None else {"answer": "grounded"}
        self._resolve_result = resolve_result if resolve_result is not None else self.spec

    def discover(self, _identity):
        return [{"qualified_name": self.spec.qualified_name}]

    def resolve(self, _name, _identity):
        return self._resolve_result

    async def call(self, name, arguments, identity):
        self.calls.append((name, arguments, identity.tenant))
        return self._call_result


def _client(monkeypatch, registry=None):
    app = StatelessMCP20260728App()
    registry = registry if registry is not None else _Registry()
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


def test_20260728_rejects_non_post_methods(monkeypatch):
    client, _ = _client(monkeypatch)
    response = client.get("/mcp", headers=_headers())

    assert response.status_code == 405
    body = response.json()
    assert body["error"]["code"] == -32600
    assert body["error"]["message"] == "POST JSON-RPC required"


def test_20260728_malformed_json_body_is_a_parse_error(monkeypatch):
    client, _ = _client(monkeypatch)
    response = client.post(
        "/mcp", headers=_headers(**{"Mcp-Method": "ping", "Content-Type": "application/json"}),
        content=b"{not json",
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32700
    assert body["error"]["message"] == "Parse error"
    assert body["id"] is None


def test_20260728_missing_jsonrpc_version_is_invalid_request(monkeypatch):
    client, _ = _client(monkeypatch)
    response = client.post("/mcp", headers=_headers(**{"Mcp-Method": "ping"}), json={
        "id": 1, "method": "ping",
    })

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32600
    assert body["error"]["message"] == "Invalid Request"
    assert body["id"] == 1


def test_20260728_non_string_method_is_invalid_request(monkeypatch):
    client, _ = _client(monkeypatch)
    response = client.post("/mcp", headers=_headers(**{"Mcp-Method": "ping"}), json={
        "jsonrpc": "2.0", "id": 1, "method": 123,
    })

    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32600


def test_20260728_non_object_params_is_rejected(monkeypatch):
    client, _ = _client(monkeypatch)
    response = client.post("/mcp", headers=_headers(**{"Mcp-Method": "ping"}), json={
        "jsonrpc": "2.0", "id": 1, "method": "ping", "params": "not-an-object",
    })

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32602
    assert body["error"]["message"] == "params must be an object"


def test_20260728_non_object_arguments_is_rejected(monkeypatch):
    client, _ = _client(monkeypatch)
    response = client.post("/mcp", headers=_headers(**{
        "Mcp-Method": "tools/call", "Mcp-Name": "kg.example@1.0.0",
    }), json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "kg.example@1.0.0", "arguments": "not-an-object"},
    })

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32602
    assert body["error"]["message"] == "arguments must be an object"


def test_20260728_unknown_method_is_method_not_found(monkeypatch):
    client, _ = _client(monkeypatch)
    response = client.post("/mcp", headers=_headers(**{"Mcp-Method": "notamethod"}), json={
        "jsonrpc": "2.0", "id": "x", "method": "notamethod",
    })

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32601
    assert body["error"]["message"] == "Method not found"
    assert body["id"] == "x"


def test_20260728_ping_returns_an_empty_result(monkeypatch):
    client, _ = _client(monkeypatch)
    response = client.post("/mcp", headers=_headers(**{"Mcp-Method": "ping"}), json={
        "jsonrpc": "2.0", "id": 7, "method": "ping",
    })

    assert response.status_code == 200
    body = response.json()
    assert body["result"] == {}
    assert body["id"] == 7


def test_20260728_denied_tool_call_reports_isError_with_reason(monkeypatch):
    denied = DeniedCapabilityCall(capability="kg.example@1.0.0", reason="missing_scope", detail="need read scope")
    client, registry = _client(monkeypatch, registry=_Registry(call_result=denied))
    response = client.post("/mcp", headers=_headers(**{
        "Mcp-Method": "tools/call", "Mcp-Name": "kg.example@1.0.0",
    }), json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "kg.example@1.0.0", "arguments": {"question": "q"}},
    })

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["isError"] is True
    assert result["structuredContent"] == {
        "denied": True, "capability": "kg.example@1.0.0",
        "reason": "missing_scope", "detail": "need read scope",
    }


def test_20260728_tools_list_omits_entries_denied_at_resolve_time(monkeypatch):
    denied = DeniedCapabilityCall(capability="kg.example@1.0.0", reason="tenant_mismatch")
    client, _ = _client(monkeypatch, registry=_Registry(resolve_result=denied))
    response = client.post("/mcp", headers=_headers(**{"Mcp-Method": "tools/list"}), json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })

    assert response.status_code == 200
    assert response.json()["result"]["tools"] == []


def test_header_lookup_is_case_insensitive():
    scope = {"headers": [(b"mcp-protocol-version", b"2026-07-28")]}

    assert _header(scope, b"mcp-protocol-version") == "2026-07-28"


def test_header_lookup_defaults_to_empty_string_when_absent():
    scope = {"headers": [(b"content-type", b"application/json")]}

    assert _header(scope, b"mcp-protocol-version") == ""


def test_json_type_maps_known_python_types():
    assert _json_type(str) == "string"
    assert _json_type(int) == "integer"
    assert _json_type(float) == "number"
    assert _json_type(bool) == "boolean"


def test_json_type_defaults_to_string_for_an_unmapped_type():
    assert _json_type(list) == "string"
    assert _json_type(None) == "string"


class _MultiArgSpec:
    qualified_name = "kg.multi@1.0.0"
    title = "Multi"
    arg_schema = {
        "question": {"type": str, "required": True},
        "top_k": {"type": int, "required": False},
        "tenant": {"type": str, "required": True},
    }


def test_tool_schema_excludes_tenant_and_reports_required_fields():
    schema = _tool_schema(_MultiArgSpec())

    assert set(schema["properties"].keys()) == {"question", "top_k"}
    assert schema["properties"]["question"] == {"type": "string"}
    assert schema["properties"]["top_k"] == {"type": "integer"}
    assert schema["required"] == ["question"]
    assert schema["additionalProperties"] is False


def test_tool_schema_omits_required_key_when_nothing_is_required():
    class _Spec:
        arg_schema = {"limit": {"type": int}}

    schema = _tool_schema(_Spec())

    assert "required" not in schema


async def _collect_body(*chunks: bytes, disconnect: bool = False) -> bytes:
    messages = [
        {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
        for i, chunk in enumerate(chunks)
    ]
    if disconnect:
        messages.append({"type": "http.disconnect"})
    remaining = list(messages)

    async def receive():
        return remaining.pop(0)

    return await StatelessMCP20260728App._body(receive)


@pytest.mark.asyncio
async def test_body_assembles_multiple_chunks_across_more_body_messages():
    body = await _collect_body(b"hello, ", b"world")

    assert body == b"hello, world"


@pytest.mark.asyncio
async def test_body_returns_empty_bytes_on_disconnect_before_completion():
    body = await _collect_body(disconnect=True)

    assert body == b""


@pytest.mark.asyncio
async def test_protocol_version_dispatch_routes_modern_requests_to_modern_app(monkeypatch):
    monkeypatch.setattr(
        CallerIdentity, "current",
        classmethod(lambda cls: CallerIdentity(subject="agent", tenant="acme", authenticated=True)),
    )
    dispatch = ProtocolVersionDispatch(legacy_app=None)
    dispatch.modern_app.registry = _Registry()
    scope = {
        "type": "http", "method": "POST",
        "headers": [(b"mcp-protocol-version", MCP_PROTOCOL_VERSION.encode()), (b"mcp-method", b"ping")],
    }
    body_sent = [{"type": "http.request", "body": b'{"jsonrpc": "2.0", "id": 1, "method": "ping"}', "more_body": False}]
    sent = []

    async def receive():
        return body_sent.pop(0)

    async def send(message):
        sent.append(message)

    await dispatch(scope, receive, send)

    assert sent[0]["status"] == 200


@pytest.mark.asyncio
async def test_protocol_version_dispatch_routes_legacy_requests_to_the_legacy_app():
    calls = []

    async def legacy_app(scope, receive, send):
        calls.append(scope)

    dispatch = ProtocolVersionDispatch(legacy_app=legacy_app)
    scope = {"type": "http", "method": "POST", "headers": []}

    await dispatch(scope, None, None)

    assert calls == [scope]
