"""Edge-case coverage for the stateless MCP 2026-07-28 adapter.

Added after a mutation campaign (docs/REMAINING_AUDIT_BACKLOG.md #8) showed half of the adapter's
mutants surviving: error codes and messages, status codes, header rules, schema mapping, result
wrapping and chunked-body handling were not pinned.
"""
from __future__ import annotations

import asyncio
import datetime

import pytest

from mcp_server import transport_20260728 as t
from mcp_server.registry import DeniedCapabilityCall
from mcp_server.transport_20260728 import HEADER_MISMATCH, MCP_PROTOCOL_VERSION, StatelessMCP20260728App
from tests.unit.test_mcp_transport_20260728 import _client, _headers

NAME = "kg.example@1.0.0"


def _post(client, headers, body=None, **kw):
    if body is not None:
        kw["json"] = body
    return client.post("/mcp", headers=_headers(**headers), **kw)


def _rpc(method, **extra):
    return {"jsonrpc": "2.0", "id": 7, "method": method, **extra}


def test_error_envelope_shape_and_protocol_header(monkeypatch):
    client, _ = _client(monkeypatch)
    r = _post(client, {"Mcp-Method": "nope"}, _rpc("nope"))
    assert r.status_code == 400 and r.headers["mcp-protocol-version"] == MCP_PROTOCOL_VERSION
    assert r.json() == {"jsonrpc": "2.0", "id": 7, "error": {"code": -32601, "message": "Method not found"}}


def test_parse_error_is_400_with_null_id(monkeypatch):
    client, _ = _client(monkeypatch)
    r = _post(client, {"Mcp-Method": "x"}, content=b"{not json")
    assert r.status_code == 400
    assert r.json() == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
    assert _post(client, {"Mcp-Method": "x"}, content=b"\xff\xfe").json()["error"]["code"] == -32700


def test_only_post_is_accepted(monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.get("/mcp", headers=_headers())
    assert r.status_code == 405
    assert r.json()["error"] == {"code": -32600, "message": "POST JSON-RPC required"}


@pytest.mark.parametrize("body", [
    [1, 2], "text", {"jsonrpc": "1.0", "id": 5, "method": "ping"}, {"id": 5, "method": "ping"},
    {"jsonrpc": "2.0", "id": 5, "method": 3}, {"jsonrpc": "2.0", "id": 5},
])
def test_invalid_request_shapes(monkeypatch, body):
    client, _ = _client(monkeypatch)
    r = _post(client, {"Mcp-Method": "ping"}, body)
    assert r.status_code == 400
    assert r.json()["error"] == {"code": -32600, "message": "Invalid Request"}
    assert r.json()["id"] == (body.get("id") if isinstance(body, dict) else None)


def test_ping_returns_empty_result(monkeypatch):
    client, _ = _client(monkeypatch)
    r = _post(client, {"Mcp-Method": "ping"}, _rpc("ping"))
    assert r.status_code == 200 and r.json() == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_header_mismatch_error_text(monkeypatch):
    client, _ = _client(monkeypatch)
    r = _post(client, {"Mcp-Method": "ping"}, _rpc("tools/list"))
    assert r.json()["error"] == {"code": HEADER_MISMATCH, "message": "Mcp-Method does not match JSON-RPC method"}
    assert HEADER_MISMATCH == -32020
    assert _post(client, {}, _rpc("ping")).json()["error"]["code"] == HEADER_MISMATCH   # header absent


def test_params_must_be_an_object(monkeypatch):
    client, _ = _client(monkeypatch)
    r = _post(client, {"Mcp-Method": "ping"}, _rpc("ping", params=[1]))
    assert r.status_code == 400
    assert r.json()["error"] == {"code": -32602, "message": "params must be an object"}
    assert _post(client, {"Mcp-Method": "ping"}, _rpc("ping")).status_code == 200      # params optional


@pytest.mark.parametrize("params,headers", [
    ({"arguments": {}}, {"Mcp-Name": "x"}),                       # name missing
    ({"name": 5, "arguments": {}}, {"Mcp-Name": "5"}),           # name not a string
    ({"name": "a", "arguments": {}}, {"Mcp-Name": "b"}),         # header mismatch
    ({"name": "a", "arguments": {}}, {}),                        # header absent
])
def test_tools_call_name_header_rules(monkeypatch, params, headers):
    client, registry = _client(monkeypatch)
    r = _post(client, {"Mcp-Method": "tools/call", **headers}, _rpc("tools/call", params=params))
    assert r.status_code == 400
    assert r.json()["error"] == {"code": HEADER_MISMATCH, "message": "Mcp-Name does not match tools/call name"}
    assert registry.calls == []


def test_tools_call_arguments_must_be_an_object_and_default_to_empty(monkeypatch):
    client, registry = _client(monkeypatch)
    h = {"Mcp-Method": "tools/call", "Mcp-Name": NAME}
    r = _post(client, h, _rpc("tools/call", params={"name": NAME, "arguments": "no"}))
    assert r.status_code == 400
    assert r.json()["error"] == {"code": -32602, "message": "arguments must be an object"}
    assert registry.calls == []
    r = _post(client, h, _rpc("tools/call", params={"name": NAME}))
    assert r.status_code == 200 and registry.calls == [(NAME, {}, "acme")]


def test_tool_result_is_serialised_sorted_and_flagged(monkeypatch):
    client, registry = _client(monkeypatch)

    async def call(name, arguments, identity):
        return {"b": 1, "a": {"when": datetime.date(2026, 1, 2)}}
    registry.call = call
    h = {"Mcp-Method": "tools/call", "Mcp-Name": NAME}
    r = _post(client, h, _rpc("tools/call", params={"name": NAME, "arguments": {}}))
    result = r.json()["result"]
    assert result["isError"] is False
    assert result["content"] == [{"type": "text", "text": '{"a": {"when": "2026-01-02"}, "b": 1}'}]
    assert result["structuredContent"] == {"a": {"when": "2026-01-02"}, "b": 1}
    assert r.json()["id"] == 7


def test_non_dict_results_are_wrapped_and_denials_are_errors(monkeypatch):
    client, registry = _client(monkeypatch)
    h = {"Mcp-Method": "tools/call", "Mcp-Name": NAME}
    body = _rpc("tools/call", params={"name": NAME, "arguments": {}})

    async def scalar(name, arguments, identity):
        return [1, 2]
    registry.call = scalar
    out = _post(client, h, body).json()["result"]
    assert out["structuredContent"] == {"result": [1, 2]} and out["isError"] is False

    async def denied(name, arguments, identity):
        return DeniedCapabilityCall(capability=NAME, reason="scope", detail="needs read")
    registry.call = denied
    out = _post(client, h, body).json()["result"]
    assert out["isError"] is True
    assert out["structuredContent"] == {"denied": True, "capability": NAME, "reason": "scope", "detail": "needs read"}


def test_result_payload_helper_directly():
    assert t._result_payload({"x": 1}) == ({"x": 1}, False)
    assert t._result_payload("s") == ({"result": "s"}, False)
    d = DeniedCapabilityCall(capability="c", reason="r", detail="d")
    assert t._result_payload(d) == ({"denied": True, "capability": "c", "reason": "r", "detail": "d"}, True)


def test_tools_list_skips_denied_specs_and_describes_tools(monkeypatch):
    client, registry = _client(monkeypatch)
    registry.discover = lambda identity: [{"qualified_name": "ok"}, {"qualified_name": "hidden"}]

    def resolve(name, identity):
        return registry.spec if name == "ok" else DeniedCapabilityCall(capability=name, reason="r", detail="d")
    registry.resolve = resolve
    tools = _post(client, {"Mcp-Method": "tools/list"}, _rpc("tools/list")).json()["result"]["tools"]
    assert [x["name"] for x in tools] == [NAME]
    assert tools[0]["title"] == "Example" and tools[0]["description"] == "Example"
    assert tools[0]["inputSchema"] == {"type": "object", "properties": {"question": {"type": "string"}},
                                       "additionalProperties": False, "required": ["question"]}


def test_json_type_mapping():
    assert [t._json_type(x) for x in (str, int, float, bool, None, bytes)] == \
        ["string", "integer", "number", "boolean", "string", "string"]


def test_tool_schema_required_only_when_present_and_tenant_hidden():
    class S:
        arg_schema = {"a": {"type": int}, "b": {"type": bool, "required": False},
                      "tenant": {"type": str, "required": True}}
    assert t._tool_schema(S) == {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}},
                                 "additionalProperties": False}

    class R:
        arg_schema = {"a": {"type": float, "required": True}, "b": {"type": str, "required": True},
                      "c": {"type": str}}
    assert t._tool_schema(R)["required"] == ["a", "b"]


def test_header_lookup_is_case_insensitive_latin1_and_defaults_empty():
    scope = {"headers": [(b"mcp-method", b"tools/list"), (b"x-name", "café".encode("latin-1"))]}
    assert t._header(scope, b"mcp-method") == "tools/list"
    assert t._header(scope, b"x-name") == "café"
    assert t._header(scope, b"missing") == ""
    assert t._header({}, b"mcp-method") == ""
    assert t._header({"headers": [(b"Mcp-Method", b"x")]}, b"mcp-method") == "x"


def _feed(*messages):
    it = iter(messages)

    async def receive():
        return next(it)
    return receive


def test_body_is_assembled_from_chunks_and_a_disconnect_yields_nothing():
    app = StatelessMCP20260728App.__new__(StatelessMCP20260728App)
    chunks = _feed({"type": "http.request", "body": b"ab", "more_body": True},
                   {"type": "lifespan.noise"},
                   {"type": "http.request", "body": b"cd", "more_body": False})
    assert asyncio.run(app._body(chunks)) == b"abcd"
    cut = _feed({"type": "http.request", "body": b"ab", "more_body": True}, {"type": "http.disconnect"})
    assert asyncio.run(app._body(cut)) == b""
    assert asyncio.run(app._body(_feed({"type": "http.request"}))) == b""


def test_dispatch_routes_by_protocol_version_header():
    seen = []

    async def legacy(scope, receive, send):
        seen.append("legacy")

    async def modern(scope, receive, send):
        seen.append("modern")
    d = t.ProtocolVersionDispatch(legacy)
    d.modern_app = modern

    def run(headers):
        asyncio.run(d({"type": "http", "headers": headers}, None, None))
    run([(b"mcp-protocol-version", MCP_PROTOCOL_VERSION.encode())])
    run([(b"mcp-protocol-version", b"2025-06-18")])
    run([])
    assert seen == ["modern", "legacy", "legacy"]


def test_module_constants_and_exports():
    assert MCP_PROTOCOL_VERSION == "2026-07-28"
    assert t.__all__ == ["HEADER_MISMATCH", "MCP_PROTOCOL_VERSION", "ProtocolVersionDispatch",
                         "StatelessMCP20260728App"]
