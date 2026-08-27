"""Stateless MCP 2026-07-28 adapter over the governed capability registry.

The installed Python MCP SDK is still 1.x and retains session-oriented
Streamable HTTP. This adapter implements the 2026-07-28 stateless core for
``tools/list`` and ``tools/call`` while retaining the SDK app as a legacy path.
Both paths invoke the same entitlement-filtered registry.
"""

from __future__ import annotations

import json
from typing import Any

from mcp_server.capabilities import build_registry
from mcp_server.identity import CallerIdentity
from mcp_server.registry import DeniedCapabilityCall
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MCP_PROTOCOL_VERSION = "2026-07-28"
HEADER_MISMATCH = -32020


def _header(scope: Scope, name: bytes) -> str:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1")
    return ""


def _json_type(value: type | None) -> str:
    return {str: "string", int: "integer", float: "number", bool: "boolean"}.get(value, "string")


def _tool_schema(spec) -> dict[str, Any]:
    properties = {
        name: {"type": _json_type(rule.get("type"))}
        for name, rule in spec.arg_schema.items() if name != "tenant"
    }
    required = [name for name, rule in spec.arg_schema.items() if name != "tenant" and rule.get("required")]
    schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


def _result_payload(result: object) -> tuple[dict[str, Any], bool]:
    if isinstance(result, DeniedCapabilityCall):
        return ({"denied": True, "capability": result.capability, "reason": result.reason, "detail": result.detail}, True)
    return (result if isinstance(result, dict) else {"result": result}, False)


class StatelessMCP20260728App:
    """ASGI app for a single stateless modern MCP JSON-RPC request."""

    def __init__(self) -> None:
        self.registry = build_registry()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self._send_error(send, None, -32600, "POST JSON-RPC required", status=405)
            return
        body = await self._body(receive)
        try:
            message = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            await self._send_error(send, None, -32700, "Parse error", status=400)
            return
        request_id = message.get("id") if isinstance(message, dict) else None
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            await self._send_error(send, request_id, -32600, "Invalid Request", status=400)
            return
        method = message["method"]
        if _header(scope, b"mcp-method") != method:
            await self._send_error(send, request_id, HEADER_MISMATCH, "Mcp-Method does not match JSON-RPC method", status=400)
            return
        params = message.get("params", {})
        if not isinstance(params, dict):
            await self._send_error(send, request_id, -32602, "params must be an object", status=400)
            return
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str) or _header(scope, b"mcp-name") != name:
                await self._send_error(send, request_id, HEADER_MISMATCH, "Mcp-Name does not match tools/call name", status=400)
                return
            arguments = params.get("arguments", {})
            if not isinstance(arguments, dict):
                await self._send_error(send, request_id, -32602, "arguments must be an object", status=400)
                return
            result = await self.registry.call(name, arguments, CallerIdentity.current())
            payload, is_error = _result_payload(result)
            await self._send_result(send, request_id, {
                "content": [{"type": "text", "text": json.dumps(payload, sort_keys=True, default=str)}],
                "structuredContent": payload, "isError": is_error,
            })
            return
        if method == "tools/list":
            tools = []
            identity = CallerIdentity.current()
            for entry in self.registry.discover(identity):
                spec = self.registry.resolve(entry["qualified_name"], identity)
                if isinstance(spec, DeniedCapabilityCall):
                    continue
                tools.append({"name": spec.qualified_name, "title": spec.title, "description": spec.title,
                              "inputSchema": _tool_schema(spec)})
            await self._send_result(send, request_id, {"tools": tools})
            return
        if method == "ping":
            await self._send_result(send, request_id, {})
            return
        await self._send_error(send, request_id, -32601, "Method not found", status=400)

    @staticmethod
    async def _body(receive: Receive) -> bytes:
        parts: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return b""
            if message["type"] != "http.request":
                continue
            parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                return b"".join(parts)

    @staticmethod
    async def _send_result(send: Send, request_id: Any, result: dict[str, Any]) -> None:
        await JSONResponse(
            {"jsonrpc": "2.0", "id": request_id, "result": result},
            headers={"MCP-Protocol-Version": MCP_PROTOCOL_VERSION},
        )(
            {"type": "http"}, _empty_receive, send,
        )

    @staticmethod
    async def _send_error(send: Send, request_id: Any, code: int, message: str, *, status: int) -> None:
        await JSONResponse(
            {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}},
            status_code=status, headers={"MCP-Protocol-Version": MCP_PROTOCOL_VERSION},
        )({"type": "http"}, _empty_receive, send)


class ProtocolVersionDispatch:
    """Route current stateless requests to the adapter and legacy requests to SDK."""

    def __init__(self, legacy_app: ASGIApp) -> None:
        self.legacy_app = legacy_app
        self.modern_app = StatelessMCP20260728App()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if _header(scope, b"mcp-protocol-version") == MCP_PROTOCOL_VERSION:
            await self.modern_app(scope, receive, send)
            return
        await self.legacy_app(scope, receive, send)


async def _empty_receive() -> Message:
    return {"type": "http.disconnect"}


__all__ = ["HEADER_MISMATCH", "MCP_PROTOCOL_VERSION", "ProtocolVersionDispatch", "StatelessMCP20260728App"]
