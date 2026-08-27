"""Authenticated Streamable HTTP transport for the GraphRAG MCP server.

The stdio entry point in :mod:`mcp_server.server` remains the local developer
transport. This module exposes that same ``FastMCP`` instance at ``/mcp`` for
remote clients. It adds only transport concerns -- bearer authentication,
bounded request bodies, and correlation IDs -- so authorization remains in the
versioned capability registry.

Run with ``python -m mcp_server.remote`` behind TLS termination. Tokens are
accepted only in the ``Authorization`` header; permissive CORS is deliberately
not enabled.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
import string

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from graphrag.core.resource_identifiers import mcp_resource
from graphrag.observability.correlation import correlation_context
from graphrag.observability.agent_telemetry import transport_context
from mcp_server.identity import CallerIdentity
from mcp_server.oauth_metadata import (
    SCOPES_SUPPORTED,
    challenge_header,
    metadata_path,
    protected_resource_metadata,
)
from mcp_server.server import mcp
from mcp_server.transport_20260728 import ProtocolVersionDispatch

DEFAULT_MAX_REQUEST_BYTES = 1_048_576
_CORRELATION_CHARS = frozenset(string.ascii_letters + string.digits + "-_.:")


def _max_request_bytes() -> int:
    raw = os.environ.get("GRAPHRAG_MCP_MAX_REQUEST_BYTES", str(DEFAULT_MAX_REQUEST_BYTES))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("GRAPHRAG_MCP_MAX_REQUEST_BYTES must be an integer") from exc
    if value <= 0:
        raise RuntimeError("GRAPHRAG_MCP_MAX_REQUEST_BYTES must be positive")
    return value


def _allowed_origins() -> frozenset[str]:
    """Exact browser origins allowed to reach the remote MCP transport."""
    raw = os.environ.get("GRAPHRAG_MCP_ALLOWED_ORIGINS", "")
    origins = frozenset(item.strip().rstrip("/") for item in raw.split(",") if item.strip())
    if "*" in origins:
        raise RuntimeError("GRAPHRAG_MCP_ALLOWED_ORIGINS must not contain '*'")
    return origins


def _header(scope: Scope, name: bytes) -> str:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1")
    return ""


def _safe_correlation_id(scope: Scope) -> str:
    value = _header(scope, b"x-correlation-id")
    if 0 < len(value) <= 128 and all(char in _CORRELATION_CHARS for char in value):
        return value
    return ""


class RemoteMCPAuthMiddleware:
    """Per-request auth boundary for FastMCP's long-lived HTTP sessions.

    A small ASGI wrapper is intentionally used instead of BaseHTTPMiddleware:
    request identity must remain bound for the whole streaming response and
    must reset even if the client disconnects or a tool raises.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_request_bytes: int | None = None,
        allowed_origins: frozenset[str] | set[str] | None = None,
    ) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes or _max_request_bytes()
        self.allowed_origins = frozenset(
            origin.rstrip("/") for origin in (
                _allowed_origins() if allowed_origins is None else allowed_origins
            )
        )
        # Resolved once so a mid-flight environment change cannot make the
        # audience this server enforces disagree with the one it advertises.
        self.resource = mcp_resource()
        self.metadata_path = metadata_path(self.resource)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # /health and the RFC 9728 metadata document are the two unauthenticated
        # surfaces. The metadata document MUST be reachable without a token --
        # it is what a client with no usable token reads to find out where to
        # get one.
        if scope["path"] in ("/health", self.metadata_path):
            await self.app(scope, receive, send)
            return

        # MCP Streamable HTTP requires Origin validation to prevent a browser
        # from using DNS rebinding to reach a local or internal gateway. CLI
        # clients usually omit Origin and continue to work unchanged.
        origin = _header(scope, b"origin").rstrip("/")
        if origin and origin not in self.allowed_origins:
            await self._reject(send, 403, "Origin is not allowed")
            return

        raw_length = _header(scope, b"content-length")
        if raw_length:
            try:
                if int(raw_length) > self.max_request_bytes:
                    await self._reject(send, 413, "MCP request body exceeds configured limit")
                    return
            except ValueError:
                await self._reject(send, 400, "invalid Content-Length")
                return

        authorization = _header(scope, b"authorization")
        if not authorization.lower().startswith("bearer "):
            await self._reject(
                send, 401, "Bearer authentication is required",
                challenge=challenge_header(scope=" ".join(SCOPES_SUPPORTED)),
            )
            return
        # Audience validation is a MUST for an MCP resource server: a token
        # minted for the REST API must not buy access to the governed MCP
        # tool surface, and a token with no audience at all is not evidence
        # that any authorization server intended it for this resource.
        identity = await CallerIdentity.from_token_checked(
            authorization[7:].strip(),
            audience=self.resource,
            strict_audience=True,
        )
        if not identity.authenticated:
            await self._reject(
                send, 401, "invalid or incomplete bearer token",
                challenge=challenge_header(
                    error="invalid_token",
                    error_description=(
                        "token must be valid, unexpired, tenant-scoped, and issued "
                        f"for resource {self.resource}"
                    ),
                    scope=" ".join(SCOPES_SUPPORTED),
                ),
            )
            return

        identity_token = CallerIdentity.bind_request(identity)
        body = await self._read_bounded_body(receive)
        if body is None:
            try:
                await self._reject(send, 413, "MCP request body exceeds configured limit")
            finally:
                CallerIdentity.reset_request(identity_token)
            return

        body_delivered = False

        async def replay_receive() -> Message:
            """Give downstream exactly the verified request body once.

            Starlette turns exceptions raised while it reads a body into a 500.
            Buffering up to the small configured bound here means FastMCP only
            ever receives a valid request or does not run at all.
            """
            nonlocal body_delivered
            if not body_delivered:
                body_delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        async def tracked_send(message: Message) -> None:
            await self._correlated_send(send, correlation_id)(message)

        with correlation_context(_safe_correlation_id(scope)) as correlation_id:
            with transport_context("streamable_http"):
                try:
                    await self.app(scope, replay_receive, tracked_send)
                finally:
                    CallerIdentity.reset_request(identity_token)

    async def _read_bounded_body(self, receive: Receive) -> bytes | None:
        """Read a complete request body without allowing unbounded buffering."""
        parts: list[bytes] = []
        bytes_seen = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return b""
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            bytes_seen += len(chunk)
            if bytes_seen > self.max_request_bytes:
                return None
            if chunk:
                parts.append(chunk)
            if not message.get("more_body", False):
                return b"".join(parts)

    @staticmethod
    def _correlated_send(send: Send, correlation_id: str) -> Send:
        async def _send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-correlation-id", correlation_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)
        return _send

    @staticmethod
    async def _reject(
        send: Send, status: int, detail: str, *, challenge: str | None = None,
    ) -> None:
        headers = {"WWW-Authenticate": challenge} if challenge else None
        await JSONResponse(
            {"detail": detail}, status_code=status, headers=headers,
        )({"type": "http"}, _no_receive, send)


async def _no_receive() -> Message:
    """Stand-in receive for a response rendered outside the ASGI request cycle.

    Starlette's Response never reads from `receive`, but the ASGI contract says
    it is awaitable, and a plain `lambda: None` would raise if that ever
    changed. Disconnect is the only honest answer here: the body has already
    been consumed or rejected.
    """
    return {"type": "http.disconnect"}


async def _health(_request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok", "transport": "streamable-http",
        "protocol_versions": ["2025-03-26", "2026-07-28"],
    })


async def _protected_resource_metadata(_request: Request) -> JSONResponse:
    """Serve the RFC 9728 document that points clients at the auth server."""
    return JSONResponse(
        protected_resource_metadata(),
        headers={"Cache-Control": "public, max-age=3600"},
    )


def create_remote_app() -> ASGIApp:
    routes = [
        Route("/health", _health, methods=["GET"]),
        Route(
            metadata_path(),
            _protected_resource_metadata,
            methods=["GET"],
        ),
    ]
    try:
        # Metrics include capability/router/evaluation counters registered in
        # this process. The outer auth middleware intentionally protects this
        # endpoint as well; configure the scraper with a least-privilege JWT.
        from prometheus_client import make_asgi_app
        routes.append(Mount("/metrics", app=make_asgi_app()))
    except ImportError:  # pragma: no cover - Prometheus is a production dep
        pass
    # New MCP clients are stateless under 2026-07-28; older SDK clients keep
    # the session-oriented FastMCP path during the published migration window.
    routes.append(Mount("/", app=ProtocolVersionDispatch(mcp.streamable_http_app())))

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        # A mounted Starlette application does not automatically run its
        # lifespan. FastMCP's Streamable HTTP session manager needs its task
        # group entered before it can accept initialize/tools/list requests.
        async with mcp.session_manager.run():
            yield

    app = Starlette(routes=routes, lifespan=lifespan)
    return RemoteMCPAuthMiddleware(app)


app = create_remote_app()


def main() -> None:
    uvicorn.run(
        "mcp_server.remote:app",
        # Local runs bind to loopback. Container deployments explicitly set
        # 0.0.0.0 after applying their network boundary and authentication.
        host=os.environ.get("GRAPHRAG_MCP_HOST", "127.0.0.1"),
        port=int(os.environ.get("GRAPHRAG_MCP_PORT", "8002")),
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
