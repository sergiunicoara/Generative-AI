"""Transport-boundary tests for authenticated Streamable HTTP MCP."""

from __future__ import annotations

from unittest.mock import patch

from starlette.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from mcp_server.identity import CallerIdentity
from mcp_server.remote import RemoteMCPAuthMiddleware, create_remote_app


async def _identity_echo(_request: Request) -> JSONResponse:
    identity = CallerIdentity.current()
    return JSONResponse({"subject": identity.subject, "tenant": identity.tenant})


async def _body_reading_echo(request: Request) -> JSONResponse:
    # Forces the ASGI receive wrapper to observe every body chunk.
    await request.body()
    return JSONResponse({"ok": True})


def _protected_app(
    *, max_request_bytes: int = 1024, allowed_origins: set[str] | None = None,
) -> RemoteMCPAuthMiddleware:
    return RemoteMCPAuthMiddleware(
        Starlette(routes=[Route("/mcp", _identity_echo, methods=["POST"])]),
        max_request_bytes=max_request_bytes,
        allowed_origins=allowed_origins,
    )


def _body_reading_app(*, max_request_bytes: int = 1024) -> RemoteMCPAuthMiddleware:
    return RemoteMCPAuthMiddleware(
        Starlette(routes=[Route("/mcp", _body_reading_echo, methods=["POST"])]),
        max_request_bytes=max_request_bytes,
    )


class TestRemoteMCPAuth:
    def test_health_is_public_but_mcp_requires_a_bearer_token(self):
        client = TestClient(create_remote_app())
        assert client.get("/health").status_code == 200
        response = client.post("/mcp", content=b"{}")
        assert response.status_code == 401
        assert "Bearer" in response.json()["detail"]
        assert client.get("/metrics").status_code == 401

    def test_verified_token_is_bound_only_for_the_request(self):
        client = TestClient(_protected_app())
        with patch("mcp_server.identity.decode_access_token_async", return_value={
            "sub": "agent-1", "tenant": "automotive", "scope": "read", "type": "m2m",
        }):
            response = client.post(
                "/mcp", content=b"{}",
                headers={"Authorization": "Bearer valid", "X-Correlation-ID": "remote-1"},
            )
        assert response.status_code == 200
        assert response.json() == {"subject": "agent-1", "tenant": "automotive"}
        assert response.headers["x-correlation-id"] == "remote-1"
        assert CallerIdentity.current() == CallerIdentity.anonymous()

    def test_invalid_token_and_oversized_body_are_rejected_before_dispatch(self):
        client = TestClient(_protected_app(max_request_bytes=3))
        with patch("mcp_server.identity.decode_access_token_async", side_effect=ValueError("bad")):
            invalid = client.post("/mcp", content=b"{}", headers={"Authorization": "Bearer invalid"})
        assert invalid.status_code == 401
        with patch("mcp_server.identity.decode_access_token_async", return_value={
            "sub": "agent-1", "tenant": "aerospace", "scope": "read",
        }):
            oversized = client.post("/mcp", content=b"too-long", headers={"Authorization": "Bearer valid"})
        assert oversized.status_code == 413

    async def test_chunked_body_is_limited_even_without_content_length(self):
        async def chunks():
            yield b"ab"
            yield b"cd"

        transport = ASGITransport(app=_body_reading_app(max_request_bytes=3))
        with patch("mcp_server.identity.decode_access_token_async", return_value={
            "sub": "agent-1", "tenant": "aerospace", "scope": "read",
        }):
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/mcp", content=chunks(), headers={"Authorization": "Bearer valid"},
                )
        assert response.status_code == 413

    def test_browser_origin_must_be_explicitly_allowed(self):
        client = TestClient(_protected_app(allowed_origins={"https://agent.example"}))
        with patch("mcp_server.identity.decode_access_token_async", return_value={
            "sub": "agent-1", "tenant": "aerospace", "scope": "read",
        }):
            denied = client.post(
                "/mcp",
                content=b"{}",
                headers={"Authorization": "Bearer valid", "Origin": "https://evil.example"},
            )
            allowed = client.post(
                "/mcp",
                content=b"{}",
                headers={"Authorization": "Bearer valid", "Origin": "https://agent.example"},
            )

        assert denied.status_code == 403
        assert allowed.status_code == 200
