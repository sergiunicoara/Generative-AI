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


class TestSessionsAreBoundToTheirCreator:
    """Regression for audit-2026-10-01.md H1: the Streamable HTTP session
    manager only refuses a different caller's use of a session when the
    transport sets scope["user"]. This middleware never did, so every
    session's owner was None and any valid token could send tool calls to any
    Mcp-Session-Id -- which then ran as the session's CREATOR, since tool
    handlers run in the session's server task. Exercised against the real SDK
    session manager, not a stub."""

    _INITIALIZE = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26", "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0"},
        },
    }
    _HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}

    @staticmethod
    def _claims(token: str) -> dict:
        return {
            "alice-token": {"sub": "alice", "tenant": "acme", "scope": "read", "type": "m2m",
                            "iss": "https://issuer-a.example"},
            "bob-token": {"sub": "bob", "tenant": "globex", "scope": "read", "type": "m2m",
                          "iss": "https://issuer-a.example"},
            # Same subject and tenant as alice, but wider scopes.
            "alice-admin-token": {"sub": "alice", "tenant": "acme", "scope": "read admin",
                                  "type": "m2m", "iss": "https://issuer-a.example"},
            # Same subject, tenant and scopes as alice, different issuer.
            "alice-other-issuer-token": {"sub": "alice", "tenant": "acme", "scope": "read",
                                         "type": "m2m", "iss": "https://issuer-b.example"},
        }[token]

    def _call(self, client, token: str, session_id: str | None, body: dict):
        headers = {**self._HEADERS, "Authorization": f"Bearer {token}"}
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        return client.post("/mcp", json=body, headers=headers)

    def test_only_the_creating_identity_can_use_a_session(self):
        async def fake_decode(token, **_kwargs):
            return self._claims(token)

        with patch("mcp_server.identity.decode_access_token_async", side_effect=fake_decode), \
             patch("graphrag.core.token_revocation.get_revocation_store") as store:
            store.return_value.is_revoked = _async_false
            with TestClient(create_remote_app(), base_url="http://localhost:8001") as client:
                created = self._call(client, "alice-token", None, self._INITIALIZE)
                assert created.status_code == 200, created.text
                session_id = created.headers["mcp-session-id"]

                follow_up = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
                # Anyone else -- a different tenant/subject, the same subject
                # with wider scopes, or the same identity from another issuer --
                # is told the session does not exist.
                for intruder in ("bob-token", "alice-admin-token", "alice-other-issuer-token"):
                    resp = self._call(client, intruder, session_id, follow_up)
                    assert resp.status_code == 404, (intruder, resp.status_code, resp.text)

                # The creator is not locked out.
                owner = self._call(client, "alice-token", session_id, follow_up)
                assert owner.status_code != 404, owner.text


async def _async_false(_claims) -> bool:
    return False
