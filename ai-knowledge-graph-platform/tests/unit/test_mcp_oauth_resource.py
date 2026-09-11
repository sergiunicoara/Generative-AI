"""MCP resource-server conformance: audience binding and RFC 9728 discovery.

The MCP 2026-07-28 authorization specification requires an MCP server acting
as an OAuth resource server to (a) implement OAuth 2.0 Protected Resource
Metadata (RFC 9728), and (b) validate that access tokens were issued
specifically for it as the intended audience (RFC 8707), accepting no other
tokens. These tests pin both, because the failure mode of getting either wrong
is silent: a REST API token simply keeps working against the governed MCP tool
surface, and nothing logs an anomaly.
"""

from __future__ import annotations

import jwt as pyjwt
import pytest
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from api.auth.jwt import create_access_token, decode_access_token
from graphrag.core.config import get_settings
from graphrag.core.resource_identifiers import (
    InvalidResourceIdentifier,
    api_resource,
    canonical_resource_uri,
    mcp_resource,
    resolve_requested_resource,
)
from mcp_server.identity import CallerIdentity
from mcp_server.oauth_metadata import (
    challenge_header,
    metadata_path,
    protected_resource_metadata,
    resource_metadata_url,
)
from mcp_server.remote import RemoteMCPAuthMiddleware, create_remote_app


async def _identity_echo(_request: Request) -> JSONResponse:
    identity = CallerIdentity.current()
    return JSONResponse({"subject": identity.subject, "tenant": identity.tenant})


def _protected_app() -> RemoteMCPAuthMiddleware:
    return RemoteMCPAuthMiddleware(
        Starlette(routes=[Route("/mcp", _identity_echo, methods=["POST"])]),
        max_request_bytes=4096,
        allowed_origins=set(),
    )


def _token(audience: str | None, **overrides) -> str:
    claims = {"sub": "agent-1", "tenant": "aerospace", "scope": "read", "type": "m2m"}
    claims.update(overrides)
    return create_access_token(claims, audience=audience)


def _raw_token(**claims) -> str:
    """Mint a token bypassing create_access_token, to model legacy shapes."""
    return pyjwt.encode(claims, get_settings().jwt_secret_key, algorithm="HS256")


class TestCanonicalResourceIdentifiers:
    def test_scheme_and_host_are_lowercased_and_trailing_slash_dropped(self):
        assert canonical_resource_uri("HTTPS://MCP.Example.COM/mcp/") == "https://mcp.example.com/mcp"

    @pytest.mark.parametrize("value", [
        "",
        "mcp.example.com",                  # no scheme
        "https://mcp.example.com#frag",     # RFC 8707 forbids a fragment
        "https://mcp.example.com?a=b",      # a query is not part of an identifier
        "ftp://mcp.example.com",
    ])
    def test_non_canonical_identifiers_are_rejected(self, value):
        with pytest.raises(InvalidResourceIdentifier):
            canonical_resource_uri(value)

    def test_unknown_resource_is_not_mintable(self):
        with pytest.raises(InvalidResourceIdentifier):
            resolve_requested_resource("https://attacker.example/mcp")

    def test_omitted_resource_defaults_to_the_api(self):
        assert resolve_requested_resource(None) == api_resource()


class TestAudienceBinding:
    def test_api_and_mcp_are_distinct_resources(self):
        assert api_resource() != mcp_resource()

    def test_default_token_is_bound_to_the_api(self):
        claims = decode_access_token(_token(None), audience=api_resource(), strict=True)
        assert claims["sub"] == "agent-1"

    def test_api_token_is_rejected_by_the_mcp_resource(self):
        with pytest.raises(ValueError):
            decode_access_token(_token(None), audience=mcp_resource(), strict=True)

    def test_token_without_audience_is_rejected_under_strict_validation(self):
        legacy = _raw_token(sub="agent-1", tenant="aerospace", exp=4102444800)
        with pytest.raises(ValueError):
            decode_access_token(legacy, audience=mcp_resource(), strict=True)
        # ...but the non-strict REST path still accepts it during the ramp.
        assert decode_access_token(legacy)["sub"] == "agent-1"

    def test_token_without_expiry_is_always_rejected(self):
        forever = _raw_token(sub="agent-1", tenant="aerospace", aud=mcp_resource())
        with pytest.raises(ValueError):
            decode_access_token(forever, audience=mcp_resource())

    def test_strict_validation_requires_naming_the_audience(self):
        with pytest.raises(ValueError, match="requires an audience"):
            decode_access_token(_token(None), strict=True)

    def test_mcp_token_is_rejected_by_the_api_resource(self):
        # The symmetric half: closing only API-token-reaches-MCP would leave
        # the same confused-deputy shape running the other way.
        with pytest.raises(ValueError):
            decode_access_token(_token(mcp_resource()), audience=api_resource())

    def test_a_wrong_audience_is_rejected_even_without_strict(self):
        # PyJWT cannot express "reject wrong, tolerate missing" -- supplying an
        # audience makes a missing claim a hard error too -- so this pins that
        # the hand-rolled check keeps the two cases distinct.
        legacy = _raw_token(sub="agent-1", tenant="aerospace", exp=4102444800)
        assert decode_access_token(legacy, audience=api_resource())["sub"] == "agent-1"
        with pytest.raises(ValueError):
            decode_access_token(_token(mcp_resource()), audience=api_resource())

    def test_multi_valued_audience_matches_any_member(self):
        # RFC 7519 Section 4.1.3 permits an array; a deployment fronting both
        # resources with one token must still validate at either.
        both = _raw_token(
            sub="agent-1", tenant="aerospace", exp=4102444800,
            aud=[api_resource(), mcp_resource()],
        )
        assert decode_access_token(both, audience=api_resource(), strict=True)["sub"] == "agent-1"
        assert decode_access_token(both, audience=mcp_resource(), strict=True)["sub"] == "agent-1"

    def test_malformed_audience_claim_grants_nothing(self):
        malformed = _raw_token(
            sub="agent-1", tenant="aerospace", exp=4102444800, aud={"not": "a list"},
        )
        with pytest.raises(ValueError):
            decode_access_token(malformed, audience=api_resource())


class TestRemoteTransportAudienceEnforcement:
    def test_mcp_audience_token_is_accepted(self):
        client = TestClient(_protected_app())
        response = client.post(
            "/mcp",
            content=b"{}",
            headers={"Authorization": "Bearer " + _token(mcp_resource())},
        )
        assert response.status_code == 200
        assert response.json() == {"subject": "agent-1", "tenant": "aerospace"}

    def test_api_audience_token_cannot_reach_the_mcp_tool_surface(self):
        client = TestClient(_protected_app())
        response = client.post(
            "/mcp",
            content=b"{}",
            headers={"Authorization": "Bearer " + _token(api_resource())},
        )
        assert response.status_code == 401
        challenge = response.headers["WWW-Authenticate"]
        assert 'error="invalid_token"' in challenge
        assert 'resource_metadata="' + resource_metadata_url() + '"' in challenge

    def test_missing_token_challenge_points_at_the_metadata_document(self):
        client = TestClient(_protected_app())
        response = client.post("/mcp", content=b"{}")
        assert response.status_code == 401
        challenge = response.headers["WWW-Authenticate"]
        assert challenge.startswith("Bearer ")
        assert 'resource_metadata="' + resource_metadata_url() + '"' in challenge
        assert 'scope="read"' in challenge


class TestTokenEndpointResourceParameter:
    """RFC 8707 `resource` round-trip through the real /auth/token route.

    The unit tests above prove the primitives; this proves they are actually
    wired, which is the part that silently regresses.
    """

    @staticmethod
    def _client():
        from fastapi import FastAPI

        from api.routes import auth as auth_routes

        app = FastAPI()
        # No app.state.limiter: the limiter is a FastAPI dependency now, not a
        # slowapi extension needing app-level registration (see api/limiter.py).
        app.include_router(auth_routes.router, prefix="/auth")
        return TestClient(app)

    @staticmethod
    def _registered_client(monkeypatch) -> tuple[str, str]:
        import hashlib

        from api.routes import auth as auth_routes

        secret = "test-client-secret"
        record = {
            "client_name": "evidence-agent",
            "scopes": ["read", "tenant:aerospace"],
            "secret_hash": hashlib.sha256(secret.encode()).hexdigest(),
            "owner": "tests",
            "tenant": "aerospace",
        }
        monkeypatch.setattr(
            auth_routes, "_client_get",
            lambda client_id: record if client_id == "graphrag_test" else None,
        )
        return "graphrag_test", secret

    def _request(self, monkeypatch, **extra):
        client_id, secret = self._registered_client(monkeypatch)
        body = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": secret,
            "scope": "read",
        }
        body.update(extra)
        return self._client().post("/auth/token", json=body)

    def test_omitting_resource_still_mints_an_api_token(self, monkeypatch):
        response = self._request(monkeypatch)
        assert response.status_code == 200
        assert response.json()["resource"] == api_resource()
        token = response.json()["access_token"]
        assert decode_access_token(token, audience=api_resource(), strict=True)

    def test_requesting_the_mcp_resource_mints_an_mcp_token(self, monkeypatch):
        response = self._request(monkeypatch, resource=mcp_resource())
        assert response.status_code == 200
        assert response.json()["resource"] == mcp_resource()
        token = response.json()["access_token"]
        assert decode_access_token(token, audience=mcp_resource(), strict=True)
        with pytest.raises(ValueError):
            decode_access_token(token, audience=api_resource(), strict=True)

    def test_uppercase_scheme_host_and_trailing_slash_are_normalised(self, monkeypatch):
        # The MCP specification asks servers to accept uppercase scheme and
        # host "for robustness and interoperability"; RFC 3986 makes only
        # those two components case-insensitive.
        scheme, rest = mcp_resource().split("://", 1)
        host, _, path = rest.partition("/")
        spelled = f"{scheme.upper()}://{host.upper()}/{path}/"
        response = self._request(monkeypatch, resource=spelled)
        assert response.status_code == 200
        assert response.json()["resource"] == mcp_resource()

    def test_path_case_is_significant(self, monkeypatch):
        # Paths are case-sensitive per RFC 3986, so /MCP is a different
        # resource from /mcp and must not be silently accepted as a match.
        response = self._request(monkeypatch, resource="http://localhost:8002/MCP")
        assert response.status_code == 400

    def test_an_unhosted_resource_is_invalid_target(self, monkeypatch):
        response = self._request(monkeypatch, resource="https://attacker.example/mcp")
        assert response.status_code == 400
        assert "invalid_target" in response.json()["detail"]

    def test_invalid_target_is_decided_before_the_credential_check(self, monkeypatch):
        # Otherwise the response code doubles as a client_id oracle.
        self._registered_client(monkeypatch)
        response = self._client().post("/auth/token", json={
            "grant_type": "client_credentials",
            "client_id": "graphrag_does_not_exist",
            "client_secret": "wrong",
            "scope": "read",
            "resource": "https://attacker.example/mcp",
        })
        assert response.status_code == 400


class TestProtectedResourceMetadata:
    def test_document_is_served_unauthenticated_at_the_rfc9728_path(self):
        client = TestClient(create_remote_app())
        response = client.get(metadata_path())
        assert response.status_code == 200
        body = response.json()
        assert body["resource"] == mcp_resource()
        assert body["authorization_servers"] == [api_resource()]
        assert body["bearer_methods_supported"] == ["header"]

    def test_well_known_prefix_precedes_the_resource_path(self):
        # RFC 9728 Section 3.1 inserts the well-known string between host and
        # path; the inverted form is a document no client ever fetches.
        assert metadata_path("https://h.example/server/mcp") == (
            "/.well-known/oauth-protected-resource/server/mcp"
        )
        assert metadata_path("https://h.example") == "/.well-known/oauth-protected-resource"

    def test_offline_access_is_not_advertised(self):
        # The specification tells resource servers not to request refresh-token
        # scope: refresh tokens are a client concern, not a resource requirement.
        assert "offline_access" not in protected_resource_metadata()["scopes_supported"]

    def test_challenge_params_are_quoted_and_escaped(self):
        header = challenge_header(error='bad"quote', scope="read write")
        assert 'error="bad\\"quote"' in header
        assert 'scope="read write"' in header
