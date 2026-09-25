"""Multi-issuer OAuth/JWT federation: trust dispatch, JWKS fetch/cache, and
issuer-scoped audience enforcement.

Four properties are pinned:

1. **Backward compatibility.** A self-issued token -- including one minted
   before `iss` existed at all -- verifies exactly as before.
2. **Deny-by-default on the issuer dimension.** An `iss` this deployment does
   not recognize is rejected outright, even when the token carries a
   perfectly valid *local* signature. A regression here would mean an
   attacker who can get any locally-valid signature accepted regardless of
   what issuer they claim -- the entire point of naming an issuer at all.
3. **Issuer-scoped audience.** A trusted issuer's tokens may only claim the
   audience(s) it was explicitly configured for, never any other resource
   this deployment happens to host.
4. **Blast-radius containment.** A JWKS fetch failure for one issuer does not
   affect verification of self-issued tokens in the same process.

No Docker, no real network: the remote JWKS fetch is exercised through an
injected fake async client (see `RemoteJWKSCache.http_client_factory`).
"""

from __future__ import annotations

import base64
import time

import jwt as pyjwt
import pytest

from api.auth.jwt import create_access_token, decode_access_token, decode_access_token_async
from graphrag.core import issuer_trust
from graphrag.core.config import TrustedIssuerConfig
from graphrag.core.issuer_trust import RemoteJWKSCache
from graphrag.core.resource_identifiers import api_resource, mcp_resource

TRUSTED_ISSUER = "https://idp.partner.example"
TRUSTED_JWKS_URI = "https://idp.partner.example/.well-known/jwks.json"


@pytest.fixture(autouse=True)
def _reset_jwks_cache():
    issuer_trust.reset_jwks_cache()
    yield
    issuer_trust.reset_jwks_cache()


def _external_keypair():
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def _b64(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _jwks_document_for(public_key, *, kid: str = "ext-key-1") -> dict:
    numbers = public_key.public_numbers()
    return {"keys": [{"kty": "RSA", "kid": kid, "n": _b64(numbers.n), "e": _b64(numbers.e)}]}


def _external_token(private_key, *, iss: str, aud: str, kid: str | None = "ext-key-1", **overrides) -> str:
    payload = {
        "sub": "ext-agent", "tenant": "aerospace",
        "exp": int(time.time()) + 600, "iss": iss, "aud": aud,
    }
    payload.update(overrides)
    headers = {"kid": kid} if kid else {}
    return pyjwt.encode(payload, private_key, algorithm="RS256", headers=headers)


def _patch_trusted_issuers(
    monkeypatch,
    *,
    audiences: list[str],
    allowed_tenants: list[str] = ("aerospace",),
    max_scopes: list[str] = ("read",),
) -> None:
    """Make this deployment trust TRUSTED_ISSUER, scoped to `audiences`.

    Patches `graphrag.core.config.get_settings` -- issuer_trust.trusted_issuers()
    imports get_settings *inside* the function (the codebase-wide convention
    for avoiding config import cycles), so the patch must target the config
    module rather than issuer_trust itself. Delegates every other attribute
    to the real settings via __getattr__ -- create_access_token's own
    get_settings() calls (algorithm selection, signing secret) must keep
    working unchanged in these tests, not just the two fields under test.
    """
    import graphrag.core.config as config_module

    real_settings = config_module.get_settings()

    class _FakeSettings:
        jwt_trusted_issuers = [
            TrustedIssuerConfig(
                issuer=TRUSTED_ISSUER, jwks_uri=TRUSTED_JWKS_URI, audiences=audiences,
                allowed_tenants=list(allowed_tenants), max_scopes=list(max_scopes),
            ),
        ]
        jwt_issuer_jwks_cache_ttl_seconds = 300

        def __getattr__(self, name):
            return getattr(real_settings, name)

    monkeypatch.setattr(config_module, "get_settings", lambda: _FakeSettings())


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient, injected via http_client_factory."""

    def __init__(self, calls: list[str], document: dict | None = None, *, fail: bool = False):
        self._calls = calls
        self._document = document
        self._fail = fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url: str):
        self._calls.append(url)
        if self._fail:
            raise RuntimeError("network unreachable")
        return _FakeResponse(self._document)


def _seed_cache(monkeypatch, *, document: dict | None = None, fail: bool = False, calls: list[str] | None = None):
    calls = calls if calls is not None else []
    cache = RemoteJWKSCache(http_client_factory=lambda: _FakeAsyncClient(calls, document, fail=fail))
    monkeypatch.setattr(issuer_trust, "_cache", cache)
    return calls


class TestSelfIssuedUnchanged:
    def test_self_issued_token_still_verifies(self):
        token = create_access_token({"sub": "a", "tenant": "t"})
        claims = decode_access_token(token, audience=api_resource())
        assert claims["sub"] == "a"
        assert claims["iss"] == api_resource()

    def test_legacy_token_with_no_iss_claim_still_verifies(self):
        # Minted before `iss` existed at all -- the compat ramp `_is_self_issued`
        # documents, same shape as the existing missing-`aud` tolerance.
        from graphrag.core import signing_keys as sk

        algorithm, key_material, kid = sk.signing_key()
        payload = {
            "sub": "legacy-user", "tenant": "t",
            "exp": int(time.time()) + 600, "aud": api_resource(), "jti": "legacy-1",
        }
        token = pyjwt.encode(payload, key_material, algorithm=algorithm, headers={"kid": kid})
        claims = decode_access_token(token, audience=api_resource())
        assert claims["sub"] == "legacy-user"


class TestUnknownIssuerRejected:
    def test_token_with_unconfigured_issuer_is_rejected_despite_a_valid_local_signature(self):
        # Signed with THIS deployment's own real key, but claiming an issuer
        # this deployment does not recognize. This is the critical negative
        # test: proves there is no fallback to local keys just because the
        # signature happens to check out.
        token = create_access_token({"sub": "attacker", "tenant": "t"}, issuer="https://evil.example")
        with pytest.raises(ValueError):
            decode_access_token(token)


class TestExternalIssuerTrust:
    async def test_trusted_external_issuer_token_verifies(self, monkeypatch):
        _patch_trusted_issuers(monkeypatch, audiences=[mcp_resource()])
        private_key, public_key = _external_keypair()
        calls = _seed_cache(monkeypatch, document=_jwks_document_for(public_key))

        token = _external_token(private_key, iss=TRUSTED_ISSUER, aud=mcp_resource())
        claims = await decode_access_token_async(token, audience=mcp_resource())

        assert claims["sub"] == "ext-agent"
        assert len(calls) == 1

    async def test_issuer_scoped_audience_rejects_a_resource_outside_its_configured_scope(self, monkeypatch):
        # Trusted only for mcp_resource() -- a token claiming api_resource()
        # (a real local resource, just not one this issuer is scoped for)
        # must still be rejected.
        _patch_trusted_issuers(monkeypatch, audiences=[mcp_resource()])
        private_key, public_key = _external_keypair()
        _seed_cache(monkeypatch, document=_jwks_document_for(public_key))

        token = _external_token(private_key, iss=TRUSTED_ISSUER, aud=api_resource())
        with pytest.raises(ValueError):
            await decode_access_token_async(token)

    async def test_jwks_fetch_failure_rejects_only_that_token(self, monkeypatch):
        _patch_trusted_issuers(monkeypatch, audiences=[mcp_resource()])
        private_key, _public_key = _external_keypair()
        _seed_cache(monkeypatch, fail=True)

        token = _external_token(private_key, iss=TRUSTED_ISSUER, aud=mcp_resource())
        with pytest.raises(ValueError):
            await decode_access_token_async(token)

        # Blast-radius containment: a self-issued token in the same process
        # is unaffected by the other issuer's fetch failure.
        local_token = create_access_token({"sub": "a", "tenant": "t"})
        claims = await decode_access_token_async(local_token, audience=api_resource())
        assert claims["sub"] == "a"

    async def test_jwks_is_cached_within_ttl(self, monkeypatch):
        _patch_trusted_issuers(monkeypatch, audiences=[mcp_resource()])
        private_key, public_key = _external_keypair()
        calls = _seed_cache(monkeypatch, document=_jwks_document_for(public_key))

        first = _external_token(private_key, iss=TRUSTED_ISSUER, aud=mcp_resource(), jti="t1")
        second = _external_token(private_key, iss=TRUSTED_ISSUER, aud=mcp_resource(), jti="t2")
        await decode_access_token_async(first)
        await decode_access_token_async(second)

        assert len(calls) == 1


class TestIssuerScopedTenantAndScopes:
    """A trusted IdP may only mint tokens for its allowed tenants and scopes."""

    async def _decode(self, monkeypatch, **claims):
        _patch_trusted_issuers(monkeypatch, audiences=[mcp_resource()])
        private_key, public_key = _external_keypair()
        _seed_cache(monkeypatch, document=_jwks_document_for(public_key))
        token = _external_token(private_key, iss=TRUSTED_ISSUER, aud=mcp_resource(), **claims)
        return await decode_access_token_async(token, audience=mcp_resource())

    async def test_allowed_tenant_and_scope_verify(self, monkeypatch):
        claims = await self._decode(monkeypatch, scope="read tenant:aerospace")
        assert claims["tenant"] == "aerospace"

    async def test_tenant_claim_outside_allow_list_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError):
            await self._decode(monkeypatch, tenant="banking")

    async def test_missing_tenant_claim_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError):
            await self._decode(monkeypatch, tenant=None)

    async def test_tenant_scope_outside_allow_list_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError):
            await self._decode(monkeypatch, scope="read tenant:banking")

    async def test_scope_above_max_scopes_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError):
            await self._decode(monkeypatch, scope="read admin")


class TestSyncColdCache:
    def test_sync_decode_rejects_an_external_token_on_a_cold_cache(self, monkeypatch):
        # decode_access_token (sync) never fetches -- it only tries keys
        # already warm in the cache. A configured-but-cold issuer's token is
        # rejected here, documenting the accepted sync-path limitation (see
        # CallerIdentity.from_token's docstring in mcp_server/identity.py).
        _patch_trusted_issuers(monkeypatch, audiences=[mcp_resource()])
        private_key, _public_key = _external_keypair()

        token = _external_token(private_key, iss=TRUSTED_ISSUER, aud=mcp_resource())
        with pytest.raises(ValueError):
            decode_access_token(token)
