"""Trust configuration and remote key material for externally-issued tokens.

Why this exists
----------------
Until now, the token issuer and the resource server verifying it were always
the same process: ``api/auth/jwt.py`` tried every locally-configured key
against a presented token regardless of who signed it, because there was no
concept of "who signed it" at all. That is fine for a single service, and
becomes a real gap the moment a second, independently-operated identity
provider needs to hand this deployment a token it should honour.

The property this module exists to guarantee is **deny-by-default on the
issuer dimension**: an ``iss`` this deployment does not recognize must never
fall through to trying local keys just because a key happens to verify.
Verification is dispatched by ``iss`` first (see ``api/auth/jwt.py``), and
only a configured, explicitly-scoped issuer's own keys are ever tried against
its own tokens.

Design notes
------------
**RS256 only.** A JWKS document has no symmetric-key concept, so HS256 is
excluded from federation structurally, not by a runtime check -- ``_parse_jwks``
only ever produces ``RS256`` entries. Federating a shared HMAC secret would
defeat HS256's own threat model anyway (see ``graphrag/core/signing_keys.py``):
anyone who can verify could also mint.

**Issuer-scoped audience.** Each trusted issuer names the specific
resource(s) its tokens may claim (``TrustedIssuerConfig.audiences`` in
``graphrag/core/config.py``). A trusted issuer is not automatically trusted
for every resource this deployment hosts -- an external IdP compromised or
misconfigured to mint an ``aud`` outside its configured scope is still
rejected, even though that audience is a real local resource.

**Cache shape mirrors ``token_revocation.py``.** A plain dict with lazy
sweep-on-read TTL, no Redis -- a JWKS cache needs no cross-replica
coordination the way revocation does, so the simpler shape is the right one
here. Unlike ``token_revocation.get_revocation_store()``, ``get_jwks_cache()``
needs no lock around its cold-start: constructing ``RemoteJWKSCache`` does no
I/O (fetches happen lazily per issuer in ``ensure_keys``), so there is no
``await`` between the None-check and the assignment for two concurrent
callers to race across.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass

import httpx
import structlog

from graphrag.core.resource_identifiers import (
    InvalidResourceIdentifier,
    canonical_resource_uri,
)
from graphrag.core.signing_keys import RS256, VerificationKey, _rsa_thumbprint

log = structlog.get_logger(__name__)

_DEFAULT_FETCH_TIMEOUT_SECONDS = 10


class IssuerTrustError(RuntimeError):
    """Base for issuer-trust failures."""


class JWKSFetchError(IssuerTrustError):
    """Raised when a trusted issuer's JWKS could not be fetched or parsed."""


@dataclass(frozen=True)
class TrustedIssuer:
    """One issuer this deployment has been configured to trust."""

    issuer: str
    jwks_uri: str
    audiences: frozenset[str]
    allowed_tenants: frozenset[str] = frozenset()
    max_scopes: frozenset[str] = frozenset()


def trusted_issuers() -> dict[str, TrustedIssuer]:
    """Configured trust anchors keyed by canonical issuer URL.

    Re-derived from ``get_settings()`` on every call, matching
    ``signing_keys.accepted_algorithms()``'s precedent -- the list is small
    and rarely read on a hot path, so a cache would buy nothing but staleness
    risk after a config reload.
    """
    from graphrag.core.config import get_settings

    result: dict[str, TrustedIssuer] = {}
    for entry in get_settings().jwt_trusted_issuers:
        try:
            issuer = canonical_resource_uri(entry.issuer)
        except InvalidResourceIdentifier:
            # Settings validation already rejects an unusable issuer URL at
            # load time; this is defensive only, never expected to trigger.
            continue
        result[issuer] = TrustedIssuer(
            issuer=issuer,
            jwks_uri=entry.jwks_uri,
            audiences=frozenset(entry.audiences),
            allowed_tenants=frozenset(entry.allowed_tenants),
            max_scopes=frozenset(entry.max_scopes),
        )
    return result


def find_trusted_issuer(iss: str | None) -> TrustedIssuer | None:
    """Return the configured trust anchor for `iss`, or None if untrusted."""
    if not iss:
        return None
    try:
        canonical = canonical_resource_uri(iss)
    except InvalidResourceIdentifier:
        return None
    return trusted_issuers().get(canonical)


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _jwk_to_pem(jwk: dict) -> str:
    """RSA public JWK (``n``, ``e``) to a PEM public key."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    n = int.from_bytes(_b64url_decode(jwk["n"]), "big")
    e = int.from_bytes(_b64url_decode(jwk["e"]), "big")
    public_key = rsa.RSAPublicNumbers(e=e, n=n).public_key()
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")


def _parse_jwks(document: dict, issuer: str) -> list[VerificationKey]:
    """Extract usable RS256 keys from a fetched JWKS document.

    One unreadable entry is logged and skipped rather than failing the whole
    document -- the same "one bad key must not hide the rest" philosophy as
    ``signing_keys._split_pems``'s callers. A non-RSA entry (``kty`` other
    than ``"RSA"``) is skipped silently: JWKS has no symmetric-key concept, so
    this is where HS256 federation becomes structurally impossible rather
    than a runtime check.
    """
    keys: list[VerificationKey] = []
    entries = document.get("keys", []) if isinstance(document, dict) else []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("kty") != "RSA":
            continue
        try:
            pem = _jwk_to_pem(entry)
            kid = entry.get("kid") or _rsa_thumbprint(pem)
        except Exception as exc:  # noqa: BLE001 - one bad JWK must not hide the rest
            log.warning("issuer_trust.unreadable_jwk", issuer=issuer, error=str(exc))
            continue
        keys.append(VerificationKey(kid=kid, algorithm=RS256, material=pem))
    return keys


@dataclass
class _CacheEntry:
    keys: list[VerificationKey]
    expires_at: float


class RemoteJWKSCache:
    """Per-issuer cache of a trusted external issuer's RS256 keys.

    ``http_client_factory`` is injectable so tests can substitute a fake
    async client returning a canned JWKS document -- no real network, no
    mocking library, mirroring how the codebase has no shared HTTP wrapper
    (see ``api/auth/google.py`` for the bare-``httpx.AsyncClient`` house
    style this follows).
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = 300,
        http_client_factory=None,
    ) -> None:
        self._ttl = max(1, ttl_seconds)
        self._entries: dict[str, _CacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._http_client_factory = http_client_factory or (
            lambda: httpx.AsyncClient(timeout=_DEFAULT_FETCH_TIMEOUT_SECONDS)
        )

    def cached_keys(self, issuer: str) -> list[VerificationKey]:
        """Already-cached keys for `issuer`, or empty if cold. Zero I/O."""
        self._sweep()
        entry = self._entries.get(issuer)
        return list(entry.keys) if entry is not None else []

    async def ensure_keys(self, trusted: TrustedIssuer) -> list[VerificationKey]:
        """Return `trusted`'s keys, fetching and caching them if cold.

        Raises ``JWKSFetchError`` on network/parse failure -- that issuer's
        tokens fail to verify, the rest of the service is unaffected.
        """
        self._sweep()
        entry = self._entries.get(trusted.issuer)
        if entry is not None:
            return list(entry.keys)
        lock = self._locks.setdefault(trusted.issuer, asyncio.Lock())
        async with lock:
            entry = self._entries.get(trusted.issuer)  # a concurrent caller may have warmed it
            if entry is not None:
                return list(entry.keys)
            try:
                async with self._http_client_factory() as client:
                    response = await client.get(trusted.jwks_uri)
                    response.raise_for_status()
                    document = response.json()
            except Exception as exc:  # noqa: BLE001 - network/parse errors all collapse to one clear failure
                raise JWKSFetchError(
                    f"could not fetch JWKS for issuer {trusted.issuer!r} "
                    f"from {trusted.jwks_uri!r}: {exc}"
                ) from exc
            keys = _parse_jwks(document, trusted.issuer)
            self._entries[trusted.issuer] = _CacheEntry(
                keys=keys, expires_at=time.time() + self._ttl,
            )
            return list(keys)

    def _sweep(self) -> None:
        now = time.time()
        for key in [k for k, e in self._entries.items() if e.expires_at <= now]:
            self._entries.pop(key, None)


_cache: RemoteJWKSCache | None = None


def get_jwks_cache() -> RemoteJWKSCache:
    """Process singleton. See the module docstring for why no lock is needed."""
    global _cache
    if _cache is None:
        from graphrag.core.config import get_settings

        _cache = RemoteJWKSCache(ttl_seconds=get_settings().jwt_issuer_jwks_cache_ttl_seconds)
    return _cache


def reset_jwks_cache() -> None:
    """Drop the cached singleton. Used by tests."""
    global _cache
    _cache = None
