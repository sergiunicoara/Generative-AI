"""JWT creation and validation.

Audience binding
----------------
Every token names the resource server it may be presented to, in the ``aud``
claim (RFC 8707 "Resource Indicators"). Both resource servers verify it, so a
token minted for the REST API cannot reach the governed MCP tool surface and a
token minted for MCP cannot reach the REST API. See
``graphrag/core/resource_identifiers.py`` for why, and for the canonical URI
rules.

Tokens issued before audience binding existed carry no ``aud`` claim at all.
The audience check below distinguishes that from a *wrong* audience: a missing
claim is tolerated while ``strict`` is False, which is the compatibility ramp
for the one-hour lifetime of already-issued tokens, but a claim naming some
other resource is always rejected. PyJWT cannot express that distinction --
supplying an ``audience=`` makes a missing claim a hard error -- so the check
is done here rather than delegated to ``jwt.decode``.

Signing and rotation
--------------------
Key material, algorithm selection, and key ids live in
``graphrag/core/signing_keys.py``. Tokens carry a ``kid`` header so several
keys can be trusted simultaneously, which is what makes a key rotation
invisible to callers. The accepted algorithm list comes from configuration and
never from the token header -- see that module for why that ordering is the
whole defence against ``alg`` confusion.

Revocation
----------
Every token carries a ``jti``. Verification is *offline* here: this module
proves the signature, expiry, and audience, and returns the claims. Whether the
token has since been revoked is an I/O question, answered by
``graphrag/core/token_revocation.py`` at the request boundary
(``api/auth/dependencies.py`` and the auth middleware). Keeping the two apart
means a purely offline check stays synchronous and testable, and the network
call happens exactly once per request rather than once per decode.

Multi-issuer federation
------------------------
Every token now carries an ``iss`` claim (this deployment's own resource
identifier, by default). Verification dispatches on it: a self-issued token
(``iss`` missing -- a legacy token minted before this claim existed -- or
matching this deployment) is verified exactly as before, against
``signing_keys.verification_keys()``. An externally-issued token is verified
only against *that* issuer's own keys, fetched and cached via
``graphrag/core/issuer_trust.py``, and **only if that issuer is explicitly
configured as trusted** (``jwt_trusted_issuers`` in ``graphrag/core/config.py``)
-- an unrecognized ``iss`` is rejected outright, never falling through to try
local keys just because a key happens to verify. That fall-through would
defeat the entire point of naming an issuer at all.

A trusted external issuer's tokens are further restricted to the specific
audience(s) it was configured for (``_assert_issuer_scoped_claims``), even
when the caller only asked for a generic decode -- a compromised or
misconfigured external issuer must not be able to mint a token for a resource
outside its configured scope, even a resource this deployment genuinely hosts.

``decode_access_token`` never performs I/O -- it only tries keys already warm
in ``issuer_trust``'s cache, so a cold external issuer is rejected. Use
``decode_access_token_async`` from an async call site to warm that cache
first; see its docstring for which call sites need it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from jwt.exceptions import InvalidTokenError

from graphrag.core.issuer_trust import JWKSFetchError, find_trusted_issuer, get_jwks_cache
from graphrag.core.resource_identifiers import (
    InvalidResourceIdentifier,
    api_resource,
    canonical_resource_uri,
)
from graphrag.core.signing_keys import (
    accepted_algorithms,
    signing_key,
    verification_keys,
)

ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Retained for callers that imported it before algorithm selection became
# configurable. It names the historical default, not the effective algorithm --
# read `graphrag.core.signing_keys.configured_algorithm()` for that.
ALGORITHM = "HS256"


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
    *,
    audience: str | None = None,
    issuer: str | None = None,
) -> str:
    """Mint a signed access token bound to a single resource server.

    ``audience`` defaults to this deployment's REST API resource so existing
    call sites keep issuing usable API tokens without change. A caller that
    wants an MCP token must ask for it explicitly, which is what makes the two
    surfaces separately compromisable rather than jointly.

    ``issuer`` defaults to this deployment's own resource identifier -- this
    process only ever mints self-issued tokens; ``iss`` naming some other
    issuer only ever appears on a token this process *verifies*, never one it
    signs.
    """
    algorithm, key_material, kid = signing_key()
    payload = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload.update({
        "exp": expire,
        "iat": now,
        "aud": audience or api_resource(),
        "iss": issuer or api_resource(),
        # A per-token id is what makes single-token revocation possible at all.
        # Generated here rather than by callers so no issuance path can forget.
        "jti": payload.get("jti") or uuid.uuid4().hex,
    })
    return jwt.encode(payload, key_material, algorithm=algorithm, headers={"kid": kid})


def _token_audiences(claims: dict) -> set[str] | None:
    """Return the token's declared audiences, or None when it declares none.

    RFC 7519 Section 4.1.3 allows ``aud`` to be either a single string or an
    array of strings; anything else is malformed and is reported as declaring
    no usable audience rather than being coerced.
    """
    raw = claims.get("aud")
    if raw is None:
        return None
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, (list, tuple)):
        return {item for item in raw if isinstance(item, str)}
    return set()


def _assert_audience(claims: dict, audience: str, *, strict: bool) -> None:
    declared = _token_audiences(claims)
    if declared is None:
        if strict:
            raise ValueError("Invalid token")
        return  # legacy token, tolerated during the rollout window
    if audience not in declared:
        raise ValueError("Invalid token")


def _order_by_kid_hint(token: str, keys: list) -> list:
    """Reorder `keys` so a ``kid`` match (if any) is tried first.

    The ``kid`` header selects a key when it matches one we hold. It is only a
    *hint*: an unknown or absent ``kid`` falls through to trying every
    candidate key, so a token minted before ``kid`` headers existed still
    verifies, and a forged ``kid`` gains nothing because the signature still
    has to check out against material we chose.
    """
    try:
        kid = jwt.get_unverified_header(token).get("kid")
    except InvalidTokenError:
        return keys
    if not kid:
        return keys
    preferred = [key for key in keys if key.kid == kid]
    return preferred + [key for key in keys if key.kid != kid]


def _local_candidate_keys(token: str):
    """This deployment's own verification keys, most likely first."""
    keys = verification_keys()
    allowed = set(accepted_algorithms())
    usable = [key for key in keys if key.algorithm in allowed]
    return _order_by_kid_hint(token, usable)


def _unverified_claim(token: str, name: str) -> str | None:
    """Peek at claim `name` without verifying the signature.

    Same "peek, never trust until the signature is proven" pattern as the
    ``kid`` header peek in ``_order_by_kid_hint`` -- this only decides *which*
    keys to try; the claim is re-read from ``jwt.decode``'s verified output
    everywhere it is actually relied upon.
    """
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
    except InvalidTokenError:
        return None
    value = payload.get(name)
    return value if isinstance(value, str) else None


def _is_self_issued(iss: str | None) -> bool:
    """True if `iss` names this deployment, or is absent (legacy compat ramp).

    Same shape as ``_assert_audience``'s missing-``aud`` tolerance: a token
    minted before ``iss`` existed is treated as self-issued rather than
    rejected, for the lifetime of already-issued tokens.
    """
    if not iss:
        return True
    try:
        return canonical_resource_uri(iss) == api_resource()
    except InvalidResourceIdentifier:
        return False


def _assert_issuer_scoped_claims(trusted, claims: dict) -> None:
    """Restrict an externally-issued token to that issuer's configured scope.

    Enforced unconditionally, even when the caller only asked for a generic
    decode with no ``audience=`` -- a trusted external issuer's tokens may
    only ever claim the audience(s) it was explicitly configured for in
    ``jwt_trusted_issuers``, never any other resource this deployment
    happens to host.
    """
    declared = _token_audiences(claims)
    if not declared or not declared.issubset(trusted.audiences):
        raise ValueError("Invalid token")
    # Same closed-allow-list rule for tenant and scope: otherwise a trusted
    # IdP could mint a token for any tenant, with admin, and it would be
    # honoured as-is.
    token_tenant = claims.get("tenant")
    if token_tenant not in trusted.allowed_tenants:
        raise ValueError("Invalid token")
    for scope in str(claims.get("scope") or "").split():
        if scope.startswith("tenant:"):
            # A self-issued M2M client can legitimately hold more than one
            # tenant:<name> scope -- register_client() only grants them by
            # intersecting against an already-authorized local admin's own
            # scopes, a real authorization step. An external issuer has no
            # such step: allowed_tenants only bounds which tenants it may
            # EVER touch across all its tokens, not what one token may
            # combine. Without pinning every tenant:<name> scope to this
            # token's own tenant claim, an issuer trusted for several
            # tenants could mint a single token claiming tenant "acme" but
            # scoped tenant:globex, which ToolPolicy's cross-tenant guard
            # (graphrag/agents/tool_policy.py) honours independently of the
            # tenant claim -- letting that token drive write/erase tools
            # against globex despite operating as acme everywhere else.
            if scope.removeprefix("tenant:") != token_tenant:
                raise ValueError("Invalid token")
        elif scope not in trusted.max_scopes:
            raise ValueError("Invalid token")


def decode_access_token(
    token: str,
    *,
    audience: str | None = None,
    strict: bool = False,
) -> dict:
    """Verify `token` and return its claims.

    ``audience``  -- the canonical resource identifier of the resource server
                     doing the verifying. When given, a token whose ``aud``
                     names some other resource is rejected.
    ``strict``    -- also reject a token that carries no ``aud`` claim at all.
                     Requires ``audience``; a resource server cannot demand an
                     audience without naming which one it is.

    ``exp`` is required unconditionally: a token with no expiry is valid
    forever, so its absence must be an error rather than a silently accepted
    shape.

    Dispatches on ``iss`` first -- see the module docstring's "Multi-issuer
    federation" section. This performs no I/O: an externally-issued token
    only verifies against keys already warm in ``issuer_trust``'s cache, so a
    cold external issuer is rejected here. Use ``decode_access_token_async``
    to warm that cache first.

    This does not consult the revocation deny-list -- see the module docstring.
    """
    if strict and audience is None:
        raise ValueError("strict audience validation requires an audience")

    iss = _unverified_claim(token, "iss")
    trusted = None
    if _is_self_issued(iss):
        candidates = _local_candidate_keys(token)
    else:
        trusted = find_trusted_issuer(iss)
        if trusted is None:
            # No fallback to local keys: an unrecognized issuer must never be
            # trusted just because a key happens to verify -- see the module
            # docstring. Deliberately the same generic message as every other
            # rejection path here, so a caller cannot distinguish "unknown
            # issuer" from "bad signature" by probing.
            raise ValueError("Invalid token")
        candidates = _order_by_kid_hint(token, get_jwks_cache().cached_keys(trusted.issuer))

    claims: dict | None = None
    for key in candidates:
        try:
            claims = jwt.decode(
                token,
                key.material,
                # Single-element allow-list per key. Passing the key's own
                # algorithm (never the token's) is what stops an RS256 public
                # key from being replayed as an HS256 shared secret.
                algorithms=[key.algorithm],
                # Audience is checked below, not here -- see the module docstring.
                options={"require": ["exp"], "verify_aud": False},
            )
            break
        except InvalidTokenError:
            continue

    if claims is None:
        raise ValueError("Invalid token")
    if trusted is not None:
        _assert_issuer_scoped_claims(trusted, claims)
    if audience is not None:
        _assert_audience(claims, audience, strict=strict)
    return claims


async def decode_access_token_async(
    token: str,
    *,
    audience: str | None = None,
    strict: bool = False,
) -> dict:
    """``decode_access_token``, but warms a cold external-issuer JWKS cache first.

    Every already-async call site should use this instead of the sync
    ``decode_access_token`` -- ``api/auth/dependencies.py``,
    ``api/auth/default_auth.py``, ``api/routes/auth.py``, and
    ``mcp_server/identity.py``'s ``from_token_checked`` (the remote HTTP MCP
    transport). The one deliberate exception is ``CallerIdentity.from_token``,
    used by the stdio MCP transport and by sync capability tests: it cannot
    await a fetch, so it only verifies self-issued tokens and external-issuer
    tokens whose keys are already warm in this process -- stdio is already
    bound to one trusted launcher rather than an arbitrary network caller
    presenting a token from a freshly-trusted external issuer, so this is an
    accepted, documented limitation rather than a gap.
    """
    iss = _unverified_claim(token, "iss")
    if not _is_self_issued(iss):
        trusted = find_trusted_issuer(iss)
        if trusted is not None:
            try:
                await get_jwks_cache().ensure_keys(trusted)
            except JWKSFetchError as exc:
                raise ValueError("Invalid token") from exc
    return decode_access_token(token, audience=audience, strict=strict)


async def assert_not_revoked(claims: dict) -> None:
    """Raise ValueError if `claims` names a token that has been revoked.

    Separated from `decode_access_token` so the offline proof stays
    synchronous. Called once per request at the auth boundary.
    """
    from graphrag.core.token_revocation import get_revocation_store

    store = await get_revocation_store()
    if await store.is_revoked(claims):
        raise ValueError("Token has been revoked")
