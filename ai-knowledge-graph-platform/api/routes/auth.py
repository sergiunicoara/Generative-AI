"""
OAuth 2.0 routes.

Browser flow  : GET /auth/login  →  Google  →  GET /auth/callback  →  JWT cookie
M2M flow      : POST /auth/clients  (register)
                POST /auth/token   (client_credentials grant)  →  Bearer JWT

Security notes
--------------
- Cookie secure flag is driven by settings.env ("production" → secure=True).
- The `next` redirect parameter is validated to be a safe relative path to
  prevent open-redirect attacks.
- M2M client registry is stored in Redis when available so all API worker
  replicas share the same client table (in-memory dict was per-process).
"""

from __future__ import annotations

import hashlib
import json
import secrets
from urllib.parse import urlparse

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from api.auth.dependencies import get_current_user, get_tenant, require_scope
from api.limiter import AUTH_LIMIT, rate_limit
from graphrag.core.config import get_settings, is_dev_env
from graphrag.core.redis_support import (
    redis_error_types as _redis_error_types,
    sync_redis_client as _get_redis_sync,
)
from graphrag.core.resource_identifiers import (
    InvalidResourceIdentifier,
    known_resources,
    resolve_requested_resource,
)
from graphrag.core.scopes import FIXED_SCOPES, tenant_scope, validate_scopes
from api.auth.google import build_authorization_url, exchange_code_for_userinfo  # pop_state removed (was dead code)
from api.auth.jwt import ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token
from api.auth.user_provisioning import (
    UserIdentityConflict,
    bind_user_identity,
    delete_user_record,
    get_user_record,
    get_user_record_by_identity,
    list_user_records,
    normalize_email,
    set_user_record,
)

router = APIRouter()
log = structlog.get_logger(__name__)
_GOOGLE_ISSUER = "https://accounts.google.com"

# ── Helpers ────────────────────────────────────────────────────────────────────

def _cookie_secure() -> bool:
    """True in production so auth cookies are never sent over plain HTTP."""
    try:
        from graphrag.core.config import get_settings
        return get_settings().env == "production"
    except Exception:  # noqa: BLE001
        return False


def _safe_next(url: str | None, default: str = "/docs") -> str:
    """Return ``url`` if it is a safe relative path, otherwise ``default``.

    Prevents open-redirect attacks: rejects any URL that contains a scheme
    (http://, https://) or a protocol-relative URL (//evil.com).
    """
    if not url:
        return default
    parsed = urlparse(url)
    # A safe relative URL has no scheme and no netloc
    if parsed.scheme or parsed.netloc:
        return default
    # Reject protocol-relative paths like //evil.com
    if url.startswith("//"):
        return default
    return url


# ── M2M client registry (Redis-backed, in-memory fallback) ────────────────────

_CLIENTS_KEY    = "graphrag:m2m_clients"
_m2m_clients_mem: dict[str, dict] = {}   # fallback for non-Redis environments


def _log_client_registry_fallback(operation: str, exc: BaseException) -> None:
    """Record that the M2M client registry just diverged from shared storage.

    A client written to process memory during a Redis outage authenticates on
    exactly one replica; every other replica answers 401 for the same valid
    credential. That is indistinguishable from a bad secret at the caller, so
    it needs to be visible from this side.
    """
    log.warning(
        "auth.client_registry_redis_unavailable",
        operation=operation,
        exception_type=type(exc).__name__,
        impact="falling back to per-process storage; clients will not be shared across replicas",
    )


def _client_get(client_id: str) -> dict | None:
    r = _get_redis_sync()
    if r is not None:
        try:
            raw = r.hget(_CLIENTS_KEY, client_id)
            return json.loads(raw) if raw else None
        except _redis_error_types() as exc:
            # The actual TCP connect happens lazily on this first command,
            # not at from_url() inside _get_redis_sync -- a Redis outage
            # surfaces here, not there.
            _log_client_registry_fallback("client_get", exc)
    return _m2m_clients_mem.get(client_id)


def _client_set(client_id: str, data: dict) -> None:
    r = _get_redis_sync()
    if r is not None:
        try:
            r.hset(_CLIENTS_KEY, client_id, json.dumps(data))
            return
        except _redis_error_types() as exc:
            _log_client_registry_fallback("client_set", exc)
    _m2m_clients_mem[client_id] = data


# ── Dev login (no credentials — development only) ──────────────────────────────

@router.get("/dev-login", summary="⚡ Dev login — issues cookie without Google (dev only)",
            dependencies=[Depends(rate_limit(AUTH_LIMIT))],
            include_in_schema=True)
async def dev_login(request: Request, response: Response, next: str = "/docs"):
    if not is_dev_env(get_settings().env):
        raise HTTPException(status_code=403, detail="Only available in development")

    tenant = get_settings().default_tenant
    # Dev-only bootstrap credential: full scope set so a developer can mint
    # properly scoped-down M2M clients via POST /auth/clients afterward
    # (which enforces requested-scopes-subset-of-caller-scopes) rather than
    # every dev session being permanently capped at "read write" with no way
    # to ever reach biz:write/biz:approve. Never reachable outside is_dev_env().
    token = create_access_token({
        "sub": "dev-user",
        "email": "dev@localhost",
        "name": "Dev User",
        "picture": "",
        "type": "browser",
        "scope": " ".join(sorted(FIXED_SCOPES | {tenant_scope(tenant)})),
        "tenant": tenant,
    })
    secure = _cookie_secure()
    redirect_to = _safe_next(next)
    r = RedirectResponse(redirect_to, status_code=302)
    r.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        secure=secure,
    )
    return r


@router.post("/dev-token",
             dependencies=[Depends(rate_limit(AUTH_LIMIT))],
             summary="⚡ Dev token — returns Bearer JWT as JSON (dev + CLI only)",
             include_in_schema=True)
async def dev_token(request: Request, tenant: str | None = None):
    """Returns a Bearer token as JSON for CLI / PowerShell use in development.

    `tenant` overrides the default tenant for this token — useful for
    minting tokens for a specific tenant during local multi-tenant testing
    without touching global config. Still dev-only; see dev_login's note on
    why this grants the full scope set.
    """
    if not is_dev_env(get_settings().env):
        raise HTTPException(status_code=403, detail="Only available in development")

    effective_tenant = tenant or get_settings().default_tenant
    token = create_access_token({
        "sub": "dev-user",
        "email": "dev@localhost",
        "name": "Dev User",
        "picture": "",
        "type": "m2m",
        "scope": " ".join(sorted(FIXED_SCOPES | {tenant_scope(effective_tenant)})),
        "tenant": effective_tenant,
    })
    return {"access_token": token, "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60}


# ── Browser: Google OAuth 2.0 ─────────────────────────────────────────────────

@router.get("/login", summary="Redirect browser to Google sign-in")
async def login(request: Request, next: str = "/docs"):
    redirect_uri = str(request.base_url).rstrip("/") + "/auth/callback"
    url, state = build_authorization_url(redirect_uri)
    request.session["oauth_state"]  = state
    request.session["next"]         = _safe_next(next)   # validate before storing
    return RedirectResponse(url, status_code=302)


@router.get("/callback", summary="Google OAuth callback — issues JWT cookie")
async def callback(request: Request, code: str, state: str):
    saved_state = request.session.pop("oauth_state", None)
    if not saved_state or not secrets.compare_digest(saved_state, state):
        raise HTTPException(status_code=400, detail="Invalid OAuth state — possible CSRF")

    redirect_uri = str(request.base_url).rstrip("/") + "/auth/callback"
    try:
        userinfo = await exchange_code_for_userinfo(code, redirect_uri)
    except Exception as exc:
        log.warning(
            "auth.google_token_exchange_failed",
            correlation_id=getattr(request.state, "correlation_id", ""),
            exception_type=type(exc).__name__,
        )
        raise HTTPException(status_code=400, detail="Google token exchange failed") from exc

    # Previously this accepted ANY Google account and unconditionally issued
    # a token for settings.default_tenant — there was no persistent mapping
    # from a Google identity to a tenant anywhere, so every real user landed
    # in the same tenant and production multi-tenancy was never actually
    # exercised by real users (only dev-token/M2M ever produced a non-default
    # tenant). An unprovisioned account now gets NO access rather than
    # shared access. See docs/context_graph_gap_plan.md F14.
    subject = userinfo.get("sub")
    email_value = userinfo.get("email")
    if not isinstance(subject, str) or not subject or not isinstance(email_value, str) or not email_value:
        raise HTTPException(status_code=400, detail="Google account is missing a stable identity or email")
    if userinfo.get("email_verified") is not True:
        raise HTTPException(status_code=403, detail="Google account email is not verified")

    # Google documents `sub` as the stable, never-reassigned account key.  An
    # email only bootstraps a new binding after Google verifies ownership.
    record = get_user_record_by_identity(_GOOGLE_ISSUER, subject)
    email = normalize_email(email_value)
    if record is None:
        try:
            record = bind_user_identity(email, issuer=_GOOGLE_ISSUER, subject=subject)
        except KeyError:
            record = None
        except UserIdentityConflict:
            raise HTTPException(
                status_code=403,
                detail="This Google identity is not authorized for the provisioned account",
            )
    if record is None:
        raise HTTPException(
            status_code=403,
            detail="This Google account is not provisioned for any tenant. "
                   "Contact an administrator to be added via POST /auth/users.",
        )
    tenant = record["tenant"]
    # Scopes were already capped at provisioning time (set_user_records's
    # caller intersects against the provisioning admin's own scopes — the
    # same escalation guard register_client uses), so nothing further to
    # intersect here; tenant_scope is always included by that same guard.
    token = create_access_token({
        "sub":     subject,
        "email":   email_value,
        "name":    userinfo.get("name", ""),
        "picture": userinfo.get("picture", ""),
        "type":    "browser",
        "scope":   " ".join(record["scopes"]),
        "tenant":  tenant,
    })

    next_url  = request.session.pop("next", "/docs")
    secure    = _cookie_secure()
    response  = RedirectResponse(next_url, status_code=302)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        secure=secure,
    )
    response.set_cookie(
        key="csrf_token",
        value=secrets.token_urlsafe(32),
        httponly=False,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        secure=secure,
    )
    return response


@router.get("/me", summary="Return current user info")
async def me(user: dict = Depends(get_current_user)):
    return {k: v for k, v in user.items() if k not in ("exp", "iat")}


@router.post("/logout", summary="Clear session cookie")
async def logout(response: Response):
    response.delete_cookie("access_token")
    response.delete_cookie("csrf_token")
    return {"status": "logged_out"}


# ── M2M: Client Credentials grant ─────────────────────────────────────────────

class M2MRegisterRequest(BaseModel):
    client_name: str
    scopes: list[str] = ["read", "write"]


class M2MRegisterResponse(BaseModel):
    client_id: str
    client_secret: str   # shown ONCE — store it securely
    client_name: str
    scopes: list[str]
    tenant: str
    note: str = "Save client_secret now — it will not be shown again."


@router.post(
    "/clients",
    response_model=M2MRegisterResponse,
    summary="Register an M2M client (requires an authenticated write-scoped session)",
    dependencies=[Depends(rate_limit(AUTH_LIMIT))],
)
async def register_client(
    request: Request,
    req: M2MRegisterRequest,
    user: dict = Depends(require_scope("write")),
    tenant: str = Depends(get_tenant),
):
    """Register an M2M client.

    The new client can never exceed the registering caller: its scopes are
    intersected with the caller's own, and it inherits the caller's tenant.
    Previously this depended on ``get_current_user`` alone and stored
    ``req.scopes`` verbatim, so a read-only user could mint themselves a
    write-scoped client — a self-service privilege escalation.
    """
    caller_scopes = set(user.get("scope", "").split())
    # Reject any requested scope that isn't a real scope before the
    # subset check, so a typo'd or made-up scope can't silently pass an
    # intersection against a caller who (harmlessly) also holds it as a
    # stray/malformed entry in an old token.
    requested = set(validate_scopes(req.scopes))
    granted = sorted(requested & caller_scopes)
    if not granted:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"None of the requested scopes {req.scopes} are held by the caller",
        )
    # The client is bound to `tenant` for its whole lifetime (below), so it
    # always carries that tenant's scope regardless of whether the caller
    # explicitly requested it -- omitting it would mint a client that can
    # never pass ToolPolicy's/the capability registry's tenant guard.
    granted = sorted(set(granted) | {tenant_scope(tenant)})

    client_id     = "graphrag_" + secrets.token_urlsafe(16)
    client_secret = secrets.token_urlsafe(40)
    _client_set(client_id, {
        "client_name":  req.client_name,
        "scopes":       granted,
        "secret_hash":  hashlib.sha256(client_secret.encode()).hexdigest(),
        "owner":        user.get("email", user.get("sub")),
        "tenant":       tenant,
    })
    return M2MRegisterResponse(
        client_id=client_id,
        client_secret=client_secret,
        client_name=req.client_name,
        scopes=granted,
        tenant=tenant,
    )


class TokenRequest(BaseModel):
    grant_type: str = "client_credentials"
    client_id: str
    client_secret: str
    scope: str = "read write"
    # RFC 8707 Resource Indicators. The MCP authorization specification makes
    # this mandatory for MCP clients and makes audience validation a MUST for
    # the MCP server, so a caller that wants to reach the MCP transport asks
    # for a token bound to it. Omitted means "the REST API", which keeps every
    # existing client_credentials caller working unchanged.
    resource: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    scope: str
    # Echoing the bound audience lets a client verify it received a token for
    # the resource it asked for, rather than discovering the mismatch later as
    # an opaque 401 from the resource server.
    resource: str


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Issue Bearer JWT for M2M access (client_credentials)",
    dependencies=[Depends(rate_limit(AUTH_LIMIT))],
)
async def token(request: Request, req: TokenRequest):
    if req.grant_type != "client_credentials":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported grant_type: {req.grant_type}",
        )

    try:
        resource = resolve_requested_resource(req.resource)
    except InvalidResourceIdentifier as exc:
        # RFC 8707 Section 2 -- an unrecognised resource is invalid_target.
        # Resolved before the credential check so a bad request never depends
        # on whether the client_id happened to exist.
        log.info("auth.invalid_target", requested=req.resource, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "invalid_target: resource must be one of "
                f"{', '.join(known_resources())}"
            ),
        ) from exc

    client = _client_get(req.client_id)
    if not client:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Unknown client_id")

    secret_hash = hashlib.sha256(req.client_secret.encode()).hexdigest()
    if not secrets.compare_digest(secret_hash, client["secret_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid client_secret")

    client_tenant = client.get("tenant") or get_settings().default_tenant
    requested = set(req.scope.split())
    allowed   = set(client["scopes"])
    granted   = requested & allowed
    # The client's tenant scope is intrinsic to its registration, not
    # something a token request has to remember to name explicitly --
    # TokenRequest.scope defaults to "read write" with no tenant: entry, and
    # requiring every caller to spell out tenant:<name> by hand would mean
    # the default request path never actually carries it. Grant it whenever
    # the client is entitled to it (i.e. it's already in `allowed`),
    # independent of what this particular request asked for.
    client_tenant_scope = tenant_scope(client_tenant)
    if client_tenant_scope in allowed:
        granted = granted | {client_tenant_scope}

    access_token = create_access_token({
        "sub":         req.client_id,
        "client_name": client["client_name"],
        "scope":       " ".join(sorted(granted)),
        "type":        "m2m",
        # Clients registered before tenant binding existed have no stored
        # tenant; fall back to the deployment default rather than issuing a
        # tenantless token that get_tenant would reject with a 403.
        "tenant":      client_tenant,
    }, audience=resource)
    return TokenResponse(
        access_token=access_token,
        scope=" ".join(sorted(granted)),
        resource=resource,
    )


# ── User provisioning: email -> tenant mapping for OAuth login (F14) ──────────
#
# GET /auth/callback rejects any Google account not provisioned here (403),
# rather than defaulting it into settings.default_tenant. See
# docs/context_graph_gap_plan.md F14 and api/auth/user_provisioning.py.

class UserProvisionRequest(BaseModel):
    email: str
    scopes: list[str] = ["read", "write"]


class UserProvisionResponse(BaseModel):
    email: str
    tenant: str
    scopes: list[str]
    added_by: str
    added_at: str


@router.post(
    "/users",
    response_model=UserProvisionResponse,
    summary="Provision a Google account for this tenant (admin only)",
)
async def provision_user(
    req: UserProvisionRequest,
    user: dict = Depends(require_scope("admin")),
    tenant: str = Depends(get_tenant),
):
    """Provision `req.email` to sign in (via /auth/login) as this tenant.

    `tenant` is deliberately NOT a body field — it comes from the caller's
    own token (Depends(get_tenant)), exactly like the /kg/sources fix in F12:
    an admin must not be able to provision a user into a tenant other than
    their own just by naming it in the request body.

    Scopes are intersected with the caller's own, same escalation guard as
    register_client — an admin can never provision a user above their own
    privilege. tenant_scope(tenant) is always included regardless of what was
    requested, so the issued token can pass ToolPolicy's tenant guard.
    """
    caller_scopes = set(user.get("scope", "").split())
    requested = set(validate_scopes(req.scopes))
    granted = sorted((requested & caller_scopes) | {tenant_scope(tenant)})

    existing = get_user_record(req.email)
    if existing is not None and existing.get("tenant") != tenant:
        raise HTTPException(
            status_code=409,
            detail="This Google account is already provisioned for another tenant",
        )

    record = set_user_record(
        req.email,
        tenant=tenant,
        scopes=granted,
        added_by=user.get("email", user.get("sub", "")),
    )
    return UserProvisionResponse(**record)


@router.get(
    "/users",
    response_model=list[UserProvisionResponse],
    summary="List Google accounts provisioned for this tenant (admin only)",
)
async def list_provisioned_users(
    user: dict = Depends(require_scope("admin")),
    tenant: str = Depends(get_tenant),
):
    records = list_user_records(tenant=tenant)
    return [UserProvisionResponse(**r) for r in records]


@router.delete(
    "/users/{email}",
    summary="Revoke a provisioned Google account (admin only)",
)
async def revoke_user(
    email: str,
    user: dict = Depends(require_scope("admin")),
    tenant: str = Depends(get_tenant),
):
    # 404 rather than 403 on a cross-tenant match: don't confirm to caller A
    # that some OTHER tenant provisioned this email at all.
    record = get_user_record(email)
    if record is None or record.get("tenant") != tenant:
        raise HTTPException(status_code=404, detail="No provisioned user with that email in this tenant")
    delete_user_record(email)
    return {"status": "revoked", "email": normalize_email(email)}


# ── Token revocation ─────────────────────────────────────────────────────────
#
# A JWT is valid until it expires and nothing else, so without this the only
# answer to "that token leaked" was rotating the signing key and logging every
# other caller out too. See graphrag/core/token_revocation.py.


class TokenRevokeRequest(BaseModel):
    # RFC 7009 names this field `token`; we accept the decoded id directly as
    # well so an operator holding only a log line (which records `jti`, never
    # the token itself) can act without needing the credential back.
    token: str | None = None
    jti: str | None = None
    # Revoke every token issued to a subject before now. This is the usual
    # incident-response shape: you have the client id, not the tokens.
    subject: str | None = None
    reason: str = ""


class TokenRevokeResponse(BaseModel):
    revoked_tokens: int
    revoked_subjects: int
    durable: bool


def _subject_tenant(subject: str) -> str | None:
    """Best-effort lookup of the tenant that owns `subject` (a `sub` claim).

    Checked against both places a subject can be registered: the M2M client
    registry (`sub` == `client_id`) and a provisioned Google identity (`sub`
    is the Google account id) -- see register_client and the callback route
    above. Returns None when `subject` is not found in either, which callers
    must treat as "cannot verify ownership", not "belongs to no one".
    """
    client = _client_get(subject)
    if client is not None:
        return client.get("tenant")
    record = get_user_record_by_identity(_GOOGLE_ISSUER, subject)
    if record is not None:
        return record.get("tenant")
    return None


@router.post(
    "/revoke",
    response_model=TokenRevokeResponse,
    summary="Revoke an access token or every token for a subject (admin only)",
    dependencies=[Depends(rate_limit(AUTH_LIMIT))],
)
async def revoke_token(
    request: Request,
    req: TokenRevokeRequest,
    user: dict = Depends(require_scope("admin")),
    tenant: str = Depends(get_tenant),
):
    """Deny a leaked credential without rotating the signing key.

    Admin-scoped: revocation is a denial-of-service primitive as much as a
    security one, so the ability to invalidate another caller's session is
    gated at the same level as user provisioning.
    """
    from api.auth.jwt import decode_access_token_async
    from graphrag.core.token_revocation import get_revocation_store

    store = await get_revocation_store()
    revoked_tokens = 0
    revoked_subjects = 0
    durable = True

    target_jti = (req.jti or "").strip()
    target_subject = (req.subject or "").strip()

    if req.token:
        # Decode without an audience: a token being revoked may well have been
        # minted for the MCP resource, and refusing to revoke it here because
        # it is not an API token would be exactly backwards.
        try:
            claims = await decode_access_token_async(req.token)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token could not be verified, so its identifier cannot be trusted",
            ) from exc
        # Tenant-scope the action: an admin of one tenant must not be able to
        # revoke another tenant's credentials.
        if claims.get("tenant") != tenant:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Token belongs to a different tenant",
            )
        target_jti = target_jti or str(claims.get("jti") or "")
        if not target_jti:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token predates revocation support and carries no jti; revoke the subject instead",
            )

    if target_subject:
        # req.token (above) already tenant-checked its own claims -- this
        # covers the subject-only path, which previously had NO tenant check
        # at all: an admin for tenant B who merely knew (from a log line, a
        # shared client id, a leaked doc) another tenant's client_id or
        # Google subject could silently log every one of that tenant's
        # sessions out. Fail open only when the subject cannot be identified
        # at all (unknown to both registries) -- consistent with
        # revocation's existing deny-list, fail-open-by-default design; see
        # graphrag/core/token_revocation.py's module docstring.
        owner_tenant = _subject_tenant(target_subject)
        if owner_tenant is not None and owner_tenant != tenant:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Subject belongs to a different tenant",
            )

    if target_jti:
        durable = await store.revoke_token(
            target_jti, subject=target_subject, reason=req.reason,
        ) and durable
        revoked_tokens = 1

    if target_subject:
        durable = await store.revoke_subject(target_subject, reason=req.reason) and durable
        revoked_subjects = 1

    if not revoked_tokens and not revoked_subjects:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide one of: token, jti, or subject",
        )

    log.info(
        "auth.revocation_requested",
        actor=user.get("sub"), tenant=tenant,
        jti=target_jti[:12] if target_jti else "",
        subject=target_subject, durable=durable,
    )
    return TokenRevokeResponse(
        revoked_tokens=revoked_tokens,
        revoked_subjects=revoked_subjects,
        durable=durable,
    )
