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

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from api.auth.dependencies import get_current_user, get_tenant, require_scope
from api.limiter import AUTH_LIMIT, limiter
from graphrag.core.config import get_settings, is_dev_env
from graphrag.core.scopes import FIXED_SCOPES, tenant_scope, validate_scopes
from api.auth.google import build_authorization_url, exchange_code_for_userinfo  # pop_state removed (was dead code)
from api.auth.jwt import ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token
from api.auth.user_provisioning import (
    delete_user_record,
    get_user_record,
    list_user_records,
    normalize_email,
    set_user_record,
)

router = APIRouter()

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


def _get_redis_sync():
    """Return a sync Redis client for the M2M registry, or None."""
    try:
        import redis as redis_lib
        from graphrag.core.config import get_settings
        redis_url = get_settings().retrieval.get("redis_url", "")
        if not redis_url:
            return None
        return redis_lib.from_url(redis_url, socket_connect_timeout=1,
                                  socket_timeout=1, decode_responses=True)
    except (ImportError, OSError, ConnectionError, ValueError):
        return None


def _redis_error_types() -> tuple[type[BaseException], ...]:
    """redis.exceptions.RedisError covers TimeoutError/ConnectionError/etc.
    Lazily imported to match _get_redis_sync's existing convention (no hard
    top-level dependency on the redis package). Falls back to just the
    stdlib types if redis truly isn't installed -- _get_redis_sync would
    already have returned None in that case, so this branch is defensive."""
    try:
        import redis as redis_lib
        return (redis_lib.exceptions.RedisError, OSError, ConnectionError, ValueError)
    except ImportError:
        return (OSError, ConnectionError, ValueError)


def _client_get(client_id: str) -> dict | None:
    r = _get_redis_sync()
    if r is not None:
        try:
            raw = r.hget(_CLIENTS_KEY, client_id)
            return json.loads(raw) if raw else None
        except _redis_error_types():
            # The actual TCP connect happens lazily on this first command,
            # not at from_url() inside _get_redis_sync -- a Redis outage
            # surfaces here, not there.
            pass
    return _m2m_clients_mem.get(client_id)


def _client_set(client_id: str, data: dict) -> None:
    r = _get_redis_sync()
    if r is not None:
        try:
            r.hset(_CLIENTS_KEY, client_id, json.dumps(data))
            return
        except _redis_error_types():
            pass
    _m2m_clients_mem[client_id] = data


# ── Dev login (no credentials — development only) ──────────────────────────────

@router.get("/dev-login", summary="⚡ Dev login — issues cookie without Google (dev only)",
            include_in_schema=True)
@limiter.limit(AUTH_LIMIT)
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
             summary="⚡ Dev token — returns Bearer JWT as JSON (dev + CLI only)",
             include_in_schema=True)
@limiter.limit(AUTH_LIMIT)
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
        raise HTTPException(status_code=400, detail=f"Google token exchange failed: {exc}")

    # Previously this accepted ANY Google account and unconditionally issued
    # a token for settings.default_tenant — there was no persistent mapping
    # from a Google identity to a tenant anywhere, so every real user landed
    # in the same tenant and production multi-tenancy was never actually
    # exercised by real users (only dev-token/M2M ever produced a non-default
    # tenant). An unprovisioned account now gets NO access rather than
    # shared access. See docs/context_graph_gap_plan.md F14.
    email = normalize_email(userinfo["email"])
    record = get_user_record(email)
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
        "sub":     userinfo["sub"],
        "email":   userinfo["email"],
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
    return response


@router.get("/me", summary="Return current user info")
async def me(user: dict = Depends(get_current_user)):
    return {k: v for k, v in user.items() if k not in ("exp", "iat")}


@router.post("/logout", summary="Clear session cookie")
async def logout(response: Response):
    response.delete_cookie("access_token")
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
)
@limiter.limit(AUTH_LIMIT)
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


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    scope: str


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Issue Bearer JWT for M2M access (client_credentials)",
)
@limiter.limit(AUTH_LIMIT)
async def token(request: Request, req: TokenRequest):
    if req.grant_type != "client_credentials":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported grant_type: {req.grant_type}",
        )

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
    })
    return TokenResponse(
        access_token=access_token,
        scope=" ".join(sorted(granted)),
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
