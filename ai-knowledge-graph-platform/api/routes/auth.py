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

from api.auth.dependencies import get_current_user
from api.auth.google import build_authorization_url, exchange_code_for_userinfo  # pop_state removed (was dead code)
from api.auth.jwt import ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token

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


def _client_get(client_id: str) -> dict | None:
    r = _get_redis_sync()
    if r is not None:
        try:
            raw = r.hget(_CLIENTS_KEY, client_id)
            return json.loads(raw) if raw else None
        except (OSError, ConnectionError, ValueError):
            pass
    return _m2m_clients_mem.get(client_id)


def _client_set(client_id: str, data: dict) -> None:
    r = _get_redis_sync()
    if r is not None:
        try:
            r.hset(_CLIENTS_KEY, client_id, json.dumps(data))
            return
        except (OSError, ConnectionError, ValueError):
            pass
    _m2m_clients_mem[client_id] = data


# ── Dev login (no credentials — development only) ──────────────────────────────

@router.get("/dev-login", summary="⚡ Dev login — issues cookie without Google (dev only)",
            include_in_schema=True)
async def dev_login(response: Response, next: str = "/docs"):
    from graphrag.core.config import get_settings
    if get_settings().env != "development":
        raise HTTPException(status_code=403, detail="Only available in development")

    token = create_access_token({
        "sub": "dev-user",
        "email": "dev@localhost",
        "name": "Dev User",
        "picture": "",
        "type": "browser",
        "scope": "read write",
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
async def dev_token():
    """Returns a Bearer token as JSON for CLI / PowerShell use in development."""
    from graphrag.core.config import get_settings
    if get_settings().env != "development":
        raise HTTPException(status_code=403, detail="Only available in development")

    token = create_access_token({
        "sub": "dev-user",
        "email": "dev@localhost",
        "name": "Dev User",
        "picture": "",
        "type": "m2m",
        "scope": "read write",
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

    token = create_access_token({
        "sub":     userinfo["sub"],
        "email":   userinfo["email"],
        "name":    userinfo.get("name", ""),
        "picture": userinfo.get("picture", ""),
        "type":    "browser",
        "scope":   "read write",
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
    note: str = "Save client_secret now — it will not be shown again."


@router.post(
    "/clients",
    response_model=M2MRegisterResponse,
    summary="Register an M2M client (requires browser session)",
)
async def register_client(
    req: M2MRegisterRequest,
    user: dict = Depends(get_current_user),
):
    client_id     = "graphrag_" + secrets.token_urlsafe(16)
    client_secret = secrets.token_urlsafe(40)
    _client_set(client_id, {
        "client_name":  req.client_name,
        "scopes":       req.scopes,
        "secret_hash":  hashlib.sha256(client_secret.encode()).hexdigest(),
        "owner":        user.get("email", user.get("sub")),
    })
    return M2MRegisterResponse(
        client_id=client_id,
        client_secret=client_secret,
        client_name=req.client_name,
        scopes=req.scopes,
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
async def token(req: TokenRequest):
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

    requested = set(req.scope.split())
    allowed   = set(client["scopes"])
    granted   = requested & allowed

    access_token = create_access_token({
        "sub":         req.client_id,
        "client_name": client["client_name"],
        "scope":       " ".join(sorted(granted)),
        "type":        "m2m",
    })
    return TokenResponse(
        access_token=access_token,
        scope=" ".join(sorted(granted)),
    )
