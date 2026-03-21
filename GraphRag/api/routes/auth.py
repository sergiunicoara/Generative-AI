"""
OAuth 2.0 routes.

Browser flow  : GET /auth/login  →  Google  →  GET /auth/callback  →  JWT cookie
M2M flow      : POST /auth/clients  (register)
                POST /auth/token   (client_credentials grant)  →  Bearer JWT
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from api.auth.dependencies import get_current_user
from api.auth.google import build_authorization_url, exchange_code_for_userinfo, pop_state
from api.auth.jwt import ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token

router = APIRouter()

# ── In-memory M2M client registry (swap for DB / Vault in prod) ───────────────
_m2m_clients: dict[str, dict] = {}


# ── Dev login (no credentials — development only) ──────────────────────────────

@router.get("/dev-login", summary="⚡ Dev login — issues cookie without Google (dev only)", include_in_schema=True)
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
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        secure=False,
    )
    from fastapi.responses import RedirectResponse as RR
    r = RR(next, status_code=302)
    r.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        secure=False,
    )
    return r


# ── Browser: Google OAuth 2.0 ─────────────────────────────────────────────────

@router.get("/login", summary="Redirect browser to Google sign-in")
async def login(request: Request, next: str = "/docs"):
    redirect_uri = str(request.base_url).rstrip("/") + "/auth/callback"
    url, state = build_authorization_url(redirect_uri)
    request.session["oauth_state"] = state
    request.session["next"] = next
    return RedirectResponse(url, status_code=302)


@router.get("/callback", summary="Google OAuth callback — issues JWT cookie")
async def callback(request: Request, code: str, state: str):
    saved_state = request.session.pop("oauth_state", None)
    if not saved_state or not secrets.compare_digest(saved_state, state):
        raise HTTPException(status_code=400, detail="Invalid OAuth state — possible CSRF")

    # also clear from in-memory store
    pop_state(state)

    redirect_uri = str(request.base_url).rstrip("/") + "/auth/callback"
    try:
        userinfo = await exchange_code_for_userinfo(code, redirect_uri)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Google token exchange failed: {exc}")

    token = create_access_token(
        {
            "sub": userinfo["sub"],
            "email": userinfo["email"],
            "name": userinfo.get("name", ""),
            "picture": userinfo.get("picture", ""),
            "type": "browser",
            "scope": "read write",
        }
    )

    next_url = request.session.pop("next", "/docs")
    response = RedirectResponse(next_url, status_code=302)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        secure=False,  # set True behind HTTPS
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
    client_secret: str  # shown ONCE — store it securely
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
    client_id = "graphrag_" + secrets.token_urlsafe(16)
    client_secret = secrets.token_urlsafe(40)
    _m2m_clients[client_id] = {
        "client_name": req.client_name,
        "scopes": req.scopes,
        "secret_hash": hashlib.sha256(client_secret.encode()).hexdigest(),
        "owner": user.get("email", user.get("sub")),
    }
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

    client = _m2m_clients.get(req.client_id)
    if not client:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown client_id")

    secret_hash = hashlib.sha256(req.client_secret.encode()).hexdigest()
    if not secrets.compare_digest(secret_hash, client["secret_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid client_secret")

    requested = set(req.scope.split())
    allowed = set(client["scopes"])
    granted = requested & allowed

    access_token = create_access_token(
        {
            "sub": req.client_id,
            "client_name": client["client_name"],
            "scope": " ".join(sorted(granted)),
            "type": "m2m",
        }
    )
    return TokenResponse(
        access_token=access_token,
        scope=" ".join(sorted(granted)),
    )
