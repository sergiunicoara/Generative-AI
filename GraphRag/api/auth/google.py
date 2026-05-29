"""Google OAuth 2.0 — Authorization Code flow.

State validation uses the session cookie (request.session["oauth_state"]) —
not the old _state_store dict.  The dict was dead code: the redirect_uri
stored in it was never read (it was recomputed at callback time), and the
state value was already validated from the session before pop_state was called.
"""

from __future__ import annotations

import secrets
from urllib.parse import urlencode

import httpx

from graphrag.core.config import get_settings

GOOGLE_AUTH_URL   = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL  = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def build_authorization_url(redirect_uri: str) -> tuple[str, str]:
    """Return (redirect_url, state) for the browser login flow."""
    settings = get_settings()
    state    = secrets.token_urlsafe(32)

    params = {
        "client_id":     settings.google_oauth_client_id,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         state,
        "access_type":   "offline",
        "prompt":        "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}", state


async def exchange_code_for_userinfo(code: str, redirect_uri: str) -> dict:
    """Exchange authorization code for Google userinfo."""
    settings = get_settings()
    async with httpx.AsyncClient(timeout=10) as client:
        token_resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code":          code,
                "client_id":     settings.google_oauth_client_id,
                "client_secret": settings.google_oauth_client_secret,
                "redirect_uri":  redirect_uri,
                "grant_type":    "authorization_code",
            },
        )
        token_resp.raise_for_status()
        tokens = token_resp.json()

        userinfo_resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        userinfo_resp.raise_for_status()
        return userinfo_resp.json()
