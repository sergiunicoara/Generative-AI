"""FastAPI dependencies for route protection."""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.auth.jwt import decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> dict:
    """Accept Bearer header only — no cookie/browser session."""
    token: Optional[str] = None
    if credentials:
        token = credentials.credentials

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_scope(scope: str):
    """Dependency factory — enforce a specific scope on ALL token types.

    Previously this only checked scopes when ``type == "m2m"``, which meant
    any browser token (type="browser") bypassed the scope gate entirely.
    The check is now unconditional: if the token doesn't carry the required
    scope, access is denied regardless of how the token was issued.
    """

    async def _check(user: dict = Depends(get_current_user)) -> dict:
        granted = set(user.get("scope", "").split())
        if scope not in granted:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{scope}' required",
            )
        return user

    return _check
