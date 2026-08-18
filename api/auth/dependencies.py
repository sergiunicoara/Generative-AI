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


async def get_tenant(user: dict = Depends(get_current_user)) -> str:
    """Return the tenant bound to the caller's token.

    Tenant is an *authorization* decision, so it is read from the signed JWT
    and never from the request body or query string. Routes that previously
    accepted ``tenant: str = "default"`` from the client let any token holder
    name any tenant; combined with ``"default"`` having been a read-everything
    wildcard in the Cypher layer, that made the whole graph readable with a
    single ``read`` token.
    """
    tenant = user.get("tenant")
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token carries no tenant claim — re-authenticate to obtain a scoped token",
        )
    return tenant


def assert_request_tenant(client_tenant: str, token_tenant: str) -> None:
    """Reject a request whose client-supplied tenant disagrees with the token.

    Some routes take a domain object as the request body (SourceSystem,
    SourceMapping, CGAction, ...) and that object legitimately carries its own
    ``tenant`` field, or name the tenant in the URL path. Either way the value
    is client-controlled, and tenant is an *authorization* decision that must
    come from the signed token (``get_tenant`` above).

    Reject rather than silently overwriting with the token's tenant: an
    overwrite turns both a genuine client bug and a deliberate cross-tenant
    write attempt into an unremarkable 200, so neither is ever noticed.

    403 (not 404) is right here — unlike a resource read, the caller already
    told us which tenant they meant, so there is nothing to conceal by
    pretending the route doesn't exist.
    """
    if client_tenant != token_tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Request tenant {client_tenant!r} does not match the "
                f"authenticated tenant {token_tenant!r}"
            ),
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
