"""§13 — 'workspace_id comes from trusted request/authentication context, not a
user-controlled body field.' This vertical slice has no real identity provider
yet (§13's own closing line: the slice 'is not described as production-
authorized until a real identity provider and policy implementation exist') — a
trusted header stands in for that until one exists. Every endpoint depends on
this function rather than reading a header directly, so swapping it for a real
JWT/session-derived workspace_id later changes one function, not every route.
"""

from __future__ import annotations

from fastapi import Header, HTTPException


async def get_workspace_id(x_workspace_id: str = Header(..., alias="X-Workspace-Id")) -> str:
    if not x_workspace_id or not x_workspace_id.strip():
        raise HTTPException(status_code=401, detail="X-Workspace-Id is required")
    return x_workspace_id
