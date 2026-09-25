"""Canonical OAuth-style scope vocabulary for this platform.

Before this module, every token-issuing site (api/routes/auth.py) emitted
only the literal string "read write" — no scope has ever named a tenant, a
capability area, or an approval right. `ToolPolicy`'s cross-tenant guard
(graphrag/agents/tool_policy.py) checks for a `tenant:<name>` scope prefix
that no real token has ever carried, which made that guard unreachable in
practice: every real caller failed it, and no test caught this because the
tests construct scopes by hand rather than through real issuance.

This module is the single source of truth for what a scope string is allowed
to mean, so `register_client` (POST /auth/clients) can reject a nonsense
scope instead of storing it verbatim and letting `require_scope` silently
never match it.
"""
from __future__ import annotations

import re

# Fixed, finite scopes. Adding a new one is a deliberate code change here,
# not something a caller can invent by asking for it in POST /auth/clients.
FIXED_SCOPES = frozenset({
    "read",
    "write",
    "admin",
    "biz:read",
    "biz:write",
    "biz:approve",
})

# Single source of truth for "what is a valid tenant name" -- every other
# tenant-name pattern in the codebase (api/routes/kg/knowledge.py's request-path
# validator, graphrag/graph/triplestore.py's remote-SPARQL tenant binding) is
# built from this one, so a name accepted in one place can never be rejected
# (or normalized differently) in another.
TENANT_NAME_PATTERN = r"[a-z0-9][a-z0-9_-]{0,63}"
TENANT_NAME_RE = re.compile(rf"^{TENANT_NAME_PATTERN}$")

# tenant:<name> is a family, not a fixed set -- one exists per tenant.
_TENANT_SCOPE_RE = re.compile(rf"^tenant:{TENANT_NAME_PATTERN}$")


def is_valid_scope(scope: str) -> bool:
    """True if `scope` is a recognized fixed scope or a well-formed tenant:<name>."""
    return scope in FIXED_SCOPES or bool(_TENANT_SCOPE_RE.fullmatch(scope))


def tenant_from_scope(scope: str) -> str | None:
    """The tenant name if `scope` is a well-formed tenant:<name> scope, else None.

    Single parser for the convention this module owns -- api/auth/jwt.py and
    graphrag/agents/tool_policy.py each used to hand-parse `tenant:` scopes
    independently (one via removeprefix, the other via split(":", 1)), so a
    future change to the convention (e.g. case-insensitive names) had two
    call sites to update instead of one.
    """
    if not scope.startswith("tenant:"):
        return None
    name = scope[len("tenant:"):]
    return name if TENANT_NAME_RE.fullmatch(name) else None


def validate_scopes(scopes: list[str]) -> list[str]:
    """Return `scopes` filtered to only recognized ones.

    Deliberately filters rather than raises: callers (register_client) already
    intersect the request against the caller's OWN scopes before this runs, so
    by the time validate_scopes sees a list, an unrecognized entry is either a
    typo or a scope name that doesn't exist -- either way, dropping it silently
    denies the unrecognized part rather than blocking the whole request over
    one bad entry. Call sites that want to know what was dropped should diff
    the input against the output themselves.
    """
    return [s for s in scopes if is_valid_scope(s)]


def tenant_scope(tenant: str) -> str:
    """Build the tenant:<name> scope string for `tenant`."""
    return f"tenant:{tenant}"
