"""§10 — tenant-safe query execution modes.

`tenant_query` is the only mode repositories should use for anything touching
sales-domain nodes. It structurally rejects Cypher that doesn't scope a matched
node/relationship by `workspace_id` — via a `{...}` property map (scoped_match(),
the pattern every src/graph/repositories/*.py MATCH/MERGE uses) or via an
explicit `x.workspace_id = $workspace_id` equality (the pattern full-text/vector
procedure calls need, since `CALL db.index.fulltext.queryNodes(...) YIELD node`
has no property-map MATCH pattern to scope at all — the WHERE-clause form is how
§10's 'full-text retrieval must also be tenant-safe; do not obtain a global
top-k and merely discard other workspaces afterward' gets satisfied for those).
A bare `$workspace_id` parameter that's merely passed but never matched against
either way is exactly the failure mode §10 calls out ("not merely include a
parameter named $workspace_id").

`schema_query` and `operational_query` are allowlisted by call-site convention —
src/graph/schema.py and src/graph/migrations/* for the former, health/index-
metadata reads (e.g. /ready, SHOW INDEXES) for the latter. This vertical slice has
no separate caller-identity/ACL layer to enforce that mechanically; §13 already
states the slice "is not production-authorized until a real identity provider and
policy implementation exist" — that gap is inherited here, not hidden.
"""

from __future__ import annotations

import re

import structlog

from src.core.neo4j_client import Neo4jClient, get_neo4j

log = structlog.get_logger(__name__)


class TenantScopingError(RuntimeError):
    """Raised when tenant_query() Cypher doesn't scope a matched node/relationship
    by workspace_id as a property — see module docstring."""


_WORKSPACE_PROP_MAP_PATTERN = re.compile(r"\{[^{}]*\bworkspace_id\s*:\s*\$workspace_id\b[^{}]*\}")
_WORKSPACE_WHERE_EQUALITY_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.workspace_id\s*=\s*\$workspace_id\b")


def _is_workspace_scoped(cypher: str) -> bool:
    return bool(_WORKSPACE_PROP_MAP_PATTERN.search(cypher) or _WORKSPACE_WHERE_EQUALITY_PATTERN.search(cypher))


def scoped_match(label: str, var: str, **extra_props: str) -> str:
    """Build a node pattern that always scopes `var` by workspace_id as a matched
    property, never only as a passed parameter.

    `extra_props` maps additional property names to the Cypher parameter name
    that supplies them, e.g.::

        scoped_match("Account", "a", account_id="account_id")
        # -> "(a:Account {workspace_id: $workspace_id, account_id: $account_id})"
    """
    props = {"workspace_id": "workspace_id", **extra_props}
    props_cypher = ", ".join(f"{key}: ${param}" for key, param in props.items())
    return f"({var}:{label} {{{props_cypher}}})"


class GraphExecutor:
    def __init__(self, client: Neo4jClient | None = None):
        self._client = client or get_neo4j()

    async def tenant_query(self, cypher: str, *, workspace_id: str, **params) -> list[dict]:
        if not _is_workspace_scoped(cypher):
            raise TenantScopingError(
                "tenant_query() requires either 'workspace_id: $workspace_id' "
                "inside a node/relationship property map (see scoped_match()) "
                "or an explicit 'x.workspace_id = $workspace_id' equality — a "
                "bare $workspace_id parameter alone does not scope the query. "
                f"Offending query: {cypher!r}"
            )
        return await self._client.run(cypher, workspace_id=workspace_id, **params)

    async def schema_query(self, cypher: str, **params) -> list[dict]:
        """Allowlisted for migrations/schema bootstrap only — src/graph/schema.py."""
        return await self._client.run(cypher, **params)

    async def operational_query(self, cypher: str, **params) -> list[dict]:
        """Allowlisted for health/index-metadata reads only (e.g. /ready, SHOW INDEXES)."""
        return await self._client.run(cypher, **params)
