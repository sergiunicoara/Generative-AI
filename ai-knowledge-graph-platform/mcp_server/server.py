"""MCP server exposing the versioned GraphRAG capability registry.

The local stdio transport resolves its identity from `GRAPHRAG_MCP_TOKEN`;
the remote Streamable HTTP transport binds a verified bearer identity to each
request. In both cases, a missing, expired, or tenant-less token resolves to
an anonymous identity and every capability call is denied with a structured
result rather than letting a process-global identity leak between callers.

Each `@mcp.tool()` function below is a thin wrapper around
`CapabilityRegistry.call()` -- the registry (`mcp_server/capabilities/`), not
this file, is the source of truth for what is exposed, at what version,
under what scopes. `graph_stats` and `query_graph_facts` are registered
under their original bare name as a `legacy_alias`, so an existing MCP
client registration needs no change.

`create_work_order` is the one mutating tool. It requires `biz:write` (and,
for CRITICAL/HIGH-severity findings or any agent-initiated command,
`biz:approve` on a separate approval call before a retry succeeds) --
`CallerIdentity.resolve()` fail-closes to anonymous for a token without
those scopes, so an unscoped or missing `GRAPHRAG_MCP_TOKEN` denies every
write attempt rather than executing one.

Run:
    python mcp_server/server.py

Add to Claude Code as a stdio MCP server, e.g.:
    claude mcp add graphrag --env GRAPHRAG_MCP_TOKEN=<token> -- python mcp_server/server.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

# MCP stdio reserves stdout for JSON-RPC. Configure before importing any
# GraphRAG module that may emit structlog events, otherwise a diagnostic line
# can corrupt the client protocol stream. Remote MCP can safely use stderr too.
import structlog
structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.server.fastmcp.server import Settings as FastMCPSettings  # noqa: E402

from mcp_server.capabilities import build_registry  # noqa: E402
from mcp_server.identity import CallerIdentity  # noqa: E402
from mcp_server.registry import DeniedCapabilityCall  # noqa: E402
from graphrag.observability.correlation import current_correlation_id  # noqa: E402

# The installed FastMCP settings model includes a generic ``lifespan`` forward
# reference. Resolve it before BaseSettings reads environment sources, so MCP
# startup stays warning-free and settings values remain fully typed.
FastMCPSettings.model_rebuild()
mcp = FastMCP("graphrag")
_registry = build_registry()


def _result_to_dict(result: object) -> dict:
    if isinstance(result, DeniedCapabilityCall):
        return {
            "denied": True,
            "capability": result.capability,
            "reason": result.reason,
            "detail": result.detail,
        }
    return result  # type: ignore[return-value]


@mcp.tool()
async def discover_capabilities() -> dict:
    """List versioned capabilities available to this authenticated caller.

    Discovery is entitlement-filtered: unavailable write tools are omitted,
    rather than advertised and then denied on invocation.
    """
    result = await _registry.call(
        "platform.capabilities.discover@1.0.0", {}, CallerIdentity.current(),
    )
    return _result_to_dict(result)


@mcp.tool()
async def graph_stats(tenant: str = "aerospace") -> dict:
    """Return entity, edge, and community counts for a tenant's knowledge graph.

    Read-only. Backed by capability `kg.graph.stats@1.0.0`. `tenant` is
    accepted for wire compatibility with the original tool signature, but is
    now an assertion, not an authority: it must match the caller's
    identity-bound tenant, or the call is denied.
    """
    result = await _registry.call("graph_stats", {"tenant": tenant}, CallerIdentity.current())
    return _result_to_dict(result)


@mcp.tool()
async def query_graph_facts(question: str, tenant: str = "aerospace", limit: int = 25) -> dict:
    """Answer a supported natural-language graph fact question.

    Read-only. Accepts no raw Cypher or SPARQL -- only a fixed set of
    parameterized, tenant-scoped query templates. Backed by capability
    `kg.facts.query@1.0.0`.
    """
    result = await _registry.call(
        "kg.facts.query", {"question": question, "tenant": tenant, "limit": limit}, CallerIdentity.current(),
    )
    return _result_to_dict(result)


@mcp.tool()
async def query_knowledge_graph(
    question: str,
    tenant: str = "aerospace",
    mode: str = "hybrid",
    session_id: str = "",
) -> dict:
    """Return a grounded GraphRAG answer with citations.

    Backed by ``kg.answer.query@1.0.0``.  Retrieval routing stays inside the
    existing HybridRetriever; MCP supplies only authenticated, tenant-bound
    invocation.
    """
    result = await _registry.call(
        "kg.answer.query@1.0.0",
        {"question": question, "tenant": tenant, "mode": mode, "session_id": session_id},
        CallerIdentity.current(),
    )
    return _result_to_dict(result)


@mcp.tool()
async def lookup_entity(
    name: str,
    tenant: str = "aerospace",
    as_of: str | None = None,
    limit: int = 25,
) -> dict:
    """Resolve an entity and return tenant-scoped evidence and relations.

    Backed by ``kg.entity.lookup@1.0.0``. Ambiguous entities are returned as
    candidates; the tool never guesses a canonical identity.
    """
    result = await _registry.call(
        "kg.entity.lookup@1.0.0",
        {"name": name, "tenant": tenant, "as_of": as_of, "limit": limit},
        CallerIdentity.current(),
    )
    return _result_to_dict(result)


@mcp.tool()
async def find_context_precedents(
    policy_version_id: str,
    tenant: str = "aerospace",
    limit: int = 10,
) -> dict:
    """Find tenant-scoped, outcome-backed decision precedents for a policy.

    Backed by ``cg.precedent.find@1.0.0``. Returned scores expose policy
    compatibility, observed outcome state, and feedback tied to that outcome;
    an agent cannot write or self-promote precedent data through this tool.
    """
    result = await _registry.call(
        "cg.precedent.find@1.0.0",
        {"policy_version_id": policy_version_id, "tenant": tenant, "limit": limit},
        CallerIdentity.current(),
    )
    return _result_to_dict(result)


@mcp.tool()
async def create_work_order(
    reason_code: str,
    originating_finding_id: str,
    title: str,
    description: str = "",
    assignee: str = "",
    expected_version: int | None = None,
    dry_run: bool = False,
    approval_id: str | None = None,
    command_id: str | None = None,
) -> dict:
    """Create a remediation work order from a compliance finding.

    Requires `biz:write`. Backed by capability `biz.workorder.create@1.0.0`.
    CRITICAL/HIGH-severity findings (and any agent-initiated call) escalate
    to human approval: the first call returns `outcome: "approval_required"`
    with an `approval_id`; a human with `biz:approve` decides it out of
    band (`POST /business/approvals/{approval_id}/decide`), and retrying
    this call with the same arguments plus that `approval_id` executes it.
    `expected_version` guards against a stale read of the finding -- pass
    the version last observed for it; a mismatch returns
    `outcome: "stale_version"` with the current version, and no write of
    any kind occurs.
    """
    result = await _registry.call(
        "biz.workorder.create@1.0.0",
        {
            "reason_code": reason_code, "originating_finding_id": originating_finding_id,
            "title": title, "description": description, "assignee": assignee,
            "expected_version": expected_version, "dry_run": dry_run,
            "approval_id": approval_id, "command_id": command_id,
            "correlation_id": current_correlation_id(),
        },
        CallerIdentity.current(),
    )
    return _result_to_dict(result)


@mcp.tool()
async def compensate_work_order(
    reason_code: str,
    work_order_id: str,
    original_command_id: str,
    expected_version: int,
    expected_finding_version: int,
    dry_run: bool = False,
    approval_id: str | None = None,
    command_id: str | None = None,
) -> dict:
    """Cancel a work order and reopen its finding through human-approved compensation.

    This does not delete or overwrite the original command.  It records new,
    linked lifecycle transitions and an immutable receipt.  Every compensation
    requires a separate ``biz:approve`` decision, including human requests.
    """
    result = await _registry.call(
        "biz.workorder.compensate@1.0.0",
        {
            "reason_code": reason_code, "work_order_id": work_order_id,
            "original_command_id": original_command_id,
            "expected_version": expected_version,
            "expected_finding_version": expected_finding_version,
            "dry_run": dry_run, "approval_id": approval_id,
            "command_id": command_id, "correlation_id": current_correlation_id(),
        },
        CallerIdentity.current(),
    )
    return _result_to_dict(result)


if __name__ == "__main__":
    mcp.run()
