"""Tool implementations for the GraphRAG MCP server.

Kept free of any ``mcp`` SDK import so these functions can be unit-tested
with the same AsyncMock-over-``get_neo4j`` convention used elsewhere in the
test suite, without the ``mcp`` package installed or a transport running.

All functions in this module are registered through versioned capability
adapters. They still accept ``tenant`` only because the adapter needs a stable
function signature; ``CapabilityRegistry`` replaces it with the
identity-bound tenant after rejecting mismatches.
"""

from __future__ import annotations

import structlog

from graphrag.graph.alias_registry import AmbiguousMatch, load_alias_registry
from graphrag.graph.controlled_query import ControlledQueryError, execute_controlled_query
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.core.config import get_settings
from graphrag.enterprise.models import AccessContext

log = structlog.get_logger(__name__)


async def query_knowledge_graph(
    retriever,
    question: str,
    mode: str = "hybrid",
    tenant: str = "default",
    session_id: str = "",
    access_context: AccessContext | None = None,
) -> dict:
    kwargs = {"question": question, "mode": mode, "tenant": tenant, "session_id": session_id}
    # Preserve the legacy capability call shape when no identity adapter is
    # involved; the registered MCP capability always supplies a context.
    if access_context is not None:
        kwargs["access_context"] = access_context
    result = await retriever.retrieve_and_answer(**kwargs)
    return result.model_dump(mode="json")


async def lookup_entity(
    name: str,
    tenant: str = "default",
    as_of: str | None = None,
    limit: int = 25,
) -> dict:
    if get_settings().access_control.get("enabled", False):
        return {
            "resolved": False,
            "message": "Entity lookup is unavailable while document ACL enforcement is active.",
            "reason": "acl_derived_graph_not_materialized",
        }
    registry = await load_alias_registry(tenant=tenant)
    resolved = registry.resolve(name)

    if resolved is None:
        return {"resolved": False, "query": name, "message": "No matching entity found."}

    if isinstance(resolved, AmbiguousMatch):
        return {
            "resolved": False,
            "query": name,
            "ambiguous_candidate": {
                "name": resolved.candidate[0],
                "type": resolved.candidate[1],
                "score": resolved.score,
                "match_type": resolved.match_type,
            },
            "message": "Match is below the auto-merge confidence threshold.",
        }

    canonical_name, canonical_type = resolved
    client = get_neo4j()
    neighbors = await client.get_relations_for_entity(
        canonical_name, canonical_type, tenant=tenant, as_of=as_of, limit=limit,
    )
    pagerank = await client.get_pagerank_by_entity_names([canonical_name], tenant=tenant)

    return {
        "resolved": True,
        "query": name,
        "canonical_name": canonical_name,
        "canonical_type": canonical_type,
        # None = no PageRank signal computed for this entity yet — never
        # coerce to 0, that would misrepresent "unknown" as "unimportant".
        "importance_pagerank": pagerank.get(canonical_name),
        "relations": neighbors,
    }


async def query_graph_facts(question: str, tenant: str = "default", limit: int = 25) -> dict:
    """Run a supported natural-language graph fact query safely.

    The implementation accepts no raw Cypher or SPARQL. It only executes a
    fixed, parameterized and tenant-scoped query template selected by the
    deterministic controlled-query planner.
    """
    if get_settings().access_control.get("enabled", False):
        return {
            "supported": False,
            "tenant": tenant,
            "message": "Controlled fact queries are unavailable while document ACL enforcement is active.",
            "reason": "acl_derived_graph_not_materialized",
        }
    try:
        return await execute_controlled_query(
            get_neo4j(), question, tenant=tenant, limit=limit,
        )
    except ControlledQueryError as exc:
        return {"supported": False, "tenant": tenant, "message": str(exc)}
