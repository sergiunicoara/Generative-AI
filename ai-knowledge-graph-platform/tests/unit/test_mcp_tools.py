"""Unit tests for mcp_server.tools — the business logic behind the MCP
server's two tools. No ``mcp`` SDK import needed here (tools.py is kept
free of it), so these exercise the same AsyncMock-over-get_neo4j convention
used throughout the suite (see tests/unit/test_pagerank.py's
``_make_computer()`` helper).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from graphrag.core.models import QueryResult
from graphrag.graph.alias_registry import AmbiguousMatch
from mcp_server.capabilities import build_registry
from mcp_server.identity import CallerIdentity
from mcp_server.registry import DeniedCapabilityCall
from mcp_server.tools import lookup_entity, query_knowledge_graph
from mcp_server.tools import query_graph_facts as query_graph_facts_impl


# ── query_knowledge_graph ────────────────────────────────────────────────────

class TestQueryKnowledgeGraph:
    async def test_passes_through_result_and_params(self) -> None:
        retriever = AsyncMock()
        retriever.retrieve_and_answer.return_value = QueryResult(
            query_id="q1",
            question="What supersedes AD-2022?",
            answer="AD-2024 supersedes AD-2022.",
            contexts=["ctx"],
            citations=["AD-2024"],
            latency_ms=123.4,
            retrieval_mode="hybrid",
            model_version="deepseek-v4-pro",
        )

        result = await query_knowledge_graph(
            retriever, "What supersedes AD-2022?",
            mode="hybrid", tenant="aerospace", session_id="sess-1",
        )

        retriever.retrieve_and_answer.assert_awaited_once_with(
            question="What supersedes AD-2022?", mode="hybrid",
            tenant="aerospace", session_id="sess-1",
        )
        assert result["answer"] == "AD-2024 supersedes AD-2022."
        assert result["citations"] == ["AD-2024"]
        assert result["retrieval_mode"] == "hybrid"


# ── lookup_entity ─────────────────────────────────────────────────────────────

def _patch_registry(resolve_return):
    registry = MagicMock()
    registry.resolve.return_value = resolve_return
    return patch(
        "mcp_server.tools.load_alias_registry",
        AsyncMock(return_value=registry),
    )


class TestLookupEntity:
    async def test_resolved_entity_fetches_neighbors_and_pagerank(self) -> None:
        neo4j = AsyncMock()
        neo4j.get_relations_for_entity.return_value = [
            {"name": "Boeing 737 MAX", "type": "AIRCRAFT_MODEL",
             "weight": None, "confidence": 0.9, "extracted_at": "2026-01-01",
             "source_doc_id": "fleet.pdf", "direction": "outgoing"},
        ]
        neo4j.get_pagerank_by_entity_names.return_value = {"FAA": 0.42}

        with (
            _patch_registry(("FAA", "ORG")),
            patch("mcp_server.tools.get_neo4j", return_value=neo4j),
        ):
            result = await lookup_entity("faa", tenant="aerospace", limit=10)

        neo4j.get_relations_for_entity.assert_awaited_once_with(
            "FAA", "ORG", tenant="aerospace", as_of=None, limit=10,
        )
        neo4j.get_pagerank_by_entity_names.assert_awaited_once_with(
            ["FAA"], tenant="aerospace",
        )
        assert result["resolved"] is True
        assert result["canonical_name"] == "FAA"
        assert result["canonical_type"] == "ORG"
        assert result["importance_pagerank"] == 0.42
        assert result["relations"] == neo4j.get_relations_for_entity.return_value

    async def test_no_match_returns_unresolved_without_neo4j_calls(self) -> None:
        neo4j = AsyncMock()
        with (
            _patch_registry(None),
            patch("mcp_server.tools.get_neo4j", return_value=neo4j),
        ):
            result = await lookup_entity("nonexistent entity")

        assert result == {
            "resolved": False,
            "query": "nonexistent entity",
            "message": "No matching entity found.",
        }
        neo4j.get_relations_for_entity.assert_not_awaited()
        neo4j.get_pagerank_by_entity_names.assert_not_awaited()

    async def test_ambiguous_match_returns_candidate_without_neo4j_calls(self) -> None:
        neo4j = AsyncMock()
        ambiguous = AmbiguousMatch(candidate=("Acme Corp", "ORG"), score=0.91, match_type="embedding")
        with (
            _patch_registry(ambiguous),
            patch("mcp_server.tools.get_neo4j", return_value=neo4j),
        ):
            result = await lookup_entity("Acme Corporation")

        assert result["resolved"] is False
        assert result["ambiguous_candidate"] == {
            "name": "Acme Corp", "type": "ORG", "score": 0.91, "match_type": "embedding",
        }
        neo4j.get_relations_for_entity.assert_not_awaited()
        neo4j.get_pagerank_by_entity_names.assert_not_awaited()

    async def test_missing_pagerank_signal_is_none_not_zero(self) -> None:
        """pagerank.get() on a key that was never computed must surface as
        None, not silently coerce to 0 — a real 0 and 'never computed' mean
        different things (see get_pagerank_by_entity_names's own docstring)."""
        neo4j = AsyncMock()
        neo4j.get_relations_for_entity.return_value = []
        neo4j.get_pagerank_by_entity_names.return_value = {}  # entity absent = no signal

        with (
            _patch_registry(("Obscure Entity", "CONCEPT")),
            patch("mcp_server.tools.get_neo4j", return_value=neo4j),
        ):
            result = await lookup_entity("obscure entity")

        assert result["importance_pagerank"] is None

    async def test_as_of_threaded_through_to_neighbors_lookup(self) -> None:
        neo4j = AsyncMock()
        neo4j.get_relations_for_entity.return_value = []
        neo4j.get_pagerank_by_entity_names.return_value = {}

        with (
            _patch_registry(("AD-2024", "AIRWORTHINESS_DIRECTIVE")),
            patch("mcp_server.tools.get_neo4j", return_value=neo4j),
        ):
            await lookup_entity("AD-2024", as_of="2026-06-01")

        neo4j.get_relations_for_entity.assert_awaited_once_with(
            "AD-2024", "AIRWORTHINESS_DIRECTIVE",
            tenant="default", as_of="2026-06-01", limit=25,
        )


# ── Capability registry wiring (kg.graph.stats, kg.facts.query) ────────────────

class TestCapabilityRegistryWiring:
    def test_graph_stats_registered_with_legacy_alias(self):
        registry = build_registry()
        spec = registry.resolve("graph_stats", CallerIdentity.from_claims(
            {"sub": "reader", "tenant": "aerospace", "scope": "read"},
        ))
        assert spec.qualified_name == "kg.graph.stats@1.0.0"

    def test_facts_query_registered_re_admitting_orphaned_tool(self):
        registry = build_registry()
        spec = registry.resolve("kg.facts.query@1.0.0", CallerIdentity.from_claims(
            {"sub": "reader", "tenant": "aerospace", "scope": "read"},
        ))
        assert spec.fn is query_graph_facts_impl

    async def test_graph_stats_denies_cross_tenant_caller(self):
        registry = build_registry()
        identity = CallerIdentity(
            subject="agent-1", tenant="aerospace",
            scopes=frozenset({"read"}), authenticated=True,
        )
        result = await registry.call("graph_stats", {"tenant": "other-tenant"}, identity)
        assert isinstance(result, DeniedCapabilityCall)
        assert result.reason == "tenant_mismatch"

    async def test_graph_stats_denies_anonymous_caller(self):
        registry = build_registry()
        result = await registry.call("graph_stats", {}, CallerIdentity.anonymous())
        assert isinstance(result, DeniedCapabilityCall)
        assert result.reason == "missing_scope"

    async def test_context_precedent_capability_is_tenant_bound(self):
        registry = build_registry()
        neo4j = MagicMock()
        neo4j.run = AsyncMock(return_value=[])
        identity = CallerIdentity(
            subject="agent-1", tenant="aerospace", scopes=frozenset({"read", "tenant:aerospace"}),
            authenticated=True,
        )
        with patch("mcp_server.capabilities.context_precedent.get_neo4j", return_value=neo4j):
            result = await registry.call(
                "cg.precedent.find", {"policy_version_id": "policy-v3", "tenant": "aerospace"}, identity,
            )
        assert result == []
        assert neo4j.run.await_args.kwargs["tenant"] == "aerospace"


# ── Neo4jClient.get_relations_for_entity ────────────────────────────────────────

class TestGetRelationsForEntityQueryShape:
    async def test_threads_params_and_omits_as_of_when_not_given(self) -> None:
        from graphrag.graph.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)  # bypass __init__ (no real driver)
        client.run = AsyncMock(return_value=[])

        await client.get_relations_for_entity("FAA", "ORG", tenant="aerospace", limit=5)

        _, kwargs = client.run.call_args
        assert kwargs["name"] == "FAA"
        assert kwargs["type"] == "ORG"
        assert kwargs["tenant"] == "aerospace"
        assert kwargs["limit"] == 5
        assert "as_of" not in kwargs

    async def test_includes_as_of_and_temporal_filter_when_given(self) -> None:
        from graphrag.graph.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[])

        await client.get_relations_for_entity(
            "AD-2024", "AIRWORTHINESS_DIRECTIVE", as_of="2026-06-01",
        )

        args, kwargs = client.run.call_args
        cypher = args[0]
        assert kwargs["as_of"] == "2026-06-01"
        assert "valid_from" in cypher and "valid_to" in cypher
