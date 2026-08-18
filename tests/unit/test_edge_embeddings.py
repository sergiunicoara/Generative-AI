"""Unit tests for EdgeEmbeddingService — tenant scoping of RelationEmbedding
nodes (F13).

RelationEmbedding nodes used to be keyed on relation name alone, so one
tenant's TransE training silently overwrote the vector every other tenant's
link prediction read. Two sources have different ownership: 'derived' is a
pure function of the relation name (safe to share, cached under
DERIVED_SCOPE); 'trained' is fitted to one tenant's edges (must be scoped).

See docs/context_graph_gap_plan.md F13. No test previously existed for this
class at all.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.graph.edge_embeddings import DERIVED_SCOPE, EdgeEmbeddingService


def _svc(rows=None):
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(return_value=rows or [])
    return EdgeEmbeddingService(neo4j), neo4j


class TestGetRelationEmbeddingRequiresTenant:
    async def test_tenant_is_required_keyword(self):
        svc, _ = _svc()
        with pytest.raises(TypeError):
            await svc.get_relation_embedding("CEO_OF")


class TestGetRelationEmbeddingReadPath:
    async def test_queries_filter_by_tenant_and_derived_scope(self):
        """The read must never be able to return an arbitrary other tenant's
        row — only this tenant's own or the shared derived one."""
        svc, neo4j = _svc(rows=[])
        await svc.get_relation_embedding("CEO_OF", tenant="acme")
        _, kwargs = neo4j.run.call_args
        assert kwargs["tenant"] == "acme"
        assert kwargs["derived"] == DERIVED_SCOPE
        query = neo4j.run.call_args[0][0]
        assert "re.tenant IN [$tenant, $derived]" in query

    async def test_own_trained_vector_used_when_present(self):
        svc, _ = _svc(rows=[{"embedding": [1.0, 2.0], "tenant": "acme"}])
        result = await svc.get_relation_embedding("CEO_OF", tenant="acme")
        assert result == [1.0, 2.0]

    async def test_falls_back_to_derivation_when_nothing_stored(self):
        svc, _ = _svc(rows=[])
        result = await svc.get_relation_embedding("CEO_OF", tenant="acme")
        assert result == svc._derive_relation_embedding("CEO_OF")

    async def test_cache_is_keyed_by_tenant_not_just_relation(self):
        """Two tenants must not share a cache slot for the same relation name
        — a plain-string cache key would let tenant B's read return whatever
        tenant A's earlier call cached."""
        svc, neo4j = _svc(rows=[{"embedding": [1.0, 2.0], "tenant": "acme"}])
        await svc.get_relation_embedding("CEO_OF", tenant="acme")
        assert ("acme", "CEO_OF") in svc._rel_emb
        assert "CEO_OF" not in svc._rel_emb  # no legacy bare-string key

        neo4j.run = AsyncMock(return_value=[])  # tenant "other" has nothing stored
        result = await svc.get_relation_embedding("CEO_OF", tenant="other")
        assert result != [1.0, 2.0]  # must NOT reuse acme's cached vector
        assert neo4j.run.await_count == 1  # cache miss forced a real query

    async def test_second_call_same_tenant_hits_cache(self):
        svc, neo4j = _svc(rows=[{"embedding": [1.0, 2.0], "tenant": "acme"}])
        await svc.get_relation_embedding("CEO_OF", tenant="acme")
        await svc.get_relation_embedding("CEO_OF", tenant="acme")
        assert neo4j.run.await_count == 1


class TestSeedRelationEmbeddings:
    async def test_writes_under_derived_scope_not_a_real_tenant(self):
        svc, neo4j = _svc(rows=[{"n": 0}])
        await svc.seed_relation_embeddings(["CEO_OF"])
        write_call = [c for c in neo4j.run.await_args_list if "MERGE" in c.args[0]][0]
        assert write_call.kwargs["tenant"] == DERIVED_SCOPE

    async def test_existence_check_is_scoped_to_derived(self):
        svc, neo4j = _svc(rows=[{"n": 1}])  # already exists
        results = await svc.seed_relation_embeddings(["CEO_OF"], overwrite=False)
        check_call = neo4j.run.await_args_list[0]
        assert check_call.kwargs["tenant"] == DERIVED_SCOPE
        assert results == {"CEO_OF": False}  # skipped, already seeded


class TestDerivedEmbeddingDeterminism:
    def test_same_name_same_vector(self):
        svc, _ = _svc()
        assert svc._derive_relation_embedding("CEO_OF") == svc._derive_relation_embedding("CEO_OF")

    def test_different_names_different_vectors(self):
        svc, _ = _svc()
        assert svc._derive_relation_embedding("CEO_OF") != svc._derive_relation_embedding("FOUNDED_BY")
