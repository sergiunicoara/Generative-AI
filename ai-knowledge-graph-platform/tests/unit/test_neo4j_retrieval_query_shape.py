"""Regression guards for interpolated retrieval Cypher fragments."""

from unittest.mock import AsyncMock

from graphrag.graph.neo4j_client import Neo4jClient


def _client() -> Neo4jClient:
    client = Neo4jClient.__new__(Neo4jClient)
    client.run = AsyncMock(return_value=[])
    return client


async def test_entity_neighbors_interpolates_bitemporal_filters() -> None:
    client = _client()

    await client.get_entity_neighbors(
        ["chunk-a"],
        as_of="2026-01-01",
        transaction_at="2026-02-01",
        tenant="acme",
    )

    cypher = client.run.await_args.args[0]
    assert "{temporal_filter}" not in cypher
    assert "{transaction_filter}" not in cypher
    assert "r.valid_from" in cypher
    assert "r.recorded_at" in cypher
    assert client.run.await_args.kwargs["tenant"] == "acme"


async def test_multihop_interpolates_depth_tenant_temporal_and_semantic_score() -> None:
    client = _client()

    await client.get_multihop_chunks(
        ["chunk-a"],
        hops=3,
        as_of="2026-01-01",
        transaction_at="2026-02-01",
        tenant="acme",
        query_embedding=[0.1, 0.2],
        semantic_weight=0.5,
    )

    cypher = client.run.await_args.args[0]
    for placeholder in (
        "{hops}", "{temporal_filter}", "{transaction_filter}",
        "{tenant_filter}", "{sem_sim_expr}", "{score_expr}",
    ):
        assert placeholder not in cypher
    assert "[:RELATES_TO*1..3]" in cypher
    assert "ALL(r IN relationships(path) WHERE r.tenant = $tenant)" in cypher
    assert "vector.similarity.cosine" in cypher
    assert client.run.await_args.kwargs["sem_w"] == 0.5


async def test_relation_subgraph_interpolates_bitemporal_filters() -> None:
    client = _client()

    await client.get_entity_relations_subgraph(
        [{"name": "SpaceX", "type": "ORG"}],
        as_of="2026-01-01",
        transaction_at="2026-02-01",
        tenant="acme",
    )

    cypher = client.run.await_args.args[0]
    assert "{temporal_filter}" not in cypher
    assert "{transaction_filter}" not in cypher
    assert "r.valid_from" in cypher
    assert "r.recorded_at" in cypher
    assert client.run.await_args.kwargs["tenant"] == "acme"
