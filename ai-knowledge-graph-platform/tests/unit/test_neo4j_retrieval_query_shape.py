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


# include_superseded defaults to True (include, today's pre-existing
# behavior unchanged) rather than False (exclude) -- a stronger intervention
# than downweighting that hasn't yet been validated against the aerospace
# golden eval (tasks/lessons.md A128 precedent). These tests assert the
# WHERE fragment exists and both explicit values thread through correctly;
# they intentionally do NOT assert which value is the default, since that's
# a one-line, deliberate flip once golden-eval validation passes -- see
# neo4j_client.vector_search_chunks's docstring.

async def test_vector_search_chunks_supports_superseded_exclusion() -> None:
    client = _client()

    await client.vector_search_chunks([0.1, 0.2], tenant="acme", include_superseded=False)

    cypher = client.run.await_args.args[0]
    assert "d.superseded_by IS NULL" in cypher
    assert client.run.await_args.kwargs["include_superseded"] is False


async def test_vector_search_chunks_filtered_index_path_supports_superseded_exclusion() -> None:
    client = _client()
    client._filtered_vector_search = True

    await client.vector_search_chunks([0.1, 0.2], tenant="acme", include_superseded=False)

    cypher = client.run.await_args.args[0]
    assert "d.superseded_by IS NULL" in cypher
    assert client.run.await_args.kwargs["include_superseded"] is False


async def test_bm25_search_chunks_supports_superseded_exclusion() -> None:
    client = _client()

    await client.bm25_search_chunks("query", tenant="acme", include_superseded=False)

    cypher = client.run.await_args.args[0]
    assert "d.superseded_by IS NULL" in cypher
    assert client.run.await_args.kwargs["include_superseded"] is False


async def test_bm25_search_entities_supports_superseded_exclusion() -> None:
    client = _client()

    await client.bm25_search_entities("query", tenant="acme", include_superseded=False)

    cypher = client.run.await_args.args[0]
    assert "d.superseded_by IS NULL" in cypher
    assert client.run.await_args.kwargs["include_superseded"] is False


async def test_get_linked_document_chunks_supports_superseded_exclusion() -> None:
    client = _client()

    await client.get_linked_document_chunks(["chunk-a"], tenant="acme", include_superseded=False)

    cypher = client.run.await_args.args[0]
    assert "source.superseded_by IS NULL" in cypher
    assert "target.superseded_by IS NULL" in cypher
    assert client.run.await_args.kwargs["include_superseded"] is False


async def test_get_best_chunk_for_document_supports_superseded_exclusion() -> None:
    client = _client()

    await client.get_best_chunk_for_document(
        "doc.txt", [0.1, 0.2], tenant="acme", include_superseded=False,
    )

    cypher = client.run.await_args.args[0]
    assert "d.superseded_by IS NULL" in cypher
    assert client.run.await_args.kwargs["include_superseded"] is False


async def test_vector_search_chunks_defaults_to_including_superseded() -> None:
    """Today's pre-existing behavior, unchanged -- see the module-level note
    above for why this isn't False by default yet."""
    client = _client()

    await client.vector_search_chunks([0.1, 0.2], tenant="acme")

    assert client.run.await_args.kwargs["include_superseded"] is True


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
