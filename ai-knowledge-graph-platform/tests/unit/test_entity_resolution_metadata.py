"""Cypher-shape guards for entity-resolution audit trail (docs/IMPLEMENTATION_AUDIT.md #8).

resolution_status/resolution_method must land in ON CREATE SET only, never
ON MATCH SET -- a later re-ingestion of the same (name, type, tenant) must
not overwrite the original creation-time resolution record.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from graphrag.core.models import Entity
from graphrag.graph.neo4j_client import Neo4jClient


def _client() -> Neo4jClient:
    client = Neo4jClient.__new__(Neo4jClient)
    client.run = AsyncMock(return_value=[])
    return client


def _entity(**overrides) -> Entity:
    defaults = dict(name="SpaceX", type="ORG", tenant="acme")
    defaults.update(overrides)
    return Entity(**defaults)


async def test_merge_entity_sets_resolution_fields_on_create_only() -> None:
    client = _client()

    await client.merge_entity(
        _entity(resolution_status="created_new", resolution_method="new"), tenant="acme",
    )

    cypher = client.run.await_args.args[0]
    on_create, on_match = cypher.split("ON MATCH SET")
    assert "e.resolution_status = $resolution_status" in on_create
    assert "e.resolution_method = $resolution_method" in on_create
    assert "resolution_status" not in on_match
    assert "resolution_method" not in on_match
    assert client.run.await_args.kwargs["resolution_status"] == "created_new"
    assert client.run.await_args.kwargs["resolution_method"] == "new"


async def test_merge_entities_batch_sets_resolution_fields_on_create_only() -> None:
    client = _client()

    await client.merge_entities_batch(
        [_entity(resolution_status="needs_review", resolution_method="fuzzy")], tenant="acme",
    )

    cypher = client.run.await_args.args[0]
    on_create, on_match = cypher.split("ON MATCH SET")
    assert "e.resolution_status = row.resolution_status" in on_create
    assert "e.resolution_method = row.resolution_method" in on_create
    assert "resolution_status" not in on_match
    assert "resolution_method" not in on_match
    rows = client.run.await_args.kwargs["rows"]
    assert rows[0]["resolution_status"] == "needs_review"
    assert rows[0]["resolution_method"] == "fuzzy"


async def test_merge_entities_batch_semantic_properties_cannot_override_resolution_fields() -> None:
    client = _client()

    entity = _entity(
        resolution_status="created_new", resolution_method="new",
        semantic_properties={"resolution_status": "smuggled", "real_field": "kept"},
    )
    await client.merge_entities_batch([entity], tenant="acme")

    rows = client.run.await_args.kwargs["rows"]
    assert "resolution_status" not in rows[0]["semantic_properties"]
    assert rows[0]["semantic_properties"] == {"real_field": "kept"}


async def test_set_entity_resolution_metadata_query_shape() -> None:
    client = _client()

    await client.set_entity_resolution_metadata(
        name="SpaceX", type="ORG", tenant="acme",
        resolution_status="auto_resolved", resolution_method="embedding",
        resolution_score=0.96,
    )

    cypher = client.run.await_args.args[0]
    assert "MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})" in cypher
    assert "e.resolution_status" in cypher
    assert "e.resolution_method" in cypher
    assert "e.resolution_score" in cypher
    kwargs = client.run.await_args.kwargs
    assert kwargs == {
        "name": "SpaceX", "type": "ORG", "tenant": "acme",
        "resolution_status": "auto_resolved",
        "resolution_method": "embedding",
        "resolution_score": 0.96,
    }
