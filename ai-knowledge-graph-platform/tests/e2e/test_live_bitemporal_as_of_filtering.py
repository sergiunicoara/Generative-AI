"""Live Neo4j proof that `as_of` bitemporal filtering actually filters.

tests/unit/test_neo4j_retrieval_query_shape.py only asserts that the
interpolated Cypher *string* contains `r.valid_from` / `r.valid_to` -- it
never executes the query, so it couldn't catch that four methods in
neo4j_client.py compared a stored `datetime` property directly against the
raw `$as_of` string parameter instead of `datetime($as_of)`. Per Cypher's
comparison semantics, `datetime <= string` (or `>`) evaluates to `null`
(neither true nor false) rather than raising, which silently excluded every
`RELATES_TO` edge whenever `as_of` was passed to `get_entity_neighbors`,
`get_multihop_chunks`, `get_entity_relations_subgraph`, or
`get_relations_for_entity` -- regardless of whether the edge was actually
valid at that timestamp. See tasks/lessons.md (entry following A175) for the
fix and docs/IMPLEMENTATION_AUDIT.md item #8.

This test seeds one edge that's expired before `as_of` and one that's still
valid at `as_of`, then proves `get_entity_neighbors` and
`get_multihop_chunks` return the still-valid neighbor and exclude the
expired one -- which the pre-fix code could not do (it excluded both).
"""

from __future__ import annotations

import uuid

import pytest

from graphrag.graph.neo4j_client import Neo4jClient


def _docker_and_testcontainers_available() -> bool:
    try:
        import docker
        import testcontainers  # noqa: F401

        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_and_testcontainers_available(),
    reason="Docker or testcontainers-python not available",
)

_PASSWORD = "bitemporal-as-of-e2e-password"


@pytest.fixture(scope="module")
def neo4j_container():
    from testcontainers.community.neo4j import Neo4jContainer
    from testcontainers.core.config import testcontainers_config

    previous_max_tries = testcontainers_config.max_tries
    testcontainers_config.max_tries = max(previous_max_tries, 300)
    try:
        with Neo4jContainer("neo4j:5.20-community", password=_PASSWORD) as container:
            yield container
    finally:
        testcontainers_config.max_tries = previous_max_tries


def _client_for(driver) -> Neo4jClient:
    client = Neo4jClient.__new__(Neo4jClient)
    client._driver = driver
    client._filtered_vector_search = False
    client._filtered_vector_indexes = set()
    client._in_flight = 0
    return client


async def _seed_expiring_and_current_edges(client: Neo4jClient, tenant: str) -> None:
    """One entity with two RELATES_TO edges: to `Expired` (valid only
    2020-01-01..2021-01-01) and to `Current` (valid from 2020-01-01,
    open-ended). Querying with as_of="2025-06-01" must return only Current.
    """
    await client.run(
        """
        CREATE (c:Chunk {id: $chunk_id, tenant: $tenant, text: "probe"})
        CREATE (anchor:Entity {name: "Anchor", type: "ENTITY", tenant: $tenant,
          description: "d", confidence: 0.9, quarantined: false})
        CREATE (expired:Entity {name: "Expired", type: "ENTITY", tenant: $tenant,
          description: "d", confidence: 0.9, quarantined: false})
        CREATE (current:Entity {name: "Current", type: "ENTITY", tenant: $tenant,
          description: "d", confidence: 0.9, quarantined: false})
        CREATE (c)-[:MENTIONS]->(anchor)
        CREATE (anchor)-[:RELATES_TO {relation: "LINKED", tenant: $tenant,
          confidence: 0.9, valid_from: datetime("2020-01-01T00:00:00Z"),
          valid_to: datetime("2021-01-01T00:00:00Z")}]->(expired)
        CREATE (anchor)-[:RELATES_TO {relation: "LINKED", tenant: $tenant,
          confidence: 0.9, valid_from: datetime("2020-01-01T00:00:00Z")}]->(current)
        """,
        tenant=tenant, chunk_id=f"chunk-{tenant}",
    )


class TestLiveBitemporalAsOfFiltering:
    async def test_get_entity_neighbors_filters_expired_edge_by_as_of(self, neo4j_container) -> None:
        from neo4j import AsyncGraphDatabase

        tenant = f"as-of-neighbors-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD),
        )
        client = _client_for(driver)
        try:
            await _seed_expiring_and_current_edges(client, tenant)

            rows = await client.get_entity_neighbors(
                [f"chunk-{tenant}"], as_of="2025-06-01T00:00:00Z", tenant=tenant,
            )

            anchor_row = next(r for r in rows if r["entity"] == "Anchor")
            assert anchor_row["neighbors"] == ["Current"]
        finally:
            await client.run("MATCH (n {tenant: $tenant}) DETACH DELETE n", tenant=tenant)
            await driver.close()

    async def test_get_entity_neighbors_includes_expired_edge_within_its_window(self, neo4j_container) -> None:
        """Same fixture, as_of moved inside the expired edge's own validity
        window -- proves the filter genuinely evaluates the timestamp rather
        than e.g. always-excluding everything with an as_of set at all.
        """
        from neo4j import AsyncGraphDatabase

        tenant = f"as-of-neighbors-inwindow-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD),
        )
        client = _client_for(driver)
        try:
            await _seed_expiring_and_current_edges(client, tenant)

            rows = await client.get_entity_neighbors(
                [f"chunk-{tenant}"], as_of="2020-06-01T00:00:00Z", tenant=tenant,
            )

            anchor_row = next(r for r in rows if r["entity"] == "Anchor")
            assert set(anchor_row["neighbors"]) == {"Expired", "Current"}
        finally:
            await client.run("MATCH (n {tenant: $tenant}) DETACH DELETE n", tenant=tenant)
            await driver.close()

    async def test_get_multihop_chunks_filters_expired_edge_by_as_of(self, neo4j_container) -> None:
        from neo4j import AsyncGraphDatabase

        tenant = f"as-of-multihop-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD),
        )
        client = _client_for(driver)
        try:
            await _seed_expiring_and_current_edges(client, tenant)
            await client.run(
                """
                MATCH (expired:Entity {name: "Expired", tenant: $tenant})
                MATCH (current:Entity {name: "Current", tenant: $tenant})
                CREATE (ec:Chunk {id: $expired_chunk, tenant: $tenant, text: "expired hop"})
                CREATE (cc:Chunk {id: $current_chunk, tenant: $tenant, text: "current hop"})
                CREATE (ec)-[:MENTIONS]->(expired)
                CREATE (cc)-[:MENTIONS]->(current)
                """,
                tenant=tenant,
                expired_chunk=f"expired-hop-{tenant}",
                current_chunk=f"current-hop-{tenant}",
            )

            rows = await client.get_multihop_chunks(
                [f"chunk-{tenant}"], as_of="2025-06-01T00:00:00Z", tenant=tenant,
            )

            hop_chunk_ids = {r["chunk_id"] for r in rows}
            assert hop_chunk_ids == {f"current-hop-{tenant}"}
        finally:
            await client.run("MATCH (n {tenant: $tenant}) DETACH DELETE n", tenant=tenant)
            await driver.close()
