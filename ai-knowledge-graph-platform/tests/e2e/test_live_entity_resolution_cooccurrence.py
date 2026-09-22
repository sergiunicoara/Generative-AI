"""Live Neo4j proof that AliasRegistry's co-occurrence tie-break
(docs/IMPLEMENTATION_AUDIT.md "contextual disambiguation") actually reads
real graph structure, not just query-shape.

The tie-break is new Cypher (AliasRegistry._cooccurrence_tiebreak) added
this session, config-gated and disabled by default. Per this session's own
A174-derived rule -- a query-shape/mocked unit test proves the string is
well-formed, never that it filters or ranks anything correctly when
executed -- this exercises it against a real Neo4j instance.

Seeds two embedding-adjacent SUPPLIER candidates (identical embeddings, so
their ANN scores tie exactly) that both lexically corroborate the raw
mention. Only one of the two shares a RELATES_TO edge with a third,
"co-mentioned" entity. With the tie-break enabled, the graph-connected
candidate must win even though embedding score alone cannot distinguish them.
"""

from __future__ import annotations

import uuid

import pytest

from graphrag.graph.alias_registry import AliasRegistry
from tests.e2e.test_live_bitemporal_as_of_filtering import (
    _PASSWORD,
    _client_for,
    _docker_and_testcontainers_available,
)

pytestmark = pytest.mark.skipif(
    not _docker_and_testcontainers_available(),
    reason="Docker or testcontainers-python not available",
)

_EMBEDDING_DIM = 3072


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


class TestLiveEntityResolutionCooccurrence:
    async def test_cooccurrence_tiebreak_favors_graph_connected_candidate(
        self, neo4j_container,
    ) -> None:
        from neo4j import AsyncGraphDatabase

        tenant = f"cooc-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD),
        )
        client = _client_for(driver)
        try:
            await client.init_schema()

            # Identical embeddings -> the ANN search ties exactly (score
            # diff 0.0), so embedding score alone cannot break the tie.
            shared_embedding = [0.1] * _EMBEDDING_DIM
            connected_name = "PlastiAuto SA"
            unconnected_name = "PlastiAuto S.R.L."
            co_mentioned_name = "Bosch Romania"

            await client.run(
                """
                CREATE (connected:Entity {name: $connected_name, type: "SUPPLIER",
                  tenant: $tenant, embedding: $embedding, description: "d",
                  confidence: 0.9, quarantined: false})
                CREATE (unconnected:Entity {name: $unconnected_name, type: "SUPPLIER",
                  tenant: $tenant, embedding: $embedding, description: "d",
                  confidence: 0.9, quarantined: false})
                CREATE (co:Entity {name: $co_mentioned_name, type: "SUPPLIER",
                  tenant: $tenant, embedding: $embedding, description: "d",
                  confidence: 0.9, quarantined: false})
                CREATE (connected)-[:RELATES_TO {relation: "SUPPLIES_TO", tenant: $tenant,
                  confidence: 0.9}]->(co)
                """,
                connected_name=connected_name, unconnected_name=unconnected_name,
                co_mentioned_name=co_mentioned_name, tenant=tenant,
                embedding=shared_embedding,
            )
            await client.run("CALL db.awaitIndexes()")

            registry = AliasRegistry(client, tenant=tenant)
            registry._cooccurrence_tiebreak_enabled = True
            registry._embedding_threshold = 0.5  # identical vectors score 1.0, well above

            result = await registry.find_duplicate_by_embedding(
                embedding=shared_embedding,
                entity_type="SUPPLIER",
                exclude_name="PlastiAuto SRL",  # lexically close to both candidates
                with_detail=True,
                co_mentioned_names=[co_mentioned_name],
            )

            assert result is not None
            assert result.name == connected_name
            assert result.runner_ups and result.runner_ups[0][0] == unconnected_name
        finally:
            await client.run("MATCH (n {tenant: $tenant}) DETACH DELETE n", tenant=tenant)
            await driver.close()
