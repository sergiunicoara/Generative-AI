"""Live Neo4j proof for the audit follow-ups: bitemporal datetime comparison, multi-hop relation
allowlist, and superseded-document edge exclusion."""

from __future__ import annotations

import uuid

import pytest

from graphrag.graph.bitemporal import BitemporalStore
from tests.e2e.test_live_bitemporal_as_of_filtering import (
    _PASSWORD,
    _client_for,
    _docker_and_testcontainers_available,
)
from tests.e2e.test_live_core_out_of_order_ingestion import neo4j_container  # noqa: F401

pytestmark = pytest.mark.skipif(
    not _docker_and_testcontainers_available(),
    reason="Docker or testcontainers-python not available",
)


async def _driver_client(container):
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(container.get_connection_url(), auth=("neo4j", _PASSWORD))
    client = _client_for(driver)
    await client.init_schema()
    return driver, client


class TestLiveBitemporalStore:
    async def test_stored_datetimes_compare_with_string_parameters(self, neo4j_container):  # noqa: F811
        t = f"bt-{uuid.uuid4().hex[:8]}"
        driver, client = await _driver_client(neo4j_container)
        try:
            await client.run(
                "CREATE (:Entity {tenant:$t, id:$t+'a', name:'Dated', type:'ORG', "
                "valid_from: datetime('2020-01-01T00:00:00Z'), recorded_at: datetime('2020-01-02T00:00:00Z')})", t=t)
            store = BitemporalStore(client)
            hit = await store.as_of_entities("2021-01-01T00:00:00", "2021-01-01T00:00:00", tenant=t)
            assert [r["name"] for r in hit] == ["Dated"]
            before_valid = await store.as_of_entities("2019-01-01T00:00:00", "2021-01-01T00:00:00", tenant=t)
            assert before_valid == []
            before_recorded = await store.as_of_entities("2021-01-01T00:00:00", "2020-01-01T00:00:00", tenant=t)
            assert before_recorded == []
        finally:
            await driver.close()


class TestLiveMultihopControls:
    async def _seed(self, client, t):
        await client.run(
            """
            CREATE (a:Entity {tenant:$t, id:$t+'a', name:'Acme', type:'ORG'})
            CREATE (b:Entity {tenant:$t, id:$t+'b', name:'Bolt', type:'ORG'})
            CREATE (n:Entity {tenant:$t, id:$t+'n', name:'Nut', type:'ORG'})
            CREATE (seed:Chunk {tenant:$t, id:$t+'seed', text:'seed'})-[:MENTIONS]->(a)
            CREATE (cb:Chunk {tenant:$t, id:$t+'cb', text:'about bolt'})-[:MENTIONS]->(b)
            CREATE (cn:Chunk {tenant:$t, id:$t+'cn', text:'about nut'})-[:MENTIONS]->(n)
            CREATE (d0:Document {tenant:$t, id:$t+'d0', name:'seed'})
            CREATE (dold:Document {tenant:$t, id:$t+'dold', name:'old', superseded_by:'newer'})
            CREATE (dok:Document {tenant:$t, id:$t+'dok', name:'ok'})
            CREATE (seed)-[:PART_OF]->(d0)
            CREATE (cb)-[:PART_OF]->(dok)
            CREATE (cn)-[:PART_OF]->(dok)
            CREATE (a)-[:RELATES_TO {relation:'OWNS', tenant:$t, confidence:0.9, source_doc_id:$t+'dok'}]->(b)
            CREATE (a)-[:RELATES_TO {relation:'SUPPLIES', tenant:$t, confidence:0.9, source_doc_id:$t+'dold'}]->(n)
            """, t=t)

    async def test_default_traverses_everything(self, neo4j_container):  # noqa: F811
        t = f"mh-{uuid.uuid4().hex[:8]}"
        driver, client = await _driver_client(neo4j_container)
        try:
            await self._seed(client, t)
            rows = await client.get_multihop_chunks([t + "seed"], hops=1, tenant=t)
            assert {r["chunk_id"] for r in rows} == {t + "cb", t + "cn"}
        finally:
            await driver.close()

    async def test_relation_allowlist_restricts_traversal(self, neo4j_container):  # noqa: F811
        t = f"mh-{uuid.uuid4().hex[:8]}"
        driver, client = await _driver_client(neo4j_container)
        try:
            await self._seed(client, t)
            rows = await client.get_multihop_chunks([t + "seed"], hops=1, tenant=t, allowed_relations=["OWNS"])
            assert {r["chunk_id"] for r in rows} == {t + "cb"}
        finally:
            await driver.close()

    async def test_edges_from_superseded_documents_are_not_traversed(self, neo4j_container):  # noqa: F811
        t = f"mh-{uuid.uuid4().hex[:8]}"
        driver, client = await _driver_client(neo4j_container)
        try:
            await self._seed(client, t)
            rows = await client.get_multihop_chunks([t + "seed"], hops=1, tenant=t, include_superseded=False)
            assert {r["chunk_id"] for r in rows} == {t + "cb"}
            kept = await client.get_entity_neighbors([t + "seed"], tenant=t)
            assert set(kept[0]["neighbors"]) == {"Bolt", "Nut"}
            filtered = await client.get_entity_neighbors([t + "seed"], tenant=t, include_superseded=False)
            assert set(filtered[0]["neighbors"]) == {"Bolt"}
        finally:
            await driver.close()
