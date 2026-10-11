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


class TestLiveEntityTrust:
    async def test_entities_get_origin_and_a_reviewer_verification_survives_remerge(self, neo4j_container):  # noqa: F811
        from graphrag.core.models import Entity

        t = f"et-{uuid.uuid4().hex[:8]}"
        driver, client = await _driver_client(neo4j_container)
        try:
            await client.merge_entities_batch([Entity(name="Acme", type="ORG", extraction_model="m1")], tenant=t)
            row = (await client.run("MATCH (e:Entity {tenant:$t, name:'Acme'}) RETURN e.origin AS o, "
                                    "e.verification_status AS v, e.generated_by AS g", t=t))[0]
            assert row == {"o": "EXTRACTED", "v": "UNVERIFIED", "g": "m1"}
            await client.run("MATCH (e:Entity {tenant:$t, name:'Acme'}) SET e.verification_status='VERIFIED'", t=t)
            await client.merge_entities_batch([Entity(name="Acme", type="ORG")], tenant=t)
            again = (await client.run("MATCH (e:Entity {tenant:$t, name:'Acme'}) RETURN e.verification_status AS v", t=t))[0]
            assert again["v"] == "VERIFIED"          # re-ingest never downgrades a reviewer's decision
        finally:
            await driver.close()


class TestLiveCommunityEvidenceFilters:
    async def _seed(self, client, t):
        await client.run(
            """
            CREATE (cur:Community {tenant:$t, id:$t+'cur', summary:'current', level:0, embedding:$emb})
            CREATE (old:Community {tenant:$t, id:$t+'old', summary:'superseded', level:0, embedding:$emb})
            CREATE (q:Community {tenant:$t, id:$t+'q', summary:'quarantined', level:0, embedding:$emb})
            CREATE (e1:Entity {tenant:$t, id:$t+'e1', name:'A', type:'ORG'})-[:MEMBER_OF]->(cur)
            CREATE (e2:Entity {tenant:$t, id:$t+'e2', name:'B', type:'ORG'})-[:MEMBER_OF]->(old)
            CREATE (e3:Entity {tenant:$t, id:$t+'e3', name:'C', type:'ORG', quarantined:true})-[:MEMBER_OF]->(q)
            CREATE (d1:Document {tenant:$t, id:$t+'d1'}), (d2:Document {tenant:$t, id:$t+'d2', superseded_by:'x'})
            CREATE (c1:Chunk {tenant:$t, id:$t+'c1'})-[:PART_OF]->(d1), (c1)-[:MENTIONS]->(e1)
            CREATE (c2:Chunk {tenant:$t, id:$t+'c2'})-[:PART_OF]->(d2), (c2)-[:MENTIONS]->(e2)
            CREATE (c3:Chunk {tenant:$t, id:$t+'c3'})-[:PART_OF]->(d1), (c3)-[:MENTIONS]->(e3)
            """, t=t, emb=[0.1] * 3072)

    async def test_filters(self, neo4j_container):  # noqa: F811
        t = f"cm-{uuid.uuid4().hex[:8]}"
        driver, client = await _driver_client(neo4j_container)
        try:
            await self._seed(client, t)
            emb = [0.1] * 3072
            default = await client.vector_search_communities(emb, top_k=10, tenant=t)
            ids = {r["community_id"] for r in default}
            assert t + "q" not in ids                                  # all members quarantined: always dropped
            assert {t + "cur", t + "old"} <= ids                       # supersession filter is opt-in
            current = await client.vector_search_communities(emb, top_k=10, tenant=t, include_superseded=False)
            assert {r["community_id"] for r in current} == {t + "cur"}
        finally:
            await driver.close()


class TestLiveCommunitySnapshotFilters:
    async def test_snapshot_branch_filters_by_evidence_documents_and_members(self, neo4j_container):  # noqa: F811
        t = f"cs-{uuid.uuid4().hex[:8]}"
        driver, client = await _driver_client(neo4j_container)
        try:
            await client.run(
                """
                CREATE (:Document {tenant:$t, id:$t+'ok'}), (:Document {tenant:$t, id:$t+'old', superseded_by:'x'})
                CREATE (:Entity {tenant:$t, id:$t+'live', name:'L', type:'ORG'})
                CREATE (:Entity {tenant:$t, id:$t+'bad', name:'B', type:'ORG', quarantined:true})
                CREATE (:CommunitySummarySnapshot {tenant:$t, id:$t+'s-ok', community_id:'k1', summary:'ok', level:0,
                        embedding:$emb, document_ids:[$t+'ok'], entity_ids:[$t+'live']})
                CREATE (:CommunitySummarySnapshot {tenant:$t, id:$t+'s-old', community_id:'k2', summary:'old', level:0,
                        embedding:$emb, document_ids:[$t+'old'], entity_ids:[$t+'live']})
                CREATE (:CommunitySummarySnapshot {tenant:$t, id:$t+'s-q', community_id:'k3', summary:'q', level:0,
                        embedding:$emb, document_ids:[$t+'ok'], entity_ids:[$t+'bad']})
                """, t=t, emb=[0.1] * 3072)
            emb = [0.1] * 3072
            default = await client.vector_search_communities(emb, top_k=10, tenant=t, valid_at="2030-01-01T00:00:00Z")
            assert {r["community_id"] for r in default} == {"k1", "k2"}       # quarantined-only dropped
            current = await client.vector_search_communities(emb, top_k=10, tenant=t, valid_at="2030-01-01T00:00:00Z",
                                                             include_superseded=False)
            assert {r["community_id"] for r in current} == {"k1"}
        finally:
            await driver.close()
