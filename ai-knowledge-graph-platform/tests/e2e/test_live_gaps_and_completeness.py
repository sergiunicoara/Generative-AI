"""Live Neo4j proof for the ingestion-gap check and the completeness report (the Cypher is not exercised by unit tests)."""

from __future__ import annotations

import uuid

import pytest

from graphrag.enterprise.gaps import find_ingestion_gaps
from graphrag.graph.completeness import graph_completeness_report
from tests.e2e.test_live_audit_followups import _driver_client
from tests.e2e.test_live_bitemporal_as_of_filtering import _docker_and_testcontainers_available
from tests.e2e.test_live_core_out_of_order_ingestion import neo4j_container  # noqa: F401

pytestmark = pytest.mark.skipif(
    not _docker_and_testcontainers_available(),
    reason="Docker or testcontainers-python not available",
)


async def _seed(client, t):
    await client.run(
        """
        CREATE (ok:Document {tenant:$t, id:$t+'ok', source_id:'s', external_id:'ok', source_version:'v1', status:'done'})
        CREATE (old:Document {tenant:$t, id:$t+'old', source_id:'s', external_id:'changed', source_version:'v1', status:'done'})
        CREATE (half:Document {tenant:$t, id:$t+'half', source_id:'s', external_id:'half', source_version:'v1', status:'processing'})
        CREATE (gone:Document {tenant:$t, id:$t+'gone', source_id:'s', external_id:'gone', source_version:'v1', status:'done'})
        CREATE (back:Document {tenant:$t, id:$t+'back', source_id:'s', external_id:'back', source_version:'v1', status:'done', is_deleted:true})
        CREATE (other:Document {tenant:$t, id:$t+'other', source_id:'other-src', external_id:'new', source_version:'v1', status:'done'})
        CREATE (okc:Chunk {tenant:$t, id:$t+'okc', text:'t', embedding:[0.1,0.2]})-[:PART_OF]->(ok)
        CREATE (oldc:Chunk {tenant:$t, id:$t+'oldc', text:'t', embedding:[0.1]})-[:PART_OF]->(old)
        CREATE (gonec:Chunk {tenant:$t, id:$t+'gonec', text:'t'})-[:PART_OF]->(gone)
        CREATE (backc:Chunk {tenant:$t, id:$t+'backc', text:'t', embedding:[0.1]})-[:PART_OF]->(back)
        CREATE (a:Entity {tenant:$t, id:$t+'a', name:'A', type:'ORG', description:'d', embedding:[0.1]})
        CREATE (b:Entity {tenant:$t, id:$t+'b', name:'B', type:'ORG', description:'', quarantined:false})
        CREATE (q:Entity {tenant:$t, id:$t+'q', name:'Q', type:'ORG', quarantined:true})
        CREATE (okc)-[:MENTIONS]->(a)
        CREATE (okc)-[:MENTIONS]->(b)
        CREATE (a)-[:RELATES_TO {tenant:$t, relation:'SUPPLIES', confidence:0.9, source_doc_id:$t+'ok',
                                 valid_from: datetime('2020-01-01T00:00:00Z')}]->(b)
        CREATE (b)-[:RELATES_TO {tenant:$t, relation:'', valid_from: datetime('2020-01-01T00:00:00Z')}]->(a)
        """, t=t)


async def test_gaps_against_a_seeded_graph_are_scoped_to_source_and_tenant(neo4j_container):  # noqa: F811
    t = f"gap-{uuid.uuid4().hex[:8]}"
    driver, client = await _driver_client(neo4j_container)
    try:
        await _seed(client, t)
        await _seed(client, t + "-other-tenant")        # must not leak into the report
        inventory = {"ok": "v1", "changed": "v2", "half": "v1", "back": "v1", "new": "v1"}
        report = await find_ingestion_gaps(client, tenant=t, source_id="s", inventory=inventory)
        assert report["counts"] == {"missing": 1, "stale": 1, "restored_in_source": 1, "not_ingested": 1,
                                    "duplicate_in_graph": 0, "unverifiable_version": 0, "would_tombstone": 1}
        assert report["samples"]["missing"] == ["new"]          # 'new' exists only under another source_id
        assert report["samples"]["would_tombstone"] == ["gone"]
        assert report["graph_documents"] == 5 and report["in_sync"] == 1
    finally:
        await driver.close()


async def test_completeness_counts_match_the_seed_and_exclude_deleted_and_quarantined(neo4j_container):  # noqa: F811
    t = f"cmp-{uuid.uuid4().hex[:8]}"
    driver, client = await _driver_client(neo4j_container)
    try:
        await _seed(client, t)
        await _seed(client, t + "-other-tenant")
        report = await graph_completeness_report(client, t)
        m = report["measures"]
        assert report["sizes"] == {"documents": 5, "chunks": 3, "entities": 2, "relations": 2}
        assert m["documents_chunked"] == {"share": 0.6, "numerator": 3, "denominator": 5}
        assert m["documents_with_entities"]["numerator"] == 1
        assert m["documents_ingested"]["numerator"] == 4                    # 'half' is still processing
        assert m["chunks_embedded"] == {"share": 0.6667, "numerator": 2, "denominator": 3}
        assert m["entities_described"]["numerator"] == 1 and m["entities_embedded"]["numerator"] == 1
        assert m["entities_connected"]["numerator"] == 2 and m["entities_with_evidence"]["numerator"] == 2
        assert m["relations_named"]["numerator"] == 1 and m["relations_sourced"]["numerator"] == 1
        scoped = await graph_completeness_report(client, t, "s")
        assert scoped["sizes"]["documents"] == 4
    finally:
        await driver.close()
