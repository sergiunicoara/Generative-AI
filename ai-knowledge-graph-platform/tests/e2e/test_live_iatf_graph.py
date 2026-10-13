"""Live Neo4j proof for the graph-backed IATF evidence view and the opt-in superseded-description filter."""

from __future__ import annotations

import uuid

import pytest

from graphrag.domains.automotive import audit
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
        CREATE (new:Document {tenant:$t, id:$t+'-csr23', filename:'CSR-CLIENT-2023.txt', status:'done',
                              recorded_at: datetime('2026-06-13T08:00:00Z')})
        CREATE (old:Document {tenant:$t, id:$t+'-csr21', filename:'CSR-CLIENT-2021.txt', status:'done',
                              recorded_at: datetime('2026-06-12T08:00:00Z'), superseded_by:$t+'-csr23'})
        CREATE (gone:Document {tenant:$t, id:$t+'-ilnc', filename:'IL-NC-06-rev2.txt', status:'done',
                               is_deleted:true, recorded_at: datetime('2026-06-13T08:00:00Z')})
        CREATE (:Chunk {tenant:$t, id:$t+'-c1', chunk_index:1, text:'Cerinte specifice:\\nKPI livrare la timp 95 %'})
               -[:PART_OF]->(new)
        CREATE (:Chunk {tenant:$t, id:$t+'-c0', chunk_index:0, text:'Introducere'})-[:PART_OF]->(new)
        CREATE (oc:Chunk {tenant:$t, id:$t+'-o1', chunk_index:0, text:'KPI livrare la timp 98 %'})-[:PART_OF]->(old)
        CREATE (:Chunk {tenant:$t, id:$t+'-g1', chunk_index:0, text:'neconformitate'})-[:PART_OF]->(gone)
        CREATE (nc:Chunk {tenant:$t, id:$t+'-n1', chunk_index:0, text:'KPI curent'})-[:PART_OF]->(new)
        CREATE (k:Entity {tenant:$t, id:$t+'-k', name:'Livrare la timp', type:'KPI',
                          description:'tinta 98% (CSR 2021)', source_doc_id:$t+'-csr21'})
        CREATE (nc)-[:MENTIONS]->(k)
        CREATE (oc)-[:MENTIONS]->(k)
        """, t=t)


async def test_graph_corpus_feeds_the_same_view_with_chunk_locators_and_supersession(neo4j_container):  # noqa: F811
    t = f"iatfg-{uuid.uuid4().hex[:8]}"
    driver, client = await _driver_client(neo4j_container)
    try:
        await _seed(client, t)
        await _seed(client, t + "-other")
        docs = await audit.load_graph_corpus(client, t)
        by_id = {d.doc_id: d for d in docs}
        assert set(by_id) == {"CSR-CLIENT-2023", "CSR-CLIENT-2021"}            # deleted doc excluded, tenant scoped
        assert by_id["CSR-CLIENT-2021"].superseded_by == "CSR-CLIENT-2023" and not by_id["CSR-CLIENT-2021"].current
        assert by_id["CSR-CLIENT-2023"].lines[0] == "Introducere"              # chunks in chunk_index order
        clause = audit.Clause("8.4.2.4", "Supplier monitoring", ("CSR-CLIENT",), ("kpi",))
        item = audit.clause_evidence(clause, docs, max_age_days=10_000, now=by_id["CSR-CLIENT-2023"].recorded_at)
        assert [e["doc_id"] for e in item["evidence"]] == ["CSR-CLIENT-2023"]
        assert [e["doc_id"] for e in item["superseded_evidence"]] == ["CSR-CLIENT-2021"]
        assert "chunk 1, line 2" in [p["locator"] for p in item["evidence"][0]["passages"]]
    finally:
        await driver.close()


async def test_superseded_entity_description_is_hidden_only_when_asked(neo4j_container):  # noqa: F811
    t = f"iatfd-{uuid.uuid4().hex[:8]}"
    driver, client = await _driver_client(neo4j_container)
    try:
        await _seed(client, t)
        default = await client.get_entity_neighbors([t + "-n1"], tenant=t, include_superseded=False)
        assert default[0]["description"] == "tinta 98% (CSR 2021)" and default[0]["description_superseded"] is False
        hidden = await client.get_entity_neighbors([t + "-n1"], tenant=t, include_superseded=False,
                                                   hide_superseded_descriptions=True)
        assert hidden[0]["description"] == "" and hidden[0]["description_superseded"] is True
        await client.run("MATCH (d:Document {id: $id}) REMOVE d.superseded_by", id=t + "-csr21")
        current = await client.get_entity_neighbors([t + "-n1"], tenant=t, include_superseded=False,
                                                    hide_superseded_descriptions=True)
        assert current[0]["description"] == "tinta 98% (CSR 2021)"
    finally:
        await driver.close()
