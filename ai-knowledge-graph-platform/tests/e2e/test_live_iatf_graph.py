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


async def test_graph_evidence_respects_document_acls(neo4j_container):  # noqa: F811
    from graphrag.enterprise.models import AccessContext

    t = f"iatfa-{uuid.uuid4().hex[:8]}"
    driver, client = await _driver_client(neo4j_container)
    try:
        await client.run(
            """
            CREATE (open:Document {tenant:$t, id:$t+'-o', filename:'CSR-CLIENT-2023.txt', acl_state:'known',
                                   access_mode:'tenant'})
            CREATE (qa:Document {tenant:$t, id:$t+'-q', filename:'RFA-REG-01-rev5.txt', acl_state:'known',
                                 access_mode:'restricted', allow_principals:['group:quality']})
            CREATE (deny:Document {tenant:$t, id:$t+'-d', filename:'PQ-07-rev3.txt', acl_state:'known',
                                   access_mode:'tenant', deny_principals:['user:eve']})
            CREATE (unknown:Document {tenant:$t, id:$t+'-u', filename:'IL-NC-06-rev2.txt'})
            """, t=t)

        async def visible(claims, enabled=True):
            docs = await audit.load_graph_corpus(client, t, access_context=AccessContext.from_claims(claims),
                                                 acl_enabled=enabled)
            return {d.doc_id for d in docs}

        everyone = {"CSR-CLIENT-2023", "RFA-REG-01-rev5", "PQ-07-rev3", "IL-NC-06-rev2"}
        assert await visible({"sub": "bob"}, enabled=False) == everyone
        assert await visible({"sub": "bob", "groups": []}) == {"CSR-CLIENT-2023", "PQ-07-rev3"}
        assert await visible({"sub": "ann", "groups": ["quality"]}) == {"CSR-CLIENT-2023", "PQ-07-rev3",
                                                                         "RFA-REG-01-rev5"}
        assert await visible({"sub": "ann"}) == {"CSR-CLIENT-2023", "PQ-07-rev3"}   # groups claim missing
        assert await visible({"sub": "eve", "groups": []}) == {"CSR-CLIENT-2023"}     # explicit deny wins
    finally:
        await driver.close()


async def test_effective_from_is_read_from_valid_from(neo4j_container):  # noqa: F811
    t = f"iatfe-{uuid.uuid4().hex[:8]}"
    driver, client = await _driver_client(neo4j_container)
    try:
        await client.run(
            "CREATE (:Document {tenant:$t, id:$t+'-x', filename:'CSR-CLIENT-2023.txt', "
            "recorded_at: datetime('2026-01-01T00:00:00Z'), valid_from: datetime('2026-03-01T00:00:00Z')})", t=t)
        doc = (await audit.load_graph_corpus(client, t, acl_enabled=False))[0]
        assert doc.effective_from.isoformat().startswith("2026-03-01")
        assert doc.recorded_at.isoformat().startswith("2026-01-01")
    finally:
        await driver.close()


async def test_description_provenance_follows_the_document_that_filled_it(neo4j_container):  # noqa: F811
    from graphrag.core.models import Entity

    t = f"iatfp-{uuid.uuid4().hex[:8]}"
    driver, client = await _driver_client(neo4j_container)
    try:
        await client.merge_entity(Entity(name="Livrare", type="KPI", description="", source_doc_id=t + "-a"),
                                  tenant=t)
        await client.merge_entity(Entity(name="Livrare", type="KPI", description="tinta 98%",
                                         source_doc_id=t + "-b"), tenant=t)
        await client.merge_entity(Entity(name="Livrare", type="KPI", description="tinta 95%",
                                         source_doc_id=t + "-c"), tenant=t)
        row = (await client.run("MATCH (e:Entity {tenant:$t, name:'Livrare'}) "
                                "RETURN e.source_doc_id AS src, e.description_doc_id AS dsrc, e.description AS d",
                                t=t))[0]
        assert row == {"src": t + "-a", "dsrc": t + "-b", "d": "tinta 98%"}
        await client.merge_entities_batch([Entity(name="Stoc", type="KPI", description="x", source_doc_id=t + "-z")],
                                          tenant=t)
        batch = (await client.run("MATCH (e:Entity {tenant:$t, name:'Stoc'}) RETURN e.description_doc_id AS d",
                                  t=t))[0]
        assert batch["d"] == t + "-z"
    finally:
        await driver.close()
