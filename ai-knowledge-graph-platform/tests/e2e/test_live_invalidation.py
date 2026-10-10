"""Live Neo4j proof of targeted invalidation: SUPERSEDES, corrected ownership,
expired facts, schema change, multi-hop inferred edges, idempotent repeats and
tenant isolation. Runs in CI (testcontainers); skipped without Docker."""

from __future__ import annotations

import uuid

import pytest

from graphrag.graph.inference_engine import ForwardChainingEngine, InferenceRule
from graphrag.graph.invalidation import EntityRef, EventKind, InvalidationEvent, RelationRef
from graphrag.graph.invalidation.expiry import sweep_expired
from graphrag.graph.invalidation.service import InvalidationService
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

RULES = [
    InferenceRule(name="owns_transitive", rule_type="transitivity", relation="OWNS", max_depth=3),
    InferenceRule(name="owns_controls", rule_type="inverse", relation="OWNS", derived_relation="OWNED_BY"),
]


async def seed(client, t: str, p: str = "") -> None:
    """Docs d1 (Acme OWNS Bolt), d2 (Bolt OWNS Nut), d9 (Zed OWNS Yak, unrelated).
    One chunk per doc mentioning its entities; a snapshot and a decision per chunk."""
    await client.run(
        """
        UNWIND $docs AS d
        CREATE (doc:Document {tenant: $t, id: d.id, filename: d.id + '.txt'})
        CREATE (c:Chunk {tenant: $t, id: d.chunk, document_id: d.id})-[:PART_OF]->(doc)
        WITH d, c
        UNWIND [d.src, d.tgt] AS name
        MERGE (e:Entity {tenant: $t, name: name, type: 'ORG'}) ON CREATE SET e.id = $t + ':' + name
        MERGE (c)-[:MENTIONS]->(e)
        WITH d
        MATCH (s:Entity {tenant: $t, name: d.src}), (o:Entity {tenant: $t, name: d.tgt})
        CREATE (s)-[:RELATES_TO {tenant: $t, relation: 'OWNS', source_doc_ids: [d.id],
                                 confidence: 0.9, source_type: 'document'}]->(o)
        """,
        t=t, docs=[
            {"id": f"{p}d1", "chunk": f"{p}c1", "src": "Acme", "tgt": "Bolt"},
            {"id": f"{p}d2", "chunk": f"{p}c2", "src": "Bolt", "tgt": "Nut"},
            {"id": f"{p}d9", "chunk": f"{p}c9", "src": "Zed", "tgt": "Yak"},
        ])
    await client.run(
        """
        UNWIND $items AS it
        CREATE (:CommunitySummarySnapshot {tenant: $t, id: 's-' + it.c, chunk_ids: [it.c],
                document_ids: [it.d], entity_ids: [], summary: 'x', transaction_to: null})
        CREATE (m:CGContextManifest {tenant: $t, id: 'm-' + it.c, chunk_ids: [it.c],
                document_ids: [it.d], ontology_version: 'o@1#aaa'})
        CREATE (run:CGAgentRun {tenant: $t, id: 'r-' + it.c})-[:USED_CONTEXT]->(m)
        CREATE (run)-[:PRODUCED_DECISION]->(:CGDecision {tenant: $t, id: 'dec-' + it.c})
        """,
        t=t, items=[{"c": f"{p}c1", "d": f"{p}d1"}, {"c": f"{p}c2", "d": f"{p}d2"},
                                  {"c": f"{p}c9", "d": f"{p}d9"}])
    await ForwardChainingEngine(client, rules=RULES).run(tenant=t, max_iterations=3)


async def states(client, t):
    rows = await client.run(
        "MATCH (a:DerivedArtifactState {tenant: $t}) RETURN a.kind AS kind, a.artifact_id AS id, a.state AS state",
        t=t)
    return {(r["kind"], r["id"]): r["state"] for r in rows}


class TestLiveInvalidation:
    async def test_end_to_end(self, neo4j_container):  # noqa: F811
        from neo4j import AsyncGraphDatabase

        t, other = f"inv-{uuid.uuid4().hex[:8]}", f"inv-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD))
        client = _client_for(driver)
        try:
            await client.init_schema()
            await seed(client, t)
            await seed(client, other, p="other-")   # Document / Chunk ids are globally unique
            inferred = await client.run(
                "MATCH (:Entity {tenant:$t, name:'Acme'})-[r:RELATES_TO {relation:'OWNS'}]->(:Entity {name:'Nut'}) "
                "RETURN r.premise_keys AS p, r.rule_version AS v", t=t)
            assert inferred and set(inferred[0]["p"]) == {"ORG:Acme|OWNS|ORG:Bolt", "ORG:Bolt|OWNS|ORG:Nut"}
            assert inferred[0]["v"]

            svc = InvalidationService(client, cache_getter=_no_cache)

            # corrected ownership: Acme no longer OWNS Bolt (retracted)
            await client.run(
                "MATCH (:Entity {tenant:$t, name:'Acme'})-[r:RELATES_TO {relation:'OWNS'}]->(:Entity {name:'Bolt'}) "
                "SET r.confidence_state = 'RETRACTED'", t=t)
            ev = InvalidationEvent(tenant=t, kind=EventKind.RELATION_CHANGED, cause="retract-1",
                                   relations=[RelationRef(src_name="Acme", src_type="ORG", relation="OWNS",
                                                          tgt_name="Bolt", tgt_type="ORG")])
            out = await svc.handle(ev)
            st = await states(client, t)
            assert st[("inferred_edge", "ORG:Acme|OWNS|ORG:Nut")] == "INSUFFICIENT_EVIDENCE"   # recomputed
            # multi-hop: Nut OWNED_BY Acme is derived from the retracted inferred edge
            assert ("inferred_edge", "ORG:Nut|OWNED_BY|ORG:Acme") in st
            assert ("decision", "dec-c1") in st and ("decision", "dec-c9") not in st
            assert ("community_snapshot", "s-c9") not in st
            retracted = await client.run(
                "MATCH (:Entity {tenant:$t, name:'Acme'})-[r:RELATES_TO {relation:'OWNS'}]->(:Entity {name:'Nut'}) "
                "RETURN r.confidence_state AS s", t=t)
            assert retracted == [{"s": "RETRACTED"}]   # kept, not deleted

            # idempotent repeat
            again = await svc.handle(ev)
            assert again["duplicate"] is True and again["event_id"] == out["event_id"]

            # tenant isolation: the other tenant's identical graph is untouched
            assert await states(client, other) == {}
            other_edge = await client.run(
                "MATCH (:Entity {tenant:$t, name:'Acme'})-[r:RELATES_TO {relation:'OWNS'}]->(:Entity {name:'Nut'}) "
                "RETURN r.confidence_state AS s", t=other)
            assert other_edge == [{"s": "INFERRED"}]

            # SUPERSEDES: d2 superseded -> its snapshot is closed (history kept)
            await client.run("MATCH (d:Document {tenant:$t, id:'d2'}) SET d.superseded_by = 'd2b'", t=t)
            await svc.handle(InvalidationEvent(tenant=t, kind=EventKind.SUPERSEDED, document_ids=["d2"], cause="d2b"))
            snap = await client.run(
                "MATCH (s:CommunitySummarySnapshot {tenant:$t, id:'s-c2'}) RETURN s.transaction_to IS NOT NULL AS closed, "
                "s.summary AS summary", t=t)
            assert snap == [{"closed": True, "summary": "x"}]

            # expiry: d9 expires -> its dependents only
            await client.run("MATCH (d:Document {tenant:$t, id:'d9'}) SET d.valid_to = datetime('2020-01-01T00:00:00Z')", t=t)
            first = await sweep_expired(client, t)
            assert first["documents"] == 1
            assert ("decision", "dec-c9") in await states(client, t)
            assert (await sweep_expired(client, t))["event"] is None   # processed once per valid_to

            # schema change: decisions recorded under another version need review
            await svc.handle(InvalidationEvent(tenant=t, kind=EventKind.SCHEMA_CHANGED, schema_version="o@2#bbb",
                                               cause="h2"))
            st = await states(client, t)
            assert {k for k in st if k[0] == "decision"} == {("decision", "dec-c1"), ("decision", "dec-c2"),
                                                             ("decision", "dec-c9")}

            # quarantined entity reaches premises through it
            await svc.handle(InvalidationEvent(tenant=t, kind=EventKind.FACT_CORRECTED, cause="q",
                                               entities=[EntityRef(name="Zed", type="ORG")]))
            assert ("community_snapshot", "s-c9") in await states(client, t)
        finally:
            await driver.close()


async def _no_cache():
    return None
