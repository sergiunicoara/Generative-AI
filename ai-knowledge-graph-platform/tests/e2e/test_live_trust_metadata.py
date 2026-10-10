"""Live Neo4j proof that retrieval reads exclude retracted/expired edges and
return trust metadata, and that writes record origin without verifying."""

from __future__ import annotations

import uuid

import pytest

from graphrag.core.models import Relation
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


class TestLiveTrustMetadata:
    async def test_current_edges_only_and_trust_fields(self, neo4j_container):  # noqa: F811
        from neo4j import AsyncGraphDatabase

        t = f"trust-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD))
        client = _client_for(driver)
        try:
            await client.init_schema()
            for n in ("Acme", "Bolt", "Nut", "Old"):
                await client.run("CREATE (:Entity {tenant:$t, id:$t+$n, name:$n, type:'ORG'})", t=t, n=n)
            await client.merge_relation(Relation(source_entity_id="a", target_entity_id="b", relation="OWNS",
                                                 extraction_model="m1"), "Acme", "ORG", "Bolt", "ORG", tenant=t)
            await client.merge_relation(Relation(source_entity_id="a", target_entity_id="c", relation="OWNS"),
                                        "Acme", "ORG", "Nut", "ORG", tenant=t)
            await client.merge_relation(Relation(source_entity_id="a", target_entity_id="d", relation="OWNS"),
                                        "Acme", "ORG", "Old", "ORG", tenant=t)
            await client.run("MATCH (:Entity {tenant:$t,name:'Acme'})-[r]->(:Entity {name:'Nut'}) "
                             "SET r.confidence_state='RETRACTED'", t=t)
            await client.run("MATCH (:Entity {tenant:$t,name:'Acme'})-[r]->(:Entity {name:'Old'}) "
                             "SET r.valid_to = datetime('2020-01-01T00:00:00Z')", t=t)

            rels = await client.get_relations_for_entity("Acme", "ORG", tenant=t)
            assert [r["name"] for r in rels] == ["Bolt"]           # retracted + expired excluded, no as_of
            bolt = rels[0]
            assert bolt["origin"] == "EXTRACTED" and bolt["verification_status"] == "UNVERIFIED"
            assert bolt["generated_by"] == "m1"

            sub = await client.get_entity_relations_subgraph(
                [{"name": n, "type": "ORG"} for n in ("Acme", "Bolt", "Nut", "Old")], tenant=t)
            assert {r["tgt"] for r in sub} == {"Bolt"}

            # historical question: the expired edge is valid as of 2019
            past = await client.get_relations_for_entity("Acme", "ORG", tenant=t, as_of="2019-06-01T00:00:00Z")
            assert {r["name"] for r in past} == {"Bolt", "Old"}

            # re-ingest never verifies; a reviewer approval does
            await client.merge_relation(Relation(source_entity_id="a", target_entity_id="b", relation="OWNS"),
                                        "Acme", "ORG", "Bolt", "ORG", tenant=t)
            from graphrag.graph.confidence_lifecycle import ConfidenceLifecycleService
            await ConfidenceLifecycleService(client).transition_relation(
                "Acme", "ORG", "OWNS", "Bolt", "ORG", "APPROVED", tenant=t, changed_by="rev")
            row = (await client.get_relations_for_entity("Acme", "ORG", tenant=t))[0]
            assert row["verification_status"] == "VERIFIED" and row["verified_by"] == "rev"
        finally:
            await driver.close()
