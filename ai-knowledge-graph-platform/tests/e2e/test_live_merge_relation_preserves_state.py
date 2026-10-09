"""Live Neo4j proof that re-merging an edge keeps manual / lifecycle state."""

from __future__ import annotations

import uuid

import pytest

from graphrag.core.models import Relation, SourceType
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


async def _edge(client, tenant):
    rows = await client.run(
        "MATCH (:Entity {tenant:$t})-[r:RELATES_TO]->(:Entity) "
        "RETURN r.confidence_state AS state, r.source_type AS stype, "
        "toString(r.valid_to) AS valid_to",
        t=tenant,
    )
    assert len(rows) == 1
    return rows[0]


async def _merge(client, tenant, **kw):
    rel = Relation(source_entity_id="a", target_entity_id="b", relation="OWNS",
                   source_doc_id=str(uuid.uuid4()), **kw)
    await client.merge_relation(rel, "Acme", "ORG", "Bolt", "ORG", tenant=tenant)


class TestLiveMergeRelationPreservesState:
    async def test_retraction_manual_source_and_expiry_survive_remerge(self, neo4j_container):  # noqa: F811
        from neo4j import AsyncGraphDatabase

        tenant = f"keep-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD))
        client = _client_for(driver)
        try:
            await client.init_schema()
            for n in ("Acme", "Bolt"):
                await client.run(
                    "CREATE (:Entity {id:$id, name:$n, type:'ORG', tenant:$t})",
                    id=str(uuid.uuid4()), n=n, t=tenant)
            await _merge(client, tenant)
            assert (await _edge(client, tenant))["state"] == "ASSERTED"

            # Operator retracts, marks manual and expires the edge.
            await client.run(
                "MATCH (:Entity {tenant:$t})-[r:RELATES_TO]->() "
                "SET r.confidence_state='RETRACTED', r.source_type='manual', "
                "r.valid_to=datetime('2024-01-01T00:00:00Z')", t=tenant)

            # A later ingest of the same fact must not resurrect it.
            await _merge(client, tenant, source_type=SourceType.DOCUMENT,
                         valid_to=None)
            edge = await _edge(client, tenant)
            assert edge["state"] == "RETRACTED"
            assert edge["stype"] == "manual"
            assert edge["valid_to"].startswith("2024-01-01")
        finally:
            await driver.close()
