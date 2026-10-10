"""Live Neo4j proof that per-document relation provenance is kept per document."""

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


async def _prov(client, tenant):
    rows = await client.run(
        "MATCH (:Entity {tenant:$t})-[r:RELATES_TO]->(:Entity) "
        "RETURN r.source_doc_ids AS docs, r.doc_chunk_ids AS chunks, "
        "r.doc_extraction_models AS models, r.doc_observed_at AS seen", t=tenant)
    assert len(rows) == 1
    return rows[0]


async def _merge(client, tenant, doc, chunk, model):
    rel = Relation(source_entity_id="a", target_entity_id="b", relation="OWNS", source_doc_id=doc,
                   source_chunk_id=chunk, extraction_model=model)
    await client.merge_relation(rel, "Acme", "ORG", "Bolt", "ORG", tenant=tenant)


class TestLiveRelationProvenance:
    async def test_each_document_keeps_its_own_chunk_and_model(self, neo4j_container):  # noqa: F811
        from neo4j import AsyncGraphDatabase

        tenant = f"prov-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD))
        client = _client_for(driver)
        try:
            await client.init_schema()
            for n in ("Acme", "Bolt"):
                await client.run("CREATE (:Entity {id:$id, name:$n, type:'ORG', tenant:$t})",
                                 id=str(uuid.uuid4()), n=n, t=tenant)
            await _merge(client, tenant, "d1", "c1", "model-a")
            await _merge(client, tenant, "d2", "c2", "model-b")
            p = await _prov(client, tenant)
            assert p["docs"] == ["d1", "d2"]
            assert p["chunks"] == ["c1", "c2"] and p["models"] == ["model-a", "model-b"]
            assert len(p["seen"]) == 2

            # Re-ingesting d1 replaces only d1's slot.
            await _merge(client, tenant, "d1", "c9", "model-c")
            p = await _prov(client, tenant)
            assert p["docs"] == ["d1", "d2"]
            assert p["chunks"] == ["c9", "c2"] and p["models"] == ["model-c", "model-b"]

            # Reconciling d1 removes its slot and keeps d2's, still aligned.
            await client.reconcile_document_evidence("d1", tenant=tenant)
            p = await _prov(client, tenant)
            assert p["docs"] == ["d2"] and p["chunks"] == ["c2"] and p["models"] == ["model-b"]
        finally:
            await driver.close()
