"""Live Neo4j proof for the publication-gate quarantine store and read sessions."""

from __future__ import annotations

import uuid

import pytest

from graphrag.core.models import Entity
from graphrag.graph.validation import validate_batch
from graphrag.graph.validation.quarantine_store import QuarantineRecordStore
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


class TestLivePublicationGate:
    async def test_quarantine_roundtrip_is_idempotent_and_tenant_scoped(self, neo4j_container):  # noqa: F811
        from neo4j import AsyncGraphDatabase

        tenant, other = f"q-{uuid.uuid4().hex[:8]}", f"q-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD))
        client = _client_for(driver)
        try:
            await client.init_schema()
            store = QuarantineRecordStore(client)
            rejected = validate_batch([Entity(name="", type="ORG", tenant=tenant)], [],
                                      tenant=tenant, source="e2e").rejected
            ids = await store.save(rejected, tenant=tenant, source="e2e", document_id="d1",
                                   document_key="f.txt")
            assert ids == await store.save(rejected, tenant=tenant, source="e2e", document_id="d1",
                                           document_key="f.txt")
            rows = await store.list(tenant=tenant)
            assert [r["id"] for r in rows] == ids
            assert rows[0]["rule_ids"] == ["ENT-REQ-001"]
            assert await store.list(tenant=other) == []
            assert await store.get(tenant=other, record_id=ids[0]) is None
            assert (await store.summary(tenant=tenant))["by_rule"] == {"ENT-REQ-001": 1}
            # No published entity was created.
            assert await client.run("MATCH (e:Entity {tenant:$t}) RETURN e", t=tenant) == []
        finally:
            await driver.close()

    async def test_run_read_refuses_writes_server_side(self, neo4j_container):  # noqa: F811
        from neo4j import AsyncGraphDatabase
        from neo4j.exceptions import ClientError

        driver = AsyncGraphDatabase.driver(neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD))
        client = _client_for(driver)
        try:
            with pytest.raises(ClientError):
                await client.run_read("CREATE (:ShouldNotExist)")
        finally:
            await driver.close()
