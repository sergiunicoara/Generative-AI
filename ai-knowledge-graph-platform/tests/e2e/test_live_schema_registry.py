"""Live Neo4j proof of the schema-registry Cypher: activation, deactivation,
version change, rollback, idempotency, legacy rows and tenant isolation."""

from __future__ import annotations

import uuid

import pytest

from graphrag.graph.schema_registry import SchemaIdentity, SchemaRegistry, UnknownSchemaError
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


def ident(tenant: str, h: str, dataset: str | None = None, version="1.0.0") -> SchemaIdentity:
    return SchemaIdentity(tenant=tenant, dataset_id=dataset or tenant, name="onto", version=version,
                          content_hash=h * 64, profile_hash="p" * 64, types=("ORG",))


async def versions(client, tenant):
    return {r["h"]: r for r in await client.run(
        "MATCH (o:OntologyVersion {tenant:$t}) RETURN coalesce(o.content_hash,o.schema_hash) AS h, "
        "o.active AS active, o.id AS id, labels(o) AS labels", t=tenant)}


async def conforms(client, tenant, dataset=None):
    return [r["h"] for r in await client.run(
        "MATCH (:Dataset {tenant:$t, id:$d})-[:CONFORMS_TO]->(o:SchemaVersion) RETURN o.content_hash AS h",
        t=tenant, d=dataset or tenant)]


class TestLiveSchemaRegistry:
    async def test_lifecycle_activation_change_rollback_and_isolation(self, neo4j_container):  # noqa: F811
        from neo4j import AsyncGraphDatabase

        a, b = f"sa-{uuid.uuid4().hex[:8]}", f"sb-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD))
        client = _client_for(driver)
        try:
            await client.init_schema()
            reg = SchemaRegistry(client)

            # first registration: dataset, version, profile, edges; one active version
            r1 = await reg.register_and_activate(ident(a, "1"))
            assert r1["created"] and not r1["was_active"] and r1["prior_hashes"] == []
            vs = await versions(client, a)
            assert vs["1" * 64]["active"] is True and {"OntologyVersion", "SchemaVersion"} <= set(vs["1" * 64]["labels"])
            assert await conforms(client, a) == ["1" * 64]
            profile = await client.run(
                "MATCH (:SchemaVersion {tenant:$t})-[:USES_VALIDATION_PROFILE]->(p:ValidationProfile) "
                "RETURN p.tenant AS tenant", t=a)
            assert profile == [{"tenant": a}]

            # idempotent: same hash again changes nothing
            r1b = await reg.register_and_activate(ident(a, "1"))
            assert r1b["version_id"] == r1["version_id"] and not r1b["created"] and r1b["was_active"]
            assert len(await versions(client, a)) == 1 and await conforms(client, a) == ["1" * 64]

            # version change: new hash active, old retained but inactive, edge repointed
            r2 = await reg.register_and_activate(ident(a, "2", version="1.1.0"))
            assert r2["created"] and r2["prior_hashes"] == ["1" * 64]
            vs = await versions(client, a)
            assert vs["2" * 64]["active"] is True and vs["1" * 64]["active"] is False
            assert await conforms(client, a) == ["2" * 64]
            assert (await reg.check(ident(a, "1")))["status"] == "drift"
            assert (await reg.check(ident(a, "1")))["known"] is True
            assert (await reg.check(ident(a, "2")))["status"] == "match"
            assert (await reg.check(ident(a, "9")))["known"] is False

            # tenant isolation: same hash under another tenant is a separate history
            await reg.register_and_activate(ident(b, "2"))
            assert (await versions(client, b))["2" * 64]["id"] != vs["2" * 64]["id"]
            assert (await versions(client, a))["2" * 64]["active"] is True
            with pytest.raises(UnknownSchemaError):
                await reg.activate(b, b, vs["1" * 64]["id"])  # a's version id, asked for under b

            # rollback: previous version active again, current one retained
            await reg.rollback(a, a)
            vs = await versions(client, a)
            assert vs["1" * 64]["active"] is True and vs["2" * 64]["active"] is False
            assert await conforms(client, a) == ["1" * 64]
            assert (await versions(client, b))["2" * 64]["active"] is True  # untouched

            # files revert/advance to a known hash -> reactivation, not a duplicate
            r3 = await reg.register_and_activate(ident(a, "2", version="1.1.0"))
            assert not r3["created"] and r3["prior_hashes"] == ["1" * 64]
            assert len(await versions(client, a)) == 2

            # deactivate leaves the dataset with no active schema, history intact
            assert await reg.deactivate(a, a) == 1
            assert await reg.get_active(a, a) is None and await conforms(client, a) == []
            assert (await reg.check(ident(a, "2")))["status"] == "unregistered"
            assert len(await versions(client, a)) == 2

            # imports stay inside one tenant
            await reg.register_and_activate(ident(a, "3"))
            va = await versions(client, a)
            assert await reg.add_import(a, va["3" * 64]["id"], va["1" * 64]["id"]) is True
            assert await reg.add_import(b, va["3" * 64]["id"], va["1" * 64]["id"]) is False
        finally:
            await driver.close()

    async def test_legacy_version_rows_are_retained_and_deactivated(self, neo4j_container):  # noqa: F811
        from neo4j import AsyncGraphDatabase

        t = f"leg-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD))
        client = _client_for(driver)
        try:
            await client.init_schema()
            # pre-registry row: schema_hash only, never deactivated
            await client.run(
                "CREATE (:OntologyVersion {id:'legacy-1', schema_hash:'abcdef0123456789', tenant:$t, active:true})",
                t=t)
            res = await SchemaRegistry(client).register_and_activate(ident(t, "5"))
            assert res["prior_hashes"] == ["abcdef0123456789"]
            vs = await versions(client, t)
            assert vs["abcdef0123456789"]["active"] is False and vs["5" * 64]["active"] is True
            # proposals/events still attach to the (retained) legacy node by id
            assert (await client.run(
                "MATCH (o:OntologyVersion {id:'legacy-1', tenant:$t}) RETURN count(o) AS n", t=t))[0]["n"] == 1
        finally:
            await driver.close()
