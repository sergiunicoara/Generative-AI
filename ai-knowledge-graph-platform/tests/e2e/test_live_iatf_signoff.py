"""Live Neo4j proof that concurrent IATF sign-offs never fork the hash chain, and that tampering is detected."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from graphrag.domains.automotive.signoff import SignoffStore
from tests.e2e.test_live_audit_followups import _driver_client
from tests.e2e.test_live_bitemporal_as_of_filtering import _docker_and_testcontainers_available
from tests.e2e.test_live_core_out_of_order_ingestion import neo4j_container  # noqa: F401

pytestmark = pytest.mark.skipif(
    not _docker_and_testcontainers_available(),
    reason="Docker or testcontainers-python not available",
)


async def test_concurrent_appends_form_one_unbroken_chain(neo4j_container):  # noqa: F811
    t = f"iatf-{uuid.uuid4().hex[:8]}"
    driver, client = await _driver_client(neo4j_container)
    try:
        store = SignoffStore(client, max_retries=50)
        results = await asyncio.gather(*[
            store.append(tenant=t, clause=f"8.{i % 3}", decision="accepted", reviewer=f"r{i}",
                         evidence_digest=f"d{i}")
            for i in range(12)
        ])
        assert sorted(r["seq"] for r in results) == list(range(1, 13))
        verdict = await store.verify(t)
        assert verdict == {"valid": True, "entries": 12, "head": verdict["head"]}
        other = await store.verify(t + "-other")
        assert other["entries"] == 0 and other["valid"] is True
    finally:
        await driver.close()


async def test_editing_a_stored_signoff_is_detected(neo4j_container):  # noqa: F811
    t = f"iatf-{uuid.uuid4().hex[:8]}"
    driver, client = await _driver_client(neo4j_container)
    try:
        store = SignoffStore(client)
        for i in range(3):
            await store.append(tenant=t, clause="10.2", decision="needs_more_evidence", reviewer="a",
                               evidence_digest=f"d{i}")
        await client.run("MATCH (s:IatfSignoff {tenant: $t, seq: 2}) SET s.decision = 'accepted'", t=t)
        verdict = await store.verify(t)
        assert verdict["valid"] is False and verdict["broken_at_seq"] == 2 and verdict["reason"] == "content changed"
        with pytest.raises(Exception):     # the (tenant, seq) constraint blocks a second entry with an existing seq
            await client.run("CREATE (:IatfSignoff {tenant: $t, seq: 1})", t=t)
    finally:
        await driver.close()
