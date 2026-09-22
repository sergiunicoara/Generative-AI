"""Live Neo4j proof that core (non-Energy-domain) bitemporal filtering
handles genuinely out-of-order ingestion: a document describing an EARLIER
period, ingested LATER (real wall-clock), must not be visible at a
transaction-time checkpoint that predates its actual ingestion -- even
though its valid-time period covers the query's valid_at.

Distinct from test_live_bitemporal_as_of_filtering.py (static seeding, single
ingestion pass, RELATES_TO edges via raw Cypher CREATE) and from
tests/unit/test_energy_temporal_correctness.py (Energy-domain unit test on a
SQLite/rdflib stack that shares no code with this codebase's core Neo4j
path). This test drives the real core ingestion methods
(Neo4jClient.merge_document / merge_chunk) across two separate calls in real
time, to prove ingestion order cannot corrupt bitemporal correctness.

See docs/IMPLEMENTATION_AUDIT.md ("core live/integration temporal test").

Deliberately does NOT touch RELATES_TO edges: merge_relation/
merge_relations_batch are documented (docs/IMPLEMENTATION_AUDIT.md,
"Contradiction detection" row) as last-write-wins on same-edge re-merges --
a separate, already-tracked, out-of-scope gap. Using two independent
documents (different filenames, no merge-identity collision) avoids any
dependency on that unfixed path.
"""

from __future__ import annotations

import uuid

import pytest

from graphrag.core.models import Chunk
from tests.e2e.test_live_bitemporal_as_of_filtering import (
    _PASSWORD,
    _client_for,
    _docker_and_testcontainers_available,
)

pytestmark = pytest.mark.skipif(
    not _docker_and_testcontainers_available(),
    reason="Docker or testcontainers-python not available",
)


@pytest.fixture(scope="module")
def neo4j_container():
    from testcontainers.community.neo4j import Neo4jContainer
    from testcontainers.core.config import testcontainers_config

    previous_max_tries = testcontainers_config.max_tries
    testcontainers_config.max_tries = max(previous_max_tries, 300)
    try:
        with Neo4jContainer("neo4j:5.20-community", password=_PASSWORD) as container:
            yield container
    finally:
        testcontainers_config.max_tries = previous_max_tries


class TestLiveCoreOutOfOrderIngestion:
    async def test_late_arriving_earlier_period_document_not_retroactively_visible(
        self, neo4j_container,
    ) -> None:
        from neo4j import AsyncGraphDatabase

        tenant = f"ooo-{uuid.uuid4().hex[:8]}"
        marker = f"marker-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD),
        )
        client = _client_for(driver)
        try:
            await client.init_schema()

            # Document A: open-ended period starting June 2024, ingested FIRST.
            doc_a_id = await client.merge_document(
                doc_id=str(uuid.uuid4()), filename=f"doc-a-{tenant}.txt",
                ingested_at="2024-06-01T00:00:00Z",
                valid_from="2024-06-01T00:00:00Z", valid_to=None,
                tenant=tenant,
            )
            await client.merge_chunk(
                Chunk(document_id=doc_a_id, chunk_index=0,
                      text=f"Document A period fact {marker}-a", tenant=tenant),
                tenant=tenant,
            )

            checkpoint_rows = await client.run("RETURN toString(datetime()) AS now")
            checkpoint = checkpoint_rows[0]["now"]

            # Document B: an EARLIER period (Jan-Jun 2024), but arrives SECOND
            # in real wall-clock time -- the out-of-order, late-arriving
            # correction for a past period that this test exists to prove.
            doc_b_id = await client.merge_document(
                doc_id=str(uuid.uuid4()), filename=f"doc-b-{tenant}.txt",
                ingested_at="2024-08-01T00:00:00Z",
                valid_from="2024-01-01T00:00:00Z", valid_to="2024-06-01T00:00:00Z",
                tenant=tenant,
            )
            await client.merge_chunk(
                Chunk(document_id=doc_b_id, chunk_index=0,
                      text=f"Document B period fact {marker}-b", tenant=tenant),
                tenant=tenant,
            )
            await client.run("CALL db.awaitIndexes()")

            # 1. At the checkpoint (before B was ingested), March 2024 (inside
            #    B's period) must return nothing -- B wasn't known yet, and
            #    A's period doesn't start until June.
            before_b = await client.bm25_search_chunks(
                marker, tenant=tenant, valid_at="2024-03-01T00:00:00Z",
                transaction_at=checkpoint,
            )
            assert before_b == []

            # 2. Now (after both ingested), March 2024 resolves to B only.
            after_both_march = await client.bm25_search_chunks(
                marker, tenant=tenant, valid_at="2024-03-01T00:00:00Z",
            )
            assert [r["text"] for r in after_both_march] == [
                f"Document B period fact {marker}-b"
            ]

            # 3. July 2024 (outside B's Jan-Jun window, inside A's open-ended
            #    June+ window) resolves to A only, regardless of ingestion order.
            after_both_july = await client.bm25_search_chunks(
                marker, tenant=tenant, valid_at="2024-07-01T00:00:00Z",
            )
            assert [r["text"] for r in after_both_july] == [
                f"Document A period fact {marker}-a"
            ]
        finally:
            await client.run("MATCH (n {tenant: $tenant}) DETACH DELETE n", tenant=tenant)
            await driver.close()
