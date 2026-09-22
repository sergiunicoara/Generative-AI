"""Live Neo4j proof that tenant isolation actually holds against real data.

Most of the existing "tenant isolation" unit tests (tests/unit/test_tenant_isolation.py)
assert that a Cypher query string contains a `tenant: $tenant` filter against a
mocked Neo4j client -- they prove query *shape*, not that isolation actually
holds against a real graph. `scripts/verify_tenant_isolation.py` does real,
live checks (missing-tenant nodes, cross-tenant RELATES_TO edges, cross-tenant
PART_OF/MENTIONS/MEMBER_OF links) but was a manual ops script, never wired
into the automated test suite (see docs/IMPLEMENTATION_AUDIT.md).

This test runs that same check against a real, disposable Neo4j container:
one pass proving a clean two-tenant graph is reported clean, and one pass
per violation type proving the check actually *catches* a real cross-tenant
leak rather than passing vacuously.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import verify_tenant_isolation  # noqa: E402


def _docker_and_testcontainers_available() -> bool:
    try:
        import docker
        import testcontainers  # noqa: F401

        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_and_testcontainers_available(),
    reason="Docker or testcontainers-python not available",
)

_PASSWORD = "tenant-isolation-e2e-password"


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


def _client_for(driver):
    """Use Neo4jClient.run() without reading the developer's settings file."""
    from graphrag.graph.neo4j_client import Neo4jClient

    client = Neo4jClient.__new__(Neo4jClient)
    client._driver = driver
    client._filtered_vector_search = False
    client._filtered_vector_indexes = set()
    client._in_flight = 0
    return client


async def _verify_with_disposable_client(container, tenant_filter):
    """Run verify_tenant_isolation.verify() against its own connection.

    verify() calls `await neo4j.close()` internally (it's written to be
    invoked as a standalone script, once, then exit) -- so it must never be
    handed a driver the caller still needs afterward for cleanup.
    """
    from neo4j import AsyncGraphDatabase
    from unittest.mock import patch

    driver = AsyncGraphDatabase.driver(container.get_connection_url(), auth=("neo4j", _PASSWORD))
    client = _client_for(driver)
    with patch("graphrag.graph.neo4j_client.get_neo4j", return_value=client):
        return await verify_tenant_isolation.verify(tenant_filter)


async def _seed_two_clean_tenants(client, tenant_a: str, tenant_b: str) -> None:
    for tenant in (tenant_a, tenant_b):
        await client.run(
            """
            CREATE (e1:Entity {name: "Supplier-1", type: "SUPPLIER", tenant: $tenant,
              description: "d", confidence: 0.9, quarantined: false})
            CREATE (e2:Entity {name: "Standard-1", type: "STANDARD", tenant: $tenant,
              description: "d", confidence: 0.9, quarantined: false})
            CREATE (e1)-[:RELATES_TO {relation: "CERTIFIED_UNDER", tenant: $tenant,
              confidence: 0.9, source_doc_ids: ["doc-1"]}]->(e2)
            CREATE (d:Document {id: $doc_id, tenant: $tenant, filename: "f.txt"})
            CREATE (c:Chunk {id: $chunk_id, tenant: $tenant, text: "text",
              document_id: $doc_id, chunk_index: 0})
            CREATE (c)-[:PART_OF]->(d)
            CREATE (c)-[:MENTIONS]->(e1)
            """,
            tenant=tenant,
            doc_id=f"doc-{tenant}",
            chunk_id=f"chunk-{tenant}",
        )


class TestLiveTenantIsolation:
    async def test_two_clean_tenants_report_no_violations(self, neo4j_container) -> None:
        from neo4j import AsyncGraphDatabase

        tenant_a = f"iso-e2e-a-{uuid.uuid4().hex[:8]}"
        tenant_b = f"iso-e2e-b-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD),
        )
        client = _client_for(driver)
        try:
            await _seed_two_clean_tenants(client, tenant_a, tenant_b)

            exit_code = await _verify_with_disposable_client(neo4j_container, [tenant_a, tenant_b])

            assert exit_code == 0
        finally:
            for tenant in (tenant_a, tenant_b):
                await client.run("MATCH (n {tenant: $tenant}) DETACH DELETE n", tenant=tenant)
            await driver.close()

    async def test_cross_tenant_relates_to_edge_is_caught(self, neo4j_container) -> None:
        from neo4j import AsyncGraphDatabase

        tenant_a = f"iso-e2e-leak-a-{uuid.uuid4().hex[:8]}"
        tenant_b = f"iso-e2e-leak-b-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD),
        )
        client = _client_for(driver)
        try:
            await _seed_two_clean_tenants(client, tenant_a, tenant_b)
            # Deliberately create a leak: an edge whose endpoints belong to
            # different tenants -- exactly what tenant-scoped multi-hop
            # traversal must never be able to produce.
            await client.run(
                """
                MATCH (a:Entity {name: "Supplier-1", tenant: $tenant_a})
                MATCH (b:Entity {name: "Standard-1", tenant: $tenant_b})
                CREATE (a)-[:RELATES_TO {relation: "LEAKED", tenant: $tenant_a,
                  confidence: 0.9, source_doc_ids: ["doc-1"]}]->(b)
                """,
                tenant_a=tenant_a, tenant_b=tenant_b,
            )

            exit_code = await _verify_with_disposable_client(neo4j_container, [tenant_a, tenant_b])

            assert exit_code == 1
        finally:
            for tenant in (tenant_a, tenant_b):
                await client.run("MATCH (n {tenant: $tenant}) DETACH DELETE n", tenant=tenant)
            await driver.close()

    async def test_supersession_exclusion_holds_against_live_data(self, neo4j_container) -> None:
        """Prove neo4j_client.py's superseded_by exclusion (opt-in via
        include_superseded=False -- see that method's docstring for why it
        isn't the default yet) actually filters at the Cypher/index level,
        not just in the mocked query-shape unit tests
        (test_neo4j_retrieval_query_shape.py). Also proves the pre-existing
        default (include_superseded=True) genuinely includes both, i.e. this
        isn't a vacuous pass where nothing was ever excludable.
        """
        from neo4j import AsyncGraphDatabase

        tenant = f"iso-e2e-supersede-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD),
        )
        client = _client_for(driver)
        try:
            await client.init_schema()
            await client.run(
                """
                CREATE (old:Document {id: $old_id, tenant: $tenant, filename: "old.txt",
                  superseded_by: $new_id})
                CREATE (new:Document {id: $new_id, tenant: $tenant, filename: "new.txt"})
                CREATE (oc:Chunk {id: $old_chunk, tenant: $tenant,
                  text: "SupersessionMarkerXYZ retirement torque spec"})
                CREATE (nc:Chunk {id: $new_chunk, tenant: $tenant,
                  text: "SupersessionMarkerXYZ current torque spec"})
                CREATE (oc)-[:PART_OF]->(old)
                CREATE (nc)-[:PART_OF]->(new)
                """,
                tenant=tenant, old_id="doc-old", new_id="doc-new",
                old_chunk="chunk-old", new_chunk="chunk-new",
            )
            await client.run("CALL db.awaitIndexes()")

            default_rows = await client.bm25_search_chunks(
                "SupersessionMarkerXYZ", tenant=tenant,
            )
            excluded_rows = await client.bm25_search_chunks(
                "SupersessionMarkerXYZ", tenant=tenant, include_superseded=False,
            )

            default_ids = {r["chunk_id"] for r in default_rows}
            excluded_ids = {r["chunk_id"] for r in excluded_rows}
            assert default_ids == {"chunk-old", "chunk-new"}
            assert excluded_ids == {"chunk-new"}
        finally:
            await client.run("MATCH (n {tenant: $tenant}) DETACH DELETE n", tenant=tenant)
            await driver.close()

    async def test_node_missing_tenant_property_is_caught(self, neo4j_container) -> None:
        from neo4j import AsyncGraphDatabase

        tenant_a = f"iso-e2e-null-{uuid.uuid4().hex[:8]}"
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD),
        )
        client = _client_for(driver)
        try:
            await client.run(
                """
                CREATE (e1:Entity {name: "Supplier-1", type: "SUPPLIER", tenant: $tenant,
                  description: "d", confidence: 0.9, quarantined: false})
                """,
                tenant=tenant_a,
            )
            # Application code enforces tenant on every write path
            # (require_tenant in graphrag/core/tenancy.py); Neo4j Community
            # edition cannot enforce this at the schema level (no property
            # existence constraints without Enterprise -- confirmed directly
            # against this project's own Neo4j instance, see
            # docs/IMPLEMENTATION_AUDIT.md). This is the achievable
            # equivalent: a node that somehow got no tenant (a bug, a manual
            # Cypher mistake, a partial write) must still be detectable.
            await client.run(
                "CREATE (e2:Entity {name: \"Orphan\", type: \"SUPPLIER\", "
                "description: \"d\", confidence: 0.9, quarantined: false})"
            )

            exit_code = await _verify_with_disposable_client(neo4j_container, [tenant_a])

            assert exit_code == 1
        finally:
            await client.run("MATCH (n {tenant: $tenant}) DETACH DELETE n", tenant=tenant_a)
            await client.run("MATCH (n:Entity {name: \"Orphan\"}) DETACH DELETE n")
            await driver.close()


async def _seed_retrieval_fixture(client, tenant: str, marker: str) -> None:
    """Seed one document/chunk/entity/relation set with a unique marker string,
    so a cross-tenant leak in the real retrieval path (not just Cypher shape)
    is unambiguous in the results.
    """
    await client.run(
        f"""
        CREATE (d:Document {{id: $doc_id, tenant: $tenant, filename: $filename}})
        CREATE (c:Chunk {{id: $chunk_id, tenant: $tenant,
          text: "Retrieval isolation probe mentions " + $marker + " certification"}})
        CREATE (c)-[:PART_OF]->(d)
        CREATE (e1:Entity {{name: $marker + "-Supplier", type: "SUPPLIER", tenant: $tenant,
          description: "d", confidence: 0.9, quarantined: false}})
        CREATE (e2:Entity {{name: $marker + "-Standard", type: "STANDARD", tenant: $tenant,
          description: "d", confidence: 0.9, quarantined: false}})
        CREATE (c)-[:MENTIONS]->(e1)
        CREATE (hc:Chunk {{id: $hop_chunk_id, tenant: $tenant,
          text: $marker + " hop-reachable chunk"}})
        CREATE (hc)-[:MENTIONS]->(e2)
        CREATE (e1)-[:RELATES_TO {{relation: "CERTIFIED_UNDER", tenant: $tenant,
          confidence: 0.9, source_doc_ids: ["doc-1"]}}]->(e2)
        """,
        tenant=tenant, marker=marker,
        doc_id=f"doc-{marker}", filename=f"{marker}.txt",
        chunk_id=f"chunk-{marker}", hop_chunk_id=f"hop-{marker}",
    )


class TestLiveRetrievalPathTenantIsolation:
    """Proves a real LocalSearch.search() call cannot return another
    tenant's data. The existing tests above prove graph *state* is clean
    (via scripts/verify_tenant_isolation.py); this exercises the actual
    retrieval query path (BM25 + entity context + multi-hop) that a live
    query uses, per docs/IMPLEMENTATION_AUDIT.md item #3.

    Vector search, the reranker, and the GNN are disabled via
    config_overrides — none of the three needs an embedding/model, and with
    them off, LocalSearch.search() makes no external (OpenAI/HF) calls at
    all, so this runs fully offline against the disposable Neo4j container.
    """

    async def test_local_search_never_returns_other_tenants_data(self, neo4j_container) -> None:
        from unittest.mock import patch

        from neo4j import AsyncGraphDatabase

        tenant_a = f"iso-e2e-retrieval-a-{uuid.uuid4().hex[:8]}"
        tenant_b = f"iso-e2e-retrieval-b-{uuid.uuid4().hex[:8]}"
        marker_a, marker_b = "AlphaMarker", "BetaMarker"
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD),
        )
        client = _client_for(driver)
        try:
            await client.init_schema()
            await _seed_retrieval_fixture(client, tenant_a, marker_a)
            await _seed_retrieval_fixture(client, tenant_b, marker_b)
            await client.run("CALL db.awaitIndexes()")

            with (
                patch("graphrag.retrieval.local_search.get_neo4j", return_value=client),
                patch("graphrag.retrieval.bm25_search.get_neo4j", return_value=client),
            ):
                from graphrag.retrieval.local_search import LocalSearch

                local_search = LocalSearch()

            result = await local_search.search(
                "Retrieval isolation probe certification",
                tenant=tenant_a,
                config_overrides={
                    "vector_search_enabled": False,
                    "reranker_enabled": False,
                    "gnn_enabled": False,
                },
            )

            chunk_ids = {c["chunk_id"] for c in result["chunks"]}
            entity_names = {e.get("entity", "") for e in result["entities"]}
            referenced_entities = set(result["referenced_entities"])
            referenced_chunks = set(result["referenced_chunks"])

            for collected in (chunk_ids, entity_names, referenced_entities, referenced_chunks):
                for value in collected:
                    assert marker_b not in value, (
                        f"tenant {tenant_a}'s query returned tenant {tenant_b}'s "
                        f"data: {value!r}"
                    )

            # Belt-and-suspenders: directly confirm every returned chunk_id
            # resolves only to tenant_a in the graph, not tenant_b.
            if chunk_ids:
                leaked = await client.run(
                    "MATCH (c:Chunk) WHERE c.id IN $ids AND c.tenant <> $tenant "
                    "RETURN c.id AS chunk_id, c.tenant AS tenant",
                    ids=list(chunk_ids), tenant=tenant_a,
                )
                assert leaked == []
        finally:
            for tenant in (tenant_a, tenant_b):
                await client.run("MATCH (n {tenant: $tenant}) DETACH DELETE n", tenant=tenant)
            await driver.close()
