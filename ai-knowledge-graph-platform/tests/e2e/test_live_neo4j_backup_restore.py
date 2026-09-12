"""Live Neo4j proof for the platform NDJSON backup and restore tool.

This test uses the production ``scripts/kg_backup.py`` code against an
isolated Neo4j container. It proves a backup, destructive tenant-scoped wipe,
restore, and Cypher re-query round trip. It is intentionally separate from the
Energy GraphDB test: the Energy POC owns an RDF graph and never writes Neo4j.
"""

from __future__ import annotations

import sys
import uuid
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import kg_backup  # noqa: E402


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

_PASSWORD = "backup-restore-e2e-password"


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


class TestLiveNeo4jBackupRestore:
    async def test_backup_wipe_restore_and_requery_tenant_graph(self, neo4j_container, tmp_path) -> None:
        from neo4j import AsyncGraphDatabase

        tenant = f"backup-restore-e2e-{uuid.uuid4().hex[:10]}"
        backup_path = tmp_path / "tenant-graph.ndjson"
        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD),
        )
        client = _client_for(driver)
        try:
            await client.run(
                """
                CREATE (source:Entity {name: "WT-01", type: "WIND_TURBINE", tenant: $tenant,
                  description: "Synthetic wind turbine", confidence: 0.98,
                  valid_from: "2026-01-01T00:00:00Z", valid_to: "2026-12-31T00:00:00Z",
                  wikidata_qid: "Q-test", quarantined: false})
                CREATE (target:Entity {name: "WO-9001", type: "WORK_ORDER", tenant: $tenant,
                  description: "Synthetic maintenance work order", confidence: 0.91,
                  quarantined: true})
                CREATE (source)-[:RELATES_TO {relation: "HAS_OPEN_WORK_ORDER", tenant: $tenant,
                  confidence: 0.96, source_doc_ids: ["SAP-WO-9001"],
                  valid_from: "2026-08-28T00:00:00Z", valid_to: "2026-12-31T00:00:00Z",
                  source_type: "synthetic_sap_export"}]->(target)
                CREATE (:Chunk {id: "chunk-WT-01", tenant: $tenant, text: "WT-01 has WO-9001",
                  document_id: "SAP-WO-9001", chunk_index: 0, redacted: false})
                """,
                tenant=tenant,
            )

            with patch("graphrag.graph.neo4j_client.get_neo4j", return_value=client):
                await kg_backup.do_backup(Namespace(tenant=tenant, output=str(backup_path)))

                await client.run("MATCH (node {tenant: $tenant}) DETACH DELETE node", tenant=tenant)
                assert await client.run("MATCH (node {tenant: $tenant}) RETURN count(node) AS count", tenant=tenant) == [
                    {"count": 0},
                ]

                await kg_backup.do_restore(Namespace(input=str(backup_path), tenant=""))

            restored = await client.run(
                """
                MATCH (source:Entity {name: "WT-01", type: "WIND_TURBINE", tenant: $tenant})
                MATCH (target:Entity {name: "WO-9001", type: "WORK_ORDER", tenant: $tenant})
                MATCH (source)-[relation:RELATES_TO {relation: "HAS_OPEN_WORK_ORDER", tenant: $tenant}]->(target)
                MATCH (chunk:Chunk {id: "chunk-WT-01", tenant: $tenant})
                RETURN source.valid_to AS source_valid_to, target.quarantined AS target_quarantined,
                       relation.valid_to AS relation_valid_to, relation.source_doc_ids AS source_doc_ids,
                       chunk.text AS chunk_text
                """,
                tenant=tenant,
            )
            assert restored == [{
                "source_valid_to": "2026-12-31T00:00:00Z",
                "target_quarantined": True,
                "relation_valid_to": "2026-12-31T00:00:00Z",
                "source_doc_ids": ["SAP-WO-9001"],
                "chunk_text": "WT-01 has WO-9001",
            }]
        finally:
            await client.run("MATCH (node {tenant: $tenant}) DETACH DELETE node", tenant=tenant)
            await driver.close()
