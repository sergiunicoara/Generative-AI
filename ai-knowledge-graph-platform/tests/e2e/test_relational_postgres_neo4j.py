"""Docker-backed PostgreSQL-to-Neo4j relational ingestion verification.

This test is opt-in by infrastructure availability, not by environment secrets:
it starts isolated PostgreSQL and Neo4j containers, writes synthetic
sustainability records, and verifies the complete relational ingestion path.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest


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


def _asyncpg_url(url: str) -> str:
    return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1).replace(
        "postgresql://", "postgresql+asyncpg://", 1,
    )


RELATIONAL_NEO4J_PASSWORD = "relational-e2e-password"


@pytest.fixture(scope="module")
def postgres_container():
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="module")
def neo4j_container():
    from testcontainers.community.neo4j import Neo4jContainer
    from testcontainers.core.config import testcontainers_config

    previous_max_tries = testcontainers_config.max_tries
    testcontainers_config.max_tries = max(previous_max_tries, 300)
    try:
        with Neo4jContainer(
            "neo4j:5.20-community", password=RELATIONAL_NEO4J_PASSWORD
        ) as container:
            yield container
    finally:
        testcontainers_config.max_tries = previous_max_tries


class TestRelationalPostgresToNeo4j:
    async def test_postgres_import_persists_tenant_scoped_graph_and_provenance(
        self, postgres_container, neo4j_container,
    ) -> None:
        from neo4j import AsyncGraphDatabase
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        from graphrag.graph.controlled_query import execute_controlled_query
        from graphrag.graph.neo4j_client import Neo4jClient
        from graphrag.ingestion.graph_writer import GraphWriter
        from graphrag.ingestion.relational import (
            EntityTableMapping,
            PostgreSQLSourceConnector,
            RelationTableMapping,
            RelationalGraphIngestor,
            RelationalGraphMapping,
        )

        tenant = f"sustainability-e2e-{uuid.uuid4().hex[:10]}"
        database_url = _asyncpg_url(postgres_container.get_connection_url())
        engine = create_async_engine(database_url)
        async with engine.begin() as connection:
            for statement in [
                "CREATE TABLE suppliers (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT)",
                "CREATE TABLE materials (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT)",
                "CREATE TABLE emissions_records (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT)",
                "CREATE TABLE evidence (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT)",
                "CREATE TABLE supplies (supplier_id TEXT, material_id TEXT, confidence REAL)",
                "CREATE TABLE reported (supplier_id TEXT, emissions_id TEXT, confidence REAL)",
                "CREATE TABLE has_evidence (emissions_id TEXT, evidence_id TEXT, confidence REAL)",
            ]:
                await connection.execute(text(statement))
            await connection.execute(text("INSERT INTO suppliers VALUES "
                                          "('s1', 'Northwind Components', 'synthetic supplier'), "
                                          "('s2', 'Apex Alloy', 'synthetic supplier')"))
            await connection.execute(text("INSERT INTO materials VALUES "
                                          "('m1', 'Recycled Aluminium', 'synthetic material')"))
            await connection.execute(text("INSERT INTO emissions_records VALUES "
                                          "('e1', 'Northwind Scope 3 record', 'synthetic report'), "
                                          "('e2', 'Apex Scope 3 record', 'synthetic report')"))
            await connection.execute(text("INSERT INTO evidence VALUES "
                                          "('v1', 'Verified emissions attestation', 'synthetic evidence')"))
            await connection.execute(text("INSERT INTO supplies VALUES "
                                          "('s1', 'm1', 0.98), ('s2', 'm1', 0.96)"))
            await connection.execute(text("INSERT INTO reported VALUES "
                                          "('s1', 'e1', 0.97), ('s2', 'e2', 0.95)"))
            await connection.execute(text("INSERT INTO has_evidence VALUES ('e1', 'v1', 0.99)"))

        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", RELATIONAL_NEO4J_PASSWORD),
        )
        client = Neo4jClient.__new__(Neo4jClient)
        client._driver = driver
        client._filtered_vector_search = False
        client._filtered_vector_indexes = set()
        client._in_flight = 0
        mapping = RelationalGraphMapping(
            id="sustainability-e2e",
            version="1.0.0",
            source_id="synthetic-postgres-e2e",
            tenant=tenant,
            entities=[
                EntityTableMapping(table="suppliers", entity_type="SUPPLIER", id_column="id", name_column="name", description_column="description"),
                EntityTableMapping(table="materials", entity_type="MATERIAL", id_column="id", name_column="name", description_column="description"),
                EntityTableMapping(table="emissions_records", entity_type="EMISSIONS_RECORD", id_column="id", name_column="name", description_column="description"),
                EntityTableMapping(table="evidence", entity_type="EVIDENCE", id_column="id", name_column="name", description_column="description"),
            ],
            relations=[
                RelationTableMapping(table="supplies", source_table="suppliers", target_table="materials", source_column="supplier_id", target_column="material_id", relation="SUPPLIES", confidence_column="confidence"),
                RelationTableMapping(table="reported", source_table="suppliers", target_table="emissions_records", source_column="supplier_id", target_column="emissions_id", relation="REPORTED", confidence_column="confidence"),
                RelationTableMapping(table="has_evidence", source_table="emissions_records", target_table="evidence", source_column="emissions_id", target_column="evidence_id", relation="HAS_EVIDENCE", confidence_column="confidence"),
            ],
            ontology_version="synthetic-sustainability-supply-chain@1.0.0",
        )
        try:
            writer = GraphWriter(changed_by="relational-e2e", neo4j_client=client)
            with patch("graphrag.graph.alias_registry._get_redis", AsyncMock(return_value=None)):
                report = await RelationalGraphIngestor(
                    PostgreSQLSourceConnector(database_url), writer,
                ).ingest(mapping)

            assert report.shacl_conforms is True
            persisted = await client.run(
                "MATCH (s:KGSource {tenant: $tenant})<-[:INGESTED_FROM]-(d:Document {tenant: $tenant}) "
                "MATCH (d)<-[:PART_OF]-(c:Chunk {tenant: $tenant}) "
                "RETURN s.id AS source_id, d.source_id AS document_source_id, count(c) AS chunks",
                tenant=tenant,
            )
            assert persisted == [{"source_id": "synthetic-postgres-e2e", "document_source_id": "synthetic-postgres-e2e", "chunks": 1}]
            gaps = await execute_controlled_query(
                client, "Which suppliers lack verified emissions evidence?", tenant=tenant,
            )
            assert gaps["intent"] == "suppliers_missing_emissions_evidence"
            assert [row["supplier"] for row in gaps["rows"]] == ["Apex Alloy"]
        finally:
            async with driver.session() as session:
                await session.run("MATCH (n {tenant: $tenant}) DETACH DELETE n", tenant=tenant)
            await driver.close()
            await engine.dispose()
