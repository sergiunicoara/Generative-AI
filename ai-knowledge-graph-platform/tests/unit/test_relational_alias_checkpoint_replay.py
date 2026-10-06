"""Regression tests for alias relation endpoints and partial-run replay."""

from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag.graph.neo4j_client import Neo4jClient
from graphrag.ingestion.graph_writer import GraphWriter
from graphrag.ingestion.relational import (
    EntityTableMapping,
    RelationTableMapping,
    RelationalGraphIngestor,
    RelationalGraphMapping,
    SQLiteSourceConnector,
)


def _database(tmp_path):
    path = tmp_path / "alias-replay.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE suppliers (id TEXT PRIMARY KEY, name TEXT NOT NULL);
            CREATE TABLE materials (id TEXT PRIMARY KEY, name TEXT NOT NULL);
            CREATE TABLE supplies (supplier_id TEXT, material_id TEXT);
            INSERT INTO suppliers VALUES ('s1', 'Supplier Alias');
            INSERT INTO materials VALUES ('m1', 'Material Alias');
            INSERT INTO supplies VALUES ('s1', 'm1');
        """)
    return path


def _mapping():
    return RelationalGraphMapping(
        id="alias-regression", version="1.0.0", source_id="alias-db", tenant="tenant-a",
        entities=[
            EntityTableMapping(table="suppliers", entity_type="SUPPLIER", id_column="id", name_column="name"),
            EntityTableMapping(table="materials", entity_type="MATERIAL", id_column="id", name_column="name"),
        ],
        relations=[RelationTableMapping(
            table="supplies", source_table="suppliers", target_table="materials",
            source_column="supplier_id", target_column="material_id", relation="SUPPLIES",
        )],
    )


class _CheckpointGraph:
    def __init__(self):
        self.row_hashes = {}
        self.ingest_complete = False
        self.was_complete_on_claim = False
        self.lease_run_id = None
        self.lease_expires_at = None
        self.tombstoned = []

    async def run(self, *_args, **kwargs):
        return [{"id": kwargs.get("id", "ok")}]

    async def begin_relational_ingest_run(self, tenant, source_id, run_id, lease_seconds):
        now = time.time()
        if self.lease_run_id is not None and self.lease_expires_at >= now and self.lease_run_id != run_id:
            return False
        self.was_complete_on_claim = self.ingest_complete
        self.ingest_complete = False
        self.lease_run_id = run_id
        self.lease_expires_at = now + lease_seconds
        return True

    async def get_relational_source_state(self, tenant, source_id):
        return {
            "row_hashes": dict(self.row_hashes),
            "ingest_complete": self.ingest_complete,
            "was_complete_on_claim": self.was_complete_on_claim,
        }

    async def complete_relational_ingest_run(self, tenant, source_id, run_id, row_hashes):
        assert self.lease_run_id == run_id
        self.row_hashes = dict(row_hashes)
        self.ingest_complete = True
        self.was_complete_on_claim = False
        self.lease_run_id = None
        self.lease_expires_at = None

    async def tombstone_relational_rows(self, tenant, source_id, entity_ids):
        self.tombstoned.extend(entity_ids)
        return len(entity_ids)


def _alias_writer(graph):
    writer = GraphWriter.__new__(GraphWriter)
    writer._neo4j = graph
    writer._audit = SimpleNamespace(
        log_entities_batch=AsyncMock(), log_relations_batch=AsyncMock(),
    )
    writer._changed_by = "test"
    writer._semantic_validator = None
    writer._cfg = SimpleNamespace(ingestion={"review_queue_enabled": False})
    writer._ontology = SimpleNamespace(validate_relation_triplet=lambda *_: (True, "SUPPLIES"))
    registry = MagicMock()
    registry.resolve.side_effect = lambda name, **_: {
        "Supplier Alias": ("Supplier Canonical", "SUPPLIER"),
        "Material Alias": ("Material Canonical", "MATERIAL"),
    }.get(name)
    registry.register_alias = AsyncMock()
    writer._get_registry = lambda tenant: registry
    writer._ensure_registry = AsyncMock()
    writer.write_document = AsyncMock(side_effect=lambda document: document.id)
    writer.write_chunks = AsyncMock()
    graph.merge_mentions = AsyncMock()
    graph.set_entity_resolution_metadata = AsyncMock()
    graph.merge_entities_batch = AsyncMock(return_value=[])
    graph.merge_mentions_batch = AsyncMock()
    graph.merge_contextual_entity_representations = AsyncMock()
    graph.merge_relations_batch = AsyncMock()
    return writer


@pytest.mark.parametrize("incremental", [False, True], ids=["snapshot", "incremental"])
async def test_relational_alias_endpoints_are_written_with_canonical_names(tmp_path, incremental):
    graph = _CheckpointGraph()
    writer = _alias_writer(graph)
    ingestor = RelationalGraphIngestor(SQLiteSourceConnector(_database(tmp_path)), writer)

    if incremental:
        await ingestor.ingest_incremental(_mapping(), run_id="run-1")
    else:
        await ingestor.ingest(_mapping())

    graph.merge_entities_batch.assert_awaited_once_with([], tenant="tenant-a")
    relation_rows = graph.merge_relations_batch.await_args.args[0]
    assert len(relation_rows) == 1
    assert {key: relation_rows[0][key] for key in ("src_name", "src_type", "tgt_name", "tgt_type")} == {
        "src_name": "Supplier Canonical", "src_type": "SUPPLIER",
        "tgt_name": "Material Canonical", "tgt_type": "MATERIAL",
    }
    assert graph.merge_relations_batch.await_args.kwargs == {"tenant": "tenant-a"}


class _ReplayWriter:
    def __init__(self, graph, *, fail_after_entities=False):
        self.neo4j_client = graph
        self.fail_after_entities = fail_after_entities
        self.documents = []
        self.entities = []

    async def write_document(self, document):
        self.documents.append(document)
        return document.id

    async def write_chunks(self, chunks):
        return chunks

    async def write_entities(self, entities, chunk):
        self.entities.extend(entities)
        if self.fail_after_entities:
            raise RuntimeError("partial graph write")
        return entities

    async def write_relations(self, relations, entity_map, doc_id, tenant):
        assert all(rel.source_entity_id in entity_map and rel.target_entity_id in entity_map for rel in relations)


async def test_failed_partial_run_replays_when_source_reverts_to_completed_hashes(tmp_path):
    path = _database(tmp_path)
    graph = _CheckpointGraph()
    def ingestor(writer):
        return RelationalGraphIngestor(SQLiteSourceConnector(path), writer)
    baseline = await ingestor(_ReplayWriter(graph)).ingest_incremental(_mapping(), run_id="baseline")
    assert baseline.skipped is False

    with sqlite3.connect(path) as db:
        db.execute("UPDATE suppliers SET name = 'Changed During Failure' WHERE id = 's1'")
    with pytest.raises(RuntimeError, match="partial graph write"):
        await ingestor(_ReplayWriter(graph, fail_after_entities=True)).ingest_incremental(
            _mapping(), run_id="failed",
        )
    assert graph.ingest_complete is False
    assert graph.lease_run_id == "failed"
    graph.lease_expires_at = time.time() - 1

    with sqlite3.connect(path) as db:
        db.execute("UPDATE suppliers SET name = 'Supplier Alias' WHERE id = 's1'")
    replay_writer = _ReplayWriter(graph)
    replay = await ingestor(replay_writer).ingest_incremental(_mapping(), run_id="retry")

    assert replay.skipped is False
    assert replay.upserted == [] and replay.deleted == []
    assert len(replay_writer.documents) == 1
    assert len(replay_writer.entities) == 2
    assert graph.ingest_complete is True
    assert graph.lease_run_id is None


async def test_retry_after_partial_run_still_tombstones_deleted_rows(tmp_path):
    path = _database(tmp_path)
    graph = _CheckpointGraph()
    def ingestor(writer):
        return RelationalGraphIngestor(SQLiteSourceConnector(path), writer)
    await ingestor(_ReplayWriter(graph)).ingest_incremental(_mapping(), run_id="baseline")

    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM supplies")
        db.execute("DELETE FROM materials")
    with pytest.raises(RuntimeError, match="partial graph write"):
        await ingestor(_ReplayWriter(graph, fail_after_entities=True)).ingest_incremental(
            _mapping(), run_id="failed",
        )
    graph.lease_expires_at = time.time() - 1

    replay = await ingestor(_ReplayWriter(graph)).ingest_incremental(_mapping(), run_id="retry")

    assert replay.skipped is False
    assert set(replay.deleted) == {"entity:materials:m1", "relation:supplies:s1:m1"}
    assert len(graph.tombstoned) == 1
    assert graph.ingest_complete is True


async def test_neo4j_lease_claim_invalidates_completion_without_changing_hashes():
    client = Neo4jClient.__new__(Neo4jClient)
    client.run = AsyncMock(return_value=[{"claimable": True}])

    assert await client.begin_relational_ingest_run("tenant-a", "alias-db", "run-2", 300)

    query = client.run.await_args.args[0]
    assert "c.was_complete_on_claim = was_complete" in query
    assert "c.ingest_complete = false" in query
    assert "c.row_hashes_json = $row_hashes_json" not in query
    assert "CASE WHEN claimable THEN [1] ELSE [] END" in query


async def test_neo4j_checkpoint_state_exposes_preclaim_completion():
    client = Neo4jClient.__new__(Neo4jClient)
    client.run = AsyncMock(return_value=[{
        "row_hashes_json": '{"entity:suppliers:s1":"hash"}',
        "ingest_complete": False,
        "was_complete_on_claim": True,
        "run_id": "run-2", "lease_expires_at": None,
    }])

    state = await client.get_relational_source_state("tenant-a", "alias-db")

    assert state["ingest_complete"] is False
    assert state["was_complete_on_claim"] is True
    assert state["row_hashes"] == {"entity:suppliers:s1": "hash"}
