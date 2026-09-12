"""Durable, row-level incremental ingestion (item 5 of the follow-up
platform critique): RelationalGraphIngestor.ingest_incremental() --
row-level upsert/delete detection, a durable checkpoint, crash recovery
and replay, and concurrent-run protection via a lease.

No live Neo4j: `_StatefulFakeNeo4j` holds real in-memory checkpoint state
(unlike tests/unit/test_relational_incremental.py's `_FakeNeo4j`, which is
a stateless stub for SourceCatalogRepository only) so lease contention,
checkpoint round-trips, and tombstoning can be genuinely exercised across
multiple simulated ingest runs sharing one "durable" backend.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

import pytest

from graphrag.ingestion.relational import (
    ConcurrentIngestError,
    EntityTableMapping,
    RelationTableMapping,
    RelationalGraphIngestor,
    RelationalGraphMapping,
    SQLiteSourceConnector,
)


def _db(tmp_path, supplier_name="Supplier One", include_material_two=False, filename="supply.db"):
    path = tmp_path / filename
    with sqlite3.connect(path) as db:
        db.executescript(f"""
        CREATE TABLE suppliers (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT);
        CREATE TABLE materials (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT);
        CREATE TABLE supplies (supplier_id TEXT NOT NULL, material_id TEXT NOT NULL, confidence REAL);
        INSERT INTO suppliers VALUES ('s1', '{supplier_name}', 'supplier');
        INSERT INTO materials VALUES ('m1', 'Material One', 'material');
        INSERT INTO supplies VALUES ('s1', 'm1', 0.9);
        """)
        if include_material_two:
            db.execute("INSERT INTO materials VALUES ('m2', 'Material Two', 'material')")
    return path


def _mapping():
    return RelationalGraphMapping(
        id="supply-chain",
        version="1.0.0",
        source_id="supplier-db",
        tenant="sustainability",
        entities=[
            EntityTableMapping(table="suppliers", entity_type="SUPPLIER", id_column="id", name_column="name"),
            EntityTableMapping(table="materials", entity_type="MATERIAL", id_column="id", name_column="name"),
        ],
        relations=[RelationTableMapping(
            table="supplies", source_table="suppliers", target_table="materials",
            source_column="supplier_id", target_column="material_id",
            relation="SUPPLIES", confidence_column="confidence",
        )],
    )


@dataclass
class _CheckpointState:
    ingest_complete: bool = False
    row_hashes: dict[str, str] = field(default_factory=dict)
    lease_run_id: str | None = None
    lease_expires_at: float | None = None


class _StatefulFakeNeo4j:
    """Genuine in-memory checkpoint/lease state, keyed by (tenant, source_id)
    -- shared across multiple RelationalGraphIngestor instances in a test to
    simulate one durable backend surviving across separate ingest runs."""

    def __init__(self):
        self._checkpoints: dict[tuple[str, str], _CheckpointState] = {}
        self.tombstoned_calls: list[tuple[str, str, list[str]]] = []

    async def run(self, *args, **kwargs):
        return [{"id": kwargs.get("id", "ok")}]

    async def begin_relational_ingest_run(self, tenant, source_id, run_id, lease_seconds):
        state = self._checkpoints.setdefault((tenant, source_id), _CheckpointState())
        now = time.time()
        claimable = (
            state.lease_run_id is None
            or state.lease_expires_at is None
            or state.lease_expires_at < now
            or state.lease_run_id == run_id
        )
        if claimable:
            state.lease_run_id = run_id
            state.lease_expires_at = now + lease_seconds
        return claimable

    async def get_relational_source_state(self, tenant, source_id):
        state = self._checkpoints.get((tenant, source_id))
        if state is None:
            return None
        return {
            "row_hashes": dict(state.row_hashes),
            "ingest_complete": state.ingest_complete,
            "run_id": state.lease_run_id,
            "lease_expires_at": state.lease_expires_at,
        }

    async def complete_relational_ingest_run(self, tenant, source_id, run_id, row_hashes):
        state = self._checkpoints.get((tenant, source_id))
        if state is None or state.lease_run_id != run_id:
            return  # superseded run -- must never clobber a newer completion
        state.ingest_complete = True
        state.row_hashes = dict(row_hashes)
        state.lease_run_id = None
        state.lease_expires_at = None

    async def tombstone_relational_rows(self, tenant, source_id, entity_ids):
        self.tombstoned_calls.append((tenant, source_id, list(entity_ids)))
        return len(entity_ids)


class _RecordingWriter:
    def __init__(self, neo4j: _StatefulFakeNeo4j):
        self.neo4j_client = neo4j
        self.documents: list = []
        self.entity_batches: list = []
        self.relation_batches: list = []

    async def write_document(self, document):
        self.documents.append(document)
        return document.id

    async def write_chunks(self, chunks):
        return chunks

    async def write_entities(self, entities, chunk):
        self.entity_batches.append(entities)
        return entities

    async def write_relations(self, relations, entity_map, doc_id, tenant):
        self.relation_batches.append(relations)
        return relations


class _AssertNotCalledWriter:
    def __init__(self, neo4j: _StatefulFakeNeo4j):
        self.neo4j_client = neo4j

    async def write_document(self, _doc):
        raise AssertionError("skip must not reach write_document")

    async def write_chunks(self, _chunks):
        raise AssertionError("skip must not reach write_chunks")

    async def write_entities(self, _entities, _chunk):
        raise AssertionError("skip must not reach write_entities")

    async def write_relations(self, *_a, **_kw):
        raise AssertionError("skip must not reach write_relations")


class TestRowLevelDiffAndDurableCheckpoint:
    async def test_first_run_writes_everything_and_completes_the_checkpoint(self, tmp_path):
        neo4j = _StatefulFakeNeo4j()
        writer = _RecordingWriter(neo4j)
        ingestor = RelationalGraphIngestor(SQLiteSourceConnector(_db(tmp_path)), writer)

        report = await ingestor.ingest_incremental(_mapping(), run_id="run-1")

        assert report.skipped is False
        assert len(writer.documents) == 1
        assert len(report.upserted) == 3  # 2 entity rows + 1 relation row
        assert report.deleted == []
        state = await neo4j.get_relational_source_state("sustainability", "supplier-db")
        assert state["ingest_complete"] is True
        assert state["row_hashes"]  # persisted

    async def test_unchanged_second_run_skips_with_zero_writes(self, tmp_path):
        neo4j = _StatefulFakeNeo4j()
        db_path = _db(tmp_path)
        await RelationalGraphIngestor(
            SQLiteSourceConnector(db_path), _RecordingWriter(neo4j),
        ).ingest_incremental(_mapping(), run_id="run-1")

        second = await RelationalGraphIngestor(
            SQLiteSourceConnector(db_path), _AssertNotCalledWriter(neo4j),
        ).ingest_incremental(_mapping(), run_id="run-2")

        assert second.skipped is True
        assert second.upserted == []
        assert len(second.unchanged) == 3

    async def test_one_changed_row_is_reported_as_upserted_and_triggers_a_write(self, tmp_path):
        neo4j = _StatefulFakeNeo4j()
        db_path = _db(tmp_path, supplier_name="Supplier One")
        await RelationalGraphIngestor(
            SQLiteSourceConnector(db_path), _RecordingWriter(neo4j),
        ).ingest_incremental(_mapping(), run_id="run-1")

        changed_path = _db(tmp_path, supplier_name="Supplier One Renamed", filename="supply-v2.db")
        writer = _RecordingWriter(neo4j)
        report = await RelationalGraphIngestor(
            SQLiteSourceConnector(changed_path), writer,
        ).ingest_incremental(_mapping(), run_id="run-2")

        assert report.skipped is False
        assert report.upserted == ["entity:suppliers:s1"]
        assert len(report.unchanged) == 2  # the material and the supply relation
        assert len(writer.documents) == 1  # still a full rewrite -- see module docstring

    async def test_a_deleted_row_is_reported_and_tombstoned(self, tmp_path):
        neo4j = _StatefulFakeNeo4j()
        db_with_two_materials = _db(tmp_path, include_material_two=True)
        mapping = _mapping()
        # Wire the second material into a relation table too so materials.m2
        # actually exists as a real, referenced entity row before it's removed.
        await RelationalGraphIngestor(
            SQLiteSourceConnector(db_with_two_materials), _RecordingWriter(neo4j),
        ).ingest_incremental(mapping, run_id="run-1")

        db_without_material_two = _db(tmp_path, include_material_two=False, filename="supply-v2.db")
        writer = _RecordingWriter(neo4j)
        report = await RelationalGraphIngestor(
            SQLiteSourceConnector(db_without_material_two), writer,
        ).ingest_incremental(mapping, run_id="run-2")

        assert report.deleted == ["entity:materials:m2"]
        assert len(neo4j.tombstoned_calls) == 1
        tenant, source_id, entity_ids = neo4j.tombstoned_calls[0]
        assert tenant == "sustainability" and source_id == "supplier-db"
        # The tombstoned id is the SAME deterministic uuid5 the entity was
        # originally written under -- recomputed, not looked up separately.
        from uuid import NAMESPACE_URL, uuid5
        expected_id = str(uuid5(NAMESPACE_URL, "supplier-db:materials:m2"))
        assert entity_ids == [expected_id]


class TestConcurrentRunProtection:
    async def test_a_second_run_while_a_lease_is_held_is_refused(self, tmp_path):
        neo4j = _StatefulFakeNeo4j()
        db_path = _db(tmp_path)
        # Claim the lease directly (simulates run-1 still in flight, never
        # having called complete_relational_ingest_run yet).
        acquired = await neo4j.begin_relational_ingest_run(
            "sustainability", "supplier-db", "run-1", lease_seconds=300,
        )
        assert acquired is True

        with pytest.raises(ConcurrentIngestError):
            await RelationalGraphIngestor(
                SQLiteSourceConnector(db_path), _AssertNotCalledWriter(neo4j),
            ).ingest_incremental(_mapping(), run_id="run-2")

    async def test_the_same_run_id_can_reacquire_its_own_lease(self, tmp_path):
        """Not a race -- a retried call with the SAME run_id (e.g. a caller
        retrying its own operation) must not be refused."""
        neo4j = _StatefulFakeNeo4j()
        await neo4j.begin_relational_ingest_run("sustainability", "supplier-db", "run-1", 300)
        reacquired = await neo4j.begin_relational_ingest_run("sustainability", "supplier-db", "run-1", 300)
        assert reacquired is True


class TestCrashRecoveryAndReplay:
    async def test_an_expired_lease_from_a_crashed_run_can_be_reclaimed(self, tmp_path):
        neo4j = _StatefulFakeNeo4j()
        db_path = _db(tmp_path)
        # Simulate run-1 crashing: it claimed the lease but never completed.
        await neo4j.begin_relational_ingest_run("sustainability", "supplier-db", "run-1", lease_seconds=300)
        state = neo4j._checkpoints[("sustainability", "supplier-db")]
        state.lease_expires_at = time.time() - 1  # force it into the past

        report = await RelationalGraphIngestor(
            SQLiteSourceConnector(db_path), _RecordingWriter(neo4j),
        ).ingest_incremental(_mapping(), run_id="run-2")

        assert report.skipped is False  # replayed the whole thing from scratch
        final_state = await neo4j.get_relational_source_state("sustainability", "supplier-db")
        assert final_state["ingest_complete"] is True
        assert final_state["run_id"] is None  # lease released

    async def test_a_superseded_run_cannot_clobber_a_newer_completion(self, tmp_path):
        """run-1's lease expired and run-2 took over and completed. If
        run-1 (e.g. a zombie process) later calls complete_relational_ingest_run,
        it must be a no-op -- it is no longer the lease holder."""
        neo4j = _StatefulFakeNeo4j()
        db_path = _db(tmp_path)
        await neo4j.begin_relational_ingest_run("sustainability", "supplier-db", "run-1", 300)
        state = neo4j._checkpoints[("sustainability", "supplier-db")]
        state.lease_expires_at = time.time() - 1

        await RelationalGraphIngestor(
            SQLiteSourceConnector(db_path), _RecordingWriter(neo4j),
        ).ingest_incremental(_mapping(), run_id="run-2")

        # run-1's late completion attempt must not override run-2's result.
        await neo4j.complete_relational_ingest_run(
            "sustainability", "supplier-db", "run-1", row_hashes={"stale": "data"},
        )
        final_state = await neo4j.get_relational_source_state("sustainability", "supplier-db")
        assert final_state["row_hashes"] != {"stale": "data"}
