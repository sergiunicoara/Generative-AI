"""Content-hash checkpoint for RelationalGraphIngestor.ingest() (item 4 of
the wishlist): unchanged snapshot -> skip with no write; changed snapshot ->
write with the correct content_hash set; no previous hash -> treated as new,
always writes.

No live Neo4j: graph_writer is a minimal fake exposing exactly the duck-typed
surface RelationalGraphIngestor.ingest() calls (neo4j_client.run,
write_document, write_chunks, write_entities, write_relations), recording
every call so a skip can be asserted as "wrote nothing" the same way
test_relational_ingestion.py's existing Writer classes already assert
"must not be called".
"""
from __future__ import annotations

import sqlite3

import pytest

from graphrag.ingestion.incremental import (
    compute_relational_snapshot_hash,
    should_skip_ingest,
)
from graphrag.ingestion.relational import (
    EntityTableMapping,
    RelationTableMapping,
    RelationalGraphIngestor,
    RelationalGraphMapping,
    SQLiteSourceConnector,
)


def _db(tmp_path, supplier_name="Supplier One", filename="supply.db"):
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


class _FakeNeo4j:
    """SourceCatalogRepository's add_mapping() treats an empty result as
    "the MATCHed source doesn't exist" and raises -- a real MERGE...RETURN
    always returns the merged row, so this stub must too."""

    async def run(self, *args, **kwargs):
        return [{"id": kwargs.get("id", "ok")}]


class _RecordingWriter:
    """Full happy-path fake — records every write instead of raising, unlike
    test_relational_ingestion.py's assert-not-called Writer classes, since
    the "changed snapshot" test needs the write path to actually succeed."""

    def __init__(self):
        self.neo4j_client = _FakeNeo4j()
        self.documents: list = []
        self.chunk_batches: list = []
        self.entity_batches: list = []
        self.relation_batches: list = []

    async def write_document(self, document):
        self.documents.append(document)
        return document.id

    async def write_chunks(self, chunks):
        self.chunk_batches.append(chunks)
        return chunks

    async def write_entities(self, entities, chunk):
        self.entity_batches.append(entities)
        return entities

    async def write_relations(self, relations, entity_map, doc_id, tenant):
        self.relation_batches.append(relations)
        return relations


class _AssertNotCalledWriter:
    """Mirrors test_relational_ingestion.py's existing pattern: any write
    call is a test failure, proving a skip performed zero writes."""

    async def write_document(self, _doc):
        raise AssertionError("skip must not reach write_document")

    async def write_chunks(self, _chunks):
        raise AssertionError("skip must not reach write_chunks")

    async def write_entities(self, _entities, _chunk):
        raise AssertionError("skip must not reach write_entities")

    async def write_relations(self, *_a, **_kw):
        raise AssertionError("skip must not reach write_relations")


class TestComputeRelationalSnapshotHash:
    def test_same_rows_same_hash_regardless_of_dict_key_order(self):
        a = [{"table": "suppliers", "row": {"id": "s1", "name": "Supplier One"}}]
        b = [{"row": {"name": "Supplier One", "id": "s1"}, "table": "suppliers"}]
        assert compute_relational_snapshot_hash(a) == compute_relational_snapshot_hash(b)

    def test_different_rows_different_hash(self):
        a = [{"table": "suppliers", "row": {"id": "s1", "name": "Supplier One"}}]
        b = [{"table": "suppliers", "row": {"id": "s1", "name": "Supplier Two"}}]
        assert compute_relational_snapshot_hash(a) != compute_relational_snapshot_hash(b)


class TestShouldSkipIngest:
    def test_matching_hash_skips(self):
        assert should_skip_ingest("abc123", "abc123") is True

    def test_different_hash_does_not_skip(self):
        assert should_skip_ingest("abc123", "def456") is False

    def test_missing_previous_hash_never_skips(self):
        assert should_skip_ingest(None, "abc123") is False
        assert should_skip_ingest("", "abc123") is False


class TestRelationalGraphIngestorIncremental:
    @pytest.mark.asyncio
    async def test_no_previous_hash_is_treated_as_new_and_writes(self, tmp_path):
        writer = _RecordingWriter()
        ingestor = RelationalGraphIngestor(SQLiteSourceConnector(_db(tmp_path)), writer)
        report = await ingestor.ingest(_mapping())

        assert report.skipped is False
        assert report.content_hash  # populated
        assert len(writer.documents) == 1
        assert writer.documents[0].content_hash == report.content_hash

    @pytest.mark.asyncio
    async def test_unchanged_snapshot_skips_with_zero_writes(self, tmp_path):
        db_path = _db(tmp_path)
        # First pass to learn the hash a real caller would have persisted.
        first_writer = _RecordingWriter()
        first_report = await RelationalGraphIngestor(
            SQLiteSourceConnector(db_path), first_writer
        ).ingest(_mapping())

        second_writer = _AssertNotCalledWriter()
        second_report = await RelationalGraphIngestor(
            SQLiteSourceConnector(db_path), second_writer
        ).ingest(_mapping(), previous_hash=first_report.content_hash)

        assert second_report.skipped is True
        assert second_report.content_hash == first_report.content_hash

    @pytest.mark.asyncio
    async def test_changed_snapshot_writes_and_updates_the_hash(self, tmp_path):
        db_path = _db(tmp_path, supplier_name="Supplier One")
        first_report = await RelationalGraphIngestor(
            SQLiteSourceConnector(db_path), _RecordingWriter()
        ).ingest(_mapping())

        changed_path = _db(tmp_path, supplier_name="Supplier One Renamed", filename="supply-v2.db")
        writer = _RecordingWriter()
        report = await RelationalGraphIngestor(
            SQLiteSourceConnector(changed_path), writer
        ).ingest(_mapping(), previous_hash=first_report.content_hash)

        assert report.skipped is False
        assert report.content_hash != first_report.content_hash
        assert len(writer.documents) == 1
        assert writer.documents[0].content_hash == report.content_hash

    @pytest.mark.asyncio
    async def test_document_filename_and_id_are_stable_across_mapping_versions(self, tmp_path):
        db_path = _db(tmp_path)
        writer = _RecordingWriter()
        mapping_v1 = _mapping()
        mapping_v2 = _mapping()
        mapping_v2.version = "2.0.0"

        await RelationalGraphIngestor(SQLiteSourceConnector(db_path), writer).ingest(mapping_v1)
        await RelationalGraphIngestor(SQLiteSourceConnector(db_path), _RecordingWriter()).ingest(
            mapping_v2, previous_hash="deliberately-different-to-force-a-write"
        )

        # Both writes must target the same Document identity (filename and
        # id), even though mapping.version differs -- that's what lets
        # merge_document's (tenant, filename) MERGE update in place across a
        # mapping-version bump instead of creating a parallel Document.
        doc_v1 = writer.documents[0]
        # second ingest used a fresh writer instance above deliberately, so
        # re-derive independently via a third run against the same writer.
        writer2 = _RecordingWriter()
        await RelationalGraphIngestor(SQLiteSourceConnector(db_path), writer2).ingest(mapping_v2)
        doc_v2 = writer2.documents[0]

        assert doc_v1.filename == doc_v2.filename == "relational://supplier-db"
        assert doc_v1.id == doc_v2.id
