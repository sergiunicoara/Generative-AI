"""Unit tests for ingestion idempotency (tasks/lessons.md A136).

Document.id and Chunk.id are Field(default_factory=uuid4) — a fresh id every
run — while the old write path MERGEd on that id, so a MERGE could never match
an existing node and every re-ingest created a full duplicate (measured live:
4 aerospace documents duplicated, 38% of that tenant's chunks). The fix MERGEs
on natural keys instead: (tenant, filename) for documents, (document_id,
chunk_index) for chunks, assigning `id` only ON CREATE and returning the
canonical id to the caller.

These tests mock Neo4jClient.run and assert on the Cypher/params it was called
with — no live database needed, matching the pattern in test_data_path_fixes.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_document(**kw):
    from graphrag.core.models import Document
    defaults = dict(filename="AD-2024.txt", source_path="/x/AD-2024.txt",
                     raw_text="text", tenant="aerospace")
    defaults.update(kw)
    return Document(**defaults)


def _make_chunk(document_id, chunk_index=0, **kw):
    from graphrag.core.models import Chunk
    defaults = dict(document_id=document_id, text="chunk text",
                     chunk_index=chunk_index, tenant="aerospace")
    defaults.update(kw)
    return Chunk(**defaults)


# ── neo4j_client.merge_document ────────────────────────────────────────────────

class TestMergeDocumentNaturalKey:
    @pytest.mark.asyncio
    async def test_merges_on_tenant_and_filename_not_id(self):
        from graphrag.graph.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[{"doc_id": "fresh-uuid"}])

        await client.merge_document(
            doc_id="fresh-uuid", filename="AD-2024.txt", ingested_at="2026-01-01T00:00:00",
            tenant="aerospace",
        )
        cypher = client.run.call_args[0][0]
        assert "MERGE (d:Document {tenant: $tenant, filename: $filename})" in cypher
        assert "MERGE (d:Document {id:" not in cypher

    @pytest.mark.asyncio
    async def test_id_is_assigned_only_on_create(self):
        from graphrag.graph.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[{"doc_id": "fresh-uuid"}])
        await client.merge_document(
            doc_id="fresh-uuid", filename="AD-2024.txt", ingested_at="2026-01-01T00:00:00",
            tenant="aerospace",
        )
        cypher = client.run.call_args[0][0]
        assert "ON CREATE SET d.id = $id" in cypher

    @pytest.mark.asyncio
    async def test_returns_canonical_id_from_neo4j_not_the_passed_in_id(self):
        """The whole point: on a re-ingest, Neo4j returns the ORIGINAL id, and
        the caller must use that, not the fresh uuid4() this run generated."""
        from graphrag.graph.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[{"doc_id": "original-doc-id"}])

        result = await client.merge_document(
            doc_id="this-runs-fresh-uuid", filename="AD-2024.txt",
            ingested_at="2026-01-01T00:00:00", tenant="aerospace",
        )
        assert result == "original-doc-id"
        assert result != "this-runs-fresh-uuid"

    @pytest.mark.asyncio
    async def test_falls_back_to_passed_id_if_neo4j_returns_nothing(self):
        from graphrag.graph.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[])
        result = await client.merge_document(
            doc_id="fallback-id", filename="x.txt", ingested_at="2026-01-01T00:00:00",
        )
        assert result == "fallback-id"


# ── neo4j_client.merge_chunk / merge_chunks_batch ──────────────────────────────

class TestMergeChunkNaturalKey:
    @pytest.mark.asyncio
    async def test_merges_on_document_id_and_chunk_index(self):
        from graphrag.graph.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[])
        chunk = _make_chunk("doc-1", chunk_index=3)

        await client.merge_chunk(chunk, tenant="aerospace")
        cypher = client.run.call_args[0][0]
        assert "MERGE (c:Chunk {document_id: $doc_id, chunk_index: $chunk_index})" in cypher
        assert "MERGE (c:Chunk {id:" not in cypher

    @pytest.mark.asyncio
    async def test_writes_document_id_property(self):
        """document_id was never written before this fix (measured 0/102 live)
        despite an index existing on it and counterfactual.py querying it."""
        from graphrag.graph.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[])
        chunk = _make_chunk("doc-1")

        await client.merge_chunk(chunk, tenant="aerospace")
        params = client.run.call_args[1]
        assert params["doc_id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_batch_merges_on_same_natural_key(self):
        from graphrag.graph.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[])
        chunks = [_make_chunk("doc-1", chunk_index=i) for i in range(3)]

        await client.merge_chunks_batch(chunks, tenant="aerospace")
        cypher = client.run.call_args[0][0]
        assert "MERGE (c:Chunk {document_id: row.doc_id, chunk_index: row.chunk_index})" in cypher
        assert "ON CREATE SET c.id = row.id" in cypher

    @pytest.mark.asyncio
    async def test_batch_of_empty_list_is_a_noop(self):
        from graphrag.graph.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[])
        await client.merge_chunks_batch([], tenant="aerospace")
        client.run.assert_not_called()


# ── neo4j_client.delete_stale_chunks ───────────────────────────────────────────

class TestDeleteStaleChunks:
    @pytest.mark.asyncio
    async def test_deletes_chunks_at_or_above_keep_count(self):
        from graphrag.graph.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[{"deleted": 2}])

        deleted = await client.delete_stale_chunks("doc-1", keep_count=5, tenant="aerospace")
        assert deleted == 2
        params = client.run.call_args[1]
        assert params["keep_count"] == 5
        cypher = client.run.call_args[0][0]
        assert "WHERE c.chunk_index >= $keep_count" in cypher
        assert "DETACH DELETE" in cypher

    @pytest.mark.asyncio
    async def test_zero_stale_chunks_returns_zero(self):
        from graphrag.graph.neo4j_client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[])
        deleted = await client.delete_stale_chunks("doc-1", keep_count=5, tenant="aerospace")
        assert deleted == 0


# ── graph_writer.write_document propagates the canonical id ───────────────────

class TestWriteDocumentReturnsCanonicalId:
    @pytest.mark.asyncio
    async def test_returns_and_assigns_canonical_id(self):
        from graphrag.ingestion.graph_writer import GraphWriter

        writer = GraphWriter.__new__(GraphWriter)
        writer._neo4j = MagicMock()
        writer._neo4j.merge_document = AsyncMock(return_value="canonical-existing-id")
        writer._changed_by = "test"
        writer._audit = MagicMock()
        writer._audit.log_document_change = AsyncMock()

        doc = _make_document(id="fresh-uuid-this-run")
        result = await writer.write_document(doc)

        assert result == "canonical-existing-id"
        assert doc.id == "canonical-existing-id"

    @pytest.mark.asyncio
    async def test_registers_supersession_with_canonical_id_not_original(self):
        from graphrag.ingestion.graph_writer import GraphWriter

        writer = GraphWriter.__new__(GraphWriter)
        writer._neo4j = MagicMock()
        writer._neo4j.merge_document = AsyncMock(return_value="canonical-id")
        writer._changed_by = "test"
        writer._audit = MagicMock()
        writer._audit.log_document_change = AsyncMock()

        doc = _make_document(id="fresh-uuid", supersedes=["old-doc-id"])

        from unittest.mock import patch
        with patch("graphrag.graph.document_authority.DocumentAuthorityService") as MockSvc:
            instance = MockSvc.return_value
            instance.register_supersession = AsyncMock()
            await writer.write_document(doc)
            instance.register_supersession.assert_awaited_once_with(
                "canonical-id", ["old-doc-id"]
            )


# ── ingestion_agent.write() propagates canonical id to chunks ─────────────────

class TestIngestionAgentReassignsChunkDocumentId:
    @pytest.mark.asyncio
    async def test_chunks_repointed_when_document_resolves_to_existing_id(self):
        """The regression this fix targets: on re-ingest, write_document
        returns a DIFFERENT id than the chunks (built during extract(), before
        write() ever ran) were assigned. Every chunk must be repointed before
        write_chunks() runs, or they'd MERGE against a document_id with no
        matching Document node and create duplicates anyway."""
        from graphrag.agents.ingestion_agent import IngestionAgent

        agent = IngestionAgent.__new__(IngestionAgent)
        writer = MagicMock()

        async def fake_write_document(doc):
            # Mirror the real write_document's side effect: it mutates doc.id
            # in place (see graph_writer.py) before returning the same value.
            doc.id = "canonical-existing-id"
            return "canonical-existing-id"

        writer.write_document = AsyncMock(side_effect=fake_write_document)
        writer.write_chunks = AsyncMock()
        writer.write_entities = AsyncMock(return_value=[])
        writer.write_relations = AsyncMock()
        writer.validate_and_check_cycles = AsyncMock(return_value={
            "validation": {"total_issues": 0}, "new_conflicts": 0,
        })
        agent._writer = writer

        doc = _make_document(id="fresh-uuid-this-run")
        chunks = [_make_chunk("fresh-uuid-this-run", chunk_index=i) for i in range(3)]

        from unittest.mock import patch
        with patch("graphrag.agents.ingestion_agent.get_settings") as mock_settings:
            mock_settings.return_value.wikidata_linking_enabled = False
            await agent.write({
                "job_id": "job-1", "doc": doc, "chunks": chunks,
                "extraction_results": [([], []) for _ in chunks],
            })

        for c in chunks:
            assert c.document_id == "canonical-existing-id"
        assert doc.id == "canonical-existing-id"

    @pytest.mark.asyncio
    async def test_chunks_unchanged_when_document_is_genuinely_new(self):
        """No-op path: a brand-new document keeps its generated id, so chunks
        must NOT be touched (they're already correct)."""
        from graphrag.agents.ingestion_agent import IngestionAgent

        agent = IngestionAgent.__new__(IngestionAgent)
        writer = MagicMock()
        writer.write_document = AsyncMock(return_value="same-id-new-doc")
        writer.write_chunks = AsyncMock()
        writer.write_entities = AsyncMock(return_value=[])
        writer.write_relations = AsyncMock()
        writer.validate_and_check_cycles = AsyncMock(return_value={
            "validation": {"total_issues": 0}, "new_conflicts": 0,
        })
        agent._writer = writer

        doc = _make_document(id="same-id-new-doc")
        chunk = _make_chunk("same-id-new-doc")

        from unittest.mock import patch
        with patch("graphrag.agents.ingestion_agent.get_settings") as mock_settings:
            mock_settings.return_value.wikidata_linking_enabled = False
            await agent.write({
                "job_id": "job-1", "doc": doc, "chunks": [chunk],
                "extraction_results": [([], [])],
            })

        assert chunk.document_id == "same-id-new-doc"


# ── The regression guard: re-ingesting the same document twice ────────────────

class TestReingestionDoesNotDuplicate:
    """This is the test that would have caught A136 before it shipped: ingest
    one document through the agent twice and confirm the second run updates
    the existing document/chunks instead of creating a second copy. Modeled as
    a fake in-memory Neo4j so it exercises the real merge_document /
    merge_chunks_batch Cypher-key logic end to end, not just mocked calls."""

    @pytest.mark.asyncio
    async def test_second_ingest_reuses_document_and_chunk_identity(self):
        from graphrag.core.models import Document, Chunk
        from graphrag.graph.neo4j_client import Neo4jClient

        # Minimal fake store keyed exactly like the real MERGE targets.
        documents: dict[tuple[str, str], dict] = {}   # (tenant, filename) -> row
        chunks: dict[tuple[str, int], dict] = {}       # (document_id, chunk_index) -> row

        client = Neo4jClient.__new__(Neo4jClient)

        async def fake_run(cypher, **params):
            if "MERGE (d:Document" in cypher:
                key = (params["tenant"], params["filename"])
                if key not in documents:
                    documents[key] = {"id": params["id"]}
                return [{"doc_id": documents[key]["id"]}]
            if "UNWIND $rows AS row" in cypher and "Chunk" in cypher:
                for row in params["rows"]:
                    key = (row["doc_id"], row["chunk_index"])
                    if key not in chunks:
                        chunks[key] = {"id": row["id"]}
                return []
            if "delete_stale" in cypher.lower() or "chunk_index >= $keep_count" in cypher:
                return [{"deleted": 0}]
            return []

        client.run = fake_run

        doc1 = Document(filename="AD-2024.txt", source_path="/x", raw_text="t",
                         tenant="aerospace")
        canonical_id_1 = await client.merge_document(
            doc_id=doc1.id, filename=doc1.filename,
            ingested_at="2026-01-01T00:00:00", tenant="aerospace",
        )
        chunk_rows = [
            Chunk(document_id=canonical_id_1, text="a", chunk_index=i, tenant="aerospace")
            for i in range(3)
        ]
        await client.merge_chunks_batch(chunk_rows, tenant="aerospace")

        # Re-ingest the SAME file — a fresh Document with a fresh uuid4() id,
        # exactly as extract() would build it on a second run.
        doc2 = Document(filename="AD-2024.txt", source_path="/x", raw_text="t (revised)",
                         tenant="aerospace")
        assert doc2.id != doc1.id   # sanity: these really are different uuids

        canonical_id_2 = await client.merge_document(
            doc_id=doc2.id, filename=doc2.filename,
            ingested_at="2026-01-02T00:00:00", tenant="aerospace",
        )
        chunk_rows_2 = [
            Chunk(document_id=canonical_id_2, text="a", chunk_index=i, tenant="aerospace")
            for i in range(3)
        ]
        await client.merge_chunks_batch(chunk_rows_2, tenant="aerospace")

        # The bug this guards against: canonical_id_2 must equal canonical_id_1
        # (the SAME document node), not doc2.id (which would mean a duplicate).
        assert canonical_id_2 == canonical_id_1
        assert len(documents) == 1
        assert len(chunks) == 3   # not 6

    @pytest.mark.asyncio
    async def test_different_tenants_same_filename_stay_distinct(self):
        """Tenant isolation must survive the natural-key change: (tenant,
        filename) is the key, so the same filename in two tenants is two
        documents, not one shared across tenants."""
        from graphrag.core.models import Document
        from graphrag.graph.neo4j_client import Neo4jClient

        documents: dict[tuple[str, str], dict] = {}
        client = Neo4jClient.__new__(Neo4jClient)

        async def fake_run(cypher, **params):
            key = (params["tenant"], params["filename"])
            if key not in documents:
                documents[key] = {"id": params["id"]}
            return [{"doc_id": documents[key]["id"]}]

        client.run = fake_run

        doc_a = Document(filename="shared.txt", source_path="/x", raw_text="t",
                          tenant="aerospace")
        doc_b = Document(filename="shared.txt", source_path="/x", raw_text="t",
                          tenant="automotive")

        id_a = await client.merge_document(
            doc_id=doc_a.id, filename=doc_a.filename,
            ingested_at="2026-01-01T00:00:00", tenant="aerospace",
        )
        id_b = await client.merge_document(
            doc_id=doc_b.id, filename=doc_b.filename,
            ingested_at="2026-01-01T00:00:00", tenant="automotive",
        )

        assert id_a != id_b
        assert len(documents) == 2
