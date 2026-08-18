"""Content hashing + tombstones — incremental ingestion (P5).

Covers graphrag/core/content_hash.py and the Neo4j-layer plumbing that uses
it. See docs/context_graph_gap_plan.md.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest

from graphrag.core.content_hash import (
    NO_HASH,
    compute_content_hash,
    content_changed,
)


class TestComputeContentHash:
    def test_is_deterministic(self):
        assert compute_content_hash("abc") == compute_content_hash("abc")

    def test_is_sensitive_to_change(self):
        assert compute_content_hash("abc") != compute_content_hash("abd")

    def test_returns_full_sha256_hex(self):
        h = compute_content_hash("abc")
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)

    def test_handles_unicode(self):
        """The aerospace/automotive corpora contain Romanian diacritics; a
        hash that raised on non-ASCII would break ingestion for a whole
        tenant (cf. tasks/lessons.md A106/A88)."""
        assert compute_content_hash("neconformitate șiîăț") != compute_content_hash("neconformitate")

    def test_empty_string_hashes(self):
        assert len(compute_content_hash("")) == 64


class TestContentChanged:
    def test_missing_stored_hash_means_changed(self):
        """Pre-migration documents have no hash. 'Unknown' must mean
        're-ingest', never 'assume unchanged' — otherwise a document that
        predates hashing would be frozen out of every future ingest, exactly
        the bug the binary ingest_complete checkpoint had."""
        assert content_changed(NO_HASH, compute_content_hash("x")) is True
        assert content_changed("", compute_content_hash("x")) is True

    def test_identical_hash_means_unchanged(self):
        h = compute_content_hash("x")
        assert content_changed(h, h) is False

    def test_different_hash_means_changed(self):
        assert content_changed(compute_content_hash("x"), compute_content_hash("y")) is True


class TestDocumentModelFields:
    def test_defaults_are_safe(self):
        from graphrag.core.models import Document
        d = Document(filename="f.txt", source_path="/f.txt", raw_text="x")
        assert d.content_hash == ""       # -> treated as "changed"
        assert d.is_deleted is False
        assert d.deleted_at is None


class TestMergeDocumentPersistsHash:
    @pytest.mark.asyncio
    async def test_content_hash_written(self):
        from graphrag.graph.neo4j_client import Neo4jClient
        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[{"doc_id": "d1"}])

        await client.merge_document(
            doc_id="d1", filename="f.txt", ingested_at="2026-01-01T00:00:00Z",
            tenant="acme", content_hash="deadbeef",
        )
        cypher = client.run.call_args[0][0]
        params = client.run.call_args[1]
        assert "d.content_hash    = $content_hash" in cypher
        assert params["content_hash"] == "deadbeef"

    @pytest.mark.asyncio
    async def test_reingest_clears_tombstone(self):
        """A file reappearing on disk is a resurrection: its chunks must
        become retrievable again rather than staying invisible."""
        from graphrag.graph.neo4j_client import Neo4jClient
        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[{"doc_id": "d1"}])

        await client.merge_document(
            doc_id="d1", filename="f.txt", ingested_at="2026-01-01T00:00:00Z", tenant="acme",
        )
        cypher = client.run.call_args[0][0]
        assert "d.is_deleted      = false" in cypher
        assert "d.deleted_at      = null" in cypher


class TestTombstoneDocuments:
    @pytest.mark.asyncio
    async def test_empty_list_is_a_noop(self):
        from graphrag.graph.neo4j_client import Neo4jClient
        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[])
        assert await client.tombstone_documents([], tenant="acme") == 0
        client.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_is_tenant_scoped_and_soft(self):
        from graphrag.graph.neo4j_client import Neo4jClient
        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[{"tombstoned": 2}])

        n = await client.tombstone_documents(["a.txt", "b.txt"], tenant="acme")
        cypher = client.run.call_args[0][0]
        params = client.run.call_args[1]

        assert n == 2
        assert "{tenant: $tenant, filename: fname}" in cypher
        assert params["tenant"] == "acme"
        # Soft-delete only — never a physical delete. Checked as whole Cypher
        # clauses, not a bare "DELETE" substring, which would false-positive
        # on the `is_deleted` property name itself.
        assert not re.search(r"DETACH\s+DELETE", cypher, re.I)
        assert not re.search(r"(?<!_)DELETE\s+\w", cypher, re.I)
        assert "d.is_deleted = true" in cypher

    @pytest.mark.asyncio
    async def test_does_not_recount_already_tombstoned(self):
        """Safe to run on every ingest — re-tombstoning must be a no-op."""
        from graphrag.graph.neo4j_client import Neo4jClient
        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[{"tombstoned": 0}])
        await client.tombstone_documents(["a.txt"], tenant="acme")
        assert "WHERE coalesce(d.is_deleted, false) = false" in client.run.call_args[0][0]


class TestGetDocumentStates:
    @pytest.mark.asyncio
    async def test_reports_hash_deletion_and_completion(self):
        from graphrag.graph.neo4j_client import Neo4jClient
        client = Neo4jClient.__new__(Neo4jClient)
        client.run = AsyncMock(return_value=[
            {"filename": "a.txt", "content_hash": "h1", "is_deleted": False, "ingest_complete": True},
            {"filename": "b.txt", "content_hash": "", "is_deleted": True, "ingest_complete": False},
        ])
        states = await client.get_document_states(tenant="acme")
        assert states["a.txt"] == {"content_hash": "h1", "is_deleted": False, "ingest_complete": True}
        assert states["b.txt"]["is_deleted"] is True
        assert client.run.call_args[1]["tenant"] == "acme"


class TestPartialIngestIsNotSkipped:
    """The subtle one: merge_document writes content_hash at the START of a
    document's write, so a run that crashes midway leaves a hash already
    matching the file on disk. Skipping on hash ALONE would freeze a
    half-ingested document out of every future run. The CLI therefore
    requires hash-match AND a completed prior write."""

    def test_skip_requires_both_conditions(self):
        h = compute_content_hash("same text")

        def safe_to_skip(state, current):
            return (not content_changed(state.get("content_hash", ""), current)
                    and state.get("ingest_complete", False))

        # hash matches but the previous write never finished -> must re-ingest
        assert safe_to_skip({"content_hash": h, "ingest_complete": False}, h) is False
        # hash matches and previous write completed -> safe to skip
        assert safe_to_skip({"content_hash": h, "ingest_complete": True}, h) is True
        # completed before, but the file changed -> must re-ingest
        assert safe_to_skip({"content_hash": "old", "ingest_complete": True}, h) is False
