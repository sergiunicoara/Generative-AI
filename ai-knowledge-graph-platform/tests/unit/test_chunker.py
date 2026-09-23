"""graphrag.ingestion.chunker — section-aware splitting.

No test file for this module existed before this one, despite it being the
first stage of the ingestion pipeline. These cases pin down a real defect
found and fixed in the aerospace corpus: `_HEADING_RE` only recognised
markdown and numbered headings, so any document using plain un-numbered
ALL-CAPS section titles (routine in compliance/regulatory reports) fell
entirely to naive fixed-size splitting — see CON-02 in evals/golden_set.json
for the retrieval failure this caused.
"""

from __future__ import annotations

from datetime import datetime, timezone

from graphrag.core.models import Document
from graphrag.ingestion.chunker import _HEADING_RE, chunk_document


def _doc(raw_text: str) -> Document:
    return Document(
        filename="test.txt",
        source_path="test.txt",
        raw_text=raw_text,
        tenant="test",
        ingested_at=datetime.now(timezone.utc),
    )


class TestHeadingDetection:
    def test_matches_markdown_heading(self):
        assert _HEADING_RE.search("## Overview\n")

    def test_matches_numbered_all_caps_heading(self):
        assert _HEADING_RE.search("3. BUDGET & KPI TARGETS\n")

    def test_matches_numbered_heading_with_a_decimal(self):
        """Regression: the numbered branch's char class previously excluded
        '.', so "3. CHANGES IN VERSION 2.0" (a real corpus heading) silently
        failed to match — a pre-existing gap fixed alongside the un-numbered
        branch."""
        assert _HEADING_RE.search("3. CHANGES IN VERSION 2.0\n")

    def test_matches_plain_all_caps_title(self):
        assert _HEADING_RE.search("CRITICAL FINDING\n")

    def test_matches_all_caps_title_with_trailing_colon(self):
        assert _HEADING_RE.search("REQUIRED ACTIONS:\n")

    def test_matches_all_caps_title_with_em_dash(self):
        assert _HEADING_RE.search("CRITICAL FINDING — NON-COMPLIANCE IDENTIFIED\n")

    def test_rejects_prose_sentence(self):
        assert not _HEADING_RE.match("FAA has approved an AMOC for this product.")

    def test_rejects_all_caps_metadata_line(self):
        """'KEY: value' (trailing content after the colon) must not be treated
        as a section title — only a bare title ending in ':' should match."""
        assert not _HEADING_RE.match("MSN: 44567")

    def test_rejects_bullet_list_item(self):
        assert not _HEADING_RE.match("- 737-700, 737-800, 737-900ER")


class TestSectionSoftCap:
    def test_modestly_oversized_section_stays_whole(self):
        """A section slightly over chunk_size must not be split at whatever
        paragraph boundary happens to fall nearest the budget — that boundary
        is often between two paragraphs that belong together (see CON-02:
        a compliance status line and the sentence immediately qualifying it,
        split into separate chunks that never reach the LLM together)."""
        heading = "CRITICAL FINDING\n\n"
        # ~150 chars of "status" content + ~150 chars of qualifying "note" —
        # comfortably over chunk_size=512 once combined with padding below,
        # but well under a 1.6x soft cap.
        status = "Status: IS_NON_COMPLIANT_WITH. " + ("Detail line. " * 20)
        note = "NOTE: non-compliance is anticipated; the aircraft remains airworthy until the deadline. " * 3
        section = heading + status + "\n\n" + note
        assert 512 < len(section) <= 512 * 1.6, f"fixture must land in the soft-cap gap, got {len(section)} chars"

        doc = _doc(section)
        chunks = chunk_document(doc)

        assert len(chunks) == 1
        assert "IS_NON_COMPLIANT_WITH" in chunks[0].text
        assert "remains airworthy" in chunks[0].text

    def test_genuinely_long_section_still_gets_split(self):
        """The soft cap must not disable splitting entirely — a section far
        beyond the cap still needs to be broken up."""
        heading = "APPENDIX\n\n"
        body = "This is one sentence of filler content. " * 100  # ~4200 chars
        doc = _doc(heading + body)

        chunks = chunk_document(doc)

        assert len(chunks) > 1
        assert all(len(c.text) <= 512 * 1.6 + 100 for c in chunks)  # small margin for prepended heading

    def test_heading_is_prepended_to_each_split_piece(self):
        heading = "APPENDIX\n\n"
        body = "This is one sentence of filler content. " * 100
        doc = _doc(heading + body)

        chunks = chunk_document(doc)

        # Doc-identity prefix (see TestDocumentIdentityPrefix) comes first,
        # heading immediately after it.
        assert all(c.text.startswith("[test]\n\nAPPENDIX") for c in chunks)


class TestDocumentIdentityPrefix:
    """A chunk that only ever refers to its own document as "this AD" is
    unsearchable on that document's ID otherwise — see INF-01 in
    evals/golden_set.json, root-caused in docs/audit-2026-08-13.md."""

    def test_every_chunk_starts_with_the_document_label(self):
        doc = Document(
            filename="FAA-AD-2024-01-02.txt",
            source_path="FAA-AD-2024-01-02.txt",
            raw_text="REFERENCES:\n\n- AD 2020-05-11: fully superseded by this AD.",
            tenant="test",
            ingested_at=datetime.now(timezone.utc),
        )

        chunks = chunk_document(doc)

        assert all(c.text.startswith("[FAA-AD-2024-01-02]") for c in chunks)

    def test_label_is_the_filename_without_extension(self):
        doc = Document(
            filename="G-ABCD_AD_compliance_2024-03.txt",
            source_path="G-ABCD_AD_compliance_2024-03.txt",
            raw_text="Some content.",
            tenant="test",
            ingested_at=datetime.now(timezone.utc),
        )

        chunks = chunk_document(doc)

        assert chunks[0].text.startswith("[G-ABCD_AD_compliance_2024-03]")
        assert ".txt" not in chunks[0].text.split("\n", 1)[0]

    def test_prefix_does_not_hide_the_original_content(self):
        doc = _doc("Filler prose with no section titles at all. " * 40)

        chunks = chunk_document(doc)

        assert all("Filler prose" in c.text for c in chunks)


class TestUnheadedDocument:
    def test_document_with_no_headings_falls_back_to_fixed_size_splitting(self):
        body = "Filler prose with no section titles at all. " * 40
        doc = _doc(body)

        chunks = chunk_document(doc)

        assert len(chunks) >= 1
        assert all(len(c.text) <= 512 * 1.6 for c in chunks)


class TestChunkIdIsStableAcrossReingestion:
    """Regression coverage for the 2026-09-23 audit finding: Chunk.id used to
    default to a fresh uuid4() on every call, so re-chunking the same
    document (a re-ingest, or a queue retry re-running extract()) produced
    chunks whose ids matched nothing already in the graph, and every
    MENTIONS edge for that document silently failed to write
    (merge_mentions_batch MATCHes Chunk by id -- see neo4j_client.py).

    Chunk.id must be seeded from the document's real identity -- (tenant,
    filename), matching Neo4jClient.merge_document()'s own natural key --
    not from document.id, which is itself a fresh uuid4() at extract() time
    and only becomes stable after write_document() resolves it later in the
    pipeline (see ingestion_agent.write()).
    """

    def test_same_document_rechunked_twice_gets_identical_chunk_ids(self):
        # Two separate Document instances for "the same file" -- exactly what
        # a second extract() run builds on re-ingest: same tenant/filename,
        # but a fresh, different document.id each time (the default_factory).
        first = _doc("## Section One\n\nSome content.\n\n## Section Two\n\nMore content.")
        second = _doc("## Section One\n\nSome content, revised.\n\n## Section Two\n\nMore content.")
        assert first.id != second.id  # sanity: these really are different uuids

        chunks_first = chunk_document(first)
        chunks_second = chunk_document(second)

        assert len(chunks_first) == len(chunks_second)
        assert [c.id for c in chunks_first] == [c.id for c in chunks_second]

    def test_different_documents_get_different_chunk_ids(self):
        doc_a = _doc("## Section\n\nContent A.")
        doc_b = Document(
            filename="other.txt", source_path="other.txt", raw_text=doc_a.raw_text,
            tenant="test", ingested_at=datetime.now(timezone.utc),
        )

        chunks_a = chunk_document(doc_a)
        chunks_b = chunk_document(doc_b)

        assert chunks_a[0].id != chunks_b[0].id

    def test_chunk_id_does_not_depend_on_document_id(self):
        """The id most available at chunk_document() call time -- document.id
        -- is exactly the one that must NOT be used, since it is not yet the
        canonical id a re-ingest will resolve to."""
        doc = _doc("## Section\n\nContent.")
        chunks = chunk_document(doc)

        doc.id = "a-completely-different-id"
        chunks_after_id_change = chunk_document(doc)

        assert [c.id for c in chunks] == [c.id for c in chunks_after_id_change]
