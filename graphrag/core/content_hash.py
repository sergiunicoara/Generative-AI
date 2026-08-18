"""Content hashing for incremental ingestion.

Lets the pipeline answer "did this source document actually change?" without
re-chunking, re-embedding and re-calling the LLM extractor for a file whose
bytes are identical to what was ingested last time.

This is not only a cost optimisation. Before content hashing, the bulk
ingest CLI's checkpoint was binary: a Document node carrying
``ingest_complete = true`` was skipped on every subsequent run, so a source
file that had been EDITED was never re-ingested at all unless the operator
passed ``--wipe`` and rebuilt the whole tenant. Comparing hashes turns that
into the correct three-way decision: unchanged -> skip, changed ->
re-ingest, new -> ingest.
"""

from __future__ import annotations

import hashlib

# Marker for "no hash recorded" — documents ingested before content hashing
# existed. Comparison code must treat this as "unknown, assume changed"
# rather than as a real hash that could accidentally match another document.
NO_HASH = ""


def compute_content_hash(text: str) -> str:
    """Return the sha256 hex digest of `text`.

    Hashes the decoded text rather than the raw file bytes on purpose: the
    same document re-saved with different line endings or a different
    encoding is the same document for extraction purposes, and re-running a
    multi-minute LLM extraction because a file picked up CRLF would be a
    false positive. `document_loader.load_document` already normalises to
    str for every supported format (PDF/DOCX/TXT/MD), so this is the only
    representation all of them share.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_changed(stored_hash: str, current_hash: str) -> bool:
    """True if the document should be re-ingested.

    An absent stored hash (pre-migration data) returns True — "unknown"
    must mean "re-ingest to be safe", never "assume unchanged", or a
    document that predates hashing would be frozen out of every future
    ingest exactly the way the old binary checkpoint froze edited files.
    """
    if not stored_hash or stored_hash == NO_HASH:
        return True
    return stored_hash != current_hash
