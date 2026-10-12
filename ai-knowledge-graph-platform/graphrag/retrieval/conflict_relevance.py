"""Question-relevant conflict signal for the sufficiency gate (docs/REMAINING_AUDIT_BACKLOG.md #2).

``assess_retrieval_sufficiency`` treats *any* open ``Conflict`` reachable from the retrieved entities as a
reason to abstain. With a corpus that is seeded with contradictions that abstains on nearly every query
(automotive golden set 7/10 -> 2/10 when the gate was enabled), even when the conflict has nothing to do
with what was asked.

This module narrows the set of conflicts that may block an answer to those that plausibly affect it:

* the question names one of the conflict's entities, or
* the conflict's entity appears in the text of the top-ranked chunks the answer is built from AND at least
  one of the conflict's source documents is among those chunks' documents.

Anything that cannot be shown to be unrelated stays relevant (a conflict with no recorded source documents,
an unparsable ``sources`` value). The filter only ever *removes* conflicts from the abstention decision; the
conflict annotation shown to the model and the policy trace still use the full list.

Status: implemented, unit-tested, **off by default** (``conflict_relevance_filter_enabled``). It has not
been validated against the golden sets, so no improvement is claimed.
"""
from __future__ import annotations

import ast
import re
from typing import Any

_EXT = re.compile(r"\.(txt|pdf|md|docx?|html?)$", re.I)


def _norm(value: Any) -> str:
    return _EXT.sub("", str(value or "").strip()).casefold()


def conflict_document_keys(raw: Any) -> set[str]:
    """Normalised document ids / names recorded on a conflict (``Conflict.sources``)."""
    try:
        parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return set()
    values = parsed.values() if isinstance(parsed, dict) else (parsed or [])
    keys: set[str] = set()
    for value in values:
        for item in (value if isinstance(value, (list, tuple, set)) else [value]):
            if isinstance(item, str) and item.strip():
                keys.add(_norm(item))
    return keys


def _chunk_score(chunk: dict) -> float:
    for field in ("final_score", "rerank_score", "score"):
        value = chunk.get(field)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _top_chunks(chunks: list[dict], limit: int) -> list[dict]:
    # Stable: equal scores keep retrieval order.
    return sorted(chunks, key=_chunk_score, reverse=True)[: max(1, int(limit))] if chunks else []


def _chunk_doc_keys(chunk: dict) -> set[str]:
    return {_norm(chunk.get(f)) for f in ("document_id", "_doc_name", "source") if chunk.get(f)}


def filter_relevant_conflicts(
    conflicts: list[dict],
    *,
    question: str,
    chunks: list[dict],
    max_ranked_chunks: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Split ``conflicts`` into (relevant, ignored) for this question and evidence."""
    asked = (question or "").casefold()
    top = _top_chunks(chunks or [], max_ranked_chunks)
    evidence_text = " ".join(str(c.get("text") or "") for c in top).casefold()
    evidence_docs: set[str] = set()
    for chunk in top:
        evidence_docs |= _chunk_doc_keys(chunk)

    relevant: list[dict] = []
    ignored: list[dict] = []
    for conflict in conflicts or []:
        entities = [str(conflict.get(f)).casefold() for f in ("src", "tgt") if conflict.get(f)]
        sources = conflict_document_keys(conflict.get("sources"))
        if any(e in asked for e in entities):
            relevant.append(conflict)
        elif not sources:
            relevant.append(conflict)              # cannot show it is unrelated
        elif sources & evidence_docs and any(e in evidence_text for e in entities):
            relevant.append(conflict)
        else:
            ignored.append(conflict)
    return relevant, ignored


__all__ = ["conflict_document_keys", "filter_relevant_conflicts"]
