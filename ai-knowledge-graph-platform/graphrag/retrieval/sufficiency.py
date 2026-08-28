"""Deterministic retrieval-sufficiency assessment before answer synthesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalSufficiency:
    sufficient: bool
    reason_code: str
    evidence_count: int
    source_count: int
    average_score: float
    conflict_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_retrieval_sufficiency(
    *,
    chunks: list[dict],
    citations: list[str],
    conflicts: list[dict],
    min_evidence: int = 1,
    min_average_score: float = 0.0,
) -> RetrievalSufficiency:
    """Decide whether retrieved evidence clears a conservative synthesis gate.

    Citations count as evidence for global/community retrieval, which has no
    local chunks.  Scores are intentionally advisory: a non-zero threshold is
    opt-in because scorer scales differ by retrieval source.
    """
    evidence_count = max(len(chunks), len(citations))
    sources = {
        str(chunk.get("source") or chunk.get("_doc_name") or "")
        for chunk in chunks
        if chunk.get("source") or chunk.get("_doc_name")
    }
    source_count = len(sources) if sources else len(set(citations))
    scores = [
        max(0.0, float(chunk.get("final_score", chunk.get("rerank_score", chunk.get("score", 0.0))) or 0.0))
        for chunk in chunks
    ]
    average_score = sum(scores) / len(scores) if scores else 0.0
    if conflicts:
        reason = "unresolved_conflict"
    elif evidence_count < max(1, min_evidence):
        reason = "insufficient_evidence"
    elif average_score < max(0.0, min_average_score):
        reason = "low_evidence_score"
    else:
        reason = "sufficient"
    return RetrievalSufficiency(
        sufficient=reason == "sufficient",
        reason_code=reason,
        evidence_count=evidence_count,
        source_count=source_count,
        average_score=average_score,
        conflict_count=len(conflicts),
    )


def abstention_message(reason_code: str) -> str:
    messages = {
        "unresolved_conflict": "I can’t provide a grounded answer because the retrieved evidence contains an unresolved conflict.",
        "low_evidence_score": "I can’t provide a grounded answer because the retrieved evidence did not meet the configured confidence threshold.",
        "insufficient_evidence": "I can’t provide a grounded answer because no sufficient authorized evidence was retrieved.",
    }
    return messages.get(reason_code, "I can’t provide a grounded answer from the retrieved evidence.")


__all__ = ["RetrievalSufficiency", "assess_retrieval_sufficiency", "abstention_message"]
