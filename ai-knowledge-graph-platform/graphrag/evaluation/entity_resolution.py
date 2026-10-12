"""Deterministic entity-resolution evaluation for domain fixture sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from graphrag.graph.alias_registry import AmbiguousMatch

ExpectedDisposition = Literal["matched", "quarantined", "new"]


@dataclass(frozen=True)
class EntityResolutionCase:
    raw_name: str
    entity_type: str
    expected: ExpectedDisposition
    canonical_name: str = ""
    canonical_type: str = ""
    category: str = "general"


def evaluate_entity_resolution(
    cases: list[EntityResolutionCase],
    resolver: Callable[[str], tuple[str, str] | AmbiguousMatch | None],
) -> dict:
    """Score resolver outcomes without treating an expected new entity as error.

    Precision/recall measure correct automatic canonical matches only; review
    routing and correct-new decisions remain visible in the overall accuracy
    and disposition counts rather than being silently discarded.
    """
    results: list[dict] = []
    true_positive = false_positive = false_negative = 0
    disposition_counts = {"matched": 0, "quarantined": 0, "new": 0}

    for case in cases:
        resolved = resolver(case.raw_name)
        if isinstance(resolved, AmbiguousMatch):
            actual = "quarantined"
            canonical_name = resolved.candidate[0]
            canonical_type = resolved.candidate[1]
        elif resolved is None:
            actual = "new"
            canonical_name = ""
            canonical_type = ""
        else:
            actual = "matched"
            canonical_name, canonical_type = resolved
        disposition_counts[actual] += 1

        correct = (
            actual == case.expected
            and (actual != "matched" or (canonical_name, canonical_type) == (case.canonical_name, case.canonical_type))
        )
        if actual == "matched":
            if case.expected == "matched" and correct:
                true_positive += 1
            else:
                false_positive += 1
        elif case.expected == "matched":
            false_negative += 1
        error = _error_kind(case, actual, canonical_name, canonical_type)
        results.append({
            "raw_name": case.raw_name,
            "entity_type": case.entity_type,
            "category": case.category,
            "expected": case.expected,
            "actual": actual,
            "canonical_name": canonical_name,
            "canonical_type": canonical_type,
            "correct": correct,
            "error": error,
        })

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    return {
        "total": len(cases),
        "correct": sum(item["correct"] for item in results),
        "accuracy": sum(item["correct"] for item in results) / len(cases) if cases else 1.0,
        "auto_match_precision": precision,
        "auto_match_recall": recall,
        "auto_match_f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "dispositions": disposition_counts,
        "errors": _error_counts(results),
        "by_category": _by_category(results),
        "results": results,
    }


# Error taxonomy (additive to the original report). Different errors have different downstream cost:
#   wrong_link     - merged into the WRONG canonical entity (corrupts the graph; hardest to notice)
#   over_merge     - a genuinely new / distinct entity was merged into an existing one
#   missed_match   - a true duplicate was left as a new entity (fragmentation)
#   over_quarantine- a clear match or clear new entity was sent to review (reviewer load, not corruption)
#   missed_review  - an ambiguous mention was resolved without review (silent guess)
ERROR_KINDS = ("wrong_link", "over_merge", "missed_match", "over_quarantine", "missed_review")


def _error_kind(case: EntityResolutionCase, actual: str, name: str, type_: str) -> str | None:
    if case.expected == "matched":
        if actual == "matched":
            return None if (name, type_) == (case.canonical_name, case.canonical_type) else "wrong_link"
        return "missed_match" if actual == "new" else "over_quarantine"
    if case.expected == "new":
        return {"new": None, "matched": "over_merge", "quarantined": "over_quarantine"}[actual]
    return {"quarantined": None, "matched": "missed_review", "new": "missed_review"}[actual]


def _error_counts(results: list[dict]) -> dict[str, int]:
    return {kind: sum(1 for item in results if item["error"] == kind) for kind in ERROR_KINDS}


def _by_category(results: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for item in results:
        bucket = out.setdefault(item["category"], {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(item["correct"])
    return out


__all__ = ["ERROR_KINDS", "EntityResolutionCase", "evaluate_entity_resolution"]
