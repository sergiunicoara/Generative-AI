"""Schema and failure taxonomy for domain evaluation sets."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from pathlib import Path
import json


class FailureCategory(StrEnum):
    EXTRACTION = "extraction"
    ENTITY_RESOLUTION = "entity_resolution"
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    CITATION = "citation"
    GENERATION = "generation"


REQUIRED_CASE_FIELDS = {"id", "type", "question", "expected_citations"}
ALLOWED_CASE_TYPES = {
    "single_hop", "multi_hop", "contradiction", "negative", "reasoning",
    "authority_chain",
}


def validate_dataset(data: dict, *, source: str = "dataset") -> dict:
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append("root must be an object")
        return {"valid": False, "errors": errors, "count": 0}
    if not isinstance(data.get("tenant"), str) or not data["tenant"]:
        errors.append("tenant is required")
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        errors.append("questions must be a non-empty list")
        questions = []
    seen: set[str] = set()
    for index, case in enumerate(questions):
        if not isinstance(case, dict):
            errors.append(f"questions[{index}] must be an object")
            continue
        missing = REQUIRED_CASE_FIELDS - set(case)
        if missing:
            errors.append(f"questions[{index}] missing {sorted(missing)}")
        case_id = case.get("id")
        if case_id in seen:
            errors.append(f"duplicate case id '{case_id}'")
        seen.add(case_id)
        if case.get("type") not in ALLOWED_CASE_TYPES:
            errors.append(f"questions[{index}] has invalid type '{case.get('type')}'")
        if not isinstance(case.get("expected_citations"), list):
            errors.append(f"questions[{index}].expected_citations must be a list")
        for field in ("expected_surfaces", "expected_evidence_ids", "expected_graph_edges"):
            if field in case and not isinstance(case[field], list):
                errors.append(f"questions[{index}].{field} must be a list")
        if "tool_budget" in case and (
            not isinstance(case["tool_budget"], int) or case["tool_budget"] < 0
        ):
            errors.append(f"questions[{index}].tool_budget must be a non-negative integer")
    return {"valid": not errors, "errors": errors, "count": len(questions), "source": source}


def classify_failure(
    *,
    extraction_ok: bool = True,
    entity_resolution_ok: bool = True,
    retrieval_ok: bool = True,
    reasoning_ok: bool = True,
    citation_ok: bool = True,
    generation_ok: bool = True,
) -> FailureCategory | None:
    """Return the first failed pipeline layer, making eval failures actionable."""
    for category, passed in (
        (FailureCategory.EXTRACTION, extraction_ok),
        (FailureCategory.ENTITY_RESOLUTION, entity_resolution_ok),
        (FailureCategory.RETRIEVAL, retrieval_ok),
        (FailureCategory.REASONING, reasoning_ok),
        (FailureCategory.CITATION, citation_ok),
        (FailureCategory.GENERATION, generation_ok),
    ):
        if not passed:
            return category
    return None


def summarize_failures(results: list[dict]) -> dict:
    counts = Counter(r.get("failure_category") for r in results if r.get("failure_category"))
    return {"total": len(results), "passed": sum(not r.get("failure_category") for r in results), "by_category": dict(counts)}


def load_and_validate(path: str | Path) -> dict:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return validate_dataset(data, source=str(p))
