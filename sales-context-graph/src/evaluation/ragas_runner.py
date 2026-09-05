"""Run optional RAGAS metrics against a versioned golden JSONL dataset.

This module deliberately imports RAGAS and the judge client only inside
``run_ragas``. The application and its normal test suite must not require an
LLM judge, an API key, or outbound network access.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"question", "answer", "contexts", "ground_truth"}
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
CATEGORY_VALUES = {"answerable", "refusal"}
SAFE_REFUSAL_PREFIXES = (
    "i cannot",
    "i can't",
    "i can’t",
    "i am unable",
    "i'm unable",
    "i’m unable",
    "no citable",
    "insufficient",
    "not enough",
    "i need",
)
SAFE_REFUSAL_SUPPORT_CUES = (
    "provide",
    "before",
    "verify",
    "no citable",
    "insufficient",
    "not present",
    "not available",
    "not found",
    "does not include",
)


def load_golden(path: Path) -> list[dict[str, Any]]:
    """Load and validate a RAGAS-compatible JSONL dataset."""
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields: {sorted(missing)}")
        if not isinstance(row["contexts"], list) or not all(isinstance(item, str) for item in row["contexts"]):
            raise ValueError(f"{path}:{line_number}: contexts must be a list of strings")
        if "category" in row and row["category"] not in CATEGORY_VALUES:
            raise ValueError(f"{path}:{line_number}: category must be one of {sorted(CATEGORY_VALUES)}")
        if "constraints" in row:
            if not isinstance(row["constraints"], list):
                raise ValueError(f"{path}:{line_number}: constraints must be a list")
            for constraint in row["constraints"]:
                if not isinstance(constraint, dict) or constraint.get("metric") not in METRIC_NAMES:
                    raise ValueError(
                        f"{path}:{line_number}: each constraint must reference a known metric {sorted(METRIC_NAMES)}"
                    )
                if "min" not in constraint and "max" not in constraint:
                    raise ValueError(f"{path}:{line_number}: each constraint must declare min and/or max")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def _as_float(value: Any) -> float | None:
    """Normalize a RAGAS scalar/cell value without hiding NaN failures."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _mean(values: list[float | None]) -> float | None:
    """Return the mean of finite metric values, preserving missing data."""
    numeric = [value for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _safe_refusal_status(row: dict[str, Any]) -> bool | None:
    """Check the deterministic refusal contract for one labeled row.

    This deliberately checks only the refusal shape, not whether an LLM
    agrees with the answer. A refusal must lead with an explicit inability or
    missing-evidence cue and state what is missing or what should happen next.
    Non-refusal rows are excluded from the denominator.
    """
    if row.get("category") != "refusal":
        return None
    answer = str(row.get("answer", "")).strip().casefold()
    has_refusal_prefix = answer.startswith(SAFE_REFUSAL_PREFIXES)
    has_supporting_cue = any(cue in answer for cue in SAFE_REFUSAL_SUPPORT_CUES)
    return has_refusal_prefix and has_supporting_cue


def safe_refusal_metrics(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    """Return refusal-only deterministic safety metrics.

    ``None`` is returned when a dataset has no refusal cases, making an absent
    refusal slice distinguishable from a failing slice.
    """
    statuses = [_safe_refusal_status(row) for row in rows]
    refusal_statuses = [status for status in statuses if status is not None]
    return {
        "safe_refusal_rate": (
            sum(status is True for status in refusal_statuses) / len(refusal_statuses)
            if refusal_statuses
            else None
        ),
        "safe_refusal_cases": sum(status is True for status in refusal_statuses),
        "refusal_cases": len(refusal_statuses),
    }


def _constraint_status(row: dict[str, Any], scores: dict[str, float | None]) -> bool | None:
    """Check whether one row's declared constraints all hold simultaneously.

    Unlike ``_safe_refusal_status``, this needs the row's own computed
    per-metric scores (not just the row), since a constraint thresholds
    RAGAS-judge output rather than a deterministic string property.
    Fail-closed: a constraint whose metric has no score (``None``) counts
    as failed, not excluded -- a declared constraint should never silently
    pass. Rows with no ``constraints`` are excluded from the denominator,
    same convention as ``_safe_refusal_status``.
    """
    constraints = row.get("constraints")
    if not constraints:
        return None
    for constraint in constraints:
        value = scores.get(constraint["metric"])
        if value is None:
            return False
        if "min" in constraint and value < constraint["min"]:
            return False
        if "max" in constraint and value > constraint["max"]:
            return False
    return True


def constraints_metrics(
    rows: list[dict[str, Any]],
    per_metric: dict[str, list[float | None]],
) -> dict[str, int | float | None]:
    """Return constraint-only all-or-nothing metrics.

    This is what per-metric averaging can't show: how often a single answer
    satisfies every declared constraint at once, not just each one on
    average. ``None`` is returned when no row declares constraints, making
    an absent constraint slice distinguishable from a failing slice.
    """
    statuses = [
        _constraint_status(row, {name: values[index] for name, values in per_metric.items()})
        for index, row in enumerate(rows)
    ]
    applicable = [status for status in statuses if status is not None]
    return {
        "constraints_pass_rate": (
            sum(status is True for status in applicable) / len(applicable) if applicable else None
        ),
        "constraints_passed_cases": sum(status is True for status in applicable),
        "constraints_cases": len(applicable),
    }


def summarize_ragas(
    rows: list[dict[str, Any]],
    per_metric: dict[str, list[float | None]],
    *,
    model: str,
) -> dict[str, Any]:
    """Build stable overall, per-category, and per-row RAGAS output.

    Category reporting keeps answerable and evidence-refusal cases visible
    without silently dropping either from the overall benchmark.
    """
    per_row = [
        {
            "id": row.get("id", str(index)),
            "category": row.get("category", "unclassified"),
            "safe_refusal": _safe_refusal_status(row),
            "all_constraints_passed": _constraint_status(
                row, {name: values[index] for name, values in per_metric.items()}
            ),
            **{name: values[index] for name, values in per_metric.items()},
        }
        for index, row in enumerate(rows)
    ]
    metrics = {name: _mean(values) for name, values in per_metric.items()}
    categories: dict[str, dict[str, float | None]] = {}
    for category in sorted({item["category"] for item in per_row}):
        indexes = [index for index, item in enumerate(per_row) if item["category"] == category]
        categories[category] = {
            name: _mean([values[index] for index in indexes])
            for name, values in per_metric.items()
        }
    return {
        "model": model,
        "rows": len(rows),
        "metrics": metrics,
        **safe_refusal_metrics(rows),
        **constraints_metrics(rows, per_metric),
        "by_category": categories,
        "per_row": per_row,
    }


def run_ragas(
    rows: list[dict[str, Any]],
    *,
    model: str = "gpt-4o-mini",
    generations: int = 3,
) -> dict[str, Any]:
    """Evaluate rows with RAGAS and an OpenAI-compatible judge.

    RAGAS metrics are LLM-judge metrics; scores are evidence from the supplied
    dataset, not a replacement for deterministic grounding or recall tests.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to run the RAGAS judge")
    if not 1 <= generations <= 8:
        raise ValueError("generations must be between 1 and 8")
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:  # pragma: no cover - exercised in optional envs
        raise RuntimeError("Install optional evaluation dependencies with: pip install -e '.[eval]'") from exc

    dataset = Dataset.from_list(rows)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    # RAGAS requests multiple generations for several metrics. Request the
    # configured count explicitly; providers that return fewer generations are
    # surfaced by RAGAS as warnings rather than hidden in the output.
    judge = LangchainLLMWrapper(ChatOpenAI(model=model, temperature=0, n=generations))
    # Annotated Any because ragas types `evaluate()` as returning
    # `EvaluationResult | Executor`; only the former is subscriptable, and
    # mypy therefore rejects the `result[metric_name]` lookup below. The
    # lookup is already defensive at runtime -- it is wrapped in
    # `except (KeyError, TypeError, ValueError)` and falls back to None per
    # metric -- so the union is handled, just not in a way mypy can see.
    result: Any = evaluate(dataset, metrics=metrics, llm=judge)
    per_metric: dict[str, list[float | None]] = {}
    for metric_name in METRIC_NAMES:
        try:
            raw = result[metric_name]
            values = raw if isinstance(raw, (list, tuple)) else [raw]
            normalized = [_as_float(value) for value in values]
            per_metric[metric_name] = normalized
        except (KeyError, TypeError, ValueError):
            per_metric[metric_name] = [None] * len(rows)
    output = summarize_ragas(rows, per_metric, model=model)
    output["generations"] = generations
    return output


def main(input_path: Path, output_path: Path, *, model: str, generations: int = 3) -> None:
    result = run_ragas(load_golden(input_path), model=model, generations=generations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
