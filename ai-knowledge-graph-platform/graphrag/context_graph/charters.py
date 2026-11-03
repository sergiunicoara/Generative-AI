"""Use-case charters: per use case, the question set, metrics and thresholds.

Progress is judged against measures agreed for each use case rather than one
global score.  A charter (``evals/charters/*.yml``) names the question set, the
split it belongs to, who owns the labels, who set the thresholds, and for
each metric a minimum or maximum.  The runner reports every metric with its
denominator and fails the use case when any threshold is missed.

The thresholds belong to the domain experts who own the use case, not to
this repository: a charter without a named label owner, or without the
source of its thresholds, is rejected.  The bundled example charter is
explicitly synthetic and says so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphrag.context_graph.knowledge_metrics import evaluate_cases


class Threshold(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _one_bound(self):
        if self.min is None and self.max is None:
            raise ValueError("a threshold needs min or max")
        return self


class Charter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_\-]{0,63}$")
    use_case: str = Field(min_length=1, max_length=500)
    kind: Literal["contextual_knowledge"] = "contextual_knowledge"
    question_set: str = Field(min_length=1)
    split: Literal["development", "held_out", "generated"]
    label_owner: str = Field(min_length=1, max_length=256)
    threshold_source: str = Field(min_length=1, max_length=500)
    metrics: dict[str, Threshold] = Field(min_length=1)

    @model_validator(mode="after")
    def _owners(self):
        if not self.label_owner.strip() or not self.threshold_source.strip():
            raise ValueError(f"charter {self.id}: a named label owner and threshold source are required")
        return self


class CharterError(ValueError):
    """A charter is malformed or names an unknown question set or metric."""


def load_charter(path: Path) -> Charter:
    try:
        return Charter.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise CharterError(f"{path.name}: {exc}") from exc


def _metric_value(report: dict, name: str) -> tuple[float | None, int | None]:
    if name == "time_to_answer_p95_ms":
        return report["time_to_answer_ms"]["p95"], report["time_to_answer_ms"]["cases"]
    if name == "cases_passed_ratio":
        total = report["cases"]["total"]
        return (report["cases"]["passed"] / total if total else None), total
    if name not in report["metrics"]:
        raise CharterError(f"unknown metric {name!r}")
    metric = report["metrics"][name]
    return metric["value"], metric["denominator"]


def run_charter(charter: Charter, *, root: Path) -> dict:
    question_set = root / charter.question_set
    if not question_set.is_file():
        raise CharterError(f"charter {charter.id}: question set {charter.question_set} not found")
    report = evaluate_cases(question_set, root=root)
    if report["split"] != charter.split:
        raise CharterError(f"charter {charter.id}: question set is split {report['split']!r}, "
                           f"charter says {charter.split!r}")
    results = {}
    for name, threshold in sorted(charter.metrics.items()):
        value, denominator = _metric_value(report, name)
        ok = value is not None and (threshold.min is None or value >= threshold.min) and (
            threshold.max is None or value <= threshold.max)
        results[name] = {"value": value, "denominator": denominator, "min": threshold.min, "max": threshold.max,
                         "passed": ok}
    return {
        "charter": charter.id, "use_case": charter.use_case, "split": charter.split,
        "question_set": charter.question_set, "label_owner": charter.label_owner,
        "threshold_source": charter.threshold_source, "label": report["label"],
        "passed": all(r["passed"] for r in results.values()), "metrics": results,
    }


__all__ = ["Charter", "CharterError", "Threshold", "load_charter", "run_charter"]
