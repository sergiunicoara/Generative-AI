"""Transparent comparison metrics with explicit handling for unavailable results."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

from .schemas import EvaluationCase, EvaluatorResult


def case_accounting(
    cases: Iterable[EvaluationCase], results: Iterable[EvaluatorResult], evaluator: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Map every scored case to exactly one TP/TN/FP/FN outcome."""
    by_case: dict[str, EvaluatorResult] = {}
    for result in results:
        if result.evaluator == evaluator and result.status == "ok" and result.detected_failure is not None:
            by_case[result.case_id] = result

    rows: list[dict[str, Any]] = []
    totals = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "unscored": 0}
    for case in cases:
        result = by_case.get(case.case_id)
        prediction = result.detected_failure if result else None
        outcome: str | None
        if prediction is None:
            outcome = None
            totals["unscored"] += 1
        elif case.expected_failure and prediction:
            outcome = "TP"
            totals["TP"] += 1
        elif not case.expected_failure and not prediction:
            outcome = "TN"
            totals["TN"] += 1
        elif not case.expected_failure and prediction:
            outcome = "FP"
            totals["FP"] += 1
        else:
            outcome = "FN"
            totals["FN"] += 1
        rows.append(
            {
                "case_id": case.case_id,
                "historical_or_synthetic": "synthetic" if case.synthetic else "historical",
                "category": case.category,
                "ground_truth_failure": case.expected_failure,
                "ground_truth_failure_type": "; ".join(case.expected_failure_types),
                "limina_prediction": prediction if evaluator == "limina" else None,
                "limina_failure_type": "; ".join(result.failure_types) if evaluator == "limina" and result else "",
                "evaluator_prediction": prediction,
                "evaluator_failure_type": "; ".join(result.failure_types) if result else "",
                "correct": outcome in {"TP", "TN"},
                "TP_TN_FP_FN": outcome or "unscored",
            }
        )
    totals["scored"] = totals["TP"] + totals["TN"] + totals["FP"] + totals["FN"]
    totals["dataset_size"] = len(rows)
    if totals["scored"] + totals["unscored"] != totals["dataset_size"]:
        raise AssertionError("Case accounting totals do not equal the dataset size")
    return rows, totals


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def confusion_metrics(cases: Iterable[EvaluationCase], results: Iterable[EvaluatorResult]) -> dict[str, Any]:
    """Return binary detection metrics only for cases with an actual evaluation."""
    expected = {case.case_id: case.expected_failure for case in cases}
    tp = tn = fp = fn = evaluated = 0
    for result in results:
        if result.status != "ok" or result.detected_failure is None or result.case_id not in expected:
            continue
        evaluated += 1
        actual, prediction = expected[result.case_id], result.detected_failure
        if actual and prediction:
            tp += 1
        elif not actual and not prediction:
            tn += 1
        elif not actual and prediction:
            fp += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    accuracy = (tp + tn) / evaluated if evaluated else None
    specificity = tn / (tn + fp) if tn + fp else None
    false_positive_rate = fp / (fp + tn) if fp + tn else None
    false_negative_rate = fn / (fn + tp) if fn + tp else None
    return {
        "evaluated_cases": evaluated,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def category_metrics(cases: Iterable[EvaluationCase], results: Iterable[EvaluatorResult]) -> dict[str, dict[str, Any]]:
    by_category: dict[str, list[EvaluationCase]] = defaultdict(list)
    for case in cases:
        by_category[case.category].append(case)
    by_id: dict[str, list[EvaluatorResult]] = defaultdict(list)
    for result in results:
        by_id[result.case_id].append(result)
    out: dict[str, dict[str, Any]] = {}
    for category, category_cases in sorted(by_category.items()):
        ids = {case.case_id for case in category_cases}
        subset = [result for case_id in ids for result in by_id.get(case_id, [])]
        out[category] = confusion_metrics(category_cases, subset)
    return out


def latency_cost_metrics(results: Iterable[EvaluatorResult]) -> dict[str, float | None]:
    complete = [result for result in results if result.status == "ok"]
    latencies = [result.latency_ms for result in complete if result.latency_ms is not None]
    costs = [result.estimated_cost_usd for result in complete if result.estimated_cost_usd is not None]
    return {
        "mean_latency_ms": mean(latencies) if latencies else None,
        "p50_latency_ms": _quantile(latencies, 0.50),
        "p95_latency_ms": _quantile(latencies, 0.95),
        "min_latency_ms": min(latencies) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
        "mean_cost_usd": mean(costs) if costs else None,
        "estimated_cost_per_1000_usd": mean(costs) * 1000 if costs else None,
    }


def repeatability_metrics(results: Iterable[EvaluatorResult]) -> dict[str, Any]:
    """Measure exact raw output, class, score, and failure-type consistency by case."""
    grouped: dict[str, list[EvaluatorResult]] = defaultdict(list)
    for result in results:
        if result.status == "ok":
            grouped[result.case_id].append(result)
    exact_values: list[float] = []
    class_values: list[float] = []
    score_variances: list[float] = []
    latency_variances: list[float] = []
    type_values: list[float] = []
    for repeats in grouped.values():
        if len(repeats) < 2:
            continue
        raw_fingerprints = {
            hashlib.sha256(json.dumps(item.raw_result, sort_keys=True, default=str).encode()).hexdigest()
            for item in repeats
        }
        classes = {item.detected_failure for item in repeats}
        types = {tuple(sorted(item.failure_types)) for item in repeats}
        scores = [item.score for item in repeats if item.score is not None]
        latencies = [item.latency_ms for item in repeats if item.latency_ms is not None]
        exact_values.append(1.0 if len(raw_fingerprints) == 1 else 0.0)
        class_values.append(1.0 if len(classes) == 1 else 0.0)
        type_values.append(1.0 if len(types) == 1 else 0.0)
        if len(scores) > 1:
            score_variances.append(sum((score - mean(scores)) ** 2 for score in scores) / len(scores))
        if len(latencies) > 1:
            latency_variances.append(sum((latency - mean(latencies)) ** 2 for latency in latencies) / len(latencies))
    return {
        "repeated_cases": len(exact_values),
        "exact_consistency": mean(exact_values) if exact_values else None,
        "classification_consistency": mean(class_values) if class_values else None,
        "failure_type_consistency": mean(type_values) if type_values else None,
        "mean_score_variance": mean(score_variances) if score_variances else None,
        "mean_latency_variance_ms2": mean(latency_variances) if latency_variances else None,
    }


def summarize(cases: list[EvaluationCase], results: list[EvaluatorResult]) -> dict[str, Any]:
    evaluators = sorted({result.evaluator for result in results})
    return {
        "case_count": len(cases),
        "source_counts": {source: sum(case.source == source for case in cases) for source in ("historical", "synthetic")},
        "categories": category_metrics(cases, results),
        "confusion": confusion_metrics(cases, results),
        "latency_and_cost": latency_cost_metrics(results),
        "repeatability": repeatability_metrics(results),
        "by_evaluator": {
            evaluator: {
                "confusion": confusion_metrics(cases, [r for r in results if r.evaluator == evaluator]),
                "categories": category_metrics(cases, [r for r in results if r.evaluator == evaluator]),
                "latency_and_cost": latency_cost_metrics([r for r in results if r.evaluator == evaluator]),
                "repeatability": repeatability_metrics([r for r in results if r.evaluator == evaluator]),
            }
            for evaluator in evaluators
        },
        "result_statuses": {
            status: sum(result.status == status for result in results)
            for status in ("ok", "skipped", "error")
        },
    }
