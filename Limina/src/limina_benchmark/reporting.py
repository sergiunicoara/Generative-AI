"""Write raw JSON/CSV and a deliberately evidence-limited Markdown report."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import summarize
from .metrics import case_accounting
from .schemas import EvaluationCase, EvaluatorResult


def _display(value: Any) -> str:
    return "not measured" if value is None else str(value)


def write_artifacts(cases: list[EvaluationCase], results: list[EvaluatorResult], destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    summary = summarize(cases, results)
    raw_path = destination / "benchmark.json"
    csv_path = destination / "benchmark.csv"
    report_path = destination / "summary.md"
    raw_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
                "cases": [case.model_dump(mode="json") for case in cases],
                "results": [result.model_dump(mode="json") for result in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["evaluator", "case_id", "status", "detected_failure", "failure_types", "score", "latency_ms", "estimated_cost_usd", "error"])
        writer.writeheader()
        for result in results:
            writer.writerow({
                **result.model_dump(exclude={"raw_result"}),
                "failure_types": "; ".join(result.failure_types),
            })
    confusion = summary["confusion"]
    latency = summary["latency_and_cost"]
    evaluator_sections = ""
    for evaluator, values in summary["by_evaluator"].items():
        ev_confusion = values["confusion"]
        ev_latency = values["latency_and_cost"]
        ev_repeat = values["repeatability"]
        category_lines = "".join(
            f"- {category}: recall={_display(category_values['recall'])}, FPR={_display(category_values['false_positive_rate'])}, n={category_values['evaluated_cases']}\n"
            for category, category_values in values["categories"].items()
        )
        evaluator_sections += (
            f"### {evaluator}\n\n"
            f"TP/TN/FP/FN: {ev_confusion['tp']}/{ev_confusion['tn']}/{ev_confusion['fp']}/{ev_confusion['fn']}; "
            f"precision={_display(ev_confusion['precision'])}, recall={_display(ev_confusion['recall'])}, "
            f"F1={_display(ev_confusion['f1'])}, accuracy={_display(ev_confusion['accuracy'])}, "
            f"specificity={_display(ev_confusion['specificity'])}, FPR={_display(ev_confusion['false_positive_rate'])}, "
            f"FNR={_display(ev_confusion['false_negative_rate'])}.\n\n"
            f"Latency mean/p50/p95/min/max: {_display(ev_latency['mean_latency_ms'])}/{_display(ev_latency['p50_latency_ms'])}/{_display(ev_latency['p95_latency_ms'])}/{_display(ev_latency['min_latency_ms'])}/{_display(ev_latency['max_latency_ms'])} ms. "
            f"Repeatability exact/classification/type: {_display(ev_repeat['exact_consistency'])}/{_display(ev_repeat['classification_consistency'])}/{_display(ev_repeat['failure_type_consistency'])}; "
            f"score variance={_display(ev_repeat['mean_score_variance'])}, latency variance={_display(ev_repeat['mean_latency_variance_ms2'])}.\n\n"
            f"Category detection:\n{category_lines}\n"
        )
    report_path.write_text(
        "# Recruiter Agent Limina Benchmark\n\n"
        "## Evidence status\n\n"
        f"- Cases: {summary['case_count']} ({summary['source_counts']['historical']} historical, {summary['source_counts']['synthetic']} synthetic).\n"
        f"- Evaluator results: ok={summary['result_statuses']['ok']}, skipped={summary['result_statuses']['skipped']}, error={summary['result_statuses']['error']}.\n"
        "- This report never substitutes missing evaluator output with a score, cost, or verdict.\n\n"
        "## Overall comparison\n\n"
        "| TP | TN | FP | FN | Precision | Recall | F1 | Accuracy | Specificity | FPR | FNR |\n"
        "| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |\n"
        f"| {confusion['tp']} | {confusion['tn']} | {confusion['fp']} | {confusion['fn']} | {_display(confusion['precision'])} | {_display(confusion['recall'])} | {_display(confusion['f1'])} | {_display(confusion['accuracy'])} | {_display(confusion['specificity'])} | {_display(confusion['false_positive_rate'])} | {_display(confusion['false_negative_rate'])} |\n\n"
        "## Per-evaluator results\n\n"
        + evaluator_sections
        + "## Latency and cost\n\n"
        + f"Mean latency: {_display(latency['mean_latency_ms'])} ms; p50: {_display(latency['p50_latency_ms'])} ms; p95: {_display(latency['p95_latency_ms'])} ms.\n\n"
        + f"Mean estimated cost: {_display(latency['mean_cost_usd'])} USD; estimated / 1,000: {_display(latency['estimated_cost_per_1000_usd'])} USD.\n\n"
        + "## Limitation and recommendation\n\n"
        + "The imported Recruiter Agent history contains only healthy, passing golden cases. It cannot establish failure-detection recall, evaluator superiority, repeatability, or a production architecture recommendation. Capture and independently label failures from each documented category, then run both evaluators on the same sanitized trajectories. Until then, retain the existing judge and use deterministic checks for executable state/tool-policy assertions; do not claim Limina is superior or deterministic.\n",
        encoding="utf-8",
    )
    return {"json": raw_path, "csv": csv_path, "summary": report_path}


def write_case_accounting(
    cases: list[EvaluationCase], results: list[EvaluatorResult], evaluator: str, destination: Path
) -> dict[str, Path]:
    """Persist auditable per-case classification rows and verify their total."""
    destination.mkdir(parents=True, exist_ok=True)
    rows, totals = case_accounting(cases, results, evaluator)
    csv_path = destination / f"{evaluator}_case_accounting.csv"
    markdown_path = destination / f"{evaluator}_case_accounting.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    table_rows = "\n".join(
        "| {case_id} | {historical_or_synthetic} | {category} | {ground_truth_failure} | {ground_truth_failure_type} | {limina_prediction} | {limina_failure_type} | {correct} | {TP_TN_FP_FN} |".format(**row)
        for row in rows
    )
    markdown_path.write_text(
        "# Case accounting\n\n"
        f"Evaluator: `{evaluator}`. Dataset size={totals['dataset_size']}; scored={totals['scored']}; unscored={totals['unscored']}; "
        f"TP={totals['TP']}; TN={totals['TN']}; FP={totals['FP']}; FN={totals['FN']}.\n\n"
        "| case_id | historical_or_synthetic | category | ground_truth_failure | ground_truth_failure_type | limina_prediction | limina_failure_type | correct | TP/TN/FP/FN |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        f"{table_rows}\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": markdown_path}


def write_head_to_head_comparison(
    cases: list[EvaluationCase],
    results: list[EvaluatorResult],
    destination: Path,
    left_evaluator: str = "limina",
    right_evaluator: str = "recruiter_llm_judge",
) -> Path:
    """Write a single-run, same-case evaluator ledger without inferring missing results."""
    destination.mkdir(parents=True, exist_ok=True)
    by_key = {(result.evaluator, result.case_id): result for result in results}
    summary = summarize(cases, results)

    def label(result: EvaluatorResult | None) -> str:
        if result is None or result.status != "ok":
            return "unscored"
        return "failure" if result.detected_failure else "healthy"

    def types(result: EvaluatorResult | None) -> str:
        return "; ".join(result.failure_types) if result and result.failure_types else ""

    rows: list[str] = []
    disagreements: list[str] = []
    for case in cases:
        left = by_key.get((left_evaluator, case.case_id))
        right = by_key.get((right_evaluator, case.case_id))
        left_label, right_label = label(left), label(right)
        truth = "failure" if case.expected_failure else "healthy"
        left_correct = str(left_label == truth).lower() if left_label != "unscored" else "unscored"
        right_correct = str(right_label == truth).lower() if right_label != "unscored" else "unscored"
        rows.append(
            f"| {case.case_id} | {case.source} | {case.category} | {truth} | {left_label} | {types(left)} | {left_correct} | {right_label} | {types(right)} | {right_correct} |"
        )
        if left_label != right_label:
            disagreements.append(
                f"| {case.case_id} | {truth} | {left_label} | {right_label} | {left_correct} | {right_correct} |"
            )

    def metric_line(evaluator: str) -> str:
        values = summary["by_evaluator"].get(evaluator, {})
        confusion = values.get("confusion", {})
        latency = values.get("latency_and_cost", {})
        return (
            f"| {evaluator} | {confusion.get('tp')} | {confusion.get('tn')} | {confusion.get('fp')} | {confusion.get('fn')} | "
            f"{_display(confusion.get('precision'))} | {_display(confusion.get('recall'))} | {_display(confusion.get('f1'))} | "
            f"{_display(confusion.get('accuracy'))} | {_display(confusion.get('specificity'))} | "
            f"{_display(confusion.get('false_positive_rate'))} | {_display(confusion.get('false_negative_rate'))} | "
            f"{_display(latency.get('mean_latency_ms'))} | {_display(latency.get('p50_latency_ms'))} | {_display(latency.get('p95_latency_ms'))} |"
        )

    output = destination / "head_to_head_comparison.md"
    output.write_text(
        "# Live head-to-head comparison\n\n"
        "Both evaluators consumed the same labelled case envelopes. `unscored` is retained rather than treated as a verdict.\n\n"
        "| Evaluator | TP | TN | FP | FN | Precision | Recall | F1 | Accuracy | Specificity | FPR | FNR | Mean ms | p50 ms | p95 ms |\n"
        "| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |\n"
        f"{metric_line(left_evaluator)}\n{metric_line(right_evaluator)}\n\n"
        "## Per-case ledger\n\n"
        "| case_id | source | category | ground truth | Limina | Limina failure type | Limina correct | Live LLM judge | Live LLM failure type | LLM correct |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(rows)
        + "\n\n## Binary disagreements\n\n"
        + ("| case_id | ground truth | Limina | Live LLM judge | Limina correct | LLM correct |\n| --- | --- | --- | --- | --- | --- |\n" + "\n".join(disagreements) if disagreements else "None. Both evaluators produced the same binary label for every scored case.")
        + "\n",
        encoding="utf-8",
    )
    return output
