"""Metrics for a contextual-knowledge question set (one split), with denominators.

Shared by ``scripts/run_knowledge_eval.py`` and the use-case charter runner.
Deterministic and in process: the pack is loaded from YAML, no database,
network or model.  Every metric carries its numerator and denominator.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from graphrag.context_graph.knowledge_evaluation import KnowledgeReleaseRef
from graphrag.context_graph.knowledge_release import case_mismatches, load_cases, load_pack, run_case


def ratio(num: int, den: int) -> dict:
    return {"value": round(num / den, 4) if den else None, "numerator": num, "denominator": den}


def evaluate_cases(cases_path: Path, *, root: Path) -> dict:
    raw = json.loads(cases_path.read_text(encoding="utf-8"))
    pack = load_pack(root / raw["pack"])
    cases = load_cases(cases_path)
    ref = KnowledgeReleaseRef(release_id="eval-candidate", domain=pack.domain, version="eval")

    app_tp = app_pred = app_exp = 0
    val_ok = val_total = 0
    conf_tp = conf_pred = conf_exp = 0
    esc_ok = esc_total = 0
    overall_ok = overall_total = 0
    cite_ok = cite_total = 0
    latencies: list[float] = []
    per_case = []
    for case in cases:
        started = perf_counter()
        decision = run_case(case, items=pack.published_items(), exceptions=pack.published_exceptions(),
                            domain=pack.domain, release=ref)
        latencies.append((perf_counter() - started) * 1000)
        exp = case.expected
        predicted = {k.split("@v")[0] for k in decision.applicable}
        expected = set(exp.applicable)
        app_tp += len(predicted & expected)
        app_pred += len(predicted)
        app_exp += len(expected)
        by_item = {r.item_id: r for r in decision.results}
        for item_id, outcome in exp.outcomes.items():
            val_total += 1
            actual = by_item[item_id].outcome.value if item_id in by_item else "absent"
            val_ok += actual == outcome
        if exp.conflicts is not None:
            actual_c = {(c.kind, tuple(sorted(k.split("@v")[0] for k in c.item_keys)), c.resolved)
                        for c in decision.conflicts}
            expected_c = {(c["kind"], tuple(sorted(c["items"])), bool(c.get("resolved", False)))
                          for c in exp.conflicts}
            conf_tp += len(actual_c & expected_c)
            conf_pred += len(actual_c)
            conf_exp += len(expected_c)
        if exp.human_review_required is not None:
            esc_total += 1
            esc_ok += decision.human_review_required == exp.human_review_required
        if exp.overall is not None:
            overall_total += 1
            overall_ok += decision.overall.value == exp.overall
        for item_id, sources in exp.citations.items():
            cite_total += 1
            cited = {c.source_id for c in by_item[item_id].citations} if item_id in by_item else set()
            cite_ok += set(sources) <= cited
        mismatches = case_mismatches(case, decision)
        per_case.append({"id": case.id, "covers": case.covers, "passed": not mismatches,
                         "mismatches": mismatches, "overall": decision.overall.value,
                         "human_review_required": decision.human_review_required,
                         "decision_id": decision.decision_id})

    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))] if ordered else None
    return {
        "dataset": str(cases_path.relative_to(root)).replace("\\", "/"),
        "split": raw.get("split", "development"),
        "label": raw.get("label", ""),
        "result_source": "deterministic in-process evaluation of the YAML pack (no DB, no network, no model)",
        "cases": {"passed": sum(c["passed"] for c in per_case), "total": len(per_case)},
        "metrics": {
            "applicability_precision": ratio(app_tp, app_pred),
            "applicability_recall": ratio(app_tp, app_exp),
            "validation_outcome_accuracy": ratio(val_ok, val_total),
            "conflict_precision": ratio(conf_tp, conf_pred),
            "conflict_recall": ratio(conf_tp, conf_exp),
            "escalation_accuracy": ratio(esc_ok, esc_total),
            "overall_outcome_accuracy": ratio(overall_ok, overall_total),
            "citation_correctness": ratio(cite_ok, cite_total),
        },
        "time_to_answer_ms": {"p95": round(p95, 3) if p95 is not None else None, "cases": len(ordered)},
        "per_case": per_case,
    }


__all__ = ["evaluate_cases", "ratio"]
