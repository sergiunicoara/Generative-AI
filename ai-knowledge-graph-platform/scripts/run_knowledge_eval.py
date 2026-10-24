"""Run the deterministic contextual-knowledge evaluation set.

    python scripts/run_knowledge_eval.py
    python scripts/run_knowledge_eval.py --cases evals/knowledge_eval/packaging_labeling_cases.json

No network, no model calls, no database: the pack is loaded from YAML and
evaluated in process.  Metrics are reported separately, each with its
denominator, and written to ``evals/knowledge_eval/last_run.json``.

The dataset is a SYNTHETIC ENGINEERING EVALUATION: invented rules and
expectations, written by the developer, with no independent domain review.
The numbers measure agreement with those expectations, nothing more.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.context_graph.knowledge_evaluation import KnowledgeReleaseRef  # noqa: E402
from graphrag.context_graph.knowledge_release import (  # noqa: E402
    case_mismatches, load_cases, load_pack, run_case,
)

DEFAULT_CASES = ROOT / "evals" / "knowledge_eval" / "packaging_labeling_cases.json"


def _ratio(num: int, den: int) -> dict:
    return {"value": round(num / den, 4) if den else None, "numerator": num, "denominator": den}


def evaluate(cases_path: Path) -> dict:
    raw = json.loads(cases_path.read_text(encoding="utf-8"))
    pack = load_pack(ROOT / raw["pack"])
    cases = load_cases(cases_path)
    ref = KnowledgeReleaseRef(release_id="eval-candidate", domain=pack.domain, version="eval")

    app_tp = app_pred = app_exp = 0
    val_ok = val_total = 0
    conf_tp = conf_pred = conf_exp = 0
    esc_ok = esc_total = 0
    overall_ok = overall_total = 0
    cite_ok = cite_total = 0
    per_case = []
    for case in cases:
        decision = run_case(case, items=pack.published_items(), exceptions=pack.published_exceptions(),
                            domain=pack.domain, release=ref)
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

    return {
        "dataset": str(cases_path.relative_to(ROOT)).replace("\\", "/"),
        "label": raw.get("label", ""),
        "result_source": "deterministic in-process evaluation of the YAML pack (no DB, no network, no model)",
        "cases": {"passed": sum(c["passed"] for c in per_case), "total": len(per_case)},
        "metrics": {
            "applicability_precision": _ratio(app_tp, app_pred),
            "applicability_recall": _ratio(app_tp, app_exp),
            "validation_outcome_accuracy": _ratio(val_ok, val_total),
            "conflict_precision": _ratio(conf_tp, conf_pred),
            "conflict_recall": _ratio(conf_tp, conf_exp),
            "escalation_accuracy": _ratio(esc_ok, esc_total),
            "overall_outcome_accuracy": _ratio(overall_ok, overall_total),
            "citation_correctness": _ratio(cite_ok, cite_total),
        },
        "per_case": per_case,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=ROOT / "evals" / "knowledge_eval" / "last_run.json")
    args = parser.parse_args()
    report = evaluate(args.cases.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["label"])
    print(f"cases passed: {report['cases']['passed']}/{report['cases']['total']}")
    for name, metric in report["metrics"].items():
        value = "n/a" if metric["value"] is None else f"{metric['value']:.4f}"
        print(f"  {name:30s} {value}  ({metric['numerator']}/{metric['denominator']})")
    for case in report["per_case"]:
        if not case["passed"]:
            print("FAIL", case["id"], case["mismatches"])
    return 0 if report["cases"]["passed"] == report["cases"]["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
