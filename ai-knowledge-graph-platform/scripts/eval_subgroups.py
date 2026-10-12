"""Subgroup breakdown of the recorded eval results. Reads files only: no API, no LLM, no cost.

    python scripts/eval_subgroups.py            # print
    python scripts/eval_subgroups.py --write    # also write evals/subgroup_report.json

Sources: evals/last_run.json (golden set, by question type), evals/routing_eval_results.json is not re-derived;
the routing cases are re-scored with the deterministic router (by expected class and by set), and
evals/entity_resolution_results.json is grouped by case category.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.evaluation.subgroups import subgroup_report  # noqa: E402

EVALS = ROOT / "evals"


def _load(name: str):
    path = EVALS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def golden_report() -> dict | None:
    data = _load("last_run.json")
    return subgroup_report(data["questions"], group_key="type") if data else None


def routing_report() -> dict | None:
    from graphrag.retrieval.query_router import route_query

    rows = []
    for path in sorted(EVALS.glob("routing_cases*.json")):
        for case in json.loads(path.read_text(encoding="utf-8"))["cases"]:
            got = route_query(case["question"]).route.value
            rows.append({"set": path.stem, "expected": case["expected"], "passed": got == case["expected"]})
    if not rows:
        return None
    return {"by_expected_class": subgroup_report(rows, group_key="expected"),
            "by_set": subgroup_report(rows, group_key="set")}


def entity_resolution_report() -> dict | None:
    data = _load("entity_resolution_results.json")
    return subgroup_report(data["results"], group_key="category", passed_key="correct", min_n=2) if data else None


def build() -> dict:
    return {"golden_set_by_type": golden_report(), "routing": routing_report(),
            "entity_resolution_by_category": entity_resolution_report(),
            "note": "Descriptive only. Small groups have wide intervals; a flag means 'look here', not 'broken'."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = build()
    for name, section in report.items():
        if not isinstance(section, dict):
            continue
        sections = section if "groups" not in section else {name: section}
        for label, sub in sections.items():
            if not sub:
                continue
            print(f"{label}: overall={sub['overall_rate']} n={sub['n']} flagged={sub['flagged']} max_gap={sub['max_gap']}")
    if args.write:
        (EVALS / "subgroup_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
