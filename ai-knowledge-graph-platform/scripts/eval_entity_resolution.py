"""Score AliasRegistry.resolve on the labelled cases in evals/entity_resolution_cases.json.

Offline and deterministic: the registry is populated in memory (exact, stem and fuzzy tiers only; the embedding tier
needs Neo4j and an embedding provider, so it is NOT measured here). No LLM, no network, no cost.

    python scripts/eval_entity_resolution.py            # print the report
    python scripts/eval_entity_resolution.py --write    # also write evals/entity_resolution_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.evaluation.entity_resolution import EntityResolutionCase, evaluate_entity_resolution  # noqa: E402
from graphrag.graph.alias_registry import AliasRegistry, _normalize, _normalize_ro  # noqa: E402

CASES = ROOT / "evals" / "entity_resolution_cases.json"
RESULTS = ROOT / "evals" / "entity_resolution_results.json"


def load_cases(path: Path = CASES) -> tuple[list[tuple[str, str]], list[EntityResolutionCase]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    canonical = [(name, type_) for name, type_ in data["canonical"]]
    cases = [
        EntityResolutionCase(
            raw_name=c["raw_name"], entity_type=c["entity_type"], expected=c["expected"],
            canonical_name=c.get("canonical_name", ""), canonical_type=c.get("canonical_type", ""),
            category=c.get("category", "general"),
        )
        for c in data["cases"]
    ]
    return canonical, cases


def build_registry(canonical: list[tuple[str, str]]) -> AliasRegistry:
    registry = AliasRegistry(None, tenant="er-eval")
    for name, type_ in canonical:
        registry._exact[_normalize(name)] = (name, type_)
        registry._stemmed[_normalize_ro(name)] = (name, type_)
    registry._rebuild_fuzzy_index()
    return registry


def run(path: Path = CASES) -> dict:
    canonical, cases = load_cases(path)
    report = evaluate_entity_resolution(cases, build_registry(canonical).resolve)
    report["tiers_measured"] = ["exact", "regulatory-prefix", "romanian-stem", "fuzzy"]
    report["tiers_not_measured"] = ["embedding", "cooccurrence-tiebreak"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = run()
    summary = {k: v for k, v in report.items() if k != "results"}
    print(json.dumps(summary, indent=2))
    for r in report["results"]:
        if r["error"]:
            print(f"  {r['error']:16} {r['raw_name']!r:48} expected={r['expected']} got={r['actual']} -> {r['canonical_name']!r}")
    if args.write:
        RESULTS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
