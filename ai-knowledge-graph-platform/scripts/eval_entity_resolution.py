"""Score AliasRegistry.resolve on labelled entity-resolution cases.

Offline and deterministic: the registry is populated in memory (exact, stem and fuzzy tiers plus the lexical guards;
the embedding tier needs Neo4j and an embedding provider, so it is NOT measured here). No LLM, no network, no cost.
Each case's entity type is passed to resolve(), and a case file's "curated_aliases" are applied as curated aliases.

    python scripts/eval_entity_resolution.py                    # original set, print the report
    python scripts/eval_entity_resolution.py --set heldout      # held-out set written 2026-11-05
    python scripts/eval_entity_resolution.py --no-guards        # ratio-only behaviour, for comparison
    python scripts/eval_entity_resolution.py --write            # also write the results file
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
SETS = {
    "original": (CASES, RESULTS),
    "heldout": (ROOT / "evals" / "entity_resolution_cases_heldout.json",
                ROOT / "evals" / "entity_resolution_results_heldout.json"),
}


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


def build_registry(canonical: list[tuple[str, str]], curated: list[dict] | None = None, *,
                   guards: bool = True) -> AliasRegistry:
    registry = AliasRegistry(None, tenant="er-eval")
    registry._lexical_guards = guards
    for name, type_ in canonical:
        registry._exact[_normalize(name)] = (name, type_)
        registry._stemmed[_normalize_ro(name)] = (name, type_)
    registry._rebuild_fuzzy_index()
    if curated:
        registry.apply_curated_aliases(curated)
    return registry


def run(path: Path = CASES, *, guards: bool = True) -> dict:
    canonical, cases = load_cases(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    curated = [{"alias": a, "canonical": n, "type": t} for a, n, t in data.get("curated_aliases", [])]
    registry = build_registry(canonical, curated, guards=guards)
    report = evaluate_entity_resolution(
        cases, lambda name, type_: registry.resolve(name, entity_type=type_), typed=True)
    report["lexical_guards"] = guards
    report["curated_aliases"] = len(curated)
    report["tiers_measured"] = ["exact", "regulatory-prefix", "romanian-stem", "fuzzy"] + (
        ["lexical-guards", "type-scope", "acronym-review", "containment-review"] if guards else [])
    report["tiers_not_measured"] = ["embedding", "cooccurrence-tiebreak"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--set", choices=sorted(SETS), default="original")
    parser.add_argument("--no-guards", action="store_true")
    args = parser.parse_args()
    cases_path, results_path = SETS[args.set]
    report = run(cases_path, guards=not args.no_guards)
    summary = {k: v for k, v in report.items() if k != "results"}
    print(json.dumps(summary, indent=2))
    for r in report["results"]:
        if r["error"]:
            print(f"  {r['error']:16} {r['raw_name']!r:48} expected={r['expected']} got={r['actual']} -> {r['canonical_name']!r}")
    if args.write:
        results_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
