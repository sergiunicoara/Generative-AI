"""Aggregate the per-profile ablation result files into the report tables.

    python scripts/aggregate_ablation.py evals/ablation/aerospace-v2.3-deepseek

Reads the valid runs (``summary.n_errors == 0``; the ``invalid/`` sub-folder is ignored) and prints:
per-profile means and ranges, a per-question table (runs passed / faithfulness), per-type means, the
pass rate with citation-recall failures ignored (a content-only view), and the result file paths.
Standard library only; the paired statistics live in compare_ablation_profiles.py.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

PROFILES = ["vector_only", "text_hybrid", "full", "full_context"]
CITATION_FAILURE = "insufficient citation recall"


def load(directory: Path, profile: str) -> list[dict]:
    runs = []
    for path in sorted(directory.glob(f"*__{profile}__*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["summary"].get("n_errors") == 0:
            report["_path"] = str(path)
            runs.append(report)
    return runs


def span(values: list) -> str:
    values = [v for v in values if v is not None]
    return f"{st.mean(values):.3f} [{min(values):.3f}-{max(values):.3f}]" if values else "n/a"


def citation_only(row: dict) -> bool:
    """A failed row whose only failures are citation recall (the answer content passed)."""
    failures = row["golden_failures"] or []
    return bool(failures) and all(f.startswith(CITATION_FAILURE) for f in failures)


def content_pass_rate(run: dict) -> float:
    rows = run["results"]
    return sum(bool(r["golden_pass"]) or citation_only(r) for r in rows) / len(rows)


def per_profile(runs: dict[str, list[dict]]) -> None:
    print("## per profile (mean [min-max] over the valid runs)")
    for name, group in runs.items():
        s = [r["summary"] for r in group]
        print(f"{name} (runs={len(group)}) | pass {span([x['golden_pass_rate'] for x in s])}"
              f" | pass ignoring citation recall {span([content_pass_rate(r) for r in group])}"
              f" | faith {span([x['faithfulness_mean'] for x in s])}"
              f" | scored {[x['n_scored'] for x in s]} refused {[x['n_refused'] for x in s]}"
              f" unscorable {[x['n_unscorable'] for x in s]}"
              f" | e2e p50 {span([x['latency_e2e_ms']['p50'] for x in s])} p95 {span([x['latency_e2e_ms']['p95'] for x in s])}"
              f" | retrieval p95 {span([x['latency_retrieval_ms']['p95'] for x in s])}"
              f" | fallbacks {[x['n_agentic_fallback'] for x in s]}"
              f" | golden citation basis: {group[0]['run'].get('citation', {}).get('golden_citation_basis', 'legacy')}")
        for which in ("answer", "context"):    # headline attribution comparison: answer
            sets = [x.get("citations", {}).get(which, {}) for x in s]
            print(f"    {which} citations: recall {span([c.get('recall') for c in sets])}"
                  f" | precision {span([c.get('precision') for c in sets])}"
                  f" | macro F1 {span([c.get('f1_macro') for c in sets])}"
                  f" | distinct docs/answer {span([c.get('mean_distinct_docs') for c in sets])}"
                  + (f" | tag compliance {span([c.get('compliance') for c in sets])}"
                     f" | tags resolving {span([c.get('tag_resolution_rate') for c in sets])}"
                     if which == "answer" else ""))


def per_question(runs: dict[str, list[dict]]) -> dict:
    table: dict = collections.OrderedDict()
    for name, group in runs.items():
        for run in group:
            for row in run["results"]:
                cell = table.setdefault(row["id"], {"type": row["type"]}).setdefault(
                    name, {"pass": [], "content": [], "faith": [], "runs": 0})
                cell["runs"] += 1
                cell["pass"].append(bool(row["golden_pass"]))
                cell["content"].append(bool(row["golden_pass"]) or citation_only(row))
                if row["faithfulness"] is not None:
                    cell["faith"].append(row["faithfulness"])
    return table


def print_questions(table: dict, names: list[str]) -> None:
    print("\n## per question: runs passed / runs (mean faithfulness or -)")
    print("id | type | " + " | ".join(names))
    for qid, row in table.items():
        cells = []
        for name in names:
            c = row[name]
            faith = f"{st.mean(c['faith']):.2f}" if c["faith"] else "-"
            cells.append(f"{sum(c['pass'])}/{c['runs']} ({faith})")
        print(f"{qid} | {row['type']} | " + " | ".join(cells))


def print_types(table: dict, names: list[str]) -> None:
    print("\n## per type: (a) pass share / (b) pass share ignoring citation recall / mean faithfulness")
    for qtype in sorted({row["type"] for row in table.values()}):
        ids = [i for i, row in table.items() if row["type"] == qtype]
        cells = []
        for name in names:
            rate = st.mean(sum(table[i][name]["pass"]) / table[i][name]["runs"] for i in ids)
            content = st.mean(sum(table[i][name]["content"]) / table[i][name]["runs"] for i in ids)
            faith = [x for i in ids for x in table[i][name]["faith"]]
            cells.append(f"{rate:.2f}/{content:.2f}/{st.mean(faith):.2f}" if faith else f"{rate:.2f}/{content:.2f}/n/a")
        print(f"{qtype} ({len(ids)}) | " + " | ".join(cells))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--profiles", nargs="+", default=PROFILES)
    args = parser.parse_args(argv)
    runs = {name: load(args.directory, name) for name in args.profiles}
    missing = [name for name, group in runs.items() if not group]
    if missing:
        raise SystemExit(f"no valid runs found for: {', '.join(missing)} in {args.directory}")
    per_profile(runs)
    table = per_question(runs)
    print_questions(table, args.profiles)
    print_types(table, args.profiles)
    print("\n## files")
    for group in runs.values():
        for run in group:
            print(run["_path"])


if __name__ == "__main__":
    sys.exit(main())
