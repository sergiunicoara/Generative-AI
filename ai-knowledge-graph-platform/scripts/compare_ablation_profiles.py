"""Paired comparison of one ablation profile against the others, from the per-question result files.

    python scripts/compare_ablation_profiles.py evals/ablation/aerospace-v2.3 \\
        --baseline full --others vector_only text_hybrid full_context

Per profile, the valid runs (``summary.n_errors == 0``; the ``invalid/`` sub-folder is ignored) are
collapsed to one value per question, so every comparison is paired on the same questions:

* golden pass: the question passes when it passes in more than half of that profile's runs;
* faithfulness: the mean over the runs in which the question was *scored* (refusals and unscorable
  answers are excluded, so a question counts only when scored in both profiles);
* citation recall (files written by the strict-citation benchmark): the mean over the runs of the
  answer-set ``recall_citable`` and of the context-set ``recall_raw``, compared with a paired bootstrap.

Tests: exact McNemar (two-sided binomial on the discordant pairs) for pass/fail, with Holm adjustment
across the compared profiles; paired percentile bootstrap (seeded) for the mean faithfulness
difference. A difference is called *distinguishable* only if the McNemar p (Holm-adjusted) is < 0.05,
or the bootstrap 95% CI excludes 0. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

MULTI_HOP_TYPE = "multi_hop"


def load_runs(directory: Path, profile: str) -> list[dict]:
    """Valid result files of ``profile`` (top level of ``directory`` only)."""
    runs = []
    for path in sorted(directory.glob(f"*__{profile}__*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["summary"].get("n_errors") == 0 and report["run"]["retrieval_profile"] == profile:
            report["_path"] = str(path)
            runs.append(report)
    return runs


def per_question(runs: list[dict]) -> dict[str, dict]:
    """Collapse runs to {question id: {type, passed, faithfulness|None, n_runs}}."""
    seen: dict[str, dict] = {}
    for run in runs:
        for row in run["results"]:
            entry = seen.setdefault(row["id"], {"type": row.get("type"), "passes": [], "faith": [],
                                                "recall_answer": [], "recall_context": []})
            metrics = row.get("citation_metrics") or {}
            for key, which, field in (("recall_answer", "answer", "recall_citable"),
                                      ("recall_context", "context", "recall_raw")):
                value = (metrics.get(which) or {}).get(field)
                if value is not None:
                    entry[key].append(float(value))
            entry["passes"].append(bool(row["golden_pass"]))
            if row["status"] == "scored" and row["faithfulness"] is not None:
                entry["faith"].append(float(row["faithfulness"]))
    return {
        qid: {
            "type": e["type"],
            "passed": sum(e["passes"]) * 2 > len(e["passes"]),
            "faithfulness": statistics.mean(e["faith"]) if e["faith"] else None,
            "n_runs": len(e["passes"]),
            "recall_answer": statistics.mean(e["recall_answer"]) if e["recall_answer"] else None,
            "recall_context": statistics.mean(e["recall_context"]) if e["recall_context"] else None,
        }
        for qid, e in seen.items()
    }


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts (1.0 when there are none)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(diffs: list[float], *, resamples: int = 10_000, seed: int = 2026,
                        level: float = 0.95) -> tuple[float, float] | None:
    """Percentile bootstrap CI of the mean of ``diffs`` (None when there are no pairs)."""
    if not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    lo = means[int((1 - level) / 2 * resamples)]
    hi = means[min(resamples - 1, int((1 + level) / 2 * resamples))]
    return lo, hi


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def compare(base: dict[str, dict], other: dict[str, dict], *, only_type: str | None = None) -> dict:
    ids = sorted(q for q in base if q in other and (only_type is None or base[q]["type"] == only_type))
    only_base = [q for q in ids if base[q]["passed"] and not other[q]["passed"]]
    only_other = [q for q in ids if other[q]["passed"] and not base[q]["passed"]]
    both_scored = [q for q in ids if base[q]["faithfulness"] is not None and other[q]["faithfulness"] is not None]
    diffs = [base[q]["faithfulness"] - other[q]["faithfulness"] for q in both_scored]
    ci = paired_bootstrap_ci(diffs)

    def paired(key: str) -> dict:
        pairs = [q for q in ids if base[q].get(key) is not None and other[q].get(key) is not None]
        deltas = [base[q][key] - other[q][key] for q in pairs]
        interval = paired_bootstrap_ci(deltas)
        return {"n_pairs": len(deltas), "mean_diff": statistics.mean(deltas) if deltas else None,
                "ci95": interval, "distinguishable": bool(interval and (interval[0] > 0 or interval[1] < 0))}

    return {
        "recall_answer": paired("recall_answer"), "recall_context": paired("recall_context"),
        "n_questions": len(ids),
        "pass_base": sum(base[q]["passed"] for q in ids), "pass_other": sum(other[q]["passed"] for q in ids),
        "only_base_passed": len(only_base), "only_other_passed": len(only_other),
        "only_base_ids": only_base, "only_other_ids": only_other,
        "mcnemar_p": mcnemar_exact(len(only_base), len(only_other)),
        "n_faithfulness_pairs": len(diffs),
        "faithfulness_mean_diff": statistics.mean(diffs) if diffs else None,
        "faithfulness_ci95": ci,
        "faithfulness_distinguishable": bool(ci and (ci[0] > 0 or ci[1] < 0)),
    }


def analyse(directory: Path, baseline: str, others: list[str]) -> dict:
    profiles = {name: per_question(load_runs(directory, name)) for name in [baseline, *others]}
    missing = [name for name, data in profiles.items() if not data]
    if missing:
        raise SystemExit(f"no valid runs found for: {', '.join(missing)} in {directory}")
    out = {"baseline": baseline, "n_runs": {}, "comparisons": {}}
    for name in profiles:
        out["n_runs"][name] = len(load_runs(directory, name))
    p_all = [compare(profiles[baseline], profiles[o])["mcnemar_p"] for o in others]
    adjusted = holm(p_all)
    for other, p_holm in zip(others, adjusted):
        result = {"all": compare(profiles[baseline], profiles[other]),
                  "multi_hop": compare(profiles[baseline], profiles[other], only_type=MULTI_HOP_TYPE)}
        result["all"]["mcnemar_p_holm"] = p_holm
        result["all"]["pass_distinguishable"] = p_holm < 0.05
        out["comparisons"][other] = result
    return out


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render(report: dict) -> str:
    base = report["baseline"]
    lines = [f"Baseline: {base}.  Valid runs per profile: "
             + ", ".join(f"{k}={v}" for k, v in report["n_runs"].items()), "",
             "| vs | scope | n | pass (base/other) | only base | only other | McNemar p | Holm p | "
             "faith pairs | mean diff | 95% CI | distinguishable |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for other, result in report["comparisons"].items():
        for scope in ("all", "multi_hop"):
            r = result[scope]
            ci = r["faithfulness_ci95"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
            holm_p = r.get("mcnemar_p_holm")
            distinct = []
            if scope == "all" and r.get("pass_distinguishable"):
                distinct.append("pass")
            if r["faithfulness_distinguishable"]:
                distinct.append("faithfulness")
            lines.append(
                f"| {other} | {scope} | {r['n_questions']} | {r['pass_base']}/{r['pass_other']} | "
                f"{r['only_base_passed']} | {r['only_other_passed']} | {_fmt(r['mcnemar_p'])} | "
                f"{_fmt(holm_p)} | {r['n_faithfulness_pairs']} | {_fmt(r['faithfulness_mean_diff'])} | "
                f"{ci_text} | {', '.join(distinct) or 'no'} |")
    lines += ["", "Paired bootstrap of per-question citation recall (baseline minus other; answer set = recall_citable, "
              "context set = recall_raw; no multiplicity adjustment):", "",
              "| vs | scope | answer pairs | answer mean diff | answer 95% CI | context pairs | context mean diff | "
              "context 95% CI | distinguishable |", "|---|---|---|---|---|---|---|---|---|"]
    for other, result in report["comparisons"].items():
        for scope in ("all", "multi_hop"):
            a, c = result[scope]["recall_answer"], result[scope]["recall_context"]
            fmt = lambda r: "n/a" if r["ci95"] is None else f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"  # noqa: E731
            distinct = [name for name, r in (("answer", a), ("context", c)) if r["distinguishable"]]
            lines.append(f"| {other} | {scope} | {a['n_pairs']} | {_fmt(a['mean_diff'])} | {fmt(a)} | {c['n_pairs']} | "
                         f"{_fmt(c['mean_diff'])} | {fmt(c)} | {', '.join(distinct) or 'no'} |")
    lines += ["", "Holm adjustment is applied to the all-questions McNemar tests across the compared profiles; "
              "multi_hop rows are descriptive (a handful of questions cannot reach significance)."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="folder with the per-profile result files")
    parser.add_argument("--baseline", default="full")
    parser.add_argument("--others", nargs="+", default=["vector_only", "text_hybrid", "full_context"])
    parser.add_argument("--json", type=Path, help="also write the full numbers to this file")
    args = parser.parse_args(argv)
    report = analyse(args.directory, args.baseline, args.others)
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
