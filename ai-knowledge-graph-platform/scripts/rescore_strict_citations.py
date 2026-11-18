"""Re-score stored ablation runs with the strict citation normaliser (no model calls, no database).

    python scripts/rescore_strict_citations.py evals/ablation/aerospace-v2.3-deepseek

Per stored run: the raw citation strings of every question go through graphrag.evaluation.citations
(canonical ids only), giving strict recall and precision. The strict golden pass keeps every stored
non-citation failure and replaces the citation-recall failure by the strict one, so nothing else in
the verdict changes. full_context's stored citations are document names found in its answer under a
different prompt, so its row is labelled: the new [DOC: id] prompt can only be measured by a re-run.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.evaluation import citations  # noqa: E402

PROFILES = ["vector_only", "text_hybrid", "full", "full_context"]


def load(directory: Path, profile: str) -> list[dict]:
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob(f"*__{profile}__*.json"))]
    return [r for r in reports if r["summary"].get("n_errors") == 0]


def rescore_run(report: dict, questions: dict[str, dict]) -> dict:
    recalls_old, recalls, precisions, passes, passes_old, f1s, distinct = [], [], [], [], [], [], []
    for row in report["results"]:
        question = questions[row["id"]]
        strict = citations.strict_citations(list(row["citations"]))
        if row["citation_recall"] is not None:
            recalls_old.append(row["citation_recall"])
        recall = citations.recall(question, strict)
        if recall is not None:
            recalls.append(recall)
        precision = citations.precision(question, strict)
        if precision is not None:
            precisions.append(precision)
        failures = [f for f in row["golden_failures"] if not f.startswith(citations.CITATION_FAILURE)]
        failure = citations.citation_failure(question, strict)
        f1 = citations.f1(question, strict)
        if f1 is not None:
            f1s.append(f1)
        distinct.append(len(strict))
        passes.append(not failures and not failure)
        passes_old.append(bool(row["golden_pass"]))
    mean = lambda values: st.mean(values) if values else None  # noqa: E731
    return {"recall_old": mean(recalls_old), "recall": mean(recalls), "precision": mean(precisions),
            "pass_old": mean(passes_old), "pass": mean(passes), "f1": mean(f1s), "distinct": mean(distinct)}


def span(values: list) -> str:
    values = [v for v in values if v is not None]
    return f"{st.mean(values):.3f} [{min(values):.3f}-{max(values):.3f}]" if values else "n/a"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--golden", type=Path, default=ROOT / "evals" / "golden_set.json")
    args = parser.parse_args(argv)
    questions = {q["id"]: q for q in json.loads(args.golden.read_text(encoding="utf-8"))["questions"]}
    print("| profile | runs | recall old | recall strict | precision strict | macro F1 strict | distinct ids / answer | golden pass old | golden pass strict |")
    print("|---|---|---|---|---|---|---|---|---|")
    for profile in PROFILES:
        scored = [rescore_run(report, questions) for report in load(args.directory, profile)]
        if not scored:
            continue
        label = profile + (" (labelled: other prompt, not comparable)" if profile == "full_context" else "")
        print(f"| {label} | {len(scored)} | " + " | ".join(
            span([s[key] for s in scored]) for key in ("recall_old", "recall", "precision", "f1", "distinct", "pass_old", "pass")) + " |")


if __name__ == "__main__":
    sys.exit(main())
