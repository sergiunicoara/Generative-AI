"""Classify every failed golden-set row of an ablation folder (offline: stored results only, no model calls).

    python scripts/classify_golden_misses.py evals/ablation/aerospace-v2.3-strict
    python scripts/classify_golden_misses.py <folder> --output-dir <dir>

Reads the valid runs (``summary.n_errors == 0``; the ``invalid/`` sub-folder is ignored) and gives each row with
``golden_pass == False`` exactly one class, by an ordered rule on what failed and what the stored evidence shows:

* ``citation_uncitable_artifact`` - only the citation check failed and, once the expected documents the prompt could
  not offer (superseded ones) are ignored, nothing is missing; no pipeline change can fix it;
* ``retrieval_miss_citation`` - only the citation check failed and an expected, citable document is missing from the
  retrieved set (retrieval profiles);
* ``citation_only`` - only the citation check failed, the retrieved set holds every expected document;
* ``refused_with_evidence`` / ``refused_no_evidence`` - a content check failed and the answer was a refusal; the
  evidence is "every expected document is in the retrieved set";
* ``retrieval_miss`` - a content check failed and an expected document is missing from the retrieved set;
* ``answer_error`` - a content check failed although every expected document was retrieved (``full_context`` has no
  retrieval step: the whole corpus is in the prompt, so a content failure is an answer error by construction).

Not decidable from stored data, and therefore never assigned: a graph or schema gap (the graph state records counts
only) and "right document, wrong chunk" (the retrieved context text is not stored, only document ids). The classes
are document-level proxies. A question's class over the runs of one profile is the most frequent one among its
failed runs. Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROFILES = ["vector_only", "text_hybrid", "full", "full_context"]
CITATION_FAILURE = "insufficient citation recall"
CLASSES = ["citation_uncitable_artifact", "retrieval_miss_citation", "citation_only", "refused_with_evidence",
           "refused_no_evidence", "retrieval_miss", "answer_error"]
LIMITATION = (
    "Classes are document-level proxies. The retrieved context text is not stored (only document ids), so "
    "'right document, wrong chunk' is counted as answer_error. A graph or schema gap cannot be decided from stored "
    "data (the graph state records counts only) and is never assigned. For full_context the whole corpus is in the "
    "prompt, so a content failure is an answer error by construction.")


def load(directory: Path, profile: str) -> list[dict]:
    runs = []
    for path in sorted(directory.glob(f"*__{profile}__*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["summary"].get("n_errors") == 0:
            report["_path"] = str(path)
            runs.append(report)
    return runs


def _retrieved_all_expected(row: dict) -> bool | None:
    """True/False from the context citation set (raw recall over every expected document); None without one."""
    context = (row.get("citation_metrics") or {}).get("context") or {}
    recall = context.get("recall_raw")
    return None if recall is None else recall >= 1.0


def classify(row: dict) -> str | None:
    """The class of one failed row, or None when the row passed."""
    if row.get("golden_pass"):
        return None
    failures = row.get("golden_failures") or []
    content = [f for f in failures if not f.startswith(CITATION_FAILURE)]
    retrieved_all = _retrieved_all_expected(row)
    if not content:                                   # only the citation check failed
        if not row.get("golden_failures_citable"):
            return "citation_uncitable_artifact"
        return "retrieval_miss_citation" if retrieved_all is False else "citation_only"
    if row.get("status") == "refused":
        return "refused_with_evidence" if retrieved_all else "refused_no_evidence"
    return "retrieval_miss" if retrieved_all is False else "answer_error"


def classify_runs(runs: dict[str, list[dict]]) -> dict:
    """``{"rows": [...], "by_profile": {profile: Counter}, "by_question": {profile: {qid: class}}}``."""
    rows: list[dict] = []
    for profile, group in runs.items():
        for index, run in enumerate(group, start=1):
            for row in run["results"]:
                label = classify(row)
                if label is not None:
                    rows.append({"profile": profile, "run": index, "id": row["id"], "type": row["type"],
                                 "class": label, "status": row.get("status"), "failures": row.get("golden_failures")})
    by_profile: dict[str, collections.Counter] = {p: collections.Counter() for p in runs}
    per_question: dict[str, dict[str, collections.Counter]] = {p: collections.defaultdict(collections.Counter)
                                                               for p in runs}
    for item in rows:
        by_profile[item["profile"]][item["class"]] += 1
        per_question[item["profile"]][item["id"]][item["class"]] += 1
    by_question = {p: {qid: counts.most_common(1)[0][0] for qid, counts in sorted(table.items())}
                   for p, table in per_question.items()}
    return {"rows": rows, "by_profile": {p: dict(c) for p, c in by_profile.items()}, "by_question": by_question,
            "n_runs": {p: len(group) for p, group in runs.items()}}


def render(result: dict) -> str:
    profiles = list(result["by_profile"])
    lines = ["# Golden-set miss classes", "", "Failed rows per profile and class (all valid runs, 34 questions each):", "",
             "| class | " + " | ".join(profiles) + " |", "|---|" + "---|" * len(profiles)]
    for label in CLASSES:
        lines.append(f"| {label} | " + " | ".join(str(result["by_profile"][p].get(label, 0)) for p in profiles) + " |")
    lines.append("| **failed rows** | " + " | ".join(
        f"**{sum(result['by_profile'][p].values())}** of {result['n_runs'][p] * 34}" for p in profiles) + " |")
    lines += ["", "Most frequent class per question (failed runs only):", "",
              "| profile | question: class |", "|---|---|"]
    for profile in profiles:
        cells = ", ".join(f"{qid}: {label}" for qid, label in result["by_question"][profile].items())
        lines.append(f"| {profile} | {cells or 'no failures'} |")
    lines += ["", f"Limitation: {LIMITATION}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--profiles", nargs="+", default=PROFILES)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where miss_classes.json and miss_classes.md are written (default: the run folder)")
    args = parser.parse_args(argv)
    runs = {name: load(args.directory, name) for name in args.profiles}
    missing = [name for name, group in runs.items() if not group]
    if missing:
        raise SystemExit(f"no valid runs found for: {', '.join(missing)} in {args.directory}")
    result = classify_runs(runs)
    result["limitation"] = LIMITATION
    out = args.output_dir or args.directory
    out.mkdir(parents=True, exist_ok=True)
    (out / "miss_classes.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    text = render(result)
    (out / "miss_classes.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
