#!/usr/bin/env python
"""Re-apply the current golden checker to answers already saved by a previous run. No LLM, no API, no cost.

    python scripts/rescore_saved_answers.py                       # automotive (default)
    python scripts/rescore_saved_answers.py --write               # also write evals/automotive_rescored.json

The automotive eval stores each question's full answer and resolved citations, and its pass/fail is decided only by
``_check`` (faithfulness is reported, not gated). So when the *checker* changes, the saved run can be re-scored
exactly, without re-generating answers. This measures the checker change, not the model: the answers are the
2026-09-21 ones. It does not apply to evals/last_run.json, which keeps only a truncated answer snippet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_golden_eval import _check  # noqa: E402

RESULTS = ROOT / "evals" / "automotive_eval_results.json"
QUESTIONS = ROOT / "data" / "eval_golden" / "queries_automotive.json"
OUT = ROOT / "evals" / "automotive_rescored.json"

# Human reading of saved answers that the substring checker scores wrongly. Kept next to the score, never folded in.
REVIEW_NOTES = {
    "SH-03": "Checker pass is NOT a genuine pass: the answer declines to give one value, reporting a conflict between "
             "95 % and 98 % taken from the entity context. 98% comes from the superseded CSR-CLIENT-2021 / a different "
             "KPI line in CSR-CLIENT-2023. Root cause: superseded or unrelated KPI facts reach the answer context.",
    "CON-05": "Genuine failure: answer says the context has no information (in English) although has_context is true. "
              "Also the question premise (RFA-REG-01 says annual) is not supported by the corpus: RFA-REG-01-rev5 "
              "line 74 also says critical suppliers are re-evaluated semestrially. Label left unchanged; flagged.",
}


def rescore(results_path: Path = RESULTS, questions_path: Path = QUESTIONS) -> dict:
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    specs = {q["id"]: q for q in json.loads(questions_path.read_text(encoding="utf-8"))["questions"]}
    rows = []
    for item in saved["golden"]["questions"]:
        spec = specs.get(item["id"])
        if spec is None or "answer" not in item:
            rows.append({"id": item["id"], "skipped": "no spec or no saved answer"})
            continue
        ok, failures = _check(spec, {"answer": item["answer"], "citations": item.get("citations") or []})
        rows.append({"id": item["id"], "type": item.get("type"), "saved_passed": bool(item.get("passed")),
                     "rescored_passed": ok, "changed": ok != bool(item.get("passed")), "failures": failures})
    scored = [r for r in rows if "skipped" not in r]
    return {
        "source_run_at": saved.get("run_at"),
        "saved_pass_rate": saved["golden"].get("pass_rate"),
        "rescored_passed": sum(r["rescored_passed"] for r in scored),
        "total": len(scored),
        "rescored_pass_rate": round(sum(r["rescored_passed"] for r in scored) / len(scored), 4) if scored else None,
        "changed": [r["id"] for r in scored if r["changed"]],
        "checker_passes_judged_not_genuine": [i for i in REVIEW_NOTES
                                              if any(r["id"] == i and r["rescored_passed"] for r in scored)],
        "review_notes": REVIEW_NOTES,
        "questions": rows,
        "note": "Saved answers re-checked with the current checker. No new generation; answers are from source_run_at.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    report = rescore()
    print(json.dumps({k: v for k, v in report.items() if k != "questions"}, indent=2, ensure_ascii=False))
    for r in report["questions"]:
        if r.get("failures"):
            print(f"  still failing {r['id']}: {r['failures']}")
    if args.write:
        OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
