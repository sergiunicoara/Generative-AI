#!/usr/bin/env python
"""
Golden GraphRAG evaluation set — regression runner.

Submits each question in evals/golden_set.json to the live API,
checks the answer against expected citations and required phrases,
and exits non-zero if the pass rate is below the configured threshold.

Usage:
    py -3.11 scripts/run_golden_eval.py                          # all questions
    py -3.11 scripts/run_golden_eval.py --type multi_hop         # subset by type
    py -3.11 scripts/run_golden_eval.py --ids MH-01 MH-02 SH-03  # specific IDs
    py -3.11 scripts/run_golden_eval.py --dry-run                 # print plan only

Requires:
    - API running on GRAPHRAG_API_URL (default http://localhost:8000)
    - Tenant seeded: py -3.11 scripts/seed_demo_data.py --commit

Exit codes:
    0  All checks passed and pass rate >= threshold
    1  One or more checks failed or pass rate below threshold
    2  API unreachable or fatal error
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import structlog

# Make project root importable
sys.path.insert(0, str(Path(__file__).parents[1]))

log = structlog.get_logger("golden_eval")

GOLDEN_SET  = Path(__file__).parents[1] / "evals" / "golden_set.json"
API_BASE    = __import__("os").getenv("GRAPHRAG_API_URL", "http://localhost:8000")
API_TIMEOUT = 60


def _submit_query(client: httpx.Client, question: str, tenant: str) -> dict:
    """Submit a query and poll until complete. Returns the result dict."""
    resp = client.post(f"{API_BASE}/query", json={"question": question, "tenant": tenant},
                       timeout=API_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    query_id = data.get("query_id", "")

    # Poll for result
    for _ in range(60):
        r = client.get(f"{API_BASE}/query/{query_id}", timeout=10)
        r.raise_for_status()
        result = r.json()
        if result.get("status") == "completed":
            return result
        time.sleep(1)

    return {"status": "timeout", "answer": "", "citations": []}


def _check(question_spec: dict, result: dict) -> tuple[bool, list[str]]:
    """
    Evaluate one result against the golden spec.
    Returns (passed: bool, failures: list[str]).
    """
    failures = []
    answer   = (result.get("answer") or "").lower()
    citations = [str(c) for c in (result.get("citations") or [])]

    # Required phrases
    for phrase in question_spec.get("answer_must_contain", []):
        if phrase.lower() not in answer:
            failures.append(f"answer missing required phrase: {phrase!r}")

    # Forbidden phrases
    for phrase in question_spec.get("answer_must_not_contain", []):
        if phrase.lower() in answer:
            failures.append(f"answer contains forbidden phrase: {phrase!r}")

    # Citation recall
    expected_cits = question_spec.get("expected_citations", [])
    if expected_cits:
        cited_str = " ".join(citations).lower()
        missing = [c for c in expected_cits if c.lower() not in cited_str]
        if missing:
            pct = len(missing) / len(expected_cits)
            threshold = question_spec.get("citation_recall_min",
                                          question_spec.get("thresholds", {}).get("citation_recall_min", 0.60))
            if pct > (1 - threshold):
                failures.append(f"insufficient citation recall — missing: {missing}")

    return len(failures) == 0, failures


def run(
    questions: list[dict],
    tenant: str,
    dry_run: bool,
    global_thresholds: dict,
) -> int:
    """Run the eval. Returns exit code."""

    if dry_run:
        print(f"\nDRY RUN — {len(questions)} questions would be submitted to {API_BASE}\n")
        for q in questions:
            print(f"  [{q['id']:8s}] ({q['type']:15s}) {q['question'][:80]}")
        return 0

    try:
        with httpx.Client() as client:
            client.get(f"{API_BASE}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"\n✗ API unreachable at {API_BASE}: {e}\n"
              "  Start the API: uvicorn api.main:app --port 8000\n", file=sys.stderr)
        return 2

    results  = []
    passed   = 0
    failed   = 0
    errored  = 0

    print(f"\n{'='*70}")
    print(f"  Golden GraphRAG Eval  |  {len(questions)} questions  |  tenant: {tenant}")
    print(f"  API: {API_BASE}")
    print(f"{'='*70}\n")

    with httpx.Client() as client:
        for q in questions:
            qid    = q["id"]
            qtype  = q["type"]
            label  = f"[{qid:8s}] ({qtype:15s})"
            try:
                result = _submit_query(client, q["question"], tenant)
                ok, failures = _check(q, result)

                if ok:
                    passed += 1
                    print(f"  ✓  {label} {q['question'][:55]}")
                else:
                    failed += 1
                    print(f"  ✗  {label} {q['question'][:55]}")
                    for f in failures:
                        print(f"       → {f}")

                results.append({
                    "id": qid, "type": qtype, "passed": ok,
                    "failures": failures,
                    "answer_snippet": (result.get("answer") or "")[:120],
                    "citations": result.get("citations", []),
                })
            except Exception as exc:
                errored += 1
                print(f"  ⚠  {label} ERROR: {exc}")
                results.append({"id": qid, "type": qtype, "passed": False,
                                 "failures": [str(exc)], "error": True})

    total       = passed + failed + errored
    pass_rate   = passed / total if total else 0
    min_pass    = global_thresholds.get("pass_rate_min", 0.80)

    print(f"\n{'='*70}")
    print(f"  Results:  {passed}/{total} passed  ({pass_rate:.0%})  "
          f"[threshold: {min_pass:.0%}]")
    if errored:
        print(f"  Errors:   {errored} questions failed to submit")

    # Per-type breakdown
    by_type: dict[str, list[bool]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r["passed"])
    print("\n  By type:")
    for qtype, outcomes in sorted(by_type.items()):
        n   = len(outcomes)
        ok  = sum(outcomes)
        print(f"    {qtype:20s}  {ok}/{n}  ({ok/n:.0%})")
    print(f"{'='*70}\n")

    # Write results file
    out_path = Path(__file__).parents[1] / "evals" / "last_run.json"
    out_path.write_text(json.dumps({
        "pass_rate": round(pass_rate, 4),
        "passed": passed, "failed": failed, "errored": errored,
        "by_type": {k: {"passed": sum(v), "total": len(v)} for k, v in by_type.items()},
        "questions": results,
    }, indent=2))
    print(f"  Results written to: {out_path}\n")

    if pass_rate < min_pass:
        print(f"✗ FAIL — pass rate {pass_rate:.0%} below threshold {min_pass:.0%}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type",    help="Filter by question type (e.g. multi_hop)")
    parser.add_argument("--ids",     nargs="+", help="Run specific question IDs only")
    parser.add_argument("--tenant",  default=None, help="Tenant override (default: from golden set)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without calling API")
    args = parser.parse_args()

    gs      = json.loads(GOLDEN_SET.read_text())
    tenant  = args.tenant or gs.get("tenant", "aerospace")
    all_qs  = gs["questions"]
    thresholds = gs.get("thresholds", {})

    # Filter
    if args.ids:
        all_qs = [q for q in all_qs if q["id"] in args.ids]
    if args.type:
        all_qs = [q for q in all_qs if q["type"] == args.type]

    if not all_qs:
        print("No questions matched the filter.", file=sys.stderr)
        sys.exit(1)

    sys.exit(run(all_qs, tenant=tenant, dry_run=args.dry_run,
                 global_thresholds=thresholds))


if __name__ == "__main__":
    main()
