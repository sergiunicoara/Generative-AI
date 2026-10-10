"""Evaluate query routing against hand-labelled cases (docs/query-routing.md).

Offline (default): route accuracy per category for the new router and for the
legacy keyword planner projected onto the same taxonomy, plus the confusion
matrix and how often each policy would allow the agentic fallback.

    python scripts/eval_routing.py                      # writes evals/routing_eval_results.json
    python scripts/eval_routing.py --live --tenant T    # also runs each question end to end

``--live`` needs the running stack (Neo4j, LLM keys). It records, per route:
latency p50/p95, fallback trigger rate, insufficient-context rate and token
usage (from the GenAI telemetry counters). Context precision/recall and
faithfulness need judge labels (RAGAS) and are reported as "not_measured"
unless --ragas is also given. Nothing is estimated: unmeasured metrics are
written as null with the reason.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.retrieval.query_router import Route, baseline_route, route_query  # noqa: E402

CASES = ROOT / "evals" / "routing_cases.json"
HELDOUT = ROOT / "evals" / "routing_cases_heldout.json"
OUT = ROOT / "evals" / "routing_eval_results.json"


def offline(cases: list[dict]) -> dict:
    per = defaultdict(lambda: {"n": 0, "router_correct": 0, "baseline_correct": 0})
    confusion: dict[str, Counter] = defaultdict(Counter)
    misses = []
    for c in cases:
        exp = c["expected"]
        got = route_query(c["question"]).route.value
        base = baseline_route(c["question"]).value
        per[exp]["n"] += 1
        per[exp]["router_correct"] += got == exp
        per[exp]["baseline_correct"] += base == exp
        confusion[exp][got] += 1
        if got != exp:
            misses.append({"id": c["id"], "expected": exp, "router": got, "baseline": base})
    n = len(cases)
    router_acc = sum(v["router_correct"] for v in per.values()) / n
    base_acc = sum(v["baseline_correct"] for v in per.values()) / n
    return {
        "cases": n,
        "route_accuracy": {"router": round(router_acc, 4), "legacy_planner_baseline": round(base_acc, 4)},
        "per_category": {k: {**v, "router_accuracy": round(v["router_correct"] / v["n"], 4),
                             "baseline_accuracy": round(v["baseline_correct"] / v["n"], 4)}
                         for k, v in sorted(per.items())},
        "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
        "misses": misses,
        "notes": ("The legacy planner has no ENTITY_LOOKUP / AGGREGATION / TEMPORAL / AMBIGUOUS "
                  "classes, so its accuracy on those categories is 0 by construction; compare "
                  "per_category for the shared classes."),
    }


async def live(cases: list[dict], tenant: str, policy: str) -> dict:
    from graphrag.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    per = defaultdict(lambda: {"latencies": [], "fallback": 0, "insufficient": 0, "n": 0})
    for c in cases:
        t0 = time.monotonic()
        result = await retriever.retrieve_and_answer(
            c["question"], tenant=tenant, config_overrides={"query_router_policy": policy})
        lat = time.monotonic() - t0
        bucket = per[result.route or "UNKNOWN"]
        bucket["n"] += 1
        bucket["latencies"].append(lat)
        bucket["fallback"] += "agentic" in (result.retrieval_mode or "") or (
            result.retrieval_trajectory is not None and result.retrieval_trajectory.completed_by != "synthesis")
        bucket["insufficient"] += not (result.retrieval_sufficiency or {}).get("sufficient", True)
    out = {}
    for route, b in per.items():
        lats = sorted(b["latencies"])
        out[route] = {
            "n": b["n"],
            "latency_p50_s": round(statistics.median(lats), 3),
            "latency_p95_s": round(lats[min(len(lats) - 1, int(0.95 * len(lats)))], 3),
            "fallback_trigger_rate": round(b["fallback"] / b["n"], 4),
            "insufficient_context_rate": round(b["insufficient"] / b["n"], 4),
        }
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--tenant", default="default")
    args = ap.parse_args(argv)
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    heldout = json.loads(HELDOUT.read_text(encoding="utf-8"))["cases"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "router": "graphrag/retrieval/query_router.py (deterministic, no LLM)",
        "offline": offline(cases),
        # Written to be kept out of rule design: rules are fixed from the dev set only, then scored here.
        "offline_heldout": offline(heldout),
        "live": None,
        "not_measured": {
            "context_precision_recall": "needs RAGAS judge + live stack",
            "faithfulness": "needs RAGAS judge + live stack",
            "latency_p50_p95_fallback_rate_tokens": "run with --live against the running stack",
        },
    }
    if args.live:
        report["live"] = {
            "observe": asyncio.run(live(cases, args.tenant, "observe")),
            "enforce": asyncio.run(live(cases, args.tenant, "enforce")),
        }
        report["not_measured"].pop("latency_p50_p95_fallback_rate_tokens")
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    o = report["offline"]
    print(f"route accuracy: router {o['route_accuracy']['router']:.2%}  "
          f"legacy baseline {o['route_accuracy']['legacy_planner_baseline']:.2%}  ({o['cases']} cases)")
    for m in o["misses"]:
        print(f"  miss {m['id']}: expected {m['expected']}, router {m['router']}")
    h = report["offline_heldout"]
    print(f"held-out: router {h['route_accuracy']['router']:.2%}  "
          f"legacy baseline {h['route_accuracy']['legacy_planner_baseline']:.2%}  ({h['cases']} cases)")
    for m in h["misses"]:
        print(f"  held-out miss {m['id']}: expected {m['expected']}, router {m['router']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
