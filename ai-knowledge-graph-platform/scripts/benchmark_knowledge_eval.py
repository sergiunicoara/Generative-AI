"""Latency benchmark for the contextual-knowledge evaluator.

    python scripts/benchmark_knowledge_eval.py
    python scripts/benchmark_knowledge_eval.py --sizes 100 1000 5000 --requests 50

Seeded synthetic packs (``graphrag/context_graph/knowledge_synthetic.py``),
evaluated in process: no database, network or model.  Reports per-request
p50/p95/max and the share of time per stage, with pack size, request count,
hardware and commit, and writes ``evals/knowledge_eval/benchmark.json``.
Budgets are stated per pack size and the exit code is 1 when one is missed.
The numbers are local measurements on this machine, not a service SLO.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.context_graph.knowledge_evaluation import KnowledgeReleaseRef, evaluate_knowledge  # noqa: E402
from graphrag.context_graph.knowledge_synthetic import (  # noqa: E402
    synthetic_context, synthetic_facts, synthetic_items,
)
from graphrag.enterprise.models import AccessContext  # noqa: E402

# p95 budget in milliseconds per pack size (in-process, single request)
BUDGET_MS = {100: 25.0, 1000: 250.0, 5000: 1500.0}


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(size: int, requests: int, seed: int = 11) -> dict:
    items = synthetic_items(size, seed=seed)
    access = AccessContext.from_claims({"sub": "benchmark"})
    ref = KnowledgeReleaseRef(release_id=f"bench-{size}", domain="synthetic", version=1)
    rng = random.Random(seed)
    latencies: list[float] = []
    stages: dict[str, float] = {}
    applicable = 0
    for _ in range(requests):
        context, facts = synthetic_context(rng), synthetic_facts(rng)
        started = perf_counter()
        decision = evaluate_knowledge(items=items, context=context, facts=facts, tenant="syn", access=access,
                                      domain="synthetic", release=ref, timings=stages)
        latencies.append((perf_counter() - started) * 1000)
        applicable += len(decision.applicable)
    total = sum(stages.values()) or 1.0
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    budget = BUDGET_MS.get(size)
    return {
        "pack_size": size, "requests": requests,
        "latency_ms": {"p50": round(median(ordered), 2), "p95": round(p95, 2), "max": round(ordered[-1], 2)},
        "stage_share": {stage: round(seconds / total, 3) for stage, seconds in sorted(stages.items())},
        "mean_applicable_items": round(applicable / requests, 1),
        "budget_p95_ms": budget, "within_budget": None if budget is None else p95 <= budget,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 5000])
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--output", type=Path, default=ROOT / "evals" / "knowledge_eval" / "benchmark.json")
    args = parser.parse_args()
    results = [run(size, args.requests) for size in args.sizes]
    report = {
        "commit": _commit(), "python": platform.python_version(), "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(), "system": platform.system(),
        "note": "local in-process measurement on synthetic packs; not a service SLO",
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in results:
        print(f"{row['pack_size']:>6} items  p50 {row['latency_ms']['p50']:>8} ms  p95 {row['latency_ms']['p95']:>8} ms  "
              f"budget {row['budget_p95_ms']}  ok={row['within_budget']}  stages={row['stage_share']}")
    return 0 if all(row["within_budget"] is not False for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
