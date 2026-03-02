import json, threading, time, os
from typing import Callable, Dict, List, Optional
from utils import make_random_vectors, make_metadata, timed, p95, p99, avg
from scenarios.config import cfg


def _make_worker(factory: Optional[Callable], primary_eng, queries: list, latencies: list):
    """Return a worker function that runs all queries and appends latencies.

    If a factory is provided each worker creates its own engine instance
    (own TCP connection / socket) so threads truly run as independent clients.
    Falls back to the shared primary engine when no factory is available.
    """
    def worker():
        if factory is not None:
            # True independent-client simulation: fresh engine per thread
            w_eng = factory(skip_index=True)
        else:
            w_eng = primary_eng
        try:
            for q in queries:
                _, t = timed(w_eng.search, q, 10)
                latencies.append(t)   # list.append is GIL-atomic in CPython
        finally:
            if factory is not None:
                w_eng.close()
    return worker


def run_scenario(engines, out_path="results/concurrency_burst.json",
                 engine_factories: Optional[Dict[str, Callable]] = None):
    """Concurrency-burst scenario.

    Parameters
    ----------
    engines:
        Dict of pre-indexed engine instances (used for indexing and as fallback).
    engine_factories:
        Dict of factory callables returned by ``get_engine_factory()``.
        When provided, each worker thread gets its own engine instance
        (independent TCP connection), which is the correct simulation of
        N independent clients.  When absent, all threads share the primary
        engine instance (old behaviour, kept for backwards compatibility).
    """
    os.makedirs("results", exist_ok=True)
    n = cfg.pick(5000, 500)
    q_count = cfg.pick(200, 40)
    vectors = make_random_vectors(n)
    metadata = make_metadata(n)
    queries = make_random_vectors(q_count)

    print("[concurrency_burst] Indexing...")
    for name, eng in engines.items():
        eng.index(vectors, metadata)
        eng.flush()

    levels = cfg.pick([1, 5, 10, 20, 50], [1, 5, 10])
    results = {}

    for name, eng in engines.items():
        results[name] = {}
        factory = (engine_factories or {}).get(name)
        if factory:
            print(f"  [{name}] using per-thread engine instances (independent clients)")
        else:
            print(f"  [{name}] no factory available — falling back to shared engine")

        for level in levels:
            latencies: List[float] = []

            threads = [
                threading.Thread(
                    target=_make_worker(factory, eng, queries, latencies),
                    daemon=True,
                )
                for _ in range(level)
            ]

            start = time.time()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            duration = time.time() - start

            results[name][str(level)] = {
                "avg_ms":         avg(latencies),
                "p95_ms":         p95(latencies),
                "p99_ms":         p99(latencies),
                "throughput_qps": len(queries) * level / duration,
            }
            print(f"  engine={name} level={level}: {results[name][str(level)]}")

    with open(out_path, "w") as f:
        json.dump({"levels": levels, "results": results}, f, indent=2)
    print(f"[concurrency_burst] Saved to {out_path}")
