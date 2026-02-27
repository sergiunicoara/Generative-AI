import json, threading, time, os
from typing import Dict, List
from utils import make_random_vectors, make_metadata, timed, p95, p99, avg
from scenarios.config import cfg


def run_scenario(engines, out_path="results/concurrency_burst.json"):
    os.makedirs("results", exist_ok=True)
    n = cfg.pick(5000, 500)
    q = cfg.pick(200, 40)
    vectors = make_random_vectors(n)
    metadata = make_metadata(n)
    queries = make_random_vectors(q)

    print("[concurrency_burst] Indexing...")
    for name, eng in engines.items():
        eng.index(vectors, metadata)
        eng.flush()

    levels = cfg.pick([1, 5, 10, 20, 50], [1, 5, 10])
    results = {}

    for name, eng in engines.items():
        results[name] = {}
        for level in levels:
            latencies = []

            def worker():
                for q in queries:
                    _, t = timed(eng.search, q, 10)
                    latencies.append(t)

            threads = [threading.Thread(target=worker) for _ in range(level)]
            start = time.time()
            for t in threads: t.start()
            for t in threads: t.join()
            duration = time.time() - start

            results[name][str(level)] = {
                "avg_ms": avg(latencies),
                "p95_ms": p95(latencies),
                "p99_ms": p99(latencies),
                "throughput_qps": len(queries) * level / duration,
            }
            print(f"  engine={name} level={level}: {results[name][str(level)]}")

    with open(out_path, "w") as f:
        json.dump({"levels": levels, "results": results}, f, indent=2)
    print(f"[concurrency_burst] Saved to {out_path}")