import json, os
from utils import make_random_vectors, make_metadata, timed, p95, p99, avg
from scenarios.config import cfg


def run_scenario(engines, out_path="results/metadata_filter_penalty.json"):
    os.makedirs("results", exist_ok=True)
    n = cfg.pick(20000, 2000)
    q = cfg.pick(300, 60)
    num_tags = cfg.pick(50, 20)
    vectors = make_random_vectors(n)
    metadata = make_metadata(n, num_tags=num_tags)
    queries = make_random_vectors(q)

    print("[metadata_filter_penalty] Indexing...")
    for name, eng in engines.items():
        eng.index(vectors, metadata)
        eng.flush()

    results = {}
    for name, eng in engines.items():
        no_filter = [timed(eng.search, q, 10)[1] for q in queries]
        with_filter = [timed(eng.search_with_filter, q, 10, {"field": "tag", "value": "tag-0"})[1] for q in queries]
        results[name] = {
            "no_filter_avg_ms": avg(no_filter), "no_filter_p95_ms": p95(no_filter),
            "filter_avg_ms": avg(with_filter), "filter_p95_ms": p95(with_filter), "filter_p99_ms": p99(with_filter),
        }
        print(f"  engine={name}: {results[name]}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[metadata_filter_penalty] Saved to {out_path}")