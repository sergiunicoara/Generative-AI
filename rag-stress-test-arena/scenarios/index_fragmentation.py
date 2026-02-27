import json, os
from utils import make_random_vectors, make_metadata, timed, p95, p99, avg
from scenarios.config import cfg


def run_scenario(engines, out_path="results/index_fragmentation.json"):
    os.makedirs("results", exist_ok=True)
    base_n = cfg.pick(10000, 2000)
    q = cfg.pick(300, 60)
    insert_n = cfg.pick(5000, 1000)
    delete_n = cfg.pick(3000, 500)
    base_vectors = make_random_vectors(base_n)
    base_metadata = make_metadata(base_n)
    queries = make_random_vectors(q)

    print("[index_fragmentation] Indexing base...")
    for name, eng in engines.items():
        eng.index(base_vectors, base_metadata)
        eng.flush()

    results = {}
    for name, eng in engines.items():
        base_lat = [timed(eng.search, q, 10)[1] for q in queries]

        eng.insert(make_random_vectors(insert_n), make_metadata(insert_n))
        eng.delete([m["id"] for m in base_metadata[:delete_n]])
        eng.flush()

        frag_lat = [timed(eng.search, q, 10)[1] for q in queries]
        results[name] = {
            "baseline_avg_ms": avg(base_lat), "baseline_p95_ms": p95(base_lat),
            "frag_avg_ms": avg(frag_lat), "frag_p95_ms": p95(frag_lat), "frag_p99_ms": p99(frag_lat),
        }
        print(f"  engine={name}: {results[name]}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[index_fragmentation] Saved to {out_path}")