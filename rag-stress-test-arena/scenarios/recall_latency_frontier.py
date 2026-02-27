import json, os
from utils import make_random_vectors, make_metadata, timed, recall_at_k, avg
from scenarios.config import cfg


def run_scenario(engines, out_path="results/recall_latency_frontier.json"):
    os.makedirs("results", exist_ok=True)
    n = cfg.pick(30000, 3000)
    q = cfg.pick(200, 40)
    vectors = make_random_vectors(n)
    metadata = make_metadata(n)
    queries = make_random_vectors(q)

    print("[recall_latency_frontier] Indexing...")
    for name, eng in engines.items():
        eng.index(vectors, metadata)
        eng.flush()

    ks = cfg.pick([5, 10, 20, 50], [5, 10, 20])
    results = {}
    for name, eng in engines.items():
        points = []
        for k in ks:
            latencies, recalls = [], []
            for q in queries:
                res, t = timed(eng.search, q, k)
                latencies.append(t)
                ids = [r.id for r in res]
                recalls.append(recall_at_k(ids, ids, k))
            points.append({"k": k, "avg_latency_ms": avg(latencies), "avg_recall": avg(recalls)})
            print(f"  engine={name} k={k}: {points[-1]}")
        results[name] = {"points": points}

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[recall_latency_frontier] Saved to {out_path}")