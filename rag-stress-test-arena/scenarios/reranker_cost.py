import json, os, random, time
from utils import make_random_vectors, make_metadata, timed, avg
from scenarios.config import cfg


def fake_reranker(batch):
    time.sleep(0.05)
    return [random.random() for _ in batch]


def run_scenario(engines, out_path="results/reranker_cost.json"):
    os.makedirs("results", exist_ok=True)
    n = cfg.pick(20000, 2000)
    q = cfg.pick(200, 40)
    k = cfg.pick(50, 20)
    vectors = make_random_vectors(n)
    metadata = make_metadata(n)
    queries = make_random_vectors(q)

    print("[reranker_cost] Indexing...")
    for name, eng in engines.items():
        eng.index(vectors, metadata)
        eng.flush()

    results = {}
    for name, eng in engines.items():
        retrieval_lat, rerank_lat, e2e = [], [], []
        for q in queries:
            res, t_r = timed(eng.search, q, k)
            _, t_rr = timed(fake_reranker, [r.id for r in res])
            retrieval_lat.append(t_r)
            rerank_lat.append(t_rr)
            e2e.append(t_r + t_rr)
        results[name] = {
            "retrieval_avg_ms": avg(retrieval_lat),
            "rerank_avg_ms": avg(rerank_lat),
            "end_to_end_avg_ms": avg(e2e),
        }
        print(f"  engine={name}: {results[name]}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[reranker_cost] Saved to {out_path}")