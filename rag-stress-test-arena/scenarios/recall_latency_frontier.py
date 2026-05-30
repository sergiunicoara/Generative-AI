import json, os
import numpy as np
from utils import make_random_vectors, make_metadata, timed, recall_at_k, avg
from scenarios.config import cfg


def _brute_force_top_k(query: list, corpus: np.ndarray, k: int) -> list:
    """Return the doc-ids of the k exact nearest neighbours (cosine)."""
    q = np.array(query, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    norms = np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9
    scores = (corpus / norms) @ q_norm
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [f"doc-{i}" for i in top_indices.tolist()]


def run_scenario(engines, out_path="results/recall_latency_frontier.json"):
    os.makedirs("results", exist_ok=True)
    n = cfg.pick(30000, 3000)
    num_queries = cfg.pick(200, 40)
    vectors = make_random_vectors(n)
    metadata = make_metadata(n)
    queries = make_random_vectors(num_queries)

    print("[recall_latency_frontier] Indexing...")
    for name, eng in engines.items():
        eng.index(vectors, metadata)
        eng.flush()

    # Precompute brute-force ground truth once (shared across all engines/ks)
    corpus_np = np.array(vectors, dtype=np.float32)
    ks = cfg.pick([5, 10, 20, 50], [5, 10, 20])
    max_k = max(ks)
    print("[recall_latency_frontier] Computing brute-force ground truth...")
    ground_truth = [_brute_force_top_k(q, corpus_np, max_k) for q in queries]

    results = {}
    for name, eng in engines.items():
        points = []
        for k in ks:
            latencies, recalls = [], []
            for idx, q in enumerate(queries):
                res, t = timed(eng.search, q, k)
                latencies.append(t)
                retrieved_ids = [r.id for r in res]
                recalls.append(recall_at_k(ground_truth[idx], retrieved_ids, k))
            points.append({"k": k, "avg_latency_ms": avg(latencies), "avg_recall": avg(recalls)})
            print(f"  engine={name} k={k}: {points[-1]}")
        results[name] = {"points": points}

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[recall_latency_frontier] Saved to {out_path}")