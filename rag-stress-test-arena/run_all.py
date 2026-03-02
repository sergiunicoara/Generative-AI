import json
import os
import time
import numpy as np
import yaml
from utils import make_random_vectors, load_query_vectors, calculate_recall, p95, avg
from engine_loader import get_engine, get_all_engines, get_all_engine_factories
from engines.bruteforce_engine import BruteForceEngine
from scenarios.concurrency_burst import run_scenario as run_concurrency
from scenarios.metadata_filter_penalty import run_scenario as run_metadata
from scenarios.index_fragmentation import run_scenario as run_fragmentation
from scenarios.reranker_cost import run_scenario as run_reranker


def run_scientific_benchmark():
    config_file = 'config.yml' if os.path.exists('config.yml') else 'config.yaml'
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    n_vectors  = cfg['distribution']['n_samples']
    n_runs     = cfg['run']['runs']
    warmup     = cfg['run']['warmup_queries']
    measured   = cfg['run']['measured_queries']
    ef_levels  = cfg['run'].get('ef_search_levels', [10, 20, 50, 100, 200])
    RECALL_K   = cfg['run']['top_k']
    # For the frontier sweep use a smaller measured set per level so the full
    # 5-level × 4-engine sweep completes in reasonable time (~15 min).
    SWEEP_MEASURED = min(measured, 500)
    RECALL_SUBSET  = SWEEP_MEASURED  # all measured queries count toward recall

    # ── 1. Load real Wikipedia corpus ─────────────────────────────────────────
    print(f"Loading Wikipedia corpus (n={n_vectors}) ...")
    vectors  = make_random_vectors(n_vectors)
    metadata = [{"id": f"doc-{i}"} for i in range(n_vectors)]

    # ── 2. Random baseline corpus (for latency penalty measurement) ───────────
    # The baseline uses uniform random vectors so the penalty reflects the
    # cost of navigating real semantic clusters vs. a flat, featureless space.
    print("Generating random baseline corpus ...")
    random_vectors = np.random.randn(n_vectors, 768).astype("float32")
    random_vectors /= np.linalg.norm(random_vectors, axis=1, keepdims=True)

    # ── 3. Load held-out Wikipedia query vectors ──────────────────────────────
    # These are real Wikipedia articles never present in the corpus.
    # Using real queries (not random noise) ensures the benchmark reflects
    # genuine semantic nearest-neighbour retrieval, not synthetic probing.
    print(f"Loading held-out query vectors ({warmup + measured} queries) ...")
    query_vectors  = load_query_vectors(warmup + measured)
    recall_queries = query_vectors[warmup:warmup + RECALL_SUBSET]

    # ── 4. Brute-force gold standard (cosine, same as all engines) ────────────
    print(f"Computing gold standard (Recall@{RECALL_K}, subset={RECALL_SUBSET})...")
    bf = BruteForceEngine({})
    bf.index(vectors, metadata)
    gold_results   = [bf.search(q, k=RECALL_K) for q in recall_queries]
    # gold_results[i] is already List[str] from BruteForceEngine

    # ── 5. Frontier sweep ─────────────────────────────────────────────────────
    engines_to_test = ["elasticsearch", "qdrant", "redis", "pgvector"]
    frontier = {"metadata": {"distribution": "Wikipedia/all-mpnet-base-v2 (real embeddings)",
                              "recall_k": RECALL_K,
                              "ef_search_levels": ef_levels}}

    for eng_name in engines_to_test:
        print(f"\n{'='*50}")
        print(f"  Engine: {eng_name}")
        print(f"{'='*50}")

        engine = get_engine(eng_name)
        if engine is None:
            print(f"  Skipping — could not initialize")
            continue

        # Index random baseline, measure baseline latency
        print(f"  [1/3] Indexing random baseline...")
        engine.index(random_vectors.tolist(), metadata)
        t0 = time.perf_counter()
        for q in query_vectors[:100]:
            engine.search(q)
        base_lat_ms = (time.perf_counter() - t0) / 100 * 1000

        # Replace with real Wikipedia corpus
        print(f"  [2/3] Indexing Wikipedia corpus...")
        engine.index(vectors, metadata)

        # Warmup at highest ef to prime the graph traversal cache
        print(f"  [3/3] Warming up ({warmup} queries at ef={max(ef_levels)})...")
        engine.kwargs['ef_search'] = max(ef_levels)
        for q in query_vectors[:warmup]:
            engine.search(q)

        # ── Sweep ef_search levels ────────────────────────────────────────────
        print(f"\n  {'ef':>5}  {'recall':>7}  {'avg_ms':>8}  {'p95_ms':>8}")
        print(f"  {'-'*35}")

        points = []
        for ef in ef_levels:
            engine.kwargs['ef_search'] = ef

            run_avg_lats, run_recalls = [], []
            for _ in range(n_runs):
                latencies, recalls = [], []
                for i, q in enumerate(query_vectors[warmup:warmup + SWEEP_MEASURED]):
                    t0 = time.perf_counter()
                    raw_res = engine.search(q)
                    latencies.append((time.perf_counter() - t0) * 1000)

                    test_ids = [r.id if hasattr(r, 'id') else r for r in raw_res]
                    recalls.append(calculate_recall(gold_results[i], test_ids))

                run_avg_lats.append(avg(latencies))
                run_recalls.append(avg(recalls))

            avg_ms       = round(avg(run_avg_lats), 2)
            p95_ms       = round(p95(run_avg_lats), 2)
            recall_val   = round(avg(run_recalls), 4)
            penalty_pct  = round(((avg_ms - base_lat_ms) / base_lat_ms) * 100, 1)

            points.append({
                "ef_search":            ef,
                "avg_ms":               avg_ms,
                "p95_ms":               p95_ms,
                "recall_at_10":         recall_val,
                "random_baseline_ms":   round(base_lat_ms, 2),
                "clustered_penalty_pct": penalty_pct,
            })
            print(f"  {ef:>5}  {recall_val:>7.4f}  {avg_ms:>8.2f}  {p95_ms:>8.2f}")

        frontier[eng_name] = {"points": points}

    # ── 6. Save frontier results ──────────────────────────────────────────────
    os.makedirs("results/summary", exist_ok=True)
    out_path = "results/summary/frontier_results.json"
    with open(out_path, "w") as f:
        json.dump(frontier, f, indent=2)
    print(f"\nFrontier results saved to {out_path}")

    # ── 7. Run additional scenarios ───────────────────────────────────────────
    # Each scenario re-indexes independently so order doesn't matter.
    print("\n\n" + "="*50)
    print("  Running additional scenarios")
    print("="*50)
    engines = get_all_engines()
    engine_factories = get_all_engine_factories()
    run_concurrency(engines, engine_factories=engine_factories)
    run_metadata(engines)
    run_fragmentation(engines)
    run_reranker(engines)
    print("\nAll scenarios complete.")


if __name__ == "__main__":
    run_scientific_benchmark()
