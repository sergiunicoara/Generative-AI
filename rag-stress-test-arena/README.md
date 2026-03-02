# rag-stress-test-arena

Stress-test suite proving **why most RAG benchmarks are lying to you**.

This repo focuses on **realistic RAG workloads**, using a **Clustered Semantic Distribution** (768d) rather than synthetic noise. It measures how vector engines behave under real pressure — and forces you to compare them fairly.

---

## 🏟️ The "Real World" Difference
Unlike standard benchmarks that use synthetic noise or hand-crafted GMM distributions, this suite uses **actual text embeddings from Wikipedia**:

| Property | This benchmark | Typical benchmark |
|---|---|---|
| Data source | wikimedia/wikipedia 20231101.en (6.5 M articles) | Uniform random / hand-crafted GMM |
| Embedding model | sentence-transformers/all-mpnet-base-v2 (768-d) | Random Gaussian |
| Cluster structure | Emergent from real topics (science, history, sport, …) | Artificial, symmetric |
| Cluster density | Non-uniform — dense topics vs. sparse topics | Uniform by construction |
| Corpus size | 20 000 articles (indexed) | Varies |
| Query set | 3 000 held-out articles — **never indexed** | Random probes |
| Reproducibility | SHA-256 checksums printed after generation | Seed-only |

Real embeddings expose issues that synthetic distributions hide:
- **Anisotropic clusters** — topics are elongated in embedding space, not spherical.
- **Hierarchical density** — "Machine learning" sub-topics are far denser than "Ancient numismatics".
- **True query/corpus overlap** — held-out queries share semantic neighbours with indexed articles, exactly as in production RAG.

This forces HNSW to navigate the same kind of dense semantic neighbourhoods it faces in a real deployment.

---

## Engines

All four engines run under **identical conditions** — same HNSW build parameters, same distance metric, same corpus — so results reflect engine quality, not config choices.

| Engine        | Index Config              | Distance | ef_search range tested |
|---------------|---------------------------|----------|------------------------|
| Qdrant        | m=16, ef_construction=100 | Cosine   | 10 → 200               |
| Elasticsearch | m=16, ef_construction=100 | Cosine   | 10 → 200               |
| pgvector      | m=16, ef_construction=100 | Cosine   | 10 → 200               |
| Redis Stack   | m=16, ef_construction=100 | Cosine   | 10 → 200               |

---

## Scenario 1 — Recall-Latency Frontier

The industry-standard way to compare ANN engines is the **recall-latency frontier**: sweep `ef_search` from low to high and plot recall vs. latency. A better engine sits **upper-left** (higher recall, lower latency) at every operating point.

Each cell shows `recall@10 / avg latency`. Corpus: 20,000 × 768d **real Wikipedia embeddings** (all-mpnet-base-v2). Ground truth: brute-force cosine similarity.

| Engine            | ef=10              | ef=20              | ef=50              | ef=100             | ef=200             |
|-------------------|--------------------|--------------------|--------------------|--------------------|---------------------|
| **Redis Stack**   | 0.8218 / 2.84 ms   | 0.9212 / 2.97 ms   | 0.9784 / 3.45 ms   | 0.9926 / 3.59 ms   | 0.9976 / 4.52 ms   |
| **Qdrant**        | 0.9368 / 20.32 ms  | 0.9812 / 20.74 ms  | 0.9978 / 22.29 ms  | 0.9998 / 22.22 ms  | 1.0000 / 24.28 ms  |
| **Elasticsearch** | 0.9191 / 26.80 ms  | 0.9719 / 25.84 ms  | 0.9930 / 33.08 ms  | 0.9983 / 29.24 ms  | 0.9995 / 29.67 ms  |
| **pgvector**      | 0.8802 / 41.25 ms  | 0.9458 / 42.99 ms  | 0.9876 / 47.45 ms  | 0.9954 / 47.52 ms  | 0.9984 / 50.72 ms  |

> Plot: `results/summary/frontier.png`

### Key Findings

- **Redis Stack is 5–10× faster than everything else** — 3–5 ms vs 20–51 ms. Clear winner for latency-sensitive production serving.
- **Real data exposes recall ceilings** — on synthetic data all engines hit 100% at ef=200. On real Wikipedia embeddings, Redis Stack caps at **99.76%** and pgvector at **99.84%** even at maximum ef. Real topic clusters create hard negatives that HNSW misses permanently.
- **Qdrant is the only engine to reach 100% recall** — at ef=200 it achieves perfect recall while maintaining consistent ~22ms latency.
- **Clustering bonus:** All engines show *negative* latency penalty vs. random-baseline — real semantic clusters actually make HNSW **faster** (fewer hops to navigate coherent topic neighbourhoods vs. featureless random space).
- **Flat latency curves (ES, pgvector):** HTTP/connection overhead dominates. Raising ef is essentially free recall.

---

## Scenario 2 — Concurrency Burst

Simulates a production traffic spike: 1 → 5 → 10 → 20 → 50 concurrent clients each issuing 200 queries.

| Engine            | 1 thread | 5 threads | 10 threads | 20 threads | 50 threads | Peak QPS |
|-------------------|----------|-----------|------------|------------|------------|----------|
| **Redis Stack**   | 3.8 ms   | 6.5 ms    | 11.5 ms    | 23.1 ms    | 55.9 ms    | **845**  |
| **Qdrant**        | 20.6 ms  | 37.7 ms   | 65.0 ms    | 123.7 ms   | 349.5 ms   | 159      |
| **pgvector**      | 58.2 ms  | 66.4 ms   | 115.2 ms   | 196.1 ms   | 488.6 ms   | 101      |
| **Elasticsearch** | 27.0 ms  | 61.3 ms   | 128.2 ms   | 261.7 ms   | 653.8 ms   | 81       |

> Plot: `plots/concurrency_avg_ms.png`, `plots/concurrency_p95_ms.png`

### Key Findings

- **Redis dominates throughput at 845 QPS** — single-threaded event loop scales with concurrency far better than multi-process engines.
- **Elasticsearch bottlenecks hardest** — QPS plateaus at 74 after just 5 threads; adding more clients only inflates latency, not throughput.
- **pgvector scales better than expected** — connection pooling keeps throughput growing to 50 threads (101 QPS peak), but latency blows out to 489ms.
- **Qdrant peaks at 10–20 threads** (151–159 QPS) then degrades — HNSW graph locks become contention points at extreme concurrency.

---

## Scenario 3 — Metadata Filter Penalty

Measures the additional latency cost of adding a tag filter (`tag = "tag-0"`, selectivity ~2%) to every query. Uses 20k corpus with 50 unique tags.

| Engine            | No-filter avg | Filter avg | Penalty    | No-filter p95 | Filter p95 |
|-------------------|---------------|------------|------------|---------------|------------|
| **Redis Stack**   | 3.0 ms        | 3.0 ms     | **0%**     | 4.5 ms        | 5.0 ms     |
| **Qdrant**        | 25.1 ms       | 27.4 ms    | +9%        | 35.5 ms       | 43.9 ms    |
| **Elasticsearch** | 24.9 ms       | 21.9 ms    | **−12%**   | 37.8 ms       | 28.4 ms    |
| **pgvector**      | 44.2 ms       | 44.9 ms    | +2%        | 60.3 ms       | 62.5 ms    |

> Plot: `plots/metadata_filter_penalty_p95.png`

### Key Findings

- **Redis: zero filter penalty** — the tag index in Redis Search is a native inverted index; pre-filtering adds no overhead.
- **Elasticsearch gets *faster* with filters** — the KNN filter is applied pre-graph-traversal, pruning candidates early and reducing computation.
- **Qdrant pays a modest 9% penalty** — HNSW filtered search requires an extra post-filter scan pass.
- **pgvector near parity** — sequential scan dominates regardless of filter.

---

## Scenario 4 — Index Fragmentation

Measures latency before and after inserting 5,000 new vectors + deleting 3,000 old ones from a 10k base corpus — simulating a live production index with continuous writes.

| Engine            | Baseline avg | Post-frag avg | Delta    | Baseline p95 | Post-frag p95 |
|-------------------|--------------|---------------|----------|--------------|---------------|
| **Redis Stack**   | 4.6 ms       | 3.3 ms        | **−28%** | 6.5 ms       | 5.2 ms        |
| **Qdrant**        | 25.1 ms      | 20.1 ms       | **−20%** | 34.9 ms      | 34.7 ms       |
| **pgvector**      | 42.8 ms      | 43.3 ms       | +1%      | 56.8 ms      | 58.9 ms       |
| **Elasticsearch** | 20.6 ms      | 29.6 ms       | **+44%** | 24.1 ms      | 42.0 ms       |

> Plot: `plots/index_fragmentation_p95.png`

### Key Findings

- **Elasticsearch degrades hardest (+44%)** — segment fragmentation from bulk insert/delete forces expensive multi-segment merges during query time.
- **Redis and Qdrant get *faster* after writes** — both engines perform internal graph rebalancing/compaction, improving traversal efficiency. Counterintuitive but confirmed across runs.
- **pgvector is fragmentation-neutral** — sequential scan is unaffected by structural changes to the index.

---

## Scenario 5 — Reranker Cost Breakdown

Models a realistic two-stage RAG pipeline: vector retrieval (top-50) followed by a cross-encoder reranker (simulated at 50ms fixed cost). Shows how retrieval speed matters once a reranker is added.

| Engine            | Retrieval avg | Rerank avg | End-to-end avg | Retrieval share |
|-------------------|---------------|------------|----------------|-----------------|
| **Redis Stack**   | 4.8 ms        | 50.6 ms    | 55.4 ms        | 9%              |
| **Qdrant**        | 32.7 ms       | 50.7 ms    | 83.4 ms        | 39%             |
| **pgvector**      | 58.8 ms       | 50.8 ms    | 109.6 ms       | 54%             |
| **Elasticsearch** | 106.7 ms      | 51.4 ms    | 158.0 ms       | 67%             |

> Plot: `plots/reranker_cost.png`

### Key Findings

- **Reranking equalises slow engines — but not fast ones.** Once you add a 50ms reranker, Elasticsearch's 107ms retrieval becomes the dominant cost (67% of e2e). Redis's 5ms retrieval still delivers the lowest e2e latency.
- **Redis: reranker is the bottleneck (91% of e2e).** If you use Redis + reranker, optimising the reranker matters far more than tuning HNSW.
- **Practical takeaway:** If your pipeline includes a reranker ≥ 30ms, pgvector becomes more cost-competitive (e2e 110ms vs Qdrant 83ms) because the retrieval gap narrows.

---

## Methodology & Reproducibility

To ensure scientific defensibility, all benchmarks follow these constraints:

- **Real corpus:** 20 000 Wikipedia articles encoded with `all-mpnet-base-v2` (768-d, L2-normalised).
- **Real queries:** 3 000 held-out Wikipedia articles — never present in the indexed corpus.
- **Identical conditions:** All engines use the same HNSW params (m=16, ef_construction=100) and cosine distance.
- **Warmup:** 500 queries per engine before measurement to ensure JIT/buffer warming.
- **Steady-state:** 500 queries per ef_search level (5 levels × 4 engines for the full frontier sweep).
- **Averaging:** Results are the mean of 3 independent runs per level.
- **Ground truth:** Brute-force cosine similarity on pre-normalised 768d vectors.
- **Confidence Intervals:** P95 and P99 variance tracked in raw JSON outputs.
- **Reproducibility:** SHA-256 checksums of corpus and query files printed at generation time.

---

## Hardware & Environment

**Reference Machine:** Intel64 Family 6 Model 140 Stepping 1, GenuineIntel (4 physical cores, 8 threads)

**RAM:** 23.8 GB

**Storage:** 237.2 GB Disk Capacity

**Docker Limits:**
- Engines: No hard CPU/RAM limits (Full host utilization).
- Network: Bridge mode (Localhost).

---

## How to Run

```bash
# 0. Install corpus-generation dependencies (one-time)
pip install sentence-transformers datasets
pip install "optimum[onnxruntime]"   # optional: 3-4× faster CPU encoding

# 1. Generate the Wikipedia embedding corpus (deterministic — same result every time)
#    Downloads wikimedia/wikipedia 20231101.en and encodes with all-mpnet-base-v2.
#    Prints SHA-256 checksums for bit-for-bit reproducibility verification.
py -3.11 generate_distribution.py

# 2. Start all four engines
docker-compose up -d

# 3. Run all 5 benchmark scenarios
py -3.11 run_all.py

# 4. Generate all plots
py -3.11 scripts/plot_results.py   # concurrency, filter, fragmentation, reranker plots
py -3.11 plot_frontier.py          # recall-latency frontier plot
```

Results are saved to `results/` and `results/summary/`. Plots saved to `plots/` and `results/summary/`.

> **Corpus generation details:**
> `generate_distribution.py` shuffles the first 500 000 Wikipedia articles with a fixed seed (42),
> encodes 20 000 as the indexed corpus and 3 000 as a held-out query set using
> `sentence-transformers/all-mpnet-base-v2`.  Both output files are L2-normalised float32 arrays.
> The snapshot (`20231101.en`) is immutable — combined with the fixed seed and frozen model weights,
> the output is bit-for-bit reproducible across machines.

---

## How to Interpret Results

- **Recall-latency frontier:** The curve that sits upper-left wins. A single operating point (e.g. ef=100) is misleading — always compare the full curve.
- **Flat latency curves (ES, pgvector):** When latency doesn't scale with ef_search, non-graph overhead (HTTP, connection pooling, query parsing) is the bottleneck. Raising ef is free recall.
- **Steep latency curves (Redis, Qdrant at low ef):** Graph traversal is the bottleneck — these engines are doing real work. At high ef they reveal their true ceiling.
- **P95 vs Avg:** If your P95 is >2× your average, your system will feel "jittery" to users.
- **Reranker pipelines:** Engine retrieval speed matters less once a slow reranker dominates. Choose your engine based on the full pipeline latency budget.

---

## Adding Your Own Engine

1. Create `engines/my_engine.py` extending `BaseEngine`.
2. Register it in `engine_loader.py` and add its params to `config.yml`.
3. Run `py -3.11 run_all.py`.

---

## Results & Artifacts

| File | Contents |
|---|---|
| `results/summary/frontier_results.json` | Recall + latency per engine per ef_search level |
| `results/summary/frontier.png` | Recall-latency frontier plot |
| `results/concurrency_burst.json` | QPS + latency at 1/5/10/20/50 concurrent clients |
| `results/metadata_filter_penalty.json` | Filter vs. no-filter latency per engine |
| `results/index_fragmentation.json` | Baseline vs. post-write latency per engine |
| `results/reranker_cost.json` | Retrieval + rerank + end-to-end latency |
| `plots/` | All scenario plots (PNG) |

---

## Non-Goals

- This is **not** a capacity test (billions of vectors).
- This is **not** a cost-per-query analysis.
- This measures **HNSW search quality and latency** under a realistic clustered distribution.

---

## License

MIT License
