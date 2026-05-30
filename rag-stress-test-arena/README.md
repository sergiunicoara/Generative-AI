# rag-stress-test-arena

Stress-test suite proving **why most RAG benchmarks are lying to you**.

This repo focuses on **realistic RAG workloads**, not lab demos. It measures how vector engines behave under:

- Concurrency (1 → 50 clients)
- Metadata filters
- Index mutations (upserts/deletes)
- Reranking overhead

and exposes where "average latency" and toy benchmarks are misleading.

---

## Engines

| Engine        | Archetype           |
|--------------|---------------------|
| Qdrant       | Speed King          |
| Elasticsearch| Reliable Workhorse  |
| pgvector     | SQL Baseline        |
| Redis Stack  | In-Memory Speedster |

Engines are pluggable via `engines/`. You can easily add more.

---

## Scenarios

All scenarios live in `scenarios/` and are orchestrated by `run_all.py`.

1. **Concurrency Burst** – `concurrency_burst.py`
   - Ramp from **1 → 5 → 10** (quick) or **1 → 5 → 10 → 20 → 50** (full) concurrent clients
   - Measure per engine:
     - `avg_ms`
     - **`p95_ms` / `p99_ms`**
     - `throughput_qps`
   - Output: `results/concurrency_burst.json`

2. **Metadata Filter Penalty** – `metadata_filter_penalty.py`
   - Compare:
     - Plain vector search
     - Vector search + metadata filter (e.g. `tag = 'tag-0'`)
   - Shows how much **filters** actually cost at P95/P99.
   - Output: `results/metadata_filter_penalty.json`

3. **Index Fragmentation** – `index_fragmentation.py`
   - Measure latency on a clean index
   - Then apply a wave of inserts + deletes
   - Measure latency again
   - Captures the **"stale / fragmented index" tax** most benchmarks ignore.
   - Output: `results/index_fragmentation.json`

4. **Recall–Latency Frontier** – `recall_latency_frontier.py`
   - Sweep different `k` values
   - Computes **exact brute-force nearest neighbours** (cosine similarity via NumPy) as ground truth, then measures each engine's true ANN recall against it
   - Track:
     - Average latency
     - True recall@k vs brute-force ground truth
   - Produces a **recall–latency curve** per engine.
   - Output: `results/recall_latency_frontier.json`

5. **Reranker Cost** – `reranker_cost.py`
   - Simulate a reranker on top of retrieval
   - Break down:
     - Retrieval latency
     - Reranker latency
     - End‑to‑end latency
   - Output: `results/reranker_cost.json`

---

## Quickstart

### 0. Clone & enter

```bash
git clone <this-repo-url>
cd rag-stress-test-arena
```

### 1. Start engines

```bash
docker-compose up -d
# starts:
# - qdrant
# - elasticsearch
# - pgvector
# - redis-stack
```

### 2. Install deps

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run all scenarios

```bash
# Quick mode (small dataset, fewer queries — good for a first run)
python run_all.py --mode quick

# Full mode (production-scale dataset)
python run_all.py --mode full
```

Mode defaults to `quick` if `--mode` is omitted.

This will:

- Index synthetic vectors/metadata
- Run all 5 scenarios against all registered engines
- Write JSON results into `results/`:

  - `concurrency_burst.json`
  - `metadata_filter_penalty.json`
  - `index_fragmentation.json`
  - `recall_latency_frontier.json`
  - `reranker_cost.json`

### 4. Run specific engines / scenarios

```bash
python run_all.py --engines qdrant elasticsearch --scenarios concurrency metadata
```

- `--engines`: subset of `qdrant`, `elasticsearch`, `pgvector`, `redis`
- `--scenarios`: subset of `concurrency`, `metadata`, `fragmentation`, `frontier`, `reranker`

### 5. Generate plots

```bash
python scripts/plot_results.py
```

This reads JSON from `results/` and writes PNGs into `plots/`, e.g.:

- `plots/concurrency_avg_ms.png`
- `plots/concurrency_p95_ms.png`
- `plots/metadata_filter_penalty_p95.png`
- `plots/index_fragmentation_p95.png`
- `plots/recall_latency_frontier_<engine>.png`
- `plots/reranker_cost.png`

---

## Quick vs Full mode

Scenarios use `scenarios/config.py` and `ScenarioConfig.pick(full, quick)` to decide dataset sizes and parameters. The mode is set by the `--mode` CLI flag (which internally sets the `RAG_STRESS_TEST_MODE` env var).

| Parameter         | quick  | full   |
|-------------------|--------|--------|
| Concurrency levels| 1,5,10 | 1,5,10,20,50 |
| Index size (burst)| 500    | 5 000  |
| Index size (recall)| 3 000 | 30 000 |
| Queries per scenario| 40   | 200–300 |

You can also set the env var directly if you prefer:

```bash
export RAG_STRESS_TEST_MODE=quick   # or full
python run_all.py
```

---

## Latest Benchmark Results

> Quick mode — 4 engines, Docker on localhost, 768-dim random vectors, clean index per run.

### Concurrency Burst (500 vectors · 40 queries/worker)

| Engine | Clients | avg ms | p95 ms | p99 ms | QPS |
|---|---|---:|---:|---:|---:|
| Redis | 1 | 2.4 | 3.8 | 6.7 | 407 |
| Redis | 5 | 4.1 | 7.8 | 9.9 | 1 115 |
| Redis | 10 | 7.1 | 14.4 | **23.9** | 1 280 |
| Qdrant | 1 | 35.4 | 31.2 | 540 | 28 |
| Qdrant | 5 | 69.0 | 33.9 | 1 963 | 70 |
| Qdrant | 10 | 122.5 | 59.5 | **3 486** | 77 |
| Elasticsearch | 1 | 22.7 | 26.2 | 29.9 | 44 |
| Elasticsearch | 5 | 54.5 | 73.1 | 83.9 | 91 |
| Elasticsearch | 10 | 104.5 | 136.0 | **144.8** | 95 |
| pgvector | 1 | 35.6 | 53.9 | 55.3 | 28 |
| pgvector | 5 | 62.3 | 90.3 | 119.7 | 79 |
| pgvector | 10 | 95.4 | 137.0 | **151.8** | 102 |

> Key insight: Qdrant p99 explodes 6× from 540 ms (1 client) to 3 486 ms (10 clients) even though avg only 3×. Averages hide tail pain.

### Metadata Filter Penalty (2 000 vectors · 60 queries · 20 tags)

| Engine | no-filter avg | no-filter p95 | +filter avg | +filter p95 | +filter p99 |
|---|---:|---:|---:|---:|---:|
| Redis | 1.7 ms | 2.5 ms | 1.8 ms | 3.4 ms | 3.4 ms |
| Qdrant | 25.3 ms | 32.5 ms | 13.7 ms | 32.0 ms | 33.6 ms |
| Elasticsearch | 18.9 ms | 23.9 ms | 18.4 ms | 23.8 ms | 28.3 ms |
| pgvector | 37.4 ms | 54.5 ms | 36.6 ms | 54.2 ms | 58.0 ms |

### Index Fragmentation (2 000 base · +1 000 inserts · −500 deletes · 60 queries)

| Engine | baseline avg | baseline p95 | after-frag avg | after-frag p95 | after-frag p99 |
|---|---:|---:|---:|---:|---:|
| Redis | 2.1 ms | 3.2 ms | 1.7 ms | 2.3 ms | 2.5 ms |
| Qdrant | 15.0 ms | 33.4 ms | 14.7 ms | 31.6 ms | 34.5 ms |
| Elasticsearch | 18.5 ms | 22.4 ms | 18.8 ms | 22.6 ms | 23.7 ms |
| pgvector | 35.8 ms | 51.3 ms | 36.4 ms (+1.6%) | 55.9 ms | 66.0 ms |

### Recall–Latency Frontier (3 000 vectors · 40 queries · brute-force ground truth)

| Engine | k=5 lat | k=5 recall | k=10 lat | k=10 recall | k=20 lat | k=20 recall |
|---|---:|---:|---:|---:|---:|---:|
| Redis | 2.4 ms | 18.0% | 2.0 ms | 17.8% | 1.9 ms | 30.3% |
| Qdrant | 19.6 ms | **100%** | 16.1 ms | **100%** | 18.8 ms | **100%** |
| Elasticsearch | 11.6 ms | 80.5% | 18.2 ms | 82.5% | 27.9 ms | 83.0% |
| pgvector | 39.6 ms | **100%** | 42.5 ms | **100%** | 42.9 ms | **100%** |

> Redis is fastest but only ~18–30% recall. Qdrant and pgvector are exact at this dataset size. Elasticsearch KNN lands at ~80–83% recall.

### Reranker Cost (2 000 vectors · 40 queries · k=20 · simulated 50 ms reranker)

| Engine | retrieval avg | reranker avg | end-to-end avg |
|---|---:|---:|---:|
| Redis | 2.9 ms | 50.7 ms | 53.6 ms |
| Qdrant | 20.7 ms | 50.6 ms | 71.3 ms |
| Elasticsearch | 27.6 ms | 50.6 ms | 78.2 ms |
| pgvector | 43.0 ms | 50.7 ms | 93.7 ms |

> When a reranker is in the pipeline, retrieval differences shrink relative to total cost. Redis's 2.9 ms vs pgvector's 43 ms gap becomes 53.6 ms vs 93.7 ms e2e — still meaningful, but the reranker dominates.

---

## How This Supports "Most RAG Benchmarks Are Lying to You"

This suite is designed to show specific failure modes that typical "10M vectors on my laptop" benchmarks hide:

- **Tail latency vs averages**
  - Concurrency Burst exposes how **P95/P99 explode** even when averages look fine.

- **Filters vs naked ANN**
  - Metadata Filter Penalty quantifies the real cost of `WHERE tenant_id = ...` style filters.

- **Day‑30 index, not Day‑1**
  - Index Fragmentation shows how latency degrades after many upserts/deletes.

- **True recall, not self-reported**
  - Recall–Latency Frontier computes exact nearest neighbours via brute force, so recall numbers are real — not the engine grading its own homework.

- **End‑to‑end, not just ANN**
  - Reranker Cost adds a simple reranking step on top of retrieval to approximate realistic pipelines.

Use the JSON + plots directly in your own perf docs, blog posts, or internal reviews.

---

## Adding Your Own Code

### New engine

1. Create `engines/my_engine.py` extending `BaseEngine`:

   - `index(vectors, metadata)`
   - `search(query, k)`
   - `search_with_filter(query, k, filter_query)`
   - `insert(vectors, metadata)`
   - `delete(ids)`
   - `flush()`

2. Register it in `engine_loader.py` by adding an entry to `_ENGINE_FACTORIES`:

   ```python
   _ENGINE_FACTORIES: Dict[str, Callable[[], BaseEngine]] = {
       "qdrant": lambda: _load_class("engines.qdrant_engine", "QdrantEngine")(),
       "elasticsearch": lambda: _load_class("engines.elasticsearch_engine", "ElasticsearchEngine")(),
       "pgvector": lambda: _load_class("engines.pgvector_engine", "PgVectorEngine")(),
       "redis": lambda: _load_class("engines.redis_engine", "RedisEngine")(),
       "my_engine": lambda: _load_class("engines.my_engine", "MyEngine")(),  # add this
   }
   ```

3. Run:

   ```bash
   python run_all.py --engines my_engine
   ```

### New scenario

1. Add `scenarios/my_scenario.py` with a `run_scenario(engines, out_path=...)` function.
2. Import and wire it in `run_all.py`.
3. Add it to the `--scenarios` dispatch.

---

## Notes & Caveats

- Data is synthetic (random vectors + synthetic metadata).
  Use it to study **behavior and curves**, not absolute numbers.
- For serious evaluation, plug in:
  - Your embeddings
  - Your document sizes and distributions
  - Your real filter patterns and access patterns

PRs, issues, and benchmark results from other stacks are very welcome.
