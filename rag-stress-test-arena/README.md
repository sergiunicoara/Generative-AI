# rag-stress-test-arena

Stress-test suite proving **why most RAG benchmarks are lying to you**.

This repo focuses on **realistic RAG workloads**, not lab demos. It measures how vector engines behave under:

- Concurrency (1 → 50 clients)
- Metadata filters
- Index mutations (upserts/deletes)
- Reranking overhead

and exposes where “average latency” and toy benchmarks are misleading.

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
   - Ramp from **1 → 5 → 10 → 20 → 50** concurrent clients  
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
   - Captures the **“stale / fragmented index” tax** most benchmarks ignore.  
   - Output: `results/index_fragmentation.json`

4. **Recall–Latency Frontier** – `recall_latency_frontier.py`  
   - Sweep different `k` values  
   - Track:
     - Average latency
     - Approximate recall proxy  
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

### 3. Choose mode: quick vs full

Scenarios use `scenarios/config.py` and `ScenarioConfig.pick(full, quick)` to decide sizes:

- **quick**: small dataset, fewer queries, levels `[1, 5, 10]`
- **full**: larger dataset, more queries, levels `[1, 5, 10, 20, 50]`

Control via env var:

```bash
# Full, production-like stress
export RAG_STRESS_TEST_MODE=full      # Windows cmd: set RAG_STRESS_TEST_MODE=full
# or quick/dev mode
export RAG_STRESS_TEST_MODE=quick
```

If you don’t set it, it defaults to `quick`.

### 4. Run all scenarios

```bash
python run_all.py
```

This will:

- Index synthetic vectors/metadata
- Run all 5 scenarios against all registered engines
- Write JSON results into `results/`:

  - `concurrency_burst.json`
  - `metadata_filter_penalty.json`
  - `index_fragmentation.json`
  - `recall_latency_frontier.json`
  - `reranker_cost.json`

### 5. Run specific engines / scenarios

```bash
python run_all.py --engines qdrant elasticsearch --scenarios concurrency metadata
```

- `--engines`: subset of `qdrant`, `elasticsearch`, `pgvector`, `redis`
- `--scenarios`: subset of `concurrency`, `metadata`, `fragmentation`, `frontier`, `reranker`

### 6. Generate plots

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

## How This Supports “Most RAG Benchmarks Are Lying to You”

This suite is designed to show specific failure modes that typical “10M vectors on my laptop” benchmarks hide:

- **Tail latency vs averages**  
  - Concurrency Burst exposes how **P95/P99 explode** even when averages look fine.

- **Filters vs naked ANN**  
  - Metadata Filter Penalty quantifies the real cost of `WHERE tenant_id = ...` style filters.

- **Day‑30 index, not Day‑1**  
  - Index Fragmentation shows how latency degrades after many upserts/deletes.

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

2. Register it in `engine_loader.py`:

   ```python
   from engines.my_engine import MyEngine

   def get_all_engines():
       return {
           "qdrant": QdrantEngine(),
           "elasticsearch": ElasticsearchEngine(),
           "pgvector": PgVectorEngine(),
           "redis": RedisEngine(),
           "my_engine": MyEngine(),
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
