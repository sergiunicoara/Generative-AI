# Performance Metrics Inventory

This document catalogs every metric the platform measures, where it's recorded, how to access it, and what it means.

---

## Overview

| Layer | Storage | Granularity | API | Status |
|---|---|---|---|---|
| Query-level KPIs | SQLite by default; optional TimescaleDB via `KPI_BACKEND=timescale` | Per query | `GET /kpis/summary`, `/kpis/timeseries` | **Active** |
| Graph health | Neo4j (`GraphHealthSnapshot` nodes) | Per snapshot (24h default) | `GET /kg/snapshots` | Implemented, needs Neo4j running |
| Confidence calibration | Neo4j (`CalibrationSample` nodes) | Per model version | Internal (no public API) | Implemented, needs Neo4j running |
| GNN scoring | In-flight (retrieval results) | Per chunk | Returned in the `POST /query` response | Active |
| Context composition | Prometheus histograms | Per assembled context | `graphrag_context_section_tokens{section}`, `graphrag_context_total_tokens` | **Active** |

---

## Query-Level Metrics (KPI Events)

### What they measure
Every user query fires a `KPIEvent` that captures the full lifecycle: retrieval latency, RAGAS evaluation, cost, and operational metadata.

### Storage
**Database:** SQLite at `results/kpi_snapshots/kpis.db` by default. For durable
time-series workloads, set `KPI_BACKEND=timescale` and
`TIMESCALE_DB_URL=postgresql+asyncpg://...`; the same KPI schema is initialized
as a TimescaleDB hypertable.

**Schema:**
```
CREATE TABLE kpi_events (
    event_id TEXT PRIMARY KEY,
    query_id TEXT,
    recorded_at DATETIME,
    latency_ms FLOAT,
    faithfulness FLOAT,
    answer_relevancy FLOAT,
    context_precision FLOAT,
    context_recall FLOAT,
    cost_usd FLOAT,
    retrieval_mode TEXT,
    model_version TEXT
)
```

### Metrics

| Metric | Type | Range | Meaning | Default if N/A |
|---|---|---|---|---|
| `latency_ms` | Float | 0–∞ | Total query duration (retrieval + LLM) in milliseconds | N/A |
| `faithfulness` | Float | 0.0–1.0 | RAGAS: was the answer grounded in the retrieved context? Higher = better | 0.0 |
| `answer_relevancy` | Float | 0.0–1.0 | RAGAS: does the answer address the query? Higher = better | 0.0 |
| `context_precision` | Float | 0.0–1.0 | RAGAS: is the retrieved context helpful? Higher = better | 0.0 |
| `context_recall` | Float | 0.0–1.0 | RAGAS: did we retrieve all relevant context? Higher = better | 0.0 |
| `cost_usd` | Float | 0.0–∞ | Total cost of the query (LLM + embedding API calls) | 0.0 |
| `retrieval_mode` | Text | "hybrid", "local", "global", "agentic" | Which retrieval path was taken | "hybrid" |
| `model_version` | Text | e.g., "groq:openai/gpt-oss-120b" | Which synthesis-tier LLM generated the answer | "" |

### Sampling strategy

- **Latency**: 100% — every query is timed
- **RAGAS scores**: ~20% sample — evaluating every query is expensive; the current judge order is DeepSeek, Groq fallback, then Gemini compatibility fallback

### Access patterns

**Via API (requires auth):**
```bash
GET /kpis/summary?window_days=7
# Returns: {
#   "avg_latency_ms": 1234.5,
#   "p95_latency_ms": 2100.0,
#   "avg_faithfulness": 0.78,
#   "avg_answer_relevancy": 0.82,
#   "total_queries": 156,
#   "total_cost_usd": 2.34
# }

GET /kpis/timeseries?metric=latency_ms&window_days=7
# Returns: [
#   { "bucket": "2026-03-20T19:00:00Z", "mean": 1200, "p95": 2000, "count": 45 },
#   ...
# ]
```

**Direct SQLite query:**
```python
import sqlite3
db = sqlite3.connect('results/kpi_snapshots/kpis.db')
cursor = db.cursor()

# P95 latency in the last 7 days
cursor.execute('''
    SELECT 
      AVG(latency_ms) as mean_latency,
      COUNT(*) as query_count,
      MAX(latency_ms) as max_latency
    FROM kpi_events
    WHERE recorded_at > datetime('now', '-7 days')
''')
result = cursor.fetchone()
```

### Sample data

From `results/kpi_snapshots/kpis.db` (104 queries measured 2026-06-03) —
**superseded by the live re-measurement below; kept here for historical
record, not as a current claim:**

```
Aggregate across 104 queries:
  Avg hybrid latency:  ~2.2s p95 (94 hybrid queries)
  Avg agentic latency: ~3.4s p95 (10 agentic/IRCoT queries)
  Faithfulness:        0.937 on answerable / 0.842 overall  (correct refusals excluded from answerable)
  Context Precision:   0.907
  Context Recall:      0.867
  Retrieval Mode:      hybrid (90%) / agentic (10%)
  Generation model:    Groq configured large model (get_llm() default) / DeepSeek fallback / Cerebras opt-in
  Embedding model:     text-embedding-3-large (OpenAI, 3072d)
```

**Corrected, 2026-07-29** — the 2.2s figure above no longer reflects
reality and should not be cited. Live re-measurement: 10 real questions
(`data/eval_golden/queries_automotive.json`, the automotive tenant — 3,013
entities, the largest live tenant) run sequentially through
`HybridRetriever.retrieve_and_answer` in one warm process, reranker
pre-warmed:

```
n=10  min=6.8s  p50=22.4s  p95=45.9s  max=45.9s  mean=22.0s
mode: 100% hybrid (agentic fallback never triggered in this sample)
```

**Caveats, stated plainly:**
- n=10 is not large enough for a statistically meaningful p95 — with 10
  samples, "p95" and "max" are the same point. Treat this as a rough
  order-of-magnitude correction, not a production SLA number. A real p95
  needs 30+ samples at minimum.
- This is a **cold-ish run**: 10 different questions touching different
  entity subsets of a 3,013-entity tenant, so the entity embedding cache
  (`graphrag/graph/embedding_cache.py`, `tasks/lessons.md` A143) only
  partially warms across the run — `chunk_entities_edges` ranged 657ms to
  6,203ms question-to-question, not the ~26x-faster warm-cache number
  measured when the *same* question repeats. A long-running server
  (`mcp_server/`, `workers/query_worker.py`) will trend faster than this
  sample as its cache saturates a tenant's entities.
- `global_search.reduce` — the single largest per-call cost found in
  A143 (13.9-17.6s) — only fired on 2 of 10 questions here; the other 8
  took the cheaper early-return path (`reason=all_map_results_not_relevant`).
  So the map-only queries in this sample understate how bad the tail gets
  when reduce *does* fire, and the ones that hit it (45.9s, 27.1s) show it.
- One question (`NEG-02`) hit `global_search.no_communities` — a real,
  separate gap (automotive appears to be missing Community nodes for at
  least part of its graph; the code's own hint suggests
  `scripts/community_rebuild.py`), not investigated further here.

**Corrected again, 2026-07-29 (same day, post-fix)** — `global_search.reduce`
(flagged above as "the next latency investigation") is now fixed:
root-caused as unbounded LLM generation for a call whose output is never
user-facing, plus usually nothing to synthesize (3 of 4 live occurrences
had only one partial answer). Fixed with a short-circuit for the
single-partial case and a `max_tokens` cap for the genuine multi-partial
case — see `tasks/lessons.md` A144. Re-running the **same 10-question
sample**:

```
n=10  min=5.1s  p50=20.7s  p95=33.9s  max=33.9s  mean=18.3s
```

p95 45.9s → 33.9s (-26%), mean 22.0s → 18.3s (-17%). The same n=10 caveat
applies — this is still not a statistically meaningful p95. The single
worst case from the whole investigation (an aerospace question that
previously triggered a 13.9-17.6s reduce call) dropped from 30-43s total
to **9.1s** on a warm cache.

**Corrected again, 2026-07-30 — parallelized local_search + global_search
(A145)**: the two had no data dependency but ran back-to-back; switched to
`asyncio.TaskGroup` so global search's latency hides behind local search's
instead of adding to it — see `tasks/lessons.md` A145. Also, this round
finally replaces the n=10 sample: combined the 10-question automotive set
with the 34-question aerospace golden set (`evals/golden_set.json`) for
**n=44** across two tenants — the first sample in this whole investigation
large enough to treat as more than a rough order-of-magnitude check:

```
n=44  min=6.3s  p50=13.2s  p95=26.4s  max=52.8s  mean=15.2s
tenant breakdown: aerospace=34, automotive=10
```

p50 20.7s → 13.2s, p95 33.9s → 26.4s — consistent with the earlier n=10
automotive-only re-run of this same fix (p50 14.75s, p95 27.1s), which is
itself reassuring: the two samples roughly agree despite different tenants
and a 4x larger n. The max (52.8s, one aerospace multi-partial-reduce case)
is a reminder the tail is still real, not eliminated.

**Honest summary for a pitch**: don't quote "2.2s p95" — it's stale, even
after three rounds of fixes this session. Current live behavior (n=44,
two tenants) is p50 ~13s, p95 ~26s — down from p95 ~46s before this
session's work, but still roughly an order of magnitude over the old
documented claim. The remaining cost is round-trip *count* (query rewrite,
embed, map, occasionally reduce, final synthesis) — most of that chain is
still fully sequential; parallelizing local+global search removed only one
join point. Further round-trip reduction is not yet scoped.

---

## Graph Health Metrics (GraphHealthSnapshot)

### What they measure
The state of the knowledge graph — is it growing, staying clean, avoiding orphaned entities and contradictions?

### Storage
**Database:** Neo4j as `(:GraphHealthSnapshot)` nodes

**Cypher to query:**
```cypher
MATCH (h:GraphHealthSnapshot) 
RETURN h.recorded_at, h.entity_resolution_quality, h.contradiction_rate
ORDER BY h.recorded_at DESC 
LIMIT 10
```

### Metrics

| Metric | Type | Range | Meaning | Healthy threshold |
|---|---|---|---|---|
| `entity_resolution_quality` | Float | 0.0–1.0 | What fraction of extracted entities were successfully merged into canonical forms? | > 0.85 |
| `relation_precision` | Float | 0.0–1.0 | What fraction of extracted relations are semantically valid? (inverse of false positives) | > 0.80 |
| `contradiction_rate` | Float | 0.0–∞ | **Conflicts per 1,000 edges** (not a fraction). Lower is better. A rate of 1.0 means one contradiction per thousand edges. | < 2.0 /1k |
| `orphan_growth_rate` | Float | 0.0–∞ | What fraction of new entities have zero incoming or outgoing edges? | < 0.20 |
| `merge_split_error_proxy` | Float | 0.0–1.0 | False positive + false negative rate of entity resolution (estimated) | < 0.15 |
| `community_coherence` | Float | 0.0–1.0 | Do communities detected by Leiden algorithm capture meaningful clusters? (internal edge density / external edge density) | > 0.50 |

### Sampling strategy

Snapshots are persisted on a maintenance schedule (default: every 24 hours). Each snapshot captures the full graph state at that moment.

### How they're computed

**Entity Resolution Quality:**
```python
# Fraction of entities with aliases successfully merged
canonical_entities = count of unique (name, type, tenant) tuples
extracted_mentions = count of original entity extractions
quality = (extracted_mentions - unmerged_aliases) / extracted_mentions
```

**Relation Precision:**
```python
# Estimated via cross-encoder confidence threshold
relations_above_threshold = count of relations where confidence >= 0.75
all_relations = total relation count
precision ≈ relations_above_threshold / all_relations
```

**Contradiction Rate (conflicts per 1,000 edges):**
```python
# Conflicts per 1,000 edges — a density measure, not a fraction
conflict_count = count of (:Conflict) nodes in the tenant
total_edges = count of (:Entity)-[:RELATES_TO]->(:Entity) in the tenant
rate_per_1k = (conflict_count / total_edges) * 1000
# Healthy: < 2.0 /1k  |  Warning: > 3.0 /1k  |  Critical: > 5.0 /1k
```

**Orphan Growth Rate:**
```python
# Fraction of new entities with degree = 0
new_entities_this_window = entities extracted in last interval
orphans = count of new entities with no RELATES_TO edges
rate = orphans / new_entities_this_window
```

**Community Coherence:**
```python
# Modularity of Leiden communities
Q = (internal_edges / total_edges) - (expected_by_chance)
coherence = 0.5 + 0.5 * Q  # normalized to [0, 1]
```

### Access patterns

**Via API (requires auth):**
```bash
GET /kg/snapshots
# Returns the latest snapshot with all metrics
```

**Direct Neo4j query:**
```cypher
MATCH (h:GraphHealthSnapshot)
WITH h ORDER BY h.recorded_at DESC LIMIT 1
RETURN {
  timestamp: h.recorded_at,
  entity_quality: h.entity_resolution_quality,
  contradiction_rate: h.contradiction_rate,
  orphan_rate: h.orphan_growth_rate,
  community_coherence: h.community_coherence
}
```

### Interpretation example — real aerospace corpus (2026-06-04)

```
Real data from Neo4j (12-doc aerospace regulatory corpus):
  Entities:             362  (after alias dedup from 684 extracted — 47% reduction)
  Relations:            382  (420 pipeline extracted → 371 asserted + 11 inferred = 382 in graph)
  Inferred edges:       11   (forward-chaining: 9 supersedes_transitivity + 1 certifies_inverse + 1 related_to)
  Contradiction Rate:   48.26 /1k edges
  Orphan Rate:          17%  (61 isolated entities — no relations resolved)
  Community Coherence:  90% (39 Leiden communities, real corpus) ✓
  Open Conflicts:       18 (detected by contradiction detector, live in Neo4j)
```

⚠ **This is a dated illustrative example (2026-06-04), not a current baseline —
do not quote these numbers as "the real corpus figures" today.** LLM extraction
is non-deterministic at temperature=0; every fresh `--wipe --commit` of the same
12-doc corpus reshapes the graph (entity count alone has read 362, 364, and 368
on three different runs, two of them on the *same day*, 2026-06-07 — see
`tasks/lessons.md` A96/A98). Use this block to learn **how to read** a graph-health
report — contradiction rate, orphan rate, inferred-edge breakdown — then re-run
the live queries below for
whatever the actual current numbers are before presenting.

The contradiction rate (48.26 /1k, *as measured on 2026-06-04 — will differ today*)
reflects genuine document disagreements in the aerospace regulatory corpus. The
orphan rate (17% on that date) reflects entities mentioned in
passing in documents without enough context to extract relations — expected in
technical manuals where tools and part numbers appear without explicit relationships.
A production run with richer documents would reduce orphan rate through second-pass
relation extraction.

---

## Confidence Calibration Metrics

### What they measure
How well are the confidence scores the system assigns actually predictive of truth?

### Storage
**Database:** Neo4j as `(:CalibrationSample)` nodes

**Schema:**
```
(:CalibrationSample) {
  model_version: "llama-3.3-70b",
  bin: 0.1,                    # Confidence bucket: [0.0-0.1), [0.1-0.2), etc.
  predicted_confidence: 0.05,  # Mean predicted confidence in this bin
  actual_accuracy: 0.08,       # Fraction of relations in this bin that were correct
  sample_count: 145            # Number of relations in this bin
}
```

### Metrics

| Metric | Type | Meaning |
|---|---|---|
| `brier_score` | Float | Mean squared error between predicted confidence and actual correctness. 0 = perfect calibration, 1 = worst. |
| `calibration_curve` | Array of (pred, actual) tuples | Calibration curve at 10% bins. Perfect calibration is a 45° line. |
| `isotonic_offset` | Float | Bias correction learned from historical samples. Added to all future predictions. |

### Brier score formula

```
brier_score = mean((predicted_confidence - actual_correctness)^2)
```

**Example:**
- Relation extracted with confidence 0.8, turned out wrong (actual = 0) → contribution = (0.8 - 0)² = 0.64
- Relation extracted with confidence 0.95, turned out right (actual = 1) → contribution = (0.95 - 1)² = 0.0025
- Average over all relations = Brier score

**Interpretation:**
- **0.0–0.15:** Excellent calibration
- **0.15–0.25:** Good calibration
- **0.25–0.40:** Acceptable but degraded
- **> 0.40:** Poor; model's confidence is unreliable

### Calibration curve

The curve answers: "When the model says confidence=0.5, is it right 50% of the time?"

**Perfect calibration:** Predicted confidence = actual accuracy at every bin
**Overconfident:** Predicted > actual (model is too sure)
**Underconfident:** Predicted < actual (model is too timid)

### Access patterns

**Via Neo4j:**
```cypher
MATCH (s:CalibrationSample {model_version: "llama-3.3-70b"})
RETURN s.bin, s.predicted_confidence, s.actual_accuracy, s.sample_count
ORDER BY s.bin
```

**Python API:**
```python
from graphrag.graph.confidence_calibration import CalibrationService
svc = CalibrationService()
curve = svc.get_calibration_curve("llama-3.3-70b")
brier = svc.get_brier_score("llama-3.3-70b")
```

---

## Retrieval Pipeline Metrics

### What they measure
Performance of each retrieval stage in the five-stage retrieval pipeline; LLM synthesis is measured separately.

### Metrics returned in the `POST /query` response

```json
{
  "answer": "...",
  "citations": [...],
  "retrieval_breakdown": {
    "stage_1_vector_ann_ms": 45,
    "stage_2_bm25_ms": 12,
    "stage_3_cross_encoder_ms": 210,
    "stage_4_multihop_ms": 88,
    "stage_5_gnn_scoring_ms": 156,
    "stage_6_llm_synthesis_ms": 1348,
    "total_ms": 1859
  },
  "final_scores": [
    {
      "chunk_id": "doc-0-chunk-15",
      "text": "...",
      "cross_encoder_score": 0.92,
      "gnn_score": 0.67,
      "final_score": 0.82,
      "final_score_breakdown": {
        "alpha": 0.6,
        "beta": 0.4,
        "calculation": "0.6 * 0.92 + 0.4 * 0.67 = 0.82"
      }
    }
  ]
}
```

### Metric definitions

| Metric | Type | Meaning |
|---|---|---|
| `stage_N_*_ms` | Float | Latency of each retrieval stage | 
| `cross_encoder_score` | Float [0–1] | Reranker's confidence that this chunk answers the query |
| `gnn_score` | Float [0–1] | Graph proximity score: how close is this chunk's entity to the query entity? |
| `final_score` | Float [0–1] | Weighted blend: `α·cross_encoder + β·gnn_score` (default α=0.6, β=0.4) |

### Interpretation

**Fast vs slow queries:**
- Stage 1+2 (retrieval): < 100ms = fast
- Stage 3 (reranking): 100–300ms = normal (cold start slower)
- Stage 5 (GNN): > 200ms = large graph, consider caching
- Stage 6 (LLM): > 1000ms = provider latency/rate limiting or complex synthesis

**Score blend α/β:**
- `α=0.6, β=0.4` (default): Trust textual relevance over graph structure
- `α=0.4, β=0.6`: Trust graph proximity (for structured queries like "who leads this company?")
- Configurable via `settings.yml: retrieval.gnn_weight`

---

## Context Composition Metrics

### What they measure

`ContextBuilder.build()` assembles the prompt from up to seven section kinds
(`chunks`, `linked_chunks`, `document_links`, `entity_context`,
`graph_relationships`, `conflicts`, `community_knowledge`). Six are bounded
by a count (top_k, `[:5]`, `[:10]`), not by tokens — five long entity
descriptions cost more than ten short edges — and `community_knowledge` is
bounded by nothing at all. These metrics attribute "the context is too big"
to a specific section instead of leaving it a guess.

### Storage

Prometheus histograms, recorded per assembled context via
`graphrag.observability.context_size.record_context_composition()`. Never
raises — a metrics-backend problem must not fail a query.

### Metrics

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `graphrag_context_section_tokens` | Histogram | `section` (one of the seven kinds above) | Estimated tokens contributed by that section |
| `graphrag_context_total_tokens` | Histogram | — | Estimated total tokens in the assembled context |

Buckets: `16, 64, 256, 1024, 4096, 16384, 65536` — spans a single short edge
line up to a context far larger than any current model window.

### Token estimation

`graphrag/core/tokens.py: estimate_tokens(text)` — `tiktoken` with the
`cl100k_base` encoding (cached), falling back to a character-count heuristic
if `tiktoken` is unavailable. This is an estimate for context-composition
visibility, not a hard provider token-limit check.

### Token budget

`ContextBuilder.build(..., token_budget: int | None = None)` caps tokens
spent on the primary top-k chunk admission loop only. It does **not** bound
the two additive slots (reserved multi-hop slots, the document-link section)
— those can still push the assembled context above `token_budget`. The first
admitted chunk always counts toward the budget even though it is admitted
unconditionally, so an oversized leading chunk still spends budget rather
than being exempted.

### Access patterns

**Via Prometheus:**
```promql
histogram_quantile(0.95, sum(rate(graphrag_context_section_tokens_bucket[5m])) by (le, section))
histogram_quantile(0.95, sum(rate(graphrag_context_total_tokens_bucket[5m])) by (le))
```

---

## Rate Limiting & Quota Metrics

### What they measure

Per-client and per-conversation request throughput, and per-tenant
consumption ceilings.

### Storage

`AsyncRateLimiter` (`api/limiter.py`) — a moving-window limiter backed by
Redis when `REDIS_URL` is set (shared across replicas), with an in-process
fallback otherwise. State lives in the limiter's own store, not a
general-purpose metrics backend; no dedicated Prometheus counter for 429
events exists yet.

### Limits

| Limit | Env var | Default | Bucket key |
|---|---|---|---|
| Per-client (IP/subject) | `GRAPHRAG_RATE_LIMIT_*` (per route) | route-specific | `client_key` — subject-or-IP |
| Per-conversation | `GRAPHRAG_RATE_LIMIT_CONVERSATION` | `20/minute` | `conversation_key` — caller **and** session, deliberately never session-alone, to prevent one caller draining another's bucket by session ID |
| Search | `GRAPHRAG_RATE_LIMIT_SEARCH` | `60/minute` | `client_key` |
| Tenant quota | `graphrag/core/tenant_quota.py` | per-tenant configured ceiling | tenant |

### Access patterns

A 429 response body states which ceiling was hit and when it resets
(`api/quota.py`); this is currently visible per-request, not yet aggregated
as a metric. Adding a `graphrag_rate_limit_rejections_total{route,reason}`
counter is a plausible next step (see "Next steps to instrument further"
below) but does not exist today — do not cite one.

---

## Derived Metrics (Dashboard)

Computed from raw metrics for visibility:

| Metric | Formula | Frequency |
|---|---|---|
| P95 latency | 95th percentile of all latencies in window | On-demand (per dashboard refresh) |
| Answer quality score | (faithfulness + answer_relevancy) / 2 | Computed at summary time |
| Graph health score | (entity_resolution_quality + clamp(1 − contradiction_rate_per1k / 5, 0, 1)) / 2 | Per snapshot |

---

## Alerting Thresholds

The system emits alerts when metrics fall outside healthy ranges:

| Metric | Alert threshold | Severity | Action |
|---|---|---|---|
| `p95_latency_ms` (hybrid) | > 3000 | ⚠️ Warning | **Stale relative to measured behavior (2026-07-30, post-A145 fix, n=44 across two tenants): real p95 is ~26.4s, ~9x this threshold** — down from ~46x before this session's three latency fixes, still nowhere close. This is real active config (`graphrag/monitoring/alerts.py:21,53`), not just a doc claim — as configured today, any live deployment would be in constant alert. Not recalibrated here; whether to tighten (treat current latency as an active incident) or loosen (match reality, alert only on further regression) is a product decision, not made in this pass. |
| `p95_latency_ms` (agentic) | > 10000 | ⚠️ Warning | Agentic/IRCoT runs 3–4 LLM rounds by design; 4–8s is expected and correct. Alert only on outliers. |
| `agentic_rate` | > 20% | ⚠️ Warning | If >20% of queries trigger agentic fallback, the hybrid confidence threshold is too loose — tighten `_is_low_confidence` trigger. |
| `faithfulness` | < 0.80 (3-sample window) | ⚠️ Warning | Check recent document ingestions; may have extraction errors. Target is **≥ 0.85**; 0.80 is the alert floor, not the goal. Measured: 0.937 on answerable questions; 0.842 overall (correct refusals score 0 and are excluded from answerable). |
| `contradiction_rate` | > 3.0 /1k edges | ⚠️ Warning | Moderate contradiction density — review recent ingestion batch |
| `contradiction_rate` | > 5.0 /1k edges | 🔴 Critical | High contradiction density — indicates schema drift or malformed source docs |
| `orphan_growth_rate` | > 0.30 | ⚠️ Warning | Entities not connecting to rest of graph; review extraction |
| `brier_score` | > 0.35 | ⚠️ Warning | Confidence calibration degraded; retrain or recalibrate. Note: 0.18 is achievable **after isotonic regression correction**; raw LLM confidence before correction typically scores 0.20–0.35. |

---

## How to use this in a pitch

**For a CTO evaluating the platform:**

1. **Start with KPI data**: "Live-measured across two tenants (automotive +
   aerospace), 44 real questions, post-fix: p50 13.2s, p95 26.4s, mean 15.2s
   (n=44 — the first sample in this investigation large enough to be more
   than an order-of-magnitude check; see the 'Corrected again, 2026-07-30'
   note above for the full breakdown — three root-caused fixes this session
   cut p95 from 45.9s to 26.4s). The remaining cost is round-trip count
   (query rewrite, embed, map, occasionally reduce, final synthesis), not a
   single bottleneck — most of that chain is still fully sequential.
   Faithfulness is 0.937 on answerable questions (0.842 overall including
   correct refusals). Context precision is 0.907, meaning almost everything
   we retrieve is relevant." **Do not cite "2.2s p95" — that number is
   stale even after three rounds of live fixes.**

2. **Show graph health**: "The knowledge graph is built from 12 aerospace regulatory
   documents — FAA/EASA airworthiness directives, manufacturer records, fleet data —
   run through the full real LLM extraction pipeline (368 entities, 422 edges, 7 open
   conflicts, 90% community coherence — verified live, 2026-06-07). Raw extraction counts
   resolve down to that through 4-stage alias deduplication; the exact raw-vs-final ratio
   varies run to run because LLM extraction is non-deterministic at temperature=0
   (see `tasks/lessons.md` A96/A98) — don't quote a specific dedup percentage from
   memory, re-run the live count first. On a production-scale corpus these health
   metrics scale automatically."

3. **Demonstrate calibration**: "The calibration pipeline uses isotonic regression to
   correct raw LLM confidence scores. Raw llama-3.3-70b confidence on technical corpora
   typically scores around Brier 0.31 — the isotonic correction targets Brier < 0.20.
   The pipeline is implemented and wired; the trajectory you see in the dashboard
   represents the expected correction curve."

4. **Explain the latency breakdown** — this entire section (the "~0.5s graph
   search, ~1.4s synthesis, ~2.4s avg" table that used to live here) has been
   **live-superseded and removed**. It was an illustrative estimate, never
   independently instrumented, and real per-stage timing (added this session
   — `local_search.py`, `global_search.py`) now contradicts it directly.
   The real breakdown, per-stage, live-measured:
   - Retrieval (BM25 + rerank + multihop + GNN): sub-second to a few seconds,
     genuinely fast — never the bottleneck.
   - `chunk_entities_edges` (Neo4j entity-embedding fetch): was the largest
     non-LLM cost (2.9-6.6s cold), fixed with an entity-keyed cache
     (`graphrag/graph/embedding_cache.py`, `tasks/lessons.md` A143) — 26x
     faster on a warm cache, live-verified.
   - `global_search.map` (concurrent per-community LLM calls): 2.6-7.4s.
   - `global_search.reduce` (single LLM call) — **was** 13.9-17.6s, the
     single largest cost anywhere in the pipeline. Root-caused: unbounded
     LLM generation for a call whose output is never user-facing (it's
     consumed only as another prompt's input), and usually nothing to
     synthesize (3 of 4 live occurrences had one partial answer, not
     several to merge). Fixed — single-partial case now short-circuits
     entirely (skips the LLM call), multi-partial case capped via
     `max_tokens` (`tasks/lessons.md` A144). Now 0s (skipped) or
     ~4.5s (genuine merge case), live-verified.
   - Final answer synthesis: single-digit seconds.

   **The honest one-line version**: retrieval is fast, the graph fetch is
   fixed, `global_search.reduce` is fixed — the pipeline's remaining cost is
   round-trip *count* (query rewrite, embed, map, occasionally reduce,
   final synthesis), not one dominant stage. Don't quote the old "synthesis
   is ~58% of a 2.4s average" framing; it undercounted by an order of
   magnitude and named the wrong stage as dominant, twice over now.

---

## Next steps to instrument further

If deployed to production, consider adding:

- **User satisfaction signals**: Explicit feedback ("was this answer helpful?") to compute real (vs. synthetic RAGAS) accuracy
- **Cohort analysis**: Segment metrics by document domain, query type, user segment
- **Trend detection**: Alert when key metrics drift significantly (e.g., entity_resolution_quality drops from 0.92 to 0.85)
- **Cost optimization**: Break down LLM costs by retrieval mode (hybrid vs. agentic) to identify expensive paths
- **Extraction error analysis**: Tag specific extraction failures in the conflict queue and track them by model version
- **Rate-limit rejection counter**: `graphrag_rate_limit_rejections_total{route,reason}` — 429s are currently visible per-request (response body) but not aggregated; a counter would let the dashboard show rejection rate by route without scraping logs
