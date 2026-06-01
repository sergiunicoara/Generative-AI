# Performance Metrics Inventory

This document catalogs every metric the platform measures, where it's recorded, how to access it, and what it means.

---

## Overview

| Layer | Storage | Granularity | API | Status |
|---|---|---|---|---|
| Query-level KPIs | SQLite (`results/kpi_snapshots/kpis.db`) | Per query | `GET /kpis/summary`, `/kpis/timeseries` | **Active** |
| Graph health | Neo4j (`GraphHealthSnapshot` nodes) | Per snapshot (24h default) | `GET /kg/health/snapshot` | Implemented, needs Neo4j running |
| Confidence calibration | Neo4j (`CalibrationSample` nodes) | Per model version | Internal (no public API) | Implemented, needs Neo4j running |
| GNN scoring | In-flight (retrieval results) | Per chunk | Returned in `/search` response | Active |

---

## Query-Level Metrics (KPI Events)

### What they measure
Every user query fires a `KPIEvent` that captures the full lifecycle: retrieval latency, RAGAS evaluation, cost, and operational metadata.

### Storage
**Database:** SQLite at `results/kpi_snapshots/kpis.db`

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
| `model_version` | Text | e.g., "gemini-2.5-flash" | Which LLM generated the answer | "" |

### Sampling strategy

- **Latency**: 100% — every query is timed
- **RAGAS scores**: ~20% sample — evaluating every query is expensive; running Groq judge on a random sample keeps costs down while maintaining statistical validity

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

From `results/kpi_snapshots/kpis.db` (as of 2026-03-20):

```
Query 5a420d85...
  Latency: 1859.0 ms
  Faithfulness: 0.667
  Context Precision: 1.000
  Context Recall: 1.000
  Retrieval Mode: hybrid
  Model: gemini-2.5-flash
  Cost: $0.000000
```

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
GET /kg/health/snapshot
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

### Interpretation example

```
Snapshot at 2026-03-20 22:00:00 UTC:
  Entity Resolution Quality: 0.92 ✓ (excellent — curated domain)
  Relation Precision: 0.81 ✓ (healthy)
  Contradiction Rate: 1.2 /1k edges ✓ (healthy — well below 2.0 /1k threshold)
  Orphan Growth Rate: 0.08 ✓ (acceptable)
  Community Coherence: 0.62 ✓ (good)
```

This indicates a clean, well-formed graph. A healthy corpus should have
entity_resolution_quality > 0.85 (on curated domain data; expect 0.70–0.85
on noisy enterprise data) and contradiction_rate < 2.0 /1k edges.

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
Performance of each stage in the 6-stage retrieval pipeline.

### Metrics returned in `/search` response

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
- Stage 6 (LLM): > 1000ms = Groq rate limit or complex synthesis

**Score blend α/β:**
- `α=0.6, β=0.4` (default): Trust textual relevance over graph structure
- `α=0.4, β=0.6`: Trust graph proximity (for structured queries like "who leads this company?")
- Configurable via `settings.yml: retrieval.gnn_weight`

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
| `p95_latency_ms` | > 3000 | ⚠️ Warning | Review retrieval stages; agentic path (IRCoT) routinely 4–8s — exclude from p95 or alert separately |
| `faithfulness` | < 0.70 (3-sample window) | ⚠️ Warning | Check recent document ingestions; may have extraction errors. Target is **≥ 0.85**; 0.70 is the alert floor, not the goal. |
| `contradiction_rate` | > 3.0 /1k edges | ⚠️ Warning | Moderate contradiction density — review recent ingestion batch |
| `contradiction_rate` | > 5.0 /1k edges | 🔴 Critical | High contradiction density — indicates schema drift or malformed source docs |
| `orphan_growth_rate` | > 0.30 | ⚠️ Warning | Entities not connecting to rest of graph; review extraction |
| `brier_score` | > 0.35 | ⚠️ Warning | Confidence calibration degraded; retrain or recalibrate. Note: 0.18 is achievable **after isotonic regression correction**; raw LLM confidence before correction typically scores 0.20–0.35. |

---

## How to use this in a pitch

**For a CTO evaluating the platform:**

1. **Start with KPI data**: "We have 156 queries recorded in the last week. Average latency is 1234ms with p95 at 2100ms. Faithfulness averages 0.78 — that's 78% of answers grounded in retrieved context."

2. **Show graph health**: "The knowledge graph has entity resolution quality of 0.92 on this curated domain — 0.70–0.85 is typical for noisier enterprise corpora. Contradiction rate is 1.2 per thousand edges — well below our 2.0 /1k health threshold. Community coherence is 0.62, which means Leiden found real semantic clusters."

3. **Demonstrate confidence**: "Our confidence calibration has a Brier score of 0.18 — in the 'good' range, after isotonic regression correction. That means when the system says a relation has 0.8 confidence, it's actually right roughly 80% of the time. Raw LLM confidence before correction typically scores 0.20–0.35."

4. **Explain the breakdown**: "Of our total latency, 45ms is retrieval, 210ms is reranking, 156ms is graph scoring, and 1348ms is LLM synthesis. The bottleneck is the LLM, not the graph."

---

## Next steps to instrument further

If deployed to production, consider adding:

- **User satisfaction signals**: Explicit feedback ("was this answer helpful?") to compute real (vs. synthetic RAGAS) accuracy
- **Cohort analysis**: Segment metrics by document domain, query type, user segment
- **Trend detection**: Alert when key metrics drift significantly (e.g., entity_resolution_quality drops from 0.92 to 0.85)
- **Cost optimization**: Break down LLM costs by retrieval mode (hybrid vs. agentic) to identify expensive paths
- **Extraction error analysis**: Tag specific extraction failures in the conflict queue and track them by model version
