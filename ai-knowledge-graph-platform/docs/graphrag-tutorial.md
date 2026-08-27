# The Complete GraphRAG Tutorial

A single-file walkthrough of everything this platform does — from knowledge
graph fundamentals to production deployment — mapped against a typical graph
engineer job description (Cypher, Neo4j, Python, graph algorithms, ML
integration, ETL pipelines, cloud).

Cross-references: [graphrag-terminology.md](graphrag-terminology.md) (A–Z
glossary), [cypher-patterns.md](cypher-patterns.md) (query cookbook),
[knowledge-graph-architecture.md](knowledge-graph-architecture.md) (data
model).

---

## Part 1 — Knowledge Graph Fundamentals

### 1.1 What is a knowledge graph?

A structured representation of entities (nodes) and typed relationships
(edges). Unlike a document store, a KG lets you *traverse* — follow a chain
of facts across documents that never mention each other.

```
(FAA-AD-2024-01-02) -[SUPERSEDES]-> (FAA-AD-2022-03-07)
(FAA-AD-2024-01-02) -[APPLIES_TO]-> (Boeing 737-800)
(Boeing 737-800)    -[MANUFACTURED_BY]-> (Boeing)
```

**In this project:** stored in Neo4j, 51 modules in `graphrag/graph/`.
Automotive tenant: 3,013 entities, 9,364 edges from 30 IATF documents.

### 1.2 Why GraphRAG over standard RAG?

| | Standard RAG | GraphRAG |
|---|---|---|
| Retrieval | vector similarity (optionally + BM25 hybrid) | vector + BM25 hybrid **+ graph traversal + inference** |
| Multi-hop questions | fails (no single chunk has the answer) | traverses entity relations across documents |
| Contradictions | invisible | detected and stored as `(:Conflict)` nodes |
| Provenance | chunk-level at best | per-edge: source doc, model, timestamp |
| Answer grounding | "trust me" | cited chunks + traceable graph paths |

Note: BM25/hybrid search is not exclusive to GraphRAG — many standard RAG
systems combine vector + BM25 too. The actual differentiator is the
**graph traversal + inference** layer, which lets the system chain facts
across documents and derive facts that were never explicitly written down
(see §4.5, Forward-chaining inference).

### 1.3 Property graph vs. triple store

This project uses a **property graph** (Neo4j): edges carry properties
(confidence, provenance, valid_from/valid_to). A triple store (RDF) needs
reification for the same. Decision record:
`docs/adr/0001-property-graph-over-triple-store.md`. RDF interop is still
available via `scripts/export_rdf.py` + `graphrag/graph/sparql_bridge.py`.

---

## Part 2 — Graph Data Modeling

### 2.1 The core schema

```
(:Document) -[:HAS_CHUNK]-> (:Chunk) -[:MENTIONS]-> (:Entity)
(:Document) -[:LINKS_TO {tenant, provenance, ACL snapshot, observed_at}]-> (:Document)
(:Entity) -[:HAS_SYSTEM_REPRESENTATION]-> (:SystemRepresentation)
(:Chunk) -[:ASSERTS_IN_CONTEXT]-> (:ContextualAssertion)
(:Entity) -[:RELATES_TO {relation, confidence, source_type, ...}]-> (:Entity)
(:Alias {value}) -[:ALIAS_OF]-> (:Entity)
(:Conflict) — contradiction records
(:Community) — Leiden cluster nodes with LLM summaries
```

Design choices worth defending in an interview:

- **Single physical label `RELATES_TO`, semantic type in a property.**
  Cypher can index and filter `{relation: 'SUPERSEDES'}` without one label
  per relation type (which would explode the schema and break generic
  traversal queries).
- **Composite entity key `(name, type, tenant)`** — the same string can be
  a different entity in a different type or tenant.
- **Bitemporal edges** — `valid_from`/`valid_to` (real-world validity) plus
  `recorded_at` (transaction time). Enables as-of queries: "what was the
  airworthiness status on 2024-02-01?"
- **Multi-tenancy by property, not by database** — every node/edge carries
  `tenant`; every query filters on it. Verified by
  `scripts/verify_tenant_isolation.py`.

Explicit HTML/Markdown/SharePoint references are the only inputs to
`LINKS_TO`; similarity never invents document links. LocalSearch follows these
edges with a bounded, ACL-aware expansion for link-dependent multi-hop queries.
System representations keep CRM and ERP contextual assertions separate beneath
the canonical tenant-scoped entity.

### 2.2 Ontology-driven validation

Entity types and relation domain/range constraints live in YAML
(`config/ontologies/*.yml`), loaded by `OntologyRegistry`. A `PERSON` cannot
`SUPERSEDES` a `REGULATION` — the triplet fails validation at write time.
Swapping the YAML retargets the whole platform to a new domain (aerospace →
automotive took one ontology file + one corpus).

---

## Part 3 — Cypher in Production

Full cookbook in [cypher-patterns.md](cypher-patterns.md). The patterns that
map directly to "writing highly optimized Cypher queries":

### 3.1 Multi-hop traversal with confidence decay

```cypher
MATCH (c:Chunk {tenant: $tenant})-[:MENTIONS]->(e:Entity)
      -[r:RELATES_TO*1..2]->(related:Entity)<-[:MENTIONS]-(c2:Chunk)
WHERE ALL(rel IN r WHERE rel.confidence >= $min_confidence)
RETURN c2, reduce(conf = 1.0, rel IN r | conf * rel.confidence) AS path_score
ORDER BY path_score DESC LIMIT $top_k
```

### 3.2 Hybrid search in one round trip

Vector ANN (`db.index.vector.queryNodes` on an HNSW index) and full-text
BM25 (`db.index.fulltext.queryNodes`) both run inside Neo4j — no separate
vector database. Results fused with Reciprocal Rank Fusion in Python.

### 3.3 Optimization techniques used here

- **Composite indexes** on `(tenant, type)` for entity lookups; vector and
  full-text indexes for chunks (see `graphrag/graph/schema.cypher`).
- **Batched writes** with `UNWIND $rows AS row MERGE ...` — ingestion
  batching cut entity/chunk write time dramatically (lessons A131–A132).
- **Scoped post-ingestion jobs** — inference and contradiction scans run
  only over the new document's entities (`run_for_document(doc_id)`), not
  the whole graph.
- **`PROFILE`/`EXPLAIN`** to verify index usage before shipping a query.

---

## Part 4 — Graph Algorithms

### 4.1 PageRank

**The algorithm:** a node is important if important nodes point to it.
Iteratively: `PR(n) = (1-d)/N + d · Σ PR(m)/outdegree(m)` over incoming
neighbors `m`, damping `d≈0.85`. In Neo4j it ships in the Graph Data
Science (GDS) library:

```cypher
CALL gds.pageRank.stream('entityGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS entity, score
ORDER BY score DESC LIMIT 10
```

**In this project:** implemented via GDS directly —
`graphrag/graph/neo4j_client.py: run_pagerank()` projects the tenant's
`Entity`/`RELATES_TO` subgraph in-memory (`gds.graph.project.cypher`,
weighted by edge confidence, dropped after use) and calls
`gds.pageRank.stream`. `graphrag/graph/pagerank.py: PageRankComputer`
orchestrates compute + persist (`e.pagerank`, `e.pagerank_computed_at` on
each `Entity` node). Exposed via:
- `POST /kg/pagerank/compute?tenant=<t>` — run and persist
- `GET /kg/pagerank/top-entities?tenant=<t>&top_k=<n>` — read results
- `python scripts/pagerank_compute.py --tenant automotive` — standalone runner

Real output on the automotive tenant (3,013 entities): top-ranked is
`furnizorii` ("the suppliers", ORG, score 43.87), then `PlastiAuto SRL`
(22.32), `ISO/IATF` (13.74), `AutoCorp GmbH` (12.13) — the entities most
referenced across the supplier-quality corpus. On aerospace (156 entities):
`airworthiness directive` (1.77) tops the list, matching the corpus's
AD-supersession-chain structure.

This is a **global, query-independent** signal — complementary to, not a
replacement for, the **GNN scorer** (`graphrag/graph/gnn_scorer.py`,
GCN/GAT), which re-scores chunks by their position in the entity subgraph
*relative to a specific query*. Message passing in a GCN is a learned
generalization of the PageRank power iteration (both propagate scores along
edges with normalization) — PageRank answers "what's important overall,"
the GNN scorer answers "what's important for this question."

### 4.2 Community detection (Louvain → Leiden)

**The algorithm family:** find groups of nodes denser internally than
externally, by maximizing modularity. **Louvain** is the classic greedy
approach; **Leiden** fixes Louvain's disconnected-community defect and is
what this project runs.

**In this project:** `graphrag/graph/community_builder.py` runs
multi-resolution Leiden via `graspologic` (multiple gamma values →
communities at several granularities). Each community gets an LLM-generated
summary (`community_summarizer.py`) used by global search for corpus-wide
thematic questions. Community coherence is tracked as a graph-health metric.

### 4.3 GNNs — GCN and GAT

- **GCN:** aggregates neighbor features with equal (degree-normalized)
  weights. Default here.
- **GAT:** learns attention weights per neighbor — an entity's edge to the
  directly-relevant directive gets amplified over its edge to background
  context. Selectable via `gnn_type: gat` in config.

Stage 5 of retrieval: chunks are re-scored by
`final_score = α·cross_encoder + β·gnn_score`.

### 4.4 Link prediction (TransE)

`graphrag/graph/link_predictor.py` — learns embeddings where
`head + relation ≈ tail`; predicts plausible unobserved edges. Used as an
extra signal for entity resolution.

### 4.5 Forward-chaining inference

Datalog-style rules (transitivity, symmetry, inverse, composition)
materialize derived edges at write time with confidence decay:
`AD-2024 supersedes AD-2022 (0.95)` + `AD-2022 supersedes AD-2020 (0.95)`
⟹ inferred `AD-2024 supersedes AD-2020 (0.9025)`. Runs to fixpoint.
`graphrag/graph/inference_engine.py`, ADR-0002.

---

## Part 5 — ETL / Ingestion Pipeline

The "automated ETL ingestion pipelines" requirement, end to end:

```
raw docs → chunker → LLM extractor → validation → entity resolution
        → Neo4j writer (batched, natural-key MERGE) → inference engine
        → contradiction scan → community rebuild → PageRank recompute
```

1. **Chunking** — `graphrag/ingestion/chunker.py`, heading-aware section
   splitting (512 tokens, 64 overlap), so table rows keep their section
   headings for embedding quality.
2. **Extraction** — Groq by default (`get_llm()`, with DeepSeek fallback;
   `LLM_INGEST_PROVIDER=deepseek` or `cerebras` selects an alternate primary,
   and Groq remains the default fast-routing provider for agentic retrieval)
   produces entities + relations as JSON with per-relation
   confidence; clamped and schema-validated in `extractor.py`.
3. **Ontology validation** — domain/range check per triplet.
4. **Entity resolution** — 4-stage alias pipeline: exact → normalized →
   embedding similarity → human review queue (`alias_registry.py`).
5. **Batched graph writes** — natural-key `MERGE` on `(tenant, filename)`
   for documents and `(document_id, chunk_index)` for chunks — re-ingesting
   the same file updates the existing nodes instead of duplicating them
   (A136). `UNWIND`-batched entity embeddings and chunk writes (A131–A132
   performance work).
6. **Bayesian confidence merge** — same relation from two independent docs:
   `fused = 1 − (1−c₁)(1−c₂)` (ADR-0003). Two 0.8s fuse to 0.96, not
   average to 0.8.
7. **Post-ingestion jobs** (`GraphWriter.validate_and_check_cycles`) —
   scoped validation, cycle detection (whole-graph, deferred during bulk
   ingestion), auto-quarantine, contradiction scan (4 types — see 6.4),
   community rebuild, PageRank recompute — the last two use a cheap
   count-based staleness check first and only pay the expensive full-graph
   cost when actually triggered (A139).
8. **Async orchestration** — RabbitMQ queues with dead-letter queues;
   idempotent re-runs via checkpoint (resume without `--wipe`).

Run: `py -3.11 scripts/ingest_corpus.py --commit` (add `--wipe` only for a
full rebuild).

### 5.1 Worked example — ingesting, then re-ingesting the same document

Fake data, tenant `aerospace`:

**First ingestion of `AD-2024-99.txt`**:
```
chunk_0: "AD-2024-99 requires Boeing to inspect the tail fin bracket
          every 500 flight hours."
→ entities: Boeing (ORG), tail fin bracket (PART)
→ relation: (AD-2024-99) -REQUIRES_INSPECTION_OF-> (tail fin bracket),
            confidence=0.85
```
`MERGE (d:Document {tenant:"aerospace", filename:"AD-2024-99.txt"})` — no
match, `ON CREATE` fires, new `Document` + `Chunk` + `Entity` + `RELATES_TO`
nodes created.

**Same file re-ingested later** (source text unchanged, or a revised PDF
with the same filename):
```
MERGE (d:Document {tenant:"aerospace", filename:"AD-2024-99.txt"})
→ matches the existing node → canonical_id = the original id
→ is_reingest = True (canonical_id != the fresh id this run generated)

MERGE (r:RELATES_TO {relation:"REQUIRES_INSPECTION_OF"})
→ old confidence 0.85, new confidence 0.85 (same source re-asserting)
→ merged: 1 - (1-0.85)(1-0.85) = 0.9775   ← boosted, not just refreshed
→ r.extracted_at reset to now (decay clock restarts)
```
Post-ingestion jobs: community rebuild is **skipped** (entity/edge counts
unchanged, no growth drift) — but PageRank recomputes anyway because
`is_reingest=True` forces it regardless of drift, since the confidence
boost above changed PageRank's weighted inputs with zero change to
entity/edge count (A139).

---

## Part 6 — The Retrieval Pipeline (AI Integration)

Six stages, each addressing a failure mode of the previous:

| Stage | What | Why |
|---|---|---|
| 1. Vector ANN | HNSW over 3072-d OpenAI embeddings | semantic recall |
| 2. BM25 | Neo4j full-text | exact identifiers ("AD-2024-01-02") embeddings blur |
| 3. RRF fusion + cross-encoder rerank | `ms-marco-MiniLM-L-6-v2` | precision on the fused pool |
| 4. Multi-hop traversal | 2-hop entity walk plus bounded explicit-document-link traversal | facts no single chunk contains; link-dependent questions |
| 5. GNN re-scoring | GCN/GAT over the query subgraph | structural relevance |
| 6. LLM synthesis | Groq by default (DeepSeek fallback), cited chunks + graph facts + open-conflict warnings | grounded, auditable answer |

Fallbacks: **agentic retrieval** (IRCoT — retrieve→reason→retrieve, max 4
steps) when confidence is low; **global search** (direct retrieval of bounded
community summaries) for corpus-wide thematic questions; **session context** (Redis)
for multi-turn follow-ups.

### 6.1 Query rewriting (Stage 0, disabled by default)

`graphrag/retrieval/query_rewriter.py` — a fast 8B-model pass that expands
the search query (acronym expansion, revision-phrasing normalization,
1-2 synonym terms) before Stage 1. Wired into
`HybridRetriever.retrieve_and_answer` immediately before the local/global
search calls. Critically, it rewrites **only the string used for
retrieval** — answer synthesis and RAGAS evaluation always use the
original question, so grading is never against a paraphrase. Fails open:
any error or malformed output falls back to the raw question.

Gated by `retrieval.query_rewrite_enabled` in `config/settings.yml`
(default `false`), because it was measured, not assumed:

**A/B on the automotive golden set (10 questions, identical corpus/judge):**

| | Pass rate | Faithfulness |
|---|---|---|
| Rewrite OFF (baseline) | **9/10 (90%)** | 0.917 |
| Rewrite ON | 8/10 (80%) | **0.967** |

Net effect was mixed at the per-question level, not just the aggregate:
faithfulness rose because the expanded query improved grounding on a
vague contradiction question (CON-02: 0.67 → 1.00), but pass rate fell
because it broke an exact single-hop lookup (SH-02: PASS → FAIL) — the
expanded query retrieved different chunks and missed the one carrying
the required "1%" figure and its citation. Query expansion trades recall
for precision, and single-hop factoid questions are precision-sensitive
in a way multi-hop/contradiction questions aren't.

Shipped disabled rather than reverted, since the module and the A/B
methodology are reusable: the likely next step is **type-aware
routing** — expand only for `multi_hop`/`contradiction` query types,
skip it for `single_hop` — rather than an unconditional Stage 0.

### 6.2 Post-synthesis claim verification (CoVe-style, disabled by default)

`graphrag/retrieval/claim_verifier.py` — a Chain-of-Verification-style pass
that re-checks each sentence of the synthesized answer against the
retrieved context and strips claims it can't ground. Gated by
`retrieval.claim_verification` (default `false`).

**Re-confirmed via A/B** (same automotive golden set, `query_rewrite_enabled`
held constant at `false`):

| | Pass rate | Faithfulness |
|---|---|---|
| OFF (baseline) | 9/10 (90%) | **0.917** |
| ON (claim verification) | 9/10 (90%) | **0.800** |

Pass rate was unaffected, but faithfulness dropped 0.117 — the verifier
strips correctly-grounded claims, not just hallucinated ones. Per-question:
CON-03, NEG-01, and NEG-02 each lost ~0.5 faithfulness while only CON-02
improved, concentrated on negative/contradiction answers where the
strict `_ANSWER_PROMPT` grounding rules are already doing the real work.

**Conclusion: a CoVe-style post-hoc verification layer is not worth
building for this system.** This is the second independent measurement
of the same failure mode (the original disable predates this re-test) —
grounding belongs at generation time via prompt constraints, not as a
verify-and-strip pass afterward. Not pursuing CoVe further unless paired
with a materially more reliable verifier model than the current one.

**Exposing graph features to ML models** (the JD's "AI Integration" line):
the graph feeds the GNN adjacency + node features, the entity-resolution
embedding comparisons, TransE link prediction, and the retrieval context
itself — the KG is a feature store for every model in the loop.

### 6.3 Worked example — a query end to end

Question: *"Which document currently governs engine mount inspection for
the Boeing 737 MAX?"*, tenant `aerospace`.

1. **Vector ANN + BM25** — pulls chunks mentioning "engine mount
   inspection," "737 MAX," and specific AD numbers; RRF-fused into one
   ranked list.
2. **Cross-encoder rerank** — re-scores the fused pool against the literal
   question text; top 5 chunks become the seed set.
3. **Multi-hop traversal** — from the seed chunks' entities (`737 MAX
   engine mount`, `AD-2022-03-07`, `AD-2024-01-02`), walks `SUPERSEDES`
   edges 2 hops out — finds `AD-2024-01-02` supersedes `AD-2022-03-07`,
   which partially supersedes `AD-2020-05-11`.
4. **GNN re-scoring** — blends text relevance with structural position;
   `AD-2024-01-02` (the current, non-superseded directive) scores highest.
5. **Conflict check** — `HybridRetriever` looks up open conflicts for the
   retrieved entities; none found here, so no warning is added to context.
6. **LLM synthesis** — answer: *"FAA-AD-2024-01-02 currently governs it —
   it supersedes AD-2022-03-07, which itself partially superseded
   AD-2020-05-11."* Citations: `[FAA-AD-2024-01-02, FAA-AD-2022-03-07]`.

If step 5 *had* found an open conflict on one of these entities (e.g. two
docs disagreeing on which directive is current), the context would include
`"⚠ Unresolved conflicts: ..."` and the LLM would be instructed to state
the disagreement rather than pick a directive silently — see 6.4.

### 6.4 Conflict detection and resolution workflow

Detected automatically during ingestion (`ContradictionDetector.scan()`,
step 7 in Part 5); resolved manually — nothing auto-resolves today.

**4 detection types, each with a fake-data example**:

| Type | Example |
|---|---|
| `directional_reversal` | Doc A: `(Boeing) -OWNS-> (Spirit AeroSystems)`. Doc B: the reverse. |
| `exclusive_state` | `(tail fin bracket)` is `"active"` per Doc A, `"deprecated"` per Doc B. |
| `functional_violation` | `(AD-2024-99) -MANUFACTURER-> (Boeing)` per Doc A, `-> (Airbus)` per Doc B — single-valued relation, two targets. |
| `positive_negative_pair` | Doc A: `(G-ABCD) -COMPLIES_WITH-> (AD-2024-99)`. Doc B: `(G-ABCD) -NOT_COMPLIES_WITH-> (AD-2024-99)`. |

**Not a conflict**: the same triple reported identically by two docs is
corroboration (`independent_source_count`), not a contradiction — this
distinction is why the retired `multi_source` strategy was wrong (A135).

**Workflow once detected**:
1. A `Conflict` node is created — `{status:"open", conflict_type, src, tgt,
   relation, sources, detected_at}`. Both conflicting facts stay in the
   graph untouched.
2. **Retrieval-side warning**: any query touching either entity gets a
   `"⚠ Unresolved conflicts:"` section appended to the LLM's context (see
   6.3, step 5) — the LLM is instructed to state the disagreement, not
   pick a side.
3. **Human resolution** — via the dashboard's Conflicts tab or
   `POST /corrections/conflict/resolve`, a person marks it:
   - `resolved_authority` — picked because one source has higher
     `authority_level` (a human decision informed by that field, not an
     automatic lookup — nothing currently auto-resolves by authority).
   - `resolved_manual` — a judgment call with no clean authority tiebreaker.
   - `false_positive` — dismissed as not a real contradiction.
4. Until resolved, visible via `GET /corrections/conflicts` and counted in
   `conflict_rate()` (open conflicts ÷ total edges) — a tracked graph
   quality metric.

---

## Part 7 — Evaluation & Observability

- **RAGAS** (`graphrag/evaluation/ragas_evaluator.py`): faithfulness,
  answer relevancy, context precision/recall — LLM-as-judge on a 20% query
  sample. No committed results file backs a per-tenant faithfulness figure
  for the automotive tenant — `evals/` holds aerospace faithfulness plus the
  hop-ranking, MMR and SPLADE benchmarks only. Re-run
  `scripts/run_golden_eval.py --tenant automotive` before quoting a number.
- **Golden datasets** (`data/eval_golden/`): 9–10 questions per tenant across
  single-hop / multi-hop / contradiction / negative types, each with
  `expected_citations`, `required_answer_terms`, `forbidden_terms` — a
  deterministic gate independent of the LLM judge. Known-failing edge
  cases documented separately (`queries_automotive_deferred.json`) with
  root-cause notes rather than silently dropped.
- **Graph health** (`graph_evaluator.py`): entity-resolution quality,
  relation precision, contradiction rate, orphan growth, community
  coherence — persisted as `GraphHealthSnapshot` nodes for trend tracking.
  RAGAS measures answers; these measure the graph itself.
- **Confidence calibration**: Brier score + isotonic calibration curves
  (`confidence_calibration.py`) — is a 0.9-confidence edge actually right
  90% of the time?
- **KPIEvents**: per-query latency, scores, retrieval mode →
  `GET /kpis/summary`, dashboard.

---

## Part 8 — APIs & Serving

- **FastAPI** (`api/main.py`): `/query` (async — publishes to RabbitMQ,
  poll for result), `/graph/entities/{id}/provenance`, `/kg/conflicts`,
  `/kg/snapshots`, `/kpis/*`, `/demo` (interactive UI with
  chain-of-thought trace steps).
- **Workers**: consume the queue, run the five-stage retrieval pipeline, perform the separate LLM synthesis step, and write results to Redis.
- Clean separation: API never touches Neo4j for queries — everything goes
  through the worker, so retrieval load can scale independently.

### 8.1 Cross-process result flow — worker writes, API polls

`ResultStore` (`graphrag/retrieval/result_store.py`) exists because the API
and the worker are **separate containers** — a module-level dict in the
worker's memory is invisible to the API process. Redis is the only thing
both sides can see.

1. API receives `POST /query` → publishes to RabbitMQ → immediately writes
   `{"status": "processing", "query_id": ...}` to Redis and returns
   `query_id` to the client.
2. Worker picks up the message, runs the five retrieval stages and then performs LLM synthesis. If the result is weakly grounded, the IRCoT fallback can issue bounded retrieve→reason→retrieve passes. During processing, each stage calls `push_progress(query_id, step)` — this does a
   Redis `GET` (read current entry), appends the step name to a `steps`
   list, `SET`s it back. This is what feeds the chain-of-thought trace in
   the `/demo` UI.
3. **Client polls `GET /query/{id}` at any point**, including mid-processing.
   There's no lock, no "in progress" error — the client just gets whatever
   was last written: `{"status": "processing", "steps": [...]}` (partial)
   or `{"status": "completed", "answer": ..., "steps": [...]}` (final).
4. Worker finishes → `set(query_id, {"status": "completed", "answer": ...,
   "steps": prior_steps})`, preserving the steps accumulated during
   processing so the trace stays intact through the final read.

**Failure mode is asymmetric with the other Redis consumers in this
codebase.** Session store, query cache, alias registry, and the alert log
all fall back to an in-memory structure when Redis is unreachable — the
request degrades (slower, or not shared across workers) but still
completes. `ResultStore` can't do that: the worker's memory and the API's
memory are different processes, so a memory fallback would just mean the
API never sees what the worker wrote. Instead, `set()`/`get()` log at **ERROR** level — not WARNING, unlike every
other Redis consumer in §8.2 — and return without touching a local dict.
The ERROR level is deliberate (see the comment at
`result_store.py:104-109`): a WARNING would suggest "degraded but fine,"
but here the result is genuinely gone. `set()` after a mid-write failure
means the client polling `GET /query/{id}` gets `None` (looks like "not
found") and eventually times out; `get()` after a mid-read failure returns
`None` even if the worker's `set()` actually succeeded moments earlier.
This is the one Redis consumer where a Redis outage produces a lost query
rather than a degraded one.

### 8.2 The other five Redis consumers — short before/after workflows

Every other Redis usage in this codebase follows the same shape: read
before the operation, write after, fall back to an in-memory structure on
Redis failure. Concrete examples:

| Consumer | Before | Redis op | After |
|---|---|---|---|
| **Session store** (`session_store.py`) | new message arrives in an existing conversation | `RPUSH graphrag:session:<id> turn_json` | next message in the session does `LRANGE` to rebuild history for the LLM prompt |
| **Governed answer cache** (`query_cache.py`) | a stateless query enters `HybridRetriever`; Neo4j supplies the tenant corpus revision | SHA-256 over canonical tenant/query/corpus/model/prompt/ontology/retrieval inputs, then `GET`; on a governed miss, `SETEX` stores the answer and source trace | unchanged inputs return immediately with a new `query_id` linked to the original `source_trace_id`; any retrieval-visible mutation changes the corpus revision, making every old key unreachable |
| **Alias registry** (`alias_registry.py`) | an ingestion batch finishes, new aliases exist (e.g. "Boeing" → "The Boeing Company") | `HSET graphrag:aliases:<tenant> alias "name\|type"`, `EXPIRE 86400` | a different worker sees "Boeing" in a query → resolves via Redis `HGET` instead of a Neo4j round-trip |
| **Alerts** (`alerts.py`) | a monitoring check fires (e.g. LLM provider unhealthy) | `LPUSH graphrag:alerts:recent alert_json`, `LTRIM 0 ALERT_HISTORY-1` | dashboard reads the list to show recent alerts, capped history |
| **Rate limiter** (`api/limiter.py`) | `POST /query` request arrives from a client | `slowapi` checks/increments the per-IP counter against `60/minute` | over limit → `429` immediately, no Neo4j/LLM touched; under limit → request proceeds |

The support caches degrade differently. Session and alias data can use local
fallbacks. The governed answer cache uses memory only when Redis was
unavailable at initialization; a runtime Redis error is a cache miss, so the
request executes live rather than trusting process-local stale data.
`ResultStore` (§8.1) remains the strict cross-process transport exception.

**RabbitMQ's failure mode is not the same shape.** It isn't a cache with a
memory substitute — it *is* the cross-process transport, so there's
nothing to fall back to. Instead (`graphrag/messaging/rabbitmq_client.py`):
connection loss is handled by `aio_pika.connect_robust`, which
auto-reconnects in the background. A handler exception on an individual
message doesn't touch the connection at all — it's retried with
exponential backoff (1s, 2s, 4s… capped at 30s, up to `MAX_RETRIES = 3`),
and after that sent to a per-queue dead-letter queue (`<queue>.dlq`) with a
structured envelope (`exception_type`, `error`, `retry_count`,
`payload_summary`) so ops can triage without parsing raw headers — the
original message is `ack()`'d either way so it doesn't block the queue.

### 8.3 RabbitMQ workflows — short before/after examples

| Flow | Before | RabbitMQ op | After |
|---|---|---|---|
| **Ingest** (`IngestionConsumer`) | document uploaded via API | publish `IngestMessage` to `INGEST_EXCHANGE`/`INGEST_QUEUE` | worker's `handle()` picks it up, runs `IngestionAgent.run(msg)` — this is the entry point into the whole ingestion pipeline (Part 5) |
| **Query** (`QueryConsumer`) | `POST /query` from client | publish `QueryMessage`; API writes queued state to Redis and returns a fresh `query_id` | `QueryAgent` enters `HybridRetriever`, which checks the versioned answer cache; a hit returns the prior answer plus source trace metadata, while a miss runs retrieval and caches only after trace persistence; the consumer writes either result to `ResultStore` |
| **Eval sampling** (`EvaluationConsumer`) | a query result just completed | `QueryConsumer` publishes an `EvalJob` for ~20% of queries (`eval_sample_rate`) — async, doesn't block the client's answer | `EvaluationConsumer` picks it up, runs RAGAS scoring (faithfulness, relevancy, etc.) against the sampled query |
| **Handler failure, any consumer** | e.g. `IngestionAgent.run()` raises (Neo4j timeout, malformed LLM output) | message headers get `x-retry-count` incremented, republished after backoff (1s → 2s → 4s, cap 30s) | after 3 failed retries, message → `<queue>.dlq` with a structured envelope (`exception_type`, `error`, `payload_summary`) for manual triage; original message acked either way so the queue isn't blocked |
| **Connection drop** (any consumer) | RabbitMQ container restarts or network blip | `aio_pika.connect_robust` detects the drop | connection and channels are re-established automatically in the background — consumers resume without code intervention, no manual reconnect logic needed |

---

## Part 9 — Cloud Deployment

Deployed to **Fly.io** (7 apps: API, workers, Neo4j, RabbitMQ, Redis,
dashboard, evaluation — Amsterdam region, private networking, persistent
volumes). Torn down when idle; local Docker is the source of truth.

**GCP translation** (the JD prefers GCP — same architecture, different
names):

| Component | Fly.io | GCP | AWS |
|---|---|---|---|
| API + workers | Machines | Cloud Run | ECS Fargate |
| Neo4j | Machine + volume | Compute Engine + Persistent Disk (or AuraDB managed) | EC2 + EBS |
| Redis | Machine | Memorystore | ElastiCache |
| RabbitMQ | Machine | Pub/Sub (rearchitect) or GCE | Amazon MQ / SQS |
| Private networking | automatic | VPC | VPC |
| Secrets | fly secrets | Secret Manager | Secrets Manager |

Honest interview framing: "I deployed the full stack as containers with
private networking, persistent volumes, and per-service scaling on Fly.io —
the same Docker images run on Cloud Run/GCE unchanged; the concepts (VPC,
volumes, service discovery, secrets) map one-to-one."

