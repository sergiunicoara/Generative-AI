# The Complete GraphRAG Tutorial

A single-file walkthrough of everything this platform does — from knowledge
graph fundamentals to production deployment — mapped against a typical graph
engineer job description (Cypher, Neo4j, Python, graph algorithms, ML
integration, ETL pipelines, cloud).

Cross-references: [graphrag-terminology.md](graphrag-terminology.md) (A–Z
glossary), [cypher-patterns.md](cypher-patterns.md) (query cookbook),
[knowledge-graph-architecture.md](knowledge-graph-architecture.md) (data
model), [jd-mapping.md](jd-mapping.md) (requirement-by-requirement evidence).

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

**In this project:** stored in Neo4j, 39 modules in `graphrag/graph/`.
Automotive tenant: 3,013 entities, 9,364 edges from 14 IATF documents.

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
        → Neo4j writer (batched) → inference engine → contradiction scan
        → community rebuild
```

1. **Chunking** — `graphrag/ingestion/chunker.py`, heading-aware section
   splitting (512 tokens, 64 overlap), so table rows keep their section
   headings for embedding quality.
2. **Extraction** — Groq LLM produces entities + relations as JSON with
   per-relation confidence; clamped and schema-validated in
   `extractor.py`.
3. **Ontology validation** — domain/range check per triplet.
4. **Entity resolution** — 4-stage alias pipeline: exact → normalized →
   embedding similarity → human review queue (`alias_registry.py`).
5. **Batched graph writes** — `UNWIND`-batched entity embeddings and chunk
   writes (A131–A132 performance work).
6. **Bayesian confidence merge** — same relation from two independent docs:
   `fused = 1 − (1−c₁)(1−c₂)` (ADR-0003). Two 0.8s fuse to 0.96, not
   average to 0.8.
7. **Post-ingestion jobs** — scoped inference, contradiction scan (5
   contradiction types), community rebuild.
8. **Async orchestration** — RabbitMQ queues with dead-letter queues;
   idempotent re-runs via checkpoint (resume without `--wipe`).

Run: `py -3.11 scripts/ingest_corpus.py --commit` (add `--wipe` only for a
full rebuild).

---

## Part 6 — The Retrieval Pipeline (AI Integration)

Six stages, each addressing a failure mode of the previous:

| Stage | What | Why |
|---|---|---|
| 1. Vector ANN | HNSW over 3072-d OpenAI embeddings | semantic recall |
| 2. BM25 | Neo4j full-text | exact identifiers ("AD-2024-01-02") embeddings blur |
| 3. RRF fusion + cross-encoder rerank | `ms-marco-MiniLM-L-6-v2` | precision on the fused pool |
| 4. Multi-hop traversal | 2-hop entity walk with confidence decay | facts no single chunk contains |
| 5. GNN re-scoring | GCN/GAT over the query subgraph | structural relevance |
| 6. LLM synthesis | Groq, with cited chunks + graph facts | grounded, auditable answer |

Fallbacks: **agentic retrieval** (IRCoT — retrieve→reason→retrieve, max 4
steps) when confidence is low; **global search** (map-reduce over community
summaries) for corpus-wide thematic questions; **session context** (Redis)
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

---

## Part 7 — Evaluation & Observability

- **RAGAS** (`graphrag/evaluation/ragas_evaluator.py`): faithfulness,
  answer relevancy, context precision/recall — LLM-as-judge on a 20% query
  sample. Automotive golden set: **0.950 faithfulness, 10/10 deterministic
  gates**. Aerospace: ~0.87.
- **Golden datasets** (`data/eval_golden/`): 10 questions per tenant across
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
  `/kg/health/snapshot`, `/kpis/*`, `/demo` (interactive UI with
  chain-of-thought trace steps).
- **Workers** (`workers/query_worker.py`): consume queue, run the 6-stage
  pipeline, write results to Redis.
- Clean separation: API never touches Neo4j for queries — everything goes
  through the worker, so retrieval load can scale independently.

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

---

## Part 10 — JD Compliance Matrix

| JD line | Evidence in this project |
|---|---|
| Cypher expertise | 39 graph modules, patterns cookbook, batched UNWIND writes, PROFILE-verified indexes |
| Neo4j | core store: graph + vector + full-text in one engine, multi-tenant |
| Python | entire platform (~30k LOC, 380 tests) |
| Graph data modeling | property-graph schema, ontology-validated, bitemporal, multi-tenant |
| PageRank | GDS `gds.pageRank.stream`, live: `POST/GET /kg/pagerank/*` + `scripts/pagerank_compute.py` (§4.1) |
| Community detection | multi-resolution Leiden + LLM summaries, coherence tracked (§4.2) |
| ML integration | GNN re-scoring, TransE, embeddings, RAGAS LLM-judge, cross-encoder (§6) |
| ETL pipelines | 8-step async ingestion with DLQs, checkpointing, batching (§5) |
| Clean APIs | FastAPI + async worker split, provenance/conflict/health endpoints (§8) |
| Cloud (GCP/AWS) | Fly.io production deploy; one-to-one GCP mapping (§9) |

Gaps to state honestly: no hands-on GCP console time (concepts transfer,
services table above).
