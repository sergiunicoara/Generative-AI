# AI Knowledge Graph & Ontology Platform

See [enterprise content governance](docs/enterprise-content-governance.md) for
ACL-aware retrieval, governed metadata, synchronization and reviewed document
lineage/obligations.

A production-grade platform for building, querying, and reasoning over enterprise knowledge graphs — with formal OWL ontology enforcement, SPARQL querying, OWL-RL reasoning, TransE link prediction, forward-chaining inference, and an LLM-augmented retrieval pipeline.

Built for domains where **graph quality is a first-class requirement**: regulatory compliance, aerospace engineering, and other knowledge-intensive fields where facts must be provenance-tracked, time-aware, and semantically consistent across sources.

---

## Knowledge Graph Layer

The graph is not a RAG index. It is a formally modeled knowledge base:

| Capability | Implementation |
|---|---|
| **Ontology enforcement** | Versioned entity type taxonomy (`SUBCLASS_OF` hierarchy); domain/range constraints on every relation write; deprecated relation auto-migration |
| **Forward-chaining inference** | Datalog-style rules (transitivity, symmetry, inverse, composition); derived edges tagged `source_type=inferred` with per-hop confidence decay |
| **Entity resolution** | 4-stage pipeline: exact/normalised → fuzzy (Levenshtein ≥ 85) → embedding cosine (≥ 0.92) → new entity |
| **Bitemporal modeling** | Valid time (`valid_from`/`valid_to`) + transaction time (`recorded_at`, immutable); `as_of(vt, tt)` queries for point-in-time reconstruction |
| **Contradiction detection** | 5 typed conflict classes: multi-source, directional reversal, exclusive state, functional violation, positive/negative pair |
| **Negative knowledge** | `NEGATIVE_RELATES_TO` edges with full provenance; conflict detection when positive and negative assertions coexist |
| **Reification** | `Statement` nodes for meta-assertions (endorsements, epistemic annotations) |
| **Confidence model** | Bayesian accumulation across sources: `1−(1−c₁)(1−c₂)`; authority-weighted decay; temporal half-life; isotonic calibration correction |
| **Document authority** | 4-level hierarchy (Regulatory → Manufacturer → Internal → Informal); `SUPERSEDES` chains penalise outdated sources |
| **Graph health metrics** | 6 semantic indicators (alias coverage, contradiction rate, orphan rate, community coherence…) with per-tenant trend snapshots and alert thresholds |
| **RDF / OWL / SKOS export** | `scripts/export_rdf.py` serialises to Turtle or JSON-LD with `owl:NamedIndividual`, `owl:ObjectProperty`, `rdfs:subClassOf`, reified confidence annotations, and tenant-scoped `skos:ConceptScheme` navigation; `--infer` applies OWL-RL closure before writing |
| **OWL-RL reasoning** | `OWLRLReasoner` (owlrl) materialises subClassOf chains, symmetric/inverse properties; `is_consistent()` detects owl:Nothing entailments |
| **SPARQL bridge** | `SPARQLBridge.from_turtle()` + `POST /kg/sparql` — SPARQL 1.1 SELECT in-process over any Turtle export; pre-built queries for entity relations, subclass hierarchy, confidence summary |
| **Link prediction** | `LinkPredictor` wraps trained `TransXTrainer` (TransE): `predict_tail(h,r,?)`, `predict_relation(h,?,t)`, `find_missing_links()` via Neo4j vector ANN; `POST /kg/predict-links` |
| **Domain ontologies** | Config-driven domain overlays (including `legal_contracts.yml`) — extend type hierarchy and relation schema without code changes; `generate_synthetic_ontology.py` generates large synthetic ontologies for load testing (no benchmark figure is committed — do not quote one) |

**Further reading:**
- [`docs/roadmap.md`](docs/roadmap.md) — current implementation status, Context Graph evaluation gate, and scaling path
- [`docs/audit-2026-08-21.md`](docs/audit-2026-08-21.md) — current architecture, security, dependency, scalability, and state-of-the-art audit
- [`docs/adr/ADR-Context-Graph-Decision-Trace.md`](docs/adr/ADR-Context-Graph-Decision-Trace.md) — decision trace, manifest, governance, and integrity contract
- [`docs/knowledge-graph-architecture.md`](docs/knowledge-graph-architecture.md) — architectural decisions, data model, LLM routing, cross-process result store
- [`docs/ontology-model.md`](docs/ontology-model.md) — formal type hierarchy, relation schema, inference rules, design decisions
- [`docs/entity-resolution.md`](docs/entity-resolution.md) — 4-stage resolution pipeline with examples
- [`docs/cypher-patterns.md`](docs/cypher-patterns.md) — 6 production Cypher patterns: multi-hop traversal, bitemporal as-of, transitive supersession, contradiction scan, community ANN search, entity resolution audit
- [`docs/runbook.md`](docs/runbook.md) — operations: startup order, common failures, backup/restore, schema migration
- [`docs/graphrag-terminology.md`](docs/graphrag-terminology.md) — every GraphRAG term defined, with examples and file references
- [`docs/performance-metrics-inventory.md`](docs/performance-metrics-inventory.md) — all 16 metrics (KPI events, graph health, calibration, retrieval stages); storage, access, interpretation, pitch guidance
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) — ADR process, PR checklist, coding standards, how to add features
- [`docs/adr/0001-property-graph-over-triple-store.md`](docs/adr/0001-property-graph-over-triple-store.md) — Why Neo4j over RDF triple stores
- [`docs/adr/0002-forward-chaining-over-backward-chaining.md`](docs/adr/0002-forward-chaining-over-backward-chaining.md) — Why materialised inference over query-time reasoning
- [`docs/adr/0003-bayesian-confidence-accumulation.md`](docs/adr/0003-bayesian-confidence-accumulation.md) — Why `1−(1−c₁)(1−c₂)` over last-write-wins
- [`docs/adr/0004-groq-over-gemini-for-text-generation.md`](docs/adr/0004-groq-over-gemini-for-text-generation.md) — LLM provider selection; two-model design rationale; OpenAI for embeddings
- [`docs/adr/0005-redis-as-cross-process-result-store.md`](docs/adr/0005-redis-as-cross-process-result-store.md) — Why Redis over PostgreSQL and RabbitMQ reply-to for result persistence
- [`docs/adr/0006-dual-llm-architecture.md`](docs/adr/0006-dual-llm-architecture.md) — Why Groq 8B routing + DeepSeek synthesis; historical latency benchmark is clearly labeled
- [`docs/adr/0007-capability-gated-neo4j-vector-search.md`](docs/adr/0007-capability-gated-neo4j-vector-search.md) — Neo4j 2026 `SEARCH` with a 5.20 compatibility fallback
- [`docs/adr/0008-adaptive-retrieval-routing.md`](docs/adr/0008-adaptive-retrieval-routing.md) — Measured tenant-scoped retrieval route selection
- [`docs/adr/0009-agent-platform-trust-boundaries.md`](docs/adr/0009-agent-platform-trust-boundaries.md) — Agent identity, capability, write, and telemetry trust boundaries
- [`docs/adr/ADR-Context-Graph-Decision-Trace.md`](docs/adr/ADR-Context-Graph-Decision-Trace.md) — Bounded Context Graph ownership, trace integrity, and privacy rules
- [`docs/mcp-operations.md`](docs/mcp-operations.md) — authenticated Streamable HTTP MCP operations and deployment gate
- [`docs/mcp-operations.md`](docs/mcp-operations.md) — authenticated MCP operations and the reusable, compatibility-tested capability contract
- [`docs/templates/production-evidence-template.json`](docs/templates/production-evidence-template.json) — source-of-truth fields for production scale and business-impact claims
- [`docs/templates/public-evaluation-report-template.md`](docs/templates/public-evaluation-report-template.md) — reproducible public evaluation report template
- [`docs/public-local-evaluation-report.md`](docs/public-local-evaluation-report.md) — generated, bounded local MCP/retrieval/write evidence
- [`docs/articles/governed-mcp-and-agent-writes.md`](docs/articles/governed-mcp-and-agent-writes.md) — governed MCP architecture walkthrough article draft
- [`docs/articles/local-evidence-walkthrough.md`](docs/articles/local-evidence-walkthrough.md) — reproducible local evidence walkthrough
- [`docs/local-evidence-runbook.md`](docs/local-evidence-runbook.md) — repeatable local MCP, load, retrieval, workflow, cost, recovery, and security evidence
- [`evals/golden_set.json`](evals/golden_set.json) — 34-question golden eval set (v2.2); run with `scripts/run_golden_eval.py`

**Live demo (no services required):**
```bash
python scripts/demo_regulatory.py
```
Runs a 6-step aerospace regulatory workflow end-to-end — ontology loading, domain/range validation, transitive inference, contradiction detection — using in-process mocks.

**Live demo against real Neo4j** (requires Neo4j — `docker compose -f compose.dev.yaml up neo4j`):
```bash
python scripts/demo_regulatory.py --live
```
Ingests two genuinely conflicting documents, runs the real inference engine, and lets the contradiction detector find the IS_AIRWORTHY / IS_UNAIRWORTHY conflict. Data persists in Neo4j — query it in the browser at `http://localhost:7474`.

**Live service tests:**

The E2E suite uses `testcontainers-python` to start isolated Neo4j and Redis
containers. Docker Desktop must be running:

```bash
python -m pytest -q tests/e2e/test_live_services.py
```

This runs five real persistence and connectivity tests and removes the
temporary containers when the test classes finish. Use `python -m pytest`
rather than the Windows `pytest` executable so the repository root is on the
import path.

---

## Context Graph Layer

The platform now includes the P0 foundation for a tenant-scoped Context Graph.
It persists `CGCase`, `CGAgentRun`, `CGToolCall`, `CGObservation`, `CGEpisode`,
`CGContextManifest`, `CGDecision`, `CGOption`, `CGPolicyVersion`, and
`CGPolicyEvaluation` in Neo4j.

Available under `/context-graph`:

- trace creation and validation;
- immutable manifest persistence with deterministic SHA-256 integrity hashing;
- deterministic policy-rule evaluation with structured reason codes;
- durable, tenant-scoped session episodes linked to the run and manifest;
- live retrieval traces linked to a separate content-addressed answer cache;
- the WPP marketing campaign-placement governed decision flow.

The two hashes have different contracts. A manifest hash protects one
historical trace, including its timestamps. The answer-cache hash identifies
reusable inference inputs: tenant, normalized question, durable corpus
revision, requested/effective retrieval mode and configuration, model route,
prompt version, and ontology version. Every request still receives a new
`query_id`; a cache hit returns `source_query_id` and `source_trace_id` for the
original governed decision. Session queries bypass this cache because their
mutable conversation history is not yet part of the key.

The Context Graph stores structured evidence and rationale only; hidden
chain-of-thought is not persisted. P1 replay/governance, P2
outcomes/precedents, and P3 proactive intelligence contracts are implemented.
The trace, replay, retention, policy, and episode paths have been validated on
local live Neo4j; production traffic and production-scale tuning remain open.

---

## Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │                  FastAPI  :8000                      │
                        │  /ingest  /query  /kpis  /evaluation  /auth          │
                        │  /corrections  (split · quarantine · conflicts)      │
                        └──────────┬──────────────┬──────────────────────────┘
                                   │              │
                    ┌──────────────▼──┐      ┌────▼───────────────┐
                    │   RabbitMQ      │      │   OAuth 2.0 (JWT)  │
                    │  :5672  :15672  │      │  Google + M2M      │
                    └──┬───────────┬──┘      └────────────────────┘
                       │           │
          ┌────────────▼──┐  ┌─────▼──────────┐  ┌──────────────────┐
          │  Ingestion    │  │  Query Worker  │  │ Evaluation Worker│
          │  Worker       │  │                │  │  (RAGAS)         │
          └────────────┬──┘  └────┬───────────┘  └──────┬───────────┘
                       │          │                      │
                  ┌────▼──────────▼──────────────────────▼────┐
                  │                  Neo4j  :7687              │
                  │   Document → Chunk → Entity → Community   │
                  │   Vector index (3072d) + BM25 fulltext     │
                  │   RELATES_TO edges with confidence,        │
                  │   source_doc_ids, authority weights        │
                  └───────────────────────────────────────────┘
                                       │
                  ┌────────────────────▼───────────────────────┐
                  │           Redis  :6379                      │
                  │  Session context store (24h TTL)           │
                  │  Query result store (1h TTL, cross-worker) │
                  │  M2M client registry (persistent)          │
                  │  Alert history (last 100)                  │
                  └────────────────────────────────────────────┘
                                       │
                  ┌────────────────────▼───────────────────────┐
                  │     Optional TimescaleDB  :5432            │
                  │       KPI Events Store (SQLite default)    │
                  └────────────────────────────────────────────┘
                                       │
                  ┌────────────────────▼───────────────────────┐
                  │         Dash Dashboard  :8050               │
                  │   latency · faithfulness · recall · ...     │
                  └────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Details |
|---------|---------|
| **Batched ingestion writes** | Entity embeddings (A131) + chunk/entity/relation writes (A129 + A132) batched via UNWIND to minimize Neo4j round-trips — 30-doc corpus ingests in ~48 min wall-clock (90+ min → 48 min after A129-A132 optimization) |
| **Five-stage retrieval pipeline + synthesis** | Vector ANN → BM25+RRF → Cross-encoder → Multi-hop → GAT/GCN GNN → LLM synthesis; IRCoT is an iterative fallback |
| **Graph Attention Network (GAT)** | GCN/GAT re-scores chunks using entity embedding propagation; attention weights by cosine similarity between neighbours |
| **Query-adaptive GNN weights** | Relational queries (e.g. "how did X cause Y") auto-shift to 50/50 text/GNN; factoid queries use default α/β |
| **BM25 + Vector hybrid search** | Vector ANN and BM25 fulltext results fused via Reciprocal Rank Fusion (RRF, k=60) |
| **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` deep pairwise query-chunk scoring before graph expansion |
| **Multi-hop graph traversal** | `Chunk → Entity → RELATES_TO* → Entity → Chunk` up to depth 2 |
| **Agentic fallback (IRCoT)** | Low-confidence answers trigger bounded iterative re-search; Groq `llama-3.1-8b-instant` handles routing, while DeepSeek handles final synthesis. The historical agentic p95 benchmark was **3.4s**; current end-to-end latency is tracked in the roadmap. |
| **Session context** | Redis-backed conversation history (24h TTL); enriches follow-up queries with prior turn entities |
| **Adaptive retrieval routing** | Per-tenant/query-class EWMA quality and latency statistics choose local, hybrid, or global retrieval after a guarded cold-start; deterministic exploration prevents route starvation |
| **Alias resolution** | Name-based + embedding-based deduplication before every entity MERGE; per-tenant registry pool |
| **Document authority hierarchy** | 4-level authority system (Regulatory → Manufacturer → Internal → Informal); superseded docs penalised |
| **Contradiction detection** | Multi-source, directional-reversal, exclusive-state, and functional-violation conflict types; scoped per tenant |
| **Community detection** | Leiden algorithm (graspologic) builds hierarchical graph summaries for global search; staleness-gated auto-rebuild |
| **Temporal community summaries** | Immutable `CommunitySummarySnapshot` versions carry valid/transaction time, integrity hashes, chunk/document versions, and evidence links for point-in-time global retrieval |
| **Graph health metrics** | 6 semantic metrics (alias coverage, relation precision, contradiction rate, orphan growth, merge/split proxy, community coherence) with per-tenant trend snapshots |
| **Ontology enforcement** | Domain/range validation on every relation write; deprecated relation names auto-migrated on ingestion |
| **Tenant isolation** | All entities, edges, conflicts, communities, and health snapshots are scoped by `(name, type, tenant)` |
| **Graph integrity guards** | Self-loop removal, cycle detection, quarantine, ingestion validation, dirty-flag propagation after every write |
| **Manual correction API** | `/corrections` endpoints: entity split, quarantine/release, edge reject/override, conflict resolution |
| **Agent tool safety** | `ToolPolicy` gate: allowlist, per-tool risk levels (low/medium/high/restricted), scope enforcement, arg validation, cross-tenant guard, dry-run mode, timeout, structured audit log; 34 guardrail tests. Dispatched over HTTP at `POST /agent/tool`, so the gate applies to real calls rather than tests alone. |
| **Governed answer cache** | Redis-backed SHA-256 cache inside `HybridRetriever`; keys include tenant, normalized question, corpus revision, retrieval configuration, model route, prompt, and ontology versions. A hit skips retrieval/LLM work and points to the original Context Graph trace. Every retrieval-visible mutation, including ingestion, corrections, ontology migrations, graph re-ranking, communities, and GNN calibration, uses `KGCorpusState.active_updates` and publishes a new revision on completion. |
| **Redis alias registry** | `AliasRegistry.load()` pushes alias table to Redis hash (`graphrag:aliases:{tenant}`, 24h TTL); parallel workers warm from Redis without full Neo4j scan; `load_alias_registry()` is Redis-first |
| **Wikidata entity linking** | Optional post-ingestion step (`WIKIDATA_LINKING=1`); grounds high-confidence entities to canonical QIDs; rate-limited to 20 entities/document |
| **RAGAS evaluation** | Faithfulness, answer relevancy, context precision, context recall — auto-sampled at 20% |
| **OAuth 2.0** | Google browser login + M2M client credentials grant (JWT Bearer) |
| **Business Matrix** | Live Plotly Dash dashboard with KPI timeseries and alert thresholds |
| **Worker health probes** | `GET /ready` + `GET /live` on each worker (`WORKER_HEALTH_PORT`); aiohttp server in `graphrag/workers/health_server.py`; compose.dev.yaml and Kubernetes readiness probes use `/ready` |
| **Structured DLQ** | Failed messages carry `exception_type`, `error`, `retry_count`, `queue`, `message_id`, `payload_summary` — full JSON envelope for automated triage |
| **Async pipeline** | RabbitMQ decouples ingestion, query, and evaluation workers with structured DLQ; `compose.dev.yaml` starts the full stack in one command |
| **Context Graph P0** | Tenant-scoped cases, agent runs, tool observations, manifests, policy evaluations, options, and decisions under `/context-graph` |
| **Source catalog** | `/kg/sources` manages tenant-scoped source systems and versioned, secret-free mapping contracts; documents can link through `INGESTED_FROM` |
| **End-to-end observability** | `X-Correlation-ID` flows HTTP -> RabbitMQ -> worker -> result -> `CGAgentRun`; Prometheus metrics and optional OTLP traces share the same request context |
| **Multimodal provenance** | Media attachments plus OCR, transcript, caption, and visual-embedding transformation links; media bytes remain in object storage |

---

## Retrieval Pipeline — 6 Stages

```
Query
  │
  ├─ [0] Session context enrichment
  │      If session_id provided: inject prior-turn entities into query
  │
  ├─ [1] Vector ANN
  │      embed(query) → 3072d cosine search on chunk_embeddings index
  │
  ├─ [2] BM25 + RRF fusion
  │      BM25 fulltext search → Reciprocal Rank Fusion (k=60) with vector results
  │
  ├─ [3] Cross-encoder reranking
  │      ms-marco-MiniLM-L-6-v2 — deep pairwise (query, chunk) scoring → top rerank_k
  │
  ├─ [4] Multi-hop graph traversal (depth=2)
  │      Chunk → MENTIONS → Entity → RELATES_TO* → Entity → MENTIONS → Chunk
  │      Bridges facts distributed across separate documents
  │
  ├─ [5] GAT/GCN scoring
  │      Build node-feature matrix H from entity embeddings
  │      Build adjacency matrix A from RELATES_TO edges (authority-weighted confidence)
  │      Propagate: final = α·sigmoid(rerank_score/5) + β·gnn_score
  │      Query-adaptive: relational queries → α=β=0.5; factoid → α=0.9, β=0.1
  │
  ├─ [6] Entity context + global community summaries
  │
  ├─► ContextBuilder (local 60% + global 40%)
  │
  ├─► DeepSeek (`get_llm()` default) generates grounded answer with chunk citations
  │
  └─► Low confidence?
        └─► AgenticRetriever (IRCoT loop, max 4 steps)
              ├─ SEARCH: <sub-query> → re-retrieve → expand context
              └─ ANSWER: <final> or "insufficient context"
```

**Why GNN on top of a reranker?**

The cross-encoder scores text similarity. It doesn't know that *Falcon 9* and *SpaceX* are structurally linked in the graph. A GAT propagates entity embeddings along RELATES_TO edges — semantically related neighbours vote on each entity's relevance. Chunks that mention graph-connected entities score higher even when their text has a weak direct match to the query.

---

## Stack

| Component | Technology |
|-----------|-----------|
| Graph DB | Neo4j 5.20 compatibility baseline; validated Neo4j 2026.06 path with tenant-filtered vector `SEARCH` via `compose.neo4j-modern.yaml` |
| Session Store | Redis 7 |
| Message Queue | RabbitMQ 3.13 |
| KPI Store | SQLite by default; optional TimescaleDB via `KPI_BACKEND=timescale` |
| Embeddings | `text-embedding-3-large` (3072d) via OpenAI |
| LLM | DeepSeek `deepseek-v4-pro` (primary generation, via `get_llm()`) + Groq `llama-3.1-8b-instant` (fast routing, via `get_fast_llm()`; also available as opt-in dev override for generation) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers) |
| GNN | PyTorch — GAT / GCN (configurable) |
| Community detection | graspologic (Leiden algorithm) |
| Agent Framework | Custom dual-LLM IRCoT (Groq 8B routing + DeepSeek large-model synthesis) |
| Evaluation | RAGAS |
| API | FastAPI + Uvicorn |
| Dashboard | Plotly Dash |
| Auth | OAuth 2.0 · python-jose JWT (HS256) |
| Runtime | Python 3.11 |

---

## Project Structure

```
ai-knowledge-graph-platform/
├── api/
│   ├── main.py                      # FastAPI app, lifespan hook, middleware, routes
│   ├── limiter.py                   # slowapi rate-limiter (20/min ingest, 60/min query, 10/min auth)
│   ├── auth/
│   │   ├── dependencies.py          # get_current_user, get_tenant (token-scoped), require_scope
│   │   ├── google.py                # Google OAuth 2.0 Authorization Code flow
│   │   └── jwt.py                   # HS256 JWT creation & validation
│   └── routes/
│       ├── auth.py                  # /auth/login, /callback, /token, /clients (Redis-backed M2M)
│       ├── ingest.py                # POST /ingest  (rate-limited)
│       ├── query.py                 # POST /query, GET /query/{id}  (rate-limited; Redis result store)
│       ├── evaluation.py            # GET /evaluation/summary  (require_scope("read"))
│       ├── kpis.py                  # GET /kpis/summary, /kpis/timeseries
│       ├── agent.py                 # POST /agent/tool — ToolPolicy-gated tool dispatch
│       ├── corrections.py           # entity split · quarantine · edge override · conflict resolve
│       ├── context_graph.py         # Bounded Context Graph decision traces
│       ├── demo.py                  # Public demo pages
│       └── kg/                      # 11 routers: knowledge, health, compliance, calibration,
│                                    #   community, confidence, embeddings, feedback, inference,
│                                    #   pagerank, review_queue, sources
│
├── graphrag/
│   ├── agents/
│   │   ├── base_agent.py            # Abstract agent base
│   │   ├── ingestion_agent.py       # Document → chunk → embed → extract → graph
│   │   ├── query_agent.py           # Question → retrieve → answer
│   │   └── evaluation_agent.py      # RAGAS scoring agent
│   ├── business_matrix/
│   │   ├── dashboard_server.py      # Plotly Dash on :8050/dashboard/
│   │   ├── kpi_store.py             # SQLAlchemy KPI event model (recorded_at indexed)
│   │   └── kpi_tracker.py           # KPI aggregation queries; real p50/p95 percentile (capped at 10k rows)
│   ├── core/
│   │   ├── config.py                # Settings (pydantic-settings, .env + YAML); production validators
│   │   ├── llm_client.py            # Central LLM router: DeepSeek primary for generation (Groq opt-in dev override; Groq 8B primary for fast routing), OpenAI for embeddings
│   │   ├── llm_utils.py             # safe_response_text() — guards legacy Gemini response.text accesses (embedding path)
│   │   ├── models.py                # Domain models: Document, Chunk, Entity, Relation, Community, SessionTurn ...
│   │   └── retry.py                 # Async exponential-backoff decorator for Neo4j transient errors
│   ├── graph/
│   │   ├── neo4j_client.py          # Async Neo4j driver, MERGE helpers, vector + BM25 search
│   │   ├── schema.cypher            # Constraints, vector indexes, fulltext indexes
│   │   ├── alias_registry.py        # Per-tenant alias pool: name-based + embedding deduplication
│   │   ├── audit_trail.py           # Immutable AuditEvent nodes for every entity/relation change
│   │   ├── community_builder.py     # Leiden community detection (graspologic); fallback to connected-components
│   │   ├── community_manager.py     # Staleness scoring (entity/edge drift); snapshot & rebuild gating
│   │   ├── community_summarizer.py  # LLM-generated community summaries
│   │   ├── contradiction_detector.py # Multi-source, directional, exclusive-state, functional conflicts
│   │   ├── cycle_detector.py        # Detect cycles in RELATES_TO graph post-ingestion
│   │   ├── document_authority.py    # Authority levels, SUPERSEDES chains, edge confidence penalties
│   │   ├── entity_splitter.py       # Detect over-merged entities; split into canonical + variant nodes
│   │   ├── gnn_scorer.py            # GAT/GCN graph-propagated re-scoring (PyTorch)
│   │   ├── graph_evaluator.py       # 6 semantic health metrics; per-tenant GraphHealthSnapshot nodes
│   │   ├── ingestion_validator.py   # Post-write graph health check; degree anomaly detection
│   │   ├── ontology_registry.py     # Domain/range rules; deprecated relation migration; schema events
│   │   └── quarantine.py            # Quarantine/release entities; auto-quarantine anomalies
│   ├── ingestion/
│   │   ├── chunker.py               # Sliding-window text chunking
│   │   ├── embedder.py              # OpenAI text-embedding-3-large (3072d) batches
│   │   ├── extractor.py             # LLM entity + relation extraction
│   │   └── graph_writer.py          # Persist chunks/entities/relations; alias resolution; validation
│   ├── messaging/
│   │   ├── rabbitmq_client.py       # aio-pika connection, publish, consume, DLQ
│   │   ├── publishers.py            # publish_document(), publish_query(), publish_eval_job()
│   │   └── consumers.py             # Message handler wiring per queue
│   └── retrieval/
│       ├── local_search.py          # Five retrieval stages: vector + BM25 + rerank + multihop + GNN; synthesis follows
│       ├── global_search.py         # Community embedding search + direct-context synthesis
│       ├── hybrid_retriever.py      # Combines local + global; agentic fallback; session turn recording
│       ├── agentic_retriever.py     # Iterative IRCoT re-search (Groq 8B routing + DeepSeek synthesis)
│       ├── bm25_search.py           # HybridBM25Search with RRF (k=60)
│       ├── reranker.py              # CrossEncoderReranker (ms-marco-MiniLM-L-6-v2)
│       ├── session_context.py       # Async session context: query enrichment from prior turns
│       ├── session_store.py         # Redis-backed turn store; in-memory fallback; strict startup mode
│       └── result_store.py          # Redis-backed query result store (cross-worker, 1h TTL)
│
├── workers/
│   ├── ingestion_worker.py          # Consumes graphrag.ingest queue; graceful SIGTERM shutdown
│   ├── query_worker.py              # Consumes graphrag.query queue; graceful SIGTERM shutdown
│   ├── evaluation_worker.py         # Consumes graphrag.eval queue; graceful SIGTERM shutdown
│   └── combined_worker.py           # Runs ingestion + query consumers on one machine (co-location mode)
│
├── scripts/                         # 73 CLI tools — ingestion, evaluation, benchmarks, migrations
│   ├── ingest_corpus.py             # Full real-pipeline corpus ingestion
│   ├── init_neo4j.py                # Idempotent schema initializer (run once after docker up)
│   ├── run_golden_eval.py           # Golden-set faithfulness evaluation
│   └── community_rebuild.py         # CLI: rebuild communities per tenant with staleness check
│
├── tests/                           # 812 tests across four tiers
│   ├── unit/                        # 71 files — in-process, fully mocked I/O
│   ├── integration/                 # AsyncMock-based; no live services (see note below)
│   ├── load/                        # Concurrency shape, AsyncMock-backed
│   └── e2e/                         # The only tier with a live Neo4j + Redis (testcontainers)
│
├── evals/                           # Golden sets + committed benchmark result files
├── mcp_server/                      # MCP tool server (hybrid retrieval, entity lookup)
├── deploy/ · infra/ · fly/          # Kubernetes manifests, Terraform, Fly.io config
│
├── config/
│   ├── settings.yml                 # All pipeline tuning (see Configuration section)
│   └── ontologies/
│       └── aerospace_regulatory.yml # Domain ontology: types, relations, inference rules, authority levels
│
├── docs/
│   ├── ontology-model.md            # Formal type hierarchy, relation schema, inference rules, design decisions
│   ├── entity-resolution.md         # 4-stage entity resolution pipeline with examples
│   └── knowledge-graph-architecture.md  # Architectural decisions, data model, scalability
│
├── docker-compose.yml
├── Dockerfile                       # Multi-stage build; non-root user; HEALTHCHECK
├── requirements.txt                 # Direct dependencies — source of truth
├── requirements-dev.txt             # pytest, pytest-asyncio, ruff, pip-tools
├── requirements.lock                # Fully-pinned lock file (regenerate: make lock)
├── requirements/                    # Per-image subsets for Docker (see requirements/README.md)
├── LICENSE                          # MIT
└── .env                             # Secrets (never commit)
```

---

## Verified System Check Results

Full end-to-end test completed 2026-03-21 (updated 2026-05-31 with Groq integration):

| Step | Component | Result |
|------|-----------|--------|
| Infrastructure | Neo4j + RabbitMQ + Redis; optional TimescaleDB | ✅ Core stack healthy; TimescaleDB deployment-specific |
| API | FastAPI + OAuth + lifespan hook | ✅ Running on :8000 |
| Ingestion | doc → chunk → embed (OpenAI text-embedding-3-large 3072d) → extract (DeepSeek default; Groq opt-in dev override) → graph | ✅ |
| Schema | Vector indexes + BM25 fulltext indexes (7 total — 4 vector, 3 fulltext — all ONLINE) | ✅ |
| Graph counts | 1 doc · 1 chunk · 5 entities · 4 relations | ✅ |
| Hybrid search | BM25=10 + vector=10 → fused=10 chunks | ✅ |
| Cross-encoder reranker | ms-marco-MiniLM-L-6-v2, top_score=9.30 | ✅ |
| GNN scoring | GAT 2-layer; α=0.9 text + β=0.1 graph | ✅ |
| Answer synthesis | DeepSeek (`get_llm()` default); citations included | ✅ |
| Session context | Redis-backed; turn recorded after answer | ✅ |
| RAGAS | 20% sampling; metrics stored in the configured KPI backend | ✅ |
| Dashboard | Live KPI charts at /dashboard/ | ✅ |

---

## Quick Start

### 1. Prerequisites

- Python 3.11
- Docker Desktop
- Google OAuth credentials → https://console.cloud.google.com/apis/credentials

### 2. Clone & install

```bash
git clone <repo>
cd ai-knowledge-graph-platform
python -m pip install -r requirements.txt
```

### 3. Configure `.env`

```env
# OpenAI — embeddings only (text-embedding-3-large, 3072d)
# Get key at: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-...

# Groq — fast routing model for agentic retrieval (get_fast_llm()); also usable
# as an opt-in text-generation override via LLM_INGEST_PROVIDER=groq
# Get key at: https://console.groq.com/keys
GROQ_API_KEY=gsk_...
# Two Groq models by design (ADR 0006): the 70B synthesises, the 8B routes.
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FAST_MODEL=llama-3.1-8b-instant

# DeepSeek — primary text generation (get_llm() default)
DEEPSEEK_API_KEY=sk-...

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphrag_dev

# RabbitMQ
RABBITMQ_URL=amqp://graphrag:graphrag_dev@localhost:5672/

# Redis (session context + cross-process query results)
REDIS_URL=redis://localhost:6379/0

# Optional TimescaleDB KPI backend (SQLite is the default)
KPI_BACKEND=timescale
TIMESCALE_DB_URL=postgresql+asyncpg://graphrag:graphrag_dev@localhost:5432/graphrag_kpis

# OAuth 2.0
JWT_SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
SESSION_SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
GOOGLE_OAUTH_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret
CORS_ORIGINS=["http://localhost:8000","http://localhost:8050"]

# App
LOG_LEVEL=INFO
ENV=development
```

### 4. Start infrastructure

```bash
docker compose -f compose.dev.yaml up   # full stack: Neo4j + RabbitMQ + Redis + API + workers + dashboards
```

**Pick one compose file — the two full stacks collide.** `compose.dev.yaml` and
`docker-compose.yml` both bind 7474, 7687, 5672, 15672, 6379, 8000 and 8050, so
running them together fails on port binding.

| File | Use it for |
|------|------------|
| `compose.dev.yaml` | **Default for local development.** One command, full stack, `dev_*` containers. This is what the rest of this README assumes. |
| `docker-compose.yml` | Production-shaped definition — pinned project name, named volumes holding the ingested tenant data, `graphrag_*` containers. `make services-up` uses this one. |
| `compose.neo4j-modern.yaml` | Not a stack — a small override layered on `docker-compose.yml` (below). |

For a fresh Neo4j 2026.06 deployment with in-index tenant filtering, use the
separate-volume override. It does not mount the existing 5.20 data volume:

```bash
docker compose -f docker-compose.yml -f compose.neo4j-modern.yaml up -d neo4j
python scripts/init_neo4j.py
python scripts/migrate_neo4j_vector_indexes.py  # dry-run on an upgraded existing database
python scripts/migrate_neo4j_vector_indexes.py --apply
```

The migration command rebuilds indexes only. Back up an existing database
before a server upgrade; do not attach a 5.20 volume directly to 2026.06.

### 5. Initialize Neo4j schema

```bash
python scripts/init_neo4j.py
```

Run once after Neo4j first starts. Creates vector indexes, fulltext indexes, constraints, and relation indexes (all idempotent).

### 6. Start workers and API

**Option A — Docker (recommended, one command):**
```bash
docker compose -f compose.dev.yaml up
```
All services start in dependency order. Workers expose `GET /ready` health probes on ports 8081–8083.

**Option B — Local Python (four terminals):**
```bash
# Terminal 1 — API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Ingestion worker  (health probe: http://localhost:8081/ready)
python workers/ingestion_worker.py

# Terminal 3 — Query worker      (health probe: http://localhost:8082/ready)
python workers/query_worker.py

# Terminal 4 — Evaluation worker (health probe: http://localhost:8083/ready)
python workers/evaluation_worker.py

# Terminal 5 — Dashboard
python graphrag/business_matrix/dashboard_server.py
```

**Single-machine shortcut:**
```bash
python workers/combined_worker.py
```

### 7. Ingest real corpus data

Ingest the 12-document aerospace regulatory corpus through the full LLM extraction
pipeline (DeepSeek default extraction → OpenAI embeddings → alias resolution → contradiction detection):

```bash
python scripts/ingest_corpus.py --commit --wipe
# Single document:
python scripts/ingest_corpus.py --commit --doc FAA-AD-2024-01-02.txt
```

**Performance:** Batched entity embeddings (A131) + batched chunk writes (A132) 
reduce write-phase round-trips by 80-90%. A 30-document automotive corpus 
(3013 entities, 9364 relations) ingests in ~48 minutes wall-clock after full 
optimization (A129 + A131 + A132) — previously 90+ minutes with unbatched 
per-entity / per-chunk writes stacked against the serialized writer phase.

This produces real graph health metrics (⚠ LLM extraction is non-deterministic —
these numbers shift on every fresh `--wipe --commit` run; verify live before
presenting, see `tasks/lessons.md` A96/A98):
**368 entities · 422 edges · 4 open conflicts · 9.48/1k contradiction rate · 58 Leiden communities** (verified live 2026-06-10, after fixing a missing document-supersession chain and rebuilding communities with supersession-aware summaries — see A99/A100)

#### Other tenants / domain corpora

`--tenant` selects the corpus, authority map, and supersession chains (see
`scripts/ingest_corpus.py` → `_corpus_config()`); each tenant's data lives in
the same Neo4j instance, partitioned by the `tenant` property:

```bash
# Automotive IATF 16949 corpus (data/automotive/, 30 docs, ontology: config/ontologies/automotive_iatf.yml)
python scripts/ingest_corpus.py --tenant automotive --commit --wipe

# Golden eval set for that tenant (data/eval_golden/queries_automotive.json)
python scripts/run_golden_eval.py --tenant automotive
```

**After every `--commit` ingestion (and especially before a `--wipe` of one
tenant), run the tenant isolation check** — it confirms no node is missing a
`tenant` property and no edge (`RELATES_TO`/`PART_OF`/`MENTIONS`/`MEMBER_OF`)
crosses between two tenants' graphs:

```bash
python scripts/verify_tenant_isolation.py
```

For a lightweight demo with hardcoded seed data (no LLM calls):

```bash
python scripts/seed_demo_data.py --commit --tenant aerospace
```

---

## Usage

### Dev token (CLI / testing)

```powershell
$resp = Invoke-RestMethod -Uri "http://localhost:8000/auth/dev-token" -Method POST
$token = $resp.access_token
$h = @{"Authorization"="Bearer $token"; "Content-Type"="application/json"}
```

### Ingest a document

```powershell
Invoke-RestMethod -Uri http://localhost:8000/ingest -Method POST -Headers $h `
  -Body '{"filename":"report.txt","text":"Company A owns Company B. Company B launched a rocket."}'
```

### Query

```powershell
$q = Invoke-RestMethod -Uri http://localhost:8000/query -Method POST -Headers $h `
  -Body '{"question":"What did Company A launch?","mode":"hybrid","session_id":"user-123"}'

# Poll for result
Invoke-RestMethod -Uri "http://localhost:8000/query/$($q.query_id)" -Method GET -Headers $h
```

### Manual corrections

```powershell
# Split an over-merged entity
Invoke-RestMethod -Uri http://localhost:8000/corrections/entity/split -Method POST -Headers $h `
  -Body '{"entity_name":"Apple","entity_type":"ORG","tenant":"default"}'

# List open contradiction conflicts
Invoke-RestMethod -Uri http://localhost:8000/corrections/conflicts -Method GET -Headers $h

# Resolve a conflict
Invoke-RestMethod -Uri http://localhost:8000/corrections/conflict/resolve -Method POST -Headers $h `
  -Body '{"conflict_id":"...","resolution":"manual_override"}'
```

### Rebuild communities (CLI)

```bash
# Check staleness, rebuild if needed
python scripts/community_rebuild.py --tenant default

# Force rebuild regardless of staleness
python scripts/community_rebuild.py --tenant default --force

# Dry-run: check without rebuilding
python scripts/community_rebuild.py --tenant default --dry-run
```

---

## Configuration

All tuning is in `config/settings.yml`:

```yaml
ingestion:
  chunk_size: 512
  chunk_overlap: 64
  embedding_batch_size: 100        # also controls chunk/entity write batch size (A131-A132)
  doc_extract_concurrency: 4       # concurrent document extractions (LLM-bound)
  extraction_concurrency: 5        # chunks per document extracted concurrently
  entity_types: [PERSON, ORG, PRODUCT, CONCEPT, LOCATION, EVENT]
  alias_embedding_threshold: 0.92  # cosine similarity to treat as duplicate entity
  alias_fuzzy_threshold: 85        # rapidfuzz ratio for soft name matching
  validate_after_ingestion: true   # run graph health check after every doc
  auto_remove_self_loops: true
  detect_cycles_after_ingestion: true

graph:
  community_levels: 3
  leiden_resolution: 1.0
  min_community_size: 3
  require_leiden: true              # fail hard if graspologic is missing
  auto_rebuild_communities: true    # rebuild when staleness exceeds threshold
  community_staleness_threshold: 0.15
  community_staleness_check_on_ingest: true
  dirty_flag_propagation: true
  default_authority_level: 4        # INFORMAL
  superseded_confidence_penalty: 0.5

retrieval:
  local_top_k: 10
  multihop_depth: 2
  rerank_top_k: 5
  bm25_enabled: true
  reranker_enabled: true
  gnn_enabled: true
  gnn_type: gat                     # "gcn" | "gat"
  gnn_layers: 2
  gnn_alpha: 0.9                    # weight for text score
  gnn_beta: 0.1                     # weight for GNN structural score
  gnn_adaptive_weights: true        # relational queries → 0.5/0.5
  authority_weighting_enabled: true
  session_context_enabled: true
  session_store: redis              # "memory" | "redis"
  session_store_strict: true        # fail startup if Redis unreachable
  redis_url: redis://localhost:6379/0
  session_ttl_seconds: 86400
  agentic_fallback: true
  agentic_max_steps: 4

ontology:
  enforce_domain_range: true
  allow_migration_renames: true
  migration_map:
    IS_CEO: CEO_OF
    FOUNDED_BY_PERSON: FOUNDED_BY

maintenance:
  stale_edge_days: 365
  low_conf_prune_threshold: 0.2
  orphan_flag_enabled: true
  cycle_check_enabled: true
```

---

## End-to-End Flow: User Asks a Question

> **Example:** *"What rockets did Elon Musk's company launch and what did they achieve?"*

This question spans 3 separate documents with no direct text overlap.

```
1. USER AUTHENTICATES
   POST /auth/dev-token  →  JWT (HS256, 60 min)

2. USER SUBMITS QUESTION
   POST /query  { "question": "...", "session_id": "user-42" }
   →  FastAPI validates JWT scope("read")
   →  Publishes to RabbitMQ: graphrag.query
   →  Returns: { query_id: "abc-123", status: "queued" }

3. QUERY WORKER
   →  QueryAgent.run(query_id, question, session_id)
   →  This example has session context, so the governed answer cache is bypassed.
      A stateless cache hit keeps the new query_id and returns source_query_id
      plus source_trace_id without rerunning retrieval or the LLM.

4. SESSION CONTEXT ENRICHMENT
   Prior turn: "Who owns SpaceX?" → answer mentioned "Elon Musk", "SpaceX"
   →  enriched_question = "What rockets did Elon Musk's company [SpaceX] launch?"

5. LOCAL SEARCH — 6 stages
   ├─ Vector ANN: embed(enriched_question) → top-10 chunks by cosine
   ├─ BM25: fulltext search → RRF fusion with vector results
   ├─ Cross-encoder: ms-marco-MiniLM-L-6-v2 → rerank top-5
   │     chunk A (achievements): rerank_score=2.97  ← best text match
   │     chunk B (products):     rerank_score=1.83
   │     chunk C (ownership):    rerank_score=0.42
   ├─ Multi-hop: chunk A mentions SpaceX → RELATES_TO → Falcon 9 → achievements.txt
   │     Cross-document bridge resolved:
   │       ownership.txt    →  "Elon Musk owns SpaceX"
   │       products.txt     →  "SpaceX manufactures Falcon 9, Starship"
   │       achievements.txt →  "Falcon 9 landed 2015, Starship flew 2023"
   └─ GAT GNN scoring:
         Falcon 9 chunk: cross-encoder score=-6.74 (weak text match — "Elon" absent)
         GAT score=0.73  (graph knows SpaceX → Falcon 9 → Starship are linked)
         final = 0.9 × sigmoid(-6.74/5) + 0.1 × 0.73 = 0.18 + 0.07 = 0.25
         Chunk stays in results. Without GNN it would drop out.

6. GLOBAL SEARCH
   embed(question) → community_embeddings ANN → cluster summaries
   →  Adds high-level SpaceX/Tesla/Musk community context

7. CONFIDENCE CHECK
   Citations found + specific answer → skip agentic fallback ✅
   (else: IRCoT loop, max 4 SEARCH steps, then "insufficient context")

8. DEEPSEEK GENERATES ANSWER
   Context: local chunks (60%) + community summaries (40%)
   "SpaceX, founded by Elon Musk, launched:
    • Falcon 9 — first booster landing 2015, 200+ missions [Chunk 8910]
    • Starship — first successful flight 2023, NASA Artemis HLS [Chunk 8910]"

9. SESSION TURN RECORDED (after answer is known)
   session_ctx.record_turn(session_id, question, answer, referenced_entities)
   →  Stored in Redis with 24h TTL

10. USER POLLS
    GET /query/abc-123
    → { status: "completed", answer: "...", citations: [...], latency_ms: 3860 }

11. RAGAS EVALUATION (20% sampled)
    faithfulness=1.0 · context_precision=1.0 · context_recall=1.0
    → Scores stored in TimescaleDB → live in dashboard
```

---

## Graph Integrity & Production Hardening

Every ingestion batch runs the following checks automatically:

| Guard | What it does |
|-------|-------------|
| **Alias resolution** | Name-based + embedding deduplication before MERGE; prevents duplicate entity nodes |
| **Ontology validation** | Domain/range rules checked on every relation; violations logged as schema events |
| **Self-loop removal** | `(e)-[r:RELATES_TO]->(e)` edges deleted automatically |
| **Cycle detection** | RELATES_TO cycles flagged after each write |
| **Contradiction detection** | Multi-source, directional, exclusive-state, and functional conflicts detected and stored as `Conflict` nodes |
| **Quarantine** | Entities flagged as degree anomalies auto-quarantined; excluded from retrieval until released |
| **Community staleness** | Entity/edge drift tracked; communities auto-rebuilt when drift exceeds threshold |
| **Graph health snapshot** | 6 metrics persisted as `GraphHealthSnapshot` nodes for trend monitoring |

**Tenant isolation** is enforced at the data layer: every entity, edge, conflict, community, and health snapshot is keyed on `(name, type, tenant)`. Cross-tenant queries never mix results.

**Strict mode** for critical dependencies:
- `require_leiden: true` → startup fails hard if `graspologic` is missing (silent fallback degrades global search quality undetectably)
- `session_store_strict: true` → FastAPI lifespan hook pings Redis at startup; fails hard if unreachable

---

## Measured Performance (live, aerospace regulatory corpus)

### Answer Quality — RAGAS (DeepSeek synthesis; DeepSeek → Groq → Gemini compatibility judge fallback)

| Metric | Measured | Target |
|--------|----------|--------|
| `faithfulness` | **0.919** — measured against golden set **v2.2 (34 questions)**, 29 scored / 0 refusals / 5 unscorable¹ | ≥ 0.85 ✓ |

Measured 2026-08-14 with a fresh aerospace ingestion (459 entities, 640 edges) via
`python scripts/run_faithfulness_eval.py`. This replaces an earlier 0.940 figure that was
measured against golden set v2.1 (39 questions, including two contradiction questions since
retired as invalid — see `evals/golden_set.json`'s changelog) — that number is no longer
current and should not be cited.

By question type (n scored): single_hop 0.875 (8) · multi_hop 0.917 (4) · temporal 0.938 (4) ·
inference 0.944 (3) · authority_chain 0.950 (2) · negative 0.833 (2) · precision 1.000 (2) ·
calibration 0.900 (1) · agentic 0.929 (1) · contextual 1.000 (2).

¹ *Two exclusion categories, both correctly excluded from the scored denominator rather than
counted as failures. **Refusals**: the corpus genuinely lacks the answer (score 0 in RAGAS by
construction) — a system that declines rather than invents is the desired behaviour; none
occurred in this run. **Unscorable**: RAGAS's own claim-decomposition step found no verifiable
statements to check, which happens on short/terse or yes-no answers — this is "metric not
applicable," not a faithfulness violation. Averaging either category in would either
underpenalize or misrepresent the model; `scripts/run_faithfulness_eval.py` filters both out of
the aggregate explicitly. `answer_relevancy`/`context_precision`/`context_recall` below are from
an earlier 10-question subset and are not re-measured here — `config/settings.yml` only enables
the `faithfulness` RAGAS metric to conserve quota, so those three are not computed by this
script at all currently.*

| Metric (10-question subset, not re-verified on full set) | Measured |
|--------|----------|
| `answer_relevancy` | 0.816 |
| `context_precision` | 0.907 |
| `context_recall` | 0.867 |

### Latency — reported per retrieval mode (A73: never combine)

Measured across 44 automotive and aerospace queries — `HybridRetriever`, live Neo4j,
live LLM, no shortcuts (`docs/roadmap.md`, "Current performance baseline"):

| Metric | Measured |
|--------|----------|
| p50 | **13.2 s** |
| p95 | **26.4 s** |
| mean | **15.2 s** |

This is slow, and honestly so. The cost is round-trip count, not a single bottleneck:
query rewrite → embed → local retrieval → optional map/reduce → final synthesis, most of
it still sequential. Three root-caused fixes cut p95 from 45.9 s to 26.4 s in one session
(`docs/performance-metrics-inventory.md`). CPU reranking alone accounts for a measured
+2,072 ms mean / +2,424 ms p95 — this deployment has no GPU (`docs/roadmap.md`, SPLADE
section). The `latency_p95_ms` alert threshold is set to 30,000 ms to match.

> An earlier version of this table claimed **2,162 ms p95** for hybrid retrieval. That
> figure is stale — `docs/performance-metrics-inventory.md` carries an explicit
> *"Do not cite 2.2s p95"* note — and has been replaced with the measured numbers above.

### Graph Health (12-doc aerospace corpus · real LLM-extracted data · 2026-06-10)

Real corpus ingested via `scripts/ingest_corpus.py` — full extraction pipeline.

⚠ **These numbers are a snapshot of one ingestion run, not a stable baseline.**
LLM extraction is non-deterministic at temperature=0 (batched GPU/LPU inference
— see `tasks/lessons.md` A96/A98): a fresh `--wipe --commit` of the *same* 12-doc
corpus produced 364/380/11, then 368/422/7, then 368/422/4 across three runs in
the same week. **Always re-run the live queries below immediately before
presenting — never quote these from memory of a prior run.**

| Metric | Real value (verified live, 2026-06-10) | Production target | Threshold |
|--------|------------|-------------------|-----------|
| **Entities** | **368** (alias-deduplicated; raw extraction count is run-dependent — don't hardcode it) | ~2,000+ | — |
| **Relations** | **422** (412 `source_type='document'` + 10 `source_type='inferred'`) | ~7,000+ | — |
| **Open conflicts** | **4** detected | — | — |
| Contradiction density | **9.48 /1k edges** | < 0.85 /1k | < 2.0 |
| Community coherence | 58 Leiden communities (rebuilt 2026-06-10 with supersession-aware summaries; coherence % not re-measured this run) | > 0.65 | > 0.50 |
| Brier score (calibration) | **0.809** (48 cumulative live samples, verdict "under-confident") — *no results file is committed for this figure; regenerate with `GET /kg/calibration/summary` before quoting* | < 0.20 | < 0.25 ✗ |

Evaluation is sampled at **20%** of queries automatically. View results:

- `GET /evaluation/summary`
- `GET /kpis/summary`
- **http://localhost:8050/dashboard/**

---

## Testing — what the suite does and does not cover

```bash
pip install -r requirements-dev.txt
pytest tests/unit/        # 794 tests, ~90s, no services required
pytest tests/e2e/         # live Neo4j + Redis via testcontainers (needs Docker)
make smoke-test           # unit tests + mock demo + API import check
```

**812 tests across four tiers, ~56% line coverage of `graphrag/`.** Being precise about
what that means, because the tier names oversell it:

| Tier | Count | What it actually exercises |
|------|-------|----------------------------|
| `tests/unit/` | 794 | In-process logic with all I/O mocked. Strong where the logic is deterministic: `core/models`, `core/retry`, `core/provider_health`, `graph/owl_reasoner`, `graph/sparql_bridge`, `graph/review_queue`, `graph/corpus_revision` are at 100%; `graph/inference_engine` 96%, `retrieval/context_builder` 89%. |
| `tests/integration/` | 34 | **Also AsyncMock** — the file docstrings say so outright. These are unit tests in a different folder, not integration tests. |
| `tests/load/` | 5 | Concurrency *shape* against AsyncMock, not throughput. Not a performance benchmark. |
| `tests/e2e/` | 5 | The only tier that runs real Cypher against a real Neo4j and a real Redis. Now runs in CI, which asserts it did not silently skip. |

Known weak spots, stated rather than hidden: the I/O boundary is thin. `ingestion/chunker`
(19%), `retrieval/bm25_search` (24%), `messaging/rabbitmq_client` (21%),
`graph/community_builder` (12%) and `retrieval/agentic_retriever` (31%) are the parts a
reviewer should assume are least protected. Most Cypher is asserted as *strings* —
substring checks that catch an accidental edit to the query text but not a query that is
syntactically valid and semantically wrong. `tests/unit/test_tenant_isolation.py` closes
that gap for the tenant-scoping invariants specifically, which is where it had already
cost us real bugs.

---

## Dashboards

Two operator dashboards share one **branded design system** (deep-navy / teal, Inter
typography, status-coloured KPI tiles, radial gauges, branded Plotly charts) — built to
look credible on a projector in front of a technical audience.

### Admin / Observability — `/admin`

Mounted directly on FastAPI (no separate process). Always serve it via the API — the
standalone Flask server 404s on Dash static assets.

| Tab | What it shows |
|-----|---------------|
| **Graph Health** | KPI tiles + **4 radial gauges** (entity resolution, relation confidence, community coherence, orphan rate) + branded contradiction-rate trend + recent alerts. |
| **Conflicts** | Themed table of open Conflict nodes. Select a row + resolution type to call `POST /corrections/conflict/resolve`. |
| **Communities** | Change-fraction + changed-entities tiles, "Rebuild Affected Communities" action, version-history table. |
| **GDPR & PII** | Erasure audit log + "Forget Entity · GDPR Article 17" form (`POST /kg/gdpr/forget-entity`). |
| **Calibration** | Brier-score rating tile + trend + isotonic calibration curve. |

```bash
export GRAPHRAG_ADMIN_TOKEN="your-secret-token"   # empty = open (dev only)
uvicorn api.main:app                              # → http://localhost:8000/admin/
```

### Business Matrix — `/dashboard/`

Query-level KPIs from the configured store (SQLite by default; optional TimescaleDB):
status-coloured tiles (queries, avg/p95 latency, faithfulness, context recall) plus
a branded metric trend with alert threshold.

```bash
python graphrag/business_matrix/dashboard_server.py   # → http://localhost:8050/dashboard/
```

### Demo mode (no backend)

To show either dashboard **fully populated** for a walkthrough or screenshots without a
running Neo4j / ingestion pipeline, set `GRAPHRAG_DASHBOARD_DEMO=1`. Each admin tab then
falls back to representative sample data (`graphrag/dashboard/demo_data.py`) **only if** the
live API is unreachable. Unset in production — real data or a real error panel is always
shown otherwise.

```bash
GRAPHRAG_DASHBOARD_DEMO=1 uvicorn api.main:app --port 8001   # → http://localhost:8001/admin/
```

---

## Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| API + Swagger | http://localhost:8000/docs | Bearer token via /auth/dev-token |
| Admin Dashboard | http://localhost:8000/admin | `GRAPHRAG_ADMIN_TOKEN` (empty = open) |
| Business Matrix Dashboard | http://localhost:8050/dashboard/ | — |
| Prometheus metrics | http://localhost:8000/metrics | — |
| Neo4j Browser | http://localhost:7474 | neo4j / graphrag_dev |
| RabbitMQ UI | http://localhost:15672 | graphrag / graphrag_dev |

---

## Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `No such vector schema index: chunk_embeddings` | Schema not initialized | Run `python scripts/init_neo4j.py` |
| `startup.session_store_unavailable` | Redis unreachable at startup | Start Redis or set `session_store_strict: false` in settings.yml |
| `graspologic is not installed` | Leiden community detection unavailable | `pip install graspologic` or set `require_leiden: false` (degrades global search) |
| `No module named 'groq'` | Groq package not installed | `pip install groq` |
| `No module named 'redis'` | redis[asyncio] not installed | `pip install "redis[asyncio]"` |
| `ImportError: redis[asyncio] is not installed but session_store=redis` | Redis package missing with strict=true | Install redis or set `session_store_strict: false` in settings.yml |
| `NotImplementedError: add_signal_handler` (Windows) | Signal handlers not supported on Windows | Fixed in workers — guarded with `if sys.platform != "win32":` |
| `size((e)-[:RELATES_TO]-()) deprecated` | Neo4j 5.x deprecation | Fixed — queries use `COUNT { (e)-[:RELATES_TO]-() }` |
| Query stuck at `status: queued` forever | Worker and API in separate processes with no shared store | Ensure Redis is running; both processes use `ResultStore` backed by Redis |
| `403 API key leaked/expired` | OpenAI or DeepSeek key expired | Regenerate at platform.openai.com or platform.deepseek.com, update `.env`, restart |
| `AMQPConnectionError` | RabbitMQ not running | `docker compose -f compose.dev.yaml up rabbitmq` |
| `Invalid token: Not enough segments` | Empty/expired JWT | Re-run `/auth/dev-token` and rebuild `$h` headers |
| Workers connecting to wrong host in Docker | `.env` uses `localhost` | Docker overrides in `docker-compose.yml` use service names |

---

## License

MIT — see [`LICENSE`](LICENSE).
