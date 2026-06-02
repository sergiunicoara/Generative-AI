# AI Knowledge Graph & Ontology Platform

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
| **RDF / OWL export** | `scripts/export_rdf.py` serialises to Turtle with `owl:NamedIndividual`, `owl:ObjectProperty`, `rdfs:subClassOf`, and reified confidence annotations; `--infer` applies OWL-RL closure before writing |
| **OWL-RL reasoning** | `OWLRLReasoner` (owlrl) materialises subClassOf chains, symmetric/inverse properties; `is_consistent()` detects owl:Nothing entailments |
| **SPARQL bridge** | `SPARQLBridge.from_turtle()` + `POST /kg/sparql` — SPARQL 1.1 SELECT in-process over any Turtle export; pre-built queries for entity relations, subclass hierarchy, confidence summary |
| **Link prediction** | `LinkPredictor` wraps trained `TransXTrainer` (TransE): `predict_tail(h,r,?)`, `predict_relation(h,?,t)`, `find_missing_links()` via Neo4j vector ANN; `POST /kg/predict-links` |
| **Domain ontologies** | Config-driven domain overlays (see `config/ontologies/aerospace_regulatory.yml`) — extend type hierarchy and relation schema without code changes; `generate_synthetic_ontology.py` benchmarks at 500 types, 170k relations/sec |

**Further reading:**
- [`docs/knowledge-graph-architecture.md`](docs/knowledge-graph-architecture.md) — architectural decisions, data model, LLM routing, cross-process result store
- [`docs/ontology-model.md`](docs/ontology-model.md) — formal type hierarchy, relation schema, inference rules, design decisions
- [`docs/entity-resolution.md`](docs/entity-resolution.md) — 4-stage resolution pipeline with examples
- [`docs/cypher-patterns.md`](docs/cypher-patterns.md) — 6 production Cypher patterns: multi-hop traversal, bitemporal as-of, transitive supersession, contradiction scan, community ANN search, entity resolution audit
- [`docs/runbook.md`](docs/runbook.md) — operations: startup order, common failures, backup/restore, schema migration
- [`docs/roadmap.md`](docs/roadmap.md) — current state, scaling limits, near/medium/long-term roadmap
- [`docs/graphrag-terminology.md`](docs/graphrag-terminology.md) — every GraphRAG term defined, with examples and file references
- [`docs/performance-metrics-inventory.md`](docs/performance-metrics-inventory.md) — all 16 metrics (KPI events, graph health, calibration, retrieval stages); storage, access, interpretation, pitch guidance
- [`docs/defensibility-drill.md`](docs/defensibility-drill.md) — 15 hard CTO questions with model answers; preparation checklist
- [`docs/presentation-playbook.md`](docs/presentation-playbook.md) — full run-of-show: setup commands, slide-by-slide script, live-demo + dashboard choreography, Q&A map, timing variants, failure fallbacks
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — ADR process, PR checklist, coding standards, how to add features
- [`docs/adr/0001-property-graph-over-triple-store.md`](docs/adr/0001-property-graph-over-triple-store.md) — Why Neo4j over RDF triple stores
- [`docs/adr/0002-forward-chaining-over-backward-chaining.md`](docs/adr/0002-forward-chaining-over-backward-chaining.md) — Why materialised inference over query-time reasoning
- [`docs/adr/0003-bayesian-confidence-accumulation.md`](docs/adr/0003-bayesian-confidence-accumulation.md) — Why `1−(1−c₁)(1−c₂)` over last-write-wins
- [`docs/adr/0004-groq-over-gemini-for-text-generation.md`](docs/adr/0004-groq-over-gemini-for-text-generation.md) — Why Groq for generation + Gemini for embeddings; two-model design rationale
- [`docs/adr/0005-redis-as-cross-process-result-store.md`](docs/adr/0005-redis-as-cross-process-result-store.md) — Why Redis over PostgreSQL and RabbitMQ reply-to for result persistence
- [`docs/adr/0006-dual-llm-architecture.md`](docs/adr/0006-dual-llm-architecture.md) — Why 8B routing + 70B synthesis cuts agentic p95 from 6.8 s to 3.4 s
- [`docs/pwc-jd-mapping.md`](docs/pwc-jd-mapping.md) — Every JD requirement mapped to file + endpoint + demo step + honest gap
- [`evals/golden_set.json`](evals/golden_set.json) — 40-question golden eval set; run with `scripts/run_golden_eval.py`

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
                  │           TimescaleDB  :5432                │
                  │              KPI Events Store               │
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
| **6-stage retrieval pipeline** | Vector ANN → BM25+RRF → Cross-encoder → Multi-hop → GAT GNN → LLM |
| **Graph Attention Network (GAT)** | GCN/GAT re-scores chunks using entity embedding propagation; attention weights by cosine similarity between neighbours |
| **Query-adaptive GNN weights** | Relational queries (e.g. "how did X cause Y") auto-shift to 50/50 text/GNN; factoid queries use default α/β |
| **BM25 + Vector hybrid search** | Vector ANN and BM25 fulltext results fused via Reciprocal Rank Fusion (RRF, k=60) |
| **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` deep pairwise query-chunk scoring before graph expansion |
| **Multi-hop graph traversal** | `Chunk → Entity → RELATES_TO* → Entity → Chunk` up to depth 2 |
| **Agentic fallback (IRCoT)** | Low-confidence answers trigger iterative re-search (max 2 steps); two-model design — llama-3.1-8b-instant for routing (~0.2s/step), llama-3.3-70b for final synthesis; agentic p95 **3.4s** |
| **Session context** | Redis-backed conversation history (24h TTL); enriches follow-up queries with prior turn entities |
| **Alias resolution** | Name-based + embedding-based deduplication before every entity MERGE; per-tenant registry pool |
| **Document authority hierarchy** | 4-level authority system (Regulatory → Manufacturer → Internal → Informal); superseded docs penalised |
| **Contradiction detection** | Multi-source, directional-reversal, exclusive-state, and functional-violation conflict types; scoped per tenant |
| **Community detection** | Leiden algorithm (graspologic) builds hierarchical graph summaries for global search; staleness-gated auto-rebuild |
| **Graph health metrics** | 6 semantic metrics (alias coverage, relation precision, contradiction rate, orphan growth, merge/split proxy, community coherence) with per-tenant trend snapshots |
| **Ontology enforcement** | Domain/range validation on every relation write; deprecated relation names auto-migrated on ingestion |
| **Tenant isolation** | All entities, edges, conflicts, communities, and health snapshots are scoped by `(name, type, tenant)` |
| **Graph integrity guards** | Self-loop removal, cycle detection, quarantine, ingestion validation, dirty-flag propagation after every write |
| **Manual correction API** | `/corrections` endpoints: entity split, quarantine/release, edge reject/override, conflict resolution |
| **Agent tool safety** | `ToolPolicy` gate: allowlist, per-tool risk levels (low/medium/high/restricted), scope enforcement, arg validation, cross-tenant guard, dry-run mode, timeout, structured audit log; 49 guardrail tests |
| **Query result cache** | Redis-backed, provenance-aware cache in `QueryConsumer`; cache hit skips all 6 retrieval stages; invalidates only queries that cited affected entities; TTL configurable via `QUERY_RESULT_TTL_SECONDS` |
| **Redis alias registry** | `AliasRegistry.load()` pushes alias table to Redis hash (`graphrag:aliases:{tenant}`, 24h TTL); parallel workers warm from Redis without full Neo4j scan; `load_alias_registry()` is Redis-first |
| **Wikidata entity linking** | Optional post-ingestion step (`WIKIDATA_LINKING=1`); grounds high-confidence entities to canonical QIDs; rate-limited to 20 entities/document |
| **RAGAS evaluation** | Faithfulness, answer relevancy, context precision, context recall — auto-sampled at 20% |
| **OAuth 2.0** | Google browser login + M2M client credentials grant (JWT Bearer) |
| **Business Matrix** | Live Plotly Dash dashboard with KPI timeseries and alert thresholds |
| **Worker health probes** | `GET /ready` + `GET /live` on each worker (`WORKER_HEALTH_PORT`); aiohttp server in `graphrag/workers/health_server.py`; compose.dev.yaml and Kubernetes readiness probes use `/ready` |
| **Structured DLQ** | Failed messages carry `exception_type`, `error`, `retry_count`, `queue`, `message_id`, `payload_summary` — full JSON envelope for automated triage |
| **Async pipeline** | RabbitMQ decouples ingestion, query, and evaluation workers with structured DLQ; `compose.dev.yaml` starts the full stack in one command |

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
  ├─► Groq (llama-3.3-70b-versatile) generates grounded answer with chunk citations
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
| Graph DB | Neo4j 5.20 |
| Session Store | Redis 7 |
| Message Queue | RabbitMQ 3.13 |
| KPI Store | TimescaleDB (PostgreSQL 16) |
| Embeddings | `gemini-embedding-001` (3072d) via Google Generative AI |
| LLM | Groq `llama-3.3-70b-versatile` (free-tier; 1500+ RPD) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers) |
| GNN | PyTorch — GAT / GCN (configurable) |
| Community detection | graspologic (Leiden algorithm) |
| Agent Framework | Google ADK |
| Evaluation | RAGAS |
| API | FastAPI + Uvicorn |
| Dashboard | Plotly Dash |
| Auth | OAuth 2.0 · python-jose JWT (HS256) |
| Runtime | Python 3.11 |

---

## Project Structure

```
AI knowledge graph platform/
├── api/
│   ├── main.py                      # FastAPI app, lifespan hook, middleware, routes
│   ├── limiter.py                   # slowapi rate-limiter singleton (20/min ingest, 60/min query)
│   ├── auth/
│   │   ├── dependencies.py          # get_current_user, require_scope (unconditional)
│   │   ├── google.py                # Google OAuth 2.0 Authorization Code flow
│   │   └── jwt.py                   # HS256 JWT creation & validation
│   └── routes/
│       ├── auth.py                  # /auth/login, /callback, /token, /clients (Redis-backed M2M)
│       ├── ingest.py                # POST /ingest  (rate-limited)
│       ├── query.py                 # POST /query, GET /query/{id}  (rate-limited; Redis result store)
│       ├── evaluation.py            # GET /evaluation/summary
│       ├── kpis.py                  # GET /kpis/summary, /kpis/timeseries
│       └── corrections.py           # entity split · quarantine · edge override · conflict resolve
│
├── graphrag/
│   ├── agents/
│   │   ├── base_agent.py            # Abstract Google ADK agent base
│   │   ├── ingestion_agent.py       # Document → chunk → embed → extract → graph
│   │   ├── query_agent.py           # Question → retrieve → answer
│   │   └── evaluation_agent.py      # RAGAS scoring agent
│   ├── business_matrix/
│   │   ├── dashboard_server.py      # Plotly Dash on :8050/dashboard/
│   │   ├── kpi_store.py             # SQLAlchemy KPI event model (recorded_at indexed)
│   │   └── kpi_tracker.py           # KPI aggregation queries; real p50/p95 percentile (capped at 10k rows)
│   ├── core/
│   │   ├── config.py                # Settings (pydantic-settings, .env + YAML); production validators
│   │   ├── llm_client.py            # Central LLM router: Groq for generation, Gemini for embeddings
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
│   │   ├── embedder.py              # Gemini embedding batches
│   │   ├── extractor.py             # LLM entity + relation extraction
│   │   └── graph_writer.py          # Persist chunks/entities/relations; alias resolution; validation
│   ├── messaging/
│   │   ├── rabbitmq_client.py       # aio-pika connection, publish, consume, DLQ
│   │   ├── publishers.py            # publish_document(), publish_query(), publish_eval_job()
│   │   └── consumers.py             # Message handler wiring per queue
│   └── retrieval/
│       ├── local_search.py          # 6-stage pipeline: vector + BM25 + rerank + multihop + GNN + context
│       ├── global_search.py         # Community embedding search + map-reduce synthesis
│       ├── hybrid_retriever.py      # Combines local + global; agentic fallback; session turn recording
│       ├── agentic_retriever.py     # Iterative IRCoT re-search (Google ADK)
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
├── scripts/
│   ├── init_neo4j.py                # Idempotent schema initializer (run once after docker up)
│   └── community_rebuild.py         # CLI: rebuild communities per tenant with staleness check
│
├── tests/
│   └── integration/
│       └── test_safety_paths.py     # Tenant isolation · ontology · contradiction · quarantine · community
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
├── requirements.txt                 # Direct dependencies
├── requirements.lock                # Fully-pinned lock file (regenerate: make lock)
└── .env                             # Secrets (never commit)
```

---

## Verified System Check Results

Full end-to-end test completed 2026-03-21 (updated 2026-05-31 with Groq integration):

| Step | Component | Result |
|------|-----------|--------|
| Infrastructure | Neo4j + RabbitMQ + TimescaleDB + Redis | ✅ Healthy |
| API | FastAPI + OAuth + lifespan hook | ✅ Running on :8000 |
| Ingestion | doc → chunk → embed (Gemini 3072d) → extract (Groq Llama) → graph | ✅ |
| Schema | Vector indexes + BM25 fulltext indexes (6 total, all ONLINE) | ✅ |
| Graph counts | 1 doc · 1 chunk · 5 entities · 4 relations | ✅ |
| Hybrid search | BM25=10 + vector=10 → fused=10 chunks | ✅ |
| Cross-encoder reranker | ms-marco-MiniLM-L-6-v2, top_score=9.30 | ✅ |
| GNN scoring | GAT 2-layer; α=0.9 text + β=0.1 graph | ✅ |
| Answer synthesis | Groq llama-3.3-70b-versatile; citations included | ✅ |
| Session context | Redis-backed; turn recorded after answer | ✅ |
| RAGAS | 20% sampling; metrics stored in TimescaleDB | ✅ |
| Dashboard | Live KPI charts at /dashboard/ | ✅ |

---

## Quick Start

### 1. Prerequisites

- Python 3.11
- Docker Desktop
- Google AI Studio API key → https://aistudio.google.com/app/apikey
- Google OAuth credentials → https://console.cloud.google.com/apis/credentials

### 2. Clone & install

```bash
git clone <repo>
cd "AI knowledge graph platform"
py -3.11 -m pip install -r requirements.txt
```

### 3. Configure `.env`

```env
# Google AI — embeddings only (3072d vectors)
GOOGLE_API_KEY=AIzaSy...
GEMINI_EMBED_MODEL=gemini-embedding-001

# Groq — text generation (free-tier, 1500+ RPD)
# Get key at: https://console.groq.com/keys
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphrag_dev

# RabbitMQ
RABBITMQ_URL=amqp://graphrag:graphrag_dev@localhost:5672/

# Redis (session context + cross-process query results)
REDIS_URL=redis://localhost:6379/0

# TimescaleDB
TIMESCALE_URL=postgresql+asyncpg://graphrag:graphrag_dev@localhost:5432/graphrag_kpis

# OAuth 2.0
JWT_SECRET_KEY=<run: py -3.11 -c "import secrets; print(secrets.token_hex(32))">
SESSION_SECRET_KEY=<run: py -3.11 -c "import secrets; print(secrets.token_hex(32))">
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

### 5. Initialize Neo4j schema

```bash
py -3.11 scripts/init_neo4j.py
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
py -3.11 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Ingestion worker  (health probe: http://localhost:8081/ready)
py -3.11 workers/ingestion_worker.py

# Terminal 3 — Query worker      (health probe: http://localhost:8082/ready)
py -3.11 workers/query_worker.py

# Terminal 4 — Evaluation worker (health probe: http://localhost:8083/ready)
py -3.11 workers/evaluation_worker.py

# Terminal 5 — Dashboard
py -3.11 graphrag/business_matrix/dashboard_server.py
```

**Single-machine shortcut:**
```bash
py -3.11 workers/combined_worker.py
```

### 7. Seed demo data (optional)

Populate the graph with a curated aerospace regulatory corpus — 20 entities, 12 relations,
2 conflict pairs (triggers the contradiction demo), health and calibration snapshots:

```bash
py -3.11 scripts/seed_demo_data.py --commit --tenant aerospace
# Or wipe and re-seed:
py -3.11 scripts/seed_demo_data.py --wipe --commit --tenant aerospace
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
Invoke-RestMethod -Uri http://localhost:8000/corrections/split -Method POST -Headers $h `
  -Body '{"entity_name":"Apple","entity_type":"ORG","tenant":"default"}'

# List open contradiction conflicts
Invoke-RestMethod -Uri http://localhost:8000/corrections/conflicts -Method GET -Headers $h

# Resolve a conflict
Invoke-RestMethod -Uri http://localhost:8000/corrections/conflict-resolve -Method POST -Headers $h `
  -Body '{"conflict_id":"...","resolution":"manual_override"}'
```

### Rebuild communities (CLI)

```bash
# Check staleness, rebuild if needed
py -3.11 scripts/community_rebuild.py --tenant default

# Force rebuild regardless of staleness
py -3.11 scripts/community_rebuild.py --tenant default --force

# Dry-run: check without rebuilding
py -3.11 scripts/community_rebuild.py --tenant default --dry-run
```

---

## Configuration

All tuning is in `config/settings.yml`:

```yaml
ingestion:
  chunk_size: 512
  chunk_overlap: 64
  embedding_batch_size: 100
  entity_types: [PERSON, ORG, PRODUCT, CONCEPT, LOCATION, EVENT]
  alias_embedding_threshold: 0.92   # cosine similarity to treat as duplicate entity
  alias_fuzzy_threshold: 85         # rapidfuzz ratio for soft name matching
  validate_after_ingestion: true    # run graph health check after every doc
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

8. GROQ/LLAMA GENERATES ANSWER
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

### Answer Quality — RAGAS (20% sample, llama-3.3-70b, 104 queries)

| Metric | Measured | Target |
|--------|----------|--------|
| `faithfulness` | **0.840** | ≥ 0.85 |
| `answer_relevancy` | **0.816** | ≥ 0.80 ✓ |
| `context_precision` | **0.907** | ≥ 0.80 ✓ |
| `context_recall` | **0.867** | ≥ 0.80 ✓ |

### Latency — reported per retrieval mode (A73: never combine)

| Mode | Avg | p95 | Notes |
|------|-----|-----|-------|
| Hybrid (6-stage) | 1,734 ms | **2,162 ms ✓** | 91% of queries |
| Agentic (IRCoT) | 2,842 ms | **3,442 ms** | 9% of queries — by design |
| Combined | 1,842 ms | 2,719 ms | Inflated by mode mix |

### Graph Health (12-doc aerospace seed corpus · pipeline targets ~2k entities at scale)

Seed corpus: 10 aerospace regulatory documents (FAA/EASA ADs, manufacturer records).
Run `py -3.11 scripts/seed_demo_data.py --commit` to populate.

| Metric | Seed | Production target | Threshold |
|--------|------|-------------------|-----------|
| Entities | **20** | ~2,000+ | — |
| Relations | **12** | ~7,000+ | — |
| Alias coverage | pipeline wired | > 90% | > 85% |
| Contradiction density | detection verified | < 0.85 /1k edges | < 2.0 |
| Community coherence | pipeline wired | > 0.65 | > 0.50 |
| Brier score (calibration pipeline) | pipeline wired | < 0.20 | < 0.25 |

Evaluation is sampled at **20%** of queries automatically. View results:

- `GET /evaluation/summary`
- `GET /kpis/summary`
- **http://localhost:8050/dashboard/**

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
| **Conflicts** | Themed table of open Conflict nodes. Select a row + resolution type to call `POST /corrections/resolve-conflict`. |
| **Communities** | Change-fraction + changed-entities tiles, "Rebuild Affected Communities" action, version-history table. |
| **GDPR & PII** | Erasure audit log + "Forget Entity · GDPR Article 17" form (`POST /kg/gdpr/forget-entity`). |
| **Calibration** | Brier-score rating tile + trend + isotonic calibration curve. |

```bash
export GRAPHRAG_ADMIN_TOKEN="your-secret-token"   # empty = open (dev only)
uvicorn api.main:app                              # → http://localhost:8000/admin/
```

### Business Matrix — `/dashboard/`

Query-level KPIs from the local SQLite store: status-coloured tiles (queries, avg/p95
latency, faithfulness, context recall) + branded metric trend with alert threshold.

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
| `No such vector schema index: chunk_embeddings` | Schema not initialized | Run `py -3.11 scripts/init_neo4j.py` |
| `startup.session_store_unavailable` | Redis unreachable at startup | Start Redis or set `session_store_strict: false` in settings.yml |
| `graspologic is not installed` | Leiden community detection unavailable | `pip install graspologic` or set `require_leiden: false` (degrades global search) |
| `No module named 'groq'` | Groq package not installed | `pip install groq` |
| `No module named 'redis'` | redis[asyncio] not installed | `pip install "redis[asyncio]"` |
| `ImportError: redis[asyncio] is not installed but session_store=redis` | Redis package missing with strict=true | Install redis or set `session_store_strict: false` in settings.yml |
| `NotImplementedError: add_signal_handler` (Windows) | Signal handlers not supported on Windows | Fixed in workers — guarded with `if sys.platform != "win32":` |
| `size((e)-[:RELATES_TO]-()) deprecated` | Neo4j 5.x deprecation | Fixed — queries use `COUNT { (e)-[:RELATES_TO]-() }` |
| Query stuck at `status: queued` forever | Worker and API in separate processes with no shared store | Ensure Redis is running; both processes use `ResultStore` backed by Redis |
| `403 API key leaked/expired` | Google revoked the Gemini key | Create new key at aistudio.google.com, update `.env`, restart |
| `AMQPConnectionError` | RabbitMQ not running | `docker compose -f compose.dev.yaml up rabbitmq` |
| `Invalid token: Not enough segments` | Empty/expired JWT | Re-run `/auth/dev-token` and rebuild `$h` headers |
| Workers connecting to wrong host in Docker | `.env` uses `localhost` | Docker overrides in `docker-compose.yml` use service names |

---

## License

MIT
