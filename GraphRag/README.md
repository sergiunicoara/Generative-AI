# GraphRAG — Enterprise Knowledge Graph RAG Pipeline

An end-to-end **GraphRAG** system that combines knowledge graph retrieval with LLMs to answer complex, multi-document questions. Built with Neo4j, RabbitMQ, Google ADK, RAGAS, and a live KPI dashboard.

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
                  │        Session context store (24h TTL)      │
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
| **Agentic fallback (IRCoT)** | Low-confidence answers trigger iterative re-search via Google ADK (max 4 steps) |
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
| **RAGAS evaluation** | Faithfulness, answer relevancy, context precision, context recall — auto-sampled at 20% |
| **OAuth 2.0** | Google browser login + M2M client credentials grant (JWT Bearer) |
| **Business Matrix** | Live Plotly Dash dashboard with KPI timeseries and alert thresholds |
| **Async pipeline** | RabbitMQ decouples ingestion, query, and evaluation workers with DLQ + TTL |

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
  ├─► Gemini generates grounded answer with chunk citations
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
| Embeddings | `gemini-embedding-001` (3072d) |
| LLM | `gemini-2.5-flash` |
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
GraphRag/
├── api/
│   ├── main.py                      # FastAPI app, lifespan hook, middleware, routes
│   ├── auth/
│   │   ├── dependencies.py          # get_current_user, require_scope
│   │   ├── google.py                # Google OAuth 2.0 Authorization Code flow
│   │   └── jwt.py                   # HS256 JWT creation & validation
│   └── routes/
│       ├── auth.py                  # /auth/login, /callback, /token, /clients
│       ├── ingest.py                # POST /ingest
│       ├── query.py                 # POST /query, GET /query/{id}
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
│   │   ├── kpi_store.py             # SQLAlchemy KPI event model
│   │   └── kpi_tracker.py           # KPI aggregation queries
│   ├── core/
│   │   ├── config.py                # Settings (pydantic-settings, .env + YAML)
│   │   └── models.py                # Domain models: Document, Chunk, Entity, Relation, Community, SessionTurn ...
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
│       └── session_store.py         # Redis-backed turn store; in-memory fallback; strict startup mode
│
├── workers/
│   ├── ingestion_worker.py          # Consumes graphrag.ingest queue
│   ├── query_worker.py              # Consumes graphrag.query queue
│   └── evaluation_worker.py         # Consumes graphrag.eval queue
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
│   └── settings.yml                 # All pipeline tuning (see Configuration section)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env                             # Secrets (never commit)
```

---

## Verified System Check Results

Full end-to-end test completed 2026-03-21:

| Step | Component | Result |
|------|-----------|--------|
| Infrastructure | Neo4j + RabbitMQ + TimescaleDB + Redis | ✅ Healthy |
| API | FastAPI + OAuth + lifespan hook | ✅ Running on :8000 |
| Ingestion | 3 documents → 19 entities, 23 relations | ✅ |
| Schema | Vector indexes + BM25 fulltext indexes | ✅ |
| Graph counts | 3 docs · 3 chunks · 19 entities | ✅ |
| Hybrid search | BM25 + vector fusion (all chunks tagged `vector+bm25`) | ✅ |
| Cross-encoder reranker | ms-marco-MiniLM-L-6-v2, top_score=2.97 | ✅ |
| GNN scoring | GAT 2-layer; α=0.9 text + β=0.1 graph | ✅ |
| Cross-doc query | Falcon 9/Starship answer spanning 3 documents | ✅ |
| RAGAS | context_precision=1.0 · context_recall=1.0 | ✅ |
| Dashboard | Live KPI charts at /dashboard/ | ✅ |
| Agentic fallback | IRCoT triggered on low-confidence query | ✅ |

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
cd GraphRag
py -3.11 -m pip install -r requirements.txt
```

### 3. Configure `.env`

```env
# Google AI
GOOGLE_API_KEY=AIzaSy...
GEMINI_INGEST_MODEL=gemini-2.5-flash
GEMINI_QUERY_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=gemini-embedding-001

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphrag_dev

# RabbitMQ
RABBITMQ_URL=amqp://graphrag:graphrag_dev@localhost:5672/

# TimescaleDB
TIMESCALE_URL=postgresql+asyncpg://graphrag:graphrag_dev@localhost:5432/graphrag_kpis

# OAuth 2.0
JWT_SECRET_KEY=<run: py -3.11 -c "import secrets; print(secrets.token_hex(32))">
GOOGLE_OAUTH_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret
CORS_ORIGINS=["http://localhost:8000","http://localhost:8050"]

# App
LOG_LEVEL=INFO
ENV=development
```

### 4. Start infrastructure

```bash
docker-compose up neo4j rabbitmq timescaledb redis
```

### 5. Initialize Neo4j schema

```bash
py -3.11 scripts/init_neo4j.py
```

Run once after Neo4j first starts. Creates vector indexes, fulltext indexes, constraints, and relation indexes (all idempotent).

### 6. Start workers and API

```bash
# Terminal 1 — API
py -3.11 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Ingestion worker
py -3.11 workers/ingestion_worker.py

# Terminal 3 — Query worker
py -3.11 workers/query_worker.py

# Terminal 4 — Evaluation worker
py -3.11 workers/evaluation_worker.py

# Terminal 5 — Dashboard
py -3.11 graphrag/business_matrix/dashboard_server.py
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

8. GEMINI GENERATES ANSWER
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

## RAGAS Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| `faithfulness` | Answer grounded in retrieved context | > 0.8 |
| `answer_relevancy` | Answer addresses the actual question | > 0.8 |
| `context_precision` | Retrieved chunks are relevant | > 0.9 |
| `context_recall` | All necessary info was retrieved | > 0.9 |

Evaluation is sampled at **20%** of queries automatically. View results:

- `GET /evaluation/summary`
- `GET /kpis/summary`
- **http://localhost:8050/dashboard/**

---

## Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| API + Swagger | http://localhost:8000/docs | Bearer token via /auth/dev-token |
| Neo4j Browser | http://localhost:7474 | neo4j / graphrag_dev |
| RabbitMQ UI | http://localhost:15672 | graphrag / graphrag_dev |
| Dashboard | http://localhost:8050/dashboard/ | — |

---

## Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `No such vector schema index: chunk_embeddings` | Schema not initialized | Run `py -3.11 scripts/init_neo4j.py` |
| `startup.session_store_unavailable` | Redis unreachable at startup | Start Redis or set `session_store_strict: false` in settings.yml |
| `graspologic is not installed` | Leiden community detection unavailable | `pip install graspologic` or set `require_leiden: false` (degrades global search) |
| `403 API key leaked/expired` | Google revoked the key | Create new key at aistudio.google.com, update `.env`, restart |
| `AMQPConnectionError` | RabbitMQ not running | `docker-compose up rabbitmq` |
| `Invalid token: Not enough segments` | Empty/expired JWT | Re-run `/auth/dev-token` and rebuild `$h` headers |
| Workers connecting to wrong host in Docker | `.env` uses `localhost` | Docker overrides in `docker-compose.yml` use service names |

---

## License

MIT
