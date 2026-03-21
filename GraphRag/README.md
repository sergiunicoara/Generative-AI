# GraphRAG — Enterprise Knowledge Graph RAG Pipeline

An end-to-end **GraphRAG** system that combines knowledge graph retrieval with LLMs to answer complex, multi-document questions. Built with Neo4j, RabbitMQ, Google ADK, RAGAS, and a live KPI dashboard.

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              FastAPI  :8000                  │
                        │  /ingest  /query  /kpis  /evaluation  /auth  │
                        └──────────┬──────────────┬────────────────────┘
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
                  └───────────────────────────────────────────┘
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
| **Graph-based retrieval** | Entities + relations stored in Neo4j; multi-hop traversal resolves cross-document facts |
| **BM25 + Vector hybrid search** | Vector ANN and BM25 fulltext results fused via Reciprocal Rank Fusion (RRF) |
| **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` re-scores RRF candidates with deep pairwise query-chunk scoring before graph expansion |
| **Multi-hop graph traversal** | `Chunk → Entity → RELATES_TO* → Entity → Chunk` up to depth 2 |
| **Agentic fallback** | Low-confidence answers trigger iterative IRCoT re-search via Google ADK |
| **RAGAS evaluation** | Faithfulness, answer relevancy, context precision, context recall — auto-sampled at 20% |
| **OAuth 2.0** | Google browser login + M2M client credentials grant (JWT Bearer) |
| **Business Matrix** | Live Plotly Dash dashboard with KPI timeseries and alert thresholds |
| **Async pipeline** | RabbitMQ decouples ingestion, query, and evaluation workers with DLQ + TTL |
| **Community detection** | Leiden algorithm builds hierarchical graph summaries for global search |

---

## Stack

| Component | Technology |
|-----------|-----------|
| Graph DB | Neo4j 5.20 + APOC + GDS |
| Message Queue | RabbitMQ 3.13 |
| KPI Store | TimescaleDB (PostgreSQL 16) |
| Embeddings | `gemini-embedding-001` (3072d) |
| LLM | `gemini-2.5-flash` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers) |
| Agent Framework | Google ADK |
| Evaluation | RAGAS |
| API | FastAPI + Uvicorn |
| Dashboard | Plotly Dash |
| Auth | OAuth 2.0 · python-jose JWT (HS256) |
| Runtime | Python 3.11 |

---

## Verified System Check Results

Full end-to-end test completed 2026-03-21:

| Step | Component | Result |
|------|-----------|--------|
| Infrastructure | Neo4j + RabbitMQ + TimescaleDB | ✅ Healthy |
| API | FastAPI + OAuth | ✅ Running on :8000 |
| Ingestion | 3 documents → 19 entities, 23 relations | ✅ |
| Schema | Vector indexes + BM25 fulltext indexes | ✅ |
| Graph counts | 3 docs · 3 chunks · 19 entities | ✅ |
| Hybrid search | BM25 + vector fusion (all chunks tagged `vector+bm25`) | ✅ |
| Cross-encoder reranker | ms-marco-MiniLM-L-6-v2, top_score=2.97 | ✅ |
| Cross-doc query | Falcon 9/Starship answer spanning 3 documents | ✅ |
| RAGAS | context_precision=1.0 · context_recall=1.0 | ✅ |
| Dashboard | Live KPI charts at /dashboard/ | ✅ |
| Agentic fallback | IRCoT triggered on low-confidence query | ✅ |

---

## Project Structure

```
GraphRag/
├── api/
│   ├── main.py                  # FastAPI app, middleware, route registration
│   ├── auth/
│   │   ├── dependencies.py      # get_current_user, require_scope
│   │   ├── google.py            # Google OAuth 2.0 Authorization Code flow
│   │   └── jwt.py               # HS256 JWT creation & validation
│   └── routes/
│       ├── auth.py              # /auth/login, /callback, /token, /clients, /dev-token
│       ├── ingest.py            # POST /ingest  (requires write scope)
│       ├── query.py             # POST /query, GET /query/{id}  (requires read scope)
│       ├── evaluation.py        # GET  /evaluation/summary
│       └── kpis.py              # GET  /kpis/summary, /kpis/timeseries
│
├── graphrag/
│   ├── agents/
│   │   ├── base_agent.py        # Abstract Google ADK agent base
│   │   ├── ingestion_agent.py   # Document → chunk → embed → extract → graph
│   │   ├── query_agent.py       # Question → retrieve → answer
│   │   └── evaluation_agent.py  # RAGAS scoring agent
│   ├── business_matrix/
│   │   ├── dashboard_server.py  # FastAPI + Plotly Dash on :8050/dashboard/
│   │   ├── kpi_store.py         # SQLAlchemy KPI event model
│   │   └── kpi_tracker.py       # KPI aggregation queries
│   ├── core/
│   │   ├── config.py            # Settings (pydantic-settings, .env + YAML)
│   │   └── models.py            # Domain models: Document, Chunk, Entity, ...
│   ├── graph/
│   │   ├── neo4j_client.py      # Async Neo4j driver, MERGE helpers, vector + BM25 search
│   │   ├── schema.cypher        # Constraints, vector indexes, fulltext indexes
│   │   ├── community_builder.py # Leiden community detection (graspologic)
│   │   └── community_summarizer.py  # LLM-generated community summaries
│   ├── ingestion/
│   │   ├── chunker.py           # Sliding-window text chunking
│   │   ├── embedder.py          # Gemini embedding batches
│   │   ├── extractor.py         # LLM entity + relation extraction
│   │   └── graph_writer.py      # Persist chunks/entities/relations to Neo4j
│   ├── messaging/
│   │   ├── rabbitmq_client.py   # aio-pika connection, publish, consume, DLQ
│   │   ├── publishers.py        # publish_document(), publish_query(), publish_eval_job()
│   │   └── consumers.py         # Message handler wiring per queue
│   └── retrieval/
│       ├── local_search.py      # Vector ANN + BM25 RRF fusion + multi-hop graph expansion
│       ├── global_search.py     # Community embedding search + synthesis
│       ├── hybrid_retriever.py  # Combines local + global, triggers agentic fallback
│       ├── agentic_retriever.py # Iterative IRCoT re-search (Google ADK)
│       ├── bm25_search.py       # HybridBM25Search with RRF (k=60)
│       └── context_builder.py   # Assembles LLM context string from results
│
├── workers/
│   ├── ingestion_worker.py      # Consumes graphrag.ingest queue
│   ├── query_worker.py          # Consumes graphrag.query queue
│   └── evaluation_worker.py     # Consumes graphrag.eval queue
│
├── scripts/
│   └── init_neo4j.py            # Idempotent schema initializer (run once after docker up)
│
├── config/
│   └── settings.yml             # Chunking, graph, retrieval, evaluation tuning
├── docker-compose.yml           # Neo4j, RabbitMQ, TimescaleDB (+ optional containerized workers)
├── Dockerfile
├── requirements.txt
└── .env                         # Secrets (never commit)
```

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
docker-compose up neo4j rabbitmq timescaledb
```

### 5. Initialize Neo4j schema

```bash
py -3.11 scripts/init_neo4j.py
```

Run this once after Neo4j first starts. It creates vector indexes, fulltext indexes, and constraints (all idempotent).

### 6. Start workers and API (each in a separate terminal)

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

### Browser (production OAuth)

1. Visit **http://localhost:8000/auth/login** — redirects to Google sign-in
2. After sign-in, redirected back with JWT Bearer token

### M2M client credentials

```bash
# Register a client
curl -X POST http://localhost:8000/auth/clients \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"client_name": "my-script", "scopes": ["read", "write"]}'

# Get a Bearer token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "client_credentials",
    "client_id": "graphrag_...",
    "client_secret": "...",
    "scope": "read write"
  }'
```

### Ingest a document

```powershell
Invoke-RestMethod -Uri http://localhost:8000/ingest -Method POST -Headers $h `
  -Body '{"filename":"report.txt","text":"Company A owns Company B. Company B launched a rocket."}'
```

### Query

```powershell
$q = Invoke-RestMethod -Uri http://localhost:8000/query -Method POST -Headers $h `
  -Body '{"question":"What did Company A launch?","mode":"hybrid","ground_truth":"Company A launched a rocket via its subsidiary Company B."}'

# Poll for result
Invoke-RestMethod -Uri "http://localhost:8000/query/$($q.query_id)" -Method GET -Headers $h
```

### Check RAGAS scores

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/evaluation/summary" -Method GET -Headers $h
```

---

## End-to-End Flow: User Asks a Question

> **Example:** *"What rockets did Elon Musk's company launch and what did they achieve?"*

This question spans 3 separate documents that have no direct text overlap.

```
1. USER AUTHENTICATES
   POST /auth/dev-token  →  JWT (HS256, 60 min)
   Authorization: Bearer eyJhbGci...

2. USER SUBMITS QUESTION
   POST /query
   { "question": "What rockets did Elon Musk company launch and what did they achieve?" }
   →  FastAPI validates JWT scope("read")
   →  Publishes message to RabbitMQ exchange: graphrag.query
   →  Returns immediately: { query_id: "abc-123", status: "queued" }

3. QUERY WORKER PICKS UP MESSAGE
   rabbitmq.consuming  exchange=graphrag.query  queue=graphrag.query.queue
   →  QueryAgent.run(query_id, question)

4. LOCAL SEARCH (parallel vector + BM25)
   ┌─ Vector ANN ──────────────────────────────────────────────────────┐
   │  embed("What rockets did Elon Musk company launch...")            │
   │  → 3072d vector via gemini-embedding-001                         │
   │  → db.index.vector.queryNodes('chunk_embeddings', 10, $vec)      │
   │  → top-3 chunks by cosine similarity                             │
   └───────────────────────────────────────────────────────────────────┘
   ┌─ BM25 Fulltext ───────────────────────────────────────────────────┐
   │  CALL db.index.fulltext.queryNodes('chunk_fulltext',             │
   │       'rockets Elon Musk launch achieve')                        │
   │  → top-3 chunks by BM25 score                                    │
   └───────────────────────────────────────────────────────────────────┘
   ┌─ RRF Fusion (k=60) ───────────────────────────────────────────────┐
   │  chunk A: vector rank 1, bm25 rank 1  → score 0.0328 [vector+bm25]│
   │  chunk B: vector rank 2, bm25 rank 2  → score 0.0320 [vector+bm25]│
   │  chunk C: vector rank 3, bm25 rank 3  → score 0.0317 [vector+bm25]│
   └───────────────────────────────────────────────────────────────────┘
   ┌─ Cross-Encoder Reranking ──────────────────────────────────────────┐
   │  model: cross-encoder/ms-marco-MiniLM-L-6-v2                      │
   │  scores every (query, chunk) pair independently — not cosine       │
   │  chunk A: rerank_score=2.97  ← achievements, most relevant        │
   │  chunk B: rerank_score=1.83  ← products                           │
   │  chunk C: rerank_score=0.42  ← ownership                          │
   │  → top rerank_k=5 passed forward to graph expansion               │
   └───────────────────────────────────────────────────────────────────┘

5. MULTI-HOP GRAPH TRAVERSAL (depth=2)
   chunk A mentions → Entity("SpaceX")
   Entity("SpaceX")  -[RELATES_TO]→  Entity("Falcon 9")
   Entity("Falcon 9") ← MENTIONS ─  chunk in achievements.txt

   Cross-document bridge resolved:
     ownership.txt    →  "Elon Musk owns SpaceX"
     products.txt     →  "SpaceX manufactures Falcon 9, Starship"
     achievements.txt →  "Falcon 9 landed 2015, Starship flew 2023, NASA Artemis"

6. GLOBAL SEARCH
   embed(question) → query community_embeddings index
   → Community nodes with pre-built cluster summaries
   → Adds high-level context about the SpaceX/Tesla/Elon Musk cluster

7. CONFIDENCE CHECK
   if citations found AND answer is specific:
       → skip agentic fallback  ✅ (this query passes)
   else:
       → trigger AgenticRetriever (IRCoT loop, max 4 steps)
          Step 1: LLM reasons → SEARCH: Elon Musk vehicles Mars
          Step 2: re-search → no new chunks found
          Step 3: SEARCH: SpaceX Mars mission → still nothing
          Step 4: ANSWER: insufficient context in knowledge base

8. GEMINI GENERATES ANSWER
   Context: local chunks (60%) + community summaries (40%)
   Model: gemini-2.5-flash

   Answer:
     "SpaceX, founded by Elon Musk, launched:
      • Falcon 9 — first booster landing 2015, 200+ missions [Chunk 8910eded]
      • Starship — first successful flight 2023, NASA Artemis HLS [Chunk 8910eded]"

9. USER POLLS FOR RESULT
   GET /query/abc-123
   → { status: "completed", answer: "...", citations: [...], latency_ms: 3860 }

10. RAGAS EVALUATION (20% sampled)
    → EvalJob published to graphrag.eval queue
    → EvaluationAgent runs:
         faithfulness      = 1.0   (answer grounded in retrieved chunks)
         context_precision = 1.0   (all retrieved chunks were relevant)
         context_recall    = 1.0   (all ground truth facts were found)
         answer_relevancy  = 0.85  (answer directly addresses the question)
    → Scores stored in TimescaleDB

11. DASHBOARD UPDATES
    http://localhost:8050/dashboard/
    → Plotly chart refreshes every 30s
    → latency timeseries + RAGAS score table + alert thresholds
```

---

## Retrieval Pipeline

```
Query
  │
  ├─► LocalSearch
  │     ├─ Vector ANN on chunk_embeddings index (3072d cosine)
  │     ├─ BM25 fulltext on chunk_fulltext index
  │     ├─ RRF fusion (k=60) of vector + BM25 results
  │     ├─ Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) → top rerank_k
  │     └─ Multi-hop graph traversal (depth=2)
  │         Chunk → MENTIONS → Entity → RELATES_TO* → Entity → MENTIONS → Chunk
  │
  ├─► GlobalSearch
  │     └─ Community embedding ANN → synthesized summaries
  │
  ├─► ContextBuilder (local 60% + global 40%)
  │
  ├─► Gemini generates answer with citations
  │
  └─► Low confidence? (no citations / vague answer)
        └─► AgenticRetriever (IRCoT loop, max 4 steps)
              ├─ Step N: LLM reasons → SEARCH: <sub-query>
              ├─ Re-search → add new chunks to context
              └─ Until ANSWER: <final> or max steps reached
```

This solves the **cross-document reasoning** problem:

> *"Company A owns Company B"* (doc 1, page 10)
> *"Company B launched a rocket"* (doc 2, page 300)
> Query: *"What did Company A launch?"* → **correctly answered via graph traversal**

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

## Configuration

All tuning is in `config/settings.yml`:

```yaml
retrieval:
  local_top_k: 10          # chunks retrieved by vector search
  multihop_depth: 2        # graph hops (2 = A→B→C)
  bm25_enabled: true       # enable BM25 fulltext search
  agentic_fallback: true   # enable iterative re-search on low confidence
  agentic_max_steps: 4     # max sub-searches before forced synthesis

ingestion:
  chunk_size: 512
  chunk_overlap: 64

graph:
  min_community_size: 3    # minimum entities to form a community
  leiden_resolution: 1.0
```

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
| `403 API key leaked/expired` | Google revoked the key | Create new key at aistudio.google.com, update `.env`, restart all processes |
| `AMQPConnectionError` | RabbitMQ not running | `docker-compose up rabbitmq` |
| `Invalid token: Not enough segments` | Empty/expired JWT | Re-run `/auth/dev-token` and rebuild `$h` headers |
| Workers connecting to wrong host in Docker | `.env` uses `localhost` | Docker overrides in `docker-compose.yml` use service names |

---

## License

MIT
