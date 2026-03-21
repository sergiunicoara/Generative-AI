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
                    └──┬──────────┬───┘      └────────────────────┘
                       │          │
          ┌────────────▼──┐  ┌────▼───────────┐  ┌──────────────────┐
          │  Ingestion    │  │  Query Worker  │  │ Evaluation Worker│
          │  Worker       │  │                │  │  (RAGAS)         │
          └────────────┬──┘  └────┬───────────┘  └──────┬───────────┘
                       │          │                      │
                  ┌────▼──────────▼──────────────────────▼────┐
                  │                  Neo4j  :7687              │
                  │   Document → Chunk → Entity → Community   │
                  └───────────────────────────────────────────┘
                                       │
                  ┌────────────────────▼───────────────────────┐
                  │           TimescaleDB / SQLite  :5432       │
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
| **Hybrid search** | Vector similarity (ANN) + graph traversal combined |
| **Agentic fallback** | Low-confidence answers trigger iterative IRCoT re-search via Google ADK |
| **RAGAS evaluation** | Faithfulness, answer relevancy, context precision, context recall — auto-sampled at 20% |
| **OAuth 2.0** | Google browser login + M2M client credentials grant (JWT Bearer) |
| **Business Matrix** | Live Plotly Dash dashboard with KPI timeseries and alert thresholds |
| **Async pipeline** | RabbitMQ decouples ingestion, query, and evaluation workers |
| **Community detection** | Leiden algorithm builds hierarchical graph summaries for global search |

---

## Stack

| Component | Technology |
|-----------|-----------|
| Graph DB | Neo4j 5.20 + APOC + GDS |
| Message Queue | RabbitMQ 3.13 |
| KPI Store | TimescaleDB (PostgreSQL 16) / SQLite (dev) |
| Embeddings | `gemini-embedding-001` (3072d) |
| LLM | `gemini-2.5-flash` |
| Agent Framework | Google ADK |
| Evaluation | RAGAS |
| API | FastAPI + Uvicorn |
| Dashboard | Plotly Dash |
| Auth | OAuth 2.0 · python-jose JWT |
| Runtime | Python 3.11 |

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
│       ├── auth.py              # /auth/login, /callback, /token, /clients
│       ├── ingest.py            # POST /ingest  (requires write scope)
│       ├── query.py             # POST /query   (requires read scope)
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
│   │   ├── dashboard_server.py  # FastAPI + Plotly Dash on :8050
│   │   ├── kpi_store.py         # SQLAlchemy KPI event model
│   │   └── kpi_tracker.py       # KPI aggregation queries
│   ├── core/
│   │   ├── config.py            # Settings (pydantic-settings, .env + YAML)
│   │   └── models.py            # Domain models: Document, Chunk, Entity, ...
│   ├── graph/
│   │   ├── neo4j_client.py      # Async Neo4j driver + MERGE helpers + vector search
│   │   ├── community_builder.py # Leiden community detection (graspologic)
│   │   └── community_summarizer.py  # LLM-generated community summaries
│   ├── ingestion/
│   │   ├── chunker.py           # Sliding-window text chunking
│   │   ├── embedder.py          # Gemini embedding batches
│   │   ├── extractor.py         # LLM entity + relation extraction
│   │   └── graph_writer.py      # Persist chunks/entities/relations to Neo4j
│   ├── messaging/
│   │   ├── rabbitmq_client.py   # aio-pika connection, publish, consume, DLQ
│   │   ├── publishers.py        # publish_document(), publish_query()
│   │   └── consumers.py         # Message handler wiring
│   └── retrieval/
│       ├── local_search.py      # Vector ANN + multi-hop graph expansion
│       ├── global_search.py     # Community embedding search + synthesis
│       ├── hybrid_retriever.py  # Combines local + global, triggers agentic fallback
│       ├── agentic_retriever.py # Iterative IRCoT re-search (Google ADK)
│       └── context_builder.py  # Assembles LLM context string from results
│
├── workers/
│   ├── ingestion_worker.py      # Consumes graphrag.ingest queue
│   ├── query_worker.py          # Consumes graphrag.query queue
│   └── evaluation_worker.py     # Consumes graphrag.eval queue
│
├── config/
│   └── settings.yml             # Chunking, graph, retrieval, evaluation tuning
├── docker-compose.yml           # Neo4j, RabbitMQ, TimescaleDB, API, workers
├── Dockerfile
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
py -3.11 -m pip install "python-jose[cryptography]" httpx itsdangerous
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
docker-compose up -d neo4j rabbitmq timescaledb
```

### 5. Initialize Neo4j schema

```bash
py -3.11 -c "
import asyncio
from graphrag.graph.neo4j_client import get_neo4j

async def main():
    neo4j = get_neo4j()
    await neo4j.init_schema()
    await neo4j.close()

asyncio.run(main())
"
```

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

### Browser (dev)

1. Visit **http://localhost:8000/auth/dev-login** — sets JWT cookie, redirects to Swagger
2. Use **http://localhost:8000/docs** — all endpoints unlocked

### Browser (production)

1. Visit **http://localhost:8000/auth/login** — redirects to Google sign-in
2. After sign-in, redirected back to `/docs` with cookie set

### M2M / CLI

```bash
# Register a client (requires browser session first)
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

# Ingest a document
curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "report.txt",
    "text": "Company A owns Company B. Company B launched a rocket.",
    "priority": "high",
    "metadata": {}
  }'

# Query
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What did Company A launch?",
    "mode": "hybrid",
    "ground_truth": "Company A launched a rocket via its subsidiary Company B."
  }'

# KPI summary
curl http://localhost:8000/kpis/summary \
  -H "Authorization: Bearer <token>"
```

---

## Retrieval Pipeline

```
Query
  │
  ├─► LocalSearch
  │     ├─ Vector ANN on chunk embeddings (top-k)
  │     └─ Multi-hop graph traversal (depth=2)
  │         Chunk → MENTIONS → Entity → RELATES_TO* → Entity → MENTIONS → Chunk
  │
  ├─► GlobalSearch
  │     └─ Community embedding ANN → synthesized summaries
  │
  ├─► ContextBuilder (local 60% + global 40%)
  │
  ├─► Gemini generates answer
  │
  └─► Low confidence? (no citations / "I don't know")
        └─► AgenticRetriever (IRCoT loop, max 4 steps)
              ├─ Step N: LLM reasons → SEARCH: <sub-query>
              ├─ Re-search → add new chunks to context
              └─ Until → ANSWER: <final answer>
```

This solves the **cross-document reasoning** problem:
- *"Company A owns Company B"* (page 10, doc 1)
- *"Company B launched a rocket"* (page 300, doc 2)
- Query: *"What did Company A launch?"* → **correctly answered**

---

## RAGAS Metrics

| Metric | What it measures |
|--------|-----------------|
| `faithfulness` | Is the answer grounded in the retrieved context? |
| `answer_relevancy` | Does the answer address the actual question? |
| `context_precision` | Are the retrieved chunks actually relevant? |
| `context_recall` | Did retrieval capture all necessary information? |

Evaluation is sampled at **20%** of queries automatically. View results at:
- `GET /evaluation/summary`
- `GET /kpis/summary`
- **http://localhost:8050/dashboard**

---

## Configuration

All tuning is in `config/settings.yml`:

```yaml
retrieval:
  local_top_k: 10          # chunks retrieved by vector search
  multihop_depth: 2        # graph hops (2 = A→B→C)
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
| API + Swagger | http://localhost:8000/docs | dev-login |
| Neo4j Browser | http://localhost:7474 | neo4j / graphrag_dev |
| RabbitMQ UI | http://localhost:15672 | graphrag / graphrag_dev |
| Dashboard | http://localhost:8050/dashboard | — |

---

## License

MIT
