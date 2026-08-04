# Sales Context Graph

CRM data (Salesforce-shaped) + sales-call transcripts (Gong-shaped) + content
engagement (Showpad-shaped) → a tenant-isolated Neo4j knowledge graph →
evidence-backed, query-specific context for one sales workflow:

> Given an opportunity, identify the objection raised by a stakeholder in the
> latest relevant call and recommend an appropriate content asset the buyer
> hasn't already viewed, with exact evidence and an explainable
> entity-resolution decision.

## Status

The full P0–P4.5 vertical slice described in [`docs/plan.md`](docs/plan.md) is
implemented and tested. 171 tests pass (unit, integration against a live
Neo4j, security, and eval), run in 8 increments — see the completion report at
the end of this document for the phase-by-phase breakdown, real measured
numbers, and known limitations.

## Architecture

```mermaid
flowchart TB
    subgraph Sources
        SF[Salesforce-shaped CRM export]
        GONG[Gong-shaped call transcripts]
        SP[Showpad-shaped content + views]
    end

    subgraph Ingestion["src/ingestion/"]
        SFA[SalesforceAdapter]
        GA[GongAdapter]
        SPA[ShowpadAdapter]
        RECON[reconciliation.py<br/>identical/changed/deleted]
    end

    subgraph Extraction["src/extraction/"]
        WIN[windowing.py]
        FIX[FixtureExtractionProvider]
        LLM[LlmExtractionProvider]
        PROMPT[prompt.py<br/>injection-resistant]
    end

    subgraph Resolution["src/resolution/ + src/review/"]
        DET[Stage A<br/>deterministic]
        CAND[candidate generation<br/>exact/fulltext/vector/relational]
        SCORE[scoring.py<br/>lexical+semantic+relational]
        POLICY[policy.py<br/>AUTO_LINKED/PENDING_REVIEW/UNRESOLVED]
        REVIEW[ReviewService<br/>targeted reconciliation]
    end

    subgraph Graph["Neo4j (src/graph/)"]
        EXEC[GraphExecutor<br/>tenant_query / schema_query / operational_query]
        REPO[Repositories<br/>Account, Contact, Opportunity, Conversation,<br/>Claim, Mention, ContentAsset, SourceRecord]
    end

    subgraph Serving["src/context_graph/ + src/usecases/"]
        CTX[ContextGraphBuilder<br/>scope → score → budget → diversity]
        UC[ObjectionContentRecommendationUseCase]
    end

    API[FastAPI — api/]

    SF --> SFA --> RECON
    GONG --> GA --> WIN --> FIX & LLM
    SP --> SPA --> RECON
    PROMPT -.delimits transcript.-> LLM
    FIX & LLM --> RECON
    RECON --> EXEC
    DET --> POLICY
    CAND --> SCORE --> POLICY
    POLICY --> REVIEW
    EXEC --> REPO --> Graph
    REPO --> CTX --> UC
    API --> Ingestion
    API --> Resolution
    API --> Serving
```

Every node carries `workspace_id` (the tenant-isolation boundary); Showpad
nodes additionally carry `division_id`. `GraphExecutor.tenant_query()`
structurally rejects Cypher that doesn't scope a matched node by `workspace_id`
— see [`docs/security-and-tenancy.md`](docs/security-and-tenancy.md).

## Setup

Requires Docker and Python 3.12+ (developed/tested against 3.11.6 — see
"Known limitations" below).

```bash
docker compose up -d neo4j
pip install -r requirements.txt
cp .env.example .env
```

`docker-compose.yml`'s `neo4j` service publishes on host ports **7475/7688**,
not Neo4j's defaults (7474/7687) — see the comment in that file for why.

## Running the tests

```bash
make test-unit          # no Neo4j required
make test-integration   # brings up neo4j, then runs integration/eval/security suites
make test                # everything
```

## Running the demo

```bash
make demo
```

Runs [`demo_volkswagen.py`](demo_volkswagen.py) end to end: seeds Volkswagen
Group + a Volkswagen Financial Services distractor, resolves a "Volks Wagen"
transcript mention (printing every candidate, each component score, the named
relational signals, the top-1/top-2 margin, and the final decision), then runs
the objection-to-content recommendation use case and prints the recommended
(unviewed) asset with its exact transcript evidence.

## Running the API

```bash
uvicorn api.main:app --reload
```

```bash
# health / readiness
curl localhost:8000/health
curl localhost:8000/ready

# ingest CRM data (workspace_id comes from the header, never the body — §13)
curl -X POST localhost:8000/api/v1/ingestions/crm \
  -H "X-Workspace-Id: ws-demo" -H "Content-Type: application/json" \
  -d '{"accounts": [{"Id": "001x", "Name": "Acme Corp", "Website": "acme.com", "IsDeleted": false}]}'

# check ingestion status
curl localhost:8000/api/v1/ingestions/<ingestion_id> -H "X-Workspace-Id: ws-demo"

# ingest a transcript (email_to_contact_id/email_to_seller_id are optional —
# omitted here, so every speaker resolves to speaker_role=UNKNOWN; pass them
# to get real BUYER/SELLER resolution, as demo_volkswagen.py does)
curl -X POST localhost:8000/api/v1/ingestions/transcripts \
  -H "X-Workspace-Id: ws-demo" -H "Content-Type: application/json" \
  -d @data/sample/gong_call.json

# list mentions awaiting human review
curl localhost:8000/api/v1/unresolved-mentions -H "X-Workspace-Id: ws-demo"

# resolve one
curl -X POST localhost:8000/api/v1/unresolved-mentions/<mention_id>/resolve \
  -H "X-Workspace-Id: ws-demo" -H "Content-Type: application/json" \
  -d '{"reviewer_id": "reviewer@example.com", "selected_entity_id": "<account_id>"}'

# build a context graph for a subject
curl -X POST localhost:8000/api/v1/context/build \
  -H "X-Workspace-Id: ws-demo" -H "Content-Type: application/json" \
  -d '{"subject_id": "<contact_id>"}'

# fetch a claim's exact evidence
curl localhost:8000/api/v1/claims/<claim_id>/evidence -H "X-Workspace-Id: ws-demo"
```

Sample request payloads for every endpoint live under
[`data/sample/`](data/sample/).

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — module layout and data flow
- [`docs/ontology.md`](docs/ontology.md) — the canonical domain model
- [`docs/entity-resolution.md`](docs/entity-resolution.md) — the implemented
  Stage A / candidate-generation / scoring / policy algorithm, with the real
  calibration data behind the thresholds
- [`docs/security-and-tenancy.md`](docs/security-and-tenancy.md) — tenant
  isolation mechanism and what's explicitly *not* production-authorized yet
- [`docs/evaluation.md`](docs/evaluation.md) — metric definitions and real
  measured results from this repo's own test runs
- [`docs/plan.md`](docs/plan.md) — the original authoritative spec this
  implementation follows

## What's ported from `ai-knowledge-graph-platform`

Eight modules under `src/graph/` were forked from a sibling project (a
different-domain GraphRAG platform) rather than built from scratch, per that
decision's own record — see each file's header comment for provenance. They
are kept working (Increment 1's `tests/unit/graph_legacy/` suite) as generic
tenant-scoped graph infrastructure, but operate on a different node shape
(`Entity`/`Statement`/`tenant`) than this repo's sales-specific model
(`Account`/`Claim`/`workspace_id`) — they are not called by the P1+
repositories built on the new model. See
[`docs/architecture.md`](docs/architecture.md) for the full reuse-vs-rewrite
breakdown per module.

## Known limitations

- **Python version mismatch**: `pyproject.toml` declares `requires-python
  >=3.12`; this repo was developed and tested against the locally available
  3.11.6. No 3.12-only syntax is used, but this hasn't been verified on 3.12
  itself.
- **No real packaging**: imports resolve via `pythonpath = ["."]`
  (`pyproject.toml`) and `PYTHONPATH=/app` (`Dockerfile`), not an installable
  package — a deliberate Increment 1 decision, documented in `src/core/config.py`'s
  header, to avoid packaging-metadata risk before there's a reason to publish
  this as a package.
- **Embedding provider**: local `sentence-transformers` (`all-MiniLM-L6-v2`,
  384-dim, no API key — `src/embedding/`), wired into `resolve_mention()`'s
  semantic scoring. The versioned vector index (`contact_embeddings_v1`) still
  exists structurally but runs unpopulated — `vector_candidates()` (candidate
  *generation* via the index) is a separate, larger milestone (embedding-on-
  write for every Contact, backfill) from semantic *scoring* of already-
  generated candidates, which is what's wired today. See
  `docs/entity-resolution.md` for the real measured calibration
  (`DEFAULT_LEXICAL_WEIGHT=0.97`) this choice drove.
- **In-process ingestion job store** (`api/state.py`) does not survive a
  process restart — proven, not just described, by
  `tests/unit/api/test_ingestion_store.py`. The interface is shaped so a
  durable (Redis/Postgres) implementation can replace it without changing any
  caller.
- **No authentication/identity provider** — `X-Workspace-Id` is a trusted
  header standing in for real auth (`api/dependencies.py`). See
  `docs/security-and-tenancy.md`.
