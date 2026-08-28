# Operational Runbook

Day-2 operations guide: startup order, monitoring, common failure patterns,
backup/restore, schema migrations, and on-call response.

---

## 1. Service Startup Order

Services have hard dependencies. Start in this order:

```
1. Infrastructure (no deps)
   docker-compose up neo4j rabbitmq redis

2. Schema initialisation (requires Neo4j)
   python scripts/init_neo4j.py
   → Wait for: "schema_ready" in output
   → Verify: `SHOW INDEXES YIELD name, state` returns every index as ONLINE

3. API (requires RabbitMQ + Redis)
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   → Verify: GET /health → {"status":"ok"}
   → Verify: GET /health/ready → {"neo4j":"ok","redis":"ok"}

4. Workers (require RabbitMQ + Neo4j + Redis)
   python workers/ingestion_worker.py   # or combined_worker.py
   python workers/query_worker.py
   → Verify: logs show "rabbitmq.consuming" for each queue

5. Dashboard (optional, requires API)
   python graphrag/business_matrix/dashboard_server.py
   → Verify: http://localhost:8050
```

**Single-machine shortcut** (dev/testing):
```bash
python workers/combined_worker.py   # runs ingestion + query in one process
```

### Scaling ingestion workers

Ingestion workers are competing consumers of the same durable RabbitMQ queue.
RabbitMQ is configured with `prefetch_count=1`, so one worker does not reserve a
batch of documents ahead of the others. Scale the development stack with:

```bash
docker compose -f compose.dev.yaml up --scale ingestion_worker=3
```

The ingestion service deliberately has no fixed `container_name` or host port;
its `/ready` probe remains available on port 8081 inside each container. In a
production deployment, add replicas when queue depth stays above 100, ingestion
lag exceeds 5 minutes, or p95 ingestion latency exceeds 30 seconds. Watch
Neo4j write contention and entity-resolution latency before increasing further.

---

## 2. Health Checks

| Check | Command | Expected |
|---|---|---|
| API live | `curl http://localhost:8000/health` | `{"status":"ok"}` |
| API ready | `curl http://localhost:8000/health/ready` | `{"neo4j":"ok","redis":"ok"}` (returns HTTP 503 if either is down) |
| Neo4j | Neo4j Browser → `:server status` | Connected |
| Schema indexes | `SHOW INDEXES YIELD name, state` | Every listed index is ONLINE |
| RabbitMQ | `curl -u graphrag:graphrag_dev http://localhost:15672/api/overview` | `{"object_totals":{...}}` |
| Redis | `redis-cli ping` | `PONG` |
| Queue depth | RabbitMQ UI → Queues tab | Near-zero for healthy throughput |
| Worker consuming | Worker logs | `rabbitmq.consuming` log line present |

---

## 3. Common Failure Patterns

### Retrieval reports insufficient evidence

`retrieval_sufficiency` is enabled by default and records evidence count,
source count, average score, conflicts, and a reason code. This is telemetry,
not by itself a service failure. Inspect the query trace and
`retrieval_sufficiency.reason_code` before changing thresholds.

The default behavior can escalate evidence shortages through the existing
agentic fallback; it does not hard-abstain. Enable
`retrieval_sufficiency_abstain_enabled` only after running
`research/linkedin_group/benchmark_plan.md` and setting tenant-appropriate
minimum evidence and score values.

### Enabling experimental retrieval controls

Conservative defaults are:

```yaml
retrieval:
  adaptive_traversal_enabled: false
  evidence_fusion_enabled: false
  retrieval_sufficiency_enabled: true
  retrieval_sufficiency_abstain_enabled: false
```

Evaluate one feature at a time against the same corpus and golden queries.
Compare answer quality, citation recall, latency, and escalation/abstention
rates. Unit-test success is not live performance validation.

### Query stuck at `status: queued`

**Cause A**: Worker not running.
Check: `ps aux | grep query_worker` / worker logs.
Fix: Restart `query_worker.py` or `combined_worker.py`.

**Cause B**: Redis not running. The result store no longer silently falls back to
an in-memory dict on a mid-operation Redis failure — it logs an ERROR and drops
the write/read instead of pretending it succeeded, so a broken Redis surfaces as
a stuck query rather than a hidden split-brain.
Check: `redis-cli ping`.
Fix: `docker-compose up redis`, restart workers and API.

**Cause C**: Message in DLQ after 3 failed retries.
Check: RabbitMQ UI → `graphrag.query.queue.dlq` depth > 0.
Fix: Investigate error in worker logs; fix root cause; re-publish from DLQ or re-submit query.

---

### Vector index not found (`chunk_embeddings`)

**Cause**: Schema not initialised or DDL silently skipped.
Fix: Run `python scripts/init_neo4j.py`. Verify with `SHOW INDEXES`.
Note: If running in a container, ensure Neo4j is fully started before init (wait for port 7687).

---

### Worker crashes with `No module named 'groq'` / `No module named 'redis'`

Fix: `pip install groq "redis[asyncio]"` in the worker's environment.
Permanent: these are now in `requirements.txt`; use `pip install -r requirements.txt`.

---

### `session_store_unavailable` at API startup

**Cause**: Redis unreachable with `session_store_strict: true` in `settings.yml`.
Fix A (dev): Set `session_store_strict: false` — API falls back to in-memory sessions.
Fix B (prod): Restore Redis connectivity; sessions and query results will be lost until Redis is back.

### Live E2E tests are skipped

### Deterministic full test collection

Use the repository launcher for the complete suite. It disables unrelated
auto-discovered plugins and explicitly enables `pytest-asyncio`, avoiding
desktop-plugin startup variance while preserving async test behavior:

```bash
python scripts/run_pytest.py --collect-only -q tests/
python scripts/run_pytest.py -q tests/
```

The launcher is also used by the `make test*` targets. Third-party plugins can
still be enabled explicitly when a test requires them.

### Local Kubernetes manifest validation

With Docker Desktop and Minikube available, validate the manifests against a
real Kubernetes API server:

```powershell
.\scripts\validate_minikube.ps1
```

The check creates or reuses a local `graphrag-local` profile, renders the
Kustomize tree, and runs a server-side dry-run. It validates admission and API
compatibility without pretending that placeholder production images or cloud
secrets are deployable locally.

For a real local MCP pod smoke test, use the disposable-secret workflow:

```powershell
.\scripts\run_minikube_smoke.ps1
```

It loads the existing local API/MCP images into Minikube, creates only
local-only credentials, scales the background workers to zero, waits for MCP
readiness, and checks `/health` through a port-forward.

The five Docker-backed service tests in `tests/e2e/test_live_services.py`
require Docker Desktop and `testcontainers-python`. Install project
dependencies and verify Docker before running them:

```bash
python -m pip install -r requirements.txt
docker version
python -m pytest -q tests/e2e/test_live_services.py
```

The tests start isolated Neo4j and Redis containers and clean them up after the
test classes finish. On Windows, use `python -m pytest`; the standalone
`pytest` executable may not put the repository root on `sys.path`, which can
break imports from `mcp_server` and `scripts`.

---

### `UnicodeEncodeError` running scripts on Windows

Cause: Terminal uses cp1252, script prints box-drawing characters (═ ─ ✓).
Fix: `set PYTHONIOENCODING=utf-8` before running, or use Windows Terminal (UTF-8 by default).
Long-term: `scripts/demo_regulatory.py` already calls `sys.stdout.reconfigure(encoding="utf-8")`.

---

### High latency (`latency_ms` > 5000)

Common causes:
1. **Reranker cold start**: first query loads `ms-marco-MiniLM-L-6-v2` (~105 weights). Subsequent queries fast.
2. **Groq rate limit**: free tier bursts; 429 errors in worker logs. Retry with backoff is automatic.
3. **Neo4j full scan**: missing index on a hot query path. Check `EXPLAIN` / `PROFILE` on slow Cypher.
4. **Community not built**: global search ANN finds no communities → skip. Build with `python scripts/community_rebuild.py --tenant default`.

---

## 4. Monitoring & Alerting

### Graph health metrics (leading indicators)

```bash
# Check current health snapshot
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/kg/snapshots

# View alert history
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/kg/health/alerts
```

Alert thresholds in `config/settings.yml` → `business_matrix.alert_thresholds`:

| Metric | Default threshold | Meaning |
|---|---|---|
| `latency_p95_ms` | 30000ms | p95 query latency ceiling — raised from 3000 to sit above the measured 26.4s baseline, so it fires on regressions rather than continuously |
| `faithfulness` | 0.8 | RAGAS faithfulness floor |
| `context_recall` | 0.6 | RAGAS recall floor |
| `contradiction_rate` | 0.05 | conflicts per 1k edges |
| `orphan_rate` | 0.10 | fraction of entities with no chunk link |
| `low_confidence_rate` | 0.30 | fraction of edges below confidence floor |

### Prometheus metrics

```
GET http://localhost:8000/metrics
```

Instrumented via `prometheus-fastapi-instrumentator`. Covers HTTP request
counts, latency histograms, and error rates per endpoint. Stage cost/latency
metrics are emitted by `cost_attribution.py`; MCP capability, deterministic
skill-router, and evaluation-job counters are emitted by
`agent_telemetry.py`. Tenant attribution stays in structured logs rather than
metric labels to prevent cardinality-driven monitoring failures. Set
`OTEL_EXPORTER_OTLP_ENDPOINT` to export API and worker spans; preserve
`X-Correlation-ID` when opening an incident so HTTP, RabbitMQ, result-store,
and Context Graph records can be joined without using it as a metric label.

### Dashboards

Two operator dashboards share one branded design system (navy/teal, Inter,
status-coloured KPI tiles, gauges, branded charts):

**Admin / Observability** — mounted on the API at `/admin` (do not run the
standalone `python graphrag/dashboard/app.py`; Dash static assets 404 under a
bare Flask server — always serve it via the API):

```
uvicorn api.main:app --host 0.0.0.0 --port 8000
→ http://localhost:8000/admin/
```

Tabs: Graph Health (gauges + contradiction trend) | Conflicts | Communities |
GDPR | Calibration. Live data requires Neo4j + the ingestion pipeline.

**Business Matrix** — query-level KPIs from the local SQLite store:

```
python graphrag/business_matrix/dashboard_server.py
→ http://localhost:8050/dashboard/
```

#### Demo mode (no backend)

To show either dashboard fully populated for a walkthrough or screenshots
**without** a running Neo4j / ingestion pipeline, set `GRAPHRAG_DASHBOARD_DEMO=1`.
When set, each admin tab falls back to representative sample data (mirroring the
healthy thresholds in `performance-metrics-inventory.md`) **only if** the live
API is unreachable. Unset in production — real data or a real error panel is
always shown otherwise.

```bash
# Windows (PowerShell)
$env:GRAPHRAG_DASHBOARD_DEMO = "1"
uvicorn api.main:app --port 8001
# → http://localhost:8001/admin/  — all tabs populated
```

Sample payloads live in `graphrag/dashboard/demo_data.py`.

---

## Context Graph operations

Context Graph reads and writes are exposed under `/context-graph` and require
the normal `read` or `write` scope. Useful checks include:

```bash
# Validate a P0 trace
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/context-graph/traces/validate?tenant=default"
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/context-graph/wpp/campaign-placement"
```

Replay, correction, approval, exception, action, outcome, feedback, precedent,
redaction-marker, and proactive policy-expiry operations are available through
the Context Graph API. Feedback that names an outcome is accepted only when
that outcome was produced by an action of the same decision; precedent scores
include observed outcome state and feedback tied through `ASSESSES`. Run
`scripts/demo_context_graph_outcomes.py` for the audited vertical slice.

## MCP gateway operations

Remote MCP uses authenticated Streamable HTTP at `/mcp`; `/metrics` requires a
Bearer token and `/health` is the only unauthenticated probe. The full local
verification, Kubernetes exposure checklist, and incident playbook are in
[`mcp-operations.md`](mcp-operations.md). Do not expose the internal service
directly or turn the production NetworkPolicy example into a blanket egress
allow rule.

### Optional TimescaleDB KPI backend

SQLite remains the default local backend. For durable time-series KPIs, set:

```powershell
$env:KPI_BACKEND = "timescale"
$env:TIMESCALE_DB_URL = "postgresql+asyncpg://user:password@host:5432/graphrag"
```

Initialize the hypertable through the `TimescaleKPIStore` startup path before
starting the dashboard or KPI workers. Keep SQLite for demos and isolated
development environments.

### Multimodal provenance

Media bytes stay in object storage. Attachments are linked in Neo4j through
`MediaAttachment`; OCR, transcript, caption, and visual-embedding outputs are
recorded as `SourceArtifact` nodes via
`MultiModalEntityService.record_transformation()`. Store the model version and
output digest with every transformation so derived evidence can be audited.

## 5. Backup & Restore

### Full graph backup (NDJSON)

`kg_backup.py` takes a **required subcommand** — `backup`, `restore` or `list`.
Every command in this section previously omitted it and died at argparse before
doing anything; `--s3-bucket`, `--s3-prefix` and `--dry-run` never existed. S3 is
addressed through the output path, not separate flags. Verify with
`python scripts/kg_backup.py --help`.

```bash
# Backup to a local directory
python scripts/kg_backup.py backup --tenant default --output backups/$(date +%Y%m%d)/

# Backup to S3
python scripts/kg_backup.py backup --tenant default --output s3://my-bucket/graphrag/

# List existing backups (local path or s3:// prefix)
python scripts/kg_backup.py list --output backups/
```

Output: three NDJSON files per tenant — `nodes.ndjson`, `edges.ndjson`, `chunks.ndjson`.

Or via make: `make backup TENANT=default` / `make backup-s3 TENANT=default S3_BUCKET=my-bucket`.

### Restore

```bash
python scripts/kg_backup.py restore --input backups/20260531/ --tenant default
```

⚠️ Restore does **not** wipe existing data — it merges (idempotent). To wipe and restore:
```cypher
-- In Neo4j Browser:
MATCH (n) DETACH DELETE n
-- Then run restore
```

### Schema backup

The schema is idempotent. Re-run `scripts/init_neo4j.py` to recreate indexes and constraints after any database wipe.

---

## 6. Schema Migrations

### Adding a new index or constraint

1. Add the Cypher statement to `graphrag/graph/schema.cypher`
2. Run `python scripts/init_neo4j.py` — all statements are idempotent (`IF NOT EXISTS`)
3. For vector indexes: specify correct dimensions (3072 for OpenAI `text-embedding-3-large`)

### Neo4j 2026 vector-index upgrade

The compatibility stack remains usable on Neo4j 5.20. To validate or deploy
Neo4j 2026.06 without attaching the old store, start the separate-volume
override:

```bash
docker compose -f docker-compose.yml -f compose.neo4j-modern.yaml up -d neo4j
python scripts/init_neo4j.py
```

On a fresh 2026 database, schema initialization creates filterable indexes
directly. For a separately backed-up and deliberately upgraded existing
database, inspect and then rebuild only the vector indexes:

```bash
python scripts/migrate_neo4j_vector_indexes.py
python scripts/migrate_neo4j_vector_indexes.py --apply
```

Verify `SHOW VECTOR INDEXES YIELD name, state, properties, indexProvider`.
`chunk_embeddings` and `community_embeddings` must be `ONLINE`, use a
`vector-2026.*` provider, and include `tenant` in `properties`. The application
detects this at startup; otherwise it keeps the 5.20 over-fetch fallback.

### Renaming an entity type across the graph

```bash
python scripts/entity_type_migration.py --old-type EXEC --new-type PERSON --tenant default --dry-run
python scripts/entity_type_migration.py --old-type EXEC --new-type PERSON --tenant default
```

This cascades to: Entity nodes, WikidataLink, Statement, RELATES_TO src_type/tgt_type, audit trail.

### Re-embedding after model change

When switching embedding models (different dimensions):
1. Update `GEMINI_EMBED_MODEL` in `.env`
2. Create a new vector index with the correct dimensions: `CREATE VECTOR INDEX chunk_embeddings_v2 ...`
3. Run `python scripts/re_embed.py --tenant default --batch-size 50`
4. Once complete, drop the old index and rename

---

## 7. Community Rebuild

Leiden communities power global search. Rebuild when:
- Staleness score > 0.15 (automatic if `auto_rebuild_communities: true`)
- Manual import of a large document batch
- First-time setup (no communities exist)

```bash
# Check staleness, rebuild if needed
python scripts/community_rebuild.py --tenant default

# Force rebuild regardless of staleness
python scripts/community_rebuild.py --tenant default --force

# Dry-run: report staleness without rebuilding
python scripts/community_rebuild.py --tenant default --dry-run
```

Without communities, `global_search.no_communities` warning appears in logs and global search returns empty context.

---

## 8. Secrets Rotation

### Rotating the JWT signing key

1. Generate new key: `python -c "import secrets; print(secrets.token_hex(32))"`
2. Update `JWT_SECRET_KEY` in `.env`
3. Restart API — **all existing tokens are immediately invalidated**
4. Users must re-authenticate (browser: new login; M2M clients: new `/auth/token` request)

Session cookies use `SESSION_SECRET_KEY` (separate from JWT). Rotating JWT does not affect browser sessions, and vice versa.

### Rotating Groq API key

1. Update `GROQ_API_KEY` in `.env`
2. Restart all workers and the API (they load `.env` at startup via `python-dotenv`)

### Rotating Neo4j password

1. Change password via Neo4j Browser: `:server change-password`
2. Update `NEO4J_PASSWORD` in `.env`
3. Restart all services that hold Neo4j connections (API, workers)

---

## 9. On-Call Decision Tree

```
User reports: "I submitted a query and it never completed"
│
├─ GET /query/{id} returns status=completed?
│  └─ YES → Client-side polling bug; check client code
│
├─ Worker logs show "rabbitmq.consuming"?
│  └─ NO → Worker not running; restart workers
│
├─ Worker logs show "rabbitmq.handler_error"?
│  └─ YES → Check error message:
│        "No module named groq" → pip install groq
│        "No module named redis" → pip install redis[asyncio]
│        Neo4j error → check Neo4j health
│        Groq 429 → rate limit; wait or upgrade tier
│
├─ Message in DLQ (3 retries exhausted)?
│  └─ YES → Root cause in logs; fix; re-submit query
│
└─ Redis down?
   └─ YES → Start Redis; restart workers (result store needs Redis for cross-process sharing)
```

---

## 10. Resource and Shutdown Safety

- `API_MAX_REQUEST_BYTES` defaults to 10 MiB. Oversized requests receive 413;
  malformed or prematurely terminated bodies receive 400. Keep the reverse
  proxy limit at or below this value so rejected bodies are not buffered twice.
- Query modes are `local`, `global`, or `hybrid`; ingestion priorities are
  `normal` or `high`. A 422 for another value is an intentional routing guard.
- The first RabbitMQ connection declares all durable exchanges, work queues,
  DLQs, TTLs, and bindings. API startup therefore makes publishing safe even
  when workers are not running yet. Alert on publish failures and DLQ growth.
- A failed query publish removes its Redis `queued` marker. A 503 from submit
  means the client may retry with its own idempotency policy.
- Graceful API/worker shutdown closes RabbitMQ, Neo4j, Redis stores, health
  servers, and the tracing provider. Give containers enough termination grace
  for in-flight handlers and telemetry flush; force-kill only after that window.
- Remote MCP rejects any supplied `Origin` not listed exactly in
  `GRAPHRAG_MCP_ALLOWED_ORIGINS`. Local execution binds to `127.0.0.1` by
  default; containers must explicitly bind `0.0.0.0` behind their network and
  ingress controls.

---

## 11. Key File Locations

| What | Where |
|---|---|
| Main config | `config/settings.yml` |
| Secrets | `.env` (never commit) |
| Neo4j schema | `graphrag/graph/schema.cypher` |
| Domain ontology | `config/ontologies/aerospace_regulatory.yml` |
| Worker logs | stdout / container logs |
| ADRs | `docs/adr/` |
| Lessons log | `tasks/lessons.md` |
| Graph health metrics | `GET /kg/snapshots` |
| Alert history | `GET /kg/health/alerts` |
