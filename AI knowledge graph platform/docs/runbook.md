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
   → Verify: SHOW INDEXES YIELD name, state — all 6 should be ONLINE

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

---

## 2. Health Checks

| Check | Command | Expected |
|---|---|---|
| API live | `curl http://localhost:8000/health` | `{"status":"ok"}` |
| API ready | `curl http://localhost:8000/health/ready` | `{"neo4j":"ok","redis":"ok"}` |
| Neo4j | Neo4j Browser → `:server status` | Connected |
| Schema indexes | `SHOW INDEXES YIELD name, state` | 6 indexes ONLINE |
| RabbitMQ | `curl -u graphrag:graphrag_dev http://localhost:15672/api/overview` | `{"object_totals":{...}}` |
| Redis | `redis-cli ping` | `PONG` |
| Queue depth | RabbitMQ UI → Queues tab | Near-zero for healthy throughput |
| Worker consuming | Worker logs | `rabbitmq.consuming` log line present |

---

## 3. Common Failure Patterns

### Query stuck at `status: queued`

**Cause A**: Worker not running.
Check: `ps aux | grep query_worker` / worker logs.
Fix: Restart `query_worker.py` or `combined_worker.py`.

**Cause B**: Redis not running — result store falls back to in-memory, worker and API can't share state.
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
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/kg/health/snapshot

# View alert history
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/kg/health/alerts
```

Alert thresholds in `config/settings.yml` → `business_matrix.alert_thresholds`:

| Metric | Default threshold | Meaning |
|---|---|---|
| `latency_p95_ms` | 3000ms | p95 query latency ceiling |
| `faithfulness` | 0.7 | RAGAS faithfulness floor |
| `context_recall` | 0.6 | RAGAS recall floor |
| `contradiction_rate` | 0.05 | conflicts per 1k edges |
| `orphan_rate` | 0.10 | fraction of entities with no chunk link |
| `low_confidence_rate` | 0.30 | fraction of edges below confidence floor |

### Prometheus metrics

```
GET http://localhost:8000/metrics
```

Instrumented via `prometheus-fastapi-instrumentator`. Covers HTTP request counts, latency histograms, and error rates per endpoint.

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

## 5. Backup & Restore

### Full graph backup (NDJSON)

```bash
# Backup to local directory
python scripts/kg_backup.py --tenant default --output backups/$(date +%Y%m%d)/

# Backup to S3
python scripts/kg_backup.py --tenant default --s3-bucket my-bucket --s3-prefix graphrag/

# Dry-run (count nodes/edges without writing)
python scripts/kg_backup.py --dry-run
```

Output: three NDJSON files per tenant — `nodes.ndjson`, `edges.ndjson`, `chunks.ndjson`.

### Restore

```bash
python scripts/kg_backup.py --restore --input backups/20260531/
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
3. For vector indexes: specify correct dimensions (3072 for Gemini `gemini-embedding-001`)

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

## 10. Key File Locations

| What | Where |
|---|---|
| Main config | `config/settings.yml` |
| Secrets | `.env` (never commit) |
| Neo4j schema | `graphrag/graph/schema.cypher` |
| Domain ontology | `config/ontologies/aerospace_regulatory.yml` |
| Worker logs | stdout / container logs |
| ADRs | `docs/adr/` |
| Lessons log | `tasks/lessons.md` |
| Graph health metrics | `GET /kg/health/snapshot` |
| Alert history | `GET /kg/health/alerts` |
