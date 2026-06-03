# Platform Roadmap & Scaling Strategy

Current architecture decisions, known scale limits, and the engineering roadmap
for production hardening and capability expansion.

---

## Current State (Baseline)

### What works today

Status wording in this section is a capability baseline, not a hiring claim that
the system has already handled real customer traffic. In interviews, describe
these as implemented, demo-ready, and production-oriented unless there is a
deployed workload and monitoring data behind the claim.

| Capability | Status | Notes |
|---|---|---|
| Graph ingestion (doc → chunk → entity → relation) | ✅ Implemented / demo-ready | Groq extraction, Gemini 3072d embeddings |
| 6-stage hybrid retrieval (vector + BM25 + reranker + GNN + multi-hop + LLM) | ✅ Implemented / demo-ready | |
| Agentic IRCoT fallback (two-model) | ✅ Implemented / demo-ready | 2-step max; 8B routing + 70B synthesis |
| Forward-chaining inference (transitivity, symmetry, inverse, composition) | ✅ Implemented / demo-ready | Post-ingestion fixpoint |
| OWL-RL reasoning over RDF export | ✅ Implemented / demo-ready | owlrl + rdflib |
| SPARQL bridge (in-process over Turtle export) | ✅ Implemented / demo-ready | SPARQL 1.1 SELECT |
| TransE link prediction | ✅ Implemented / demo-ready | entity embeddings as input |
| Entity resolution (4-stage: exact/fuzzy/embedding/queue) | ✅ Implemented / demo-ready | Cosine threshold 0.92 |
| Contradiction detection (5 types) | ✅ Implemented / demo-ready | |
| Document authority hierarchy + SUPERSEDES chains | ✅ Implemented / demo-ready | |
| Multi-tenant isolation | ✅ Implemented / demo-ready | `(name, type, tenant)` key |
| Leiden community detection (multi-resolution) | ✅ Implemented / demo-ready | graspologic |
| RAGAS evaluation (20% sampling) | ✅ Implemented / demo-ready | Groq judge (Gemini fallback) |
| OAuth 2.0 (Google browser + M2M JWT) | ✅ Implemented / demo-ready | |
| GDPR erasure | ✅ Implemented / demo-ready | cascade + audit log |
| Domain ontologies (YAML-configurable) | ✅ Implemented / demo-ready | Aerospace regulatory demo |
| GitHub Actions CI | ✅ | pytest matrix + ruff lint |

### Known scale limits

| Limit | Current | Hits at |
|---|---|---|
| Ingestion throughput | Sequential per-document | ~20 docs/min on a single worker |
| Alias resolution | In-memory dict per process | ~500k entities before RAM pressure |
| Community rebuild | Full graph per tenant | Slow beyond ~100k entities; incremental builder available |
| Result store TTL | 1 hour | Fine for interactive queries; increase for batch pipelines |
| Groq free tier | 1500 RPD / 6000 RPM | Sufficient for dev; upgrade for high-volume ingestion |
| Vector index | Neo4j native (cosine, 3072d) | Scales to ~10M chunks on adequately specced Neo4j |

---

## Open Implementation Candidates

Completed items are intentionally excluded from this list. These are the
remaining roadmap items most worth implementing for hiring signal, JD alignment,
and technical credibility.

| Priority | Candidate | Recommendation | Why it matters |
|---|---|---|---|
| 1 | **Agent tool safety layer** | Implement next | Directly matches agentic chatbot roles: allowlisted tools, scoped permissions, argument validation, timeouts, dry-run mode, denied-action behavior, and audit logs. |
| 2 | **Golden GraphRAG evaluation set** | Implement next | Proves retrieval quality with regression tests instead of relying on a polished demo. Include expected citations, answer traits, and pass/fail thresholds. |
| 3 | **JD mapping page** | Implement next | Converts the project into hiring evidence by mapping job requirements to files, endpoints, demo steps, and business value. |
| 4 | **End-to-end demo video** | Implement next | Reduces recruiter and CTO review friction. Show ingest, inference, agentic query, citations, and dashboard metrics in 2-3 minutes. |
| 5 | **Re-ranking feedback loop** | Worth implementing | Adds product maturity: citation clicks, thumbs up/down, or expand events can become implicit relevance data for future tuning. |
| 6 | **GNN pre-training scheduled job** | Worth implementing | Makes the existing GNN work feel operational, not experimental. Trigger after large ingestion batches or on a maintenance schedule. |
| 7 | **TimescaleDB KPI store** | Implement only if pitching observability hard | Useful for enterprise monitoring, but larger than it looks because compose/dev config and migration paths must be added. |
| 8 | **Multi-modal extraction pipeline** | Defer unless targeting multi-modal roles | Storage exists, but OCR/visual embedding/audio ingestion expands scope away from the core GraphRAG hiring story. |
| 9 | **Cross-encoder fine-tuning** | Defer | Hard to prove without real usage data and domain-labeled query/chunk pairs. |
| 10 | **Kafka streaming ingestion** | Defer | Interesting scale path, but RabbitMQ already supports the current MVP story. Kafka would add operational complexity before it adds hiring value. |

### Recommended Next Sprint

- [x] **Tool execution policy** — `graphrag/agents/tool_policy.py`: allowlist, per-tool scopes, arg validation, timeout, dry-run mode, denied-action records, audit trail with summary.
- [x] **Tool-call guardrail tests** — `tests/unit/test_tool_safety.py`: 21 tests across 7 classes (allowlist, scopes, arg validation, cross-tenant, destructive, dry-run, timeout). Suite: 325 passing.
- [x] **Golden GraphRAG eval set** — `evals/golden_set.json`: 40 questions (single-hop, multi-hop, contradiction, authority-chain, inference, calibration); `scripts/run_golden_eval.py`: regression runner, pass/fail thresholds, per-type breakdown.
- [x] **JD mapping artifact** — `docs/jd-mapping.md`: every JD bullet mapped to file + endpoint + demo step + business value; quick-reference numbers table; live demo sequence.
- [ ] **Demo video checklist** - define the exact 2-3 minute walkthrough sequence and the data/query set it uses.

---

## Near-Term (Next 1–2 Sprints)

### P0 — Production hardening

- [ ] **Parallel ingestion workers** — multiple `ingestion_worker.py` instances; RabbitMQ `prefetch_count=1` set; `compose.dev.yaml` starts a single instance; scale by adding replicas
- [x] **Worker health endpoint** — `GET /ready` and `GET /live` on configurable port (`WORKER_HEALTH_PORT`); `graphrag/workers/health_server.py` (aiohttp); wired to ingestion, query, evaluation workers; compose.dev.yaml health checks use it
- [x] **Query result TTL** — `QUERY_RESULT_TTL_SECONDS` env var (alias: `GRAPHRAG_RESULT_TTL`); documented in compose.dev.yaml; no redeploy needed
- [x] **Structured error responses** — DLQ messages now carry: `exception_type`, `error`, `retry_count`, `queue`, `message_id`, `payload_summary`; retry headers include `x-last-error` and `x-exception-type`

### P1 — Developer experience

- [x] **`make smoke-test`** — unit tests + mock demo + API import check; exits 0/1
- [x] **Local Docker Compose profile** — `compose.dev.yaml`; one command starts Neo4j + RabbitMQ + Redis + API + 3 workers + 2 dashboards; health checks wired to worker `/ready` endpoints
- [x] **Seed data script** — `scripts/seed_demo_data.py`; 20 entities, 12 relations, 2 conflict pairs, health + calibration snapshots; `--commit`, `--wipe`, `--tenant` flags; idempotent MERGE
- [x] **Real corpus ingestion** — `scripts/ingest_corpus.py`; full LLM extraction pipeline on 12-doc aerospace corpus; 374 entities, 456 edges, 70 conflicts in Neo4j (2026-06-03)
- [x] **Graph health snapshot fixed** — `coalesce(e.quarantined, false)` null-safe pattern in `graph_snapshots.py` + `graph_evaluator.py`; entity_count now shows 374 correctly
- [x] **Degree anomaly threshold** — raised `MAX_DEGREE_MULTIPLIER` 5→20 in `ingestion_validator.py`; prevents hub entities (FAA, Boeing) from being wrongly quarantined in sparse domain graphs
- [x] **Dashboard live mode** — JWT token with `read write` scope; `GRAPHRAG_DEFAULT_TENANT` env var; `GET /kg/snapshots` response wrapped in `{"snapshots":[]}`; all 5 tabs verified live against real aerospace data

---

## Medium-Term (1–3 Months)

### Retrieval quality

- [ ] **GNN pre-training** — `scripts/calibrate_gnn.py` ✅ exists (7.4 KB); needs wiring into a scheduled job that re-trains after each large ingestion batch
- [ ] **Re-ranking feedback loop** — store which citations users follow (click/expand signal from dashboard) and use as implicit relevance signal for future fine-tuning
- [x] **Query result caching** — `QueryConsumer` now checks `QueryCache` before dispatching; Redis-backed with provenance-aware invalidation on new ingests
- [ ] **Cross-encoder fine-tuning** — fine-tune `ms-marco-MiniLM-L-6-v2` on domain-specific query/chunk pairs gathered from RAGAS evals

### Graph quality

- [x] **Incremental community detection** — `IncrementalCommunityDetector` wired to `/kg/incremental-community/rebuild-affected` and `/kg/incremental-community/summary` API routes; dashboard Communities tab triggers it
- [x] **Wikidata linking** — wired into `IngestionAgent.run()` as optional post-write step; enabled via `WIKIDATA_LINKING=1` (default off); caps at 20 entities/doc to respect rate limits; `wikidata_links` count in job return dict
- [x] **Property schema enforcement** — `GET /kg/health/property-violations` endpoint added; admin dashboard Health tab shows violation count and per-type breakdown

### Operational

- [ ] **TimescaleDB KPI store** — KPI store lives at `graphrag/business_matrix/kpi_store.py` (SQLite); docker-compose does **not** currently define a TimescaleDB service — needs adding before the flip is possible
- [ ] **Multi-modal entities** — `graphrag/graph/multimodal.py` ✅ exists (12.4 KB) for storage; extraction pipeline (OCR + visual embeddings) **not yet built**
- [x] **Multi-worker alias registry** — `AliasRegistry.load()` pushes alias table to Redis hash (`graphrag:aliases:{tenant}`, 24h TTL); `load_alias_registry()` tries Redis warm-load first; workers share state without full Neo4j scan on startup

---

## Long-Term Vision (3–12 Months)

### Scale path

**Write throughput**: The bottleneck is entity resolution (embedding comparison) and Neo4j MERGE contention under parallel writes. Path: shard by tenant into separate Neo4j databases (Neo4j 5.x multi-db); dedicate one alias-resolution worker per tenant shard.

**Read latency**: Add Neo4j read replicas for vector ANN and BM25 queries. The write primary handles ingestion; the replica cluster serves retrieval. Redis query cache absorbs repeat queries.

**Community rebuild**: Already has incremental builder; for very large graphs (1M+ entities), partition Leiden runs by sub-graph (e.g. per document cluster or entity type subtree).

### Capability expansions

- **Streaming ingestion** — replace RabbitMQ with Kafka for event-sourced ingestion at higher throughput; maintain RabbitMQ as a simpler option for small deployments
- **Graph-native reranking** — embed the full RELATES_TO subgraph (not just entity embeddings) as a GNN input; move from static entity embeddings to dynamic multi-hop pooled representations
- **Multi-modal entities** — `graphrag/graph/multimodal.py` provides storage; add extraction pipeline for images (OCR + visual embeddings) and audio (transcript + speaker embeddings)
- **Cross-tenant federated queries** — allow read-only federation across tenants with explicit permission grants for cross-domain regulatory queries (e.g. ITAR across suppliers)
- **Domain-specific embedding models** — replace `gemini-embedding-001` with a fine-tuned domain model (e.g. BioBERT for healthcare, LegalBERT for regulatory) without changing the 3072d index if dimensions match, or via embedding registry migration otherwise

### ADRs to write

The following decisions are pending documentation:

| Decision | Status |
|---|---|
| Groq over Gemini for text generation | ✅ ADR-0004 written |
| Redis as cross-process result store | ✅ ADR-0005 written |
| Dual LLM architecture (8B routing + 70B synthesis) | ✅ ADR-0006 written |
| Session context enrichment strategy (pre-embed, not pre-LLM) | Documented in lessons.md (A03); upgrade to ADR |
| Multi-hop depth 2 default | Documented in lessons.md (A13); upgrade to ADR |

---

## Scaling Decision Reference

### When to add a worker

Add another `ingestion_worker.py` when:
- Ingestion queue depth consistently > 100 messages
- Ingestion lag > 5 minutes under normal load
- p95 ingestion latency > 30 seconds

### When to upgrade Groq tier

- Free tier: 1500 RPD / 6000 RPM — adequate for development and low-volume pipelines
- Upgrade when: ingestion rate > 1000 docs/day, or RAGAS evaluation introduces > 300 RPD additional calls

### When to switch to Neo4j Enterprise

- Parallel writes from multiple workers saturate single-writer Neo4j (check `dbms.connector.bolt.thread_pool_max_size`)
- Need read replicas for query scaling
- Require online backup without downtime

### When to add a TimescaleDB continuous aggregate

- KPI query `GET /kpis/timeseries` exceeds 100ms
- Dashboard refresh causes noticeable lag under load
- Raw KPI table exceeds ~5M rows

