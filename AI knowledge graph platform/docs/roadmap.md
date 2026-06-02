# Platform Roadmap & Scaling Strategy

Current architecture decisions, known scale limits, and the engineering roadmap
for production hardening and capability expansion.

---

## Current State (Baseline)

### What works today

| Capability | Status | Notes |
|---|---|---|
| Graph ingestion (doc → chunk → entity → relation) | ✅ Production | Groq extraction, Gemini 3072d embeddings |
| 6-stage hybrid retrieval (vector + BM25 + reranker + GNN + multi-hop + LLM) | ✅ Production | |
| Agentic IRCoT fallback (two-model) | ✅ Production | 2-step max; 8B routing + 70B synthesis |
| Forward-chaining inference (transitivity, symmetry, inverse, composition) | ✅ Production | Post-ingestion fixpoint |
| OWL-RL reasoning over RDF export | ✅ Production | owlrl + rdflib |
| SPARQL bridge (in-process over Turtle export) | ✅ Production | SPARQL 1.1 SELECT |
| TransE link prediction | ✅ Production | entity embeddings as input |
| Entity resolution (4-stage: exact/fuzzy/embedding/queue) | ✅ Production | Cosine threshold 0.92 |
| Contradiction detection (5 types) | ✅ Production | |
| Document authority hierarchy + SUPERSEDES chains | ✅ Production | |
| Multi-tenant isolation | ✅ Production | `(name, type, tenant)` key |
| Leiden community detection (multi-resolution) | ✅ Production | graspologic |
| RAGAS evaluation (20% sampling) | ✅ Production | Groq judge (Gemini fallback) |
| OAuth 2.0 (Google browser + M2M JWT) | ✅ Production | |
| GDPR erasure | ✅ Production | cascade + audit log |
| Domain ontologies (YAML-configurable) | ✅ Production | Aerospace regulatory demo |
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

## Near-Term (Next 1–2 Sprints)

### P0 — Production hardening

- [ ] **Parallel ingestion workers** — multiple `ingestion_worker.py` instances consuming from the same queue (RabbitMQ `prefetch_count=1` already set; just run N copies — not yet containerised as a multi-replica service)
- [ ] **Worker health endpoint** — expose a minimal HTTP `GET /ready` from each worker so Kubernetes/Fly.io knows when to route traffic (not yet implemented)
- [ ] **Query result TTL** — make `query_result_ttl_seconds` configurable via env var (currently YAML only; ops needs fast override without deploy)
- [ ] **Structured error responses** — DLQ messages should carry the original exception class, not just the message string, for automated triage

### P1 — Developer experience

- [ ] **`make smoke-test`** — Makefile exists but has no smoke-test target; wrap the full ingest → query → poll cycle using a pre-baked test document
- [ ] **Local Docker Compose profile** — `compose.dev.yaml` does not yet exist; contributors need four terminal windows to run the stack
- [ ] **Seed data script** — `scripts/seed_demo_data.py` does not yet exist; `scripts/demo_regulatory.py --live` seeds a small demo corpus but is not a general-purpose seed tool

---

## Medium-Term (1–3 Months)

### Retrieval quality

- [ ] **GNN pre-training** — `scripts/calibrate_gnn.py` ✅ exists (7.4 KB); needs wiring into a scheduled job that re-trains after each large ingestion batch
- [ ] **Re-ranking feedback loop** — store which citations users follow (click/expand signal from dashboard) and use as implicit relevance signal for future fine-tuning
- [ ] **Query result caching** — `graphrag/retrieval/query_cache.py` ✅ exists (10 KB, Redis-backed, provenance-aware invalidation); **not yet wired** into `QueryConsumer` — still dispatches every query to a worker
- [ ] **Cross-encoder fine-tuning** — fine-tune `ms-marco-MiniLM-L-6-v2` on domain-specific query/chunk pairs gathered from RAGAS evals

### Graph quality

- [ ] **Incremental community detection** — `graphrag/graph/incremental_community.py` ✅ exists (`IncrementalCommunityDetector`, 16.9 KB); `community_manager.py` does **not** yet call it — still does full rebuild for all tenants
- [ ] **Wikidata linking** — `graphrag/graph/entity_linker.py` ✅ exists (13.2 KB) but runs ad-hoc; **not yet wired** into the ingestion pipeline
- [ ] **Property schema enforcement** — `graphrag/graph/property_schema.py` ✅ exists (11.7 KB); `GET /kg/health/property-violations` endpoint and admin dashboard tile **not yet wired**

### Operational

- [ ] **TimescaleDB KPI store** — KPI store lives at `graphrag/business_matrix/kpi_store.py` (SQLite); docker-compose does **not** currently define a TimescaleDB service — needs adding before the flip is possible
- [ ] **Multi-modal entities** — `graphrag/graph/multimodal.py` ✅ exists (12.4 KB) for storage; extraction pipeline (OCR + visual embeddings) **not yet built**
- [ ] **Multi-worker alias registry** — persist alias tables to Redis so multiple ingestion workers share deduplication state without Neo4j round-trips

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
| Groq over Gemini for text generation | Needs ADR (current implementation lacks rationale doc) |
| Redis as cross-process result store | Needs ADR (current implementation lacks rationale doc) |
| Dual LLM architecture (generation vs. embeddings) | Needs ADR |
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
