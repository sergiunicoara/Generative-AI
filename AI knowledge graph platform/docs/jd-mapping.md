# JD Mapping — GraphRAG Knowledge Platform

Every bullet from the target role (PwC Graph RAG / Knowledge Engineering) mapped
to a specific file, endpoint, demo step, or test — with a one-line business value.

> **How to use this in an interview:**  
> When a question maps to this doc, open the file immediately. "I can show you the code
> for that right now" is more credible than any verbal answer.

---

## Must-Have Requirements

### Neo4j + Cypher expertise in production environments

| Evidence | Where |
|---|---|
| 39 KG modules, 572-line client | `graphrag/graph/neo4j_client.py` |
| Vector ANN (3072d cosine search) | `neo4j_client.py → vector_search_chunks()` |
| BM25 fulltext search | `neo4j_client.py → fulltext_search()` |
| `UNWIND`, `EXISTS {}`, `COUNT {}`, `reduce()` | Search for these in `neo4j_client.py` |
| APOC with graceful fallback | `neo4j_client.py → _try_apoc_merge()` |
| Bitemporal `as_of(vt, tt)` queries | `neo4j_client.py → as_of_query()` |
| Cypher patterns (6 production patterns) | `docs/cypher-patterns.md` |
| Schema: indexes, constraints | `graphrag/graph/schema.cypher` |
| **Demo:** Live Neo4j with inferred edges | `py -3.11 scripts/demo_regulatory.py --live` |
| **Business value:** Multi-hop traversal finds connections a vector index cannot |

---

### Deep ontology / taxonomy modeling experience

| Evidence | Where |
|---|---|
| Versioned type taxonomy (`SUBCLASS_OF` hierarchy) | `graphrag/graph/type_taxonomy.py` |
| Domain/range constraints on every relation write | `graphrag/graph/ontology_registry.py` |
| Deprecated relation auto-migration | `ontology_registry.py → migrate_relation()` |
| Config-driven domain overlays (YAML) | `config/ontologies/aerospace_regulatory.yml` |
| OWL-RL reasoning (owlrl) | `graphrag/graph/owl_reasoner.py` |
| SPARQL 1.1 SELECT bridge | `graphrag/graph/sparql_bridge.py`, `POST /kg/sparql` |
| RDF/Turtle export with owl:NamedIndividual | `scripts/export_rdf.py` |
| **Demo:** Type hierarchy load, domain/range validation | Demo steps 1–2 |
| **Business value:** Domain knowledge is config, not code — new domains need no Python changes |

---

### Python engineering skills

| Evidence | Where |
|---|---|
| 22,650 lines, fully async (asyncio) | Entire `graphrag/` package |
| 353 passing unit tests | `tests/unit/` → `pytest tests/unit -q` |
| GitHub Actions CI (pytest matrix + ruff) | `.github/workflows/` |
| Docker multi-stage build | `Dockerfile` |
| One-command dev stack | `docker compose -f compose.dev.yaml up` |
| Stack health check | `make smoke-test` |
| Retry / backoff | `graphrag/core/retry.py` |
| Structured logging (structlog) | Every module |
| 6 ADRs documenting decisions | `docs/adr/` |
| **Business value:** Code is production-oriented: tested, observable, deployable |

---

### Knowledge graph integration with LLMs, RAG, or vector search

| Evidence | Where |
|---|---|
| 6-stage hybrid retrieval pipeline | `graphrag/retrieval/hybrid_retriever.py` |
| Vector ANN (OpenAI text-embedding-3-large 3072d) | `graphrag/ingestion/embedder.py` |
| BM25 + Reciprocal Rank Fusion | `hybrid_retriever.py → _rrf_merge()` |
| Cross-encoder reranking (ms-marco-MiniLM) | `graphrag/retrieval/cross_encoder_reranker.py` |
| GNN / GAT scoring | `graphrag/retrieval/gnn_scorer.py` |
| Multi-hop graph traversal | `hybrid_retriever.py → _multihop_expand()` |
| Agentic IRCoT (8B routing + 70B synthesis) | `graphrag/retrieval/agentic_retriever.py` |
| Redis-backed query result cache | `graphrag/retrieval/query_cache.py` → `graphrag/messaging/consumers.py` |
| RAGAS evaluation (auto-sampled 20%) | `graphrag/evaluation/ragas_evaluator.py` |
| Golden eval set (40 questions, regression runner) | `evals/golden_set.json`, `scripts/run_golden_eval.py` |
| **Demo:** Live query against real Neo4j with citations | `POST /query` → `GET /query/{id}` |
| **Business value:** 6-stage pipeline catches what single-stage systems miss |

---

### Experience balancing formal semantics with pragmatic implementation

| Evidence | Where |
|---|---|
| ADR-0001: Property graph over triple store | `docs/adr/0001-property-graph-over-triple-store.md` |
| ADR-0002: Forward-chaining over backward-chaining | `docs/adr/0002-forward-chaining-over-backward-chaining.md` |
| ADR-0003: Bayesian confidence accumulation | `docs/adr/0003-bayesian-confidence-accumulation.md` |
| ADR-0004: Groq over Gemini for generation | `docs/adr/0004-groq-over-gemini-for-text-generation.md` |
| ADR-0005: Redis as result store | `docs/adr/0005-redis-as-cross-process-result-store.md` |
| ADR-0006: Dual-LLM architecture | `docs/adr/0006-dual-llm-architecture.md` |
| OWL-RL alongside property graph | `graphrag/graph/owl_reasoner.py` + `sparql_bridge.py` |
| **Business value:** Can explain every design choice — not just what was built, but why |

---

### Ability to lead technically while remaining hands-on

| Evidence | Where |
|---|---|
| 6 ADRs with alternatives evaluated | `docs/adr/` |
| 82 documented lessons (pattern library) | `tasks/lessons.md` |
| CONTRIBUTING.md with ADR process, PR checklist | `CONTRIBUTING.md` |
| Production runbook (startup, health checks, failure modes) | `docs/runbook.md` |
| Scaling roadmap with decision thresholds | `docs/roadmap.md` |
| Agent tool safety layer (allowlist, scopes, audit) | `graphrag/agents/tool_policy.py` |
| **Business value:** Can onboard a team, not just deliver a prototype |

---

## Highly Valuable Bonus Areas

### RDF / OWL

| Evidence | Where |
|---|---|
| OWL-RL reasoning (`owlrl`) — subClassOf, symmetric, inverse | `graphrag/graph/owl_reasoner.py` |
| SPARQL 1.1 SELECT (in-process over Turtle export) | `graphrag/graph/sparql_bridge.py` |
| RDF/Turtle serialisation with `owl:NamedIndividual` | `scripts/export_rdf.py` |
| `is_consistent()` — owl:Nothing entailment check | `owl_reasoner.py → is_consistent()` |

---

### Inference engines

| Evidence | Where |
|---|---|
| Datalog forward-chaining: transitivity, symmetry, inverse, composition | `graphrag/inference/forward_chaining.py` |
| Fixpoint iteration | `forward_chaining.py → run_to_fixpoint()` |
| Confidence decay per hop (`0.95^n`) | `forward_chaining.py → _decay_confidence()` |
| Provenance: derived edges tagged `source_type=inferred` | `graphrag/graph/neo4j_client.py → write_inferred_edge()` |
| **Demo:** `FAA-AD-2024 → FAA-AD-2020` derived edge, confidence 0.857 | Demo step 4 |

---

### Entity resolution

| Evidence | Where |
|---|---|
| 4-stage pipeline: exact → fuzzy → embedding (≥0.92) → queue | `graphrag/graph/alias_registry.py` |
| Redis-backed cross-worker sharing | `alias_registry.py → load_alias_registry()` |
| Embedding deduplication (cosine similarity) | `alias_registry.py → find_duplicate_by_embedding()` |
| Entity splitter (over-merged entities) | `graphrag/graph/entity_splitter.py` |
| Wikidata external grounding | `graphrag/graph/entity_linker.py` → enabled by `WIKIDATA_LINKING=1` |

---

### Legal / Regulatory data

| Evidence | Where |
|---|---|
| Aerospace regulatory ontology (28 type pairs, 12 relation rules) | `config/ontologies/aerospace_regulatory.yml` |
| Document authority hierarchy (4 levels) | `graphrag/graph/document_authority.py` |
| SUPERSEDES chain + transitive inference | Demo step 4 |
| GDPR Article 17 right-to-be-forgotten | `POST /kg/gdpr/forget-entity` |
| PII guard | `graphrag/ingestion/pii_guard.py` |
| Contradiction detection (5 conflict types) | `graphrag/graph/contradiction_detector.py` |
| **Demo:** Airworthy vs unairworthy conflict surface | Demo step 5 |

---

## Operational / Production Evidence

| Claim | Proof |
|---|---|
| "Not a tutorial project" | 353 tests, CI, Docker build, structured DLQ, 6 ADRs, 82 documented lessons |
| "Observable" | 16 metrics, RAGAS 0.840/0.907/0.867 from 104 real runs, calibration pipeline wired, admin + KPI dashboards |
| "Deployable" | `docker compose -f compose.dev.yaml up`; worker `/ready` health probes |
| "Scalable design" | Redis alias registry, incremental community detection, RabbitMQ parallel workers |
| "Regulated-client ready" | GDPR, multi-tenant isolation, audit trail, contradiction detection |
| "Day-one delivery" | `py -3.11 scripts/ingest_corpus.py --commit` → full real corpus in Neo4j |

---

## Live Demo Sequence (15 minutes)

| Minute | Action | Evidence shown |
|---|---|---|
| 0–1 | Open slide deck | Context |
| 1–2 | Slide 2: the problem | Why this matters |
| 2–4 | Slide 3: architecture | Three-layer overview |
| 4–5 | Slide 4: JD mapping | Every requirement covered |
| 5–10 | **Terminal: `demo_regulatory.py --live`** | Forward-chaining, contradiction detection, live Neo4j |
| 10–12 | **Browser: admin dashboard `/admin/`** | Health metrics, conflicts, calibration |
| 12–13 | Slide 7: technical foundation | 353 tests, 6 ADRs, smoke-test |
| 13–14 | Slide 8: client scenarios | Regulatory, audit, compliance |
| 14–15 | Slide 9: close + questions | |

---

## Quick-reference: Key numbers

| Metric | Value |
|---|---|
| Faithfulness (RAGAS) | 0.840 |
| Context precision | 0.907 |
| Context recall | 0.867 |
| Hybrid p95 | 2.2s |
| Agentic p95 | 3.4s (8B routing + 70B synthesis) |
| Agentic trigger rate | ~9% of queries |
| Entities (real corpus) | 374 (LLM-extracted from 12-doc corpus, after alias dedup) |
| Relations (real corpus) | 456 (asserted + 10 forward-chain inferred) |
| Open conflicts detected | 70 (contradiction detector, verified on real data) |
| Relation confidence | 99.6% edges ≥ 0.75 (live Neo4j snapshot) |
| Alias coverage | 14.7% entities with aliases; 600+→374 canonical (~38% reduction) |
| Orphan rate | 0.0% (all entities linked to source chunks) |
| Community coherence | 90% (39 Leiden communities, real corpus) |
| Contradiction rate | 153.51 / 1k edges (adversarial corpus — expected high) |
| Calibration (Brier) | pipeline wired; target < 0.20 on production corpus |
| Passing tests | 353 |
| KG modules | 39 |
| ADRs | 6 |
| Lessons | 82 |
| LOC | 22,650 |
