# Platform Roadmap & Scaling Strategy

This roadmap separates two distinct engineering goals:

1. **Improve the existing Knowledge Graph platform** until it is reliable,
   scalable, measurable, and production-ready.
2. **Extend the platform into a Context Graph for AI** by adding persistent
   decision context, policy evaluation, execution history, outcomes, and
   reusable organizational precedent.

Status wording distinguishes three evidence levels: **implemented and
unit-tested**, **implemented and wired**, and **live-validated**. Claims only
move to a stronger level when the corresponding test or live exercise exists.

---

## Current State (Baseline)

### Engineering workflow automation

The repository now includes a local specification-to-implementation runner in
`graphrag/engineering_workflows/`. It supports declarative task DAGs, an
allowlisted skill registry, injectable specialist-agent handlers, lifecycle
hooks, atomic persisted run state, bounded no-shell command execution, and an
explicit approval gate before Git mutations. The CLI is executable through
`scripts/run_engineering_workflow.py`; autonomous code generation and native
Codex UI commands remain outside the repository boundary.

### What works today

Status wording in this section is a capability baseline, not a hiring claim that
the system has already handled real customer traffic. In interviews, describe
these as implemented, demo-ready, and production-oriented unless there is a
deployed workload and monitoring data behind the claim.

| Capability | Notes |
|---|---|
| Graph ingestion (document → chunk → entity → relation) | Groq extraction by default (`get_llm()`), with DeepSeek fallback; `LLM_INGEST_PROVIDER=deepseek`/`cerebras` opt-in overrides; OpenAI `text-embedding-3-large`, 3072 dimensions |
| LLM provider circuit breaker | Fail-fast after 3 consecutive failures or an 80% error rate over the last 20 calls; the default `FallbackLLM` chain is Groq → DeepSeek; surfaced on `/health/ready` |
| Six-stage hybrid retrieval | Vector + BM25 + reranker + GNN + multi-hop + LLM synthesis; high-level community summaries are retrieved directly into final synthesis (legacy map-reduce remains an ablation fallback) |
| Agentic IRCoT fallback | Bounded four-step maximum from configuration; Groq fast-model routing + Groq large-model synthesis (DeepSeek fallback) |
| Forward-chaining inference | Transitivity, symmetry, inverse, and composition to fixpoint after ingestion |
| OWL-RL reasoning | `owlrl` + `rdflib` over RDF export |
| SPARQL bridge | In-process SPARQL 1.1 SELECT over Turtle export |
| TransE link prediction | Entity embeddings as input; tenant-starvation-safe ANN candidate pool (see Recent Hardening) |
| Four-stage entity resolution | Exact, fuzzy, embedding, and human review; cosine threshold 0.92; ambiguous matches are queued through `/kg/review-queue`; embedding-search ANN pool is tenant-starvation-safe |
| Contradiction detection | `directional_reversal`, `exclusive_state`, `functional_violation`, and `positive_negative_pair`; retrieval-side conflict warnings |
| Authority and supersession | Document authority hierarchy and `SUPERSEDES` chains |
| Explicit document-link topology | HTML/Markdown/SharePoint references become tenant-scoped, ACL-aware `LINKS_TO` edges with provenance and bounded retrieval traversal |
| Context-scoped entity representations | Source-system assertions are retained below canonical entities as `SystemRepresentation` and `ContextualAssertion` nodes |
| Temporal and provenance model | Valid time, transaction time, snapshots, extraction model, prompt version, spans, and source type; API retrieval can constrain chunk and graph traversal by valid and transaction time |
| Multi-tenant isolation | `(name, type, tenant)` identity key; agent-tool layer (`ToolPolicy`) enforces tenant scoping on both read and write/restricted tools (see Recent Hardening) |
| Community detection | Multi-resolution Leiden via `graspologic` |
| Evaluation | RAGAS with 20% sampling; DeepSeek judge, Groq fallback, then Gemini compatibility fallback |
| Authentication and privacy | OAuth 2.0, M2M JWT, GDPR erasure, cascade handling, and audit log |
| Domain ontologies | YAML-configurable; aerospace regulatory, automotive IATF 16949 (30-doc corpus, 5-question golden set), and marketing/adtech domains |
| Synthetic commercial-pharma demo | Tenant-scoped synthetic corpus, YAML ontology, RDF/OWL and SHACL structural validation, and deterministic commercial-content approval; intentionally excludes clinical decision support and Context Graph |
| CI | GitHub Actions, pytest matrix, and Ruff linting |
| Retrieval feedback | `graphrag/retrieval/feedback.py` — Neo4j-backed feedback capture, wired into `/feedback` API routes |
| Confidence lifecycle & evidence tracking | `graphrag/graph/confidence_lifecycle.py`, `graphrag/graph/evidence.py` — real state-machine + audit trail, wired into `/kg/confidence` routes |

### Known scale limits

### Technology evaluation boundaries

Neo4j remains the production system of record. The platform's graph
interoperability and second-backend work is deliberately evaluation-led:

| Area | Current decision | Promotion condition |
|---|---|---|
| Graph analytics | Neo4j GDS PageRank is implemented | Evaluate additional read-only GDS workloads before adoption. |
| RDF / SKOS | RDF export and SKOS vocabularies are implemented | Review external-source mappings before creating auditable equivalence links. |
| GQL | Not a production runtime dependency | Keep the bounded read-query contract for conformance tests. |
| Ultipa / other graph backends | Not integrated | Consider only a same-dataset, read-only benchmark. |
| PostgreSQL SQL/PGQ | Track as a possible lightweight deployment tier | Consider only after a standards-compliant release is available and the same-dataset read-only benchmark meets the second-backend gate. |

Second-backend results must use the same dataset fingerprint, scenario, and
query count as Neo4j. They require at least 99.9% result equivalence, tenant
isolation, 20% lower p95 latency, 25% higher throughput, and no more than 20%
higher cost before a limited read-only pilot is considered. This is a design
review gate, not production approval.

| Limit | Current | Expected pressure point |
|---|---|---|
| Ingestion throughput | Sequential per document | Approximately 20 documents/minute on one worker |
| Alias resolution | In-memory dictionary per process | Approximately 500,000 entities before memory pressure |
| Community rebuild | Full graph per tenant | Slow beyond approximately 100,000 entities; incremental builder exists |
| Result-store TTL | One hour | Appropriate for interactive queries; insufficient for some batch pipelines |
| Groq free tier | 1,500 RPD / 6,000 RPM | Gates fast routing and default Groq synthesis; DeepSeek is the fallback |
| Vector index | 3072d cosine; Neo4j 5.20 over-fetch fallback and validated Neo4j 2026.06 in-index tenant filtering | Recall/load testing is still required before claiming a 10M-chunk operating point |

### Current performance baseline

Measured across 44 automotive and aerospace queries (`HybridRetriever`, live
Neo4j + LLM, no shortcuts):

- p50: **13.2 seconds**
- p95: **26.4 seconds**
- mean: **15.2 seconds**

The remaining cost is distributed across model and retrieval round trips
(query rewrite, embedding, local retrieval, and final synthesis). The former
per-community map/reduce path is retained only for measured fallback/ablation.

### Recent hardening (2026-07-29 - 2026-08-03, A143-A162)

A live latency and security investigation, fully documented in
`tasks/lessons.md`:

- **A143** — `chunk_entities_edges` was paying Bolt-deserialization cost for
  3072-dim embeddings on every query; fixed with an entity-keyed cache
  (`graphrag/graph/embedding_cache.py`), 26x faster on a warm cache.
- **A144** — `global_search.reduce` was an unbounded LLM call, usually
  reformatting a single extraction nobody needed reformatted; short-circuited
  when there's only one partial answer, capped output otherwise.
- **A145** — `local_search` and `global_search` ran sequentially despite
  having no data dependency; parallelized via `asyncio.TaskGroup`.
- **A146** — `vector_search_communities`/`vector_search_chunks` ran Neo4j ANN
  search globally across all tenants before filtering by tenant, so a small
  top-k could starve a tenant's own relevant results out entirely (this is
  what caused the `global_search.no_communities` warning — automotive had
  200 real Community nodes, not zero). Fixed with an over-fetch-then-filter
  pattern (`fetch_k = max(top_k*20, 100)`), live-verified.
- **A147** — `ToolPolicy`'s five low/medium-risk read tools (`local_search`,
  `global_search`, `get_neighbors`, `search_graph`, `get_community`) never
  received the `tenant` value the policy layer already had, so they silently
  ran as `tenant="default"` (the established wildcard = all tenants) — a
  real, currently-reachable cross-tenant data leak on the agent-tool
  surface. Fixed by forcing the policy-level tenant onto any tool that
  doesn't declare `tenant` in its own schema. Live-verified: an
  automotive-only entity returns 6 neighbors scoped to `tenant="automotive"`,
  0 scoped to `tenant="aerospace"`.
- **A148** — Same ANN-starvation bug class as A146, in
  `link_predictor.py`/`alias_registry.py`'s entity-embedding search. Fixed
  after benchmarking the ingestion-loop cost first (k=5 vs k=100 measured
  statistically identical at current entity counts — ~31-47ms both ways).
- **A149** — `ToolPolicy`'s write/restricted tools (`ingest_document`,
  `quarantine_entity`, `erase_entity`) silently trusted any tenant an agent
  claimed in `args` whenever the caller held no `tenant:X` scope — a
  deliberately tested "unrestricted caller" design, closed because it's
  unsafe the moment those scopes are ever handed to an LLM agent rather than
  a human/ops process.
- **A150** — Replaced the query/session-only answer key with a governed,
  canonical SHA-256 key over tenant, normalized question, durable corpus
  revision, effective retrieval configuration, model route, prompt, and
  ontology version. Cache lookup now lives in `HybridRetriever` and hits keep
  a fresh `query_id` while exposing the original trace.
- **A151** — Cache entries are written only after Context Graph trace
  persistence and require a trace ID plus citations; a cache hit exposes the
  original `source_query_id` and `source_trace_id`.
- **A152** — Cache identity is the full inference context, not question text:
  tenant, normalized question, corpus revision, retrieval modes, model route,
  prompt, retrieval configuration, and ontology version.
- **A153** — Retrieval-visible mutations bracket writes with
  `KGCorpusState.active_updates`; cache reads fail closed until the final
  concurrent mutation publishes an atomic new revision.
- **A154** - Global retrieval now uses direct, token-bounded community
  summaries in the final synthesis context, eliminating per-community map LLM
  calls. The old `map_reduce` strategy remains configuration-gated for A/B.
- **A155** - `valid_at` and `transaction_at` are carried from `POST /query`
  through the worker, ANN/BM25 candidates, multi-hop traversal, cache identity,
  and Context Graph trace configuration. Community summaries are now append-only
  `CommunitySummarySnapshot` versions with evidence lineage, so temporal hybrid
  and global retrieval no longer have to fall back to local-only retrieval.
- **A156** - Reproducible local benchmark runners now exist for retrieval
  ablation (`scripts/benchmark_retrieval_ablation.py`) and per-document
  incremental-ingestion/maintenance cost
  (`scripts/benchmark_incremental_ingestion.py`). No performance delta is
  claimed until a named live run produces its report.
- **A157** - Neo4j 2026.06 support is capability-gated. Fresh modern deployments
  create filterable vector indexes and use Cypher `SEARCH` with tenant filtering
  inside the ANN index; Neo4j 5.20 keeps the tested over-fetch fallback. The
  modern image, index provider, properties, and tenant-isolated query were
  live-validated on a separate volume. Existing 5.20 volumes are not upgraded
  implicitly.
- **A158** - Community identities are deterministic and each generated summary
  creates an immutable bitemporal snapshot carrying chunk/document IDs,
  versions, a canonical content hash, and `SUPPORTED_BY`/`DERIVED_FROM` links.
- **A159** - The keyword planner is now the cold-start policy for a measured
  adaptive router. Per-tenant/query-class/mode EWMA quality and latency select
  routes after minimum sample counts; deterministic bounded exploration avoids
  starving untried routes. No production win is claimed until traffic supplies
  representative observations.
- **A160** - Retrieval policy is executable structured data, not a hard-coded
  `ALLOW`: ordered typed conditions produce `allow`, `deny`, or `escalate` with
  a matched rule and reason code. Governed traces also persist user/agent
  `CGEpisode` nodes and can load durable session memory tenant-safely.
- **A161** - A tenant-scoped source catalog and connector protocol now version
  ingestion mappings, reject credential-shaped mapping data, and link
  cataloged documents through `INGESTED_FROM`.
- **A162** - `X-Correlation-ID` and W3C trace context propagate from FastAPI
  through RabbitMQ to query workers, results, structured cost logs, and
  `CGAgentRun`. Prometheus dependencies are explicit and OTLP export activates
  when `OTEL_EXPORTER_OTLP_ENDPOINT` is configured.
- **A163** - Docker-backed service E2E coverage is executable through
  `testcontainers-python`: five tests now exercise real Neo4j and Redis
  persistence/connectivity with isolated containers. The tests are opt-in when
  Docker is unavailable and use the current `testcontainers.community`
  namespaces without deprecation warnings.
- Alert threshold `latency_p95_ms` raised from 3000 to 30000 to match
  measured reality with headroom, rather than firing continuously.

The ANN-starvation siblings and chunk-entity discovery path are now closed.
`scripts/calibrate_gnn.py` remains intentionally single-tenant and documents
that constraint; it is not an unscoped multi-tenant application query.

---

# Part I — Improve the Existing Knowledge Graph Platform

## Objective

Strengthen the platform that already exists. This part covers ontology quality,
ingestion, retrieval, graph reasoning, evaluation, security, observability,
scalability, and developer experience.

It does **not** introduce decision traces, approvals, actions, outcomes, or
organizational precedent. Those belong to Part II.

### Exit claim

After completing the material Part I items, the project may credibly be
described as:

> A production-oriented Enterprise Knowledge Graph and GraphRAG platform with
> ontology management, temporal reasoning, provenance, hybrid retrieval,
> evidence-based generation, and policy-controlled access.

## Part I status

### Core foundation — implemented and wired

1. Versioned ontology and schema validation (`graphrag/graph/domain_ontology.py`
   — real semver + migration-map validation, `graphrag/graph/ontology_migration.py`
   diff logic).
2. Statement-, source-, chunk-, and model-level provenance.
3. Valid-time and transaction-time reconstruction (`graphrag/graph/bitemporal.py`
   — three real as-of Cypher query methods, wired into snapshot retrieval).
4. Stable hybrid GraphRAG retrieval and evaluation (see performance baseline
   above; recent hardening closed both the largest latency costs and a real
   tenant-isolation gap).
5. Tenant-safe graph mutations, versioning, supersession, and audit history.

### Additional implemented capabilities

| Module | Status | Detail |
|---|---|---|
| Retrieval feedback (`graphrag/retrieval/feedback.py`) | **Implemented and wired** | Neo4j-backed feedback capture is exposed at `/feedback`; `HybridRetriever` now reads tenant-scoped aggregate signals in one batched lookup and blends them conservatively into final chunk scores. The call site and score blending are tested and fail open. |
| Document-link topology (`document_loader.py`, `neo4j_client.py`) | **Implemented and wired** | Explicit HTML/Markdown/SharePoint references persist as provenance- and ACL-bearing `LINKS_TO` edges; unresolved targets reconcile later, stale links are removed on re-ingestion, and LocalSearch performs bounded authorised expansion. |
| Contextual entity representations (`graph_writer.py`, `neo4j_client.py`) | **Implemented and wired** | Tenant/source-system `SystemRepresentation` nodes and chunk-backed `ContextualAssertion` paths preserve CRM/ERP context beneath canonical entities. |
| Evidence tracking (`graphrag/graph/evidence.py`) | **Implemented and wired** | Real `Evidence`/`SourceArtifact` Cypher writes, wired into `/kg/confidence` routes. |
| Confidence lifecycle (`graphrag/graph/confidence_lifecycle.py`) | **Implemented and wired** | Real enum-guarded state machine (`ASSERTED/INFERRED/DISPUTED/RETRACTED/APPROVED`) with an audit `ConfidenceTransition` node per transition, wired into `/kg/confidence`. |
| GNN calibration scheduler (`graphrag/graph/calibration_scheduler.py`) | **Implemented and wired** | Triggered from the RabbitMQ ingestion consumer; records scheduled/running/completed/failed states and launches `scripts/calibrate_gnn.py`. Runner injection keeps unit tests deterministic. |
| TimescaleDB KPI store (`graphrag/business_matrix/timescale_kpi_store.py`) | **Implemented and live-validated** | Provisioned in both Compose stacks, selected through `KPI_BACKEND`/`TIMESCALE_DB_URL`, and verified with a live initialize/write/read cutover. |
| Ontology migration diffing (`graphrag/graph/ontology_migration.py`) | **Implemented and wired** | Added/removed/renamed-class diff logic is applied through `OntologyRegistry.apply_ontology_migration` and `/kg/ontology/migration`. |
| Adaptive query router (`query_planner.py`, `adaptive_router.py`) | **Implemented and wired** | The planner supplies cold-start routing; measured tenant/query-class/mode EWMA quality and latency take over after bounded sample gates, with deterministic exploration and fail-open fallback. |
| Domain eval harness (`graphrag/evaluation/domain_eval.py`) | **Implemented, wired to a script only** | Used by `scripts/validate_eval_datasets.py`; not part of the running application. |
| Observability (`graphrag/observability/`) | **Implemented and wired** | Prometheus cost/latency and budget metrics are exposed at `/metrics`; HTTP/RabbitMQ/worker correlation is preserved, W3C trace context is propagated, and optional OTLP spans activate from environment configuration. |
| Source catalog (`graphrag/graph/source_catalog.py`) | **Implemented and API-wired** | `/kg/sources` owns tenant-scoped sources and immutable mapping versions; connector implementations use a provider-neutral protocol and credentials are prohibited from persisted mappings. |
| Local relational-to-KG ingestion (`graphrag/ingestion/relational.py`) | **Implemented and live-validated** | Read-only SQLite and PostgreSQL adapters share declarative entity/relation mappings; an in-memory SHACL candidate-batch gate rejects invalid imports before any Neo4j write. Imports retain deterministic source-document provenance. `tests/e2e/test_relational_postgres_neo4j.py` writes synthetic PostgreSQL data and reads it back through Neo4j. The synthetic sustainability demo also runs the PostgreSQL-to-MCP evidence-gap path. R2RML conversion and same-tenant OBDA federation are now implemented and unit-tested in `graphrag/ingestion/r2rml.py`; live external federation remains an evidence task. |
| Controlled MCP graph facts (`graphrag/graph/controlled_query.py`) | **Implemented and MCP-wired** | `query_graph_facts_tool` maps a small allowlist of natural-language fact intents, including supplier evidence gaps, to fixed read-only Cypher templates. It is parameterized, tenant-scoped, result-bounded and rejects raw Cypher or unsupported questions. |
| Synthetic supply-chain entity-resolution benchmark | **Implemented and repeatable** | `scripts/benchmark_sustainability_entity_resolution.py` runs seven synthetic name variants through the real alias registry: automatic matches, an ambiguous case quarantined for review, and a new-entity case. It is a threshold-regression check, not a production accuracy claim. |
| PROV-O RDF alignment (`scripts/export_rdf.py`) | **Implemented** | Exported entities and reified assertions retain the existing platform annotations and add standard `prov:wasDerivedFrom` and `prov:generatedAtTime` links when provenance is available. |
| Ops exercises (`graphrag/ops/`, `scripts/run_production_exercises.py`) | **Implemented and executable** | Load, security, backup/restore digest, and cost exercises have a CLI and deterministic tests. Results still describe the environment in which the command was run; they are not evidence of customer-scale traffic. |

### Evaluation and controlled route benchmarking

`graphrag.evaluation.graphrag_benchmark` accepts GraphRAG-Benchmark-compatible
JSONL (`question`/`query` plus an ID), preserves unknown task fields, and emits
dataset and route fingerprints with measured latency, answers, and citations.
Run identical tenant questions through named profiles with
`scripts/run_graphrag_benchmark.py`; official leaderboard scoring remains an
external, version-pinned evaluation step. The configured RAGAS backend falls
back to the deterministic `reference` evaluator when upstream imports, calls,
or timeouts fail, and persists the score source explicitly.

Fuzz/property checks cover prompt policy, tenant, protocol, and mapping
boundaries. `make mutation` is an opt-in Mutmut target for the high-risk
adapters; a complete mutation score still requires CI execution.

## Part I long-term scale path (3–12 months)

### Write throughput

Entity resolution and Neo4j `MERGE` contention are the likely bottlenecks under
parallel writes. The scale path is tenant-aware sharding, one alias-resolution
worker per shard, batched writes, idempotent ingestion, and backpressure.

### Read latency

Use Neo4j read replicas for vector ANN, BM25, and graph traversal. Keep the
write primary focused on ingestion and mutation. Use Redis only for bounded,
version-aware cache entries so stale graph context cannot silently survive
ontology or policy changes.

### Community rebuild

Continue incremental community updates. For graphs beyond approximately one
million entities, partition Leiden processing by tenant, document cluster,
subdomain, or entity-type subtree.

### Future Knowledge Graph capabilities

- Streaming ingestion with Kafka when RabbitMQ no longer meets throughput or
  replay requirements
- Graph-native reranking over multi-hop pooled subgraph representations
- Permissioned cross-tenant federated queries
- Domain-specific embedding models through a versioned embedding registry
- Incremental reasoning rather than full post-ingestion recomputation

The list above is conditional scale architecture, not an implementation backlog
for the current corpus. Kafka, read replicas, federation, and model registries
should be introduced only when the thresholds in the scaling reference are met.

## Full Part I maturity criteria

These criteria define full production maturity. They do not gate Part II P0.
Part I is fully mature when:

1. Ontology and schema changes are versioned, validated, and migratable.
2. Ingestion is idempotent, horizontally scalable, observable, and tenant-safe.
3. Temporal graph reconstruction is deterministic and tested.
4. Retrieval and generation evaluations isolate failure classes and use
   reproducible datasets.
5. Graph mutations, retractions, contradictions, and supersession preserve an
   auditable history.
6. Backup, restore, load, security, and cost controls are demonstrated.

---

# Part II — Extend the Platform into a Context Graph for AI

## Objective

Add a graph-native memory of AI and human decision processes. The Context Graph
must connect what the system knew, which policies applied, which alternatives
were considered, which action was selected, who approved or overrode it, and
what outcome followed.

This is a new semantic layer built on the Part I Knowledge Graph. It is not a
replacement for the existing ontology and is not simply another retrieval
stage.

## Context Graph readiness assessment

### Current assessment

**`graphrag/context_graph/` is real, working code, not a stub.** Its Pydantic
models enforce tenant consistency, cross-references, and manifest integrity;
its repository contains tenant-scoped async Neo4j persistence; and
`api/main.py` registers the Context Graph router with trace, governance,
precedent, outcome, and proactive endpoints.

`HybridRetriever._record_context_trace` records a tenant-scoped trace for
worker-path queries that have a query ID and referenced chunks. The operation
fails open so Context Graph maintenance cannot make retrieval unavailable.
`tests/integration/context_graph/test_live_neo4j.py` round-trips a trace through
Neo4j and verifies replay, approval expiry, and retention redaction.
`find_precedents` uses policy, outcome, and feedback weighting, with dedicated
ranking tests; retrieval feedback consumption is tested at the call site.

Its strongest foundations, confirmed by direct read:

- contextualized facts with provenance, confidence, and validity;
- reified statements and meta-relations;
- authority, supersession, contradiction, and negative knowledge;
- bitemporal history and graph snapshots;
- policy-gated tools, audit events, and tenant isolation (`ToolPolicy`).

The remaining caveat is production scale, not missing implementation: these
paths have run against local live infrastructure and deterministic corpora,
but not customer traffic or a production-sized Context Graph.

### Capability scorecard

| Context Graph capability | Current state | Assessment |
|---|---|---|
| Entity and relationship graph | Neo4j, ontology registry, typed domain models | Strong foundation |
| Fact-level provenance | Documents, chunks, spans, extraction model, prompt version, source type | Strong |
| Temporal context | Valid time, transaction time, snapshots, supersession, versioned community summaries | Strong; summary lineage is query-time usable |
| Confidence and epistemic state | Confidence, source type, contradiction, negative knowledge, real `confidence_lifecycle.py` state machine | Strong, wired |
| Higher-order statements | Reified relations and meta-relations | Strong foundation |
| Authority and constraints | Authority hierarchy, constraints, `ToolPolicy` (hardened A147/A149) | Strong, security-verified |
| Agent execution trace | `AgentRun`/`ToolCall`/`Observation`/`CGEpisode` + repository writes | **Implemented, wired, and live-tested** — worker retrieval records correlation, durable session episodes, executable policy results, evidence, and alternatives; maintenance fails open |
| Decision trace | Tenant-scoped `AgentRun`/`Decision` graph, real Cypher, real validation | **Implemented, wired, and live-validated** |
| Alternatives and rejection reasons | `DecisionOption.reason_code` (required field), persisted | **Implemented and unit-tested** |
| Exceptions and approvals | `CGApproval`/`CGExceptionGrant` models + append-only correction linkage | **Implemented and live-validated** — effective state is evaluated as-of a timestamp with approval and exception expiry enforcement |
| Outcomes and feedback | `record_outcome`/`record_feedback` | **Implemented and unit-tested** |
| Precedent retrieval | `find_precedents` — policy, outcome, and feedback-weighted score | **Implemented and tested**, including deterministic precision/recall/MRR metrics |
| Context assembly governance | `ContextManifest`, SHA-256 integrity hash, typed policy rules, `record_trace` | **Implemented, unit-tested, and live-tested** |
| Proactive context | Expiring-policy recommendations, as-of validity, reversible compaction | **Implemented and tested** with configurable usage/urgency thresholds and false-positive metrics; production threshold tuning remains environment-specific |

## Target three-layer ontology

```text
Domain ontology
    Entity, Organization, Regulation, Component, Requirement, Person, Case...

Knowledge and evidence ontology
    Statement, Evidence, SourceArtifact, Document, Chunk, Provenance,
    Confidence, Authority, TemporalValidity, Contradiction...

Decision and context ontology
    AgentRun, ReasoningStep, ContextManifest, Decision, Option,
    PolicyVersion, PolicyEvaluation, ToolCall, Observation, Approval,
    Exception, Action, Outcome, Feedback, Precedent...
```

The Context Graph ontology must reference domain and evidence objects rather
than copying their content into an isolated trace store.

## Minimal Context Graph model

```text
(:AgentRun)-[:HAS_STEP]->(:ReasoningStep)
(:AgentRun)-[:ADDRESSES]->(:Task|:Question|:Case)
(:AgentRun)-[:PRODUCED]->(:Decision)

(:ReasoningStep)-[:CONSUMED]->(:ContextManifest)
(:ContextManifest)-[:INCLUDED]->(:Statement|:Evidence|:Document|:Chunk)
(:ContextManifest)-[:INCLUDED_POLICY]->(:PolicyVersion)
(:ContextManifest)-[:USED_CONFIGURATION]->(:RetrievalConfiguration)

(:ReasoningStep)-[:INVOKED]->(:ToolCall)
(:ToolCall)-[:RETURNED]->(:Observation)

(:Decision)-[:DECIDED_FOR]->(:Case)
(:Decision)-[:CONSIDERED]->(:Option)
(:Decision)-[:SELECTED]->(:Option)
(:Decision)-[:REJECTED]->(:Option)
(:Decision)-[:SUPPORTED_BY]->(:Statement|:Evidence|:Observation|:Precedent)
(:Decision)-[:APPLIED_POLICY]->(:PolicyEvaluation)
(:PolicyEvaluation)-[:EVALUATED_VERSION]->(:PolicyVersion)
(:Decision)-[:USED_EXCEPTION]->(:Exception)
(:Decision)-[:APPROVED_BY]->(:Approval)
(:Decision)-[:RESULTED_IN]->(:Action)
(:Action)-[:PRODUCED]->(:Outcome)
(:Feedback)-[:EVALUATES|:CORRECTS]->(:Decision|:Outcome|:AgentRun)
(:Decision)-[:SUPERSEDES|:SIMILAR_TO]->(:Decision)
```

Every `Decision`, `AgentRun`, `ToolCall`, `ContextManifest`, `PolicyVersion`,
`PolicyEvaluation`, `Approval`, `Action`, and `Outcome` should include:

- tenant and authorization context;
- actor identity and actor type;
- correlation and causation IDs;
- recorded time and valid-time fields where applicable;
- schema and ontology version;
- integrity hash;
- concise structured rationale and reason codes.

Do not persist hidden chain-of-thought. Persist auditable inputs, constraints,
alternatives, tool observations, decisions, and outcomes.

## Part II priorities

### P0 — Foundation and first governed decision trace

`graphrag/context_graph` module, `CG*` Neo4j schema, tenant-safe immutable
persistence, deterministic context-manifest hashing, and the WPP
campaign-placement vertical slice are **implemented and live-validated**.
The worker retrieval path also creates traces through `HybridRetriever`.

### P1 — Replay, governance, and correction

Tenant-scoped replay, append-only approvals, exception grants, corrections,
supersession links, and redaction markers are **implemented and
unit-tested**. Live Neo4j replay, approval-expiry enforcement, and append-only
retention redaction are verified by the opt-in integration test. Production
retention periods remain an operator policy choice.

### P2 — Outcomes, precedent, and organizational memory

Actions, outcomes, feedback linkage, and policy-compatible precedent scoring
are **implemented and unit-tested**. The deterministic evaluation harness
reports precision@k, recall@k, and MRR for corpus exports.

### P3 — Proactive Context Graph intelligence

Expiring-policy recommendations, true as-of validity snapshots, and reversible
lossless manifest compaction are implemented. Usage and urgency thresholds are
configurable, and recommendation precision/recall/false-positive metrics are
covered by the deterministic evaluation harness. Threshold tuning against
production traffic remains operational validation, not missing code.

## Context Graph evaluation suite

Evaluation must go beyond answer relevance. Corrected against actual test
coverage (2026-08-03) — a checkmark here means a real test asserts the
behavior, not that the capability is production-validated:

- [x] Trace completeness
- [x] Evidence and provenance integrity
- [x] Context-manifest reproducibility
- [x] Valid-time and transaction-time replay contract
- [x] Policy-version and rule-evaluation linkage
- [x] Approval and exception enforcement contract
- [x] Tenant and authorization isolation
- [x] Correction and supersession integrity contract, including cycle
      prevention and full tenant-scoped chain reconstruction
- [x] Outcome-link completeness contract
- [x] Precedent relevance and policy compatibility query contract
- [x] Decision consistency under unchanged context
- [x] Appropriate decision change under changed context using
      corpus-provided expected decisions
- [x] Proactive recommendation false-positive metrics
- [x] Live-Neo4j execution of a Context Graph trace, replay, approval-expiry,
      and retention path

## Acceptance criteria for claiming "Context Graph for AI"

The platform may credibly use that label only when all of the following are
demonstrable:

1. A completed governed agent task creates an immutable, connected decision
   subgraph.
2. The exact evidence, policy versions, tool observations, retrieval settings,
   model version, and prompt version used at inference time are recoverable.
3. A point-in-time query can answer why the decision was valid then and whether
   the same decision would remain valid now.
4. Alternatives considered, selection reasons, and rejection reasons are
   represented structurally rather than only in free text.
5. Human approval, exception, correction, and override append new graph state
   without rewriting history.
6. Actions are connected to measurable outcomes and subsequent feedback.
7. A later task can retrieve authorized, temporally valid, policy-compatible
   precedents and their outcomes.
8. Evaluation covers trace, temporal, policy, security, outcome, and precedent
   quality — not only generated-answer relevance.
9. At least one full trace has been created and replayed against a live Neo4j
   instance, not just asserted against a mock.
10. A real query or agent action — not just a standalone API call — actually
    produces a Context Graph trace. **Met**: every worker-path
    `HybridRetriever.retrieve_and_answer` call with referenced chunks now
    records one (`hybrid_retriever.py:93-160,322,346`), fail-open, since
    2026-07-30.

### Exit claim

Reached at prototype and local-integration level. A real retrieval path emits
traces, and the governed trace/replay/expiry/retention path has round-tripped
through live Neo4j. This supports the label "Context Graph for AI" for the
implemented platform. It does not support a claim of production readiness or
customer-scale validation without deployment evidence.

---

# ADRs

## Knowledge Graph ADRs

| Decision | Status |
|---|---|
| Session-context enrichment strategy | Documented in `tasks/lessons.md` A03 |
| Multi-hop depth-two default | Documented in `tasks/lessons.md` A13 |
| Ontology versioning and migration semantics | Implemented and wired through `OntologyRegistry` and `/kg/ontology/migration` |
| Knowledge-state lifecycle and retraction semantics | Implemented (`confidence_lifecycle.py`, wired) |
| Temporal snapshot and integrity model | Implemented (`bitemporal.py`, wired) |
| Tenant-scoping enforcement on the agent-tool surface | Implemented and live-verified — `tasks/lessons.md` A147/A149 |
| Capability-gated Neo4j 2026 vector search | `docs/adr/0007-capability-gated-neo4j-vector-search.md`; implemented and live-validated on a separate modern volume |
| Measured adaptive retrieval routing | `docs/adr/0008-adaptive-retrieval-routing.md`; implemented and unit-tested, with production gains still unclaimed |
| Audience-bound access tokens for API and MCP | `docs/adr/0010-audience-bound-access-tokens.md`; implemented and unit-tested. Breaking for existing remote MCP clients |
| JWT key rotation, algorithm confinement, and token revocation | `docs/adr/0011-jwt-key-rotation-and-revocation.md`; implemented and unit-tested. RS256/JWKS is opt-in — `jwt_algorithm` still defaults to HS256 |

## Context Graph ADRs

| Decision | Status |
|---|---|
| Decision-trace ontology and lifecycle | `docs/adr/ADR-Context-Graph-Decision-Trace.md`; implemented, unit-tested, and live-validated |
| Context manifest, integrity hash, and replay semantics | Implemented P0/P1 contract; live replay verified |
| Structured rationale versus prohibited chain-of-thought storage | Enforced by model validation |
| Decision correction, approval, exception, and supersession semantics | Implemented P1 contract; expiry behavior live-validated |
| Trace retention, redaction, GDPR erasure, and audit preservation | Append-only retention/redaction markers implemented and live-validated; destructive erasure remains governed by the platform privacy service |
| Outcome taxonomy and precedent-ranking policy | Outcome linkage plus policy/outcome/feedback-weighted ranking implemented and tested |
| Context compaction and lossless evidence references | Reversible, lossless compaction implemented and tested |

---

# Scaling Decision Reference

## When to add an ingestion worker

Add workers when queue depth remains above the normal operating range, oldest
message age breaches the ingestion SLO, or one worker cannot meet expected peak
throughput. Validate Neo4j write contention before assuming linear scaling.

## When to upgrade the LLM provider tier

Upgrade when rate limiting is a measured production bottleneck after retries,
caching, request consolidation, and fallback routing have been evaluated.
Provider upgrades must not be used to hide inefficient round-trip design.

## When to switch to Neo4j Enterprise

Consider Neo4j Enterprise when multi-database tenant isolation, clustering,
read replicas, online backup requirements, or operational support justify the
licensing and deployment complexity.

## When to add TimescaleDB continuous aggregates

Use continuous aggregates when KPI query cost, retention volume, dashboard
latency, or SLO reporting can no longer be served reliably by ordinary indexed
queries. TimescaleDB is provisioned in both Compose files and the KPI cutover
has passed a live initialize/write/read check; continuous aggregates remain a
threshold-triggered optimization.

## When to add a SPLADE retrieval channel

Not currently planned. Measured via `scripts/benchmark_splade_impact.py`
(see `tasks/lessons.md` A157): reranking the existing BM25+vector RRF
candidate pool with `naver/splade-cocondenser-ensembledistil` costs a
measured +2072ms mean / 2424ms p95 per query on CPU (this deployment has no
GPU) — roughly doubling to tripling current per-query retrieval latency for
one extra reranking pass. The retrieval-quality gain that cost would need
to justify was **not measured** — the aerospace golden-set corpus wasn't
ingested in the environment at the time (only `pharma`, 7 docs, was
present), so no recall/MRR delta could be computed; the run wasn't
extrapolated or guessed. Revisit only if the CPU latency budget changes
(GPU becomes available) or a real recall gap shows up on non-lexical/
non-semantic-similarity queries (SPLADE's actual niche) that BM25+dense
already demonstrably miss — re-run the same script with the aerospace or
automotive corpus ingested to get the other half of this decision.

## When to add MMR (Maximal Marginal Relevance) reranking

**Not building it — measured, not just unproven.** Cost is negligible
(`scripts/benchmark_mmr_latency.py`, `tasks/lessons.md` A158: synthetic,
mean 0.36–0.49ms/selection; `scripts/benchmark_mmr_quality.py`, A159:
mean 11.3ms on real embeddings, still ~4 orders of magnitude cheaper than
SPLADE's +2072ms), but the quality delta on this pipeline's real aerospace
corpus (33 golden questions) is **mildly negative**: coverage 0.929→0.843
(−0.086), MRR 0.781→0.772 (−0.009), 1 question improved vs. 4 regressed
vs. 28 tied. Most likely cause: this pipeline already has a
document-coverage lexical-diversity step (`local_search.py:237-264`) doing
real diversity work; MMR's embedding-similarity notion of "diversity" is
document-agnostic and can demote a relevant same-document chunk in favor
of a different-document chunk that looks diverse but is less relevant —
the two mechanisms optimize different notions of "redundant" and can work
against each other rather than stacking. Revisit only if the existing
lexical-diversity step is removed or substantially changed — this verdict
is about MMR *stacked on top of* that mechanism, not MMR in isolation.

---

# 2026-08-21 Audit Follow-ons

The implementation and evidence are recorded in
`docs/archive/audits/audit-2026-08-21.md` and
`docs/archive/audits/audit-2026-08-21-second-pass.md`. Remaining work is ordered by
production value, not trend visibility.

## Production-critical

1. **Turn on RS256 in production, then federate an external IdP.** Asymmetric
   signing, `kid`, JWKS publication, rotation overlap, and token revocation are
   implemented (ADR 0011) but `jwt_algorithm` defaults to HS256 — the
   capability is inert until a deployment sets it and provides key material.
   Do that first; it is configuration plus a rotation drill. What genuinely
   remains after it is multi-issuer validation, remote JWKS fetching, and `iss`
   allow-listing.
   *Prerequisite:* key material in the deployment's secret store; then choose
   the production IdP.
   *Benefit:* a gateway or auditor can verify without being able to mint;
   remote MCP can be exposed to federated clients.
   *Complexity:* low for RS256, medium for federation.
2. Run representative 10x load and recovery tests. Check in queue-age,
   throughput, p95/p99, error-rate, provider-cost, RTO, and RPO evidence.
   *Complexity:* medium; the blocker is environment, not code.
3. **Turn on `semantic_answer_cache_strict` for any multi-replica deployment,
   and prove it.** The flag exists and defaults off. Until a deployment sets
   it, a Redis outage still silently forks the answer cache per replica and a
   correction cannot evict a sibling's copy.
   *Prerequisite:* a Redis instance whose availability is actually monitored.
   *Complexity:* low — configuration plus a failure drill.
4. Broaden the parent monorepo CI lint command to `ruff check .`; the current
   project tree passes, but the workflow still scans only selected directories.

## Recently completed

- **Property-based and concurrency testing.** `hypothesis` is now a dev
  dependency. `tests/unit/test_property_invariants.py` pins invariants across
  generated inputs — answer-cache tenant isolation above all, since a key
  collision there serves one tenant another's answer with nothing erroring, and
  no finite set of examples can establish its absence.
  `tests/unit/test_write_path_concurrency.py` covers the optimistic-concurrency
  guard, quota counters, and revocation under real `asyncio.gather` contention,
  using a state-holding fake rather than an `AsyncMock` (a canned mock answers
  identically regardless of arrival order, so it cannot demonstrate a
  concurrency property at all). Those assertions were mutation-verified.
  *Still open:* a **live** concurrency drill against real Neo4j. The fake
  models the guard; only the database can demonstrate the atomicity the guard
  relies on.

- **Async rate limiting.** `api/limiter.py` was rebuilt on `limits.aio`;
  slowapi is removed from the dependency set. Its Limiter is synchronous with
  no async variant in 0.1.x, so every Redis-backed check blocked the event loop
  — tolerable only while six low-rate endpoints were limited, and exactly wrong
  at the load where limiting matters. Enforcement is now a FastAPI dependency
  rather than a decorator, so endpoints no longer carry `request: Request`
  purely for the limiter's benefit.
- **Per-tenant quotas.** `graphrag/core/tenant_quota.py` adds a fixed-window
  budget per tenant across two dimensions (requests, cost USD), gating
  `/query` and `/ingest`. This closes a gap a rate limiter cannot: one tenant
  running steadily just under the rate limit could consume an entire day of
  shared LLM spend with every individual request looking well-behaved.
  Ceilings default to unlimited, so the feature throttles nobody until a
  deployment chooses numbers.

## Recommended

1. **Decouple the answer prompt from the aerospace corpus.** `_ANSWER_PROMPT`
   in `graphrag/retrieval/hybrid_retriever.py` hardcodes corpus-specific rules
   — revision-number formatting (`rev.2` -> `rev2`), `doc_id` metadata
   conventions, specific airworthiness phrasing. They exist because they moved
   the golden-set pass rate, and they are the single largest obstacle to the
   platform being domain-general: onboarding a second corpus today means
   editing a shared prompt that another corpus depends on.
   *Rationale:* a per-ontology prompt fragment, versioned alongside the
   ontology and composed into the base prompt, keeps the measured behaviour
   while making it additive rather than shared.
   *Prerequisite:* a runnable golden eval — this must not be changed on
   inspection alone, since the current rules are the only evidence anyone has
   about what the corpus needs.
   *Benefit:* second-corpus onboarding stops being a merge conflict.
   *Complexity:* medium; the risk is entirely in the eval, not the code.
2. Add a GraphRAG-Benchmark-compatible adapter (ICLR 2026) and compare the
   existing local/global/hybrid/agentic routes on the same datasets and cost
   envelope. The benchmark's own finding — that graph structure helps on
   multi-hop, global, and sensemaking questions and not on single-fact lookup
   — is the hypothesis to test against this corpus, not to assume.
   *Complexity:* medium. *Benefit:* replaces anecdotal route comparison with
   a quality/latency/cost triple.
3. ~~Add dashboards and alerts for oldest RabbitMQ message age, DLQ growth,
   publish failures, Neo4j pool saturation, and per-tenant model spend.~~
   **Done** — `graphrag/observability/operational_metrics.py` emits queue age,
   DLQ, publish outcome, retry, graph pool occupancy, and store-degradation
   signals; `monitoring/prometheus/alerts.yml` consumes them. A test asserts
   every metric an alert references actually exists, because a rule pointing at
   a typo'd metric never fires and never-firing looks exactly like healthy.
   **Done** — `monitoring/prometheus/prometheus.yml` wires the rules into a
   Prometheus scrape, and `monitoring/grafana/graphrag-overview.json` is
   provisioned by the Docker Compose Grafana service. SLO targets remain
   provisional until the load, soak, and restore exit criteria are met.
4. ~~Track OpenTelemetry GenAI semantic conventions and adopt the stable fields
   that map cleanly to the platform's existing traces.~~
   **Done** — `graphrag/observability/genai_telemetry.py` emits
   `gen_ai.operation.name`, `gen_ai.system`, request/response model, and
   provider-reported token usage, attached at `FallbackLLM.generate` (the
   single choke point every production model call passes through). Prompt and
   completion content are deliberately never attached: they carry customer
   document text, and a trace backend has none of the retention, tenancy, or
   erasure guarantees `graphrag/graph/gdpr.py` exists to provide.
5. **Adopt the MCP 2026-07-28 transport changes.** This pass implemented the
   specification's authorization requirements only. Its stateless protocol
   core, multi-round-trip requests, header-based routing, and cacheable list
   results remain unadopted.
   *Prerequisite:* an SDK upgrade (`mcp` is currently constrained to 1.x) and
   a client-compatibility review.
   *Complexity:* medium-high; not worth taking piecemeal.
6. Replace RAGAS if upstream does not fix its multi-modal SSRF and DiskCache
   dependency; until then keep it isolated to offline evaluation workers.

## Experimental — benchmark before implementation

1. DRIFT-style search versus the current bounded agentic fallback.
2. Query-personalized PageRank versus existing graph expansion/PageRank.
3. FastGraphRAG/LightRAG-style extraction versus current ontology-governed
   extraction, measuring indexing cost and domain-relation recall together.

## Explicitly deferred

- A second graph database (including Ultipa) without a measured Neo4j blocker.
- Always-on agentic loops.
- A LightRAG migration that would remove temporal, provenance, ontology, or
  Context Graph governance semantics.
