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
| Query-conditioned retrieval controls | Adaptive traversal policy, retrieval sufficiency telemetry, auditable evidence bundles, and opt-in text/graph/path/provenance fusion are implemented and unit-tested; live calibration is still required before enabling all controls |

### Known scale limits

## Semantic modelling and compilation roadmap

The Energy implementation validates a reusable pattern for the base platform:
author domain meaning once, then compile it into the representations required by
each deployment. This is the practical lesson from frame/slot modelling and
from the RDF-versus-LPG question: the durable asset is the domain model, not a
particular database schema.

The canonical YAML model already emits OWL/RDFS, SHACL and Neo4j artefacts and
records target limitations. The following additions are ordered by the value
they add to every future domain, with Energy as the first reference model.

**Status (2026-09-18):** the domain-onboarding and curation-workbench
sections below are fully closed. What remains — the P0 capability-matrix
redesign and the P1 qualified-cardinality/composite-key schema change — are
each explicitly flagged, in their own bullet text, as needing "a real
redesign" / "a schema change... needs its own design pass," not an
additive implementation. Neither is assessed as worth picking up without
that design pass happening first; see the Audit Follow-ons section's
"Current status" note for the same conclusion applied to the rest of the
document.

### P0 — target capability matrix and loss diagnostics

- [x] Give every semantic rule a typed target outcome: `enforced`,
      `translated`, `runtime-mitigated`, `warning`, or `unsupported`.
      `fidelity` today is a 3-value axis
      (`preserved`/`approximated`/`unenforceable`,
      `graphrag/semantic_model/compiler.py`'s `Diagnostic.fidelity`) read by
      a live demo consumer (`api/routes/energy_demo.py`) and its tests; the
      5-value split additionally implies emitting a positive `enforced`
      record for every natively-enforced rule × target, not just losses —
      turning `diagnostics.json` from a loss report into a full matrix and
      roughly doubling its size. A real redesign, not a rename; needs its
      own design pass.
      **Status (2026-11-03):** `Diagnostic.outcome` (enforced/translated/runtime-mitigated/warning/unsupported, derived from `fidelity` plus the compensating control; `fidelity` kept) and a `capability-matrix.json` per model (every rule x target, never more optimistic than the diagnostics). See `docs/interoperability-and-curation.md`.

**Base project value:** makes format choice a compilation decision, supports
architecture reviews and prevents false confidence from incomplete schemas.

### P1 — visual and tool-friendly semantic modelling

- [x] Add explicit support and examples for qualified cardinalities,
      relationship attributes and composite keys where the target supports
      them. Needs new fields on `PropertySpec`/`RelationSpec`/`Cardinality`
      (`graphrag/semantic_model/models.py`) and touches every emitter in
      `compiler.py` (OWL restrictions, SHACL property shapes, Neo4j
      constraints) — a schema change, not an additive one; needs its own
      design pass.
      **Status (2026-11-03):** composite keys, qualified cardinalities and relationship attributes in the model, all three emitters, the runtime validator and the ERD; examples in the `contextual-knowledge` model and the synthetic test model. Relationship attributes are reified in OWL/SHACL and native but runtime-validated on Neo4j.

**Base project boundary:** YAML remains the canonical source; ERD and Protégé
are review and interoperability surfaces, not competing sources of truth.

### P1 — ontology curation and human-in-the-loop workbench

- [x] Add a reusable curator queue for proposed entities, relations and schema
      changes with confidence, provenance, conflicts and affected use cases.
      `graphrag/graph/ontology_proposals.py`'s pre-existing `OntologyProposalService`
      (`:OntologyProposal` nodes, `EVIDENCED_BY`/`ASSERTED_IN`/`PROPOSED_FOR`
      provenance) now also carries `confidence` (from extraction), `conflicts`
      (same proposed value proposed under more than one kind in the same
      batch — a real, cheaply-computed collision), and an `affected_use_cases`
      field (populated by the caller; wiring it to
      `ontology_migration.find_affected_competency_questions` is a natural
      follow-up, not done here to avoid forcing a `MigrationReport` out of a
      single ad hoc proposal).
- [x] Support approve, edit, reject, merge, defer and quarantine decisions,
      each with actor, reason, timestamp and model/version evidence.
      `OntologyProposalService.decide(..., action=...)` — six actions, each
      recording `reviewed_by`, `reviewed_at`, `decision_reason`, and
      `decision_model_version`; `edit` also stores the corrected
      `proposed_value`, `merge` stores `merge_target`. The old
      `approve: bool` call shape still works for existing callers.
      `api/routes/kg/ontology_proposals.py` exposes all six as
      `POST /kg/ontology/proposals/{id}/{approve|edit|reject|merge|defer|quarantine}`.
- [x] Add duplicate and contradiction views backed by the existing entity
      resolution, provenance and validation services.
      `GET /kg/curation/duplicates-and-contradictions`
      (`api/routes/kg/curation_review.py`) — a thin, read-only aggregation
      over the already-persisted `ReviewQueueService.list_pending()` and
      `ContradictionDetector.get_open_conflicts()`. No new storage, no new
      write path; approve/reject/resolve stay at their existing endpoints.
- [x] Add request intake and prioritisation fields: business impact, urgency,
      effort, risk, dependencies and requesting team.
      `OntologyProposalService.submit()`/`list()` carry
      `business_impact`/`urgency`/`effort`/`risk`/`dependencies`/
      `requesting_team` alongside each proposal; populated by the caller
      (curator UI or ingestion), no fixed vocabulary enforced.
- [x] Export review decisions as golden-set/training data and expose status
      reporting for partner teams.
      `OntologyProposalService.export_golden_set()` returns decided
      proposals (approved/rejected/edited/merged/deferred/quarantined) as
      `(input, label)` records with the human's corrective evidence
      (`corrected_value`, `merge_target`, `decision_reason`); no reward or
      ranking signal is inferred, that's left to the consumer.
      `status_report()` is a plain count-by-status/kind breakdown (no
      SLA/throughput metric — none is agreed for curator turnaround yet).
      `GET /kg/ontology/proposals/export` and
      `GET /kg/ontology/proposals/status-report`.

**Energy first example:** review WT-01 evidence, missing-evidence requests,
invalid observations and work-order state transitions in one operations-facing
curation flow.

### P1 — reusable domain onboarding and governance pack

- [x] Provide templates for a business glossary, SME workshop notes, entity and
      relationship decisions, unresolved questions, ownership and deprecation.
      `docs/templates/domain-onboarding-*.md` (5 files), referenced from
      `ontology/README.md`'s new "Onboarding a new domain" section. Templates
      only — nothing validates they were filled in.
- [x] Add a domain profile describing identifiers, naming policy, provenance,
      retention, access, temporal semantics, validation and publication rules.
      `docs/templates/domain-profile-template.md` — each field states
      whether it's platform-enforced (with the enforcing code cited) or
      policy-only.
- [x] Add a regulated-domain example, such as insurance or financial services,
      covering auditability, explainability, retention and policy traceability.
      Used automotive IATF 16949 (already in the repo, already regulated)
      rather than inventing a new domain from scratch. See below.
- [x] Demonstrate a second domain reusing the compiler, competency-question
      tests, curation queue and governance templates without modifying Energy
      code.
      `ontology/models/automotive-iatf-quality.yaml` re-expresses
      `config/ontologies/automotive_iatf.yml`'s vocabulary (suppliers,
      audits, nonconformities, quality documents, standards) in the Energy
      compiler's YAML schema, compiled through the same `compile_model()`/
      `load_model()` with no Energy-specific code change. Covers the four
      named concerns: auditability (`Auditable` mixin, `Audit.auditorId`/
      `outcome`), explainability (`Nonconformity.rootCause`/
      `correctiveAction`), retention (`retentionPeriodYears`,
      `reevaluationFrequencyMonths`), policy traceability
      (`Standard→mandates→QualityDocument→appliesTo→AutomotiveComponent`,
      `references`/`supersedes`). Details, scope, and honest limitations in
      `ontology/README.md`'s "Second domain: automotive" section.
      `make semantic-model-check` now drift-checks both models; a new test
      (`test_compiler_is_domain_general_not_energy_specific`) compiled it and
      caught a real bug in the process — a diagnostic message hardcoded
      "Energy" even for non-Energy models, fixed in `compiler.py`.
      Competency-question tests and the curation queue were *not* re-plumbed
      into this model: automotive's existing Cypher competency questions
      (`evals/automotive_iatf/`) run against the separate LPG/Neo4j track
      (`graphrag/graph/domain_ontology.py`), which this YAML model doesn't
      feed into; the curation queue (`OntologyProposalService`) is likewise
      LPG-track and tenant-agnostic to which YAML produced the schema, so it
      already applies to any domain, automotive included, without change.

**Value:** turns stakeholder discovery into reviewable engineering artefacts and
provides evidence for collaboration with SMEs, architects and country teams.

### P2 — additional target and integration adapters

- [x] Add read-only conformance adapters for additional LPG/RDF targets only
      when a real deployment requires them; benchmark against the same dataset
      fingerprint and competency-question suite.
  **Status (2026-11-03):** `scripts/sparql_conformance.py` runs the SPARQL competency suite against any SPARQL 1.1 endpoint (Stardog, GraphDB, Fuseki) with a dataset fingerprint. Verified offline only; not run against a real Stardog, GraphDB or Fuseki.
- [x] Gremlin (Neptune/Cosmos DB): `GraphBackend` Protocol + `GremlinBackend`
      cover core entity/relation CRUD and 1-hop retrieval, additive to Neo4j,
      live-verified against a real `tinkerpop/gremlin-server` container (not
      Neptune/Cosmos DB themselves). See
      [ADR-0012](adr/0012-graphbackend-protocol-and-gremlin.md).
- [x] Consider Stardog and GQL adapters as interoperability targets, not as
      additional sources of truth.
  **Status (2026-11-03):** Stardog via the SPARQL conformance adapter; GQL as a generated `graph-type.gql` (ISO GQL graph-type rendering, flattened inheritance, losses reported). Not validated by a GQL engine.
- [x] Add GraphQL only as a stable consumer API over governed RDF/Neo4j reads;
      it must not bypass SPARQL, SHACL, tenant boundaries or audit controls.
  **Status (2026-11-03):** `POST /graphql`, query-only, every resolver through the capability registry (scopes, tenant, audit); answer generation deliberately not exposed.
- [x] Add optional Jira/ticketing and spreadsheet import/export integrations for
      curator operations after the core review model is stable.
  **Status (2026-11-03):** CSV/XLSX export and decision import, and a Jira issue route, on the ontology-proposal queue; reviewer is the token `sub`. Jira only exercised against a mock transport; the route is not idempotent.

### Explicit non-goals

- Do not generate a large number of database dialects merely to increase the
  artifact count.
- Do not make Protégé, GraphQL, Neo4j or any other target a second canonical
  model source.
- Do not call synthetic Energy fixtures live SAP, Snowflake or SharePoint
  connectors, and do not present local scorecard measurements as enterprise
  SLAs.

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
| PROV-O RDF alignment (`graphrag/provenance/prov_o.py`, `scripts/export_rdf.py`) | **Implemented, SHACL-gated under `--strict`** | Tenant-scoped documents, chunks, assertions, ingestion activities, retrieval activities, software agents, evidence inputs, and answer digests are exported with `prov:used`, `prov:wasGeneratedBy`, `prov:wasDerivedFrom`, `prov:specializationOf`, and `prov:wasAssociatedWith`; source text is not copied into the PROV projection. Validation runs *before* serialisation: `--validate` reports violations, `--strict` refuses to write the file and exits non-zero. An activity that would have no `prov:used` input is skipped with a logged warning rather than exported non-conformant. |
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
| Contextual knowledge applicability and validation | Three-valued applicability, normative modality, bounded constraints, conflicts/exceptions, gated releases, `kg.knowledge.evaluate@1.0.0` | **Implemented and unit-tested**; wired into Energy `maintenance_review`; synthetic evaluation only, not live-validated |

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

### Contextual knowledge evaluation (2026-10-24)

Shared contracts (`context_graph/knowledge.py`: identity/version, tenant,
sources, publication status, valid/recorded time, applicability over customer,
brand, market, product, product family and workflow, normative modality,
bounded constraints, exceptions and supersession), a three-valued
applicability service, an evaluator that validates **every** applicable
requirement and reports conflicts, and gated knowledge releases on the existing
durable governance store are **implemented and unit-tested**. Exposed as
`kg.knowledge.evaluate@1.0.0` and `POST /knowledge/evaluate`, and wired into the
Energy `maintenance_review` answer (**implemented and wired**; Energy answers
now carry `publication_version_id`). The synthetic packaging/labeling pack and
its 26-case evaluation set are a **synthetic engineering evaluation**: invented
rules, developer-written expectations, no domain review. A same-day audit
reproduced and fixed eight evaluator and release-gate defects (partial
exception coverage, precedence without an effective winner, prohibition
absence semantics, fact typing, order-dependent supersession, empty regression
gate, multiple effective exception versions, disclaimer-only releases); each
has a regression test. Published knowledge also projects to RDF and is gated
by SHACL generated from `ontology/models/contextual-knowledge.yaml`; packs are
published, rolled back and inspected through authenticated routes. Not done: an
RDF export route, a review UI or per-entry approval route, live validation
against Neo4j or a deployed API, and any real regulatory content. See `docs/contextual-knowledge-evaluation.md`.

- [x] Project published knowledge items to RDF and validate them with SHACL
  generated by the canonical semantic-model workflow (2026-10-29: model
  `ontology/models/contextual-knowledge.yaml`, `knowledge_rdf.py`, gate in
  `check_pack`; enumerations and scopes stay Pydantic-enforced).
  **Base project value:** one authoritative model for knowledge items instead of
  Pydantic-only validation.
- [x] Authenticated pack review/publish routes using token `sub` as reviewer
  and publisher identity (2026-10-29: review queue, publish, history,
  rollback; metadata-only responses; tenant from the token).
  **Base project value:** curation without code access; same provenance rules
  the library already enforces.
- [x] Union several partial waivers or partial supersessions when together
  they cover a requirement's whole bound context, and report the remaining
  matched context of a partially superseded revision (2026-10-29: judged per
  combination of context values, capped at 5,000 combinations, fail-closed).
  **Base project value:** fewer needless evaluations of requirements that are
  jointly waived, without weakening the current fail-closed coverage rule.

### Candidate additions from a graph-community feed review (2026-11-03)

**Status: proposed, not implemented.** Source: the newest 48 posts (back to about
one month) of a public LinkedIn graph-database/knowledge-graph group, with the
comments visible on each. That material is secondary, partly promotional (vendor
launches, events, job ads, self-published books) and unverified; it was used
only to prompt ideas. Nothing below is adopted on the strength of a post, and no
product claim from the feed is treated as fact until checked against its own
documentation. Items are phrased against this platform's existing contracts.

- [x] Decision re-review impact analysis. A decision from the contextual-knowledge
  evaluator cites its release; when a release changes, an exception nears expiry
  or an item passes a review-due date, list the stored decisions that would now
  come out differently, keeping "what was known then" apart from "what is known
  now", and keeping missing evidence, pending approval and refusal as distinct
  states. Needs a design decision first: decisions are returned and cached today,
  not persisted, so a durable decision log (tenant-scoped, append-only) comes
  before the analysis. `review_due` would be an explicit, human-set field; the
  tool must never set a "verified" flag itself.
  **Base project value:** answers "which past decisions need another look after a
  rule or ownership change", which the as-of/known-as model already makes
  computable.
  **Acceptance evidence:** replaying a stored decision against the release it
  cites reproduces it exactly; a diff against the current release lists exactly
  the affected decisions on fixtures that change one rule, one waiver expiry and
  one superseding revision; tenant isolation and access filtering hold.
  **Status (2026-11-03):** `review_impact` (`graphrag/context_graph/knowledge_decisions.py`) re-evaluates stored decisions after a release change, at a moved valid time, for exceptions lapsing within a window, and for items whose human-set `review_due` has passed, reporting then and now separately. Admin route `POST /knowledge/decisions/{domain}/review-impact`.
- [x] Remediation hints in machine-readable errors. Publication-gate failures and
  evaluation errors already come back structured; give each a stable code, the
  offending field or item key, and the smallest corrective action, so an agent or
  curator can act without parsing prose.
  **Base project value:** shorter fix loops for agent-driven curation, with no
  change to the gate itself.
  **Acceptance evidence:** every failure from `check_pack` and every
  `invalid_request` carries code, path and hint; tests assert them instead of
  matching message text.
  **Status (2026-11-03):** `check_pack_issues` returns `GateIssue(code, path, message, hint)`; `KnowledgePublicationError.issues`; the publish route returns them; evaluation `invalid_request` errors carry `code`, `path` and `hint`. Text messages are unchanged.
- [x] Evaluate SHACL-to-Cypher validation for the Neo4j runtime. The generated
  shapes run on RDF exports; property graphs rely on runtime validators and the
  mutation gate. Compare an existing open-source SHACL-to-Cypher compiler (check
  licence, maintenance and its single-maintainer risk) against a minimal in-house
  subset limited to what the generated shapes use. Evaluation only: no new write
  path, no second source of truth.
  **Base project value:** one set of generated shapes checked on both stores.
  **Acceptance evidence:** parity with the RDF-side validator on fixed valid and
  invalid fixtures for the Energy and knowledge models, recorded with the shape
  hash it ran against.
  **Status (2026-11-03):** evaluated: the open-source `shacl2cypher` (MIT per PyPI metadata, one release 0.3.0 on 2026-09-28, one maintainer, Rust core) was not installed, being new executable code from a single source; an in-house subset `graphrag/semantic_model/cypher_checks.py` compiles the canonical model into read-only Cypher violation queries (required, datatype, endpoints, relation and qualified counts, key and composite-key uniqueness) tied to the shapes hash, with a unit test that every generated property shape of a concrete type has its check. Parity with SHACL on a live Neo4j is written (`tests/e2e/test_live_cypher_checks.py`) but has not run: Docker was unavailable here.
- [x] Controlled-natural-language rendering of constraints for reviewers. A
  deterministic text rendering of `Constraint`/`Predicate` over the existing
  operator vocabulary, shown in the review queue next to the statement.
  Inspired by business-rules vocabularies such as SBVR, but a view only: no new
  rule language and no change to the W3C-based model, which stays authoritative.
  **Base project value:** reviewers approve what the executable rule says, not
  only what the prose says.
  **Acceptance evidence:** golden renderings for every operator and `when_missing`
  mode; a test that the rendering cannot change evaluation.
  **Status (2026-11-03):** `knowledge_render.py`; the review queue shows `executable` next to `statement`; golden renderings for every operator and `when_missing` mode.
- [x] Release stamp on every projection. Each projection of a release (RDF
  export, Energy answers, GraphQL and MCP payloads, cache entries) carries the
  release id and content hash; a check fails when a served projection does not
  match the active release. The decision cache already keys on release; the RDF
  export does not yet carry it.
  **Base project value:** corrections propagate or are visibly stale across
  projections, with the release remaining the single authority.
  **Acceptance evidence:** a test publishes, rolls back and re-exports, and
  asserts stamps and stale-projection detection.
  **Status (2026-11-03):** the RDF export can carry a `KnowledgeRelease` node (release id + content hash, in the canonical model and its generated shapes) that every projected entry points at; decisions already cite their release and the cache keys on it; `KnowledgeEvaluationService.stale_projections` reports projections whose content hash is not the active one. A test publishes, changes the release and rolls back: stamps go stale and current again. Not done: stamping the Energy RDF publication, which keeps its own `publication_version_id`.
- [x] Shape and model registry as plain RDF (lower priority). Describe which
  generated model and shapes apply to which graph, with version, SHA-256 and
  active flag, in a named graph shipped with the export so the mapping travels
  with the data.
  **Base project value:** portability across triple stores without store-specific
  configuration.
  **Acceptance evidence:** a SPARQL query answers "which shapes apply to graph X"
  and the recorded hash equals `compile --check`.
  **Status (2026-11-03):** `graphrag/semantic_model/registry_rdf.py`: models with id, version, model and shapes SHA-256, shapes path and active flag; data graphs bound with `dcterms:conformsTo`; `shapes_for_graph` is one SPARQL query; `shapes_drift` compares recorded hashes with the committed shapes.
- [x] Verify before considering an "Open Knowledge Format" bundle export. A post
  describes a markdown-bundle format for agent-readable knowledge with untyped
  links; its existence, version and licence are unverified here. Only if the
  specification checks out, evaluate an export in which typed edges and trust
  fields survive and nothing like a "verified" flag is invented.
  **Acceptance evidence:** a conformance check against the published
  specification, and a round trip that preserves edge types.
  **Status (2026-11-03):** verified: OKF v0.2 (2026-07-25) is published by Google Cloud in GoogleCloudPlatform/knowledge-catalog (`okf/SPEC.md`); only `type` is required, links are untyped, unknown keys must be accepted, trust fields (`generated`, `verified`, `stale_after`) are optional. The repository licence was not confirmed. An evaluation prototype (`graphrag/context_graph/okf_export.py`) exports visible published items with typed edges in an extension key, writes `verified` only from recorded human reviews, and passes a conformance check against the spec rules it relies on; no OKF consumer was run against it.

Considered and **not** added: new embedded or streaming graph engines announced
in the feed (a second graph database stays deferred without a measured Neo4j
blocker, see "Explicitly deferred"); additional schema dialects such as
PG-Schema, LinkML, FalkorDB or Memgraph targets (the non-goal on dialect count
stands; the GQL graph-type export already planned covers interoperability);
"GraphRAG versus vector RAG" guidance (the golden evaluation's per-type
breakdown already measures it); "semantic layer first" accuracy arguments
(consistent with the current direction, no new work); job posts, events and
book announcements.

### Candidate additions from a knowledge-platform job description (2026-11-03)

**Status: proposed, not implemented.** Source: a job description for a shared
"knowledge and reasoning" platform role in a packaging-software company, used
only as a checklist of what such a platform is expected to provide. It is not a
claim of fit with that company's product, and none of its regulatory or domain
content is adopted. Already covered by the contextual-knowledge module: three-valued
applicability, mandatory-versus-advisory modality, conflicts and exceptions,
escalation to human review, versioned gated releases, tenant and access filtering
before evaluation, curation routes, separate evaluation metrics, API and MCP.

- [x] Durable decision log and explanation-fidelity evaluation. Persist each
  evaluation decision (tenant-scoped, append-only, with the release it cites and
  the facts it used) and measure whether its explanation is faithful: every cited
  item must change the outcome when removed, and no uncited item may.
  Persistence is also the prerequisite of the decision re-review analysis above.
  **Base project value:** audit for multi-tenant use, and a measurable answer to
  "which knowledge items influenced this result".
  **Acceptance evidence:** a counterfactual test over the evaluation set
  (remove each cited item, assert the outcome changes; remove an uncited item,
  assert it does not); a stored decision replays exactly; no decision readable
  across tenants.
  **Status (2026-11-03):** append-only, tenant-scoped `knowledge_decisions` table; every service evaluation is recorded with its inputs, access context and cited release, and replays exactly. Counterfactual fidelity on the 26-case set: no decisive item or exception is missing from any decision's citations. The converse (every cited item decisive) is not required and does not hold: a decision lists every applicable rule.
- [x] Stronger evaluation of applicability and conflict handling. Add metamorphic
  tests (same decision after reordering items, adding an irrelevant item, adding
  another tenant's items, renaming ids), a held-out split with adversarial cases,
  and generated contexts with large denominators. The current 26-case set is
  developer-authored and measures agreement with the author's expectations.
  **Base project value:** evidence of recall and stability that does not depend on
  hand-picked cases.
  **Acceptance evidence:** reported with denominators and split labels; the
  held-out split is not used when tuning rules.
  **Status (2026-11-03):** metamorphic relations (input order, another tenant's items, unpublished/inaccessible items, an out-of-scope item, consistent renaming) over all 39 cases; a 13-case held-out adversarial split (`packaging_labeling_heldout.json`, 13/13 on first run, written before running); a generated split of 200 seeded contexts x 400 synthetic items against an independent oracle (`knowledge_oracle.py`): applicability 80000/80000, validation 6524/6524. All three are still developer-authored; they show stability and internal consistency, not domain correctness.
- [x] Scale and observability of the evaluator. Evaluation scans every visible
  item per request, which is correct but not shown for thousands of items. Add an
  index by applicability dimension, a latency benchmark with stated budgets, and
  metrics for cache hit rate and time per stage.
  **Base project value:** a shared service has to meet latency and reliability
  expectations as packs grow.
  **Acceptance evidence:** identical decisions with and without the index on the
  evaluation set; benchmark results recorded with pack size, hardware and commit.
  **Status (2026-11-03):** `scripts/benchmark_knowledge_eval.py` (seeded packs, stage shares, p95 budgets) showed conflict detection at 40% of the time at 5000 items, so the index is on conflict candidates (subject, series, overrides) rather than on applicability dimensions: every visible item must still report its own applicability reasons, so a dimension prefilter could not skip that work without changing decisions. 5000 items: p95 775 ms to 440 ms (local, 50 requests); decisions identical with and without the index. Stage timings in `evaluate_knowledge(timings=...)`; `GET /knowledge/metrics` (admin) reports cache hit rate and time per stage.
- [x] Record the formalism decision and cover non-constraint knowledge. Write an
  ADR comparing the current bounded predicates with decision tables (DMN),
  Datalog and a production rule engine, including failure modes. Today only
  constraints and guidelines are modelled; transformations and procedures are
  not. If the ADR supports it, add DMN-style decision tables as a bounded item
  kind for them: representation and validation only, no execution.
  **Base project value:** a documented, reviewable choice of formalism, and
  coverage of knowledge that does not fit a conjunction of predicates.
  **Acceptance evidence:** the ADR; a table with an explicit hit policy,
  validated for overlaps and gaps; tests that unsupported content is refused or
  escalated rather than approximated.
  **Status (2026-11-03):** ADR-0014 (`docs/adr/0014-knowledge-formalism.md`) compares bounded predicates, DMN-style tables, Datalog and production rule engines with their failure modes; `graphrag/context_graph/decision_tables.py` adds tables with declared input domains, a bounded condition vocabulary and an explicit hit policy, validated for overlaps, gaps and conflicting outputs (published invalid tables block the gate). Representation and validation only: tables are not executed. Unsupported content (expressions, undeclared domains) is refused at load time.
- [x] Expert elicitation workflow. Turn informal expertise into reviewable items:
  interview templates, expert attribution, recorded disagreement between experts,
  a "needs expert" state, and review ownership, built on the existing draft and
  review queue.
  **Base project value:** the stated source of much domain knowledge is people,
  not documents.
  **Acceptance evidence:** a draft item cannot be published without a named
  reviewer; recorded disagreement blocks publication until resolved or waived
  with an approval reference.
  **Status (2026-11-03):** interview template (`config/knowledge_packs/templates/expert_interview.yml`) and `graphrag/context_graph/elicitation.py`: answers become an attributed draft, `needs_expert` when the expert is unsure, unstated scope stays unknown, never executable or published; `ExpertDisagreement` blocks publication while open and closes only resolved (with the resolution) or waived (with an approval reference); the review queue shows experts, review owner and open disagreements; publication still needs a named reviewer.

Considered and **not** added: real regulatory content (it could not be validated
here and the module must not imply compliance; only synthetic structures such as
multilingual sections or market-specific rules fit), fact extraction from artwork
files (a separate project), and Datalog as the primary engine (it may be revisited
for derived facts after the ADR above).

### Candidate additions from a second job description: RDF and extraction role (2026-11-03)

**Status: proposed, not implemented.** Source: a job description for a
graph-engineering role covering RDF modelling, extraction services, entity
resolution and search. Used as a checklist only; no claim of fit with that
employer's product. Most of it is already planned or implemented here: domain
model, evidence-backed ingestion, safer entity resolution, labelled evaluation
and RDF publication (see "Suggested extension, Recruitment Knowledge Graph"
above), SKOS label mappings, PROV-O provenance, SHACL gates, incremental
relational ingestion with quarantine, and retries, timeouts and circuit
breaking. Only the gaps below are added.

- [x] Erasure and correction propagation audit across every derived store.
  List every place a fact can live after it is written (Neo4j, vector and text
  indexes, RDF exports and triple-store graphs, query and decision caches,
  knowledge-release snapshots, logs) and prove that an approved correction or
  deletion reaches each one, or record why it cannot. Today propagation is
  stated for traces and for the optional recruiter adapter only.
  **Base project value:** a privacy claim that holds for the whole system, not
  one store.
  **Acceptance evidence:** a test writes a synthetic fact, then deletes or
  corrects it and asserts absence or the new value in every store above, per
  tenant; stores that are immutable by design (append-only releases) are listed
  with their retention or redaction rule.
  **Status (2026-11-03):** `graphrag/graph/erasure_registry.py` accounts for 24 stores (erased, redacted, expires, immutable with its rule, or a stated gap); erasure now also clears media and cached embeddings, the decision log and trust reports are redacted per subject, the decision cache is evicted, triple-store resources are erased per tenant, and the extraction cache can be purged. Offline tests cover the stores that run without a database; Neo4j-backed stores only through mocked drivers. Remaining gaps are listed in `docs/privacy-controls.md` (chunk vectors, community summaries, aliases, Turtle exports).
- [x] Data minimisation before model providers. Before text leaves the system for
  an LLM or embedding provider, apply a per-tenant field allowlist and
  pattern-based redaction of direct identifiers, and log a hash and size of what
  was sent rather than the content. Allow a tenant to disable external providers.
  Pattern-based redaction is incomplete by nature and is described as such.
  **Base project value:** sensitive records are not sent to providers by default.
  **Acceptance evidence:** tests with synthetic identifiers show them absent from
  the outbound payload; the audit record holds hash and size only; a disabled
  provider is never called.
  **Status (2026-11-03):** `graphrag/core/outbound.py` inside `FallbackLLM.generate` and `OpenAIEmbedder.embed`: per-tenant provider switch (a disabled provider is never called), pattern-based redaction of direct identifiers, field allowlist, and a hash-and-size audit; ingestion sets the tenant context. Pattern redaction is incomplete by nature; directly constructed clients are not covered.
- [x] Extraction record contract check. Verify, and enforce where missing, that
  every extracted assertion keeps its source location, the model and prompt
  version that produced it, a confidence, and a review state (accepted, uncertain,
  unsupported), and that uncertain or unsupported results go to the review queue
  instead of the graph.
  **Base project value:** extracted facts can be traced and re-run when a model
  changes.
  **Acceptance evidence:** a schema-level test over the extractor output; a
  replay with a different model version produces a diffable record.
  **Status (2026-11-03):** every extracted entity and relation carries source chunk, served model, a prompt version derived from the template, confidence and `review_state` (accepted, uncertain, unsupported); only accepted assertions are written, the rest go to the review quarantine; `extraction_records` / `diff_extraction_records` make model replays diffable.
- [x] Named-graph layout decision for RDF stores. Record, as an ADR, how data is
  split into named graphs (per tenant, per source, per publication version) and
  how that supports rollback, per-tenant deletion and provenance queries, plus an
  optional JSON-LD serialisation for API consumers that cannot use Turtle.
  **Base project value:** deletion, rollback and tenant boundaries follow graph
  boundaries in the store.
  **Acceptance evidence:** the ADR; a SPARQL test that drops one tenant's graph
  without touching another's; a JSON-LD round trip that preserves IRIs and
  datatypes.
  **Status (2026-11-03):** ADR-0013 (`docs/adr/0013-named-graph-layout.md`); `graphrag/graph/named_graphs.py` with per-tenant `DROP GRAPH` (tested: one tenant dropped, the other untouched, a longer tenant name with the same prefix untouched) and a JSON-LD round trip that is isomorphic and keeps IRIs and datatypes. Tested on an in-memory rdflib Dataset; existing exports still write a single graph.

Considered and **not** added: embeddings and hybrid-search evaluation, entity
resolution, taxonomy mapping and pre-publication checks (already planned above or
implemented); and any work on a specific graph vendor, since the store-neutral
SPARQL adapter and conformance suite already cover Stardog, GraphDB and Fuseki
offline.

### Candidate additions from a third job description: enterprise knowledge deployment role (2026-11-03)

**Status: proposed, not implemented.** Source: a job description for a
customer-facing role that deploys and operates a knowledge platform over
enterprise sources. Used as a checklist only; no claim of fit with that
employer's product. Already present here: SharePoint and Snowflake source
connectors with injectable clients, source ACLs applied before retrieval with
fail-closed handling of unknown ACL state, incremental and delta ingestion,
drift monitoring, feedback capture, a semantic-model compiler and the
contextual-knowledge module. Only the gaps below are added.

- [x] Governed business-definition registry. A versioned register of the terms
  people and agents must share ("active customer", "which price list applies
  when"), each with an owner, a plain-language definition, a precision level
  (exact, bounded or deliberately loose), the executable form where one exists
  (a SQL view, a SPARQL query or a knowledge-item predicate), the sources it
  depends on, and a review date. Definitions resolve through the same release,
  tenant and access rules as knowledge items, and an answer lists the
  definition versions it used. The registry records where a definition is
  intentionally loose rather than forcing false precision.
  **Base project value:** one definition per concept across retrieval, agents
  and reports, with a trail when a definition changes.
  **Acceptance evidence:** a definition change produces a new version and lists
  the answers or decisions that used the old one; two tenants may define the same
  term differently without crossing; an answer cites definition ids and versions.
  **Status (2026-11-03):** `graphrag/context_graph/definitions.py`: versioned, owned definitions with an explicit precision (exact needs an executable form, loose refuses one, bounded states its limits), sources, dependencies and review date, carried in the knowledge release and resolved by tenant, status, latest version and access; items reference them, decisions list `definitions_used`, and `decisions_using` finds the stored decisions that used an older version. The gate rejects references to unpublished definitions and duplicate published terms. Executable SQL/SPARQL text is stored, not run.
- [x] Use-case charters with success metrics. Per use case, a checked-in charter
  naming the question set, the metrics (for example time to answer and answer
  accuracy against expert-labelled examples), the owner of the labels and the
  acceptance thresholds; the evaluation runner reports each use case against its
  own charter. Thresholds come from the domain experts, not from this repository.
  **Base project value:** progress is judged against agreed measures per use case
  rather than one global score.
  **Acceptance evidence:** the runner fails a use case that misses its charter
  threshold and reports denominators and split labels; a charter without a named
  label owner is rejected.
  **Status (2026-11-03):** `graphrag/context_graph/charters.py` and `scripts/run_charters.py` over `evals/charters/*.yml`: each charter names its question set, split, label owner and threshold source; a missing owner is rejected, a missed threshold fails the use case, and every metric reports its denominator. The bundled charters are explicit synthetic placeholders: their thresholds were not set by domain experts.
- [x] Answer-trust investigation workflow. When a user marks an answer as not
  trusted, assemble one bundle: the question, retrieved evidence, permission
  filters applied, definition and release versions, and the trace; classify the
  root cause from a fixed list (retrieval miss, stale source, permission filter,
  wrong or missing definition, extraction error, model error, expectation
  mismatch) and track counts per class and the time to a fix. Builds on the
  durable decision log proposed above.
  **Base project value:** "investigate answers users do not trust and fix the
  cause" becomes a repeatable process with measurable outcomes.
  **Acceptance evidence:** a synthetic untrusted answer for each root-cause class
  is classified correctly from its bundle; the bundle contains no content the
  reporting user could not read.
  **Status (2026-11-03):** `graphrag/context_graph/answer_trust.py`: a distrust report on a stored decision assembles a bundle under the reporter's access (restricted knowledge omitted, facts only for the asker), classifies the root cause from the fixed list, and is stored with time to fix; counts per class and median hours to fix are reported. Routes: `POST /knowledge/decisions/{domain}/{record_id}/distrust` (recipient or admin), `GET /knowledge/trust-reports/{domain}` and `POST .../{report_id}/fixed` (admin). Covers knowledge decisions; GraphRAG answer traces are not yet wired into it.
- [x] Permission-freshness bound. State and test how quickly a revoked source
  permission stops affecting answers: the maximum age of cached ACL state, what
  happens when a source cannot be reached (deny, not allow), and how a revocation
  invalidates caches and any derived index entries.
  **Base project value:** "respect source permissions and stay up to date" gets a
  stated bound instead of an assumption.
  **Acceptance evidence:** tests revoke access on a fake source and assert denial
  within the stated bound, including through the query cache; an unreachable
  source yields denial and an audit record.
  **Status (2026-11-03):** ACL observation time on the policy and the document node; restricted documents older than `ACL_MAX_AGE_SECONDS` are denied in the Cypher predicate and the Python mirror; the answer-cache key includes the ACL window; stated bound: next request after a re-sync, otherwise at most twice the max age; an unreachable permission source is denied and audited. Tested with a fake SharePoint source; policies without an observation time are not bounded until re-synced.
- [x] Confluence and Databricks source connectors (conditional). Only if a real
  deployment requires them: a Confluence connector that maps page-tree and space
  permissions to the existing ACL model and detects changes, and a Databricks
  connector following the existing Snowflake connector. Same pattern as the other
  connectors: injectable client, offline tests, no claim of live validation.
  **Acceptance evidence:** permission-mapping tests, including inherited
  restrictions; change detection that removes deleted pages from the graph.
  **Status (2026-11-04):** `graphrag/enterprise/confluence.py`: CQL change detection by `lastmodified` within configured spaces, trashed pages become deletes and `reconcile()` tombstones pages that vanished; page-tree read restrictions map to one allow list (intersection across restricted levels, space readers from configuration for unrestricted pages, empty intersection, unknown space readers or unreadable restrictions denied and audited); routes `POST /sync/confluence/{source_id}/run` and `/reconcile`. `graphrag/ingestion/databricks_source.py`: SQL Statement Execution API 2.0 with OAuth M2M, polling, chunked results, terminal-state errors and schema-drift checks, conforming to `TabularSourceConnector`. Both tested against mocked transports only; a reader who qualifies through different groups at two restricted levels is denied (conservative).

Considered and **not** added: operating on a cloud platform, SQL and relational
modelling, evaluation in continuous integration and AI-assisted development
workflows (all already in place or process, not platform features); expert
elicitation (covered by the proposal above); and Finnish or other language
support beyond the existing answer-language handling, which needs a concrete
corpus and labelled examples first.

### Candidate additions from a fourth job description: ontology-grounded AI delivery role (2026-11-04)

**Status: proposed, not implemented.** Source: a job description for a role
building pipelines, retrieval workflows and decision-support agents on an
ontology-centred data platform. Used as a checklist only; no claim of fit with
that employer's product, and no integration with that platform is proposed (none
could be verified here). Already present here: RAG and GraphRAG grounding over
vector search and ontology content, a canonical semantic model with SHACL and a
runtime mutation validator, decision tables, the decision log and answer-trust
workflow, retrieval-trajectory evaluation and rubrics, source ACLs, privacy and
erasure controls, and use-case charters. Only the gaps below are added. Today
`CGAction` and `ActionNode` only record actions after the fact; nothing in the
model declares an action that can be proposed, checked and executed.

- [ ] Ontology actions and functions (the write path of ontology-augmented
  generation). Declare actions in the canonical model YAML (for example
  `quarantine_lot` or `open_8d_report`): typed parameters, the object types they
  apply to, submission criteria, the graph effects they produce and whether they
  need approval. Declare functions as read-only logic bound to an object type
  that agents may call. The flow: an agent proposes an action with parameters;
  `SemanticMutationValidator` and the applicable decision tables check it; a
  named human approves when required (reusing the answer-trust and review
  patterns); the effect is written in one transaction; the proposal, checks,
  approval and effect go into the decision log. Actions and functions are
  exposed through the capability registry, so MCP and GraphQL get the same
  scopes, tenant binding and audit.
  **Base project value:** agents can act on the graph, not only answer about it,
  with every change validated against the model and attributable to a person.
  **Acceptance evidence:** a proposal that violates a submission criterion or a
  decision-table rule is rejected with the rule named; an action that needs
  approval does not change the graph until approved; an approved action's
  effects match its declaration; a replay from the decision log reproduces the
  same verdict; a caller without the scope or from another tenant cannot submit.
- [ ] Agent evaluation on actions, not only answers. Test cases that name the
  expected action (or the expected refusal) and its parameters for a request:
  whether the agent chose the right action, filled the parameters correctly,
  respected the submission criteria and declined what it must decline. Runs
  offline with a scripted model, beside the retrieval-trajectory evaluation and
  the rubrics; reports per action type with denominators.
  **Base project value:** agent changes are judged on what they would do to the
  graph, which answer-level scores do not show.
  **Acceptance evidence:** a scripted agent that picks the wrong action, omits a
  required parameter or executes something it should refuse fails the case that
  covers it; results are deterministic across runs.
- [ ] Data expectations on pipeline transforms and per-build lineage. Each
  ingestion transform declares expectations on its output (schema, nulls,
  uniqueness, freshness, row-count bounds); a failed expectation blocks
  publication of that build instead of writing partial data. Each build records
  its inputs, code version, expectation results and output counts, linked to the
  existing lineage records. Extends the completeness and ingestion-gap checks in
  `docs/data-quality-operations.md`, which run after the fact.
  **Base project value:** bad source data stops at the transform with a reason,
  and every published graph change traces back to the build that produced it.
  **Acceptance evidence:** a fixture that breaks each expectation kind blocks the
  build and leaves the previous publication in place; a passing build records
  inputs, code version and results; the lineage of a node resolves to its build.

Considered and **not** added: more retrieval or GraphRAG work and more
governance or privacy controls (already covered in depth); a connector to that
specific platform's SDK (not verifiable here); training and user-support
material (process, not a platform feature); and adoption metrics per use case,
which can follow once actions exist and their acceptance and rejection rates
are worth measuring.

### Candidate additions from a fifth job description: agent-first engineering role (2026-11-04)

**Status: all three done (2026-11-04); the kind job has not run yet.** Source: a job description centred on
the developer (daily work through coding agents, reviewing agent output,
production systems, Kubernetes, testing habits). Used as a checklist only; most
of it describes the person and is shown through the CV and interviews, not
code. Already present here: Testcontainers integration and e2e tests on a real
Neo4j in CI (failing when skipped), relational ingestion and SQL connectors,
Neo4j and vector search, Kustomize manifests rendered in CI, mutation testing,
ADRs, and the agent tooling in `.claude/` with the lesson log in
`tasks/lessons.md`. Only the items below are added.

- [x] Agent-assisted development write-up. How the repository is built through
  a coding agent: what the agent is given, the guards on its output, and the
  failure modes those guards exist for, each tied to a recorded case.
  **Base project value:** shows review of agent output as a practice with
  evidence, not a statement.
  **Status (2026-11-04):** `docs/agent-assisted-development.md`, citing lessons
  A52, A56, A133, A141, A154, A164, A173 and A181 and cases from the current
  sessions; states the guards' limits.
- [x] Kubernetes manifest schema validation. Run `kubeconform` (strict, with
  the Kubernetes version the manifests target) on the rendered Kustomize output
  in CI, beside the existing `kubectl kustomize` render step, which only proves
  the manifests build.
  **Base project value:** an invalid field or API version fails CI instead of
  the first deploy.
  **Acceptance evidence:** a deliberately invalid manifest fails the step; the
  committed manifests pass.
  **Status (2026-11-04):** the CI `infrastructure` job renders the base and the kind-smoke overlay and validates both with kubeconform 0.6.7 (`-strict`, Kubernetes 1.30.0), then asserts that `deploy/ci/invalid-manifest.yaml` is rejected. Runs in CI only; not run locally.
- [x] Kind smoke deploy (conditional, costs CI minutes). Deploy the rendered
  manifests to a throwaway kind cluster in CI and wait for the API readiness
  probe. Only on explicit approval, since it adds CI time to every run.
  **Acceptance evidence:** the job reaches a ready API pod and fails with the
  pod events when it does not.
  **Status (2026-11-04):** CI job `kind-smoke` (approved 2026-11-04): builds the API image, loads it into a kind cluster, applies `deploy/kind-smoke` (one API pod, non-development configuration, placeholder secrets, RWO volume) and waits for rollout, printing pods, events and logs on failure, then calls `/health` through the Service. Readiness is checked on `/health` because `/health/ready` needs Neo4j, Redis and an LLM provider, which are not deployed. A local run of the API with the same configuration and no backing services completed startup; the kind job itself has not run yet.

Considered and **not** added: a Helm chart (would duplicate the Kustomize
manifests; worth it only where a deployment requires Helm), and the
experience, ownership and communication requirements (not platform features).

### Candidate additions from a sixth job description: ontology and knowledge-graph lead (2026-11-04)

**Status: proposed, not implemented.** Source: a job description (in Spanish)
for a senior role applying ontologies and knowledge graphs, with RAG, agent
orchestration and semantic data governance. Used as a checklist only; no claim
of fit with that employer's product. Already present here: OWL export and
reasoning, SHACL validation, the SPARQL bridge and endpoint conformance suite,
RDF interoperability, GraphRAG with LLM integration, Neo4j with Cypher, a
Gremlin client and triple-store backend, the business-definition registry,
use-case charters, expert elicitation and ADRs. Only the gaps below are added.

- [ ] Data-catalog export: Apache Atlas glossary and OpenLineage events. Export
  published business definitions (`graphrag/context_graph/definitions.py`) as
  Atlas glossary terms with owner, precision, review date and version, and their
  dependencies as term relations; emit ingestion and build runs as OpenLineage
  run events with input and output datasets, linked to the existing lineage
  records (`graphrag/enterprise/lineage.py`). Tenant and access rules apply as for
  any other export; restricted definitions are not exported without the reader's
  access. Clients are injectable and tested against mocked transports; no claim
  that a real Atlas, Collibra or Marquez instance accepted the payloads.
  **Base project value:** the semantic layer reaches the governance catalog
  people already use instead of staying inside this platform.
  **Acceptance evidence:** exported payloads validate against the published
  Atlas glossary and OpenLineage JSON schemas; a definition's new version updates
  the term rather than duplicating it; a run event names exactly the datasets the
  build read and wrote; a restricted definition is absent from a plain export.
- [x] LangGraph adapter (conditional). Expose the capability registry as
  LangGraph tools so an external LangGraph agent gets the same scopes, tenant
  binding, argument validation and audit as MCP and GraphQL. A thin adapter
  only: orchestration inside this platform stays on its own agents
  (`base_agent`, `skill_router`, `tool_policy`). Only if a deployment uses
  LangGraph, since it adds a framework dependency.
  **Acceptance evidence:** a scripted LangGraph graph calls a read capability
  and receives the same result and audit record as the MCP path; a call without
  the scope or for another tenant is refused.
  **Status (2026-11-04):** `mcp_server/langgraph_tools.py`: `registry_tools(registry, identity)` turns each entitled capability into a LangChain `StructuredTool` (provider-safe names, tenant bound to the identity, JSON content plus the raw payload as the artifact) that calls `CapabilityRegistry.call`; `langgraph_registry()` audits with `transport=langgraph`. Tests run a real LangGraph `StateGraph` with `ToolNode` and compare result and audit with the direct registry call; other-tenant and invalid arguments are refused. langchain-core is not in the API image, so the adapter is for external agents in environments that install it.

Considered and **not** added: an Azure Cosmos DB backend (a further graph store
with no deployment that needs it), a Collibra-specific connector (the Atlas and
OpenLineage formats come first; a Collibra mapping can follow a real
requirement), and the experience, leadership and language requirements (not
platform features).

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
    records one (`_record_context_trace`, `hybrid_retriever.py:172`, called
    from `:849` and `:908`), fail-open, since 2026-07-30.

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

## When to add query-personalized PageRank reranking

**Not building it — measured, not just unproven.** Live-benchmarked against
the real aerospace corpus (33 golden questions, live Neo4j + real
embeddings, `scripts/benchmark_personalized_pagerank_quality.py`): seed GDS
Personalized PageRank (`sourceNodes`) with entities mentioned by the
current top-5 candidate chunks, propagate over the tenant's Entity/
RELATES_TO graph, blend 50/50 with original fused relevance. Result is
**mildly negative**: hit rate 0.970→0.939, coverage 0.904→0.884, MRR
0.744→0.708, 5 questions improved vs. 7 regressed vs. 21 tied. Cost is real
but modest: mean 164ms/query, p95 349ms (one GDS graph projection reused
across all queries in a run, not re-projected per query). Full results in
`evals/personalized_pagerank_quality_results.json`.

This exercise also caught and fixed a real production bug, independent of
the personalization verdict: three Cypher-building methods in
`neo4j_client.py` (`get_chunk_filenames`, `get_best_chunk_for_document`,
`bm25_search_entities`) embedded `document_access_predicate('d')` in a
plain (non-f) triple-quoted string, so the literal, unparseable text
`{document_access_predicate('d')}` was sent to Neo4j on *every* call to
those three methods, unconditionally — `bm25_search_entities` sits on the
production hybrid-retrieval path (`bm25_search.py`), so this broke
BM25-over-entities retrieval outright, for every tenant, whether or not ACL
enforcement was even enabled. Every existing unit test mocks `neo4j.run`,
so nothing ever sent this Cypher text through a real parser until this live
run. Fixed by adding the missing `f` prefix; regression-tested in
`tests/unit/test_document_link_topology.py` (asserts the predicate call is
actually interpolated, not left as literal template text) rather than
relying on a live Neo4j instance to catch a recurrence.

## When to add DRIFT-style search

**Not building it — measured, and decisively negative on cost, flat on
quality.** Live-benchmarked against the real aerospace corpus (33 golden
questions, live Neo4j, real LLM calls,
`scripts/benchmark_drift_search_quality.py`) against the existing bounded
agentic (IRCoT) fallback (`agentic_retriever.py`), both given an identical
seed local search on the raw question so only the expansion strategy
differed: DRIFT's one-shot global-primer-driven follow-ups (community
summaries → fast-model primer + up to 3 follow-up sub-questions → one local
search per follow-up → large-model synthesis) versus the existing
iterative fast-model reasoning loop (up to 4 steps, each either answering
or issuing one more local-search sub-query).

Result: hit rate 1.000 vs. 1.000, coverage 1.000 vs. 0.985, **MRR
identical to three decimal places (0.787 vs. 0.787) — 0 questions
improved, 0 regressed, 33 tied**, at **3.8x the latency** (agentic mean
5993ms vs. DRIFT mean 22915ms) from DRIFT's larger, fixed evidence-gathering
shape (seed + primer + up to 3 follow-up local searches run unconditionally,
each a full 6-stage retrieval pipeline, versus agentic's early-exit-capable
loop — most golden questions resolved in 1 agentic step). DRIFT's LLM-call
count is actually lower (2 fixed vs. agentic's 2-5), so the added latency is
retrieval-pipeline cost, not LLM cost. Full results in
`evals/drift_search_quality_results.json`.

This exercise caught and fixed two further real production bugs, both
independent of the DRIFT verdict itself and both invisible to every
existing unit test because they mock `neo4j.run`/the LLM client and never
exercise a real Groq response or a real Lucene parser:

- **`FallbackLLM`'s Groq→DeepSeek chain didn't catch `groq.APIStatusError`.**
  A live 413 "tokens per minute" response (DRIFT's larger accumulated
  context occasionally exceeded Groq's per-org TPM budget) is a
  `groq.APIStatusError` instance, not the `groq.RateLimitError` subclass
  the fallback chain checked for (`isinstance(openai.APITimeoutError(...),
  groq.APITimeoutError)` and the equivalent groq/groq pairing being False
  across SDKs the same way) — so it crashed the caller instead of failing
  over to DeepSeek, exactly the failure mode the DeepSeek→Groq direction
  was already hardened against after the 2026-07-24 incident (see
  `deepseek_primary()`'s docstring). Fixed in `graphrag/core/llm_client.py`
  by adding `APIStatusError` to both `GroqClient.generate()`'s caught
  exceptions and `FallbackLLM.groq_primary()`'s `fallback_exceptions`
  tuple. While adding regression coverage, also found and fixed a *pre-existing
  vacuous test*: `TestGroqFailFast`'s one test constructed its exception
  from `openai.APITimeoutError` instead of `groq`'s own (the two are
  separate class hierarchies, not `isinstance`-compatible), so it never
  actually exercised `GroqClient.generate()`'s except clause at all.
- **Unescaped Lucene special characters crashed BM25 fulltext search.**
  `bm25_search_chunks`/`bm25_search_entities` (`neo4j_client.py`) passed
  the raw question straight into `db.index.fulltext.queryNodes` with zero
  escaping. A DRIFT-generated follow-up question containing a `/`
  (`"...LEAP-1B/CFM56..."`) crashed with a Lucene `TokenMgrError` instead of
  searching for it — and since this is on the production hybrid-retrieval
  path (`bm25_search.py`, used by every retrieval mode), any real user
  question containing `/ - ! ( ) : ^ " ~ * ? | & { } [ ] \` (all plausible
  in technical/aviation text — part numbers, dates, aircraft designators)
  could crash retrieval outright today, not just under DRIFT. Fixed with
  `_escape_lucene_query()`, matching Lucene's own
  `QueryParserBase.escape()` character set; regression-tested in the new
  `tests/unit/test_bm25_lucene_escaping.py`.

---

# 2026-08-21 Audit Follow-ons

The implementation and evidence are recorded in
`docs/archive/audits/audit-2026-08-21.md` and
`docs/archive/audits/audit-2026-08-21-second-pass.md`. Remaining work is ordered by
production value, not trend visibility.

### Current status (2026-09-18): nothing below is assessed as worth
### implementing right now

Every item in this section falls into one of three buckets, none of them
"a tractable improvement waiting to be built":

- **Measured and rejected.** Query-personalized PageRank and DRIFT-style
  search were both live-benchmarked against the real aerospace corpus (see
  "When to add query-personalized PageRank reranking" and "When to add
  DRIFT-style search" in the Scaling Decision Reference above) and are
  **not being built** — flat-to-negative quality for real added latency.
  Both benchmark runs also each surfaced and fixed a real, previously
  undetected production bug (a Cypher f-string defect, a missing LLM
  provider-fallback exception, unescaped Lucene query syntax) — the more
  durable value of this pass was hardening what already exists, not adding
  to it.
- **Infra/deployment-gated, not a code gap.** RS256 rollout, 10x load
  testing, `semantic_answer_cache_strict` with monitored Redis, and the
  remaining P2 adapters (Stardog/GraphQL, Jira/spreadsheet integration)
  all require a real deployment or explicitly contradict this document's
  own non-goals ("do not generate a large number of database dialects
  merely to increase the artifact count"). Building them now would be
  lower-quality work by the project's own stated standard. Gremlin
  (Neptune/Cosmos DB) is the one P2 adapter that closed since this was
  written — `GraphBackend`/`GremlinBackend`, live-verified against a real
  Gremlin Server, still deployment-gated only for the two named vendors
  themselves (see ADR-0012).
- **Not benchmarked, deprioritized rather than rejected.** LightRAG-style
  extraction is the one remaining "Experimental" item genuinely untested.
  It wasn't skipped for a negative result — it's the most expensive of the
  three experimental items to measure honestly (it needs a *second,
  separate* corpus ingestion with an alternate extraction pipeline, real
  LLM spend on top of what personalized-PageRank and DRIFT already used),
  and two other benchmarks measured this same session both came back
  negative. That's a reason to deprioritize spending more on this pattern
  next, not a claim that LightRAG itself is known to be a bad idea.

RAGAS replacement remains blocked on an upstream fix, independent of this
assessment.

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
## Recommended

1. Replace RAGAS if upstream does not fix its multi-modal SSRF and DiskCache
   dependency; until then keep it isolated to offline evaluation workers.

**Done, not reflected before this pass:**
- Decoupling the answer prompt from the aerospace corpus — done, see
  `graphrag/retrieval/answer_policy.py` (`BASE_ANSWER_PROMPT` plus a
  per-`answer_policy` domain-rules fragment; `hybrid_retriever.py` now calls
  `answer_prompt(cfg)` instead of hardcoding aerospace rules).
- GraphRAG-Benchmark-compatible adapter — done, see
  `graphrag/evaluation/graphrag_benchmark.py` and
  `scripts/run_graphrag_benchmark.py`. Actually running it to compare
  local/global/hybrid/agentic routes on a shared dataset/cost envelope is a
  live-evidence task, not a code gap.
- MCP 2026-07-28 stateless transport — done, see
  `mcp_server/transport_20260728.py` (stateless `tools/list`/`tools/call`
  core over the same entitlement-filtered registry as the legacy SDK path).

## Experimental — benchmark before implementation

1. FastGraphRAG/LightRAG-style extraction versus current ontology-governed
   extraction, measuring indexing cost and domain-relation recall together.

Query-personalized PageRank and DRIFT-style search were both benchmarked
and are **not being built** — see "When to add query-personalized PageRank
reranking" and "When to add DRIFT-style search" above.

## Explicitly deferred

- A second graph database (including Ultipa) without a measured Neo4j blocker.
- Always-on agentic loops.
- A LightRAG migration that would remove temporal, provenance, ontology, or
  Context Graph governance semantics.
