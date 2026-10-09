# GraphRAG hardening — Phase 0 audit and implementation plan (2026-10-09)

Scope: the 8-phase request (validation gate, schema registry, targeted invalidation,
query routing, trust metadata, explanations, guarded MCP, shared context contract).
This file is the Phase 0 deliverable. **No spec feature has been implemented yet.**

Every status below was traced from a real call path by a read-only review pass and is
cited `file:line`. Where I quote a number it comes from a committed results file, with
its path and date. `(static)` means read from code, not executed.

## 0. Verification environment (what I can and cannot prove here)

| Capability | Available locally | Consequence |
|---|---|---|
| Unit / integration (mocked) / load tests | yes | Baseline below; run after every phase |
| Live Neo4j (`tests/e2e`, testcontainers) | **no**: Docker CLI present, daemon not running | e2e tests are written and run in CI; locally they skip. Start Docker Desktop to run them here. No Phase claim that depends on real Cypher behavior is "verified" until CI e2e is green |
| Live LLM judge (RAGAS, golden eval) | not in CI; needs API keys + a loaded stack | Quality/faithfulness deltas cannot be measured offline. Route accuracy and routing behavior can. No quality improvement is claimed without a measured before/after |

**Baseline (2026-10-09, commit 77cf41e, `tests/unit tests/integration tests/load`, 4 xdist workers):**
**2608 passed, 1 skipped, 0 failed, 28 warnings, 739.7 s.** The one skip is
`tests/integration/context_graph/test_live_neo4j.py` (needs `RUN_LIVE_NEO4J=1`; CI never sets it,
so `ContextGraphRepository`'s Cypher is never run against a real Neo4j today).
`tests/e2e` (28 tests, live Neo4j/Redis) was not run locally: no Docker daemon.
Existing recorded quality numbers (not re-measured, need the live stack): aerospace golden
pass_rate 0.8529 (29/34, `evals/aerospace_regression_results.json`, undated file, last commit
2026-09-23); automotive golden 0.6 (6/10, `evals/automotive_eval_results.json`, run 2026-09-21);
faithfulness_answerable 0.874 (`evals/faithfulness_eval_results.json`, 2026-08-21).

CI state at start: the 10-09 nightly failed (clean-install, Test, E2E) because `pyarrow`
26.0.0 dropped NumPy 1.x while `graspologic` caps `numpy<2`. Fixed and committed as
`77cf41e`, **not pushed**. The seven `service-images` jobs pass.

## 1. Runtime flow as it actually is

```
source -> IngestionAgent.extract (graphrag/agents/ingestion_agent.py)
  -> extractor.py: LLM extraction; ontology/SHACL check ONLY if registry.is_loaded (extractor.py:173-239)
  -> entity resolution: GraphWriter alias registry; AmbiguousMatch -> review queue, fails open (graph_writer.py:109-123)
  -> IngestionAgent.write: CorpusMutation { write_document, repoint chunks, reconcile, write_chunks,
        write_entities, write_relations }  -- ALL WRITTEN LIVE (ingestion_agent.py:315-358)
  -> validate_and_check_cycles AFTER the write; findings only logged/returned (ingestion_agent.py:418)
     (also DELETEs self-loops, SETs has_cycle, auto-quarantines degree anomalies)
  -> indexing (Neo4j vector/fulltext), PageRank, community drift check
query -> HybridRetriever.retrieve_and_answer (retrieval/hybrid_retriever.py)
  -> query_planner (5 keyword classes) + adaptive bandit router -> mode local|hybrid|global
  -> LocalSearch (vector+BM25+RRF, rerank, multihop, entity ctx, GNN) / GlobalSearch (communities)
  -> sufficiency gate -> LLM synthesis -> low-confidence -> AgenticRetriever (max 4 steps)
  -> QueryResult: answer + citations/evidence + evidence_bundle + trajectory + sufficiency
```

Key structural fact: **records are written first and validated after**. There is no
publication gate on the Neo4j graph.

## 2. Capability matrix (spec -> reality)

| Spec item | Status | Evidence |
|---|---|---|
| SHACL on the property graph | PARTIAL, inconsistent | RDF-temp-graph check pre-write in relational ingest (`relational.py:501-508`) and per-chunk in extractor (`extractor.py:223-239`, conditional on registry loaded); export-time only elsewhere. No SHACL->Cypher layer. |
| Required labels/props, confidence range, endpoint types | PARTIAL | Only the ingestion shape (`ontology/shapes/ingestion.shapes.ttl`) on those two paths. `Relation.confidence` has no model bounds (`models.py:243`). |
| Temporal interval validity | **MISSING** | No `valid_from <= valid_to` check anywhere. |
| SUPERSEDES refs / tenant consistency / orphans | PARTIAL | domain/range only; supersession silently no-ops on missing nodes (`document_authority.py:96-117`); orphan check is post-write advisory, LIMIT 100. |
| Cardinality, property types | **UNWIRED** | `SemanticMutationValidator` never passed to `GraphWriter` in production; `PropertySchemaValidator` is GET-only though its docstring claims it is wired into ingestion. |
| Lifecycle EXTRACTED->STAGED->VALIDATED->PUBLISHED | **MISSING** | `Document.status` is pending/processing/done/failed (unrelated). Quarantine is a post-write flag on entities (`quarantine.py:63-67`). Only the energy demo has stage->validate->quarantine->publish->rollback, in-memory/SQL on RDF (`energy/publication.py`). |
| Quarantine keeps rule IDs/source/provenance, retryable | **MISSING** | `QuarantineLog` has no rule ID, source or retry; rejected extractor assertions are dropped (only a 2000-char event), rejected relational batches leave no record. |
| Reports: severities, counts by rule/tenant/source, JUnit/SARIF | PARTIAL / MISSING | SHACL report has Violation/Warning/Info; Prometheus by target/severity/shape only; no JUnit/SARIF anywhere. |
| Schema registry (Dataset/SchemaVersion/ValidationProfile) | PARTIAL | `OntologyVersion` exists (`ontology_registry.py:203-216`) but: `active` never changed after create; hash omits semver, shapes and properties; no Dataset/CONFORMS_TO; not linked to Document/Entity/Relation; answer provenance hard-codes `"platform/v1"` (`hybrid_retriever.py:68`). |
| Schema/shape drift at ingest + startup | PARTIAL | Vocabulary drift per chunk (`ontology_registry.py:259-371`); no shape-file or startup hash check. |
| Dependency tracking | PARTIAL | `source_doc_ids`+`doc_confidences`; community `SUPPORTED_BY/DERIVED_FROM`; `ContextManifest` for answers. Inferred edges carry **no premises**; manifests record chunks/docs only; nothing reverse-walks these edges. |
| Targeted invalidation | **INCONSISTENT, mostly global** | Corrections/ingest/ontology migration bump a per-tenant revision (invalidates every cached answer). Only re-ingest of a document is targeted (`reconcile_document_evidence`). `invalidate_for_entities` is reachable only from a manual endpoint. |
| VALID/NEEDS_REVIEW/RECOMPUTING state + recompute queue + invalidation trace | **MISSING** | `PropagationService` dirty flags have no callers. |
| Query router with the 7 requested classes | PARTIAL | 5 keyword classes (factoid, relational, contradiction, multi_hop, negative); no ENTITY_LOOKUP/AGGREGATION/TEMPORAL/AMBIGUOUS; temporal handled by parameters; `"fallback: review"` is never read (`query_planner.py:52-71`, `hybrid_retriever.py:862`). |
| Structured/aggregation route | PARTIAL, MCP-only | 4 regex intents (`controlled_query.py`); no COUNT/GROUP BY; not wired into `/query`. |
| Bounded agentic fallback | FULLY IMPLEMENTED | `agentic_retriever.py:207,330,335`; off under ACL. |
| Per-category evaluation | PARTIAL | Golden sets have author `type` labels, not planner route labels; no route accuracy / fallback rate / p95 / tokens aggregate recorded anywhere. |
| Trust metadata | PARTIAL | `source_type`, `confidence`, `authority_level`, `valid_*`, `confidence_state` exist. **`confidence_state` is never read by retrieval** (RETRACTED/DISPUTED edges still retrieve). `SourceType.LLM` never assigned. `stale_after`, `verification_status`, `generated_by` absent. |
| Expired/superseded excluded consistently | **INCONSISTENT** | Vector/BM25/multihop filter doc/edge validity; entity-context, agentic, controlled query, MCP `lookup_entity`, SPARQL each differ or filter nothing. `include_superseded` defaults True and no caller overrides it. |
| Explanation structure | PARTIAL / INCONSISTENT | No answer-level `confidence`, `inferences`, `score_breakdown`, `schema_version`, `limitations`; graph paths are flat strings. The async API drops trajectory/bundle/sufficiency (`messaging/consumers.py:112-136`). |
| Explanation authorization | **MISSING (leak when ACL on)** | Decision-trace routes filter tenant only and return answer + raw chunk text (`context_graph/repository.py:343-372`). |
| No unrestricted Cypher via MCP/agents | **FULLY IMPLEMENTED** | No tool accepts Cypher; every interpolated fragment is an internal constant/clamped int. |
| Read-only execution, timeouts, result caps, approval | PARTIAL | Session is not read-only (`neo4j_client.py:126`); no timeout in the MCP path; `requires_approval` is never read by `registry.call`; `POST /agent/tool` can `erase_entity` immediately with spoofable `requested_by`. |
| Shared ContextItem | MISSING | Distinct surfaces only (EvidenceBundle, ContextManifest, CitationEvidence, ...). |

## 3. Defects found along the way (independent of the spec)

Ranked. Items marked **P0** are cheap and I recommend fixing before Phase 1 because later
phases depend on them.

1. **P0 - Schema bootstrap skips 9 of 74 statements** in the ingestion and combined workers
   and `scripts/init_schema_only.py` (`startswith("--")` on a fragment that begins with a
   comment). *Reproduced:* skips `doc_id`, `entity_name_type_tenant`, `conflict_id` uniqueness
   constraints, among others; failures only log a warning. Only `Neo4jClient.init_schema`
   strips comments first.
2. **P0 - `merge_relation(s_batch)` resets `source_type`, `confidence_state`, `valid_from/to` on
   every re-ingest** (*verified* in the Cypher: `neo4j_client.py:1418-1422` and `1529-1533` SET them unconditionally), silently undoing manual overrides, APPROVED/RETRACTED states and expiry;
   `/edge/reject` hard-DELETEs (not durable); corrections have no tests.
3. `confidence_state` is write-only, and `/kg/confidence/transition` does not bump the corpus
   revision, so a retraction does not reach cached answers.
4. Decision-trace routes and some `/kg` + agent routes leak past ACL when `access_control.enabled`
   (carried over from audit-2026-10-01 H5/H6, not yet fixed).
5. `capability` metric label is the raw client-supplied name on `not_found`
   (`registry.py:135`): unbounded Prometheus series.
6. `BitemporalStore` compares stored datetimes to raw string params; likely returns nothing on a
   live DB (static, **unverified** - needs the e2e stack).
7. Extractor validation is skipped entirely when the ontology registry is not loaded
   (`ontology_validation_skipped`).
8. Orphan community snapshots stay "current"; cache key `ontology_version` is a constant.

## 4. Design decisions (made here; flag if you disagree)

**D1 - Where the publication gate lives.** Two options: (A) validate the in-memory batch *before*
any write and keep lifecycle state on the ingestion batch plus a durable quarantine store;
(B) write records to the graph in a STAGED state and add a `publication_state` filter to every
retrieval query. **Choose A.** B requires touching every Cypher read path and repeats the
null-trap class of bug we already fixed (`NOT x.quarantined = true`). With A an invalid record
is *never written*, which is a strictly stronger guarantee than "written but filtered".
Graph-context rules (dangling SUPERSEDES, cross-tenant references) run as read-only Cypher
against the existing graph plus the batch's referenced ids before publish.

**D2 - Extend, don't duplicate, the registry.** Evolve `OntologyVersion` into the spec's
`SchemaVersion` (keep the label and ids for compatibility; add `name`, `version`,
full-content `content_hash`, `source_uri`, `loaded_at`) and add `Dataset`, `ValidationProfile`,
`CONFORMS_TO`, `USES_VALIDATION_PROFILE`, `IMPORTS`. Enforce one active version per
(tenant, dataset). Replace the hard-coded `"platform/v1"` strings with the real id.

**D3 - Invalidation is event-driven over explicit dependency edges, staged 3a/3b/3c.**
3a targeted invalidation of cached answers and community snapshots (largest concrete win);
3b inferred-edge premises + retraction; 3c state machine + recompute worker on the existing
RabbitMQ. The global revision bump stays as the safety net; targeted paths narrow it.

**D4 - Router: extend the existing planner, keep its names as aliases.** Map the legacy five
classes onto the requested taxonomy, add ENTITY_LOOKUP / AGGREGATION / TEMPORAL / AMBIGUOUS,
and wire the controlled-query route into `/query` only with measured value and never under ACL.
No LLM classifier in this pass.

**D5 - Trust: add `origin` (extracted|imported|inferred|generated|manual) and
`verification_status` rather than redefining `source_type`.** Redefining `source_type` would
change meaning for existing data. Do **not** flip `include_superseded` to False without the
golden-eval run the backlog already requires (REMAINING_AUDIT_BACKLOG item 1).

**D6 - Abstention stays off by default.** `retrieval_sufficiency_abstain_enabled=false` was tried
and reverted for the automotive tenant (settings.yml:270 comment). This pass adds explicit
*reasons* (including `all_evidence_stale`) without changing the default behavior.

## 5. Phases

Each phase = audit finding -> design -> smallest coherent change -> tests -> run -> docs ->
one focused commit. Order is chosen so each phase only depends on earlier ones.

### Step 0 (prerequisite, small): P0 defects
- Files: `workers/ingestion_worker.py`, `workers/combined_worker.py`, `scripts/init_schema_only.py` (use one shared statement loader), `graphrag/graph/neo4j_client.py` (`merge_relation`, `merge_relations_batch`: do not overwrite `source_type`/`confidence_state`/`valid_to` when already set by a manual/approved transition).
- Tests: parse the real `schema.cypher` and assert all 74 statements and the named constraints are executed; unit tests that a re-merge keeps APPROVED/RETRACTED/manual state.
- Risk: changing `merge_relation` semantics affects confidence recompute; covered by the existing idempotency suite.
- Accept: every statement bootstraps from every entry point; manual state survives re-ingest.

### Phase 1 - Validation as a publication gate (D1)
- Current: write-then-validate; SHACL on two paths; no rule IDs for graph rules; no lifecycle; no durable quarantine; no JUnit/SARIF.
- Intended: `graphrag/graph/validation/` package: a rule registry (stable `rule_id`, severity BLOCKING/WARNING/INFORMATIONAL, mapping to SHACL shape names where one exists), a pre-write batch validator (labels/props/types/confidence range/temporal intervals/tenant consistency/endpoint types/ER-output shape) and read-only graph-context checks (dangling SUPERSEDES, cross-tenant refs, orphans). `IngestionBatch` lifecycle EXTRACTED->STAGED->VALIDATED->PUBLISHED on the run manifest. Durable `:QuarantinedRecord {rule_ids, source, provenance, payload, status}` with a retry API after correction. Reports aggregated by rule/severity/tenant/source; pytest + JUnit (SARIF only if the CI already consumes it - it does not, so JUnit).
- Files: new `graphrag/graph/validation/`; `ingestion_agent.py`, `graph_writer.py`, `relational.py`, `extractor.py` (call the gate; remove the "only if registry loaded" bypass or make it explicit and recorded), `schema.cypher`, `scripts/`, CI workflow.
- Tests: invalid data rejected and quarantined with rule IDs; valid data published; validation queries proven read-only (assert no write clauses); tenant-mismatch rejected; retry after correction; unit + mocked-integration locally, live Neo4j in CI e2e.
- Risks: strict gating can reject data that used to ingest. Mitigation: severities - only BLOCKING stops a batch; first release ships WARNING for rules with no prior enforcement and a per-tenant strictness setting.
- Accept: no BLOCKING-invalid record is written; every rejection is queryable with rule IDs.

### Phase 2 - Versioned schema registry (D2)
- Files: `ontology_registry.py`, `domain_ontology.py`, new `graphrag/graph/schema_registry.py`, `schema.cypher` (unique on `(schema_hash, tenant)` - currently missing), startup hook in the API/workers, ingestion stamp on Document + validation reports + `QueryResult` + cache key.
- Tests: activation/deactivation/version change/hash mismatch/rollback; idempotent init; tenant isolation; per-dataset (not prefix-glob) selection.
- Risk: hash definition change makes every existing version "new". Mitigation: keep old rows, mark superseded, one-time idempotent backfill.
- Accept: unknown schema is refused; drift detected at ingest and startup; every report and answer names its schema version.

### Phase 3 - Dependency tracking and targeted invalidation (D3)
- 3a: extend the cache provenance index and `ContextManifest` to record docs/chunks/entities/edges/communities; a `DependencyIndex` reverse walk; an `InvalidationEvent` (+ trace of *why*); wire corrections, supersession, `confidence/transition`, expiry sweeps and ER revisions to it; close orphaned community snapshots.
- 3b: inferred edges store premise keys and rule version; retract/mark when a premise changes.
- 3c: `VALID -> NEEDS_REVIEW -> RECOMPUTING -> VALID` plus `INSUFFICIENT_EVIDENCE | VALIDATION_FAILED | RECOMPUTE_FAILED` on summaries, inferred edges and decision records; recompute consumer on RabbitMQ; idempotent on repeated events; tenant-scoped.
- Tests: SUPERSEDES, corrected ownership, expired fact, schema-version change, multi-hop downstream, idempotent replay, tenant isolation.
- Risk: largest phase; partial delivery must leave the global revision bump as the fallback so nothing regresses. Accept: changing one fact invalidates only its dependents and nothing else (asserted by test).

### Phase 4 - Query-aware routing (D4)
- Files: `query_planner.py`, `hybrid_retriever.py`, `controlled_query.py` (add count/group templates), `adaptive_router.py` (reward scale consistency), settings; new labeled eval fixtures `evals/routing_cases.json` (expected route per question: factual/entity/relational/multi_hop/aggregation/temporal/ambiguous/stale/conflicting/unauthorized).
- Metrics tooling: route accuracy, fallback rate, insufficient-context rate, p50/p95, tokens, context precision/recall, faithfulness.
- Honesty rule: route accuracy and routing behavior are measurable offline and will be reported; **quality/latency deltas need the live stack and will be reported as "not measured" if I cannot run it.**
- Accept: deterministic routes with recorded reason; no unbounded fallback; ACL path unchanged.

### Phase 5 - Trust metadata in retrieval (D5, D6)
- Files: `models.py`, `neo4j_client.py` retrieval queries, `evidence_fusion.py`, `contradiction_detector.py`, `sufficiency.py`, `document_authority.py`, `inference_engine.py`.
- Change: add `origin`, `verification_status`, `generated_by`, `verified_by`, `stale_after`; make retrieval read `confidence_state` and exclude RETRACTED, expired, stale; apply one shared temporal/validity predicate to **every** path in the E2 matrix (entity context, agentic, controlled query, MCP lookup, SPARQL); conflict resolution sets DISPUTED and uses `winner_doc_id`; always expose score components.
- Tests: conflicting sources, authority, expiry, generated-unverified, verified correction, explainability.
- Risk: excluding RETRACTED/stale changes answers; gated by config with the default matching today's behavior until the eval shows no regression.

### Phase 6 - Explanation / proof traces
- Files: `QueryResult` + new `Explanation` model, `hybrid_retriever.py`, `messaging/consumers.py` (stop dropping fields), `api/routes/query.py`, `context_graph/repository.py` + routes (ACL-filter or owner-scope traces), `inference_engine.py` (derivation via Phase 3b premises).
- Tests: cited evidence supports the answer; unauthorized nodes never appear; inferred facts show derivation; insufficient evidence yields an honest failure.

### Phase 7 - Guarded operations
- Current state is better than the spec assumes (no tool accepts Cypher). Remaining: read-only Neo4j access (`READ_ACCESS` for `kind="read"`), per-call timeout + result cap, enforce `additionalProperties:false`, central `requires_approval` enforcement in `registry.call`, bounded `capability` metric label, approval + token-derived actor on `erase_entity`/`quarantine_entity`, ACL gating of REST agent tools, `/agent/audit` actually persisted.
- Tests: the adversarial matrix (tenant-filter removal, scope override, injection incl. parameter-position fuzz, unrestricted traversal, oversized results, unauthorized mutation, fabricated operation names).

### Observability (across phases)
Low-cardinality counters/histograms only (no query text, entity ids or tenant ids as labels): validation failures by rule+severity, quarantined records, schema drift, invalidated/recomputed artifacts, route distribution, fallback and insufficient-context rates, stale-evidence attempts, ER ambiguity, retrieval latency by mode, expansion depth, authorization denials. Fix the existing `tenant` label on `graphrag_quota_rejected_total`.

### Phase 8 - ContextItem
Deferred until 1-7 are stable, per the spec. Adapters over `EvidenceBundle`/`ContextManifest`/`CitationEvidence`, no data migration.

## 6. Acceptance gate for the whole effort
Each criterion in the request maps to a phase above. A criterion is only marked done when its
tests pass **and** (for anything depending on real Cypher) the CI e2e job is green. Anything I
cannot run locally is listed under "unverified" in the final summary, not claimed.

## 7. Open questions for you
1. **D1** (pre-write gate, not in-graph staging): OK? It changes where "STAGED" lives.
2. Should I **start Docker Desktop** (or will you) so e2e/Cypher claims can be verified locally? Otherwise they are CI-only.
3. May I **push** `77cf41e` (pyarrow cap)? It fixes the failing nightly and is independent of this work.
4. Build order: I recommend Step 0 -> Phase 1 -> 2 -> 3a -> 5 -> 4 -> 6 -> 7. Phase 4 is later than the spec's numbering because routing evaluation needs Phase 5's trust predicates to be meaningful.
