# Context Graph gap plan

**Date:** 2026-08-18
**Branch:** `fix/inf01-recall-and-graph-fact-synthesis`
**Method:** repository inspection + measured test baseline, before any code change.

Status vocabulary follows `tasks/lessons.md` **A154** — a claim must match the
strongest evidence that actually exists:

| Term | Means |
|---|---|
| **live-validated** | exercised against real Neo4j / real services |
| **wired** | reachable from a production entry point (API, worker, CLI) |
| **unit-tested** | covered by tests, but only against mocks |
| **exists, unwired** | code present, no production caller |
| **missing** | not in the repository |

---

## 1. Baseline (Priority 0) — measured

| Check | Command | Result |
|---|---|---|
| Unit tests | `pytest tests/unit -q` | **1081 passed**, 0 failed, 72.7 s |
| Integration + load | `pytest tests/integration tests/load -q` | **39 passed, 1 skipped**, 43.6 s |
| E2E | `pytest tests/e2e -q` | requires Docker; 1 test self-skips without it |
| Lint | `ruff check .` | **49 pre-existing errors** (baseline — do not increase) |
| Type check | — | **not configured**; no mypy/pyright in `pyproject.toml` |

Total green baseline: **1120 passed, 1 skipped, 0 failures.**

### Major components

```
api/                  FastAPI: /query /ingest /kg/* /context-graph/*
workers/              ingestion | query | evaluation workers (RabbitMQ consumers)
graphrag/ingestion/   loader → chunker → embedder → extractor → graph_writer
graphrag/graph/       Neo4j client, ontology registry, alias registry, quarantine,
                      bitemporal, contradiction, inference, SHACL/OWL/SPARQL
graphrag/retrieval/   hybrid (BM25 + vector) → rerank → GNN → context builder
graphrag/context_graph/  CG* decision-trace model, repository, policy engine
graphrag/evaluation/  golden eval, RAGAS, entity-resolution, domain eval
graphrag/observability/  OTel tracing, Prometheus metrics, cost attribution
```

### Ingestion flow

`POST /ingest` → RabbitMQ → `IngestionConsumer` → `IngestionAgent`:
**Phase 1 (`extract`)** chunk → embed → LLM extract (semaphore-bounded) → embed entities.
**Phase 2 (`write`, serialized)** `begin_corpus_update` → `write_document` →
`write_chunks` → per chunk `write_entities` (alias → embedding dedup → MERGE) →
`write_relations` → `validate_and_check_cycles` (validate, self-loop removal,
cycle detect, auto-quarantine, contradiction scan, community rebuild, PageRank)
→ optional Wikidata linking → `complete_corpus_update`.

### Retrieval flow

query → classify (`query_planner`) → adaptive route (`AdaptiveRetrievalRouter`)
→ local (BM25 + vector) ∥ global (community) → cross-encoder rerank → GNN
rescoring → `ContextBuilder` (dedup, near-duplicate drop, top-k cap, reserved
hop slots) → LLM synthesis → citations. `HybridRetriever._record_context_trace`
writes a Context Graph trace per governed query.

### Graph model

Node identity is **natural-key MERGE**, not the in-Python `uuid4`:

| Node | Merge key |
|---|---|
| `Document` | `(tenant, filename)` |
| `Chunk` | `(document_id, chunk_index)` |
| `Entity` | `(name, type, tenant)` |
| `RELATES_TO` | `(src, tgt, relation)` |

Constraints/indexes: `graphrag/graph/schema.cypher` (48 stmts, incl. 4 vector +
3 fulltext indexes), `graphrag/context_graph/schema.py` (16 constraints +
8 indexes on 17 `CG*` labels), `graphrag/business/schema.py` (12). Applied
idempotently at startup by `neo4j_client.init_schema()`. There is **no versioned
migration framework** — schema is re-applied DDL.

---

## 2. Requirement status by priority

### P1 — SHACL validation

| Requirement | Status |
|---|---|
| SHACL engine | **wired** — `graphrag/graph/shacl_validator.py`, 11 unit tests |
| Real mutation gate | **wired**, relational path only — `ingestion/relational.py:306` raises on non-conformance |
| Post-export validation | **wired** — `scripts/export_rdf.py --validate` (log-only, no non-zero exit) |
| Shapes in `ontology/shapes/` | **missing** — shapes are inline Python strings (`shacl_validator.py:48,104`); no `.ttl` file exists anywhere |
| Error vs. warning severity | **missing** — everything defaults to `sh:Violation` |
| Machine-readable report | **missing** — only `results_text: str` |
| Metrics (validated / failures by shape / quarantined) | **missing** |
| Reject-or-quarantine on the **LLM** path | **missing** — see finding F5 |
| CI SHACL step | **missing** |
| Entity-level / affected-subgraph validation | **partial** — `validate_relational_batch` is batch-scoped, not whole-graph |

### P2 — Ontology lifecycle

| Requirement | Status |
|---|---|
| Explicit version | **wired** — `ontology.version` strict semver, all 7 YAMLs at `1.0.0` |
| Status lifecycle (`draft/active/deprecated`) | **wired** — `domain_ontology.py:78` |
| Deprecation without deletion | **wired structurally** — `deprecated_types`/`deprecated_relations` + required `migration_map`; **all empty in every shipped ontology** |
| Backward-compat check | **partial** — major-version equality only; `compatible_with` is presence-checked but **never parsed** |
| Automated ontology checks | **wired** — `validate_ontology_document` (cycles, semver, UPPER_SNAKE, domain/range) |
| Changelog | **missing** — no changelog file in the repo |
| Competency questions | **missing** |
| `ontology/README.md` | **missing** |
| Migration mechanism | **partial** — `ontology_migration.py` + `migration_map`, unpopulated |

**Dead ontology config (found):** YAML `inference_rules:` are **never wired into
production** — `api/routes/kg/inference.py:31` constructs `ForwardChainingEngine`
without `rules=`, silently falling back to 6 hardcoded `DEFAULT_RULES`.
`authority_levels`, `document_prefixes`, `supersession_chains` are not read by
the loader. `settings.ontology.enforce_domain_range` and
`allow_migration_renames` are **read by no Python file**.

### P3 — Context and decision trace

Substantially **implemented** — this is the repo's strongest area.

**Present:** 17 `CG*` labels, 25+ relationship types, bitemporal `CGBase`
(`valid_from/valid_to/transaction_from/transaction_to`), `ContextManifest` with
SHA-256 integrity hash, `Decision`/`Option`/`PolicyVersion`/`PolicyEvaluation`,
`CGApproval`/`CGExceptionGrant`/`CGCorrection`, `CGAction`/`CGOutcome`/
`CGFeedback`, `CGRedaction`, supersession with cycle guard, ~35 tests, one live
Neo4j integration test. ADR explicitly prohibits storing chain-of-thought.

**Missing concepts** (zero occurrences repo-wide): `Conversation`, `Meeting`,
`Utterance`, `Mention` (as a node), `Need`, `Objection`, `Risk`, `Commitment`.
`CGEpisode` is the nearest analogue — a session event log with no speaker
identity, no utterance timestamp, and no conversation container.

**Per-assertion fields missing:** source fragment/span, source timestamp
(distinct from write time), confidence, per-assertion verification status.
Provenance and model/prompt versions exist only at *run/manifest* level.

**Wiring gap:** `graphrag/ingestion/` and `workers/` contain **zero** imports of
`context_graph`. The decision trace is live for **queries**, dead for
**ingestion**.

### P4 — Uncertain entity resolution

**Staged resolver exists** (`alias_registry.py`, 503 lines): exact → normalized →
regulatory-prefix → Romanian stem → rapidfuzz ≥85 auto-merge → 70–84 ambiguous
band → embedding ANN ≥0.92 auto-merge → 0.85–0.92 review band → new entity.
No LLM in the resolution path (correct per the brief). Review queue with
approve/reject and 4 HTTP endpoints exists.

**Gaps:**
- **No `Mention` node**; `MENTIONS` edge carries **zero properties** — no surface
  form, no span, no confidence. Surface form survives only as an `Alias` node,
  and only when the resolved name differs.
- **No `POSSIBLY_REFERS_TO`** or any uncertainty edge — zero repo-wide.
- **No resolver version** stamped anywhere. Re-tuning a threshold leaves no way
  to identify affected merges.
- **Only the single best candidate** is retained; a genuine 3-way ambiguity is
  recorded as one pair.
- **Rejections are not durable** — see finding F6.
- `Entity.canonical_id` is declared (`core/models.py:75`) and **never assigned or
  read**.

### P5 — Incremental / idempotent ingestion

**Good:** natural-key MERGE on all hot-path writes; the A136 duplicate-chunk
regression is fixed and unit-tested (`test_ingestion_idempotency.py:312`);
RabbitMQ retry (max 3, exponential backoff) with a real DLQ envelope;
corpus-revision fencing; `ingest_complete` checkpoint in the bulk CLI.

**Gaps:**
- **No content hashing** on `Document`/`Chunk` — the pipeline cannot tell whether
  a re-ingested file changed; every re-ingest re-chunks, re-embeds, re-calls the LLM.
- **Relation confidence inflates on re-ingest** — finding F2.
- **No tombstones** — `tombstone`/`soft_delete`/`is_deleted`/`deleted_at` are
  zero-hit repo-wide; a source document that disappears is never reconciled.
- **Stale `MENTIONS` and `RELATES_TO` are never retracted** when a re-ingested
  document no longer supports them.
- **No `extractor_version` / `pipeline_version`**; `prompt_version` is a
  hardcoded `"v1"` with no bump mechanism; no `ontology_version` stamped on
  ingested nodes.
- Queue path has **no checkpointing** — `ingest_complete` is set only by the CLI.
- `Entity.valid_from/valid_to` are **queried** by `bitemporal.py:107` but never
  written by any ingestion path; no such model field exists.
- Bitemporal transaction time is a single `recorded_at` point, not the
  `system_from`/`system_to` interval the brief asks for (the real interval exists
  only on `CommunitySummarySnapshot`).

### P6 — Multi-tenant security

Audit still in progress at time of writing; `tests/unit/test_tenant_isolation.py`
carries **65 tests** and `tasks/lessons.md` A146–A149 record a closed
tenant-isolation hardening thread. To be completed before any P6 change.

### P7 — Retrieval

Already matches the brief's target flow (lexical + vector, rerank, bounded
traversal, temporal filters, citations, capped context). `docs/audit-2026-08-13.md`
holds the live evidence ledger. **No change proposed** — recent measured work
(A157–A159) shows unmeasured retrieval tuning tends to regress this pipeline.

### P8 — Evaluation

**Strong.** 17 eval/benchmark scripts, versioned `evals/golden_set.json` (v2.3)
with negative and ambiguous cases, machine-readable `evals/last_run.json`,
regression thresholds, `scripts/validate_eval_datasets.py`.
**Gaps:** `validate_eval_datasets.py` is not run by CI; extraction-layer metrics
(entity/relation P/R/F1) exist only as `graphrag/evaluation/entity_resolution.py`
with a 3-case synthetic set; no SHACL-validity-rate metric.

### P9 — Observability

**Wired.** OTel tracing (`observability/tracing.py`), Prometheus counters and
histograms, cost attribution, budgets, agent telemetry, health/readiness probes
per worker, `monitoring/alerts.py`.
**Gap:** no metric for validation/quarantine rates (depends on P1 work).

---

## 3. Confirmed defects (verified directly, not inferred)

| # | Defect | Evidence | Severity |
|---|---|---|---|
| **F1** | `pyshacl` is in `requirements.txt` but in **none** of `requirements/{api,ingestion,query,workers}.txt` — the only files `Dockerfile:21` installs. `validate_relational_batch` imports pyshacl lazily and `relational.py:306` is a hard gate, so **every container raises `ImportError` instead of validating**. CI installs `requirements-dev.txt`, so CI can never catch this. | `Dockerfile:21`, `requirements/*.txt`, `shacl_validator.py:204` | **High** — SHACL is dead in production while 11 tests pass |
| **F2** | Re-ingesting an unchanged document **inflates relation confidence**: `r.confidence = 1.0 - (1.0 - r.confidence) * (1.0 - $confidence)` runs unconditionally. 0.8 → 0.96 → 0.992. The adjacent `source_doc_ids` write **is** guarded against the same duplicate — the guard was simply not applied to confidence. | `neo4j_client.py:502-505`, `:580-583` | **High** — silent data corruption on every re-ingest |
| **F3** | `write_relations` calls `registry.resolve()` and subscripts the result. `resolve()` can return `AmbiguousMatch`, a plain dataclass — **truthy but not subscriptable** → `TypeError`. `write_entities:215` has the `isinstance` guard; the relation path does not. Fires whenever a relation endpoint lands in the fuzzy 70–84 band. | `graph_writer.py:390-395` vs `:215` | **High** — crashes ingestion |
| **F4** | `ingestion_agent.py:159` filters `e.confidence >= 0.85`, but `Entity` (`core/models.py:67-80`) has **no `confidence` field** → `AttributeError`, swallowed by the broad `except` at `:174`. **Wikidata linking is silently dead** whenever `WIKIDATA_LINKING=1`. | `ingestion_agent.py:159`, `core/models.py:67` | **Medium** — feature silently non-functional |
| **F5** | On the **LLM ingestion path** nothing is ever rejected or quarantined. `validate_extraction` **coerces**: unknown entity type → `CONCEPT`; domain/range violation → `RELATED_TO`. `validate_relation_triplet` can return `False` but the write path only uses its normalized name. No dead-letter, no rejection store — `dead_letter` is zero-hit repo-wide. | `ontology_registry.py:230-273` | **High** — brief requires reject-or-quarantine |
| **F6** | A **rejected** review-queue item is not durable. `AliasRegistry.load()` reads only `Entity`/`Alias`, never `ReviewQueueItem`; `reject()` writes no negative alias or blocklist. `enqueue()` uses `CREATE` with no dedup key and no uniqueness constraint, so **each re-ingest appends a fresh duplicate item forever**. Auto-merges are equally unrecoverable — no `MERGED_FROM` edge. | `review_queue.py:58,140-166`, `alias_registry.py:206-238` | **Medium** — brief explicitly requires this |
| **F7** | `consumers.py:55` reads `msg.tenant`, but `IngestMessage` has only `job_id`/`document`/`priority` → `AttributeError`, swallowed at `:57`. **GNN calibration never schedules from the queue path.** | `consumers.py:55`, `core/models.py:299` | **Medium** — feature silently non-functional |
| **F8** | Two diverging hardcoded copies of the relation rules: `ontology_registry._RELATION_RULES` and `ingestion_validator.RELATION_RULES:36-50` (the latter lacks `PART_OF`, `USES`, `RELATED_TO`). | `ingestion_validator.py:36-50` | **Low** — drift risk |
| **F9** | `test_ontology_lifecycle.py:32-35` hardcodes a **6-file list**; `sustainability_supply_chain.yml` is omitted, so a newly added ontology is never gated by CI. | `test_ontology_lifecycle.py:32` | **Low** |
| **F10** | **Cross-tenant answer exfiltration chain.** `GET /kpis/timeseries` had no scope and no tenant filter; `kpi_tracker.py:108` did `getattr(KPIEventRow, metric)` on unvalidated input; `KPIEventRow` has **no tenant column**. So `?metric=query_id` returned **every tenant's query IDs**. `GET /query/{query_id}` then had **no ownership check** and `result_store._key` is the bare `query_id`, so those ids redeemed the full stored answer and cited source text. uuid4 entropy was the only control, and step one removed it. | `api/routes/kpis.py:17`, `kpi_tracker.py:108`, `kpi_store.py:17-33`, `api/routes/query.py:93`, `result_store.py:106` | **Critical** |
| **F11** | **Cross-tenant session read + write.** `session_store.py:180` keys on `f"graphrag:session:{session_id}"` with **no tenant**, and `session_id` is client-supplied (`api/routes/query.py:25`). `session_context.enrich_query` splices another tenant's entity names into the prompt (`session_context.py:163`); `hybrid_retriever.py:692` writes this tenant's question *and full answer* into the attacker-chosen session key. Contrast `ContextGraphRepository.load_session_episodes`, which **is** correctly scoped — Redis was simply missed. | `session_store.py:180`, `session_context.py:128,163`, `hybrid_retriever.py:692` | **Critical** |
| **F12** | Cross-tenant write surfaces: `DELETE /kg/cache/flush/{tenant}` takes tenant from the **path** (`kg/health.py:99`); `POST /kg/sources` and `/kg/sources/{id}/mappings` take it from the **request body** (`kg/sources.py:17,36`). The tenant-isolation guard test can't see either shape — its regex only matches `tenant: str = "literal"` (`test_tenant_isolation.py:123-134`). | `kg/health.py:99`, `kg/sources.py:17` | **High** |
| **F13** | Globally-shared un-tenanted nodes let one tenant affect another: `RelationEmbedding` (`edge_embeddings.py:178`) is MERGEd by relation name only, so tenant A's training overwrites the relation semantics tenant B's link prediction reads. Same shape for `OntologyVersion`/`EntityType`; `get_schema_history()` returns **all tenants'** versions. | `edge_embeddings.py:178`, `ontology_registry.py:415` | **High** |
| **F14** | Google OAuth callback accepts **any** Google account (no `hd`, no allowlist) and hardcodes `tenant = settings.default_tenant` (`auth.py:214`). There is no user→tenant mapping anywhere in the codebase, so production multi-tenancy is effectively single-tenant and the isolation machinery is never exercised by real users. | `api/routes/auth.py:202-227` | **High** |

### Repository-topology risk (needs a human decision)

`ai-knowledge-graph-platform/` contains its own `.git` **and** is tracked by the
parent `Generative-AI` repo (660 files). The parent holds the live CI
(`.github/workflows/ai-knowledge-graph-platform-ci.yml`, path-filtered to
`ai-knowledge-graph-platform/**`). Commits made in the nested repo are invisible
to the parent, which reports them as uncommitted modifications — so **work
committed only to the nested repo never reaches CI**. The nested repo's index
also lists 1594 unrelated paths (`Citim-impreuna/`, sibling `.md` files)
inherited from the parent, which show as mass deletions.

**No action taken.** Resolving this (submodule, subtree, or removing one `.git`)
is an architectural decision for the maintainer.

---

## 4. What was actually implemented (2026-08-18)

Test baseline moved **1120 → 1134 passed, 1 skipped, 0 failures**; `ruff check .`
unchanged at 49 pre-existing errors, and every touched file is ruff-clean.

| Defect | Fix | Evidence |
|---|---|---|
| **F1** | `pyshacl>=0.29.0` added to `requirements/ingestion.txt` with a comment explaining why it is a hard dependency of that service | declaration only — **not** container-verified |
| **F2** | Confidence guard added to **both** `merge_relation` and `merge_relations_batch`, reading a `WITH`-snapshotted `prior_docs` so it cannot depend on `SET` evaluation order | **live-validated** against real Neo4j: re-ingesting one doc 3× holds 0.8; a new source still accumulates to 0.96. 3 unit tests |
| **F3** | `_resolved_pair()` narrows `resolve()` to a tuple-or-None; ambiguous band now means "unresolved" in the relation path, matching the entity path | 2 unit tests, incl. one asserting the candidate is **not** silently adopted |
| **F10** | `_ALLOWED_TIMESERIES_METRICS` allowlist (numeric measurement columns only, rejects rather than defaulting); `/kpis` now requires `read` scope; `GET /query/{id}` enforces tenant ownership, **failing closed** on entries with no recorded tenant; `set_status` takes a required `tenant`; the worker records `msg.tenant` on the completed result | 6 allowlist tests + 2 adversarial cross-tenant tests + 1 fail-closed test |

**Deliberately NOT fixed** (each needs its own iteration): **F11** session-store
tenant scoping touches ~10 call sites across `SessionStore`, `SessionContext`,
`local_search`, `hybrid_retriever` and the API — too broad to land safely
alongside the above. **F12**, **F13**, **F14** likewise. F11 in particular is
still a live cross-tenant read/write path and should be the next thing done.

**F10 residual:** the allowlist removes identifier enumeration, but `/kpis/*`
still returns **cross-tenant aggregate** latency/cost figures, because
`KPIEventRow` has no tenant column at all. Adding one is a persistence-format
change (Timescale hypertable + index), so it is deferred rather than done
silently.

## 5. Implementation plan

Ordered by value-per-risk, not by brief numbering. Each step keeps the 1120-test
baseline green and does not increase the 49-error ruff baseline.

**Step 1 — fix verified defects (P0/P1/P4/P5).** F1, F2, F3, F4, F7, each with a
regression test that fails before the fix. These are small, isolated, and
high-value; F1 and F2 are production data-correctness issues.

**Step 2 — P1 SHACL hardening.** Move shapes to version-controlled
`ontology/shapes/*.ttl` (loaded by path, inline strings kept as fallback so no
public API changes); add `sh:severity` so warnings and violations are distinct;
return a machine-readable report alongside the text; expose counts (validated,
failures, failures-by-shape); add a CI step.

**Step 3 — P2 ontology governance docs.** `ontology/README.md`, a changelog, and
competency questions; make the lifecycle test glob `config/ontologies/*.yml`
instead of a hardcoded list (fixes F9).

**Steps 4+ — deferred, documented as roadmap items.** P3 conversation/utterance
model, P4 `Mention` node + `POSSIBLY_REFERS_TO` + rejection durability, P5
content hashing and tombstones. Each is a schema change touching the write path
and warrants its own iteration with live verification, per A154 and the A124/
A128/A137 pattern of retrieval/ingestion changes that looked correct and
regressed something else.

**Explicitly not changed:** retrieval (P7). The evidence ledger in
`docs/audit-2026-08-13.md` and lessons A157–A159 show unmeasured tuning here
regresses quality; there is no new premise to justify another attempt.
