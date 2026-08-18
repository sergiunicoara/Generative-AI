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

### F11 — fixed 2026-08-18 (follow-up pass)

`SessionStore` now keys on `(tenant, session_id)` in **both** backends:
`graphrag:session:<quote(tenant)>:<session_id>` in Redis, and a
`dict[tuple[str, str], ...]` in the in-memory fallback — the fallback is a full
substitute for Redis whenever it is unconfigured or unreachable, so leaving it
unscoped would have reopened the leak on exactly the degraded path where it is
hardest to notice. `tenant` is **keyword-only and required** on every
`SessionStore` / `SessionContext` method that reaches storage, so a caller that
forgets it raises `TypeError` at the call site instead of silently reading a
shared namespace. `enrich_query` lost its `tenant: str = "default"` fallback for
the same reason (that default is gap **M9** in its own right).

`quote()` on the tenant segment stops a tenant named `a:b` from aliasing a
different `(tenant, session)` pair; ordinary names pass through unchanged so
keys stay greppable in `redis-cli`.

**Live-validated** against real Redis: an attacker reusing the victim's exact
client-supplied `session_id` reads 0 turns, their write leaves the victim's
history intact, and the two land in separate keys
(`graphrag:session:victim:…` vs `graphrag:session:attacker:…`).
7 new adversarial unit tests (read leak, write poisoning, tenant-scoped
`clear`, key shape, colon-aliasing, required-kwarg enforcement).

### F12 — fixed 2026-08-18 (follow-up pass)

All three routes now reject a client-supplied tenant that disagrees with the
token, rather than honouring it:

| Route | Was | Now |
|---|---|---|
| `DELETE /kg/cache/flush/{tenant}` | tenant from **path** | path kept (URL shape unchanged), checked against token |
| `POST /kg/sources` | tenant from **body** (`SourceSystem.tenant`) | `Depends(get_tenant)` + reject on mismatch |
| `POST /kg/sources/{id}/mappings` | tenant from **body** (`SourceMapping.tenant`) | same |

The reject-don't-overwrite helper already existed as
`context_graph._assert_body_tenant`; it moved to
`api/auth/dependencies.assert_request_tenant` (next to `get_tenant`, where a
route author looking for the tenant dependency will find it) and the old
private name is now a thin delegating alias so existing call sites and tests
are untouched.

**The guard test that missed all three is rewritten.** The old check was a
regex for `tenant: str = "literal"`, which structurally cannot see a bare path
param, a `Query(default=…)`, or a tenant nested in a request-body model — i.e.
it could never have caught any of these. The replacement walks the AST of every
route handler and requires: a `tenant`/`token_tenant` parameter to default to
`Depends(get_tenant)` or the handler to call the reject helper; and any
parameter annotated with a **tenant-bearing model** (30 such models found by
scanning `graphrag/` and `api/` for classes declaring a `tenant` field) to be
accompanied by that call. Dev-only handlers are exempt, but the exemption is
**derived from the code** (the handler calls `is_dev_env`) rather than a
hand-maintained name list, so deleting a dev gate re-arms the test instead of
silently widening the exemption. The old regex test is kept alongside it.

Tests: 1140 → 1146 passed, 1 skipped, 0 failures. ruff held at 49 (one new
`F401` introduced by the delegation was found and removed before commit).

### F13 — fixed 2026-08-18 (follow-up pass)

The audit lumped three node types into one item; code inspection found three
different problems of three different severities, so the fix is scoped
accordingly rather than uniformly:

**`RelationEmbedding` — the real corruption, the only one needing a data
migration.** Two sources have different ownership: `source='derived'` is a
pure function of the relation name (SHA-256 seed → fixed RNG draw — identical
for every tenant, safe to share); `source='trained'` is TransE fitted to one
tenant's edges. Both used to `MERGE` on `{relation}` alone. Now: derived
embeddings are cached under a `DERIVED_SCOPE` sentinel tenant value so a
shared-cache write can never collide with (and overwrite) a real tenant's
trained node of the same name; trained embeddings key on
`{relation, tenant}`. Reads prefer the caller's own trained vector, fall back
to the shared derived one, then to in-process derivation.
`EdgeEmbeddingService`'s cache and `TransXTrainer`'s shared working dict (the
same object, passed by reference) both moved from bare `relation` string keys
to `(tenant, relation)` tuples — the trainer needed matching changes or its
persist loop would have bound a tuple as a Cypher string parameter.

Migration (`scripts/migrate_relation_embeddings.py`, dry-run by default):
existing `trained` nodes carry no tenant — there was never a property to
read, so attributing them to any specific tenant would be inference, not
fact. **Deleted** rather than migrated; nothing breaks, since the code already
falls back to the deterministic derived vector, so this only degrades
retrieval to the documented fallback until each tenant re-runs
`POST /kg/edge-embeddings/train`. Existing `derived` nodes are backfilled
with `tenant = DERIVED_SCOPE` in place. **Live-validated** against real
Neo4j: seeded legacy-shaped (no-tenant) probe data, ran the migration, and
confirmed idempotency (a second `--apply` run finds nothing left to do).

**`OntologyVersion` — disclosure, not corruption, and unwired today.**
`get_schema_history()` had no route calling it (checked before fixing —
lower severity than the initial audit implied), but its unfiltered read and
the version node's `{schema_hash}`-only merge key were both fixed the same
way as the rest of F13 rather than left as a trap for whoever wires it in
next: the merge key is now `{schema_hash, tenant}` (two tenants loading a
byte-identical ontology must not collapse into one shared governance
history), and `get_schema_history(tenant)` now takes tenant as a required
parameter.

**New finding, not in the original audit:** `load()`'s "known relation types
from existing graph" query scanned every tenant's `RELATES_TO` edges
unfiltered, feeding `_known_relations` — which drives `validate_extraction`'s
drift detection. Unscoped, tenant A's vocabulary silently suppressed
"new relation" drift warnings for tenant B. Fixed with the same
belt-and-suspenders tenant filter (both endpoint entities and the edge's own
`tenant` property) used elsewhere in `neo4j_client.py`.

**`EntityType` — left alone**, per the approved scope: `MERGE` over a
module-level hierarchy constant is a shared vocabulary by design, and the
audit's own finding was that tenant-specific `extra_pairs` are only ever
passed from a demo script, not a production path.

**Live-validated** end-to-end against real Neo4j (not just mocked): a
simulated tenant-A training write followed by a tenant-B read confirms zero
leakage, tenant A reads its own vector correctly, the fallback matches direct
derivation, and the shared derived cache stays genuinely shared and correctly
tagged.

Tests: 1146 → 1166 passed, 1 skipped, 0 failures (measured, full suite,
including 4 fixed by the trainer key-shape correction). 39 new tests across 3
new/extended files — this class had zero prior coverage. ruff held at 49 (one
new `F401` from an unused test import, caught and removed before commit).

### F14 — fixed 2026-08-18 (follow-up pass)

Unlike F11–F13, this was a missing **feature**, not a missing check: closing
"any Google account gets `default_tenant`" requires *something* to decide
which tenant an email belongs to before OAuth can issue a scoped token. Three
shapes were weighed against the user (see the design note this fix followed):
Google Workspace domain (`hd` claim) auto-mapping (doesn't cover personal
accounts, so not a full fix on its own), an explicit admin-provisioned table
(chosen), and self-service tenant creation (rejected outright — a "tenant" in
this codebase is a whole isolated customer, not something a signup should be
able to spin up unilaterally).

**Implementation**, mirroring the M2M client registry (`register_client`)
already shipped in `api/routes/auth.py` — same Redis-hash-with-in-memory-
fallback storage shape, same scope-intersection escalation guard:

- `api/auth/user_provisioning.py` — `graphrag:user_tenant_map` Redis hash,
  keyed by normalized (lowercased, stripped) email.
- `POST /auth/users` — `require_scope("admin")`-gated. **Tenant is not a body
  field** — it comes from the caller's own `Depends(get_tenant)`, applying
  the F12 lesson directly: an admin must not be able to provision a user into
  a tenant other than their own by naming a different one in the request.
  Granted scopes are intersected with the provisioning admin's own (same
  guard `register_client` uses), so an admin can never grant a user privilege
  they don't hold themselves.
- `GET /auth/users`, `DELETE /auth/users/{email}` — both `admin`-gated and
  tenant-scoped; revoking a record that belongs to a different tenant returns
  404, not 403, so the caller can't confirm the email is provisioned
  *somewhere else*.
- `GET /auth/callback` rewritten: looks up the authenticated Google email in
  the table. **Found → issues a token for that tenant with exactly the
  provisioned scopes** (no widening at login time — they were already capped
  at provisioning). **Not found → 403**, not a silent default. This is the
  actual fix.

**Named limitation, not solved by this fix**: bootstrapping the *first*
admin-scoped token in a production deployment has no dedicated mechanism —
that gap pre-dates F14 and is orthogonal to it. `POST /auth/dev-token`
already grants the full scope set (including `admin`) but is `is_dev_env()`-
gated, so it covers local development; a production bootstrap story is a
separate, undecided piece of work.

**Live-validated** against real Redis (not just mocked): set/get/list/delete
round-tripped through the actual `graphrag:user_tenant_map` hash, confirmed
present via a raw `HGET`, confirmed tenant-scoped listing excludes other
tenants, confirmed cleanup.

Tests: 1166 → 1191 passed, 1 skipped, 0 failures (measured, full suite). 25
new tests across 2 new files — zero prior coverage existed for
`/auth/callback` or the M2M registry it mirrors. ruff held at 49.

**F11–F14 are all now fixed.** Remaining roadmap items are P2/P3/P8/P9
documentation and the deferred P5 schema-touching work noted in the original
plan (Priority 4/5 sections above).

### Step 2 (P1 SHACL hardening) and Step 3 (P2 governance docs) — done 2026-08-18

Resumed the original Step 2/3 plan (Section 5) after the F11–F14 detour.

**Shapes moved to version-controlled files.** `ontology/shapes/export.shapes.ttl`
and `ontology/shapes/ingestion.shapes.ttl` are now the source of truth,
loaded by path. The inline Python-string constants remain as a last-resort
fallback (used only if the file is missing at runtime — a packaging issue,
not normal operation) and are kept byte-identical to the files, verified by
a test that loads both and compares triple counts.

**Named shapes, not anonymous `[]` blank nodes.** Every shape now has a
stable IRI (`shapes:EntityLabelProperty`, `shapes:AxiomConfidenceRangeProperty`,
etc.). This is what makes "failures by shape" grouping meaningful — a blank
node's identity isn't stable across validation runs, so grouping by it
would have produced a different, meaningless key every time.

**`sh:severity` added explicitly to every constraint**, distinguishing
`sh:Violation` (fails validation) from `sh:Warning` (visible in the report
and counts, does not fail). One new, grounded warning-tier constraint: an
`owl:Axiom`'s confidence annotation may legitimately be absent —
`export_rdf.py` only writes it when the source Neo4j edge's confidence is
non-null — so its absence is now a warning, not indistinguishable from a
genuinely malformed confidence value (still a hard violation when present
but out of range or wrong datatype).

**Non-obvious pyshacl finding, verified empirically before relying on it**:
without `allow_warnings=True`, pyshacl's `conforms` flips to `False` for a
*bare* `sh:Warning` result too — not documented anywhere obvious in pyshacl's
own docs. Without this flag the whole severity distinction would have been
cosmetic: every warning would still silently reject. Confirmed both ways
with a throwaway script before writing the real implementation.

**Machine-readable report added additively.** `validate()` and
`validate_relational_batch()` keep their exact original `(bool, str)`
signatures — zero existing callers or tests needed to change. New
`validate_report()` / `validate_relational_batch_report()` return a
`ShaclReport` dataclass: `.conforms`, `.text`, `.results: list[ShaclResult]`
(per-violation focus node / message / shape / severity), `.counts` (total /
violations / warnings), `.failures_by_shape`.

**Metrics** — `graphrag_shacl_records_validated_total`,
`graphrag_shacl_validation_failures_total` (labeled by severity),
`graphrag_shacl_failures_by_shape_total`, all labeled by `target`
(`export` | `relational_batch`). Same optional-`prometheus_client` pattern
already used in `graphrag/observability/*` — a no-op if the package isn't
installed, not a hard dependency added by this change. **Not added**: a
"quarantined records" metric — nothing quarantines on a SHACL failure today
(the relational path rejects via `ValueError`; the LLM-extraction path
coerces, per F5, still open). A metric that would always read zero was not
worth fabricating; F5 remains the tracked gap for when SHACL becomes a real
quarantine trigger.

**F9 fixed**: `test_ontology_lifecycle.py`'s hardcoded 6-file list (which had
silently omitted `sustainability_supply_chain.yml`) is now a glob over
`config/ontologies/*.yml`. Verified the previously-untested file: it passes
validation, confirming this was a real coverage gap, not a file that would
have failed if it had been included.

**`ontology/README.md`** — governance/process doc, distinct from the existing
`docs/ontology-model.md` (which covers the technical model): naming
conventions and what's actually enforced (with exact line citations,
verified against the code before publishing), the lifecycle state machine,
a realistic propose → validate → review → migrate → release process matching
how this repo actually operates, and competency questions **drawn from the
live golden-eval suites** (`evals/golden_set.json`,
`data/eval_golden/queries_*.json`) rather than invented — each one is a
question the ontology is already being tested against, not an aspiration.

**`ontology/CHANGELOG.md`** — records the current baseline (all 7 ontologies
at `1.0.0`, no deprecations yet) honestly as a starting point, not
reconstructed history, plus a template for future entries that requires
linking a concrete reason and stating which golden-eval questions were
re-verified — matching this repo's own evidence-over-inference convention
(`tasks/lessons.md` A154).

**CI step — explicitly NOT added, flagged rather than silently skipped.**
The live GitHub Actions workflow
(`.github/workflows/ai-knowledge-graph-platform-ci.yml`) lives in the
**parent** `Generative-AI` repository root, not in this nested repo — GitHub
Actions only reads workflows from a repository's actual root, confirmed
earlier this session. Editing a copy inside this nested repo would have no
effect on real CI runs; editing the parent repo's workflow file is a
cross-repo, outward-facing change to live automation that wasn't part of
what was asked. Added a `make test-shacl` target instead
(`test_shacl_validator.py` + `test_ontology_lifecycle.py` +
`test_ontology_registry.py` + `test_export_rdf.py` +
`test_relational_ingestion.py`) that a CI step can call once the workflow
question is resolved.

**Live-validated end-to-end**, not just unit-mocked graphs: ran
`scripts/export_rdf.py --tenant aerospace --validate` against the real
aerospace corpus (447 entities, 648 edges, 8910 RDF triples) — loaded shapes
from the actual `ontology/shapes/export.shapes.ttl` file on disk, conformed
cleanly with 0 violations, 0 warnings.

Tests: 1191 → 1205 passed, 1 skipped, 0 failures (measured, full suite; one
test-authoring bug caught and fixed along the way — a `minCount 1` test
wrongly assumed an empty-string literal fails a minCount check, corrected to
exercise a value the code path can actually produce). ruff held at 49.

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


---

## Step 4 — P5 incremental ingestion (content hashing + tombstones), 2026-08-18

Scoped deliberately to the two items the plan named. **P3 and P4 were NOT
done here** — see "Scope correction" below.

### Content hashing — fixes a correctness bug, not just cost

The bulk ingest CLI's checkpoint was **binary**: a `Document` carrying
`ingest_complete = true` was skipped on every later run, so a source file
that had been **edited** was never re-ingested at all without a full
`--wipe`. Hashing turns that into the correct three-way decision:
unchanged → skip, changed → re-ingest, new → ingest.

- `graphrag/core/content_hash.py` — `compute_content_hash` (sha256 of the
  decoded text, not raw bytes: the same document re-saved with different
  line endings is the same document, and re-running a multi-minute LLM
  extraction over a CRLF change would be a false positive) and
  `content_changed`, where an **absent** stored hash means "re-ingest",
  never "assume unchanged".
- `Document.content_hash`, computed once in `document_loader.load_document`
  at the single point every format (PDF/DOCX/TXT/MD) converges to `str`.
- Persisted by `merge_document`; read back by the new
  `Neo4jClient.get_document_states`.

**Subtle guard, nearly missed:** `merge_document` writes the hash at the
*start* of a document's write, so a run that crashes midway leaves a hash
already matching disk. Skipping on hash **alone** would freeze a
half-ingested document out of every future run — a worse bug than the one
being fixed. The CLI therefore requires hash-match **AND**
`ingest_complete`; `ingest_complete` is kept, not removed. Pinned by
`TestPartialIngestIsNotSkipped`.

### Tombstones — soft delete, never physical

`Document.is_deleted` / `deleted_at`, set by
`Neo4jClient.tombstone_documents` when a source file disappears from the
corpus. Deliberately **not** a physical delete: erasing data is GDPR
erasure's job (`graphrag/graph/gdpr.py`) and is irreversible, whereas a file
vanishing from a corpus directory is far more often a sync glitch, partial
checkout, or rename than a genuine deletion. Tombstoning hides the
document's chunks from retrieval while leaving everything recoverable — the
same shape `quarantined` already uses for entities.

Filters added to the **six** query sites that feed retrieval or citations
(both `vector_search_chunks` branches, `bm25_search_chunks`, the BM25 entity
search, `get_best_chunk_for_document`, and community source documents). The
`OPTIONAL MATCH` sites use `(d IS NULL OR ...)` so a chunk with no
`PART_OF` edge still passes — filtering those out would silently drop
legitimately orphaned chunks, a behaviour change beyond this feature.

Resurrection is automatic: `merge_document` clears `is_deleted`/`deleted_at`,
so a file reappearing on disk becomes retrievable again rather than needing
an operator to notice.

### Verification

**Live-validated against real Neo4j** on a scratch tenant: hash written and
read back; tombstone applied; re-tombstoning returns 0 (idempotent, safe to
run every ingest); tombstoned document still present in the graph
(recoverable); its chunk correctly excluded from a retrieval-shaped query
while the sibling chunk remained visible; resurrection cleared the flag and
updated the hash.

Every modified Cypher query was `EXPLAIN`-checked against live Neo4j.
**One exception, stated rather than glossed:** the native-vector branch of
`vector_search_chunks` uses Neo4j 2026.x `SEARCH ... IN (VECTOR INDEX ...)`
syntax and cannot be parsed by the local Neo4j 5.20 — a pre-existing
limitation of the dev environment, not something this change introduced. That
branch's edit was verified by inspection only.

Tests: 1205 → **1221 passed**, 1 skipped, 0 failures (measured, full suite).
ruff held at 49.

### Scope correction — P3/P4 belong to a different repository

Work on P3 (Conversation/Utterance/Mention) and P4 (`Mention` node,
`POSSIBLY_REFERS_TO`, resolution-rejection durability) was **started and then
deliberately reverted** after checking the sibling `sales-context-graph`
repository, which already implements that entire vocabulary natively:

| Brief item | Already exists in `sales-context-graph` |
|---|---|
| Conversation, Mention | `src/domain/conversation.py` |
| Objection, Commitment | `src/domain/knowledge.py` |
| Claims / assertions | `src/domain/assertion.py` |
| Entity resolution + review queue | `src/resolution/`, `src/review/` |

That repo's own README states the brief's use case almost verbatim
("identify the objection raised by a stakeholder in the latest relevant
call... with an explainable entity-resolution decision"), over Gong-shaped
transcripts and Salesforce-shaped CRM. This platform's domain is regulatory
compliance Q&A (aerospace ADs, IATF, pharma, adtech) and its
`graphrag/context_graph/` module already has a fit-for-purpose vocabulary
(`Decision`, `Action`, `Outcome`, `ToolCall`, `CGEpisode`).

Building `Need`/`Objection`/`Commitment` here would have imported another
product's ontology into a codebase with nothing to produce or consume it.
P5 was kept because it is genuinely domain-neutral and a real gap here.
