# Current: Deterministic ingestion via LLM extraction cache (2026-06-07)

## Context
Groq/DeepSeek are NOT deterministic at temperature=0 (batched GPU/LPU inference +
mid-run Groq→DeepSeek fallback on rate limit). Each `--wipe --commit` re-ingest of
the same 12-doc corpus produces different entity/edge/conflict counts, which keeps
breaking the hiring-strategy demo script's hardcoded examples (already hit twice —
the FAA AD SUPERSEDES chain and the Boeing conflict rows both vanished between runs).

## Plan
- [x] Design: content-addressable disk cache keyed by sha256(model|temperature|json_mode|prompt),
      hooked ONLY into `Extractor.extract()` (the call site responsible for graph shape) —
      NOT retrieval/query LLM calls, which must stay live.
- [x] New module `graphrag/core/llm_cache.py` — `LLMCache` (get/set, JSON files under `.cache/llm_extraction/`) + `get_llm_cache()` singleton
- [x] New setting `llm_cache_enabled: bool = False` in `graphrag/core/config.py` (env: `LLM_CACHE_ENABLED=1`), off by default so production ingestion of new docs always hits the live LLM
- [x] Wire into `graphrag/ingestion/extractor.py::Extractor._generate()` — cache lookup before `get_llm().generate()`, store on miss (reads the flag fresh from settings each call so it works with `Extractor.__new__()` test fixtures too)
- [x] Add `.cache/` to `.gitignore`
- [x] Run unit tests — all 362 pass; smoke-tested `LLMCache` get/set/key-isolation directly
- [x] Update `tasks/lessons.md` with the pattern (A96)

## Verification
```powershell
$env:LLM_CACHE_ENABLED = "1"
py -3.11 scripts/ingest_corpus.py --commit --wipe   # run 1 — populates cache
py -3.11 scripts/ingest_corpus.py --commit --wipe   # run 2 — must replay cache, identical counts
```
Compare "Entities in Neo4j", "Edges in Neo4j", "Open conflicts", "Inferred edges" — must match exactly.
**Not yet run** (requires a live ingestion against Neo4j + Groq/DeepSeek keys, ~2-3 min
each — recommend running this once before your next pitch-prep session to confirm and
to populate the cache for good).

## Review
- Implementation is complete and unit-tested (362/362 pass + standalone cache smoke test:
  miss→set→hit, correct key isolation between distinct prompts).
- Design choice: cache is keyed by the *configured* model name, not whichever provider
  (Groq vs. DeepSeek fallback) actually served the request — intentional, since the goal
  is "replay whatever was returned the first time" for graph-shape reproducibility, not
  provider attribution.
- Off by default (`LLM_CACHE_ENABLED=0`) — zero behavior change for production ingestion
  of new documents. Only opt-in demo/pitch-prep re-ingestion runs are affected.
- End-to-end verification (two `--wipe --commit` runs producing identical counts) is the
  one remaining step — needs a live Neo4j + LLM-key environment to execute, which I don't
  want to kick off unprompted given it wipes the tenant's data. Recommend running it
  yourself with the command above before relying on it for an interview.

## Follow-up: single-provider override for the cache-populating run (2026-06-07)
**Why:** user is currently rate-limited ("out of tokens") on Groq. A naive run would
either stall on Groq's daily cap or mid-run fall through to DeepSeek — mixing two
providers' extraction "voices" into one cache baseline (defeats the determinism goal).
**Fix:** added a one-shot env override so `get_llm()` can bypass Groq/FallbackLLM
entirely and route straight to DeepSeek-V3 (already wired, ~$0.07/1M input tokens,
"generous rate limits, no sleep, no queuing" per its own docstring).

- [x] New setting `llm_ingest_provider: str = ""` in `graphrag/core/config.py`
      (env: `LLM_INGEST_PROVIDER=deepseek`) — empty string = unchanged FallbackLLM behavior
- [x] `get_llm()` in `graphrag/core/llm_client.py` checks the flag first; when set to
      `"deepseek"` returns a bare `DeepSeekLLM` (logs `llm_client.single_provider_override`)
      instead of constructing `FallbackLLM`. Return type widened `FallbackLLM` → `BaseLLM`
      (the actual common parent — both `FallbackLLM` and `DeepSeekLLM` extend it) and
      `_llm` singleton retyped to match.
- [x] Confirmed only `graphrag/ingestion/extractor.py` calls `get_llm()` anywhere in the
      ingestion path (`get_fast_llm()` — untouched — is agentic-retriever-only, not used
      during ingestion), so this override cannot have query-time side effects.
- [x] Full unit suite re-run after the change — 362/362 pass (no regressions)

**To use (when Groq quota resets or you want to skip waiting for it):**
```powershell
$env:LLM_CACHE_ENABLED     = "1"
$env:LLM_INGEST_PROVIDER   = "deepseek"
py -3.11 scripts/ingest_corpus.py --commit --wipe
# afterwards: Remove-Item Env:\LLM_INGEST_PROVIDER  — one-shot knob, not a permanent switch
```
Estimated cost: ~$0.05–$0.30 for the 100-chunk / 12-doc corpus. Estimated time: similar
to or better than the documented 20–30 min baseline (no Groq rate-limit stalls).

---

# GraphRAG — Production Hardening & Architecture Completion

All work completed across two sessions. Tracked retroactively per CLAUDE.md.

---

## Phase 1 — Core Architecture Gaps

- [x] Implement Redis-backed `SessionStore` with in-memory fallback
- [x] Make `SessionContext` fully async, backed by `SessionStore`
- [x] Add `get_session_store()` singleton reading from config
- [x] Add per-tenant `AliasRegistry` pool (`get_alias_registry(tenant)`)
- [x] Wire per-tenant registry into `GraphWriter` (remove global `self._alias_registry`)
- [x] Implement `CommunityManager` — staleness scoring, snapshot, rebuild gating
- [x] Implement `GraphEvaluator` — 6 semantic health metrics + snapshot persistence
- [x] Implement `QuarantineService` — quarantine/release entities, auto-quarantine anomalies
- [x] Implement `ContradictionDetector` — multi-source, directional, exclusive-state, functional types
- [x] Implement ontology migration rules in `OntologyRegistry` (`migration_map` from config)
- [x] Implement `scripts/community_rebuild.py` — CLI with `--tenant`, `--force`, `--dry-run`
- [x] Implement `api/routes/corrections.py` — split, quarantine, edge override, conflict resolve
- [x] Write `tests/integration/test_safety_paths.py` — tenant isolation, ontology, contradiction, quarantine, community

---

## Phase 2 — Structural Bug Fixes

- [x] Fix multi-source evidence: `merge_relation` accumulates `source_doc_ids` list (not overwrite)
- [x] Fix provenance update: tenant-scoped MATCH `(name, tenant)` on secondary SET query
- [x] Fix `EntitySplitter`: every MATCH/MERGE/DELETE includes `tenant: $tenant`
- [x] Fix `ContradictionDetector.scan()`: add `tenant` param; all detection methods filter by tenant
- [x] Fix `ContradictionDetector`: APOC-free flatten using `reduce(acc=[], lst IN collect(...) | acc+lst)`
- [x] Fix `ContradictionDetector`: remove hard-coded LIMITs; add configurable `scan_limit`
- [x] Fix all-pairs supersession check using Cypher `UNWIND range(i,j)` instead of collect+compress
- [x] Fix GNN formula: rename `_fallback_score` → `_text_score`; apply α correctly in no-entity path
- [x] Fix session turn recording: move `record_turn()` from `local_search` to `hybrid_retriever` (after LLM answer exists)
- [x] Fix community Leiden fallback visibility: log at ERROR with `impact`/`fix` fields; tag fallback communities

---

## Phase 3 — Config Drift & Runtime Gating

- [x] Wire `authority_weighting_enabled` config flag to actually gate `apply_authority_weights()`
- [x] Wire `community_staleness_check_on_ingest` config flag to gate staleness check in `_maybe_rebuild_communities`
- [x] Wire `auto_rebuild_communities` config flag to gate rebuild in `_maybe_rebuild_communities`
- [x] Wire `session_context_enabled` config flag in `LocalSearch.__init__`
- [x] Set production-safe defaults: `require_leiden: true`, `session_store: redis`, `auto_rebuild_communities: true`

---

## Phase 4 — Redis Strict Mode

- [x] Add `strict: bool = False` param to `SessionStore.__init__`
- [x] Raise `ImportError` at init when `strict=True` and `redis[asyncio]` missing
- [x] Add `async verify_connection()` — raises `ConnectionError` in strict mode on ping failure
- [x] Add `_log_op_failure()` — logs at ERROR in strict mode, WARNING otherwise; never drops live requests
- [x] Add FastAPI lifespan hook in `api/main.py` calling `await store.verify_connection()` at startup
- [x] Add `session_store_strict: true` to `config/settings.yml`
- [x] Update `get_session_store()` to read and pass `session_store_strict`

---

## Phase 5 — Tenant & Type Correctness

- [x] Fix `merge_relation`: add `src_type`/`tgt_type` params; MATCH on `(name, type, tenant)` for both endpoints
- [x] Fix provenance secondary MATCH inside `merge_relation` to also use `(name, type, tenant)`
- [x] Fix `graph_writer.write_relations`: capture `src_type`/`tgt_type` after alias resolution; pass to `merge_relation`
- [x] Fix `get_entity_relations_subgraph`: accept `entities: list[dict]` with `name`+`type`; filter target by composite `name:type` key
- [x] Fix `_fetch_subgraph_edges` in `local_search.py`: pass `(name, type)` pairs from `get_chunk_entity_embeddings`
- [x] Fix `GraphEvaluator`: add `tenant` param to all 6 metrics, `full_report`, `persist_snapshot`, `get_trend`
- [x] Fix `GraphHealthSnapshot` nodes: store `tenant` field at creation
- [x] Fix `find_conflicts_for_entity`: add `entity_type` + `tenant` params; MATCH on `(name, type, tenant)`; rewrite to use `source_doc_ids` list

---

## Phase 6 — Documentation

- [x] Update `README.md`: add GNN, session context, alias resolution, contradiction detection, tenant isolation, document authority, community staleness, graph health, ontology, quarantine, corrections API, strict startup mode, Redis, all new files
- [x] Create `tasks/lessons.md` — 15 code correctness patterns
- [x] Extend `tasks/lessons.md` — 18 knowledge graph architecture lessons

---

## Phase 7 — KG Architecture Gaps

- [x] Implement `graphrag/graph/negative_knowledge.py` — `NEGATIVE_RELATES_TO` edge type with same provenance model as positive edges; assert/retract/query API; conflict detection with `ContradictionDetector`
- [x] Implement `graphrag/graph/type_taxonomy.py` — `SUBCLASS_OF` hierarchy (`EntityType` nodes); `expand_type()` for query expansion; `least_common_ancestor()` for merge decisions; singleton + Neo4j persistence
- [x] Implement `graphrag/graph/bitemporal.py` — `BitemporalStore` with `as_of_entities()`, `as_of_edges()`, `transaction_diff()`, `time_travel_report()`; stamp `recorded_at` (transaction time) on entity and relation CREATE in `neo4j_client.py`
- [x] Implement `graphrag/graph/confidence_calibration.py` — `CalibrationService` with Brier score, calibration curve, isotonic correction, `CalibrationSnapshot` trend nodes
- [x] Implement `graphrag/graph/reification.py` — `ReificationService`; `Statement` nodes with `SUBJECT_OF`/`OBJECT_OF` edges; `endorse()`, `contradict()`, `add_meta()` methods
- [x] Implement `graphrag/graph/edge_embeddings.py` — `EdgeEmbeddingService`; TransE scoring (`‖ h + r − t ‖₂`); deterministic relation embeddings from name hash; `predict_missing_links()` for link prediction; `embed_all_edges()` batch
- [x] Implement `graphrag/graph/graph_snapshots.py` — `GraphSnapshotService`; `create_snapshot()` with health metrics; `diff_snapshots()` with per-field before/after/delta/pct; `list_snapshots()`, `restore_summary()`
- [x] Add `_detect_positive_negative_pairs()` to `ContradictionDetector.scan()` — 5th conflict type
- [x] Add `recorded_at = datetime()` ON CREATE to `merge_entity` and `merge_relation` in `neo4j_client.py`
- [x] Add `NegativeRelation`, `Statement`, `CalibrationSample`, `GraphSnapshot` models to `core/models.py`
- [x] Create `api/routes/kg_features.py` — 28 endpoints covering all 7 features
- [x] Wire `kg_features.router` into `api/main.py` at prefix `/kg`
- [x] Fix indentation bug in `corrections.py` `list_conflicts` handler

---

## Phase 8 — Remaining KG Architecture Gaps

### Core KG Integrity
- [x] Implement `graphrag/graph/embedding_registry.py` — EmbeddingRegistry: version tracking, incompatibility detection, `migrate_embeddings()` batch job; add `embedding_model`/`embedding_version` to Entity in `neo4j_client.py`
- [x] Implement `graphrag/graph/property_schema.py` — PropertySchemaValidator: per-type attribute cardinality rules, wire into `IngestionValidator`
- [x] Extend `OntologyRegistry` with `rename_entity_type()` — cascade update entities, WikidataLinks, edges, Statements, audit trail
- [x] Create `scripts/re_embed.py` — CLI: `--tenant`, `--model`, `--batch-size`, `--dry-run`; writes back to Neo4j with new version tag
- [x] Create `scripts/entity_type_migration.py` — CLI: `--old-type`, `--new-type`, `--tenant`, `--dry-run`

### Reasoning Layer
- [x] Implement `graphrag/graph/inference_engine.py` — ForwardChainingEngine: Datalog-style rules (transitivity, symmetry, inverse), mark inferred edges with `source_type=inferred`, configurable rule set
- [x] Extend `GNNScorer` — `propagate_confidence()` (product of edge confidences along multi-hop paths); `annotate_path_confidence()` stamps `path_confidence` on each chunk result
- [x] Extend `EdgeEmbeddingService` with `train()` — TransE negative-sampling SGD training loop; updates `RelationEmbedding` nodes with learned vectors

### Community & Clustering
- [x] Implement `graphrag/graph/incremental_community.py` — IncrementalCommunityDetector: track changed entities since last build via `CommunityRebuildPoint`, only rebuild affected communities
- [x] Extend `CommunityBuilder` with `build_semantic_communities()` — HDBSCAN on entity embeddings; compute Jaccard overlap vs Leiden; log divergence as quality metric; `SEMANTIC_MEMBER_OF` edges

### Data Quality & Compliance
- [x] Implement `graphrag/graph/gdpr.py` — GDPRService: `forget_entity()`, `forget_document()` (cascade + redact chunk text), `deletion_audit_log()`
- [x] Implement `graphrag/graph/pii_guard.py` — PIIGuard: regex + entity-type based PII detection, `redact()`, `tag_entity_pii()`, `scan_document()`
- [x] Implement `graphrag/retrieval/query_cache.py` — QueryCache: Redis-backed, key=hash(query+tenant+session), provenance-aware invalidation on document ingest

### Advanced Features
- [x] Implement `graphrag/graph/entity_linker.py` — WikidataEntityLinker: Wikidata API lookup, cache in `WikidataLink` nodes, store `wikidata_qid` on Entity
- [x] Implement `graphrag/graph/counterfactual.py` — CounterfactualAnalyzer: `simulate_retraction(doc_id)` dry-run showing what conflicts/entities disappear if a doc is removed
- [x] Implement `graphrag/graph/multimodal.py` — MultiModalEntityService: attach image/audio/video to entities, cross-modal embedding storage, inventory

### Operational
- [x] Create `scripts/kg_backup.py` — export all nodes/edges/chunks to NDJSON; local + S3 support; `restore` mode
- [x] Add API endpoints for all 13 new services to `api/routes/kg_features.py` (Phase 8: endpoints 8–20)
- [x] Update `tasks/lessons.md` with A25–A33

---

---

## Phase 9 — Tests, Alerts, Admin UI

### Phase 9a — Integration Tests
- [x] Create `tests/integration/test_operational_paths.py` — TestSessionStorePersistence (5 tests), TestLeidenStartupPath (3 tests), TestTenantScopedContradiction (3 tests), TestCommunityAutoRebuildLifecycle (3 tests)

### Phase 9b — Load Tests
- [x] Create `tests/load/__init__.py`
- [x] Create `tests/load/test_load_scenarios.py` — 5 async concurrency benchmark tests

### Phase 9c — Alert Service
- [x] Create `graphrag/monitoring/__init__.py`
- [x] Create `graphrag/monitoring/alerts.py` — AlertService with check/fire/check_and_fire
- [x] Edit `graphrag/graph/graph_evaluator.py` — call alert_svc.check_and_fire() in persist_snapshot()
- [x] Edit `api/routes/kg_features.py` — add GET /health/alerts endpoint
- [x] Edit `config/settings.yml` — add contradiction_rate, orphan_rate, low_confidence_rate thresholds

### Phase 9d — Admin Dashboard
- [x] Create `graphrag/dashboard/__init__.py`
- [x] Create `graphrag/dashboard/app.py` — Dash admin panel with 5 tabs
- [x] Edit `api/main.py` — mount Dash app at /admin

---

## Phase 10 — Audit Corrections

All 20 issues from the code audit corrected.

### P0 — Runtime failures (fixed)
- [x] Create `graphrag/core/retry.py` — async exponential-backoff retry decorator
- [x] Apply `@with_retry(TransientError, ServiceUnavailable)` to `Neo4jClient.run()`
- [x] Move `from pathlib import Path` to top of `neo4j_client.py` (was line 561, used at line 41)
- [x] Remove stale deferred `from pathlib import Path` comment at bottom of file
- [x] Fix `entity_linker._search_wikidata()` — wrap `urllib.request.urlopen` in `run_in_executor`
- [x] Add `before_request` auth guard to `dashboard/app.py` — login form + session cookie check; GDPR/rebuild/resolve callbacks now blocked without `GRAPHRAG_ADMIN_TOKEN`
- [x] Migrate `_recent_alerts` deque to Redis `LPUSH/LTRIM` in `alerts.py`; in-process deque is fallback

### P1 — Technical debt (fixed)
- [x] Replace `asyncio.get_event_loop()` → `asyncio.get_running_loop()` in 11 async files
- [x] Fix `agentic_retriever._llm()` — remove `loop.run_until_complete()` on already-running loop; call `generate_content()` directly (it is sync)
- [x] Wire `prometheus-fastapi-instrumentator` in `api/main.py` → exposes `GET /metrics`
- [x] Add `tenacity>=8.3.0` to `requirements.txt`

### P2 — Coverage gaps (fixed)
- [x] Create `tests/conftest.py` — shared fixtures (`neo4j_mock`, `make_entity`, `make_chunk`, `make_relation`, `make_turn`, `memory_session_store`)
- [x] Create `tests/unit/__init__.py`
- [x] Create `tests/unit/test_retry.py` — 9 async unit tests for retry decorator
- [x] Create `tests/unit/test_alert_service.py` — 12 unit tests for AlertService
- [x] Create `tests/unit/test_pii_guard.py` — 15 unit tests for PIIGuard
- [x] Create `tests/e2e/__init__.py` + `test_live_services.py` — testcontainers scaffold for Neo4j + Redis (auto-skipped without Docker)
- [x] Add `[tool.pytest.ini_options]` to `pyproject.toml` — `asyncio_mode = "auto"` so `@pytest.mark.asyncio` is not needed on every test

### P3 — Quality of life (fixed)
- [x] Add `.dockerignore` — excludes tests/, eval_data/, results/, .git, .venv, *.md, .env
- [x] Surface HTTP error cause/status in all dashboard render functions and action callbacks
- [x] Add per-tenant alert threshold overrides (`tenant_alert_thresholds` in `settings.yml`; `AlertService._effective_thresholds()`)
- [x] Add `Makefile` with `make test`, `make api`, `make dashboard`, `make backup`, `make lint`
- [x] Update README — added full Admin Dashboard section (5 tabs, auth, standalone mode, service URL table updated)
- [x] Narrow 4 key bare `except Exception` blocks to specific types (`extractor.py`, `ontology_registry.py`, `consumers.py`)
- [x] Add 5 new lessons to `tasks/lessons.md` (A38–A42)

### Bonus: 4 pre-existing bugs fixed in test_safety_paths.py
Running the full suite exposed 4 pre-existing test failures (first time safety paths were included in a full run):
- [x] `test_multi_source_conflict_created` — mock used `sources` key but code accesses `doc_ids`; fixed field name + added `independent_pairs` + `positive_negative_pairs` slot
- [x] `test_directional_reversal_detected` — phantom "CREATE for empty query" slots displaced real conflict row into exclusive_state slot; removed phantom slots
- [x] `test_functional_violation_detected` — same phantom slot issue with 2+2+8 fake slots; corrected to 1+1+4 actual slots
- [x] `test_mark_rebuilt_creates_new_snapshot` — mock used `entities`/`edges` but code accesses `entity_count`/`edge_count`; added missing `community_count` and 3rd call slot for SET milestone; updated assertion from 2 to 3 calls
- [x] Lesson A43 added to `tasks/lessons.md` — AsyncMock side_effect lists must exactly match actual conditional call counts

### Test counts after Phase 10
- Unit tests: 36 (test_retry: 9, test_alert_service: 12, test_pii_guard: 15)
- Integration tests: 33 (test_operational_paths: 14, test_safety_paths: 19)
- Load tests: 5 (test_load_scenarios)
- E2e scaffold: auto-skipped without Docker
- Total runnable without live services: **73 tests, all pass**

---

---

## Phase 11 — Audit Follow-ups

All 12 findings from the second code audit implemented.

### P0 — Security
- [x] `graphrag/core/config.py` — add `model_validator` that raises in production when `jwt_secret_key` or `neo4j_password` are default values
- [x] `graphrag/dashboard/app.py` — fail closed when `GRAPHRAG_ADMIN_TOKEN` is unset and `env=production` (previously open to all with no token)
- [x] `api/main.py` — tighten CORS: `allow_methods` to explicit list, `allow_headers` to explicit set

### P1 — Reliability / Ops
- [x] `.github/workflows/ci.yml` — create GitHub Actions CI: test matrix on push/PR, ruff lint job
- [x] `api/main.py` — add `GET /health/ready` readiness probe that pings Neo4j and Redis
- [x] `api/main.py` — replace deprecated `starlette.middleware.wsgi.WSGIMiddleware` with `a2wsgi.WSGIMiddleware`; add `a2wsgi>=1.10.0` to `requirements.txt`
- [x] Narrow key `except Exception` blocks in `alerts.py` (4 blocks), `query_cache.py` (2 blocks), `graph_snapshots.py` (1 block), `rabbitmq_client.py` (add intent comment)

### P2 — Reproducibility & Test depth
- [x] `Makefile` — add `make lock` target using `pip-compile`; add `pip-tools` to `install-dev`
- [x] `tests/unit/test_gnn_scorer.py` — 23 tests: adjacency normalisation, GCN/GAT math, edge filtering, hub dampening, text score, confidence propagation
- [x] `tests/unit/test_inference_engine.py` — 18 tests: rule defaults, dispatch, fixpoint, dry-run, tenant, run_for_document
- [x] `tests/unit/test_confidence_calibration.py` — 20 tests: add_sample clamping, Brier score, calibration curve binning, apply_calibration, summary verdict

### P3 — Repo hygiene
- [x] Move `check_db.py`, `check_embed.py`, `check_key.py`, `test_gnn.py`, `test_hybrid.py` → `scripts/dev/`; add `scripts/dev/README.md`
- [x] Move `d1.json`, `d2.json`, `d3.json` → `eval_data/`; add `eval_data/*.json` to `.gitignore`
- [x] `Dockerfile` — multi-stage build (builder + runtime stages); non-root `app` user; `HEALTHCHECK` instruction
- [x] `.gitignore` — add `eval_data/` patterns, `requirements.lock`

### Test counts after Phase 11
- Unit: 97 (prev 36 + 23 gnn_scorer + 18 inference_engine + 20 confidence_calibration)
- Integration: 33 (unchanged)
- Load: 5 (unchanged)
- **Total: 135 tests, all pass**

---

---

## Phase 12 — LLM Swap, Infrastructure Fixes & Local Smoke Test

Groq replaces Gemini for all text generation. Six critical bugs fixed. System verified end-to-end locally.

### LLM Routing
- [x] Create `graphrag/core/llm_client.py` — `GroqLLM` (async Groq SDK via executor) + `GeminiEmbedder` + singletons `get_llm()` / `get_embedder()`
- [x] Update `graphrag/core/config.py` — add `groq_api_key`, `groq_model` fields
- [x] Update `graphrag/ingestion/extractor.py` — use `get_llm().generate()` instead of Gemini `generate_content`
- [x] Update `graphrag/ingestion/embedder.py` — delegate to `get_embedder().embed()`
- [x] Update `graphrag/retrieval/hybrid_retriever.py`, `global_search.py`, `agentic_retriever.py`, `graphrag/graph/community_summarizer.py` — use `get_llm().generate()` throughout
- [x] Add `groq>=0.9.0` and `redis[asyncio]>=5.0.0` to `requirements.txt`
- [x] Update `.env` — add `GROQ_API_KEY`, `GROQ_MODEL`; keep `GOOGLE_API_KEY`, `GEMINI_EMBED_MODEL`

### Neo4j Bug Fixes
- [x] `graphrag/graph/neo4j_client.py` line 485 — replace deprecated `size((e)-[:RELATES_TO]-())` with `COUNT { (e)-[:RELATES_TO]-() }` (Neo4j 5.x)
- [x] `graphrag/graph/ingestion_validator.py` — fix nested aggregate `avg(toFloat(count(r)))` → `avg(toFloat(degree))`
- [x] `graphrag/graph/cycle_detector.py` — fix APOC availability check to use `CALL apoc.help('findCycles')`
- [x] `scripts/init_neo4j.py` — fix lazy DDL: add `await result.consume()` after every schema statement
- [x] `scripts/init_neo4j.py` — fix comment stripping: strip `-- comment` lines per-fragment before executing

### Windows Compatibility
- [x] `workers/ingestion_worker.py`, `query_worker.py`, `evaluation_worker.py`, `combined_worker.py` — guard `loop.add_signal_handler` with `if sys.platform != "win32":`

### Workers & Infrastructure
- [x] `workers/combined_worker.py` — created; runs ingestion + query consumers on one machine
- [x] `config/settings.yml` — `session_store_strict: false` (redis package may be absent in dev)

### Documentation
- [x] Update `README.md` — Stack table (Groq LLM), `.env` section, Project Structure, Verified System Check Results, Common Issues, Quick Start
- [x] Update `docs/knowledge-graph-architecture.md` — add Section 9: LLM routing + cross-process result store
- [x] Update `tasks/lessons.md` — add A57–A62 (Neo4j DDL lazy execution, `size()` deprecation, comment stripping, Windows signals, cross-process results, strict mode + missing packages)
- [x] Update `tasks/todo.md` — this entry

### Verified end-to-end
- Ingestion: `ingestion_agent.done chunks=1 entities=5 relations=4` ✅
- Hybrid search: `bm25=10 fused=10 vector=10` ✅
- Reranker: `top_score=9.30` ✅
- GNN: `gnn_scorer.skip reason=no_entity_embeddings` (expected — no community embeddings yet)
- Answer synthesis: Groq Llama generates cited answer ✅
- Redis result store: cross-process result sharing confirmed ✅

---

## Review

### What was built
A production-grade GraphRAG pipeline hardened across 6 phases. Starting from a working
but single-tenant, formula-inconsistent, silently-degrading prototype, the system now has:

- **Correct multi-source evidence accumulation** on every RELATES_TO edge
- **Full `(name, type, tenant)` isolation** at every MATCH/MERGE/analytics query
- **Fail-fast strict mode** for critical dependencies (Redis, Leiden) so operators see failures at startup, not in user reports
- **Contradiction detection** with 4 typed conflict categories, each suggesting a different resolution strategy
- **Graph health monitoring** as a leading indicator separate from RAGAS (lagging)
- **Community staleness gating** preventing unnecessary rebuilds while keeping structure fresh
- **Deferred session recording** so stored turns always have real answers
- **All config flags wired** — no more descriptive-only settings

### What was not built
- No load/performance testing of the GNN scoring layer under real concurrency
- No automated alert routing from `GraphHealthSnapshot` threshold breaches
- ~~No UI for the corrections API (currently API-only)~~ — stale: the admin
  dashboard now has a "Resolve conflict" button (was 404ing due to a wrong
  route, fixed 2026-07-24 commit `f4322f1`; UI has existed since Phase 9d)

### Key numbers
- 15 code correctness lessons documented
- 18 architecture design lessons documented
- ~30 distinct bugs/gaps found and fixed across the two sessions
- 0 of them crashed; all corrupted data or degraded quality silently

---

## Phase 13 — JD Compliance Audit & Fixes

Full audit against the "Senior Knowledge Graph & Ontology Lead" JD. Found and fixed
all gaps introduced by the Groq migration and missing leadership/ops artifacts.

### Bug fixes (code)
- [x] `graphrag/agents/query_agent.py` — `_model()` returned `gemini_query_model`; corrected to `groq_model`
- [x] `graphrag/agents/ingestion_agent.py` — same fix for `gemini_ingest_model`
- [x] `graphrag/agents/evaluation_agent.py` — same fix; added clarifying comment
- [x] `graphrag/agents/base_agent.py` — abstract `_model()` docstring said "Gemini model ID"; corrected to provider-agnostic
- [x] `graphrag/retrieval/agentic_retriever.py` — `model_version=self._model` raised `AttributeError` at runtime (`AgenticRetriever` has no `_model`); replaced both sites with `get_settings().groq_model`
- [x] `graphrag/evaluation/ragas_evaluator.py` — hardwired to `ChatGoogleGenerativeAI(gemini_ingest_model)`, re-introducing Gemini quota risk; rewritten to try `langchain_groq.ChatGroq` first, fallback to Gemini, then None
- [x] `requirements.txt` — add `langchain-groq>=0.1.0`
- [x] `tests/unit/test_data_path_fixes.py` — `TestExtractorNoneResponse` mocked old Gemini path (`run_in_executor` + `response.text`); rewrote both tests to patch `get_llm` (Groq path); now verifies real empty-response guard logic
- [x] Editable install re-run (`pip install -e .`) to fix stale `GraphRag` path in tracebacks

### Leadership artifacts (new files)
- [x] `CONTRIBUTING.md` — ADR process, 14-item PR checklist, Python/Cypher/LLM coding standards, feature extension guide
- [x] `docs/runbook.md` — startup order, 9 health checks, 8 failure patterns + root causes, monitoring, backup/restore, schema migration, secrets rotation, on-call decision tree
- [x] `docs/roadmap.md` — current state matrix, 9 scale limits, near/medium/long-term roadmap, pending ADRs, scaling decision reference
- [x] `README.md` — added links to runbook, roadmap, CONTRIBUTING

### Documentation updates (this phase)
- [x] `docs/knowledge-graph-architecture.md` — add RAGAS evaluator to "What uses Groq" list; add RAGAS judge note with priority-order explanation
- [x] `README.md` — `llm_utils.py` description corrected from "Gemini response.text" to "(embedding path)" scope
- [x] `tasks/lessons.md` — A63 (LLM swap without re-running tests); A64 (attribute ref vs method call: `self._model` bug)
- [x] `tasks/todo.md` — this entry

### Verified
- Unit suite: **304 passed, 0 failed**
- `scripts/demo_regulatory.py` runs all 6 steps on Windows without live services

---

## Done: Semantic blending for multi-hop chunk ranking (2026-06-10)

**Problem:** hop chunks were capped at 50 (`multihop_top_k`) ranked purely by
`path_score = path_confidence / path_length` — graph topology only. On dense
graphs, semantically relevant chunks could fall below the cap while
topologically-cheap-but-irrelevant ones survived.

**Change:** rank hop chunks by `(1-w)·path_score + w·cos(chunk_emb, query_emb)`
BEFORE the cap, with the cosine computed inside Neo4j
(`vector.similarity.cosine`) so no embeddings cross the wire.

- [x] `graphrag/graph/neo4j_client.py` — `get_multihop_chunks()` accepts
      `query_embedding` + `semantic_weight`; null-safe fallback for chunks
      without embeddings; blend computed in Cypher
- [x] `graphrag/retrieval/local_search.py` — passes query embedding (already
      computed for vector search — zero extra embed calls); degrades to pure
      path score when vector search disabled
- [x] `config/settings.yml` — `multihop_semantic_weight: 0.5`
- [x] `scripts/eval_hop_ranking.py` — retrieval-level A/B gate (no LLM judge):
      citation hit rate / coverage / MRR on golden set, both arms
- [x] `scripts/ingest_corpus.py` — now applies `schema.cypher` (idempotent)
      before commit writes; guards against fresh Neo4j volumes missing
      `chunk_embeddings` / `chunk_fulltext` indexes (found during eval setup:
      both were missing from the live DB — vector + BM25 retrieval would
      silently break on a fresh clone)

**Eval gate result (33 golden questions with expected citations):**
| arm | hit rate | coverage | MRR | p50 |
|---|---|---|---|---|
| baseline w=0 | 1.000 | 1.000 | 0.385 | 10.36s |
| blend w=0.5 | 1.000 | 1.000 | **0.545** | 10.67s |

MRR +0.160 (+42% relative) — expected-citation chunks rank visibly higher;
zero hit/coverage regression; +0.3s latency (cosine on ~3.7k chunks in-DB).
VERDICT: keep w=0.5. Results: `evals/hop_ranking_eval_results.json`.

**Verified:** 362/362 unit tests pass; dry-run ingest OK; eval ran against
live Neo4j (aerospace tenant, 368 entities / 422 edges).
