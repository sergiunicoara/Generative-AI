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
- No UI for the corrections API (currently API-only)

### Key numbers
- 15 code correctness lessons documented
- 18 architecture design lessons documented
- ~30 distinct bugs/gaps found and fixed across the two sessions
- 0 of them crashed; all corrupted data or degraded quality silently
