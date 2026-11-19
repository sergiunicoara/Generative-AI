# Publication Audit — Rollback History (2026-09-14)

Context: `#publication-audit` panel (active version, publish timestamp,
counts, quarantine list) already shipped (commit 22c3dac). Missing piece is
durable rollback/publication history: an API endpoint, a UI timeline, tests,
and docs.

## Plan

- [x] `graphrag/domains/energy/demo.py`: add `EnergyDemoService.publication_history_durable()`
      (durable governance-store history, falling back to in-memory
      `publication_history()` when no store is wired) — leaves the existing
      sync `publication_history()` untouched (used by `test_energy_demo.py`).
- [x] `api/routes/energy_demo.py`: add `GET /energy-demo/publication/history`
      — tenant-scoped 404 guard (same pattern as every other read route),
      returns `{active_version_id, publications:[{version_id, published_at,
      conforms, published_triple_count, quarantined_count, rolled_back_from}]}`,
      oldest first. No blob paths/hashes (not present on `PublicationReport`
      anyway) and no full quarantined-record bodies (see `/quarantine`).
- [x] Extend `_publication_audit_html()` / `_publication_audit_script()` with
      a "Publication history" sub-list, fetched client-side (own
      `loadPublicationHistory()`, independent try/catch so the existing
      `loadPublicationAudit()` error-handling test is untouched), active
      version badge, rollback vs. publication label, restored-from version.
- [x] Tests in `tests/unit/test_energy_demo_routes.py`: history returned +
      shape, chronological order with a real rollback (`rolled_back_from`
      correct), wrong-tenant 404 (extend existing combined test), read-scope
      access (extend existing combined test), route is GET-only, dashboard
      HTML contains the new section id and fetches client-side.
- [x] Docs: `docs/demos/energy_asset_intelligence.md` (route list + a short
      history/timeline paragraph), `docs/energy-architecture.html` (tooltip
      mention), `docs/presentation/energy-demo-evidence-manifest.md` (short
      dated note, matching the existing panel's note style).
- [x] `scripts/capture_energy_demo_ui.py`: wait for the history sub-section
      to leave "Loading…" before the existing `#publication-audit`
      screenshot (no new file needed — it's the same panel).
- [x] Run: targeted tests, then `run_energy_demo_e2e.py`, then
      `verify_energy_movies.py` (no movie content changes expected).

## Review

- **Tests**: `tests/unit/test_energy_demo_routes.py` (22), `test_energy_evidence_requests.py`
  (6), `test_energy_governance_store.py` (9) — 37/37 pass. `test_energy_demo.py` +
  `test_energy_publication.py` (27, the in-memory-history callers) also re-run clean —
  confirms `publication_history()`'s existing behavior/signature is untouched.
- **Live check**: started `api.main:app` locally against a scratch governance store,
  logged in via `/auth/dev-login`, and confirmed via `get_page_text` +
  `read_network_requests` that `GET /publication/history` returns 200 and the new
  "Publication history" sub-panel renders "Publication · Active · <timestamp> · version
  <id> · 0 quarantined · conforms" — exactly the intended contract. (Screenshot capture
  itself kept timing out because the pane was hidden/minimized in this session — a tool
  limitation, not a rendering issue; the page-text/network evidence is the actual proof.)
  Cleaned up the scratch server, scratch sqlite files, and the stale
  `artifacts/energy/` local dev cache afterward (all untracked, regenerable).
- **`run_energy_demo_e2e.py`**: exit code 0, full capability list intact.
- **`verify_energy_movies.py`**: fails with `ValueError: Stale narration:
  render_energy_demo_client_teaser.py 6` — confirmed via `git stash` to a clean
  checkout of `HEAD` (commit 22c3dac, before any change in this session) that this
  failure **pre-exists** this work; it is not caused by anything here (this session
  touched no renderer, movie, or narration file). Left as-is per the task's scope —
  fixing it is a separate, unrelated movie-rebuild task. Worth flagging to the user.
- **Safety boundary honored**: no button, form, or POST/PUT/DELETE route was added to
  the audit panel or the history endpoint; `test_energy_demo_routes_expose_no_new_mutation_endpoints`
  and the new `test_publication_history_endpoint_is_read_only` both confirm this
  structurally, not just by inspection.

---

# Contextual knowledge evaluation (2026-10-24)

## Existing / reuse / extend map (from code, not docs)

| Need | Existing (file) | Decision |
| --- | --- | --- |
| Trusted tenant + principal | `api/auth/dependencies.py` `get_tenant`, `require_scope`, `assert_request_tenant`; `mcp_server/identity.py` `CallerIdentity`; `enterprise/models.py` `AccessContext.from_claims`, `DocumentAccessPolicy` | **Reuse.** Knowledge items carry a `DocumentAccessPolicy`; a Python twin of `document_access_predicate` filters before anything is exposed. |
| Obligation extraction | `enterprise/models.py` `ObligationDraft` (no tenant/modality/status) | **Extend by conversion**: `KnowledgeItem.from_obligation_draft()` yields a *draft* with no executable constraint and unknown applicability. `ObligationDraft` unchanged. |
| Policy evaluation | `context_graph/policy_engine.py` first-match, no unknown state | **Reuse only for the escalation step** (one decision over summary facts). Not used as the multi-obligation evaluator. Reuse `ConditionOperator` as the bounded constraint vocabulary. |
| Escalation / human review | `business/policy.py`, `CGApproval`, `CGExceptionGrant` | **Reuse pattern**: escalation expressed as a `PolicyVersion`; exceptions modelled with explicit scope/provenance/validity mirroring `CGExceptionGrant`. |
| Domain applicability | `graph/pharma_commercial.py` exact-equality scope checks, no unknown | Leave as is; new shared service generalises it (multi-dimension, three-valued). |
| Temporal selection | `domains/energy/answers.py` `select_bulletin` current/missing/expired/conflicting | **Hook point**: shared evaluation added to `maintenance_review` answers; consistency test against `select_bulletin`. |
| Release / publication | `domains/energy/publication.py` SHACL gate; `governance_store.py` append-only versions + active pointer + rollback | **Extend the durable store** with knowledge-release tables (same engine/DB, no new store). Publisher gate runs contract + conflict lint + regression eval; failure keeps previous active release. Energy answers stamped with the active publication version. |
| Cache identity | `retrieval/query_cache.py` `QueryCacheContext` (no release/context fields) | **Extend**: add `knowledge_release` + `business_context_fingerprint`, bump schema version; evaluation results cached under tenant+access+context+view+release key. |
| API/MCP | `mcp_server/registry.py` `CapabilitySpec`, `capabilities/__init__.py`, `server.py`, `api/routes/skills.py`, contract fixture `tests/unit/contracts/mcp_capabilities_v1.json` | **Reuse**: new capability `kg.knowledge.evaluate@1.0.0` + `POST /knowledge/evaluate`; extend `_json_type` for dict args. |
| Missing | — | three-valued applicability, normative modality, bounded constraint DSL with missing-fact semantics, conflict/exception handling, validation outcome types, knowledge release record. |

## Plan
- [x] Contracts: `graphrag/context_graph/knowledge.py`
- [x] Applicability service (applies / does_not_apply / unknown)
- [x] Constraint validation + conflicts + exceptions + decision (escalation via policy engine)
- [x] Release publisher gate on the durable governance store; evaluation cache
- [x] Packaging/labeling synthetic pack + walkthrough
- [x] Energy integration on `maintenance_review` + publication stamp
- [x] MCP capability + API route + contract fixture
- [x] Deterministic eval set + runner with separate metrics
- [x] Docs + roadmap; targeted tests; final regression run

## Review (2026-10-24)
- Full unit suite: 3278 passed (703 s), one run at the end. New targeted tests: 62.
- Eval set: 22/22 synthetic cases; metrics in `evals/knowledge_eval/last_run.json`.
- Comparator sensitivity proven by mutating 7 expectations (all detected).
- Not done: RDF/SHACL projection via semantic-model compiler; pack review/publish HTTP route; any live validation.
- Side fix: `artifacts/mcp-capabilities-v1.json` was stale (read capabilities exported with empty scopes); regenerated.

## Audit and fix (2026-10-24)

Prior results above (3278 / 62 / 22-of-22) were measured before the audit and
are kept as history. Measurements below are new.

- [x] Reproduce: `tests/unit/context_graph/test_contextual_knowledge_audit.py`
  (31 tests) written first; against the pre-fix code **28 failed, 3 passed**
  (the 3 passing were behaviour that must not change: a waiver covering every
  bound value, ordinary falsity passing a prohibition, an unknown superseder
  making the older revision unknown). All 8 reported defects confirmed; none
  rejected.
- [x] Eval set extended with KE-23..KE-26 (expectations written before the
  fix). Pre-fix runner: **crashed** on KE-26 (`TypeError: '>=' not supported
  between 'str' and 'int'`); KE-23/24/25 mismatched (PKG-001 / PKG-009 /
  PKG-017 suppressed or passed; overall `compliant` instead of `non_compliant`).
- [x] Fix evaluator: latest version per id before status/access; typed
  predicates (`invalid` / `fact_type_mismatch`); absence-is-violation fails
  every modality; coverage-checked waivers/precedence with
  `partial_exception_refs`; precedence needs an effective winner, else
  `unresolved_exception` escalation (policy 1.1.0); transitive,
  order-independent supersession with cycle/unknown/partial handling.
- [x] Fix contracts: NaN/inf rejected on `Fact` and `Predicate`.
- [x] Fix release gate: mandatory regression cases; one published version per
  item/exception id and it must be the latest; content hash covers pack
  metadata + disclaimer.
- [x] Consumer example approves only `compliant` with no review flag.
- [x] Structured failures through registry and HTTP route (type mismatch ->
  `needs_review`; NaN -> `invalid_request` / 422).
- [x] Focused: 100 tests (31 audit + 69 knowledge/capability/energy) pass; eval
  26/26; walkthrough runs.
- [x] Full unit suite (one run, at the end): **3316 passed**, 28 warnings, 0 failed (749 s).
- Remaining gaps (as of the audit): partial waivers/supersessions not unioned and
  a partially superseded revision reporting its full matched context were fixed
  on 2026-10-29 (below); the eval set is developer-authored (agreement with own
  expectations, not real-world accuracy).

## Roadmap items implemented (2026-10-29)

Chosen from the open items in `docs/roadmap.md`; the others are conditional
("only when a real deployment requires them"), non-goals, or larger/riskier
(RDF/SHACL projection, typed semantic-rule outcomes, qualified cardinalities)
and were left for a later decision.

- [x] Tests first: `tests/unit/context_graph/test_knowledge_partial_union.py`
  (8) and `tests/unit/test_knowledge_pack_routes.py` (15). Pre-implementation:
  5 of 8 union tests failed (the 3 passing are fail-closed guards); 13 of 14
  route tests failed (the initial set) before the routes existed.
- [x] Union of partial waivers / supersessions, per combination of context
  values (not per dimension), capped at 5,000 combinations (fail-closed);
  `matched_context` of a partially superseded revision shows what is left;
  later exceptions cover exactly that remainder.
- [x] Routes: review queue (`write`), publish and rollback (`admin`), history
  (`read`); reviewer/publisher = token `sub`; tenant from the token; pack by
  validated domain name; metadata-only responses.
- [x] RDF/SHACL projection: canonical model `ontology/models/contextual-knowledge.yaml`
  (compiled artifacts committed, `make semantic-model-check` extended),
  `graphrag/context_graph/knowledge_rdf.py`, SHACL conformance required by
  `check_pack` (fail-closed), visibility-filtered `export_graph`; 11 tests
  incl. negative cases (missing source / tenant / review ref, duplicate version).
- Focused: union 8 + audit 31 + knowledge 46 + routes 15 + capability 11 + rdf 11
  pass; eval 26/26.
- [x] Full unit suite (one run, at the end of that batch): passed; superseded by the 2026-11-03 run below.

# Remaining roadmap items (2026-11-01)

Scope: every open checkbox in `docs/roadmap.md` (items at lines 89, 106, 278,
286, 288, 290). The proposed "Recruitment Knowledge Graph" extension is a
separate, unscheduled domain proposal, not an open checkbox, and is out of
scope. Constraints: no paid services; no live external systems (Stardog,
Jira, GraphDB) are available, so adapters are verified offline and are not
described as live-validated. Full unit suite runs once, at the end, before push.

## Plan
- [x] P0 typed target outcome: additive `outcome` per diagnostic
      (enforced/translated/runtime-mitigated/warning/unsupported) and a full
      rule x target capability matrix artifact; `fidelity` kept for existing
      consumers.
- [x] P1 qualified cardinalities, relationship attributes, composite keys in
      the canonical model and every emitter where the target supports them;
      diagnostics where it does not; regenerated artifacts + drift check.
- [x] P2 read-only SPARQL 1.1 endpoint conformance adapter (Stardog/GraphDB/
      Fuseki) running the SPARQL competency questions against the dataset
      fingerprint; offline-tested.
- [x] P2 GQL graph-type export as an interoperability surface (not a source
      of truth).
- [x] P2 GraphQL consumer API that only resolves through the governed
      capability registry (scopes, tenant, audit unchanged).
- [x] P2 curator import/export: spreadsheet (CSV/XLSX) and Jira issue payloads
      for the curator queues; outbound Jira client with injectable transport,
      disabled unless configured.
- [x] Docs + roadmap; one full unit-suite run; push.

## Results (2026-11-03)
- Implemented: typed outcomes + `capability-matrix.json`; composite keys, qualified
  cardinalities, relationship attributes (all emitters, runtime validator, ERD);
  GQL graph-type rendering; SPARQL endpoint conformance adapter (+ Fuseki URL
  builder); `POST /graphql` (read-only, via the capability registry; the
  `answer` field was removed because it writes traces / may call an LLM);
  curator CSV/XLSX export + decision import + Jira issue route.
- Caught while testing: an infinite loop I introduced (extension validation ran
  before the inheritance-cycle check) and two diagnostic-code/matrix mismatches.
- Full unit suite, one run at the end: **3567 passed**, 28 warnings, 0 failed (771 s).
- Not validated live: GQL engine, Stardog/GraphDB/Fuseki, Jira (mock transports only).
- Open: `requirements.lock` lacks `graphql-core` (`make lock`); Jira route not idempotent.

# Roadmap candidate items (2026-11-03 batch)

Scope: the 21 open items in `docs/roadmap.md` (7 feed-review candidates, 14
job-description candidates). Confluence/Databricks stays open by its own
condition (no real deployment requires it). Offline only, no paid services, no
claims of live validation. Full unit suite and push after each wave.

- [x] Wave 1 (3582 passed, full suite): durable decision log + replay + explanation-fidelity check;
      decision re-review impact analysis (release change, expiring exceptions,
      review-due items); remediation hints in gate and evaluation errors.
- [x] Wave 2: metamorphic + held-out + generated evaluation; applicability
      index, latency benchmark, stage metrics; constraint rendering for reviewers.
- [x] Wave 3: release stamp on every projection; shape/model registry as RDF;
      named-graph ADR + per-tenant drop test + JSON-LD round trip.
- [x] Wave 4: formalism ADR + decision tables (representation/validation);
      expert elicitation; business-definition registry; use-case charters;
      answer-trust investigation workflow.
- [x] Wave 5: erasure/correction propagation audit; data minimisation before
      model providers; extraction record contract; permission-freshness bound.
- [x] Wave 6 (evaluation only): SHACL-to-Cypher; "Open Knowledge Format" spec check.

## Results (2026-11-03 batch)
- 20 of 21 open items implemented or evaluated; Confluence/Databricks stays open by its own
  condition. Waves 1-5 pushed after a full unit suite each (3582, 3793, 3803, 3834, 3851 + 1 stale
  test assertion fixed). Wave 6 is evaluation: shacl2cypher not installed (single new release,
  single maintainer), in-house Cypher subset written; live Neo4j parity test written but skipped
  (no Docker here); OKF v0.2 verified and an export prototype written, licence unconfirmed.
- Everything is verified offline; no live Neo4j, Redis, triple store, SharePoint, Jira, GQL engine
  or model provider was used.



---

# Roadmap items from the 2026-11-19 feed review: failure classes, token accounting, answer-attributed citations

Context: the three highest-value items of "Candidate additions from a second
graph-community feed review (2026-11-19)" in `docs/roadmap.md`. Code mapped by
three read-only passes on 2026-11-19; nothing implemented yet. **Waiting for
plan approval.** Order: item 2 (free, offline), item 3 (instrumentation), item 1
(production answer path, largest regression risk).

Rules for the whole batch: no paid model calls without explicit approval;
default prompts, answers and response shapes unchanged when a new flag is off;
unit suite run once at the end; no push until asked; never commit
`docs/roadmap.md` changes made before this batch.

## Item 2 — classify golden-set misses (offline, no model calls)

- [x] `scripts/classify_golden_misses.py`: load the valid runs of a folder (same
      glob and `n_errors == 0` filter as `aggregate_ablation.py`), take every row
      with `golden_pass == False`, assign exactly one class by this ordered rule:
      1. `status == "unscorable"` -> `unscorable`
      2. only the citation check failed: all expected ids uncitable
         (`n_expected_uncitable == len(expected)`) or `golden_failures_citable`
         empty -> `citation_uncitable_artifact`; else if expected ids are missing
         from `citations_context` (context basis) -> `retrieval_miss_citation`;
         else `citation_only`
      3. a content check failed (required term / any-of): `refused_with_evidence`
         or `refused_no_evidence` when `status == "refused"` (evidence = context
         `recall_raw == 1.0`); otherwise `retrieval_miss` when context
         `recall_raw < 1.0`, else `answer_error`; `full_context` (no context set,
         whole corpus in the prompt) -> `answer_error` by construction
      4. `graph_or_schema_gap` is **not decidable** from stored data (graph state
         has counts only, no context text) -> reported as a stated limitation, never
         assigned
- [x] Outputs next to the runs: `miss_classes.json` and `miss_classes.md` (per
      profile x class counts, per-question majority class over the 3 runs,
      per-row table, the limitation text).
- [x] `tests/unit/test_classify_golden_misses.py` with synthetic rows (pattern of
      `test_compare_ablation_profiles.py`): one test per rule, `invalid/` and
      `n_errors > 0` ignored, `full_context` null-context path, raw versus citable
      citation split.
- [x] Run it on `evals/ablation/aerospace-v2.3-strict/`; sanity-check against
      the known tallies (for example `full`: 15 content-only failing rows, 3
      citation-plus-content).
- Limitation to state in the output: classes are document-level proxies. The
  retrieved context text is not stored, so "right document, wrong chunk" lands in
  `answer_error`. Storing context text in future runs is a separate, paid decision.

Review (2026-11-19, item 2): `scripts/classify_golden_misses.py` and
`tests/unit/test_classify_golden_misses.py` (16 tests, pass). One deliberate change from the
plan: `unscorable` is not a class, because it describes the faithfulness judge, not the golden
check, so a failed row is classified by its golden cause whatever its judge status. On
`evals/ablation/aerospace-v2.3-strict/` the failed rows (34, 30, 18, 26 of 102 for
vector_only, text_hybrid, full, full_context) match the aggregate pass rates; outputs
`miss_classes.json` and `miss_classes.md` are next to the runs, not committed. For `full`,
12 of 18 failures are `answer_error` (every expected document was retrieved), 3
`refused_with_evidence`, 3 `retrieval_miss`.

## Item 3 — token and cost accounting (measure only, answers unchanged)

- [x] `graphrag/observability/usage.py`: `UsageLedger` (buckets by kind
      {extraction, artifacts, community, embedding, generation, judge}, provider,
      model; fields prompt, completion, total, n_calls, n_calls_without_usage),
      active ledger in a ContextVar, a lock around mutation, `ledger_scope()` and
      `usage_scope(kind)`.
- [x] Report sites:
      - `graphrag/core/llm_client.py` `_report_openai_compatible_usage` (about
        93-119): also add to the active ledger (no change to return values; failed
        or retried-away attempts are not counted — document the understatement)
      - `OpenAIEmbedder.embed` (about 796-825): record
        `response.usage.total_tokens` when present, else count it as unmeasured
      - RAGAS judge: a passive `BaseCallbackHandler` passed as
        `evaluate(callbacks=[...])` at `graphrag/evaluation/ragas_evaluator.py`
        (about 264-270), tagged `judge`, scoped per question
- [x] Ingestion manifest (`graphrag/agents/ingestion_agent.py` about 196-206,
      474-482): add `prompt_tokens`, `completion_tokens`, `total_tokens`,
      `n_calls_without_usage` and a cost to `embedding`, `extraction` and a new
      `artifacts` stage of `stage_metrics`. Values are `None` unless every call
      reported usage; cost only from `estimated_cost_usd` and only for priced
      models; the `cost_status` string stays. Community summaries get a separate
      job-level record. The integrity hash changes with the new keys (accepted).
- [x] Ablation script (`scripts/benchmark_retrieval_ablation.py`): open a ledger
      per profile around the question loop (about 313) and add an optional
      `summary["tokens"]` (generation and judge prompt/completion, embedding,
      cost or `None`). No exact-key assertions on `summary` were found, only on
      `meta["git"]` and `config_overrides`.
- [x] Tests (patterns of `test_genai_token_usage.py`, `test_pricing.py`,
      `test_llm_client.py`, `test_ingestion_agent_extract_resilience.py`):
      ledger accumulation and thread safety, `None` usage stays unmeasured, a
      fallback secondary call is counted once, the embedder with and without
      `usage`, the judge handler with a mocked judge (scores unchanged), the
      manifest `None` rule and the "never invents missing cost" behaviour.
- Resolved: a bare `import ragas` does fail here (missing
  `langchain_community.chat_models.vertexai`), but the project patches a stub in
  first (`RagasEvaluator._patch_langchain_vertexai_stub`, `ragas_evaluator.py`
  about 171-211); with it, ragas 0.4.3 imports and `evaluate` accepts `callbacks`.
  Not a defect.

Review (2026-11-19, item 3): `graphrag/observability/usage.py` (ledger), reporters in
`llm_client.py` (providers and embedder), `query_rewriter.py` (kind `rewrite`),
`ragas_evaluator.py` (passive judge callback, only when a ledger is active),
`ingestion_agent.py` (per-document ledger into `stage_metrics`, new `artifacts` stage),
`benchmark_retrieval_ablation.py` (`summary["tokens"]` per profile). Tests:
`tests/unit/test_usage_ledger.py` (11) plus the neighbouring suites (99 pass in total).
Verified only with mocked providers: no real provider response, ingestion or judge call was
made (no cost), so the first real token numbers appear at the next paid run. Not counted:
provider-internal retries that failed, community summaries (separate job record, not yet
wired). Embedding cost stays `None` (no verified price row).

## Item 1 — answer-attributed citations (opt-in, off by default)

- [x] New `graphrag/retrieval/answer_citations.py` with the pure parts moved out
      of `graphrag/evaluation/citations.py`: `_norm`, `build_alias_index`,
      `strict_citations`, `parse_answer_tags`, `ACCEPTED_TAG_FORMS`,
      `citation_instruction`; none reads the disk. `graphrag/evaluation/citations.py`
      re-exports them and keeps `_load`, `ALIAS_MAP_PATH` and the scoring helpers,
      so the benchmark and `test_citations.py` keep working and the pinned
      instruction text stays byte-identical (hash `9769435d...` must not change).
- [x] Config: `retrieval.answer_citation_tags_enabled: false` in
      `config/settings.yml`. When true, the retriever builds the instruction from
      the tenant's document names (`get_document_filenames`, stems, ACL-filtered)
      and appends it through the existing `eval_citation_instruction` hook in
      `answer_policy.answer_prompt`; when false the prompt is byte-identical.
- [x] `graphrag/retrieval/hybrid_retriever.py` after `normalize_dashes` (about
      960) and before claim verification (about 966): parse the tags with
      `canonical_ids` = the tenant stems and an index built from them, set
      `cited`; strip the tags from the user-facing answer (decision D1).
- [x] `graphrag/core/models.py` `QueryResult`: `cited: list[str] = []`; set it at
      every construction (about 443, 1096, 1164 and the agentic path); persist it
      in `graphrag/messaging/consumers.py` (explicit dict, about 123) and in the
      semantic answer-cache entry; MCP picks it up through `model_dump`; update
      the two docstrings in `mcp_server/server.py` (about 116, 164).
- [x] Tests: `test_answer_policy.py` (prompt unchanged when off, instruction when
      on), `test_hybrid_retriever.py` (`cited` populated, tags stripped, empty
      when off), `test_query_consumer_failure_handling.py`,
      `test_mcp_tools.py`, `test_query_result_evidence.py` (additive field),
      re-run `test_mcp_contract_compat.py`, `test_citations.py` and
      `test_benchmark_retrieval_ablation.py`.
- [x] Acceptance on stored data, free: precision of the answer-tag set versus the
      context set per retrieval profile, already in the strict runs (answer
      0.601 to 0.734 against context 0.227 to 0.391). Production-path correctness
      is covered by the unit tests; an optional paid confirming run is decision D4.

## Decisions needed before coding

- D1 (item 1): strip the `[DOC: id]` tags from the user-facing answer when the
  flag is on? Recommended: yes, as the benchmark does; the raw tagged text is
  not exposed.
- D2 (item 1): the instruction lists every tenant document, one line each. Cap it
  (for example refuse to enable above 50 documents, fail closed) or list only the
  retrieved documents? Recommended: cap and fail closed; listing only retrieved
  documents changes what the model may cite and would need its own measurement.
- D3 (item 1): module location `graphrag/retrieval/answer_citations.py`
  (recommended, next to `answer_policy.py`) or `graphrag/core/`.
- D4 (item 1): an optional paid confirming run (one profile, 34 questions; the
  strict runs cost about $0.06 to $0.08 for the retrieval profiles) to check tag
  compliance on the production path. Default: not run.
- D5 (item 3): embeddings have no price row. Record tokens only and leave
  embedding cost `None` until you supply a verified price (recommended; no
  guessed prices).
- D6 (item 2): should later ablation runs store the retrieved context text so
  `retrieval_miss` versus `answer_error` can be decided at chunk level? Needs a
  paid re-run, so out of this batch unless you ask.

## Review (2026-11-19)

Items 2, 3 and 1 implemented in that order with the recommended decisions (D1 strip tags, D2 cap 50 and fail closed, D3 `graphrag/retrieval/answer_citations.py`, D4 no paid run, D5 embedding cost None, D6 not done). One unit-suite run at the end: 4176 passed, 0 failed (4137 before this batch plus 39 new tests). Not done: a paid confirming run for item 1, community-summary token accounting, the routing table from the roadmap (item 2 of the feed review). The measured behaviour of the new code on real provider responses is untested: all three were verified with mocks.
