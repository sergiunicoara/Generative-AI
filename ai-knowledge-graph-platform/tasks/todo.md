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
- Remaining gaps: partial waivers/supersessions are not unioned; a partially
  superseded revision reports its full matched context; the eval set is
  developer-authored (agreement with own expectations, not real-world accuracy).

---

# Contextual knowledge follow-ons (2026-10-29)

The three open roadmap items under "Contextual knowledge evaluation".

## Plan
- [x] Joint coverage: union partial waivers / partial supersessions by box
  subtraction over context-value combinations; report `in_force_context`.
- [x] Authenticated pack review/publish routes: working-pack revisions in the
  governance store (CAS on `expected_revision`), token `sub` as reviewer,
  grantor and publisher, unchanged gate with deployed regression cases.
- [x] RDF projection of published knowledge: canonical model
  `ontology/models/contextual-knowledge.yaml`, compiled artefacts,
  `knowledge_rdf.py` projection + SHACL validation, wired into the gate.
- [x] Docs, roadmap, ontology README, Makefile drift check.

## Review (2026-10-29)
- New tests: 12 coverage-union (10 failed against the pre-change evaluator,
  2 pin preserved behaviour), 13 curation-route, 15 RDF/SHACL.
- Focused knowledge/capability/energy/MCP/semantic-model tests: all pass.
- Eval: 26/26; outcomes unchanged. Decision ids in `last_run.json` changed
  because results now carry `in_force_context`.
- `make semantic-model-check` equivalent: all three models verified.
- Environment: Python 3.11 venv from `requirements.lock` without torch,
  torchvision, sentence-transformers and ragas (PyTorch's index is blocked
  here; the PyPI CUDA wheels are several GB). Test modules importing
  `sentence_transformers` could not be collected.
- Full unit suite (one run, at the end): 3169 passed, 32 failed, 10 modules
  not collectable. The same 32 fail identically on a clean checkout of
  `main` in this environment (missing torch / sentence_transformers, and
  `graphrag.agents.query_agent` failing to import without them), so none
  is caused by this change.
- Not done: review UI, draft-submission route, union of precedence
  exceptions, any live validation.
