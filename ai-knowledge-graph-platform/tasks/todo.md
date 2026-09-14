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
