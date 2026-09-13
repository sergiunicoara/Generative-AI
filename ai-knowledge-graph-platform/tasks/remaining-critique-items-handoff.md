# Handoff prompt: remaining platform-critique items

Copy everything below this line into a new session for the AI coding agent.

---

You are working in the repo at `ai-knowledge-graph-platform/`, inside a git
monorepo whose root is one level up (also contains an unrelated `Limina/`
project — **never stage or touch `Limina/` deletions or files**).

## Context: what this is

A prior session worked through an 8-item platform-credibility critique plus
three cross-cutting hardening areas, one item at a time, each fully planned,
implemented, tested, and committed separately. Items 1–6 are done and pushed
to `main`. Your job is to continue the same pattern for the items still
open: **item 7, item 8, and the end-to-end demo scenario** described below.

Read `docs/demos/energy_asset_intelligence.md`, `ontology/README.md`, and the
commit log (`git log --oneline -20`) before starting — they document exactly
what already exists and why, and will save you from re-deriving context this
prompt doesn't repeat.

## Standing rules (apply to every item below, no exceptions)

1. **Plan before implementing.** For anything non-trivial (which every item
   below is), explore the relevant code first (delegate broad exploration to
   a subagent rather than searching inline yourself), design a concrete
   plan, and get it approved before writing code. Confirmed constraints or
   real gaps found during exploration should be stated plainly in the plan,
   not glossed over.
2. **One commit per item.** Never commit or push without the user explicitly
   asking for it, and ask again before every single commit — approval for
   one item's commit does not carry over to the next.
3. **Run the full test suite once, at the very end of each item** — not
   after every file change. Use targeted test files while iterating.
4. **Verify before claiming done.** Run the tests you write, run the
   relevant regression suite, run ruff/mypy on touched files, and actually
   execute anything you claim works (a live e2e script, a CLI flag) rather
   than asserting it from reading the code.
5. **Root-cause fixes, not workarounds.** If exploration surfaces a real bug
   adjacent to the task (this has happened repeatedly in this project —
   e.g. an unclosed SQLite connection, an ill-typed RDF literal, a missing
   tenant check on a revocation endpoint), fix it at the root and disclose
   the fix clearly in the commit message, don't route around it.
6. **Never overclaim.** This codebase has a strong, consistent convention of
   stating exactly what's tested/verified vs. what's a documented starting
   point (see any module docstring under `graphrag/ingestion/` for the
   tone). Match it. If something can only be tested against a mock/fixture
   because no live credentials or service exists, say so explicitly in the
   code and in your summary — don't imply more than what was actually
   proven.
7. **Minimal-impact edits.** Touch only what each item requires. Don't
   refactor unrelated code while you're in a file, even if you notice
   something questionable — flag it in your final summary instead.

## Item 7: real database backup and restore

> Demonstrate actual database backup and restore, then rerun the Energy
> queries against the restored instance (current backup artifact only
> compares a metrics fixture).

A prior exploration pass already resolved the ambiguity here — use these
findings directly rather than re-deriving them:

- `scripts/kg_backup.py`'s `do_backup`/`do_restore` are **Neo4j-only**: both
  require a live, reachable Neo4j (`get_neo4j()`) and apply/read NDJSON via
  real Cypher `MERGE`/`MATCH`. There is no offline/file-only mode.
- `scripts/run_production_exercises.py`'s `recovery_exercise()` is
  confirmed to do *only* a SHA-256 digest comparison of two files on disk —
  it never calls `kg_backup.py`, never touches Neo4j, never runs a query.
  That's the exact "only compares a metrics fixture" gap named in the
  critique.
- **The Energy demo does not use Neo4j at all** (`EnergyDemoService`'s own
  module docstring: *"does not write to Neo4j"* — confirmed, no
  contradicting code path exists). Its queries run in-process against an
  rdflib graph via `SPARQLBridge`. A GraphDB path exists
  (`compose.energy-demo.yaml`, `graphrag/graph/triplestore.py`,
  `tests/e2e/test_live_graphdb.py`) but is explicitly documented as *"an RDF
  serving copy... not a second source of truth."*
- **Real, confirmed gap inside that GraphDB e2e test**: it queries GraphDB
  with a **hand-written SPARQL string that duplicates**
  `evals/energy_demo/sparql/maintenance_review.rq` instead of reading that
  version-controlled file — meaning no test today actually runs "the Energy
  queries" (as `EnergyDemoService.answer()` defines them: reading that exact
  file) against a live triplestore. Fix this as part of item 7, not as a
  separate cleanup — load it via the same `Path.read_text()` +
  `SPARQLBridge`-style call `demo.py` itself uses, so the live e2e test
  genuinely proves the real query file, byte for byte.
- **Reusable pattern for a genuine Neo4j backup→restore→requery proof**:
  `tests/e2e/test_relational_postgres_neo4j.py` already boots a real
  `testcontainers` `Neo4jContainer`, ingests data, and queries it via a real
  `Neo4jClient` — the closest template if the critique also wants
  `kg_backup.py` proven against a live instance (not just Energy-specific).

Given the Energy demo has no Neo4j involvement, design this as **two
focused pieces**, not one:

1. **GraphDB backup/restore for the Energy dataset** (the literal "rerun the
   Energy queries against the restored instance" ask): extend
   `tests/e2e/test_live_graphdb.py` (or a new sibling e2e test) to, after
   loading data into a live GraphDB container: (a) export a backup —
   GraphDB's own documented REST backup endpoint if it exists cleanly for
   the free/unlicensed tag already verified in this repo
   (`ontotext/graphdb:10.8.1`), otherwise a Turtle export via the existing
   `TripleStoreTarget`/SPARQL `CONSTRUCT` is a legitimate, already-tested-
   components fallback — state explicitly which you used and why; (b) wipe
   or stand up a second repository/container; (c) restore into it; (d)
   rerun the real `maintenance_review.rq` file (fixed per the bug above) via
   `TripleStoreTarget`/`SPARQLBridge` against the restored instance and
   assert the identical WT-01/WO-9001 result `test_live_graphdb.py` already
   proves for load/query/restart.
2. **`kg_backup.py` proven against a live Neo4j instance** (the platform-
   wide half of "actual database backup and restore" — `kg_backup.py` is
   the platform's only real backup tool and it's entirely untested against
   a live database today, per `tests/unit/test_kg_backup.py` only covering
   URI-parsing helpers): a new e2e test using the `Neo4jContainer` pattern
   from `test_relational_postgres_neo4j.py` — ingest real data, run
   `kg_backup.py`'s `do_backup`, wipe or start a second Neo4j container,
   run `do_restore`, and rerun a real Cypher query proving the same data
   comes back. Then rewrite `recovery_exercise()` in
   `scripts/run_production_exercises.py` to call this real path (or clearly
   deprecate/relabel the digest-only version as a narrower check, not the
   platform's recovery proof) rather than leaving the file-hash-only
   version as the only thing `docs/local-evidence-runbook.md` documents.

If, after your own reading of the critique in full context, item 7 clearly
means only the Energy/GraphDB half, piece 2 can be scoped down to "flagged
as a separate, disclosed gap in the commit message" rather than built — use
judgment, but state the decision explicitly either way.

## Item 8: Energy evaluation and load report

> Publish an Energy evaluation and load report (answer correctness,
> evidence accuracy, abstention, tenant isolation, freshness, p95 latency,
> ingestion throughput).

This repo has a mature, consistent "evidence artifact" convention
(`report_schema_version` + `claim_policy` + a JSON dump — see
`graphrag/evidence/reports.py`, and `scripts/build_release_evidence_report.py`
for the most recent, closest precedent: it aggregates several existing
evidence sources into one report, degrading each missing section to an
explicit `null`/`available: false` with a reason rather than a silent
omission). Follow that exact pattern for this report rather than inventing a
new shape.

Concretely, for each named dimension:
- **Answer correctness / evidence accuracy / abstention**: the Energy demo's
  five fixed questions (`EnergyDemoService.questions`,
  `evals/energy_demo/questions.json`, `scripts/evaluate_energy_demo.py`)
  already assert expected answers including the "insufficient_evidence"
  abstention case — turn that into a scored report (pass rate, per-question
  detail), not just a pass/fail script exit code.
- **Tenant isolation**: `EnergyDemoService.answer()` already returns
  `{"status": "not_found", ...}` for a non-`energy-demo` tenant — prove it
  with a report entry, not just a unit test assertion.
- **Freshness**: the demo's `publication_report()`
  (`graphrag/domains/energy/publication.py`, built this session) already
  carries `published_at` and quarantine state — surface it.
- **p95 latency / ingestion throughput**: check for an existing benchmark
  harness to reuse (`graphrag/evaluation/graphrag_benchmark.py`,
  `scripts/benchmark_incremental_ingestion.py`,
  `scripts/measure_controlled_query_cost.py` are the closest precedents in
  this repo for latency/throughput-style evidence) rather than writing a
  bespoke timer from scratch. If the Energy demo's dataset is too small for
  a meaningful p95 (it may well be — ten synthetic turbines), say so
  plainly in the `claim_policy` field rather than reporting a misleadingly
  precise number over trivial data.

Output should be a script (`scripts/build_energy_evaluation_report.py`,
matching this repo's `scripts/build_*_report.py` naming convention) plus
enough test coverage that the report's shape and degrade-gracefully
behavior are proven the same way `test_release_evidence_report.py` proves
it for the release-evidence report — read that test file first as your
template.

## End-to-end demo scenario

> Source update → mappings produce RDF → invalid data quarantined → valid
> data updates graph → maintenance recommendation changes with traceable
> evidence → access restrictions enforced → service survives restart →
> restored database produces the same result.

This is a single, scripted narrative tying together work already done in
this session — it should NOT require new application code, only a new
end-to-end demonstration script/test that drives the existing pieces in
sequence and asserts on the chain:

1. **Source update → mappings produce RDF**: mutate the SAP-shaped SQLite
   fixture or the Snowflake-shaped JSON fixture
   (`graphrag/domains/energy/fixtures.py`,
   `data/energy_demo/snowflake_observations.json`), then construct a new
   `EnergyDemoService` and confirm the change flows through real R2RML/RML
   execution (`materialize_r2rml`/`materialize_rml`) into the graph.
2. **Invalid data quarantined**: include one row that violates
   `ontology/shapes/energy-asset-intelligence.shapes.ttl` in that source
   update and confirm `publication_report().quarantined_records` names it
   (`graphrag/domains/energy/publication.py`'s `DatasetPublisher`).
3. **Valid data updates graph, recommendation changes with traceable
   evidence**: confirm `answer("maintenance_review", ...)` reflects the new
   data and that its evidence/`query_rows`/`answer_source` fields trace back
   to the real SPARQL query file, not a hand-waved string.
4. **Access restrictions enforced**: reuse the existing tenant-isolation
   proof (a non-`energy-demo` tenant token gets no evidence) and the
   `POST /rollback` route's `require_scope("write")` gate
   (`api/routes/energy_demo.py`) as the "access restrictions" proof point.
5. **Service survives restart**: this is exactly what
   `tests/e2e/test_live_graphdb.py`'s restart test already proves for the
   GraphDB path — reuse it, don't reinvent it, unless item 7's design
   changed what "restart" means here.
6. **Restored database produces the same result**: chain directly off
   whatever item 7 built.

Write this as one clearly-narrated integration test or script
(`tests/e2e/test_energy_demo_end_to_end_scenario.py` or
`scripts/run_energy_demo_full_scenario.py`, your call, matching whichever
convention item 7 ends up using) that runs the six steps in order with an
assertion after each one, and produces readable output a human reviewer
could follow step-by-step — this is explicitly a *demonstration*, so
legibility of the narrative matters as much as the assertions passing.

## Order and process

Do item 7 first (item 8 doesn't depend on it, but the end-to-end scenario's
last step does), then item 8, then the end-to-end scenario — same one-item-
at-a-time, plan → approve → implement → verify → ask-to-commit loop as
every prior item in this project. Do not batch multiple items into one
commit. Do not start the next item without the previous one's commit (or an
explicit instruction to skip committing and move on).
