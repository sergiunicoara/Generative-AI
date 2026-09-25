# Remaining Audit Backlog

**Updated 2026-09-25** — see the dated addendum at the end of this file for
what changed in this session and what remains genuinely blocked in a
coding-only environment with no Docker daemon (this session verified that
directly: `dockerd`/`service docker start` both fail here, so the full dev
stack — Neo4j/Redis/RabbitMQ — and any testcontainers-backed e2e test cannot
run).

**As of:** 2026-09-22. Consolidates every still-open, worth-implementing item
from the two live audit trails in this repo:
[docs/IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md) (code-level, 2026-09-21/22)
and the historical
[docs/archive/audits/audit-2026-08-22-remaining-gaps.md](archive/audits/audit-2026-08-22-remaining-gaps.md)
(infra/evidence-level, 2026-08-22). Everything else either of those files
tracked has been closed — see those files for closed-item history and
evidence. This file replaces them as the day-to-day "what's left" reference;
they stay in place as historical record.

Ordered by value, code-fixable items first.

---

## Code-fixable (no external infra required)

### 1. Flip `include_superseded` default to `False`
Mechanism exists (`include_superseded` param threaded through
`vector_search_chunks`/`bm25_search_chunks`/`bm25_search_entities`/etc.) and
is live-tested to correctly exclude superseded docs. Still defaults to
`True` (today's behavior, i.e. supersession not actually enforced) because
the validation pass never ran — the dev stack (Docker: Neo4j/Redis/RabbitMQ)
wasn't up in the session that built it.
**To close:** bring up the full dev stack, run the aerospace golden eval
with `include_superseded=False` forced, confirm no regression against the
A128 baseline (esp. AUT-03), then flip the default.
**Still blocked as of 2026-09-25:** confirmed directly this session — this
environment has no Docker daemon (`dockerd`/`service docker start` both
fail: "Operation not permitted" / no socket), so the dev stack cannot be
brought up here. Left untouched rather than flipping the default blind.

### 2. Question-relevant conflict calibration — mechanism closed 2026-09-25, re-validation still pending
`sufficiency.py`'s abstention gate is a binary global disqualifier — any
open `Conflict` node reachable in retrieved context blocks the answer, even
when unrelated to the question. Enabling it as-is previously collapsed the
automotive golden set 7/10 → 2/10. Currently disabled by default, correctly.
**Closed this session:** the root cause was that `HybridRetriever` fetched
conflicts against `local_results["referenced_entities"]` — the entity
neighborhood of *every* retrieved chunk, before top-k/rerank narrows that
pool down to what the LLM actually sees — so a conflict about an entity that
never survives into the answer's context still unconditionally set
`reason_code="unresolved_conflict"`. Added
`HybridRetriever._filter_question_relevant_conflicts()`
(`graphrag/retrieval/hybrid_retriever.py`): re-scopes the entity lookup to
the chunks that will actually be ranked into context (the same `final_score`
ordering `ContextBuilder` sorts by, generously widened to also cover its
additive hop/link slots), and drops any conflict whose `src`/`tgt` isn't
among them. Gated by `conflict_relevance_filter_enabled` (default `true`,
config/settings.yml) so the prior unfiltered behavior is one flag away.
Unit-tested in `tests/unit/test_hybrid_retriever.py::TestQuestionRelevantConflictFiltering`
against the exact NEG-01/NEG-02-shaped scenario (a conflict entity outside
vs. inside the top-k context).
**Still open:** this was built and unit-tested with mocks, not re-validated
against the aerospace/automotive golden evals — this session's environment
has no Docker daemon, so the dev stack (Neo4j/Redis/RabbitMQ) required by
`scripts/run_golden_eval.py`/`run_automotive_eval.py` could not be brought
up. `retrieval_sufficiency_abstain_enabled` therefore correctly stays
disabled by default until someone with a working dev stack re-runs both
golden sets and confirms this filter actually fixes the automotive
collapse without regressing aerospace, then flips it on.

### 3. Defensible evidence and provenance — closed 2026-09-25
`QueryResult.citations` is a flat `list[str]` — no graph path, per-citation
confidence, or timestamp. Answers can't surface "claim + source + timestamp
+ path + confidence."
**Prior state (as of 2026-09-22):** `QueryResult.evidence: list[CitationEvidence]`
already existed as an additive sibling to `citations` (source_id, source_label,
path, confidence) — but `valid_from` was defined on the model and never once
set anywhere it was constructed (`context_builder.py`, `agentic_retriever.py`),
so every evidence entry's timestamp was silently always `None`. That was the
one genuinely still-open piece of "claim + source + timestamp + path +
confidence" — `path`/`confidence` were already real.
**Closed this session:** added `Neo4jClient.get_chunk_valid_from()`
(`graphrag/graph/neo4j_client.py`) — a per-chunk lookup of the source
document's bitemporal `valid_from`, mirroring the existing
`get_chunk_filenames()` pattern (same access-predicate interpolation, same
fails-open-on-error behavior). Wired into `LocalSearch.search()`
(`graphrag/retrieval/local_search.py`) as `chunk["_valid_from"]`, and
`ContextBuilder.build()` (`graphrag/retrieval/context_builder.py`) now reads
it into `CitationEvidence.valid_from` for both the primary and
hop-reserved/link-slot chunk evidence paths. No API shape change — `citations`
is untouched, `evidence` gains real data in a field it already had. Unit
tests: `tests/unit/test_context_builder.py` (valid_from present/absent),
`tests/unit/test_local_search.py::TestChunkValidFromWiring` (wiring +
fails-open on lookup failure), `tests/unit/test_document_link_topology.py`
(regression guard for the same "missing `f`-string prefix" Cypher-interpolation
bug class this codebase has hit before on sibling queries).

---

## Blocked on live infrastructure / credentials this environment lacks

### 4. Scale/availability evidence (0%)
No load-test scripts, Neo4j cluster/read-replica config, autoscaling on
queue age, or capacity report exist anywhere in the repo.
**Needs:** real 10x/100x/1000x load tests under sustained traffic — cannot
be produced from a coding session alone.

### 5. Complete retrieval-quality evaluation
Partially advanced: `evals/drift_search_quality_results.json` shows DRIFT
search has flat quality vs. the current agentic retriever on the aerospace
golden set (identical hit_rate/MRR across 33 questions) but 3.8x latency —
not worth adopting. Automotive eval pass rate raised 20%→70% with no
aerospace regression.
**Still missing:** RAGAS unscorable/refusal-case run, PageRank/GNN
ablations, GraphRAG-Bench comparison.
**Needs:** Docker stack up + LLM API budget for a full golden-set run.

### 6. Federated MCP + tested disaster recovery
Multi-issuer OAuth trust dispatch is real, working, unit-tested code
(`graphrag/core/issuer_trust.py`, `tests/unit/test_issuer_federation.py`).
**Still open:** an integration test against a *real* external IdP (Auth0,
Okta, etc. actually issuing a token this code verifies end-to-end),
external IdP trust-establishment tooling, and an actual DR restore drill
with measured RTO/RPO. Unit-testing the federation code is not the same as
tested disaster recovery.

### 7. Branch-protection enforcement on `main`
CI (`.github/workflows/ci.yml`) runs on push/PR to `main`, but whether
GitHub's branch-protection rule actually *requires* it to pass before merge
is unverified — this is a GitHub repo-settings toggle, not a file in the
repo.
**Checked 2026-09-25, still blocked:** this session had the GitHub MCP
connector attached to this repo, but its tool surface (list/get PRs, issues,
commits, branches, files, reviews) has no branch-protection read/write
endpoint — checking or setting this needs either `gh api
repos/:owner/:repo/branches/main/protection` with a token that has repo
admin scope, or the GitHub web UI (Settings → Branches), neither of which
this session had.
**Needs:** repo-admin-scoped `gh`/GitHub REST API access specifically for
the branch-protection endpoint (not just the GitHub MCP connector's current
tool set), or a human doing it in Settings → Branches.

### 8. Complete mutation score — real bug found and fixed 2026-09-25; baseline raised from ~50% to 77.6%, not yet CI-gated
`make mutation` (opt-in Mutmut campaign) exists, but no measured score or
CI-run report exists anywhere in the repo.
**Found this session:** `make mutation` was not merely "never run" — it was
broken. It called `mutmut run --paths-to-mutate <3 files>`, but
`requirements-dev.txt` pins `mutmut>=3.2.0`, and mutmut 3.x removed the
`--paths-to-mutate` CLI flag entirely (config-file only now, `source_paths`
in `[tool.mutmut]`). The command crashed with `FileNotFoundError` before
mutating a single line, for anyone who ran it since that pin landed. A
first, naive fix (`source_paths` = the 3 files directly) crashed differently
(`ModuleNotFoundError: No module named 'graphrag.core'`): mutmut 3.x's
`setup_source_paths()` strips the real source tree from `sys.path` and only
re-adds it under `mutants/` for the conventional root names `.`/`src`/`source`
— pointing `source_paths` at individual files leaves everything *outside*
those files unimportable once the real tree is stripped.
**Fixed:** `pyproject.toml`'s new `[tool.mutmut]` uses `source_paths = ["."]`
(so the whole repo mirrors correctly into `mutants/` and imports resolve)
with `only_mutate` restricting actual mutation to the same 3 adapters the
Makefile always targeted, plus `pytest_add_cli_args_test_selection` scoping
the test run to those adapters' own mocked unit tests (no live infra needed
— this also works in this session's no-Docker environment). `make mutation`
now runs `mutmut run` (config-driven) followed by `mutmut results`.
**Recorded score progression (this session, local runs, not CI)** — 859
mutants total across the 3 files throughout (source unchanged; only test
coverage improved):

| Stage | Killed | Survived | No tests | Timeout | Kill rate |
|---|---|---|---|---|---|
| Pre-session baseline (3 pre-existing tests) | 429 | 400 | 25 | 5 | 49.9% |
| +21 tests (round 1: transport error paths, aerospace prompt, r2rml helpers) | 577 | 252 | 25 | 5 | 67.2% |
| +18 tests (round 2: transport dispatch/`_body`/`_tool_schema`, r2rml error paths + field assertions) | **667** | **187** | **4** | **1** | **77.6%** |

39 unit tests were added across the two rounds. Round 1:
`tests/unit/test_mcp_transport_20260728.py` (+10: error-response branches,
`ping`, a denied tool call, `tools/list` entitlement filtering),
`tests/unit/test_answer_policy.py` (+3: the aerospace-positive prompt path
and citation dedup), `tests/unit/test_r2rml_obda.py` (+8: the small
`_one`/`_local_name`/`_identifier_from_template` helpers). Round 2:
`tests/unit/test_mcp_transport_20260728.py` (+10 more: `_header`,
`_json_type`, `_tool_schema`'s required-field/tenant-exclusion logic,
`_body`'s chunk assembly and disconnect handling, and
`ProtocolVersionDispatch`'s modern-vs-legacy routing — none of this had any
test before), `tests/unit/test_r2rml_obda.py` (+8 more: the four previously
untested `r2rml_to_mapping` error paths, full field-level assertions on a
successful mapping, and `FederatedOBDAIngestor`'s three guard clauses).
Manually spot-checked several remaining survivors (`mutmut show <mutant>`):
the tool and its test-to-mutant attribution work correctly (`mutmut
tests-for-mutant` correctly names the exact test exercising each mutated
function) — remaining survivors are a mix of genuinely equivalent mutants
(e.g. `rsplit(x, 1)[-1]` vs. `rsplit(x)[-1]` — identical result when only the
last segment is read) and real, narrower gaps, concentrated in
`r2rml_to_mapping` (only exercised through full-mapping fixtures that
tolerate many low-level mutations) and `FederatedOBDAIngestor.validate`.
Full breakdown, per-module counts, and example diffs:
`artifacts/mutation-campaign-2026-09-25.json`.
**Correction:** an earlier revision of this entry reported "0 killed" from
this session's very first run, before any new tests were added. That number
was a measurement artifact, not a real result — `mutmut results` hides
already-killed mutants by default; only `mutmut results --all true` shows
them. The true pre-session baseline was 429/859 killed (~50%), not 0/430.
Caught and corrected before this file was finalized.
**Still open:** this is a strong local baseline, not a CI-gated campaign.
187 survivors remain (largely `graphrag/ingestion/r2rml.py`'s
`r2rml_to_mapping` and `FederatedOBDAIngestor.validate`) — the
next-highest-value place to look if this is worth investing further in, and
wiring `make mutation` into a protected (not per-PR) scheduled CI job so
this score doesn't silently regress.

---

## Deliberately conditional (not real gaps, don't schedule)

- Opportunity/deal-level ACLs — no such concept exists in the platform;
  only build if a concrete CRM product requirement emerges.
- Neo4j schema-level tenant `IS NOT NULL` constraints — blocked by Neo4j
  Community edition; closed instead by a live CI test layer. Revisit only
  if Enterprise is licensed.
- Persistent `ToolPolicy` audit-log flush — low priority while that path
  isn't used in production.
- Runner-up entity-resolution candidate ranking at query time (persistence
  is already done, just not surfaced live) — no current consumer.

---

## 2026-09-25 session addendum

This session's environment has no Docker daemon and no repo-admin-scoped
GitHub API access — both verified directly (see items #1, #2, #7 above), not
assumed from the original doc's framing. Work was scoped to what a
coding-only session can actually close or meaningfully advance:

- **Closed:** item #3 (defensible evidence and provenance) — the
  `CitationEvidence.valid_from` timestamp, the one field of the "claim +
  source + timestamp + path + confidence" shape that had no data source
  until now.
- **Mechanism closed, re-validation still pending:** item #2
  (question-relevant conflict calibration) — the exact miscalibration
  described in this doc (conflicts scoped to the full pre-rerank retrieval
  pool instead of the evidence actually used) is fixed and unit-tested, but
  not re-run against the golden evals (no Docker here). Abstention stays
  disabled by default until that validation happens.
- **Real bug found and fixed, first score recorded:** item #8's `make
  mutation` was silently broken (not just "never run") since the
  `mutmut>=3.2.0` pin — fixed, and a first baseline mutation score is now
  recorded in `artifacts/mutation-campaign-2026-09-25.json`.
- **Confirmed still blocked, more precisely than before:** item #1 (no
  Docker to run the golden eval), item #7 (the GitHub MCP connector's tool
  set has no branch-protection endpoint, distinct from "no `gh` at all").
- **Not attempted:** items #4, #5, #6 — unchanged from the prior write-up;
  genuinely need real load infrastructure, a billed full golden-set run, and
  an external IdP / DR drill respectively, none of which a coding session
  can manufacture.
