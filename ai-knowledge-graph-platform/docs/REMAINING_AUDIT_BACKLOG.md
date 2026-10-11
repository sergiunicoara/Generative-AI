# Remaining Audit Backlog

**As of:** 2026-09-22 (item 3 updated 2026-10-10; see docs/graphrag-hardening.md for the hardening work since). Consolidates every still-open, worth-implementing item
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
**Done 2026-10-11.** On the live aerospace stack (12 documents, `ENV=development`, the project's own
`scripts/run_golden_eval.py`): default `True` 30/34 (88%); forced `False` 30/34 twice, with the same four
failures (MH-02, MH-03, AUT-02, TMP-03) and AUT-03 still passing. The recorded earlier baselines were
28-29/34. `HybridRetriever.retrieve_and_answer` now defaults to `include_superseded=False`; the Neo4j
client methods keep their own `True` default and receive the value explicitly. Caveat: the graph in
that run held only 34 edges from 315 extracted relations (the general ontology rejects most aerospace
relations unless `ontology.domain_ontology_path` is set; the pre-change commit behaved the same), so
graph-dependent questions are lightly exercised. Original description follows.
Mechanism exists (`include_superseded` param threaded through
`vector_search_chunks`/`bm25_search_chunks`/`bm25_search_entities`/etc.) and
is live-tested to correctly exclude superseded docs. Still defaults to
`True` (today's behavior, i.e. supersession not actually enforced) because
the validation pass never ran — the dev stack (Docker: Neo4j/Redis/RabbitMQ)
wasn't up in the session that built it.
**To close:** bring up the full dev stack, run the aerospace golden eval
with `include_superseded=False` forced, confirm no regression against the
A128 baseline (esp. AUT-03), then flip the default.

### 2. Question-relevant conflict calibration
`sufficiency.py`'s abstention gate is a binary global disqualifier — any
open `Conflict` node reachable in retrieved context blocks the answer, even
when unrelated to the question. Enabling it as-is previously collapsed the
automotive golden set 7/10 → 2/10. Currently disabled by default, correctly.
**To close:** make the conflict signal question-relevant (does a
conflicting claim appear in the ranked evidence actually used?) and/or a
graded penalty instead of a binary gate, then re-validate against the
golden set before enabling.

### 3. Defensible evidence and provenance
**Largely closed 2026-10-10** without changing `QueryResult.citations`: every
answer now carries a structured `explanation` (evidence with trust and score
components, graph paths, inferences, grounding, limitations) and per-citation
`CitationEvidence`; relation provenance is kept per document. See
[docs/explanations.md](explanations.md) and
[docs/graphrag-hardening.md](graphrag-hardening.md). `citations` itself stays a
flat list for eval-script compatibility.

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
**Checked 2026-10-11 (read-only GitHub API):** `main` requires 4 status checks (`Test (Python 3.11)`, `Lint`,
`Validate a clean installation`, `Validate deployment definitions`) with `strict` (branch must be up to date).
`enforce_admins` is **off**, so a repository admin can push directly past the checks (as happened with the
hardening commits). Turn on "Include administrators" in the branch rule to close that. The paragraph below is
the original, now-answered question.
CI (`../.github/workflows/ai-knowledge-graph-platform-ci.yml`, at the
monorepo root — an in-project copy at `.github/workflows/ci.yml` was
deleted again on 2026-09-23 after being silently reintroduced; see
[docs/archive/audits/audit-2026-09-23.md](archive/audits/audit-2026-09-23.md)
item 8) runs on push/PR to `main`, but whether GitHub's branch-protection
rule actually *requires* it to pass before merge is unverified — this is a
GitHub repo-settings toggle, not a file in the repo.
**Needs:** authenticated `gh`/GitHub API access (`gh auth login` or the
GitHub MCP connector).

### 8. Complete mutation score
**Measured 2026-10-11** (mutmut 3.8 on Linux CI, manual workflow
`ai-knowledge-graph-platform mutation`, run 38115846381) for `graphrag/retrieval/answer_policy.py`,
`mcp_server/transport_20260728.py` and `graphrag/ingestion/r2rml.py`: **859 mutants, 425 killed
(49.5%), 5 timed out, 404 survived, 25 with no covering test.** Roughly half of the mutations
to these adapters are not detected by their tests. The surviving list is in the run's
`mutation-results.txt` artifact: most survivors are in `r2rml` (94 listed by module prefix),
the transport's JSON-schema/header helpers (`_json_type`, `_tool_schema`, `_header`) and the
answer prompt text. Next: strengthen those tests, then rerun the workflow. Not supported on native Windows.
**Re-measured after strengthening the tests (same day, run 38135880176): 861 mutants, 784 killed
(91.1%), 73 survived, 4 with no covering test.** Survivors by module: answer_policy 8, transport 34,
r2rml 35 (module-prefix counts from `mutation-results.txt`). The new tests also found and fixed a real
bug: the MCP adapter's `structuredContent` was not serialised with `default=str`, so a capability
result containing a date/datetime made the response fail. Another finding, left unchanged on purpose:
the answer prompt literals begin with a stray backslash line (`"""\` in the source); removing it is a
prompt change and needs a golden-eval run. Remaining survivors are mostly equivalent or cosmetic
mutations; inspect them before adding more tests.
Original description:
`make mutation` (opt-in Mutmut campaign) exists, but no measured score or
CI-run report exists anywhere in the repo.
**Needs:** a CI run of the mutation campaign and a recorded score.

### 9. Apply the GKE hardening Terraform
Dataplane V2, private nodes, authorized networks and Cloud NAT were added to
`infra/terraform/` on 2026-09-25 (audit-2026-09-23 closure pass) but never
`terraform validate`d or applied. Both the datapath and private-cluster
settings force cluster replacement.
**Needs:** Terraform CLI + GCP credentials, a replacement/migration plan, and
`master_authorized_networks` filled in for CI/operator access.

### 10. Upstream CVE fixes for `ragas` / `diskcache`
**Re-checked 2026-10-11:** `pip index versions` still lists ragas 0.4.3 and diskcache 5.6.3 as the latest, so no fixed release exists yet.
PYSEC-2026-3046 (`ragas==0.4.3`) and PYSEC-2026-2447 (`diskcache==5.6.3`,
pulled in only via `ragas`) have no fixed release yet — both pins are the
latest versions. Evaluation pipeline only.
**To close:** upgrade when a fix ships, then re-run the golden eval.

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
  is done, including for ambiguous matches; just not surfaced live) — no current consumer.
