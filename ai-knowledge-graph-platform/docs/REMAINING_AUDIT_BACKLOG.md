# Remaining Audit Backlog

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
`QueryResult.citations` is a flat `list[str]` — no graph path, per-citation
confidence, or timestamp. Answers can't surface "claim + source + timestamp
+ path + confidence."
**To close:** needs its own migration plan — this is a public API change
that breaks every eval script's citation-recall scoring, not a quick patch.

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
**Needs:** authenticated `gh`/GitHub API access (`gh auth login` or the
GitHub MCP connector).

### 8. Complete mutation score
`make mutation` (opt-in Mutmut campaign) exists, but no measured score or
CI-run report exists anywhere in the repo.
**Needs:** a CI run of the mutation campaign and a recorded score.

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
