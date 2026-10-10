# Query-aware retrieval routing

Status: **router implemented and evaluated offline; route-specific retrieval is
opt-in** (`query_router_policy: enforce`). Default `observe` records the route
on every query and keeps the existing retrieval behaviour, because the effect
of enforcement on answer quality, latency and tokens has **not been measured**
on the live stack.

## Routes

`graphrag/retrieval/query_router.py` (deterministic regex/phrase rules, no LLM)
extends the keyword planner (`query_planner.py`). The planner's five classes
remain as aliases (`legacy_class`) and keep their measured mode/top_k plans.

| Route | Rule (first match wins) | Retrieval under `enforce` |
|---|---|---|
| TEMPORAL | caller passed `valid_at`; or a temporal phrase ("as of", "before", "previously", "superseded", an ISO date ...) | `as_of` = the date in the question when present; validity judged at that date; no agentic fallback |
| AGGREGATION | "how many", "count", "list all", "number of", "which X have no", "average" ... | allowlisted read-only template (`controlled_query.py`) when one matches, answered deterministically with citations; never under ACL; falls back to normal retrieval when no template or no rows |
| MULTI_HOP | legacy `multi_hop` | configured multi-hop traversal + hybrid |
| RELATIONAL | legacy `relational` / `contradiction` | graph traversal + hybrid |
| AMBIGUOUS | < 2 content words, or an unresolved reference ("it", "that one") without session context | hybrid; bounded agentic fallback when evidence is insufficient |
| ENTITY_LOOKUP | "who is / what is / tell me about / describe" + short + no relation words | hybrid with entity context, multi-hop limited to 1 hop |
| FACTUAL_LOOKUP | everything else (legacy `factoid`, `negative`) | vector/BM25 hybrid; **no graph expansion** (no multi-hop, GNN or entity neighbourhood) |

Every result carries `QueryResult.route` and `route_reason`, and the trace log
`hybrid_retriever.route` records route, reason, policy and legacy class.
Tenant and authorization filters are applied inside the retrieval queries
before any evidence is returned, whatever the route.

## Fallback bounds

The agentic (IRCoT) fallback is bounded by `agentic_max_steps` and is now
**non-recursive**: a context variable marks an active fallback and no fallback
can start inside another. The agentic retriever searches with `LocalSearch`
only and never re-enters the hybrid retriever. Temporal and aggregation routes
do not use it under `enforce`; ACL mode keeps it disabled as before.

## Evaluation

`evals/routing_cases.json`: 49 hand-labelled questions across all seven routes,
including stale, conflicting and unauthorized scenarios. Labels were written
before the router was run and were not edited afterwards.
`python scripts/eval_routing.py` writes `evals/routing_eval_results.json`
(reproducibility is asserted by a unit test).

Offline route accuracy on the dev set (measured 2026-10-10, after the fixes
below; **fitted to this set**, so treat the held-out figures as the honest ones):

| Category | n | Router | Legacy planner (projected) |
|---|---|---|---|
| AGGREGATION | 6 | 1.00 | 0.00 |
| AMBIGUOUS | 5 | 1.00 | 0.00 |
| ENTITY_LOOKUP | 6 | 0.83 | 0.00 |
| FACTUAL_LOOKUP | 10 | 1.00 | 1.00 |
| MULTI_HOP | 6 | 1.00 | 0.83 |
| RELATIONAL | 9 | 1.00 | 0.56 |
| TEMPORAL | 7 | 1.00 | 0.00 |
| **all** | 49 | **0.980** | 0.408 |

Read with care: the legacy planner has no ENTITY_LOOKUP / AGGREGATION /
TEMPORAL / AMBIGUOUS classes, so much of the gap is structural.
The first version missed 8 of these 49 (F02, E02, E06, R05, R06, M05, X04,
C01). They were fixed with general rules (below), not per-question patches, and
validated on a **separate held-out set**.

### Fixes and held-out validation

Rules added: date-shaped document identifiers ("AD 2024-03-07", "SB-2023-11-04")
and non-calendar numbers are not time constraints; relation verbs ("owns",
"supplies", "depends on", "operated by", ...) and authority comparisons route
RELATIONAL; two or more relation cues, or "through which / ultimately /
intermediaries", route MULTI_HOP; a single named subject ("What is ICAO?") is
ENTITY_LOOKUP, and a bare reference with nothing to resolve it ("Is this still
valid?") is AMBIGUOUS unless a session exists or a named token is present;
ENTITY_LOOKUP needs at most 4 content words so property questions ("What is the
stated limit for ...") stay FACTUAL_LOOKUP.

`evals/routing_cases_heldout.json` (43 cases, different phrasings) was written
and scored **before** the fixes, then scored once after them. It was not used to
tune any rule:

| Set | Router before | Router after | Legacy planner |
|---|---|---|---|
| Dev (49, fitted) | 83.67% | 97.96% | 40.82% |
| Held-out (43) | 51.16% | 93.02% | 20.93% |

The held-out "before" figure shows the earlier dev score was optimistic. The
remaining misses are recorded, not tuned away:

- Dev U02 "Who is the confidential supplier for ...?" is now RELATIONAL; the
  label (ENTITY_LOOKUP) is arguably wrong, but labels are not edited after a run.
- Held-out HF02 "What does service bulletin SB-2023-11-04 require operators to
  do?" -> RELATIONAL ("operators" matches the relation-verb rule).
- HR04 "Which standards does the quality manual reference?" -> FACTUAL_LOOKUP
  ("reference" is not a relation cue).
- HR09 "Which parts are made by Bolt Supplier GmbH?" -> MULTI_HOP (the word
  "Supplier" in a name counts as a second relation cue).

Further rule changes should be validated on a new held-out set; this one is now
partly seen. Routing accuracy is a structural measure only: it says nothing
about answer quality, latency or tokens.

**Not measured** (needs the running stack): context precision/recall,
faithfulness (RAGAS judge), latency p50/p95, token usage, fallback trigger rate
and insufficient-context rate per route. `scripts/eval_routing.py --live
--tenant T` measures latency, fallback and insufficient-context rates for
`observe` vs `enforce`; no improvement is claimed until it is run.

## Metrics

`graphrag_query_routes_total{route,policy}`, `graphrag_fallback_triggers_total{reason}`,
`graphrag_insufficient_context_total{reason}`, `graphrag_stale_evidence_total`,
`graphrag_retrieval_latency_seconds{mode}`, `graphrag_graph_expansion_results`,
`graphrag_retrieval_authorization_denials_total`,
`graphrag_entity_resolution_ambiguous_total{match_type}`. Token usage and
generation latency come from the existing GenAI telemetry. No query text,
entity id or tenant id is used as a label.
