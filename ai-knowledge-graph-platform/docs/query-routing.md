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

Offline route accuracy (measured 2026-10-10):

| Category | n | Router | Legacy planner (projected) |
|---|---|---|---|
| AGGREGATION | 6 | 1.00 | 0.00 |
| AMBIGUOUS | 5 | 0.80 | 0.00 |
| ENTITY_LOOKUP | 6 | 0.83 | 0.00 |
| FACTUAL_LOOKUP | 10 | 0.90 | 1.00 |
| MULTI_HOP | 6 | 0.83 | 0.83 |
| RELATIONAL | 9 | 0.56 | 0.56 |
| TEMPORAL | 7 | 1.00 | 0.00 |
| **all** | 49 | **0.837** | 0.408 |

Read with care: the legacy planner has no ENTITY_LOOKUP / AGGREGATION /
TEMPORAL / AMBIGUOUS classes, so most of the gap is structural. On the shared
classes the router is equal (RELATIONAL, MULTI_HOP) or slightly worse
(FACTUAL_LOOKUP: one identifier containing a date, "AD 2024-03-07", is routed
TEMPORAL). Known misses (kept, not tuned away): "What is EASA?" -> AMBIGUOUS
(a one-word entity counts as too few content words); relational questions
phrased without relation keywords ("Who owns X?", "Which components does X
depend on?", authority comparisons) -> FACTUAL_LOOKUP; "Is this still valid?"
-> FACTUAL_LOOKUP. A fix should be validated on a held-out set, not this one.

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
