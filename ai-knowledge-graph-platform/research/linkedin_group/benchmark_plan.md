# Benchmark plan

Use frozen golden data split by query class: entity-local, two-hop, multi-hop, global/theme, temporal/context, ambiguous/no-entity, and adversarial unsupported.

Baseline:
BM25 + vector ANN + RRF + cross-encoder + depth=2 graph traversal + existing graph-aware score.

Experiment A:
Same baseline + query router + adaptive traversal (max depth 4, beam width 8, marginal-gain stop).

Experiment B:
A + calibrated text/graph/path/provenance fusion.

Experiment C:
A + community hierarchy and global/DRIFT-like search.

Experiment D:
A + Context Graph temporal episodes.

Measure:

- Recall@5/@10, MRR, nDCG.
- Context precision/recall, evidence coverage, path validity, contradiction rate.
- Faithfulness, groundedness, hallucination/unsupported-claim rate, abstention precision/recall.
- p50/p95 latency, token usage, Neo4j query count/cost, cache hit rate, index freshness.
- Per-tenant and per-query-class results.

Protocol:

- Same corpus, prompts, model, top-k, and budget.
- Time-based split for temporal data; tenant-disjoint validation for learned fusion.
- At least 3 repeated runs for LLM metrics; bootstrap confidence intervals for retrieval metrics.
- Report Pareto frontier, not a single aggregate score.
- Keep feature flags and log policy, seed entities, hops, paths, scores, and stop reasons.

Acceptance criteria:

- No P0 change ships if entity-local nDCG or groundedness regresses >2%.
- Adaptive traversal must improve multi-hop recall ≥5% or reduce graph cost ≥15% at equal recall.
- Fusion must improve nDCG/MRR in at least 3 query classes and reduce calibration error.
- Context Graph must pass temporal correctness, provenance, retention, and tenant-isolation tests.
- Any experiment that only improves LLM-as-a-judge without retrieval/evidence gains is rejected.

Example Neo4j expansion shape:

```cypher
MATCH (s:Entity)
WHERE s.id IN $seed_ids
CALL apoc.path.expandConfig(s, {
  maxLevel: $max_hops,
  uniqueness: 'NODE_PATH',
  bfs: true,
  limit: $candidate_limit,
  relationshipFilter: $relationship_allowlist
}) YIELD path
WITH path, reduce(c=1.0, r IN relationships(path) | c * coalesce(r.confidence, 0.5)) AS path_confidence
RETURN path, path_confidence
ORDER BY path_confidence DESC
LIMIT $beam_width
```

Use parameterized queries only; if APOC is unavailable, implement bounded Cypher patterns per hop.

