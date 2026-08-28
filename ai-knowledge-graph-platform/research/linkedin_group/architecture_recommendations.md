# Executive conclusion

After analysing 51 provisional unique LinkedIn feed records and 6 external technical sources, the highest-value improvements are:

1. Query-adaptive retrieval routing and graph traversal.
2. Confidence-aware text/graph/path fusion.
3. A temporal Context Graph episode layer.
4. Retrieval sufficiency and abstention gates.
5. Community/global retrieval for non-entity questions.

# KEEP

- Hybrid BM25 + vector retrieval and RRF.
- Cross-encoder reranking as a strong baseline.
- Provenance, bitemporality, contradiction detection, entity-resolution tiers.
- OWL-RL/SPARQL/Datalog for explicit verification where applicable.
- OpenTelemetry, Prometheus, Langfuse, golden sets, and groundedness evaluation.

# MODIFY

- Replace fixed depth=2 with query-conditioned bounded traversal.
- Replace fixed 90/10 fusion with calibrated query/risk-aware fusion.
- Move from entity-only routing to entity/local, community/global, basic/vector, and DRIFT-like modes.
- Add joint evidence-bundle reranking after bounded graph expansion for relationship questions.

# ADD

- `RetrievalMode`: BASIC, LOCAL, GLOBAL, DRIFT, TEMPORAL.
- `TraversalPolicy`: hop budget, beam width, edge allowlist, stop rule, cost budget.
- `EvidenceBundle`: claims, source spans, entities, paths, temporal intervals, confidence.
- `RetrievalSufficiency`: coverage, contradiction risk, unresolved entities, abstain reason.
- Context Graph episode nodes with retention and policy controls.

# EXPERIMENT

- Personalized PageRank seeded from linked entities and query concepts.
- Leiden community reports and multi-level community retrieval.
- LightRAG-style dual-level retrieval and incremental update path.
- Final cross-encoder ranking over graph-grounded evidence bundles.

# REJECT

- Replacing the measured baseline solely because a LinkedIn post claims deterministic systems are superior.
- GNNs as the default retrieval authority without ablation and calibration evidence.
- Full autonomous graph-to-action execution without allowlists, provenance, rollback, and human approval.
- Global map-reduce search for every query due cost and latency.

A concrete interface:

```python
class RetrievalPolicy:
    mode: Literal["basic","local","global","drift","temporal"]
    max_hops: int
    beam_width: int
    min_marginal_gain: float
    max_graph_cost: int
    require_verified_paths: bool
```

Use `graph_expand(seeds, policy)`, stop on marginal evidence gain, then score:
`alpha(query,risk)*text + beta(query,risk)*graph + gamma*path_confidence + delta*provenance`.
