# Evaluation and controlled route benchmarking

## GraphRAG-Benchmark adapter

`graphrag.evaluation.graphrag_benchmark` reads JSONL benchmark records with a
`question` (or `query`) and an ID. It retains unfamiliar benchmark fields, so
dataset-specific labels survive the adapter. Every report records a SHA-256
dataset fingerprint, the tenant, the exact named retrieval profile, its
effective overrides, route fingerprint, answer, citations, and measured local
latency.

Run routes only against the same indexed corpus and tenant:

```powershell
python scripts/run_graphrag_benchmark.py `
  --questions <GraphRAG-Benchmark-jsonl> `
  --tenant <tenant> `
  --route full:full `
  --route vector:vector_only `
  --route text:text_hybrid `
  --output artifacts/graphrag-benchmark-report.json
```

The adapter does not claim official leaderboard scoring. Feed its answer output
to the version-pinned official evaluator for the selected dataset/task, and
publish both artifacts together. A comparison is invalid if data fingerprint,
tenant corpus revision, model route, prompt version, or retrieval route differs.

## Evaluator fallback

`evaluation.backend` defaults to `ragas`. If an import, judge call, or timeout
raises an evaluation error and `ragas_fallback_to_reference` is enabled, the
worker records a deterministic `reference` score instead. It measures lexical
support of the answer in supplied context/reference; it is deliberately not
presented as semantic RAGAS faithfulness. The `evaluation_source` field is
persisted on KPI events and must accompany any reported metric.

Set `evaluation.backend: reference` for an offline, dependency-light run. Use
this only for regression signals and route comparisons, not as a substitute
for a calibrated semantic-quality result.

## Relational R2RML / OBDA mappings

`graphrag.ingestion.r2rml.r2rml_to_mapping()` converts a safe R2RML subset
into the platform's SHACL-gated mapping contract. It accepts table names,
single-column subject templates, classes, `rdfs:label` columns, and
parent-triples-map joins. Unsupported semantics fail before the connector reads
data. The versioned example is at
`ontology/mappings/supply-chain.r2rml.ttl`.

Use `FederatedOBDAIngestor` for several independently authorized sources in
the same tenant. It validates every source before starting any materialization;
it does not create an ungoverned cross-tenant or cross-region virtual join.

## Fuzz and mutation checks

`make test-fuzz` runs generated-input tests over tenant/prompt/protocol and
mapping boundaries. `make mutation` runs Mutmut over the domain-policy, MCP
transport, and R2RML adapter code. Mutation results require human review:
surviving mutants are either a missing assertion to fix or a documented
equivalent mutation; they are never silently accepted.
