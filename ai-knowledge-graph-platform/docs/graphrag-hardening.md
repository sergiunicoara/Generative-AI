# GraphRAG hardening (2026-10): overview, status, metrics, migration

This page ties together the hardening work (plan:
`tasks/graphrag-hardening-plan.md`). Per-topic detail is linked below.

## Runtime flow after the change

```
source -> extraction -> entity resolution (ambiguity metric, review queue)
  -> publication gate: EXTRACTED -> STAGED -> VALIDATED -> PUBLISHED      [graph-validation.md]
     (BLOCKING records quarantined with rule ids; nothing invalid is written)
  -> graph write (origin / verification / schema version on edges)        [trust-metadata.md]
  -> schema registry: Dataset -[CONFORMS_TO]-> SchemaVersion, drift checks [schema-registry.md]
  -> change events -> dependency closure -> targeted invalidation         [invalidation.md]
query -> deterministic route (observe | enforce)                          [query-routing.md]
  -> retrieval with the shared validity predicate + trust factors         [trust-metadata.md]
  -> sufficiency (incl. all_evidence_stale) -> synthesis / bounded fallback
  -> explanation: evidence, paths, inferences, score breakdown, limitations [explanations.md]
MCP / agent tools -> allowlisted operations in guarded execution scopes   [mcp-security.md]
```

## Status

| Area | State |
|---|---|
| Publication gate, quarantine, retry, JUnit in CI | **Implemented and verified** (unit + mocked integration); live Cypher in CI e2e |
| Versioned schema registry, drift detection, schema provenance | **Implemented and verified**; registry Cypher proven only in CI e2e |
| Targeted invalidation, inferred-edge premises, recompute states | **Implemented and verified**; graph Cypher proven only in CI e2e |
| Trust metadata, shared validity predicate, score components | **Implemented and verified** |
| Query router | **Implemented**; route-specific retrieval **optional** (`query_router_policy: enforce`, default `observe`) |
| Explanations, trace authorization | **Implemented and verified** |
| Guarded MCP / agent operations | **Implemented and verified** (adversarial suite) |
| `ContextItem` contract (`graphrag/retrieval/context_item.py`) | **Experimental**: adapters + fusion, tested; not yet used for ranking |
| Chunk authority weighting | **Optional**, default off (not evaluated) |
| `include_superseded=False` default | **Planned, not implemented** (needs a golden-eval run) |
| Automatic re-answering of invalidated decisions | **Planned, not implemented** (hook exists) |
| Live quality / latency / token comparison of the router | **Not measured** (needs the running stack; `scripts/eval_routing.py --live`) |

## Audit follow-ups (2026-10-10)

Closed after the phases above (see `docs/IMPLEMENTATION_AUDIT.md` for the rows):

| Item | State |
|---|---|
| `BitemporalStore` compared stored datetimes with raw string parameters (null in Neo4j, rows silently dropped) | **Fixed**; params wrapped in `datetime()`; live proof in `tests/e2e/test_live_audit_followups.py` (CI only) |
| Relation provenance was last-write-wins for chunk / model / time | **Implemented**: per-document slots `doc_chunk_ids`, `doc_extraction_models`, `doc_observed_at`, index-aligned with `source_doc_ids`; a re-ingested document replaces only its own slot; reconcile removes a document's slot. Live proof: `tests/e2e/test_live_relation_provenance.py` (CI only). Edges written before this change have no slots until their next write |
| `extraction_model` was the configured label | **Implemented**: the provider-reported model for the call (context-local), falling back to the label on a cache hit or when the provider reports none |
| Runner-up entity candidates for ambiguous matches | **Implemented**: `AmbiguousMatch.runner_ups`; a `needs_review` entity persists the candidates it nearly merged into (best effort) |
| Evidence coverage | **Implemented**: `QueryResult.evidence_coverage` = lexical grounding ratio (no LLM); metrics below. `ClaimVerifier.verify_with_stats` exposes the LLM verifier's sentence counts |
| Multi-hop relation allowlist | **Optional**, default off: `retrieval.multihop_allowed_relations: []` (empty = any relation). Not evaluated; enable per corpus after a golden-eval run |
| Superseded documents and edges | `include_superseded=False` now also drops multi-hop edges and entity-neighbour edges whose `source_doc_id` is superseded. The default (`True`) is unchanged. **Community (global) search is still not covered**: communities carry no document link to filter on |
| Path validity / temporal-correctness production metrics | **Not implemented**: retrieval queries enforce edge currency by construction, so a live metric would be trivially 1.0; these are offline evaluation measures |

## Observability

All labels are bounded enums; tenant, entity and query identifiers appear only
in structured logs.

| Metric | Labels | Area |
|---|---|---|
| `graphrag_validation_violations_total` | rule_id, severity | validation |
| `graphrag_quarantined_records_total` | record_kind | validation |
| `graphrag_publication_batches_total` | outcome | validation |
| `graphrag_quarantine_retries_total` | outcome | validation |
| `graphrag_schema_drift_events_total` | outcome (match, version_change, rollback, blocked, startup_drift) | schema |
| `graphrag_invalidation_events_total` | kind, outcome | invalidation |
| `graphrag_invalidated_artifacts_total` | artifact_kind | invalidation |
| `graphrag_invalidated_cached_answers_total` | – | invalidation |
| `graphrag_recomputed_artifacts_total` | artifact_kind, outcome | invalidation |
| `graphrag_query_routes_total` | route, policy | routing |
| `graphrag_fallback_triggers_total` | reason | routing |
| `graphrag_insufficient_context_total` | reason | routing |
| `graphrag_stale_evidence_total` | – | trust |
| `graphrag_retrieval_latency_seconds` | mode | routing |
| `graphrag_graph_expansion_results` | – | routing |
| `graphrag_retrieval_authorization_denials_total` | – | security |
| `graphrag_answer_evidence_coverage` | – (histogram) | trust |
| `graphrag_unsupported_statements_total`, `graphrag_answer_statements_total` | – | trust (unsupported-claim rate = ratio) |
| `graphrag_superseded_evidence_used_total` | – | trust (non-current evidence in answers) |
| `graphrag_claims_verified_total`, `graphrag_claims_stripped_total` | – | LLM claim verifier |
| `graphrag_entity_resolution_ambiguous_total` | match_type | ingestion |
| `graphrag_capability_calls_total` (existing) | transport, capability (bounded), outcome | MCP |

Route accuracy from evaluation runs is written to
`evals/routing_eval_results.json`. Token usage and generation latency come
from the existing GenAI telemetry.

## Configuration added

`config/settings.yml`:

- `ingestion.publication_gate_enabled: true`
- `ontology.schema_registry: {mode: auto, datasets: {}}`
- `invalidation: {enabled, max_depth, max_dependents_per_query, recompute_inline, recompute_inline_limit}`
- `retrieval.query_router_policy: observe`
- `retrieval.trust_weighting_enabled: true`, `retrieval.chunk_authority_weighting_enabled: false`,
  `retrieval.retrieval_sufficiency_abstain_on_stale: true`

## Migration

1. Deploy; `graphrag/graph/schema.cypher` (idempotent) adds constraints for
   `QuarantinedRecord`, `SchemaVersion`/`Dataset`/`ValidationProfile`,
   `InvalidationEvent`, `DerivedArtifactState`. Workers apply it at startup, or
   run `python scripts/init_neo4j.py`.
2. First ontology load per tenant registers a new schema version (expected
   `version_change` drift event); legacy `OntologyVersion` rows are kept and
   deactivated.
3. Answer caches miss once (schema label now in the key).
4. `scripts/export_rdf.py` now requires `--tenant`; re-export each tenant.
5. Schedule `python scripts/recompute_stale.py --tenant T --sweep-expiry`.
6. Optional: `scripts/schema_registry.py` for explicit activation in
   `enforce` mode; `scripts/eval_routing.py --live` before enabling
   `query_router_policy: enforce`.

## Known limitations and next steps

- Run the CI e2e job (live Neo4j) — the registry, invalidation, trust and
  quarantine Cypher is proven there, not locally.
- Run the golden evaluations on the live stack before flipping
  `include_superseded`, `query_router_policy: enforce` or chunk authority
  weighting; no quality improvement is claimed until then.
- Remaining route misses (docs/query-routing.md) should be validated on a new
  held-out labelled set (the first one is now partly seen).
- Entities do not yet carry origin/verification; global (community) search
  does not yet apply supersession/quarantine filters (needs a community-to-document link first).
- The new Cypher (provenance slots, relation allowlist, superseded-edge exclusion, bitemporal
  `datetime()`) is proven only by the CI e2e job.
