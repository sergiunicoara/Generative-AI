# Answer explanations and proof traces

Status: **implemented and unit-verified** (2026-10-10).

Every answer from the hybrid retriever (synthesis, agentic fallback, structured
template) now carries `QueryResult.explanation` and `QueryResult.confidence`.
The async API (`/query` via RabbitMQ) no longer drops them: the persisted result
also includes `route`, `schema_version`, `retrieval_sufficiency`,
`evidence_bundle` and `retrieval_trajectory`, which were previously lost on that
path.

## Format

```json
{
  "answer": "...",
  "confidence": 0.81,
  "confidence_components": {"grounding": 1.0, "evidence_currency": 1.0, "evidence_trust": 0.9,
                            "capped_by_insufficiency": false, "note": "heuristic, not a calibrated probability"},
  "route": "RELATIONAL", "route_reason": "legacy_relational",
  "evidence": [{"id": "chunk-id", "kind": "chunk", "source": "AD-2024-03", "document_id": "...",
                "cited": true, "excerpt": "...", "origin": "EXTRACTED", "current": true,
                "trust": {...}, "score_components": {...}, "retrieved_via": "vector+bm25"}],
  "citations": ["AD-2024-03"],
  "graph_paths": [{"edges": [{"src": "...", "relation": "...", "tgt": "...", "origin": "...",
                              "confidence_state": "...", "verification_status": "...",
                              "confidence": 0.72, "confidence_before_trust": 0.8, "trust_factors": {...}}],
                   "length": 1},
                  {"via_entity": "...", "length": 2, "path_confidence": 0.7, "reached_chunk": "..."}],
  "inferences": [{"fact": "A -OWNS-> C", "rule": "owns_transitive", "rule_version": "...",
                  "premises": ["ORG:A|OWNS|ORG:B", "ORG:B|OWNS|ORG:C"], "derivation_recorded": true}],
  "entity_resolution": [{"entity": "...", "type": "...", "resolution_status": "...", "resolution_method": "..."}],
  "score_breakdown": {"text": ..., "vector": ..., "bm25": ..., "graph": ..., "path": ...,
                      "temporal": ..., "provenance": ..., "authority": ..., "per_evidence": {...}},
  "grounding": {"statements": 3, "grounded": 3, "unsupported_statements": [], "grounding_ratio": 1.0},
  "schema_version": "name@version#hash12",
  "fallback": {"triggered": false, "reason": null},
  "insufficient_context": null,
  "limitations": [{"code": "...", "message": "..."}]
}
```

- **Origin** of each fact: EXTRACTED / IMPORTED / INFERRED / GENERATED / MANUAL
  (docs/trust-metadata.md); inferred facts list their rule, rule version and
  premises.
- **Grounding**: each answer statement (>= 3 content words) is checked against
  the retrieved evidence text and graph facts; unsupported statements are
  listed and lower the confidence. They are never presented as grounded.
- **Confidence** = grounding ratio x share of cited evidence that is current x
  mean evidence trust factor, capped at 0.25 when the sufficiency gate failed.
  Deterministic and explainable; not calibrated.
- **Limitation codes**: `unsupported_statements`, `non_current_evidence`,
  `unverified_evidence`, `disputed_facts`, `schema_unregistered`,
  `insufficient_context:<reason>`, `agentic_fallback`, `route_observed_only`,
  `inference_without_recorded_premises`, `graph_explanation_withheld`,
  `structured_template`.

## Authorization

- The explanation is built only from evidence the retrieval queries returned
  for this caller; every chunk/document read applies the tenant and ACL
  predicate, so it cannot introduce unauthorized nodes.
- Under access control the graph parts (paths, inferences, entity resolution)
  are withheld, matching the retrieval safe mode, and
  `graph_explanation_withheld` says so.
- **Decision traces** (`GET /context-graph/traces/{id}` and `/replay`): embeddings
  are never returned; with ACL enabled, chunks and documents the caller cannot
  read are removed using the same fail-closed predicate, and if anything was
  removed the decision's free text, the manifest's task input and the
  observations are redacted (`redaction` field). Previously these routes filtered
  by tenant only and returned raw chunk text.
- Internal traces (structured logs) stay tenant-scoped; no tenant or entity
  identifiers are used as metric labels.

## Limitations

- Grounding is lexical (content-word overlap), not entailment; it can miss a
  paraphrased unsupported claim and is reported as a heuristic.
- Community (global) evidence has no per-statement grounding text beyond its
  citations.
- Entity-resolution status is shown when the entity node recorded it.
