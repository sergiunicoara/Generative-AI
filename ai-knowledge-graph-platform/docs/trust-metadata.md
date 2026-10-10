# Trust metadata in retrieval

Status: **implemented and unit-verified** (2026-10-10). Live Cypher behaviour is
proven by `tests/e2e/test_live_trust_metadata.py` (CI only, no local Docker).
**No quality improvement is claimed:** the golden evaluations need the live
stack and were not re-run; ranking changes are listed below so they can be
measured.

## Fields

Extends the existing provenance (plan decision D5) instead of a parallel system:

| Field | Where | Meaning |
|---|---|---|
| `source_type` | edges, entities | unchanged (`document` / `inferred` / `llm` / `manual`) |
| `origin` | edges | EXTRACTED, IMPORTED, INFERRED, GENERATED, MANUAL. Written on ingest (`IMPORTED` for relational sources, `INFERRED` by rules, `MANUAL` by override); derived from `source_type` when absent |
| `verification_status` | edges | UNVERIFIED unless a reviewer acted: APPROVED -> VERIFIED, RETRACTED -> REJECTED (`verified_by`, `verified_at`), manual override -> VERIFIED. **Never set by ingestion; having a source is not verification.** |
| `generated_by` | edges | extraction model, or `rule:<name>` for inferred edges |
| `observed_at` | edges | the existing `extracted_at` |
| `confidence`, `confidence_state` | edges | unchanged; now read by retrieval |
| `authority_level`, `superseded_by`, `valid_from/valid_to` | documents | unchanged; now returned with every chunk |
| `stale_after` | edges (`Relation.stale_after`), documents (property) | needs re-verification after this instant |
| `schema_version` | edges, documents | active schema-registry label |
| `tenant` | everything | unchanged |

## One validity predicate

`graphrag/graph/validity.py` defines "usable as current": edge not RETRACTED,
inside its `valid_from`/`valid_to` window at `as_of` (or **now** when no
`as_of` is given), endpoints not quarantined. It is used by every fact read:
entity neighbours, multi-hop traversal, the GNN/context subgraph, MCP
`kg.entity.lookup`, the controlled fact queries (`kg.facts.query`), the agent
`get_neighbors` tool, and community building.

Behaviour changes:

- **Retracted edges are no longer retrieved anywhere** (previously
  `confidence_state` was never read).
- **Expired edges are excluded without `as_of`** (previously only when the caller
  passed `as_of`). Historical questions still see them via `as_of`.
- Controlled fact queries and the agent tool previously applied no filter at all.
- `get_entity_neighbors` is now tenant-scoped on the chunk and entity too.

DISPUTED edges stay retrievable but are down-weighted and labelled
`(disputed)` in the prompt; they are never treated as settled.

**Not changed (pending a measured golden-eval run, plan D5):**
`include_superseded` still defaults to True. Superseded evidence is retrieved
but no longer treated as current: it is down-weighted (x0.5), flagged in its
trust record, and cannot make an answer look grounded on its own (below).

## Scoring

`graphrag/graph/trust.py` assesses each edge and chunk into separate factors:
`authority`, `origin`, `verification`, `state`, `temporal`, `supersession`,
`staleness`. Nothing is collapsed silently:

- **Graph scoring:** edge confidence is multiplied by its trust factor before GNN
  scoring (`confidence_before_trust` and `trust` kept on the edge).
- **Chunk ranking:** the score is multiplied by supersession x staleness x
  temporal; document authority is applied only when
  `chunk_authority_weighting_enabled` (default **off**, not yet evaluated).
- **Score components:** raw `vector_score`, `bm25_score`, `bm25_entity_score`,
  `rrf_score` (previously overwritten by RRF), `rerank_score`, `text_score`,
  `gnn_score`, path scores, `pagerank_tiebreak`, `feedback_score`,
  `trust_factor_applied`, `score_before_trust` and `final_score` are kept per
  chunk and returned on `CitationEvidence.score_components`, with
  `CitationEvidence.trust`, `valid_from`, `valid_to`.

## Conflicts

`ContradictionDetector.suggest_resolution` (`GET
/corrections/conflict/{id}/suggestion`) ranks the conflict's source documents
by trust and suggests a winner only when one *current* source dominates by a
margin; otherwise `unresolved`. Every claim is returned with its components.
It never writes; resolution stays a human action.

## Insufficient context

New sufficiency reason `all_evidence_stale`: every retrieved chunk is
superseded, expired, not yet valid, stale, retracted or rejected. The answer
then abstains (`retrieval_sufficiency_abstain_on_stale: true`) even though
general abstention stays off (plan D6). Unauthorized evidence never reaches
the result (ACL filters in the queries), and the existing
`insufficient_evidence` / `low_evidence_score` reasons are unchanged.

## Configuration

`retrieval.trust_weighting_enabled` (true), `retrieval.chunk_authority_weighting_enabled`
(false), `retrieval.retrieval_sufficiency_abstain_on_stale` (true).

## Limitations

- Entities carry no `origin`/`verification_status` yet (edges and documents do).
- Existing edges get `origin` on their next write; until then it is derived.
- SPARQL / RDF export (`scripts/export_rdf.py`) does not apply the predicate, and
  for the `default` tenant it exports every tenant's edges (to be fixed with
  the Phase 7 guarded-operations work).
- Global (community) search does not yet apply supersession or quarantine filters.
