# Dependency tracking and targeted invalidation

Status: **implemented and unit-verified** (2026-10-10). The graph-side Cypher
(dependency walk, state transitions, recompute queries) is proven by
`tests/e2e/test_live_invalidation.py`, which runs in CI only (no local Docker).

## What changed

Before: every correction, ingest or migration bumped a per-tenant corpus
revision. The revision is part of every answer-cache key, so one corrected
edge discarded every cached answer of the tenant, and nothing else derived
from the fact (recorded decisions, community summaries, inferred edges) was
told at all. `/kg/confidence/transition` and review-queue approvals did not
even bump the revision.

Now a change emits an `InvalidationEvent`; only artifacts that depend on it are
invalidated, and they are recomputed or sent to review.

## Events and triggers

| Event | Emitted by | Revision bump? |
|---|---|---|
| `RELATION_CHANGED` | `/kg/confidence/transition` (DISPUTED/RETRACTED: targeted; ASSERTED/APPROVED: additive), `/corrections/edge/reject` (targeted), `/corrections/edge/override` (additive) | only if additive |
| `FACT_CORRECTED` | `/corrections/entity/quarantine` single entity (targeted); subgraph quarantine and release (additive) | only if additive |
| `ER_REVISED` | review-queue approval, `/corrections/entity/split` (additive) | yes |
| `SUPERSEDED`, `DOCUMENT_REINGESTED` | ingestion (inside the ingestion bracket, which already bumps) | yes (ingestion) |
| `EVIDENCE_EXPIRED` | `sweep_expired()` / `scripts/recompute_stale.py --sweep-expiry` | no |
| `SCHEMA_CHANGED` | `OntologyRegistry.load` on a version change | no (schema label is in the cache key) |

**Additive vs removal.** A removal or weakening (retract, dispute, delete,
quarantine, expire) can only make answers wrong that were built on it, so
targeted invalidation is complete. An additive change (new or re-enabled fact,
alias) can change answers that cited nothing related, so those keep the
tenant-wide revision bump; their existing dependents are still marked.

Event ids are deterministic in `(tenant, kind, subjects, cause)`. A repeated
or redelivered event is a no-op (`duplicate: true`); an artifact is never
marked twice for the same event.

## Dependencies followed

From the changed facts, read-only and tenant-scoped (`dependency_index.py`):

- document -> its chunks (`Chunk.document_id`) and the edges it sources (`source_doc_ids`)
- edge -> its endpoint entities -> chunks that mention them
- edge or entity -> **inferred edges** whose `premise_keys` contain it, then
  inferred edges built on those (multi-hop, up to `max_depth`)
- chunks / documents / entities -> current `CommunitySummarySnapshot`s
- chunks / documents -> `CGContextManifest` -> `CGDecision` (recorded answers)
- schema change -> decisions whose manifest names another schema version
- chunks / entity names -> cached answers (answer-cache provenance index; it
  now indexes chunk ids as well as entity names)

The walk over-approximates (an answer citing an entity whose edge changed is
included even if that edge was not in its context): a false positive costs a
recomputation; a false negative would serve a stale answer.

**Inferred-edge premises (3b).** `ForwardChainingEngine` now records
`premise_keys` (`Type:Name|REL|Type:Name`) and `rule_version` (hash of the
rule definition) on every inferred edge, and rules no longer fire on
retracted, expired or quarantined premises. Inferred edges written before this
change have no premises; they depend on their endpoints.

## States

```
VALID -> NEEDS_REVIEW -> RECOMPUTING -> VALID
                                      -> INSUFFICIENT_EVIDENCE | VALIDATION_FAILED | RECOMPUTE_FAILED
                                      -> NEEDS_REVIEW (human review)
any state -> NEEDS_REVIEW (a newer event)
```

State lives on `(:DerivedArtifactState {tenant, kind, artifact_id})`, not on
the artifact: decisions and manifests are integrity-hashed and immutable, and
a snapshot keeps its text. Each invalidation adds
`(state)-[:INVALIDATED_BY]->(:InvalidationEvent {kind, reason, actor, cause,
subjects_json, summary_json})`, the trace of why. `version` increases on every
transition; a recompute that finishes after a newer invalidation cannot
overwrite it. Inferred edges and snapshots also get a `review_state` marker.

## Recomputation

Deterministic, no LLM (`recompute.py`), inline after each event (bounded by
`recompute_inline_limit`) and via `scripts/recompute_stale.py` /
`POST /corrections/invalidation/recompute`:

- **inferred edge**: premises re-checked; if invalid, the rule is re-run for that
  pair only. Re-derived -> VALID with new premises. Otherwise the edge is
  **retracted, not deleted** -> INSUFFICIENT_EVIDENCE.
- **community snapshot**: evidence intact -> VALID; otherwise the snapshot is
  closed (`transaction_to`, text kept) -> INSUFFICIENT_EVIDENCE; the next
  community rebuild writes a new one. This also closes previously orphaned
  snapshots once something they depend on changes.
- **decision**: no surviving evidence -> INSUFFICIENT_EVIDENCE; otherwise back to
  NEEDS_REVIEW with `requires_human_review` (re-answering needs the model; an
  `answer_recomputer` hook exists but is not wired).
- errors -> RECOMPUTE_FAILED.

Observe: `GET /corrections/invalidation/artifacts?state=NEEDS_REVIEW`,
`GET /corrections/invalidation/artifacts/{kind}/{id}` (state + events).

## Cache soundness

- Eviction happens inside the mutation bracket, while cache reads are bypassed.
- A targeted bracket finishes with `advance_revision=False`: the revision (and
  so every cache key) is unchanged, but `KGCorpusState.invalidation_seq`
  advances. The retriever re-reads the corpus state before storing an answer
  and skips the store if the revision or `invalidation_seq` moved since the
  query started, so an answer computed before a change is never cached after it.
- **Fallback to the revision bump** whenever targeting cannot be proven: closure
  truncated (`max_dependents_per_query`, `max_depth`), the answer cache is not
  shared across processes (in-memory cache: another replica or worker cannot be
  reached), eviction failed, or any error in `emit()`. `invalidation.enabled:
  false` restores the old behaviour for every event.

## Configuration

`config/settings.yml` → `invalidation:` `enabled`, `max_depth` (5),
`max_dependents_per_query` (5000), `recompute_inline` (true),
`recompute_inline_limit` (200).

## Migration

`schema.cypher` adds `invalidation_event_key`, `derived_artifact_state_key`
and `derived_artifact_state_queue` (applied at worker startup or
`scripts/init_neo4j.py`). No backfill: existing inferred edges gain premises
the next time inference runs; until then they depend on their endpoints.
Run `scripts/recompute_stale.py --tenant T --sweep-expiry` on a schedule to
process expiries.

## Metrics

`graphrag_invalidation_events_total{kind,outcome}` (outcome: targeted,
fallback_*, duplicate, error), `graphrag_invalidated_artifacts_total{artifact_kind}`,
`graphrag_invalidated_cached_answers_total`,
`graphrag_recomputed_artifacts_total{artifact_kind,outcome}`. Tenant and artifact
ids are in structured logs only.

## Limitations

- Conflict resolution (`/corrections/conflict/resolve`), GDPR erasure, community
  rebuilds and ontology migrations still use only the revision bump (and GDPR
  also flushes the tenant cache).
- Decisions are not re-answered automatically.
- Premise keys use `|` as a separator; an entity name containing `|` makes its
  premise unparseable for re-checking (it then falls back to re-derivation).
- Expiry is processed only when the sweep runs; there is no in-process scheduler.
