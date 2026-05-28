# Lessons — GraphRAG Project

Patterns learned from corrections and bugs found during this project.
Each entry has: the mistake, the root cause, and the rule to prevent recurrence.

---

## L01 — Config drift: documented flags that don't gate anything

**What happened:**
Three flags existed in `settings.yml` with clear comments (`authority_weighting_enabled`,
`community_staleness_check_on_ingest`, `auto_rebuild_communities`). None of them were read
at runtime. The code ran unconditionally regardless of the config value.

**Root cause:**
Flag was added to config as documentation/intent, then the code was written without
wiring up the read.

**Rule:**
> Whenever a config flag is added, immediately grep for the code path it claims to gate
> and verify the read + conditional is present. A flag with no runtime effect is worse
> than no flag — it creates false confidence.

---

## L02 — Formula bugs: documented formula ≠ implemented formula

**What happened:**
The GNN blend formula was correctly documented as `final = α·text + β·gnn`.
The fallback path (no entity data available) returned the raw score without applying α.
Silent wrong scores on entity-sparse chunks. No error, no warning.

**Root cause:**
The fallback branch was written separately from the main path and didn't replicate the
full formula.

**Rule:**
> After writing a formula, trace every code path — especially edge cases and fallbacks —
> and verify each one applies the full formula. Rename helper functions to make their
> role obvious (`_fallback_score` → `_text_score`) so callers know what they're calling.

---

## L03 — Session state recorded before it exists

**What happened:**
`record_turn()` was called inside `local_search.py` where `answer=""` is unavoidable —
the LLM hadn't run yet. Every stored session turn had an empty answer, making the
session context useless for follow-up query enrichment.

**Root cause:**
The recording was placed at the nearest logical point (after retrieval) rather than the
correct point (after generation).

**Rule:**
> Session/state recording must happen after the full pipeline completes, not at the
> retrieval step. If the answer isn't available yet, the record is incomplete — wait
> for it. Move the call to the outermost orchestrator that has all the outputs.

---

## L04 — Tenant isolation: name-only entity matching

**What happened:**
Entity MERGEs, relation writes (`merge_relation`), and subgraph fetches
(`get_entity_relations_subgraph`) all matched entities by `name` + `tenant` only,
not by `(name, type, tenant)`. In a tenant with "Apple" as ORG and "Apple" as PRODUCT,
a single MERGE would collapse them. Structurally valid. Semantically corrupt.

**Root cause:**
The unique key for an entity is `(name, type, tenant)` per the data model, but MATCH
patterns were written as `{name: $name, tenant: $tenant}` — one dimension short.

**Rule:**
> Every MATCH/MERGE on an Entity node must use the full 3-part key: `(name, type, tenant)`.
> Auditing checklist: grep for `{name: $` in `.cypher` and `.py` files and verify `type`
> and `tenant` are always present alongside name.

---

## L05 — Silent degradation without startup signal

**What happened:**
Two critical dependencies silently degraded:
1. `redis[asyncio]` missing or Redis unreachable → silent in-memory fallback; sessions
   died on restart with no log entry.
2. `graspologic` missing → Leiden fell back to connected components; global search
   quality degraded undetectably.

**Root cause:**
Both were written as resilient by default — "try, warn, fall back". Reasonable for
development. Dangerous in production where the operator doesn't know the system is
running on a degraded path.

**Rule:**
> Critical dependencies that materially affect answer quality must have a `strict` mode
> that fails hard at startup (`verify_connection()`, `require_leiden: true`).
> Fallback is opt-in via explicit config, not the default.
> Strict violations belong at startup — not in hot request paths where failure drops
> live requests. Per-operation: degrade gracefully but log at ERROR.

---

## L06 — Global scoping in a multi-tenant architecture

**What happened:**
Contradiction detection, `GraphEvaluator` metrics, health snapshots, and `find_conflicts_for_entity`
were all globally scoped. In a multi-tenant deployment this caused:
- Cross-tenant conflicts (tenant A's facts contradicting tenant B's)
- Misleading aggregate metrics (a struggling tenant averaged out by a healthy one)
- Health trends mixing all tenants into a single timeseries

**Root cause:**
Features were designed and tested single-tenant, then tenant isolation was added
incrementally — but analytics/evaluation code was never revisited.

**Rule:**
> Every analytics, evaluation, and health query must include a tenant filter from day one.
> The pattern `WHERE ($tenant = 'default' OR node.tenant = $tenant)` makes the filter
> opt-out (wildcard on `'default'`) so existing callers don't break.
> When adding tenant isolation to a system, explicitly audit: aggregates, snapshots,
> conflict nodes, and any "get all X" query.

---

## L07 — Multi-source evidence compression

**What happened:**
`merge_relation` overwrote `source_doc_id` on each ingestion pass. Only the latest
document was recorded on the edge. Multi-source contradiction detection (which checks
whether the same relation was asserted by independent documents) was structurally broken
because the evidence trail was destroyed.

**Root cause:**
MERGE + SET overwrites all properties. No thought was given to properties that should
accumulate over time.

**Rule:**
> When merging edges that can be produced by multiple ingestion passes, always accumulate
> evidence in a list (`source_doc_ids`) rather than overwrite. Use the Cypher pattern:
> ```cypher
> r.source_doc_ids = CASE
>   WHEN r.source_doc_ids IS NULL THEN [$id]
>   WHEN $id IN r.source_doc_ids THEN r.source_doc_ids
>   ELSE r.source_doc_ids + [$id]
> END
> ```
> Apply this anywhere provenance matters for downstream analysis.

---

## L08 — Legacy helpers that drift from the data model

**What happened:**
`find_conflicts_for_entity` assumed multiple distinct RELATES_TO edges per `(src, rel, tgt)`
triple — the pre-accumulation model. After `source_doc_ids` was introduced (one edge per
triple, list of sources), the method became structurally wrong but didn't crash: it just
returned no conflicts because `size(sources) > 1` could never be true with the new model.

**Root cause:**
Core data model changed (from multi-edge to single-edge + list), but helper methods that
queried the old pattern weren't audited.

**Rule:**
> When refactoring a core data model assumption, audit ALL methods that query the same
> pattern — not just the primary path. Search for the old pattern in Cypher strings
> after every model change. Helper/diagnostic methods go stale silently.

---

## L09 — APOC dependency in core query paths

**What happened:**
A Cypher query used an APOC list function to flatten a nested list. APOC is an optional
Neo4j plugin — not available in all deployments. The query would silently fail or error
in environments without it.

**Root cause:**
APOC was available in the dev environment so the dependency wasn't noticed.

**Rule:**
> Never use APOC functions in core query paths without a verified fallback.
> For list operations, prefer pure Cypher equivalents:
> - Flatten: `reduce(acc = [], lst IN collect(r.ids) | acc + lst)`
> - Any/all: `any(x IN list WHERE x = $val)`
> Reserve APOC for scripts and maintenance tasks where the dependency can be explicitly
> documented.

---

## L10 — Strict mode belongs at startup, not per-operation

**What happened:**
Initial proposal was to raise on every Redis operation failure in strict mode. This would
kill a live request mid-answer on a transient blip — worse UX than silently completing
with memory fallback for that one turn.

**Root cause:**
Strict mode was conflated with "never tolerate any failure" rather than "make failures
visible at the right time".

**Rule:**
> Strict/fail-fast guarantees belong at process startup (`lifespan` hook, `__init__`).
> Per-operation failures during live requests should always degrade gracefully — but log
> at ERROR in strict mode so ops can see the drift.
> Three failure tiers:
> 1. Missing dependency at init → raise immediately (no requests served in broken state)
> 2. Unreachable dependency at startup → `verify_connection()` raises → process aborts
> 3. Transient failure mid-request → log ERROR, fall back, complete the request

---

## L11 — Alias resolution changes type, not just name

**What happened:**
After alias resolution, the code captured the canonical name (`src_canonical[0]`) but
continued using the original entity's type (`src.type`) for the ontology validator and
`merge_relation`. If the canonical entity had a different type, the MATCH would fail to
find it and relation writes would silently drop.

**Root cause:**
Alias resolution returns a `(name, type)` tuple but only the name was extracted.

**Rule:**
> After alias resolution, always capture both dimensions of the canonical identity:
> ```python
> src_name = src_canonical[0] if src_canonical else src.name
> src_type = src_canonical[1] if src_canonical else src.type
> ```
> Never use a pre-resolution attribute after resolution has run — the resolved value
> is the ground truth.

---

## L12 — Internal patterns repeat the same bug as the public interface

**What happened:**
When `merge_relation` was updated to add `src_type`/`tgt_type` to the primary MATCH,
the secondary MATCH inside the same function (for deep provenance) still used the old
name-only pattern. Same bug, different code path, same function body.

**Root cause:**
Fixed the function signature and the first MATCH, didn't scan the rest of the function
for repeated patterns.

**Rule:**
> When fixing a correctness bug in a MATCH pattern, grep the entire function body (not
> just the line reported) for the same pattern. Secondary queries inside the same method
> often repeat the same assumption. Fix all instances in the same commit.

---

## L13 — README lags behind the implementation

**What happened:**
After extensive feature additions (GNN, alias resolution, contradiction detection, tenant
isolation, Redis sessions, 12+ new files), the README still described the original 5-step
pipeline, 10 key features, and the original project structure. Users reading it would have
a fundamentally incorrect picture of the system.

**Root cause:**
README updates were deferred to "when the feature is stable" — which meant never.

**Rule:**
> Update the README in the same PR as any significant feature addition. At minimum:
> - Add the feature to the Key Features table
> - Add new files to the Project Structure
> - Update any pipeline diagrams
> If the feature touches configuration, add/update the relevant config block.

---

## L14 — Contradiction detection scope mismatch

**What happened:**
`ContradictionDetector.scan()` queried without a tenant filter. In a multi-tenant
deployment, two tenants could assert contradictory facts about entities with the same name,
generating spurious conflicts that neither tenant owned or could resolve.

**Root cause:**
Contradiction detection was built when the system was single-tenant; tenant isolation was
added later but the detector wasn't revisited.

**Rule:**
> Any query that traverses RELATES_TO edges must filter by tenant. Contradiction,
> cycle detection, and graph health checks are especially risky because they aggregate
> across the whole graph — a missing tenant filter silently mixes data from unrelated
> tenants.

---

## L15 — Health metrics must carry their tenant context

**What happened:**
`GraphHealthSnapshot` nodes were written without a `tenant` field. `get_trend()` returned
all snapshots mixed together, making it impossible to track health trends per tenant or
compare tenants over time.

**Root cause:**
Snapshots were designed for single-tenant use; tenant was not part of the data model.

**Rule:**
> Any persisted metric, snapshot, or event node must store its tenant at creation time.
> Retrofitting tenant onto existing nodes is painful and lossy. Add it on the first write,
> even if the system is currently single-tenant.
> Standard field: `tenant: $tenant` in every `CREATE` for analytics nodes.
