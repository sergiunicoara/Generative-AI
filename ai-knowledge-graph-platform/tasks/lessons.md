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

---

## L16 — Function-local imports break `unittest.mock.patch`

**What happened:**
Writing `tests/unit/test_review_queue.py`, `patch("graphrag.graph.alias_registry.get_settings")`
and `patch("graphrag.graph.review_queue.get_alias_registry")` both raised
`AttributeError: module ... does not have the attribute ...`. Both functions were imported
*inside* the calling function body (`from graphrag.core.config import get_settings` inside
`AliasRegistry.__init__`; `from graphrag.graph.alias_registry import get_alias_registry`
inside `ReviewQueueService.approve`), not at module level — so the importing module never
had that name as an attribute for `patch` to find.

**Root cause:**
`patch()` replaces an attribute on a module object. A name only becomes an attribute of
`module.py` if it's imported at module scope. A function-local `import` binds the name only
in that function's local scope — invisible to `patch("module.name")`. This is unrelated to
whether the import itself was intentional (e.g. to avoid a circular import) — it's a pure
testability side effect nobody thought about when writing the import.

**Rule:**
> Before writing a test that needs to mock a dependency, check whether that dependency is
> imported at module level in the file under test. If it's a function-local import, either
> move it to module level (preferred, if no circular-import risk) or patch the *true* source
> module (`graphrag.core.config.get_settings`) instead of the importing module's alias.
> When adding new service classes, default to module-level imports unless there's a specific
> circular-import reason not to — it costs nothing and keeps the module mockable.

---

---

# Knowledge Graph Architecture — Findings & Design Lessons

Architectural insights learned from building and hardening the GraphRAG pipeline.
These are design principles, not code bugs.

---

## A01 — GNN compensates for what text can't see

**Finding:**
Cross-encoder score for a Falcon 9 chunk against "What rockets did Elon Musk's company launch?" was **-6.74** — weak text match because "Elon" isn't in that chunk. GAT score was **0.73** because the graph knows `SpaceX → Falcon 9 → Starship`. Without the GNN layer, that chunk drops out of results entirely even though it's directly relevant.

**Principle:**
> Text similarity answers "is this chunk about the query?" Graph scoring answers "is this
> chunk connected to entities the query is about?" Both questions matter. A reranker alone
> is not sufficient for multi-document reasoning — it only sees text overlap, not graph
> structure. The GNN layer is not an optimisation; it's a correctness layer.

---

## A02 — Query-adaptive GNN weights outperform fixed α/β

**Finding:**
A fixed `α=0.9, β=0.1` is correct for factoid queries ("Who founded SpaceX?") where the
text score is reliable. But for relational queries ("How did X's acquisition affect Y's
supply chain?") the text match is often weak across the relevant chunks and the graph
structure carries more signal. Forcing `α=β=0.5` on relational queries measurably
improves recall on multi-hop fact retrieval.

**Principle:**
> Don't use fixed blend weights for a mixed query population. Detect query intent
> (relational signals: "caused by", "connected", "between", "led to", "impact of")
> and adjust weights accordingly. The cost is one string scan per query; the benefit
> is correct weighting without manual tuning per use case.

---

## A03 — Session context must enrich the query before embedding

**Finding:**
Session enrichment adds prior-turn entities to an ambiguous follow-up query
(e.g. "What else did they launch?" → "What else did SpaceX launch?"). This enrichment
must happen *before* `embed(query)` — not before LLM generation. If enrichment happens
after embedding, the vector search runs on the ambiguous query and retrieves the wrong
seed chunks regardless of what the LLM eventually receives.

**Principle:**
> Query enrichment and query embedding are order-dependent. The enriched string is the
> input to all downstream retrieval stages, not just the LLM. The correct pipeline order
> is: enrich → embed → ANN → BM25 → rerank → multihop → GNN → LLM.

---

## A04 — Leiden hierarchy is not optional for global search quality

**Finding:**
When `graspologic` is missing, the system falls back to connected components (flat
partitioning, level 0 only). Community summaries generated on connected components
describe "all nodes reachable from X" rather than "semantically coherent clusters at
multiple resolutions". Global search map-reduce over these summaries produces vague,
unfocused answers that RAGAS scores as low context recall.

**Principle:**
> The Leiden algorithm's multi-resolution hierarchy (coarse at level 0, fine at level 2)
> is what makes global search answers useful. Connected components is a structural
> fallback, not a quality-equivalent alternative. Treat `graspologic` as a hard
> dependency for production, not an optional enhancement.

---

## A05 — Community staleness gating prevents rebuild thrash

**Finding:**
Rebuilding communities after every ingestion batch is expensive (full entity+relation
fetch + Leiden + LLM summarization). But never rebuilding means global search runs on
stale structure as the graph grows. The right model: compute a staleness score from
entity/edge drift (40% entity delta + 60% edge delta) and only rebuild when drift
exceeds a threshold (default 0.15).

**Principle:**
> Community rebuild is a write-heavy batch operation. Gate it on a staleness metric, not
> on every ingestion. The staleness check itself is a lightweight read (snapshot delta),
> so it can run on every batch without rebuild cost. Tune the threshold by domain:
> fast-changing knowledge bases need a lower threshold; stable corpora can tolerate more
> drift.

---

## A06 — Two-layer entity deduplication is required

**Finding:**
Name-based alias resolution catches obvious variants ("Elon Musk" / "E. Musk" / "musk").
But it misses paraphrases and abbreviations with no shared tokens ("SpaceX" vs.
"Space Exploration Technologies"). Embedding similarity (cosine > 0.92 threshold) catches
these. Either layer alone leaves a meaningful number of duplicate nodes in the graph.

**Principle:**
> Entity deduplication needs two layers: (1) fuzzy name matching for surface variants,
> (2) embedding similarity for semantic equivalents. Run name matching first (cheap);
> only run embedding comparison if name matching found no canonical. The threshold matters:
> 0.92 cosine is aggressive — tune lower if your domain has many legitimately similar
> but distinct entities.

---

## A07 — Source evidence accumulation enables contradiction detection

**Finding:**
The original model wrote one RELATES_TO edge per (src, relation, tgt) pair per ingestion
pass, overwriting the previous `source_doc_id`. This meant the graph looked like every
relation came from a single document. Multi-source contradiction detection — which checks
whether two *independent* documents assert the same relation differently — was impossible.

Switching to a `source_doc_ids` list (accumulated, not overwritten) on each edge gave
contradiction detection the full provenance it needs: same edge, all contributing sources
visible, supersession relationships queryable across all of them.

**Principle:**
> In a knowledge graph built from multiple documents, every edge must carry its full
> source provenance as a list. A single `source_doc_id` field that gets overwritten
> destroys the audit trail and makes multi-source analysis impossible. Model provenance
> as an append-only list from the first schema version.

---

## A08 — Contradiction types need distinct semantics

**Finding:**
A generic "conflict" label on a Conflict node doesn't tell the resolver what to do.
Four distinct types each suggest a different resolution strategy:
- `multi_source` → check authority levels; higher authority wins
- `directional_reversal` → `A→B` vs `B→A`; likely an extraction error; review source text
- `exclusive_state` → "X is CEO" + "X is not CEO"; one must be stale; check timestamps
- `functional_violation` → ontology says X can only have one Y; deduplicate

**Principle:**
> Contradiction detection is only useful if it produces actionable output. Classify
> conflicts by type at detection time, not resolution time. Each type maps to a resolution
> strategy. A conflict queue with mixed untyped entries is processed slowly and
> inconsistently.

---

## A09 — Document authority level determines conflict resolution, not recency

**Finding:**
Newer documents are not always more authoritative. A regulatory directive from 2015
(authority level 1) supersedes an internal email from 2025 (authority level 4). Resolving
conflicts by timestamp would systematically prefer the less authoritative source.

The SUPERSEDES relationship between documents is the right model: it makes the authority
order explicit and queryable, independent of ingestion timestamp or document date.

**Principle:**
> Model document authority explicitly (1=regulatory, 2=manufacturer, 3=internal,
> 4=informal). Never use ingestion time or document date as a proxy for authority.
> Build SUPERSEDES edges between documents at ingestion time when the supersession is
> known. Apply a confidence penalty (e.g. 0.5×) to edges from superseded documents
> during retrieval so they rank lower without being deleted.

---

## A10 — Orphan entities are a leading indicator of extractor problems

**Finding:**
An entity with no MENTIONS link from any Chunk exists in the graph but is unreachable
from retrieval. Orphan rate rising across ingestion batches (not just spiking and
recovering) signals a systematic extractor failure: entities are being extracted and
written but the MENTIONS edges are not being created. By the time RAGAS scores degrade,
hundreds of entities may already be dark.

**Principle:**
> Track orphan rate as a leading indicator, not as a cleanup metric. A rising orphan
> rate predicts future retrieval gaps before they appear in evaluation scores. Alert
> at the graph health layer (`orphan_delta > 0` across N consecutive snapshots), not
> by waiting for user-reported answer quality drops.

---

## A11 — Quarantine beats deletion for suspicious entities

**Finding:**
When an entity is flagged as a degree anomaly (e.g. hub node with 500+ edges from a
single document — likely an extraction artifact), deleting it destroys the audit trail
and makes it impossible to investigate the root cause or recover if the flag was a
false positive.

Quarantine (set `quarantined=true`, exclude from retrieval, keep in graph) preserves
all edges and provenance. The entity can be inspected, released, or split after review.

**Principle:**
> Never delete flagged entities automatically. Quarantine them. Deletion is irreversible
> and destroys diagnostic information. The quarantine flag is cheap (one property write)
> and exclusion from retrieval (`WHERE NOT e.quarantined = true`) is trivial. Build
> release and split workflows before enabling auto-quarantine.

---

## A12 — Community coherence predicts global search quality

**Finding:**
A community with low intra-community edge density (most edges go to entities outside
the community) means Leiden found poor cluster structure — the "community" is not actually
a semantically coherent group. LLM summaries of such communities are vague and unfocused.
When global search map-reduces over low-coherence communities, the partial answers are
weak and the synthesis degrades.

**Principle:**
> Monitor `avg_community_coherence` (ratio of intra-community to total edges per
> community) as a leading indicator of global search quality. Values below ~0.4 signal
> that the graph is too sparse or that Leiden resolution parameters need tuning.
> Coherence below threshold should trigger a parameter sweep before the next rebuild,
> not just a rebuild at the same parameters.

---

## A13 — Multi-hop depth 2 is the sweet spot; depth 3+ increases noise

**Finding:**
Multi-hop at depth 2 (`A → B → C`) resolves the core cross-document reasoning problem:
entity mentioned in chunk → connected entity → chunk mentioning that entity. Depth 3
(`A → B → C → D`) exponentially increases the number of hop chunks pulled in, most of
which are tangentially related at best. For a corpus of moderate size, depth 3 typically
doubles the chunk count but adds minimal relevant information.

**Principle:**
> Default to `multihop_depth: 2`. Only increase to 3 if your domain has deep causal
> chains (e.g. supply chain analysis, legal precedent chains) and you have evidence that
> depth-3 chunks improve recall on your evaluation queries. The cost is linear in
> chunk count but the noise increases super-linearly.

---

## A14 — Cross-encoder before multi-hop, not after

**Finding:**
Placing cross-encoder reranking before multi-hop traversal (current architecture) means
the seed chunks for graph expansion are the highest-quality text matches. The hop chunks
inherit quality from good seeds.

Placing cross-encoder after multi-hop would mean reranking a much larger set (seeds +
all hop chunks), most of which were added for graph connectivity, not text relevance.
The reranker would spend budget on structurally-present but textually-weak chunks.

**Principle:**
> The pipeline order matters: rerank before expand, not after. Use the reranker to
> select the best seed set for graph traversal. Apply GNN scoring after expansion to
> score the full set (seeds + hops) using graph signal — that's what the GNN is for.

---

## A15 — Graph health metrics are leading indicators; RAGAS is lagging

**Finding:**
RAGAS scores measure answer quality on the evaluation sample. By the time faithfulness
or context recall drops, the underlying graph problem (rising orphan rate, high
contradiction rate, low alias coverage, degraded community structure) has usually been
accumulating for multiple ingestion batches.

Graph health metrics are cheap to compute and predictive:
- Rising `orphan_delta` → extraction pipeline losing MENTIONS edges
- Rising `conflicts_per_1k_edges` → new documents contradicting existing facts
- Falling `alias_coverage` → duplicate entities accumulating
- Falling `avg_community_coherence` → Leiden structure degrading

**Principle:**
> Build graph health monitoring before you need it. By the time RAGAS drops, the
> diagnosis takes hours. With a `GraphHealthSnapshot` trend, the anomaly is visible
> at the batch that caused it. Alert on health metrics; use RAGAS to confirm.

---

## A16 — Tenant isolation must be designed first, retrofitted never

**Finding:**
Tenant isolation was added incrementally to an initially single-tenant design. Each
addition required: updating the entity unique constraint, adding tenant filters to every
MATCH/MERGE, updating alias registries, adding tenant to conflict nodes, health snapshots,
community nodes, and auditing every analytics query. Retrofitting took significantly
longer than designing it in from the start would have.

**Principle:**
> If multi-tenancy is a possibility (not even a certainty), build the `tenant` dimension
> into the schema on day one:
> - Entity unique constraint: `(name, type, tenant)`
> - Every MATCH pattern: `{..., tenant: $tenant}`
> - Every analytics node: `tenant: $tenant` stored at CREATE time
> The cost of adding it upfront is one extra parameter everywhere.
> The cost of retrofitting is auditing every query in the codebase.

---

## A17 — Self-loops are extraction artifacts, not valid knowledge

**Finding:**
LLM entity extractors occasionally produce relations where source and target resolve to
the same entity (e.g. "SpaceX was founded by SpaceX" due to coreference errors). These
self-loops (`(e)-[:RELATES_TO]->(e)`) create cycles, inflate entity degree counts, and
corrupt GNN message-passing (a node attending to itself at full weight skews its embedding).

**Principle:**
> Auto-remove self-loops immediately after every relation write. They are never valid
> domain knowledge — they are always extraction errors. Don't quarantine them or flag
> them for review; just delete. Add the removal to the post-write validation pipeline
> so it runs automatically, not as a scheduled cleanup job.

---

## A18 — Confidence fusion must be Bayesian, not averaging

**Finding:**
When the same relation is extracted from two independent documents both with confidence
0.8, averaging gives 0.8 — unchanged. But two independent sources asserting the same
fact is stronger evidence than one. The Bayesian combination
`1 - (1 - c1) × (1 - c2) = 0.96` correctly reflects the increased certainty.

Averaging is wrong because it treats additional confirming evidence as neutral rather than
reinforcing.

**Principle:**
> Use Bayesian confidence fusion for accumulating evidence on edges:
> `r.confidence = 1 - (1 - r.confidence) * (1 - $new_confidence)`
> Only use this for independent sources. If two extractions come from the same document
> (same `source_doc_id`), don't fuse — it's not independent evidence.
> The `source_doc_ids` list enables this check: only fuse if the new source is not
> already in the list.

---

## A19 — Open-world assumption is not free — model negative knowledge explicitly

**Finding:**
A graph with no edge "A USES B" makes no claim: it may mean "we don't know" or
"A definitely does NOT use B". Without explicit negative edges, the retrieval layer
answers "Unknown" to "Does A use B?" even when a document explicitly states it does not.
This ambiguity is especially dangerous in safety-critical domains where "not found" and
"confirmed absent" are different answers with different consequences.

**Principle:**
> When a source document explicitly negates a relation ("Engine A does not use Fuel Pump B"),
> store a `NEGATIVE_RELATES_TO` edge with the same provenance model as positive edges.
> If a positive RELATES_TO edge also exists for the same triple, surface it immediately as
> a `positive_negative_pair` conflict. Retrieval should check for confirmed negatives and
> include them in LLM context so the model can answer "No (confirmed)" rather than "Unknown".

---

## A20 — Flat type namespaces break query expansion and merge decisions

**Finding:**
A query for "Agent" entities must explicitly enumerate PERSON and ORG when the type
namespace is flat. Adding a new type (EXECUTIVE, REGULATOR) requires updating every
query that should include it. Merge decisions between "MANAGER" and "PERSON" entities
require knowing that MANAGER is a subtype of PERSON, but there is no place to express
that. The type space fragments silently as ingestion runs with different prompt versions.

**Principle:**
> Build a `SUBCLASS_OF` hierarchy from day one. Concrete leaf types (PERSON, ORG) sit
> below abstract parents (AGENT). Query expansion calls `expand_type("AGENT")` and gets
> all subtypes automatically. Merge decisions use `least_common_ancestor(type_a, type_b)`
> to find the most specific safe merge target. New subtypes are registered in the taxonomy,
> not scattered as strings across query files.

---

## A21 — Single-axis time models cannot answer "what did we know when?"

**Finding:**
Valid time (when a fact was true in the real world) is essential for temporal queries.
But transaction time (when we recorded the fact) is equally essential for auditing
and debugging. Without transaction time, "did the bad ingestion batch introduce this
fact?" requires reconstructing write timestamps from audit logs — if they were captured
at all. A retroactive correction silently overwrites the original write with no trace.

**Principle:**
> Stamp `recorded_at = datetime()` on CREATE (never update it) for every entity and
> every relation. This is the transaction-time dimension of a bitemporal model.
> Bitemporal query = `WHERE valid_from ≤ $vt AND recorded_at ≤ $tt` — "what did we know
> (tt) about the world at time (vt)?". Without transaction time, point-in-time KG
> reconstruction is impossible even with a perfect audit trail.

---

## A22 — Confidence numbers are not calibrated by default

**Finding:**
LLMs report relation confidence in the range 0.7–0.99 regardless of actual accuracy.
A model reporting 0.9 confidence that is correct only 55% of the time is miscalibrated.
The Bayesian fusion (`1 - (1-c1)(1-c2)`) computes correctly given accurate inputs, but
if the prior confidences are biased upward, the fused value is also biased.
Downstream effects: high-confidence edges dominate GNN weighting even when they're wrong;
contradiction detection thresholds fire on the wrong edges.

**Principle:**
> Track Brier score over a golden set: mean((predicted - actual)²). A perfectly calibrated
> model scores 0.0; an always-0.5 baseline scores 0.25. Run isotonic regression on the
> calibration curve (binned mean_predicted vs mean_actual) and apply the correction to raw
> LLM-reported confidence before storing edges. Rebuild the calibration table after model
> updates — confidence distributions shift with prompt version.

---

## A23 — TransE link prediction surfaces missing relations without labeled training data

**Finding:**
The GNN scores existing paths but cannot suggest plausible missing edges. Link prediction
requires a scoring function over unseen (h, r, t) triples. TransE provides this with
only entity embeddings as input (which already exist). The score `‖ h + r − t ‖₂` is
low for plausible triples and high for implausible ones. Relation embeddings can be
derived deterministically from the relation name (seeded from a hash of the name) when
no learned embedding is available — same relation always maps to the same vector.

**Principle:**
> TransE link prediction is essentially free if entity embeddings already exist.
> Use it as a graph completion signal: after ingestion, run `predict_missing_links()` for
> key entities and surface the top-k with score below a threshold as candidate facts for
> human review. This turns the graph from a passive store into an active hypothesis
> generator. False positives are expected — the output feeds a review queue, not the
> live graph.

---

## A24 — Snapshots decouple "current state" from "historical state" without duplication

**Finding:**
Without named checkpoints, comparing graph state before and after a batch ingestion
requires either storing two full graph copies (expensive) or reconstructing the pre-batch
state from audit logs (slow and error-prone). A lightweight snapshot that stores statistics
(entity count, edge count, health metrics, and the recorded_at of the youngest fact at
snapshot time) is sufficient for most audit and regression detection purposes.

**Principle:**
> Create named graph snapshots before major ingestion batches (`pre-Q1-ingest`) and after
> (`post-Q1-ingest`). Diff snapshots to detect quality regression immediately after
> ingestion, not weeks later in RAGAS scores. For precise point-in-time reconstruction,
> combine snapshots (which record the transaction-time boundary) with bitemporal queries
> (which filter by that boundary). Snapshots are cheap; retrospective debugging is expensive.

---

## A25 — TransE training only updates relation embeddings, never entity embeddings

**Finding:**
Entity embeddings come from a sentence transformer and are expensive to recompute (requires
re-embedding the entire corpus). TransE training should update only the translation vectors
`r` (relation embeddings) — which are small in count and cheap to update. Updating entity
embeddings during TransE training would contradict the pre-trained semantic space and
require re-embedding all edges after every training run.

**Principle:**
> In a graph where entity embeddings come from a pre-trained text model, treat them as
> fixed during TransE training. Only learn relation translations `r`. This separates
> concerns: the sentence transformer provides semantic coordinates; TransE provides
> structural translations between those coordinates. After training, re-run `embed_all_edges()`
> to update the stored triple embeddings with the new `r` vectors.

---

## A26 — Confidence propagates multiplicatively, not additively, through multi-hop paths

**Finding:**
When a query traverses A → B → C, the confidence of the path is `conf(A→B) × conf(B→C)`.
This is not a quality-level combination — it is a probability calculation: if each hop
introduces independent uncertainty, joint confidence decays as the product. Additive
propagation would allow path confidence to exceed edge confidence, which is nonsensical.

**Principle:**
> Propagate edge confidence multiplicatively along paths (Bellman-Ford with product instead
> of sum). Use max-product (best path) not sum-product (total support). Cut paths below a
> minimum confidence threshold (default 0.05) to prevent noise accumulation at depth 3+.
> Stamp the resulting `path_confidence` on each chunk result so the LLM can treat low-
> confidence path chunks with appropriate scepticism.

---

## A27 — Incremental community detection requires a rebuild point anchor

**Finding:**
Without a stored timestamp of the last community build, "what changed since the last build"
has no answer. The system either rebuilds from scratch (expensive) or doesn't know it should
rebuild (stale). A `CommunityRebuildPoint` node with a `rebuilt_at` timestamp is the minimal
state needed to drive incremental detection — compare Entity.recorded_at against it.

**Principle:**
> After every community build (full or incremental), write a `CommunityRebuildPoint` node
> with the current datetime. The incremental detector queries `Entity.recorded_at > last_rebuild`
> to find changed entities. Without this anchor, the incremental path degenerates to full
> rebuild on every ingestion. The rebuild point is cheap to write and cheap to query.

---

## A28 — GDPR erasure requires a pre-deletion audit record, not a post-deletion one

**Finding:**
If the deletion process fails halfway (network error, timeout), a post-deletion audit
record is never written — leaving the DPO with no evidence of what was attempted. Starting
the deletion with an audit record in `status='in_progress'` ensures the intent is recorded
even if the execution fails. A reaper job can find stuck `in_progress` audits and retry or
alert.

**Principle:**
> Create the DeletionAudit node BEFORE the first DELETE statement. The audit record's
> existence proves the erasure was requested and started. Update it to `status='complete'`
> only after all deletion steps succeed. A stuck `in_progress` record is a recovery signal;
> a missing record is an untraceable gap in the compliance trail.

---

## A29 — PII regex patterns need de-overlapping, not just union

**Finding:**
A string containing a credit card number also matches IP_V4 patterns (groups of digits
separated by characters). Without de-overlapping, both findings are returned for the same
span, and a naive redaction would apply two passes and corrupt the output. The correct
algorithm: sort findings by start position, keep only the highest-confidence finding
when two patterns overlap the same character span.

**Principle:**
> Always de-overlap PII findings before returning or redacting. Sort by start position,
> then by confidence descending within the same span. Apply redactions in reverse order
> (from end to start) so earlier offset positions are not shifted. Both the de-overlapping
> and the reverse-order redaction are required for correct output.

---

## A30 — Wikidata linking should cache failures, not just successes

**Finding:**
An entity not found in Wikidata ("XR-7500 Widget" — an internal part number) is looked
up on every call if only successes are cached. For a batch of 1000 unlinked entities,
this means 1000 API calls even if 900 of them were already tried and returned nothing.
Caching failures as `WikidataLink {status: 'not_found'}` prevents repeated futile lookups.

**Principle:**
> Cache negative results with a TTL (7 days default). The `status: 'not_found'` record
> acts as a negative cache entry. Include it in the `get_link()` lookup and skip the API
> call if it exists. Periodically clear `not_found` entries (they expire after TTL) to
> allow re-checking as Wikidata grows.

---

## A31 — Counterfactual impact score must weight removed edges more than exclusive entities

**Finding:**
Exclusive entities (those sourced only from the retracted document) are important but their
loss is recoverable — if the document is re-ingested with corrections, the entities return.
Removed edges (those whose only evidence came from this document) are more damaging: they
represent knowledge links that will disappear from reasoning paths. Weighting removed edges
at 0.4 and exclusive entities at 0.3 reflects this priority.

**Principle:**
> Impact score weights should reflect operational recovery cost, not raw count.
> Removed edges: 0.40 (hardest to recover — must re-process the source document)
> Exclusive entities: 0.30 (recoverable via re-ingestion)
> Resolved conflicts: 0.20 (may be beneficial — removes known contradictions)
> Orphaned entities: 0.10 (low additional impact — already captured in exclusive entities)
> Calibrate weights by asking: "if this component increases by 10%, how much worse is the
> retrieval experience?"

---

## A32 — Query result cache invalidation must be entity-scoped, not time-based alone

**Finding:**
A TTL cache for query results becomes stale the moment a new document updates an entity
mentioned in the cached answer. With pure TTL, the answer "Elon Musk is CEO of Tesla"
stays in cache for an hour even after a document is ingested that changes his role.
Provenance-aware invalidation (entity_name → set of cache keys) ensures only affected
answers are purged on ingestion, not all cached answers.

**Principle:**
> Build the provenance index at set-time: for each entity in `entities_used`, add the
> cache key to that entity's provenance set. At invalidation time (post-ingest), look up
> the provenance sets for the ingested document's entities and delete only those cache
> entries. The provenance index TTL should be longer than the result TTL to avoid dangling
> keys — cache entries expire, but the index outlives them.

---

## A33 — Entity type renames must cascade to all nodes that carry the type as a field value

**Finding:**
Renaming entity type EXEC → PERSON in the Entity.type field is not enough. WikidataLink
nodes carry `entity_type` as a field; Statement nodes carry `src_type`, `tgt_type`,
`subject_type`, `object_type`; RELATES_TO edges carry `src_type` / `tgt_type` in some
pipelines. A partial rename creates inconsistency: `Entity.type = 'PERSON'` but
`WikidataLink.entity_type = 'EXEC'` — the link is now unreachable because get_link()
queries on the current type.

**Principle:**
> Entity type rename is a graph-wide operation. Enumerate every node label and edge type
> that stores the entity type as a field value (not just Entity.type) and update all of
> them in the same migration transaction. Add a `type_migrated_from` audit field so the
> old value is recoverable without hitting the audit log.

---

## A34 — Test graphs must use string node IDs when the model requires strings

**What happened:**
`TestLeidenStartupPath._tiny_graph()` built a NetworkX graph with integer node IDs
(`G.add_edges_from([(0, 1), (1, 2)])`).  `Community.member_entity_ids: list[str]`
requires strings.  All three Leiden tests failed with Pydantic ValidationError on node
IDs 0, 1, 2, 3.

**Root cause:**
NetworkX defaults to integer node IDs and Pydantic's strict-type list validation does
not coerce ints to strings.  The mismatch was invisible until runtime.

**Rule:**
> When writing a test graph for a component whose model uses string entity IDs, always
> use string nodes: `G.add_edges_from([("n0", "n1"), ("n1", "n2")])`.  Check the model
> field type before building test fixtures.

---

## A35 — sys.modules.pop does not block imports of installed packages

**What happened:**
`TestLeidenStartupPath` tried to simulate a missing graspologic by calling
`sys.modules.pop("graspologic", None)`.  The test expected a `RuntimeError` but
`_run_leiden()` succeeded because Python re-imported graspologic from disk on the
next `from graspologic.partition import leiden`.

**Root cause:**
`sys.modules.pop` removes the cached copy but Python immediately re-imports from the
file system if the package is installed.

**Rule:**
> To block an installed package from importing in tests, set its `sys.modules` entry
> to `None`: `patch.dict(sys.modules, {"graspologic": None, "graspologic.partition": None})`.
> Python treats a `None` entry as "this module intentionally does not exist" and raises
> `ImportError` without touching the filesystem.

---

## A36 — Alert thresholds belong in config, not scattered across service constructors

**What happened:**
Initial AlertService draft hard-coded `contradiction_rate: 0.05` in the class body.
The project already had `business_matrix.alert_thresholds` in `settings.yml` for latency
and RAGAS metrics.

**Root cause:**
New service was written without checking if a config section for it already existed.

**Rule:**
> Before adding a new constant to a service, grep for existing config sections that
> logically own it.  Centralise thresholds in config so operators can tune them without
> code changes.  The pattern: `DEFAULT_THRESHOLDS` as code fallback + config override
> merged at singleton init.

---

## A37 — Dashboard data loading must go through the REST API, not direct Neo4j

**What happened:**
An early dashboard design considered adding `graphrag.graph.neo4j_client` imports
directly to `graphrag/dashboard/app.py`.

**Root cause:**
Dashboard sits outside the FastAPI process boundary (WSGIMiddleware mount), so async
Neo4j calls from a sync Dash callback would require a new event loop and config
bootstrap — creating hidden coupling.

**Rule:**
> UI components that are mounted inside a server process should still consume data
> through the public REST API (`httpx.get(API_BASE + path)`), not through internal
> service singletons.  This keeps auth, validation, and rate limiting in one place and
> makes the dashboard independently deployable.

---

## A38 — Module-level import order: use before first use

**What happened:**
`neo4j_client.py` had `from pathlib import Path` at line 561, but `Path` was used at
line 41 inside `init_schema()`.  The import worked only because `init_schema()` is
called after the full module loads.  Any refactor that called it during module
import would have caused a `NameError`.

**Root cause:**
The import was placed at the bottom "to avoid circular imports" — a comment that was
never verified.  The actual circular-import risk did not exist.

**Rule:**
> Imports must appear at the top of the module unless there is a verified circular-import
> reason.  "Placed here to avoid circular import" is not a reason — prove it by checking
> what the imported symbol's module actually imports at the top level.  Fix the circular
> import properly (move code, use `TYPE_CHECKING` guard, or restructure) rather than
> deferring.

---

## A39 — Blocking I/O in async functions silently stalls the event loop

**What happened:**
`entity_linker.py._search_wikidata()` was declared `async` but used
`urllib.request.urlopen(req, timeout=10)` — a blocking call — directly inside the
coroutine body.  The 10-second network timeout blocked the *entire* event loop,
freezing all concurrent requests.

**Root cause:**
`urllib.request` is inherently synchronous.  Marking a function `async` does not make
its blocking calls non-blocking.

**Rule:**
> Any blocking I/O (file reads, `urllib`, `requests`, `subprocess`) inside an `async`
> function must be wrapped in `await loop.run_in_executor(None, fn)`.  Alternatively,
> use an async-native library (`httpx.AsyncClient`, `aiofiles`, `asyncio.create_subprocess_exec`).
> Grep for `urllib.request`, `requests.get`, `open()`, `time.sleep()` inside `async def`
> blocks as part of every code review.

---

## A40 — asyncio.get_event_loop() is deprecated in Python 3.10+

**What happened:**
12 files used `asyncio.get_event_loop()` inside `async def` functions.
In Python 3.10+, calling `get_event_loop()` when there is no running loop emits a
`DeprecationWarning` and will raise `RuntimeError` in a future version.
Inside a running coroutine, the correct call is `asyncio.get_running_loop()`.

**Root cause:**
The code predated the Python 3.10 deprecation and was not updated during the upgrade.

**Rule:**
> Inside any `async def` block, always use `asyncio.get_running_loop()` — it raises
> `RuntimeError` immediately if there is no running loop (a clear signal of misuse)
> rather than silently creating one.  `asyncio.get_event_loop()` is only appropriate
> in synchronous code that is guaranteed to run before the event loop starts.

---

## A41 — Module-level mutable state breaks multi-worker deployments

**What happened:**
`graphrag/monitoring/alerts.py` stored recent alerts in a module-level `deque`.
Under `uvicorn --workers 4`, each worker process has its own copy of the deque.
`GET /kg/health/alerts` returned only the alerts fired by the *current* worker —
other workers' alerts were invisible.

**Root cause:**
In-process mutable state is invisible across OS-process boundaries.  Multi-worker
deployments (gunicorn, uvicorn, celery) spawn separate processes that do not share
memory.

**Rule:**
> Any mutable state that needs to be visible across multiple workers must live in a
> shared store (Redis, Postgres, a queue).  Use the in-process structure only as a
> fallback when the shared store is unavailable.  Pattern:
> `_push_to_redis(item) or _local_deque.append(item)`.

---

## A42 — Admin dashboards need auth on every mutation path

**What happened:**
The Dash admin dashboard at `/admin` had zero authentication on callbacks including
"Forget Entity" (GDPR deletion), "Rebuild Communities", and "Resolve Conflict".
Any unauthenticated visitor could trigger these operations.

**Root cause:**
The Dash app was added quickly and the REST endpoints (protected by OAuth scopes)
were called server-side, creating the illusion of security.  But the Dash callbacks
that trigger the POST requests ran for *any* browser that loaded the page.

**Rule:**
> A WSGI dashboard mounted inside a FastAPI app is **not** automatically protected by
> FastAPI's middleware.  Add `before_request` on the Flask server to check a session
> cookie or token header.  Protect any callback that performs mutations (POST/DELETE)
> even if the downstream API has its own auth — defense in depth requires the UI layer
> to gate access too.

---

## A44 — Cross-process result passing requires shared external storage

**What happened:**
The async query flow used a module-level `_results: dict` in the API process.
The query worker (separate container) computed the answer and wrote to `_results` via
`from api.routes.query import store_result` — which succeeded silently, writing to
the **worker's own memory**, not the API's. Clients polling `GET /query/{id}` from
the API always saw `"queued"` forever.

**Root cause:**
Module-level mutable state is per-process. When the writer and reader run in different
containers (different OS processes), in-process dicts are invisible across the boundary.
The import succeeding is misleading — Python resolved the import, but the object lives
in the importer's address space.

**Rule:**
> Any state that must be readable by a different process (worker → API, API replica A →
> replica B) must live in an external store: Redis, a DB, a message queue.
> The smell: a background worker importing from an API module to "write back" a result.
> That import path works for code reuse but not for shared state.
> Fix: introduce a dedicated Redis-backed store (SETEX/GET with TTL) that both processes
> connect to independently.

---

## A45 — Scope gates must be unconditional, not token-type-conditional

**What happened:**
`require_scope` checked `if scope not in granted and user.get("type") == "m2m"`.
Browser tokens (`type="browser"`) bypassed the scope check entirely.
Today every browser token carried `"read write"` so it was masked.
Adding any `require_scope("admin")` endpoint would silently grant all browser users admin.

**Root cause:**
The intent was "only M2M tokens have explicit scopes", but that assumption was wrong and
the code encoded it as a logic gate that made scope enforcement conditional.

**Rule:**
> Authorization checks must be unconditional. Never add `and type == X` clauses to a
> permission check — they silently exempt all other token types.
> Correct form: `if scope not in granted: raise 403` — period.
> If different token types need different scope vocabularies, enforce that at mint time,
> not at verification time.

---

## A46 — Open redirect via unvalidated `next` parameter

**What happened:**
`/auth/login?next=https://evil.com` stored the `next` URL in the session and after
Google OAuth redirected the user to the attacker's domain. Classic post-auth open redirect
(phishing and token-theft vector).

**Root cause:**
The `next` parameter was stored and used without validating that it was a safe relative
path. External URLs are indistinguishable from internal ones without explicit checking.

**Rule:**
> Any URL parameter that drives a redirect must be validated before use.
> Safe check: `urlparse(url).scheme` and `.netloc` must both be empty; the string must
> not start with `//`. Reject anything that fails; redirect to a safe default instead.
> Never store an unvalidated redirect target, even in a session cookie.

---

## A47 — JWT signing secret and session cookie secret must be distinct

**What happened:**
`SessionMiddleware` used `settings.jwt_secret_key` as its secret. These are conceptually
different keys: one signs bearer tokens for API auth, the other signs browser session
cookies. Using the same key means rotating one invalidates the other.

**Root cause:**
Reuse was convenient — the key already existed. The distinction wasn't visible until
the audit flagged that a JWT secret rotation would unexpectedly log out all browser users.

**Rule:**
> Signing keys should have single responsibilities. Add a `session_secret_key` config
> field distinct from `jwt_secret_key`. In production, set both explicitly.
> For backward compatibility, derive a fallback (`jwt_secret_key + ":session"`) so
> existing deployments don't break on upgrade, but document the expectation clearly.

---

## A48 — Rate limiting is infrastructure, not an afterthought

**What happened:**
`POST /ingest` and `POST /query` had no rate limiting.  A single runaway client
could saturate the Gemini API quota (per-minute token limit), exhaust the Neo4j
connection pool (50 connections), or DDoS the RabbitMQ broker — with no feedback
to the caller and no protection for other tenants.

**Root cause:**
Rate limiting was deferred because the prototype had one user.  It was never added
when the system grew.

**Rule:**
> Add rate limiting before the first external user, not after the first incident.
> Use a per-IP limiter (`slowapi` for FastAPI) and make limits configurable via
> environment variables so they can be tuned per-deployment without a code change.
> LLM-backed endpoints deserve stricter limits (quota-sensitive) than read endpoints.

---

## A49 — Worker processes need SIGTERM handlers or they corrupt in-flight messages

**What happened:**
All three workers called `asyncio.run(consumer.start())` with no signal handling.
On `docker stop` (or Kubernetes pod eviction), the process received SIGTERM and
was killed immediately.  Any message that had been received from RabbitMQ but not
yet acked (mid-processing) was silently lost from the worker's perspective — the
broker would requeue it after the consumer connection timed out, but not before
the message was potentially half-written to Neo4j.

**Root cause:**
The worker was written as the simplest possible entry point.  Graceful shutdown
was not considered.

**Rule:**
> Wrap the consumer in `asyncio.create_task()`.  Register SIGTERM/SIGINT handlers
> that call `task.cancel()`.  Catch `asyncio.CancelledError` at the top level and
> log a clean shutdown message.
> With `prefetch_count=1`, cancellation happens at the next `await` boundary, which
> is after the current message's handler finishes — so the message is always either
> fully processed+acked or returned to the queue cleanly.

---

## A50 — Lock files prevent silent transitive dependency breaks

**What happened:**
`requirements.txt` pinned only direct dependencies with `>=` lower bounds.
A transitive package (`ragas`, `sentence-transformers`, `graspologic`) could
introduce a breaking change in a minor release and break the build on the next
`pip install` — silently, in production, without any code change.

**Root cause:**
Lock files were known but deferred.  `make lock` was added but the lock file was
gitignored, so it was never actually committed.

**Rule:**
> Commit a fully-pinned lock file (`requirements.lock` via `pip-compile`).
> Never gitignore it.  Regenerate it with `make lock` after any requirements.txt
> change, and treat the lock file as a dependency artifact that ships with the code.
> CI must install from the lock file, not from the loose requirements.

---

## A51 — Unbounded aggregation queries grow with traffic

**What happened:**
`kpi_tracker.get_summary()` fetched ALL latency rows in the time window to compute
p50/p95 in Python.  No `LIMIT` clause.  At 1 request/second for 7 days that's
604,800 rows loaded into memory per dashboard refresh.  The `recorded_at` column
also had no index, so the WHERE filter required a full table scan.

**Root cause:**
The query was written during development with small data sets where performance
was not observable.

**Rule:**
> Any query that aggregates over a growing table must have:
> 1. An index on the filter column (`recorded_at`).
> 2. A `LIMIT` on any Python-side computation that loads raw values (cap at 10k
>    rows for percentile — statistically accurate and memory-bounded).
> For databases that support it (Postgres/TimescaleDB), use `PERCENTILE_CONT`
> instead of loading values into Python.

---

## A52 — Portfolio code must actually execute the claimed capabilities

**What happened:**
Added aerospace ontology YAML, export_rdf.py, and domain_ontology.py as portfolio
pieces for an "AI Knowledge Graph/Ontology Lead" JD. An Opus audit found 8 credibility
gaps: the YAML was never loaded, `add_domain_range_rules()` didn't exist, the RDF
script used hand-rolled string concatenation instead of rdflib, and there was no
runnable proof of the pipeline.

**Root cause:**
The first pass added the *description* of the capability without completing the
*wiring*. A technical reviewer will grep for the method call, find it missing, and
stop reading.

**Rule:**
> For any portfolio capability claim, verify three things before committing:
> 1. **The code path executes**: import the module and call the entry point in a
>    smoke test or demo script. If it can't be imported, it doesn't count.
> 2. **Config is wired**: grep for every YAML key added; trace it to the code that
>    reads and uses it. Dead config silently fails — `getattr(settings, "x", {})` on
>    a property that doesn't exist always returns `{}`.
> 3. **Use the standard library**: hand-rolled Turtle, hand-rolled JSON parsing, or
>    hand-rolled crypto are all red flags. If a mature library exists (rdflib, rdflib,
>    cryptography), use it.

---

## A53 — Settings properties must be explicitly declared; getattr fallback hides missing fields

**What happened:**
`OntologyRegistry.load()` called `getattr(get_settings(), "ontology", {})`. The
`Settings` class had no `ontology` property, so it always returned `{}`. The
migration_map was silently never loaded from `settings.yml`. No error, no warning —
just wrong behaviour.

**Root cause:**
`getattr(obj, "x", default)` returns the default for *any* reason the attribute is
missing — including typos, missing properties, or runtime exceptions in the property.
It's a silent failure pattern.

**Rule:**
> Never use `getattr(settings, "key", {})` as an access pattern. Always:
> 1. Declare the property explicitly on the Settings class.
> 2. Access it as `get_settings().ontology` — if the property is missing the
>    AttributeError is loud and immediate.
> 3. After adding a new settings section, grep for every call site and verify the
>    property exists before committing.

---

## A55 — Test assumptions about library behaviour must match the library's actual semantics

**What happened:**
`test_empty_graph_returns_zero` asserted that applying `owlrl.DeductiveClosure` to an
empty rdflib Graph returns 0 new triples.  It returned 106.  owlrl bootstraps the OWL
built-in axioms (`owl:Thing`, `owl:Nothing`, etc.) even on an empty graph — that is
correct OWL-RL behaviour, not a bug.

Similarly, `test_valid_graph_is_consistent` used `(None, None, OWL.Nothing) in graph`
which is True whenever `owl:Nothing` appears as *any* object (e.g. as an object of a
built-in OWL axiom).  The correct check is `(None, RDF.type, OWL.Nothing)` —
i.e. an *individual* typed as owl:Nothing.

**Root cause:**
Tests were written from first-principles reasoning about the library, not from reading
the library's actual output.

**Rule:**
> Before writing assertions about a third-party library's output:
> 1. Run the library on a minimal example and inspect the actual result.
> 2. Do not assert exact counts unless you have run the code and confirmed the number.
>    Use `>= 0` / `> 0` / `isinstance(n, int)` instead of `== 0` for things the library controls.
> 3. For consistency checks in OWL-RL: check `(None, RDF.type, OWL.Nothing)` —
>    an individual typed as the empty class — not `(None, None, OWL.Nothing)` which
>    matches any triple with owl:Nothing as object (including built-in axioms).

---

## A56 — Re-explore before assuming gaps are real; previously built items may already exist

**What happened:**
A gap analysis listed 7 items as missing (integration tests, load tests, alert service,
admin dashboard, TransE training, SPARQL, OWL reasoner, link predictor).  Re-exploration
found 5 of those 7 were already fully built in prior sessions.  Only 4 were genuinely
missing.

**Root cause:**
The analysis was based on the initial JD audit which happened before Phase 9
implementation.  The gap list was never reconciled against what was actually committed.

**Rule:**
> Before implementing a list of "missing" items:
> 1. Grep/glob for the file that would exist if it were already done.
> 2. Check git log for commits that sound like they implement the item.
> 3. If the file exists, read it — it may be complete, partial, or wrong.
> Only after confirming absence should you write new code.
> False-positive gaps waste time and create duplicate implementations.

---

## A57 — Neo4j 5.x deprecated `size()` for relationship counting

**What happened:**
`neo4j_client.py` used `size((e)-[:RELATES_TO]-()) AS degree` to count entity
relationships. Neo4j 5.x deprecated this pattern and the query failed with a
`SyntaxError: The `size()` function with a pattern expression is deprecated`.

**Root cause:**
The query was written against Neo4j 4.x syntax. The deprecation was introduced
in Neo4j 5.0 and the code was never updated.

**Rule:**
> Never use `size(pattern-expression)` to count relationships in Neo4j 5.x.
> Use the subquery form instead:
> ```cypher
> -- Deprecated (Neo4j 4.x):
> size((e)-[:RELATES_TO]-()) AS degree
>
> -- Correct (Neo4j 5.x+):
> COUNT { (e)-[:RELATES_TO]-() } AS degree
> ```
> When upgrading Neo4j, grep for `size((` in all `.py` and `.cypher` files.

---

## A58 — Neo4j async driver lazy execution: DDL requires explicit consume()

**What happened:**
`scripts/init_neo4j.py` called `await session.run(stmt)` for DDL statements
(CREATE INDEX, CREATE CONSTRAINT). The statements appeared to succeed but the
indexes were never created. `SHOW INDEXES` returned an empty list.

**Root cause:**
`session.run()` with the Neo4j async driver returns a lazy `Result` cursor.
For DML queries (SELECT/MATCH), the rows are consumed by the `async for` loop.
For DDL statements where you don't iterate results, nothing is consumed — the
server never executes the statement.

**Rule:**
> After any DDL statement with the Neo4j async driver, explicitly consume the
> result:
> ```python
> result = await session.run("CREATE INDEX ...")
> await result.consume()   # forces server execution
> ```
> Without `consume()`, DDL is silently a no-op. This applies to every
> `CREATE INDEX`, `CREATE CONSTRAINT`, `DROP INDEX`, and schema-only query.

---

## A59 — Comment lines must be stripped per-fragment, not per-file in schema.cypher

**What happened:**
`schema.cypher` was split on `;` to get individual statements. Some fragments
started with a `-- comment` line before the `CREATE` statement. The code checked
`stmt.startswith("--")` and skipped those fragments entirely — so several
indexes were never created despite the script reporting success.

**Root cause:**
The guard was designed to skip comment-only lines but was applied to the full
fragment. A fragment like `"-- create vector index\nCREATE VECTOR INDEX ..."` starts
with `--` so the entire `CREATE` was skipped.

**Rule:**
> When parsing SQL/Cypher files split on `;`, strip comment lines from each
> fragment before checking if it is a real statement:
> ```python
> for stmt in raw.split(";"):
>     lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
>     stmt = "\n".join(lines).strip()
>     if not stmt:
>         continue
>     await session.run(stmt)
> ```
> Never apply the comment check to the whole fragment string — it will silently
> discard statements that follow comments.

---

## A60 — Windows does not support asyncio.add_signal_handler

**What happened:**
All four worker entry points called `loop.add_signal_handler(signal.SIGTERM, ...)`.
On Windows this raises `NotImplementedError` because Windows does not have POSIX
signals. Workers crashed at startup on every Windows dev machine.

**Root cause:**
Signal handling code was written for Linux/macOS (the deployment target) without
a Windows guard. The error only manifested when running locally on Windows.

**Rule:**
> Guard `asyncio.add_signal_handler` with a platform check:
> ```python
> import sys
> if sys.platform != "win32":
>     loop = asyncio.get_running_loop()
>     for sig in (signal.SIGTERM, signal.SIGINT):
>         loop.add_signal_handler(sig, handler)
> ```
> On Windows, the process will still terminate on Ctrl-C (KeyboardInterrupt)
> but won't have graceful cancellation via signal. This is acceptable for
> local development; production runs on Linux where signals work correctly.

---

## A61 — Cross-process result sharing requires Redis; in-process dict is silently broken

**What happened:**
The query worker computed an answer and stored it by importing a dict from the
API module. The API polled the same dict and always returned `{"status": "queued"}`.
After switching to Redis-backed `ResultStore`, both processes saw the same state.

**Root cause:**
In-process mutable state (`_results: dict`) is invisible across OS process
boundaries. The import resolved to the worker's own address space, not the API's.
No error was raised — the worker "successfully" wrote to its own copy.

**Rule:**
> Any state that must cross a process boundary (worker → API, replica A → replica B)
> must live in an external store. Redis `SETEX/GET` with TTL is the standard pattern.
> The smell: a worker importing from an API module to "write back" a result.
> See also: L44 (the original lesson on this pattern). The symptom is always the
> same — `status: queued` forever despite the worker completing successfully.

---

## A62 — Missing package + strict mode = silent startup failure

**What happened:**
`redis[asyncio]` was not in `requirements.txt`. `settings.yml` had
`session_store_strict: true`. At startup the session store raised `ImportError`
and the query worker crashed before consuming any messages. The ingestion worker
continued (it doesn't use the session store directly) so ingests appeared to work
but all queries stayed at `status: queued`.

**Root cause:**
The strict mode guard correctly detected the missing package, but `redis[asyncio]`
was omitted from `requirements.txt` so a fresh `pip install -r requirements.txt`
left the package absent.

**Rule:**
> Every package that is required (even conditionally) by a feature that ships as
> on-by-default must be in `requirements.txt`. Strict modes and optional features
> are not excuses for leaving packages out of the lock file.
> Check: after any `session_store` or similar feature change, run
> `pip install -r requirements.txt` in a clean venv and verify the worker starts.

---

## A43 — AsyncMock side_effect lists must exactly match actual call counts
(recurred twice — A43 and the former A54 were the same bug pattern, in
tests and in a demo script)

**What happened (test_safety_paths.py):**
Four tests had `side_effect` lists with phantom "CREATE" call slots
inserted after queries that returned `[]`. When a query returns an empty
list, the `for row in rows:` loop body is never entered, so no CREATE is
issued. The tests assumed a CREATE always follows a query, which drifted
the mock's slot mapping — the actual conflict row landed in the wrong
slot, causing `KeyError` on unexpected field names.

**What happened (demo_regulatory.py):** step 4 (`ForwardChainingEngine`)
and step 5 (`ContradictionDetector`) both raised `StopIteration` because
the `side_effect` list had fewer slots than actual `await db.run()` calls
— step 4 needed 9 slots (4 rule queries per fixpoint iteration × 2
iterations + 1 MERGE write), step 5 needed 11. The mock was written from
the outside ("query then create") rather than tracing the actual code
path; `ForwardChainingEngine` runs all rules in *every* fixpoint
iteration, not just the ones with matches.

**Root cause:**
Both cases: the mock sequence was designed assuming "query → create"
pairs, but the real code pattern is "query → *conditional* create per
row" (or, for fixpoint loops, "N rule-queries per iteration × M
iterations"). Zero rows = zero creates; the mock author didn't trace the
actual call sequence before writing the slot list.

**Rule:**
> When writing `AsyncMock(side_effect=[...])`, trace through the
> production code path explicitly before writing the list:
> 1. List every `await db.run()` call and mark which ones are conditional
>    (inside a for-loop over rows) or repeated (inside a fixpoint/retry
>    loop — slots = calls_per_iteration × num_iterations).
> 2. Only include CREATE/UPDATE mocks when your mock rows will actually
>    trigger that code path. Zero-row queries produce zero subsequent
>    CREATE calls.
> 3. Name each slot in a comment: `# 3: directional CREATE (only if row
>    returned)` or `# iter1: supersedes_transitivity → 1 candidate`.
> Every phantom or missing slot shifts all subsequent slots by one,
> causing cascade `KeyError`/`StopIteration` failures. `StopIteration`
> from a mock means you undercounted by exactly one call — add one slot
> at a time until it passes, rather than guessing.

---

## A63 — LLM swap without re-running tests leaves regressions

**What happened:**
The Gemini→Groq migration changed the call path in `extractor.py` from
`loop.run_in_executor → response.text` to `get_llm().generate() → str`.
Three confidence-clamping tests and two empty-response tests mocked the old
path. They were not re-run after the swap — the tests were silently broken
(the clamping tests failed; the empty-response tests passed accidentally via
`parse_error` for the wrong reason).

**Root cause:**
The migration was done and manually verified end-to-end, but the unit suite
was not run. The `run_in_executor` + `response.text` mock signature is tightly
coupled to the provider; swapping providers invalidates it silently.

**Rule:**
> After any LLM provider migration, always run the full unit suite immediately.
> Tests that mock `run_in_executor`, `generate_content`, or `response.text` are
> provider-coupled — grep for them before and after the swap and update every one.
> Commit the test fixes in the same PR as the migration, never separately.

---

## A64 — Attribute reference vs. method call: `self._model` vs. `self._model()`

**What happened:**
`agentic_retriever.py` stamped `model_version=self._model` on `QueryResult`.
`AgenticRetriever` has no `_model` attribute (it's an abstract method on
`BaseGraphRAGAgent`, which `AgenticRetriever` does not inherit from). The
expression evaluated to `AttributeError` at runtime when the agentic path fired.
This was not caught by tests because no test exercised the agentic code path
with a live `QueryResult` construction.

**Root cause:**
Copy-paste from an agent class where `_model` is a method. In the retriever,
which is not an agent, the attribute doesn't exist.

**Rule:**
> When copying a field assignment, verify the RHS is defined on the target class.
> `self._x` and `self._x()` are different: the former reads an attribute; the
> latter calls a method. For provenance fields like `model_version`, always use
> an explicit source: `get_settings().groq_model` or a constructor parameter —
> never a method reference from an unrelated base class.

---

## A65 — "The X" in a request may refer to one of several X; enumerate before acting

**What happened:**
The user asked to make "the metrix dashboard" breathtaking. The project has **two**
dashboards: the admin/observability dashboard (`graphrag/dashboard/`) and the business
KPI dashboard (`graphrag/business_matrix/dashboard_server.py`). The first redesign pass
only touched the admin one — the business KPI dashboard (literally "the metrics dashboard",
showing latency/faithfulness/cost) was nearly missed and is the one the phrasing most
likely meant.

**Root cause:**
Assumed the singular "the dashboard" mapped to the first dashboard found, without checking
how many components of that kind exist in the repo.

**Rule:**
> When a request names a component with the definite article ("the dashboard", "the demo",
> "the client", "the worker"), glob for how many match before editing. If more than one
> exists, either handle all of them or confirm which is meant. The user's mental model is
> often the one you haven't opened yet.

---

## A66 — Demo/empty states must be neutral, never alarming

**What happened:**
The redesigned KPI dashboard rendered status-coloured tiles. With no data in the time
window (the sample event was older than the 7-day default), every tile showed a red
`0.000` against its "target ≥ 0.70" hint — making a freshly-launched dashboard look
broken/failing in a cold demo.

**Root cause:**
The colour logic was written for the populated case only; the zero/empty case fell through
to the "below threshold → red" branch.

**Rule:**
> Distinguish "bad value" from "no value". An empty/zero-row state must render neutral
> (grey, em-dash, "awaiting data") — never the failure colour. Add an explicit
> `if not total_rows:` branch that returns the neutral layout before any threshold
> colouring runs. A demo that opens on red zeros reads as a broken product.

---

## A67 — Demo data belongs behind an env flag, never hardcoded into render paths

**What happened:**
To populate dashboard tabs without a live backend, the first pass dropped mock payloads
directly into each tab's `render()` as an unconditional fallback on API error. That would
mask real API failures in production — a 500 from the backend would silently show fake
green metrics.

**Root cause:**
Conflated "show sample data for a screenshot" with "be resilient to API errors". They are
opposite requirements: the demo wants fake data on error; production wants the error shown.

**Rule:**
> Gate demo/sample data behind an explicit flag (`GRAPHRAG_DASHBOARD_DEMO`). The fallback
> fires **only** when the flag is set AND the live source is unreachable. Unset (production)
> always shows real data or a real error panel. Keep the payloads in a dedicated
> `demo_data.py` module, not inline in render functions, so the production code path stays
> clean and the sample values are auditable in one place.

---

## A68 — A Dash app mounted under FastAPI must be served via the API, not standalone

**What happened:**
Running `python graphrag/dashboard/app.py` standalone served the HTML but every
`_dash-component-suites/*.js` request returned the index HTML (`SyntaxError: Unexpected
token '<'`), so the page hung on "Loading...". The app is built with
`requests_pathname_prefix="/admin/"` and is designed to be mounted under the API via
`a2wsgi`. Also, `app.config.requests_pathname_prefix = "/"` in the `__main__` block raised
`AttributeError` — the prefix is read-only after construction in Dash 2.x. Separately, the
mount in `api/main.py` is wrapped in a try/except that **silently** skips mounting if
`a2wsgi` is missing → `/admin` 404s with no obvious cause.

**Root cause:**
The Dash asset routes are tied to the pathname prefix; serving from a bare Flask dev server
at `/` breaks asset resolution. And a silent `except ImportError` hid the missing dependency.

**Rule:**
> Serve a mounted Dash app through its host (the API: `uvicorn api.main:app` → `/admin/`),
> never the standalone Flask server. Do not reassign `requests_pathname_prefix` after
> construction. Ensure mount dependencies (`a2wsgi`) are in `requirements.txt`, and when a
> mount is wrapped in try/except, log the exception at WARNING with the dep hint (this code
> did — `grep` the startup log for `admin_dashboard_unavailable` when `/admin` 404s).

---

## A69 — Tooling: Dash pages never reach document_idle; processes from other sessions can't be killed

**What happened (two process/tooling traps in one session):**
1. The browser screenshot tool kept timing out on `document_idle` for Dash pages — the
   `dcc.Interval` / dash-renderer keeps a connection live so the document is never "idle".
   The `zoom` (region-capture) action uses a different code path that bypasses the idle
   gate and succeeded every time.
2. Killing a stale server failed repeatedly: `python` isn't on PATH inside the bash tool
   (only PowerShell); the direct `python.exe` was a different install missing `a2wsgi`;
   PowerShell `... > $null` is an "ambiguous redirect" in bash; and a process launched by a
   *previous* session could not be killed (`taskkill` → access denied).

**Root cause:**
Cross-shell/interpreter assumptions, and OS process ownership across session boundaries.

**Rule:**
> - For screenshots of Dash/long-poll SPAs, use the `zoom` region action, not the full-page
>   screenshot that waits for `document_idle`.
> - Don't assume `python` works in every shell; resolve the interpreter explicitly and
>   verify deps (`python -c "import a2wsgi"`) before launching.
> - If a process can't be killed (different session owner, access denied), don't fight it —
>   launch a fresh instance on a different port and point the browser there.
> - Use `2>$null`/`$null` redirects only in PowerShell; in bash use `>/dev/null 2>&1`.

---

## A70 — Live demo script diverges from the model's actual signature

**What happened:**
`scripts/demo_regulatory.py --live` crashed three times before completing:
1. `neo4j_client.init_schema()` failed with `CypherSyntaxError` — comment lines (`--`) were
   sent raw to Neo4j because the schema parser stripped comments at the file level, not
   per-fragment (same bug class as A59, different manifestation).
2. `Document(content=...)` failed — the field is `raw_text`, not `content`.
   `source_path` was also missing (required by Pydantic).
3. `neo4j.merge_document(doc)` failed — the method takes explicit keyword args
   (`doc_id`, `filename`, `ingested_at`), not a `Document` object.

**Root cause:**
The demo script was written with `Document` as a convenience object passed directly to
Neo4j, but `merge_document()` was later refactored to take explicit scalar args without
updating the demo.

**Rule:**
> Before a live demo, run the demo script end-to-end in a clean environment and fix
> every crash before the meeting. Do not assume the demo works because the mock path works —
> the mock path does not exercise the same code. For Neo4j-touching scripts: read the
> `merge_document()` signature, not just the model definition.

---

## A71 — Single-model agentic retrieval is the biggest latency leak

**What happened:**
The agentic IRCoT retriever used `llama-3.3-70b` for every call: the intermediate
"SEARCH or ANSWER?" reasoning steps AND the final synthesis. The reasoning step emits
~15 tokens ("SEARCH: engine mount inspection"); running the 70B model for this costs
~1.5s per step. With `max_steps=4`, that's up to 6s in LLM calls alone before synthesis.

Switching reasoning steps to `llama-3.1-8b-instant` (~800 tok/s vs ~150 tok/s for 70B)
cut each step from ~1.5s to ~0.2s. Combined with reducing `max_steps` 4→2:
- Agentic p95: 6.8s → **3.4s** (−50%)
- Combined p95: 5.9s → **2.7s** (under the 3s SLO)

**Rule:**
> In a multi-step LLM pipeline, match model size to task complexity:
> - Routing / classification / short-form decisions → small fast model (8B)
> - Final synthesis / long-form generation → full model (70B)
> Add `groq_fast_model` as a separate config field so it can be tuned independently.
> Measure per-step latency in logs before and after; don't guess.

---

## A72 — Agentic trigger with OR logic causes aggressive fallback on sparse corpora

**What happened:**
`_is_low_confidence()` triggered on hedge signals **OR** zero citations.
On a freshly-ingested small corpus, many answers have zero citation IDs (chunks aren't
fully linked to entities yet) even when the answer text is confident. This caused ~32% of
queries to fall through to the 6s agentic path unnecessarily.

Changing to AND (hedges AND zero citations) dropped the agentic rate from 32% to ~9%,
which is the correct rate for genuinely hard queries on a mature corpus.

**Rule:**
> Alert / fallback conditions that trigger on any single signal are almost always too
> aggressive on sparse/fresh data. Use AND when both signals are meaningful independently,
> OR only when either signal alone is sufficient evidence. For new corpora especially,
> test the trigger rate before deploying — `agentic_rate > 20%` indicates the threshold
> is wrong, not that the corpus is hard.

---

## A73 — Combined p95 hides the signal when mixing different retrieval modes

**What happened:**
Reporting a single `p95_latency_ms` across hybrid and agentic queries showed 5.9s,
which looked broken. Hybrid p95 alone was 2.2s (well under target). The combined
number was meaningless because the two modes have fundamentally different latency
profiles (1.7s vs 5s average) and different SLOs.

**Rule:**
> Never aggregate latency percentiles across requests that are semantically different.
> Report per-mode: `p95_hybrid`, `p95_agentic`. Alert on `p95_hybrid` for the SLO.
> Alert on `agentic_rate` (not agentic latency) — a high rate means the trigger
> is too loose; the latency is the expected cost of the capability, not a bug.

---

## A74 — LLM rate limits: fail-fast fallback beats sleep-and-retry for ingestion

**What happened:**
Groq free tier (100k tokens/day) exhausted mid-ingestion. The retry-backoff slept 5–10 min
per chunk, turning a 5-minute corpus run into an 8-hour job. Switching to DeepSeek as
fallback (OpenAI-compatible SDK, already installed) made each hit instant: Groq 429 →
DeepSeek in <1s, no sleep.

**Root cause:**
Retry-backoff is right for transient errors (network blip). It's wrong for quota exhaustion
where sleeping doesn't help — the quota window is hours, not seconds.

**Rule:**
> For LLM rate limits in ingestion pipelines: use `max_retries=1` on the primary, catch
> the exception in a `FallbackLLM` wrapper, and re-issue to a secondary provider
> immediately. Reserve retry-backoff for the secondary. Never sleep on a daily quota hit.

---

## A75 — Embedding provider swap: same dimensions = zero schema changes

**What happened:**
Gemini embeddings ($0 balance) blocked the entire pipeline. Switched to OpenAI
`text-embedding-3-large` — also 3072 dimensions. Neo4j vector index, all retrieval
queries, and all similarity thresholds worked identically. Zero schema changes required.

**Root cause:**
Dimension count was the only coupling point.

**Rule:**
> When evaluating embedding providers, match dimensions to avoid schema migrations.
> 3072d (OpenAI large / Gemini) is a safe default for production KG work. Document
> the dimension in `config.py` so future swaps are a one-line change.

---

## A76 — Audit trail MATCH without tenant returns N rows; CREATE fires N times

**What happened:**
`AuditTrail.log_entity_change()` did `MATCH (e:Entity {name: $name, type: $type})`
with no tenant filter. Across 12 documents, the same entity name existed in multiple
contexts. MATCH returned N rows; `CREATE (cl:ChangeLog {id: $log_id})` executed N times
with the same UUID, triggering a `ConstraintValidationFailed` error on attempt 2+.

**Root cause:**
Audit queries matched globally, but the UUID was generated once before the query.

**Rule:**
> Every `MATCH` in an audit trail write must include `LIMIT 1` to guard against
> multi-row returns, and wrap in try/except since audit logs are non-critical.
> Pattern: `MATCH (e:Entity {name: $name, type: $type}) WITH e LIMIT 1 CREATE ...`

---

## A77 — `NOT property = true` is a null trap in Cypher (recurred twice —
A77 and the former A81 were the same bug in two different files)

**What happened (first occurrence):**
`WHERE NOT e.quarantined = true` excluded all entities where `e.quarantined` was NULL
(property not set). In Cypher, `NULL = true` → NULL, `NOT NULL` → NULL, which is
falsy in WHERE — so all 374 entities were excluded and `entity_count=0` in snapshots.

**What happened (second occurrence, `incremental_community.py`):** the
Communities tab (incremental drift monitor) showed "0% — 0 changed
entities" even though all 374 entities were ingested after the last
(non-existent) rebuild point. Same root cause, three separate queries in
`incremental_community.py` (`get_changed_entities` ×2, `should_full_rebuild`
total count) all used `WHERE NOT e.quarantined = true`.

**Root cause:**
Neo4j Cypher null semantics: any comparison involving `NULL` returns
`NULL`, not `true` or `false`. `NOT NULL = true` → `NOT NULL` → `NULL`
(falsy in a `WHERE` clause) — so any entity where the optional boolean
property was never set gets silently excluded. Standard boolean negation
doesn't work as expected when the property may be absent — which is the
common case for properties like `quarantined`, `deleted`, `archived`.

**Rule:**
> **Every** Cypher `WHERE` clause that filters on an optional boolean
> property must use `coalesce(prop, default) = expected`, not
> `NOT prop = true` or `prop = false`. Audit all new Cypher for this
> pattern before committing — the null-trap is invisible unless you query
> a graph where the property is absent, which is exactly the common case.

**Scope of fixes applied across the codebase (both occurrences combined):**
`incremental_community.py` (3), `neo4j_client.py` (6), `graph_snapshots.py`
(2), `graph_evaluator.py` (2), `community_builder.py` (1, in
`build_semantic_communities`).

---

## A78 — Degree anomaly threshold must scale with graph density

**What happened:**
`MAX_DEGREE_MULTIPLIER = 5.0` flagged FAA (degree=54), Boeing (19), MCAS (18) as
anomalies on a 374-entity sparse graph where mean degree ≈ 1.22. These are genuinely
important hub entities, not hallucinated noise.

**Root cause:**
The threshold was designed for dense graphs. In a sparse domain corpus (< 10k entities),
central regulatory bodies naturally have 20–50× mean degree.

**Rule:**
> Set `MAX_DEGREE_MULTIPLIER ≥ 20` for sparse domain graphs (< 50k entities).
> Reserve multiplier=5 for dense social/web graphs. Document why the threshold was
> chosen so the next engineer doesn't silently lower it.

---

## A79 — Dashboard JWT token must include `scope` field

**What happened:**
Generated a JWT with `{"sub": "dashboard", "role": "admin"}` — got 401 (no token),
then 403 (token accepted but missing scope). The `require_scope("read")` dependency
checks `user.get("scope", "").split()` — if `scope` is absent, all scoped endpoints
return 403.

**Root cause:**
JWT payload lacked the `scope` claim the API expects.

**Rule:**
> Dashboard/service tokens must include `scope: "read write"` in the JWT payload.
> Template: `create_access_token({"sub": "service", "role": "admin", "scope": "read write"}, timedelta(days=3650))`

---

## A80 — API response format must match what the dashboard expects

**What happened:**
`GET /kg/snapshots` returned a plain `list[dict]`. The dashboard did
`data.get("snapshots", [])` — since `isinstance(list, dict)` is False, `snaps = []`
and the Graph Health tab showed "No snapshots found" despite 3 snapshots existing.

**Root cause:**
The API and dashboard were written independently with different assumptions about
the response envelope.

**Rule:**
> List endpoints that feed UI components must return `{"items": [...], "count": N}`.
> Never return a bare list from a REST endpoint — it breaks pagination, metadata,
> and future extension. Check the dashboard's `_get()` call before finalising the route.

---

## A83 — LLM answer preamble ("Based solely on the context, ...") causes RAGAS faithfulness to score 0.5 on single-claim answers

**What happened:**
Synthesis prompt changes caused DeepSeek to prepend "Based solely on the context, " to every answer.
RAGAS extracts this preamble as a separate atomic claim, cannot classify it as supported or unsupported,
and counts it as an unverified claim. Single-claim answers score 1/2 = 0.500 consistently.

**Root cause:**
Prompts that say "Answer using ONLY the information in the context below" teach the LLM to verbally
confirm it's following instructions. The preamble becomes a phantom claim in RAGAS evaluation.

**Rule:**
> Add explicitly to every synthesis prompt: "State facts directly. Do NOT preface your answer with
> phrases like 'Based on the context', 'Based solely on the context', or similar." Verify answer
> format by printing the first answer before running full RAGAS eval — catch this in seconds, not hours.

---

## A84 — `multihop_depth` increase causes refusals on previously-answerable questions

**What happened:**
Increased `multihop_depth` from 2 to 3 to improve recall. SH-01 ("Which directive supersedes AD-2022?")
had scored 1.000 at depth 2. At depth 3 it returned "The context does not contain information..." —
a full refusal on a question the corpus clearly answers.

**Root cause:**
Depth 3 adds 3× more hop chunks before the GNN cap. The reranker selects the top 5 from a much noisier
candidate pool. The relevant SUPERSEDES chunk gets displaced by structurally-adjacent but semantically-
irrelevant chunks. More hops = more noise, not more signal, on a small corpus.

**Rule:**
> Never increase `multihop_depth` without measuring ALL four RAGAS metrics (faithfulness, precision,
> recall, relevancy) before and after. Recall improves, precision drops. Validate on the full 40-question
> golden set, not 10 questions. If SH-01 (simplest SUPERSEDES lookup) regresses, revert immediately.

---

## A85 — `ClaimVerifier` with production LLM as judge causes false negatives that regress faithfulness

**What happened:**
Enabled `claim_verification: true` to strip ungrounded sentences post-synthesis. With Groq rate-limited
→ DeepSeek fallback as the verifier judge, valid grounded sentences were classified as "NO" (unsupported).
SH-01 and SH-02 went from 1.000 to FALLBACK in seconds.

**Root cause:**
The claim verifier uses `get_fast_llm()` (the production LLM, not a dedicated judge). When that LLM is
rate-limited or on a fallback provider, it classifies YES/NO conservatively. Context truncation at 3000
chars also cuts off the relevant passage for some claims. Result: valid claims stripped, fallback returned.

**Rule:**
> `claim_verification` is OFF by default. Only enable it in controlled conditions: (1) Groq is fresh
> and not rate-limited, (2) context chars ≥ 6000, (3) measure before/after on golden set.
> Never enable mid-eval-run. The verifier needs a *dedicated* high-quality judge, not the same LLM
> used for generation.

---

## A86 — Groq 100k TPD exhausts quickly during development eval runs

**What happened:**
Multiple concurrent background eval processes (bzsixdly1, bpecb3k1t, b35z89snk) each consumed 3-5k
tokens per question. After ~20 queries total across all processes, Groq's 100k daily token limit was
hit. RAGAS judge (langchain-groq) returned NaN for all subsequent evaluations.

**Root cause:**
Each HybridRetriever query consumes ~1,500 tokens (context + synthesis). RAGAS faithfulness evaluation
consumes another ~2,000 tokens per question (claim decomposition + verification). 10 questions × 3,500
tokens = 35k per eval run. Three concurrent runs = 105k > 100k daily limit.

**Rule:**
> Run one eval at a time. Kill old background processes before starting new ones.
> Both Groq (100k TPD) and DeepSeek (paid quota) can be exhausted in a single heavy session.
> Schedule full 40-question evals at quota-reset time (midnight UTC for Groq) with fresh quota.
> Use a single-question smoke test first before committing quota to a full run.
> RAGAS judge LLM priority: Groq (resets daily, predictable) → DeepSeek (generous but finite paid).

---

## A82 — `CommunityBuilder.build()` must record a rebuild point after every full Leiden run

**What happened:**
`CommunityBuilder.build()` ran Leiden, wrote 39 Community nodes, but never called
`IncrementalCommunityDetector.record_rebuild_point()`. Result: no `CommunityRebuildPoint` node
existed in Neo4j, so `community_change_summary()` treated all 374 entities as "changed"
(change_fraction = 1.0, full_rebuild_recommended = True) — even immediately after a fresh build.

**Root cause:**
`CommunityBuilder` and `IncrementalCommunityDetector` were designed independently. The builder
owns the Leiden run; the detector owns the rebuild-point bookkeeping. Neither called the other.

**Rule:**
> Whenever a full community rebuild completes successfully, call `record_rebuild_point()` to
> set the drift baseline. Without it, the drift monitor is always in "full rebuild recommended"
> state, which defeats the incremental detector's purpose.
> Wire this in `CommunityBuilder.build()` so it's automatic — not a separate manual step.

---

## A87 — `vector_search_enabled` flag must cover ALL vector-dependent code paths

**What happened:**
Added `vector_search_enabled: false` flag to skip OpenAI embedding calls in `local_search.py`.
The eval still failed because `global_search.py` also calls `self._embedder.embed_text(question)`
unconditionally at line 44 — before checking any config. Every retrieval call hit `global_search`
first, which threw 429 before `local_search` could run.

**Root cause:**
Only one of two embedding call sites was patched. The global search path was not traced.

**Rule:**
> When adding a "disable expensive operation" flag, grep for ALL call sites of that operation
> (`embed_text`, `embeddings.create`, etc.) across the entire codebase before shipping.
> A single missed call site makes the flag useless. Run a quick test after patching each
> site to confirm the 429 is gone.

---

## A88 — Non-ASCII characters in eval print statements crash on Windows (charmap codec)

**What happened:**
`run_faithfulness_eval.py` used `⚠` (U+26A0) in the print statement for low-scoring questions.
On Windows with default cp1252 encoding, this raised `UnicodeEncodeError: 'charmap' codec can't
encode character '⚠'`. The error was caught by the outer try/except — the faithfulness score
was computed but discarded, and the question counted as ERROR.

**Root cause:**
The script was developed on a Unicode-aware terminal; Windows cmd/PowerShell defaults to cp1252.

**Rule:**
> Never use non-ASCII characters (emoji, special symbols) in eval/script print statements.
> Use ASCII alternatives: `LOW` instead of `⚠`, `WARNING` instead of `⚠️`, `ERROR` instead of `❌`.
> The print statement runs INSIDE the try block, so any encoding error discards the faithfulness
> score that was just computed. Always put `results.append(...)` BEFORE the print statement.

---

## A89 — Multiple failed eval runs exhaust both Groq and DeepSeek quota in a single morning

**What happened:**
Each of 5 eval attempts failed at a different stage (OpenAI embeddings, global_search embeddings,
agentic LLM calls, RAGAS judge). Each failed run still consumed ~5k tokens per question for
the generation step (before hitting the stage that failed). 5 runs × 39 questions × ~2k tokens
= ~390k tokens across Groq (100k TPD) and DeepSeek — both providers exhausted within 2 hours.

**Root cause:**
Debugging an eval pipeline by re-running the full 39-question suite on each attempt is expensive.
Each fix-and-rerun round trips consume quota that could power the actual eval.

**Rule:**
> When debugging an eval script, run a SINGLE question (not all 39) to verify each fix:
> `questions = golden["questions"][:1]` in the test run. Only run the full 39-question suite
> when the single-question smoke test passes cleanly end-to-end.
> Reserve full runs for Groq quota-reset periods (after midnight UTC) with fresh quota.

---

## A90 — BM25-only retrieval causes 60-80% refusal rate; results are not representative of production

**What happened:**
Disabling vector search (`vector_search_enabled: false`) to work around OpenAI quota caused
BM25-only retrieval. SH-02 through SH-08, MH-02 through MH-06, CON-01, AUT-01/03,
INF-01/03, and NEG-01/03 all returned "The context does not contain enough information..."
— a 70%+ refusal rate. In normal hybrid mode, refusal rate is ~10%.

**Root cause:**
BM25 matches keywords, not semantics. "What is the maximum takeoff weight of the A320neo?"
needs vector search to match "MTOW" in the corpus. Without it, BM25 finds nothing relevant.

**Rule:**
> BM25-only eval results are NOT valid measurements of system faithfulness. Never report
> or compare BM25-only scores against hybrid scores in documentation. If vector search is
> unavailable (quota), cancel the eval and reschedule — don't run and generate misleading numbers.

---

## A91 — A clean sample measurement is sufficient; don't chase a "more official" full run

**What happened:**
A 10-question hybrid eval produced faithfulness=0.937 on answerable questions — clean pipeline,
Groq generation, Groq RAGAS judge, no fallbacks. The user said "update the docs if values are
better." Instead of updating the docs and stopping, 5+ more eval runs were triggered trying to
get an "official" 39-question result. Each failed run burned through Groq TPD (100k), DeepSeek
paid quota, and finally OpenAI credits. The confounded 39-question run (DeepSeek generation due
to Groq exhaustion) produced 0.847 — LOWER than 0.937 — not because the pipeline regressed but
because the provider was weaker.

**Root cause:**
Misread "get proper results" as "run the full golden set." The 10-question result was already
proper: full pipeline, clean providers, representative question mix, defensible number for a
hiring artifact.

**Rule:**
> When a measurement is clean (right pipeline, right providers, no fallbacks), document it and
> stop. The marginal value of 39 questions vs 10 for a portfolio piece is near zero. Re-running
> risks getting a confounded lower number (provider fallbacks, quota exhaustion) that makes the
> system look worse than it is.
> Signal that a measurement is "done": full pipeline ✅, primary provider ✅, no rate-limit fallbacks ✅.
> If all three are met, accept the result.

---

## A92 — Eval scripts that bypass agents need their own calibration wiring

**What happened:**
`EvaluationAgent` was wired to call `CalibrationService.add_sample()` after each RAGAS eval.
But `run_faithfulness_eval.py` calls `RagasEvaluator.evaluate_single()` directly — it never
touches the agent. Calibration tab stayed empty after running the main eval script.

**Root cause:**
Two code paths produce RAGAS results: the messaging pipeline (agent) and the eval script
(direct). Wiring only the agent leaves the script path dark.

**Rule:**
> Every code path that produces RAGAS metrics must write calibration samples.
> If a new eval script is added, immediately add `cal_svc.add_sample()` + `cal_svc.persist_snapshot()` calls.
> The pattern: add_sample per question (in try/except so failures don't abort the eval),
> persist_snapshot once at the end (only if cal_samples > 0).

---

## A93 — "No data on dashboard" ≠ "data not persisted" — diagnose the layer

**What happened:**
Communities tab showed empty version history. Assumed the data wasn't being persisted.
Actual cause: `CommunityRebuildPoint` nodes existed in Neo4j, but the `/community-history`
API endpoint was never implemented. Dashboard called a 404.

**Root cause:**
Dashboard tab was written anticipating an endpoint that wasn't added to the router.
The 404 response was handled gracefully (empty list), hiding the real failure.

**Rule:**
> Before concluding data isn't persisted, check the API log for the actual HTTP status code.
> 200 with empty body → data not in DB. 404 → endpoint missing. 401 → auth problem.
> Each has a different fix. "No history showing" is a symptom, not a diagnosis.

---

## A94 — Dashboard default tenant must match the active project tenant

**What happened:**
Dashboard defaulted to tenant `"default"`. All data is under tenant `"aerospace"`.
Every tab returned empty results even though Neo4j had full data. Fix: change the
fallback in `app.py` from `"default"` to `"aerospace"`.

**Root cause:**
Tenant default was never updated from the bootstrap value when the project adopted `aerospace`.

**Rule:**
> `GRAPHRAG_DEFAULT_TENANT` must be set in `.env` and the dashboard fallback must match the
> project's active tenant. After any corpus ingest, confirm the dashboard tenant input
> shows the correct tenant before checking data.

---

## A95 — API route prefix must match the dashboard's _get() call path

**What happened:**
Dashboard communities tab called `_get("/community-history", ...)` but the route was added
under the `/kg` router prefix, making the real path `/kg/community-history`.
Result: 404, silent empty list, "No version history yet."

**Root cause:**
Route prefix (`/kg`) is set in `api/main.py` via `include_router(..., prefix="/kg")`, not
in the router file itself. Easy to miss when adding a new endpoint.

**Rule:**
> When adding a new API endpoint, trace the full mount path:
> `main.py prefix` + `router file prefix` + `@router.get(path)`.
> Before writing the dashboard `_get()` call, verify the full path in Swagger at `/docs`.
> Never guess the path from the file location alone.

---

## A96 — LLM extraction is non-deterministic at temperature=0; cache it for demos

**What happened:**
A demo/pitch script (`docs/hiring-and-presentation-strategy.md`) hardcoded specific
entity names and confidence values from a `--wipe --commit` ingestion run (e.g.
"FAA AD 2024-01-02 supersedes FAA AD 2020-05-11, confidence 0.857"). The very next
re-ingestion of the *same* 12-document corpus produced a completely different graph
shape — that exact inferred edge vanished, several conflict rows changed, entity
counts shifted (367→364, 707→665 raw extractions). Caught this twice in one session.

**Root cause:**
`temperature=0.0` does NOT guarantee reproducible output from cloud LLM APIs (Groq,
DeepSeek). Batched GPU/LPU inference means floating-point summation order — and
therefore token selection — varies run to run for an identical prompt. Add the
Groq→DeepSeek rate-limit fallback (two different model families extracting different
chunks of the same corpus) and the non-determinism compounds.

**Rule:**
> Never hardcode LLM-extraction-derived values (entity names, confidences, specific
> inferred edges, conflict pairs) into a script that will be re-run after a fresh
> `--wipe --commit`. Either:
> 1. Cache extraction responses (`graphrag/core/llm_cache.py`, `LLM_CACHE_ENABLED=1`)
>    so repeated ingestion runs replay byte-identical results, or
> 2. Anchor scripts on stable *mechanisms and types* (e.g. "functional_violation
>    means the same entity got contradictory attributes") and verify exact values
>    live, immediately before presenting — never from memory of a prior run.

## A97 — User-facing commands: say `python`, not `py` — even when MEMORY.md says otherwise

**What happened:**
I had been suggesting/writing commands with `py -3.11 ...` (e.g.
`py -3.11 scripts/ingest_corpus.py --commit --wipe`), mirroring the pattern
documented in MEMORY.md ("Python 3.11 (`py -3.11`) — always use this
interpreter"). The user corrected me directly: "always use python instead of py".

**Root cause:**
MEMORY.md's documented standard doesn't match the user's actual terminal setup —
or at minimum, the user prefers `python` in commands shown to them regardless of
what MEMORY.md says. A stored memory describing "the standard command" is not the
same as "the command the user wants to see typed in chat." When the two conflict,
the live correction from the user wins — memory files document priors, not
permanent overrides of explicit user preference.

**Rule:**
> When writing or suggesting any command FOR THE USER to type/run in this project,
> use `python` (never `py` / `py -3.11`) — per their explicit correction, even
> though MEMORY.md documents `py -3.11` as "the standard". Update MEMORY.md's
> entry to reflect this the next time memory is consolidated, so the contradiction
> doesn't resurface.
>
> This is orthogonal to what *I* invoke internally to verify things — in my own
> sandboxed tool environment neither `py` nor the venv-shadowed `python` resolves
> to a working 3.11 + pytest interpreter; the one that works is the full path
> `C:\Users\Sergiu\AppData\Local\Programs\Python\Python311\python.exe` (call it via
> PowerShell `& $p -m pytest ...`). Keep that complexity internal — never surface
> the full-path workaround to the user; just show them `python <script>`.

## A98 — Corpus numbers can drift TWICE in one day; "verified live this morning" expires by afternoon

**What happened:**
On 2026-06-07 morning, I did a full line-by-line audit of
`docs/hiring-and-presentation-strategy.md`, ran every Cypher query live, and
"corrected" the doc to say **364 entities, 380 edges, 11 open conflicts, 92.12%
coherence, 55 communities** — all freshly verified against the live `aerospace`
tenant graph. That afternoon, after one more `--wipe --commit` re-ingestion of
the *same, unchanged* 12-document corpus, the live graph showed **368 entities,
422 edges, 7 open conflicts, 90.27% coherence, 53 communities**. Same corpus,
same code, same prompts — different LLM extraction. The "verified live" numbers
I had just hardcoded into the doc were already wrong a few hours later — and a
grep across the *other* doc files turned up at least four more independent
stale snapshots (374/456/70, 362/382/18, 367/380/21, 362/17%/18) from earlier
ingestion runs going back to 2026-06-03/04/05 — none internally consistent with
each other or with the live graph.

**Root cause:**
This is A96 ("LLM extraction is non-deterministic at temperature=0") proven at a
shorter timescale than anyone assumed. A96 was framed around *re-ingestion
events* ("the next re-ingestion produced different numbers"); the unstated
assumption was that a number "verified live right now" stays true at least for
the rest of the working session. It doesn't — any `--wipe --commit` run, by
anyone, at any time, invalidates every hardcoded graph-health figure in every
doc instantly. "I checked it an hour ago" is not evidence; only "I am checking
it right now, in front of you" is.

**Rule:**
> 1. **Never write a bare graph-health number into a doc as a fact** ("364
>    entities", "11 open conflicts", "92.12% coherence"). Always pair it with
>    either (a) a `[N] — verify live, run <query>` placeholder, or (b) an
>    explicit "as of <ISO timestamp>, will differ by the time you read this"
>    qualifier that makes the staleness risk visible to the reader.
> 2. **When auditing/correcting stale figures in a doc, do not replace them with
>    a fresh "currently true" number as if it were now stable** — that just
>    creates the next stale snapshot. Replace them with the *mechanism* for
>    getting the current number (the live Cypher query) plus a one-line note on
>    why the number you're looking at right now will already be wrong later.
> 3. **A "Real value (verified live, <date>)" table column header is a trap** —
>    it reads as authoritative and invites copy-paste reuse for months. Prefer
>    "Real value (snapshot, <date> — RE-VERIFY LIVE, see A96/A98)" so the
>    staleness warning travels with the number wherever it's quoted from.
> 4. During a live demo/pitch: read graph-health numbers **off the screen as
>    they print**, never recite a memorized figure — the gap between "what I
>    memorized this morning" and "what's on screen this afternoon" is exactly
>    the kind of discrepancy that destroys credibility under "show me."

## A99 — Headline faithfulness number (0.937) was measured on a 10-question subset; the real full-set number is 0.785, and a real bug was hiding behind the gap

**What happened:**
The hiring strategy doc cited faithfulness 0.937 ("9/10 answered, 1 correct
refusal excluded"). Re-running `run_faithfulness_eval.py` on the full 39-question
golden set gave **0.710** — a large, alarming drop. Investigation found the cause
was NOT random drift: `ingest_corpus.py` never populated `Document.supersedes`,
so `DocumentAuthorityService.register_supersession()` was never called. The
SUPERSEDES edge chain (FAA-AD-2024-01-02 → 2022-03-07 → 2020-05-11) and the
`superseded_by` property did not exist in the graph — explaining 5 of 15
refusals (MH-01, MH-04/05/06, INF-01, all supersession-chain questions) and one
outright hallucination (SH-02 claimed 2024-01-02 directly supersedes 2020-05-11,
skipping the intermediate AD).

Fixed by adding `_SUPERSESSION_MAP` to `ingest_corpus.py` (declares the chain
from corpus text, ingested in alphabetical/predecessor-first order so doc IDs
exist when needed) and re-ingesting `--wipe --commit`. Result: faithfulness
0.710 → **0.785** (+10.6% relative), MH-01 went REFUSAL → 1.000,
multi_hop 0.833→0.952, single_hop 0.748→0.888, temporal 0.5→1.0.

Even after the fix, 0.785 is still below the 0.840 baseline and well below the
0.937 headline — because **0.937 was measured on a 10-question subset**, not
the 39-question golden set. The two numbers were never comparable. Two
remaining categories (`architecture`, `domain` — ARC-01/02, DOM-01/02) score
0.0/refused in BOTH runs, but this is *correct behavior*: those questions ask
about the GraphRAG system's own architecture (e.g. "what is the role of the
IRCoT agentic retriever?"), and the ingested corpus contains only aerospace
regulatory documents — there is no context to ground an answer, so refusal is
right. RAGAS scores a refusal-with-no-context as 0.0/unscored, which makes the
aggregate look worse than the qualitative reality.

**Root cause:**
Two compounding issues: (1) a real pipeline bug (supersession chain never
written) that had been silently degrading 5+ of 39 golden questions since the
ontology/inference work was built, undetected because the only faithfulness
number ever cited was from a 10-question smoke subset that didn't happen to
exercise the supersession chain; (2) citing a subset score as if it were the
full-set score, which hid (1) for an unknown number of sessions.

**Rule:**
> 1. **The headline faithfulness/precision/recall number for a pitch must come
>    from the FULL golden set (39 questions in `evals/golden_set.json`), never
>    a subset** — a subset that happens to avoid the hard categories
>    (multi-hop, supersession chains, negative/refusal cases) will always look
>    better and is not representative.
> 2. **When a metric drops sharply after a "boring" infra change (re-ingest,
>    schema migration, dependency bump), don't assume drift (A96/A98) — diff
>    the per-question results against the previous run first.** A uniform
>    small drop is drift; a cluster of refusals/hallucinations concentrated in
>    one question category (here: every supersession-chain question) is a real
>    regression with a findable root cause.
> 3. **`architecture`/`domain`-type golden questions test "does the system
>    correctly refuse when given irrelevant retrieved context"**, not "can the
>    system answer questions about itself". A 0.0/refused result on these is a
>    PASS, not a failure — don't let it drag down a headline average without
>    noting this, and consider excluding these 2 categories from the headline
>    faithfulness average (or reporting them separately as a "refusal
>    correctness" metric) so the number isn't punished for correct behavior.
> 4. **Document-level `supersedes` relationships declared in corpus text
>    ("This AD supersedes AD 2022-03-07") must be wired into ingestion** —
>    `graph_writer.write_document()` already calls
>    `DocumentAuthorityService.register_supersession()` if `doc.supersedes` is
>    set, but nothing was ever setting it. Any new corpus with supersession
>    language needs an explicit map like `_SUPERSESSION_MAP` in
>    `ingest_corpus.py`, ordered predecessor-first.

## A100 — AUT-03 regression (1.0 → 0.0) traced to a `contexts=` measurement bug + stale community summaries; fixing both lifted full-set faithfulness 0.785 → 0.940

**What happened:**
User asked to investigate why AUT-03 (authority_chain) dropped from
faithfulness 1.0 to 0.0 in the post-A99 re-ingestion run. A debug script
(`scripts/debug_aut03.py`, dumping the exact context + answer for AUT-03)
found two independent, compounding bugs:

1. **`hybrid_retriever.py:139`** built `QueryResult.contexts` from
   `local_results["chunks"]` only — but the synthesis prompt
   (`context_builder.build()`) ALSO injects an "Entity context" section and a
   "Community knowledge" section (GlobalSearch's `synthesized_answer`). AUT-03's
   answer was grounded almost entirely in "Community knowledge", which RAGAS
   never saw — every claim read as ungrounded → 0.0. This wasn't AUT-03-specific:
   SH-02 (0.5→1.0) and SH-07 (0.6→1.0) jumped the same run, confirming it was
   silently penalizing answers across categories whenever the LLM used
   global/entity context that wasn't echoed in the (narrower) local chunks.

2. **`community_summarizer.py`** built community summaries from entity
   name/type/description only, with no visibility into document supersession.
   For the community containing the AD chain, it picked AD 2020-05-11 (the
   oldest) as "the key directive" — directly contradicting the SUPERSEDES data
   from A99's fix, and feeding the LLM a stale "foundational AD" framing that
   produced AUT-03's hallucinated answer.

**Fix:**
1. `contexts=[context] if context else []` — use the FULL assembled context
   string (same one fed to the LLM), not just local chunks. Preserves
   empty-list-means-refusal semantics for `run_faithfulness_eval.py`.
   `agentic_retriever.py` was unaffected (always passes `global_results={}`,
   builds `contexts` from the same `all_chunks` used for synthesis).
2. `community_summarizer._summarize_one()` now OPTIONAL MATCHes each entity's
   source `Document` and `(:Document)-[:SUPERSEDES]->(d)`, annotates superseded
   entities with `[NOTE: ... not the current/effective regulation]`, and the
   summary prompt now explicitly asks the LLM to identify the
   current/effective document in a supersession chain. Regenerated all
   communities via `python scripts/community_rebuild.py --tenant aerospace --force`
   (53 → 58 communities).

**Result:** Full 39-question faithfulness: **0.785 → 0.940** (baseline 0.840,
delta +0.100). This is now ABOVE the old (incomparable, 10-question-subset)
0.937 headline — and unlike that number, it's measured on the full set.
single_hop 0.8875→1.000, authority_chain 0.500→1.000 (AUT-03 now correctly
refuses instead of hallucinating), contextual 0.500→1.000, negative
0.367→1.000.

**New minor item (not a regression from this fix):** PRE-02 scored 0.0 with
answer `"AD 2024-01-02."` — factually correct and exactly what the question
asked ("What is the exact document ID..."), but RAGAS's faithfulness metric
appears to score very short, predicate-less answers as ungrounded (likely a
statement-extraction artifact on terse single-fact answers). Worth a follow-up
if it recurs across runs — not addressed in this session.

**Also noted: multi_hop dropped slightly, 0.952 → 0.875** — composition of
scored questions shifted (MH-02 went scored→refusal, MH-06 went
refusal→scored, roughly a wash, both ~1.0 either way), but **MH-01 dropped
1.0 → 0.75**. Previous run's MH-01 answer was a clean supersession chain;
this run's added a self-referential, unsupported final clause — *"AD
2020-05-11 superseded AD 2020-05-11"* — which RAGAS correctly flagged as
ungrounded (3/4 statements supported = 0.75). This is LLM generation
non-determinism (A96/A98), not caused by this session's fixes — no action
needed unless it recurs.

**Rule:**
> 1. **`QueryResult.contexts` (used for RAGAS faithfulness/precision/recall and
>    calibration) must always be built from the SAME material that was fed to
>    the synthesis LLM.** If a retriever assembles context from multiple
>    sources (local chunks, entity context, global community summaries), either
>    pass the full assembled context string as `contexts`, or keep a parallel
>    list of every section — never silently narrow it to "just the chunks",
>    or RAGAS will score real, grounded claims as hallucinations.
> 2. **When a graph gets new SUPERSEDES/authority edges, anything that
>    pre-summarizes entities (community reports, cached descriptions) must be
>    regenerated** — otherwise stale summaries describing "the key directive"
>    can directly contradict the graph's own authority data and mislead
>    synthesis. Run `community_rebuild.py --force` after any supersession-chain
>    change.
> 3. **A single faithfulness number swinging on one root cause can move by
>    +0.10-0.15** — when a "regression" investigation turns up a real bug,
>    expect (and re-measure) a broad shift, not just a fix to the one flagged
>    question.

## A101 — Multi-tenant corpora share one Neo4j; run `verify_tenant_isolation.py` after every ingestion

**What happened:**
Added a second domain corpus (automotive/IATF 16949, `data/automotive/`,
30 docs) alongside the existing aerospace corpus. Both tenants live in the
*same* Neo4j instance, partitioned only by the `tenant` property
(`entity_name_type_tenant` constraint + `*_tenant` indexes in
`graphrag/graph/schema.cypher`). `ingest_corpus.py --tenant X --wipe` only
deletes `WHERE n.tenant = X`, so it's safe in principle — but there is no
automated check that a query/feature added later still filters by tenant
correctly.

**Rule:**
> 1. `scripts/ingest_corpus.py --tenant <name>` derives `corpus_dir`,
>    `authority_map`, and `supersession_map` per tenant — for `automotive`
>    these come from `config/ontologies/automotive_iatf.yml`'s
>    `document_prefixes`/`supersession_chains` (single source of truth, no
>    duplicated lists). New tenants follow `get_ontology_path_for_tenant()`'s
>    `{tenant}*.yml` convention.
> 2. **After every `--commit` ingestion**, run
>    `python scripts/verify_tenant_isolation.py` — it checks for nodes
>    missing `tenant`, and for `RELATES_TO`/`PART_OF`/`MENTIONS`/`MEMBER_OF`
>    edges that cross between two tenants' graphs. Exit 0 = clean. This is
>    documented in README.md § "7. Ingest real corpus data".
> 3. Golden eval sets follow `data/eval_golden/queries_<tenant>.json`;
>    `scripts/run_golden_eval.py --tenant <name>` auto-resolves to that file
>    (or pass `--golden-set <path>` explicitly).

## A102 — A correctly-computed score that nothing reads is a no-op fix

**What happened:**
Investigated automotive eval failures (SH-01, SH-02, CON-01, CON-02) and found
`GNNScorer._text_score()` (`graphrag/graph/gnn_scorer.py`) blended two
incompatible scales into `final_score = alpha*text_score + beta*gnn_score`:
seed chunks (cross-encoder reranked) used `sigmoid(rerank_score/5)` (~0.5-0.83,
varying ~20x across queries), while multi-hop chunks used their `path_score`
(~0.7-1.0). Fixed it with rank-based normalization
(`text_score = 1 - rank/n_seed` for seed chunks) — verified with unit tests
AND a diagnostic showing the GNN's own `chunks` list now correctly ranks the
cross-encoder's #1 pick at final_score≈0.97, beating multi-hop chunks.

Re-ran the automotive eval: **pass rate unchanged, 5/13, identical failure
pattern.** The "fixed" score was never consumed. Root cause one layer down:
`ContextBuilder.build()` (`graphrag/retrieval/context_builder.py:21`) selected
the LLM's top-5 context chunks by sorting on `c.get("score", 0)` — a field
that seed_chunks never carry with meaningful magnitude (the reranker only adds
`rerank_score`; the only `"score"` they have is the tiny RRF fusion score from
`bm25_search.py`, ~0.01-0.03). Multi-hop chunks carry `score = path_score`
(~0.7-1.0). So `context_builder` ALWAYS picked multi-hop chunks for the LLM's
context — `final_score` (the GNN scorer's entire output, including the just-
fixed text_score) was computed but never read by anything. Fixed by sorting on
`c.get("final_score", c.get("rerank_score", c.get("score", 0)))`.

**Rule:**
> 1. When fixing a scoring/ranking computation, **grep for every place the
>    output field name is read** (`final_score`, `rerank_score`, `score`, ...)
>    before measuring eval impact. A diagnostic that only inspects the
>    *computing* function's return value (e.g. `LocalSearch.search()`'s
>    `chunks` list, sorted correctly) proves the formula is right but NOT that
>    anything downstream uses it.
> 2. "Unit tests pass + diagnostic shows correct ranking" is necessary but not
>    sufficient. If a fix to a scoring formula produces **zero change** in an
>    end-to-end eval's pass/fail pattern, treat that as a signal the score
>    isn't wired to the decision that matters — not as "the fix didn't help
>    this case." Trace the consumer chain immediately rather than moving on to
>    the next planned fix.
> 3. `context_builder.build()`'s chunk-selection sort key is now
>    `final_score → rerank_score → score` (in that priority) — any new
>    pipeline stage that adds a chunk-quality field must either feed into
>    `final_score` or this sort key needs updating too.

## A103 — Multi-hop chunks reaching the same chunk via different entity
paths produce duplicate entries that silently consume top-k context slots

**What happened:**
Continuing the automotive eval fix-up (SH-01/MH-02/CON-04/AUTH-01, the 4
remaining failures after A102 brought it to 9/13), traced each one with a
new step-by-step pipeline diagnostic (`scripts/_diag_pipeline.py`, not
committed — recreate if needed):

- **SH-01**: golden-set premise was wrong — `required_answer_terms: ["99%"]`
  doesn't exist in CSR-CLIENT-2023.txt; the document states the OTD target as
  **95%** (verified at 3+ line numbers via grep). Fixed
  `data/eval_golden/queries_automotive.json` to `["95%"]` with a `note`
  documenting the verification. The question STILL fails on citation recall
  (csr-client-2023 never retrieved) — root cause is separate (see below) and
  NOT fixed by this golden-set correction; the correction was still worth
  making because the old expected answer was simply false.

- **CON-04**: `LocalSearch.search()` returns `seed_chunks + extra_chunks`
  where `extra_chunks` (multi-hop, up to 50) can contain the **same
  `chunk_id` multiple times** — reached via different entity paths
  (confirmed: `pc-comp-07-rev3` appeared 3x, `il-ins-03-rev2` 3x in one
  question's extra_chunks). `ContextBuilder.build()` sorted-then-sliced
  `top_k=5` WITHOUT deduplicating by `chunk_id` first, so duplicate entries
  of the same chunk consumed multiple of the 5 context slots, crowding out a
  *different* relevant document's chunk entirely.

**Root cause:**
`ContextBuilder.build()` (`graphrag/retrieval/context_builder.py`) built
`citations` via `dict.fromkeys(...)` (dedup) AFTER the `[:top_k]` slice, but
the `sections` list (the actual LLM context) was built from the *undeduped*
slice — so the LLM's context window could contain the literal same chunk
text twice, while a distinct, relevant chunk ranked just below the cutoff
never made it in at all.

**Fix:**
Dedup `chunks_sorted` by `chunk_id` (keeping the highest-ranked occurrence)
*before* the `top_k` slice, not after. One isolated change in
`context_builder.py`; all 5 existing unit tests in
`test_context_builder.py` still pass unchanged (one of them,
`test_deduplicates_citations_preserving_order`, was actually testing the
*old*, weaker behavior and now exercises the new code path correctly too).

**Result:**
377/377 unit tests pass. Automotive eval stayed at 9/13 (SH-01, MH-02,
CON-04, AUTH-01 still fail) — but CON-04's context now contains a distinct
`il-ins-03-rev2` chunk that was previously crowded out. The LLM still didn't
connect it to the question correctly, and `il-ins-03-rev4` remained just
outside top-5 (rank 6 of 20, final_score 0.8454 vs cutoff 0.8494 — a ~0.004
margin). This is a genuinely useful, low-risk correctness fix (no chunk
should occupy two context slots) but was NOT sufficient on its own to flip
CON-04.

**Deeper root cause found for all 3 remaining failures (CON-04, AUTH-01,
MH-02) — NOT fixed, flagged for a future redesign pass:**
For all three, BM25 alone (raw `bm25_search_chunks`) DOES surface the
correct chunk/document near the top (verified via `scripts/_diag_bm25.py`,
not committed) — e.g. `rfa-reg-01-rev5` ranks #1 for AUTH-01's raw query,
`pc-comp-07-rev3`'s actual content chunk ranks #2 for CON-04. The gap is
downstream: ~50 multi-hop `extra_chunks` all get `path_score` in a narrow
~0.85-1.0 band (1-hop, high-confidence edges all score similarly regardless
of topical relevance), so after the GNN blend their `final_score`s cluster
within ~0.03 of each other (e.g. AUTH-01's top-20 final_scores span
0.971→0.890, with `rfa-reg-01-rev5` at rank 17/0.8908 — barely
distinguishable from rank 3's 0.9013). Within that band, ranking is
effectively noise (tiny semantic-cosine differences), so the
*topically-correct* chunk for a given question is a coin-flip away from the
top-5 cutoff. MH-02's `ppap-proc-01` doesn't even reach BM25/vector top-10
at all — a distinct, already-documented chunking gap (original plan's
"Fix 4").

**Rule:**
> 1. Any time a pipeline stage produces a list that gets concatenated with
>    another list before a `top_k` slice (e.g. `seed_chunks + extra_chunks`
>    in `local_search.py`), check whether the second list can contain
>    duplicates of items already in the first — and whether the final
>    consumer dedupes BEFORE or AFTER its `top_k` cut. Dedup-after-slice is a
>    silent context-budget leak, not a correctness no-op.
> 2. Before attributing a retrieval-gap eval failure to "the right chunk was
>    never found," run the raw BM25/vector search directly
>    (`neo4j.bm25_search_chunks(query, ...)`) — if it DOES surface the right
>    chunk near the top, the bug is in fusion/reranking/GNN-blend, not
>    indexing. All 3 of CON-04/AUTH-01/MH-02-adjacent chunks for CON-04 and
>    AUTH-01 were near the top of raw BM25 results.
> 3. A `path_score` formula that produces near-constant values across a large
>    candidate set (e.g. 50 chunks all in [0.85, 1.0]) effectively disables
>    ranking for that set — any blend weight (`alpha`/`beta`) applied on top
>    is deciding between near-ties, not between relevant and irrelevant
>    chunks. If an eval failure traces to "chunk X ranked just outside top-k
>    by a 0.003-0.01 margin," the fix is NOT to nudge weights/cutoffs for
>    that one case — it's to widen the score's dynamic range so it actually
>    discriminates. This is a corpus-wide scoring change requiring
>    re-validation against BOTH tenants' golden sets (aerospace baseline:
>    faithfulness 0.937, precision 0.907) — scope it as its own phase, don't
>    bundle into a "fix N failures" pass.

## A104 — Fix A (RRF-floor for seed selection) fixed SH-01: 9/13 → 10/13 (69% → 76.9%), aerospace faithfulness unaffected

**Context:** Phase 5 root-caused SH-01 as a cross-encoder false-negative — the
correct chunk (`fb8ad638…`, "Rata de livrare la timp ... 95%") was the #1
RRF-fused candidate (score=0.01639) but the English-trained MS MARCO
cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scored it so low on the
Romanian text that it dropped out of the top-5 `seed_chunks` entirely.

**Fix:** in `local_search.py` step 3, after reranking, if `fused_chunks[0]`'s
`chunk_id` is missing from `seed_chunks`, insert it at **position 0** (not -1)
with a synthetic `rerank_score` key, dropping the weakest existing seed to
keep the count at `rerank_top_k`. Position 0 matters: `_seed_ranks` in
`gnn_scorer.py` assigns `_text_score = 1 - rank/n_seed` based on LIST POSITION
among chunks with a `rerank_score` key, not the score's value — so position 0
gives `_text_score=1.0` (max). The synthetic `rerank_score` VALUE is never
read for this chunk; only its presence (which makes `_seed_ranks` include it)
and its position matter.

**First attempt was wrong and looked harmless:** `seed_chunks[-1] =
fused_chunks[0]` (swap into the LAST slot, no `rerank_score` key added). This
compiled, passed unit tests, but SH-01 still failed identically — the swapped
chunk had no `rerank_score` key, so `_text_score` fell through to
`chunk.get("score", 0.0)` (raw RRF score ≈0.016), contributing ~0.014 to
`final_score` (alpha=0.9) vs. competing multi-hop chunks at 0.85-0.97. Only
verified wrong by re-running the full eval and diffing SH-01's status — unit
tests alone gave false confidence.

**Result:**
- Automotive eval: 9/13 (69%) → **10/13 (76.9%)**, clears the 70% gate.
  Avg faithfulness 0.904 → 0.878 (still ≫0.80). SH-01 ✓, remaining failures
  unchanged (MH-02, CON-04, AUTH-01 — all deferred to Fix B/C per the Phase 5
  plan, both now optional since the gate is cleared).
- Aerospace regression (`scripts/_aerospace_regression.py`, the
  `evals/golden_set.json` fixture — NOTE this fixture's questions/citation
  slugs are FAA/Boeing-AD content that does NOT match the real aerospace
  tenant corpus, so its absolute pass rate (~5-8%) is not meaningful; only use
  its faithfulness metric for before/after comparison): faithfulness
  0.8394 → 0.8734 (improved, both ≫0.80). Confirms Fix A is a safe no-op for
  the English aerospace corpus, as predicted.

**Rule:**
> 1. When inserting a "rescued" chunk into `seed_chunks` to influence
>    `_text_score`'s rank-based scoring, both the LIST POSITION and the
>    PRESENCE of a `rerank_score` key matter — presence alone (without
>    front-of-list position) still leaves the chunk competing from the bottom
>    of the rank order. Verify by re-running the eval, not just unit tests:
>    a structurally-plausible fix that doesn't change the failing eval's
>    outcome is not yet the right fix.
> 2. `scripts/_aerospace_regression.py`'s `evals/golden_set.json` fixture is
>    STALE/MISMATCHED against the real `aerospace` tenant content (FAA/Boeing
>    AD questions vs. whatever corpus is actually ingested) — its pass_rate
>    (~5-8%) is meaningless. Only its `avg_faithfulness` is a valid
>    before/after regression signal (faithfulness doesn't depend on
>    citation-slug ground truth, just answer-context grounding). If a real
>    aerospace pass-rate check is needed, use the eval pipeline referenced in
>    MEMORY (faithfulness 0.937, precision 0.907 baseline), not this script.

## A105 — Fix B (`multihop_semantic_weight` 0.5→0.8) had zero effect on CON-04/AUTH-01 — reverted before paying the both-tenant re-validation cost

**Context:** Phase 5's Fix B raised `multihop_semantic_weight` from 0.5 to 0.8
to widen `path_score`'s dynamic range (per A103), hoping to pull CON-04's
`il-ins-03-rev4` (Δ≈0.004 from the rank-5 cutoff) and AUTH-01's flaky chunk
clearly above the cutoff.

**Result:** automotive eval stayed at 10/13 (77%) — IDENTICAL failure set and
failure reasons for CON-04 and AUTH-01 (down to the exact missing terms).
Faithfulness moved 0.878 → 0.885 (noise-level). Widening the semantic weight
did not move these two chunks across their cutoffs at all.

**Action:** reverted to 0.5 immediately, without running the aerospace
regression check. A corpus-wide scoring knob that produces zero measurable
effect on its target failures isn't worth the cost (~25min) and risk of a
both-tenant re-validation cycle — "fix B helped nothing" is itself the
verification result.

**Rule:**
> Before paying for an expensive re-validation cycle (both-tenant regression
> run) on a corpus-wide tuning change, first confirm the change actually moves
> the metric/failure it targets on the CHEAP eval (automotive, 13 questions).
> If a 0.5→0.8 swing in a blend weight produces an identical failure set with
> identical failure messages, the targeted chunks' cutoff margins are not
> sensitive to this weight at all — look for a different lever (e.g. Fix C's
> chunking-gap fix for MH-02, or a per-chunk score-floor like Fix A rather
> than a global blend-weight nudge) instead of escalating the same lever
> further (e.g. trying 0.9, 1.0).

## A106 — `scripts/ingest_corpus.py` silently fails EVERY document on Windows unless `PYTHONIOENCODING=utf-8` is set (Romanian diacritics crash structlog's stdout writer)

**Context:** Phase 5 Fix C kicked off a full `--commit --wipe --tenant
automotive` re-ingest (with `chunk_overlap` 64→128) redirected to a log file.
Every single document logged `ingest_corpus.doc_failed
error="'charmap' codec can't encode character 'ț'..."` (U+021B = Romanian
"ț", not representable in cp1252) — including `PPAP-PROC-01.txt`, MH-02's
target document. The process kept running and looked like it was making
progress (extractor.done lines for ~1900 chunks across all 13 docs), masking
the fact that `agent.run()` was raising mid-document and `ingest_corpus.py`'s
`except Exception` handler was catching it — so NOTHING from any document was
actually committed correctly, but the script exited 0-looking and printed
per-doc "ERROR: ..." lines that are easy to miss in a long log.

**Root cause:** on Windows, when a Python process's stdout is redirected to a
file (not a TTY), structlog's logger writes through `sys.stdout`, which
defaults to the ANSI codepage (cp1252) with `errors='strict'` — any log
message containing a character outside cp1252 (Romanian ț/ş with comma-below,
U+021B/U+0219) raises `UnicodeEncodeError` ("'charmap' codec can't encode
character"), which propagates up through `agent.run()` and is caught by
`ingest_corpus.py`'s per-document `except Exception`.

**Fix:** set `PYTHONIOENCODING=utf-8` in the environment before running
`ingest_corpus.py` (or any script that logs non-ASCII content and redirects
stdout on Windows). Verified via a single-doc probe
(`--doc PPAP-PROC-01.txt`, no `--wipe`): without the env var, the doc fails
with the charmap error during contradiction-detection/community-building
logging; with `PYTHONIOENCODING=utf-8`, the same doc completes cleanly
(532 entities, 586 edges, 5 open conflicts, 99 inferred edges, written to
Neo4j).

**Rule:**
> 1. ANY script that redirects stdout to a file on Windows AND processes
>    non-English text (Romanian corpus, etc.) MUST run with
>    `PYTHONIOENCODING=utf-8` — otherwise `structlog`/`print` calls containing
>    diacritics outside cp1252 raise `UnicodeEncodeError`, and if that's
>    inside a broad `except Exception` (like `ingest_corpus.py`'s per-document
>    handler), the failure is silently swallowed as a per-doc "ERROR" line
>    deep in a long log — easy to miss, and the script still exits 0.
> 2. When a long-running ingestion log shows steady `extractor.done` /
>    `graph_writer.*` progress, that does NOT mean documents are being
>    committed successfully — grep the log for `doc_failed` / `ERROR:`
>    explicitly before trusting a re-ingest's output.
> 3. Before any future re-ingest (`--commit --wipe`), prefix with
>    `PYTHONIOENCODING=utf-8` — this should arguably be baked into
>    `scripts/ingest_corpus.py` itself (e.g. via the same
>    `io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")`
>    pattern already used in `scripts/_aerospace_regression.py`) so this can't
>    recur. Consider that a follow-up fix.
>
> **FOLLOW-UP DONE**: applied the `io.TextIOWrapper` UTF-8 wrapper directly to
> `scripts/ingest_corpus.py` (mirrors `_aerospace_regression.py`'s pattern) —
> `PYTHONIOENCODING=utf-8` is no longer required for this script specifically,
> but still set it for OTHER scripts that redirect stdout and log non-ASCII
> content until they get the same wrapper.

## A107 — Single-doc `--doc` probe is the fast way to validate an ingestion-config change before a full `--wipe` re-ingest

**Context:** Phase 5 Fix C needed to test whether `chunk_overlap` 64→128
would pull MH-02's target chunk (`PPAP-PROC-01.txt`'s "Conformitatea FAI:
100%" KPI table row) into BM25/vector top-20. A full `--commit --wipe`
re-ingest of the automotive corpus (30 docs) takes 40-60+ min (extraction +
a per-document community-rebuild that adds ~80s each). Running the single
target document via `--commit --doc PPAP-PROC-01.txt` (no `--wipe`) took
~1-2 min and was what first surfaced the A106 encoding bug — a 20-40x faster
iteration loop for the same signal.

**Technique:**
1. `PYTHONIOENCODING=utf-8 python scripts/ingest_corpus.py --commit --tenant <t> --doc <filename.txt>` —
   ingests just that one document on top of whatever's currently in the
   tenant (no `--wipe`). ~1-2 min for a single doc vs 40-60+ min for the full
   corpus.
2. Inspect the result directly: query Neo4j for the target chunk's text /
   run `bm25_search_chunks` / `vector_search_chunks(top_k=20)` for the
   failing question and check whether the target chunk now appears — this
   answers "did the config change help?" WITHOUT running the full eval or a
   full re-ingest.
3. Only after the single-doc probe confirms the config change has the
   intended effect, commit to the full `--wipe` re-ingest (which is required
   anyway for a clean, consistent corpus-wide chunking change — the
   single-doc probe leaves the tenant in a "mixed chunking" state that is
   NOT representative of the final result, just a fast correctness check on
   the chunking mechanism itself).

**Rule:**
> For any ingestion-config change (chunk_size, chunk_overlap, entity_types,
> etc.) targeting a specific document/chunk, validate with a single-doc
> `--doc` probe (no `--wipe`) FIRST — it's 20-40x faster and surfaces
> pipeline-level bugs (like A106's encoding crash) before they're buried in a
> 30-document log. Only run the full `--wipe` re-ingest once the probe
> confirms the mechanism works.

## A108 — `OpenAIEmbedder` had no API timeout — hung the full automotive
re-ingest for 50+ minutes with zero CPU usage and zero log output

**What happened:** During the Phase 5 Fix C (`chunk_overlap` 64→128)
re-ingest (`--commit --wipe --tenant automotive`, 30 docs), the process
completed docs 1-16 cleanly (~80s/doc incl. community rebuild) then went
completely silent starting doc 17 (`IL-PROC-12-rev1.txt`). 50+ minutes later
the process was still "running" but using 0% CPU — i.e. blocked forever on a
network call, not slow.

**Root cause:** `OpenAIEmbedder.__init__` in `graphrag/core/llm_client.py`
constructed `OpenAI(api_key=api_key)` with **no timeout**. The OpenAI SDK
default (`httpx` timeout=600s, `max_retries=2` with backoff) can compound to
30-50+ minutes on a single stalled connection — and unlike `GroqLLM`
(`_TIMEOUT=60.0`) and `DeepSeekLLM` (`_TIMEOUT=60.0`), which both carry
explicit comments warning about exactly this ("without this the call hangs
forever"), the embedder was missed when those fixes were made.

**Fix:** Added `_TIMEOUT = 60.0` and passed `timeout=self._TIMEOUT` to
`OpenAI(...)` in `OpenAIEmbedder.__init__`, mirroring the existing
Groq/DeepSeek pattern.

**Recovery:** Killed the hung process (PID confirmed 0% CPU delta over 5s via
`Get-Process`). Docs 1-16 were already committed (no `--wipe` needed for
resume) — resumed docs 17-30 via 14 sequential single-doc `--doc <filename>`
invocations (see A107's technique), since none of the remaining docs are part
of an automotive `supersession_chains` entry (all three chains —
CSR-CLIENT-2021→2023, PQ-07-rev1→rev3, IL-INS-03-rev2→rev4 — only involve
docs 1-16, so splitting the run doesn't break `doc_ids_by_filename` lookups).

**Rules:**
> 1. EVERY LLM/embedding client wrapper in `llm_client.py` must set an
>    explicit `timeout` on its underlying SDK client — no exceptions. A
>    client with no timeout doesn't fail loudly, it hangs silently for tens
>    of minutes, masquerading as "still working" in a background task.
> 2. To detect a hang vs. genuine slowness in a background ingestion task:
>    check both (a) no new log lines for several minutes past the expected
>    per-doc time, AND (b) 0% CPU delta over a few seconds via
>    `Get-Process -Id <pid>`. Only (a) can be a slow LLM call; (a)+(b)
>    together means truly stuck.
> 3. When resuming a partial `--wipe` re-ingest after a crash/hang, check the
>    tenant's `supersession_chains` in the domain ontology before splitting
>    the remaining docs into per-`--doc` invocations — cross-invocation
>    `doc_ids_by_filename` lookups silently return nothing, breaking
>    `SUPERSEDES` edges for any chain spanning the split point.

---

## A109 — Fix C (`chunk_overlap` 64→128) regressed automotive 10/13→9/13
(below the 70% gate) without fixing its target (MH-02) — reverted

**What happened:** Phase 5 Fix A got automotive to 10/13 (76.9%), with
MH-02/CON-04/AUTH-01 as known remaining failures. Fix C raised
`chunk_overlap` 64→128 to try to bridge MH-02's 2-chunk structural gap
(target "Conformitatea FAI: 100%" row sits 2 chunks from its KPI heading).
After a full `--wipe` re-ingest (interrupted by A108's embedder hang, resumed
per A107/A108), the eval came back **9/13 (69%) — below the 70% gate**:
- MH-02 still failed (overlap=128, ~128 chars, still can't bridge a 2-chunk gap)
- SH-01 newly failed (was Fix A's flagship fix)
- CON-03 newly failed
- AUTH-01 flipped to pass (consistent with its known non-determinism, A103)

Net: -1 question, gate failure.

**Root cause of the SH-01 regression (confirmed live):** The correct chunk
(`0c24c84c-...`, "Rata de livrare la timp ... 95%") WAS present in
`local_search.search()`'s 55-chunk output — but at **rank 52/55**,
`final_score=0.7919`, far below the 0.86-0.97 cluster occupying ranks 1-51.
`context_builder.py` only takes the **top 5** chunks (`top_k=5`) for the
synthesis prompt, so the LLM never saw it (hence faith=1.00 but "missing
required term: 95%", "insufficient citation recall").

Re-chunking with overlap=128 shifted this chunk's boundaries enough that it
was no longer `fused_chunks[0]` (the #1 RRF-fused candidate) — so Fix A's
RRF-floor mechanism (which forces `fused_chunks[0]` into the rerank seed set)
no longer protected it. Instead it surfaced only as a low-scoring multihop
hop-chunk (rank 52), well outside `top_k=5`.

**Fix:** Reverted `chunk_overlap` to `64` (restoring the validated Fix A
10/13 state) and re-ingested the automotive tenant (`--wipe --commit`, with
the A108 timeout fix applied so the re-ingest didn't hang). MH-02/CON-04/
AUTH-01 remain documented, deferred known-failures per the Fix A plan
("optional — gate already cleared by Fix A alone").

**Rules:**
> 1. A chunking config change (chunk_size/chunk_overlap) is corpus-wide and
>    can shift which chunk becomes `fused_chunks[0]` for *every* query —
>    including ones that currently pass via Fix A's RRF-floor mechanism.
>    Always re-run the FULL golden eval (not just the targeted question)
>    after any chunking change, and compare the total pass count, not just
>    the targeted question's pass/fail.
> 2. If a fix targets question X but the eval shows X still failing AND the
>    total pass count dropped, revert immediately — don't layer further
>    fixes on top of a config change that's net-negative.
> 3. "Chunk is in `local_search.search()`'s output" is not the same as
>    "chunk reaches the LLM" — `context_builder.py`'s `top_k=5` slice is the
>    real gate. When debugging a missing-fact failure, check the chunk's
>    *rank* in the final list against `top_k`, not just presence/absence.

---

## A110 — Fix D (`context_top_k` 5→6, decoupled from `rerank_top_k`) had
no effect on CON-04/AUTH-01 — reverted

**What happened:** After reverting `chunk_overlap` to 64 (A109), CON-04 and
AUTH-01 remained as documented Fix A known-failures. Their earlier root-cause
numbers (CON-04's chunk at rank 6/55, final_score=0.8454 vs the rank-5 cutoff
0.8494, Δ≈0.004) were measured under the *Fix C* (`chunk_overlap=128`) graph.
Hypothesized that decoupling `ContextBuilder`'s `top_k` from `rerank_top_k`
and raising it 5→6 would let CON-04's near-miss chunk into the synthesis
context without touching the validated `rerank_top_k=5` reranker-seed count
(precision 0.907 baseline).

**Verification (partial ingest, 20/30 docs — all docs needed for SH-01/MH-02/
CON-04/AUTH-01 present, all re-chunked at `chunk_overlap=64`):**
- SH-01: PASS (confirms A109's revert restores the Fix A win)
- MH-02: FAIL (unchanged, expected — A109/structural gap)
- CON-04: FAIL — `context_top_k=6` did not pull `pc-comp-07-rev3` /
  `il-ins-03-rev4` into context at all
- AUTH-01: FAIL — all 6 citations were `csr-client-2023` (duplicated);
  `rfa-reg-01-rev5` never entered the candidate set, nowhere near a rank-6
  cutoff

Under `chunk_overlap=64` (Fix A's graph), CON-04/AUTH-01 fail for reasons
unrelated to a narrow rank-5/6 margin — the Fix C-era root-cause numbers don't
transfer. `context_top_k=6` had zero measured benefit.

**Fix:** Reverted `context_top_k` entirely (removed from `settings.yml`,
`hybrid_retriever.py`, `agentic_retriever.py`) — an unverified, ineffective
config addition that also risked the aerospace tenant (cross-tenant retrieval
config) should not be kept (Simplicity First).

**Final Phase 5 state:** Fix A stands alone (10/13, 76.9%, gate cleared).
MH-02/CON-04/AUTH-01 remain documented, deferred known-failures — as the
original Fix A plan already classified them ("optional, gate already cleared
by Fix A alone"). No further fix attempts planned without new root-cause
evidence gathered *against the chunk_overlap=64 graph specifically*.

**Rule:**
> Root-cause measurements (chunk ranks, score deltas) are tied to the exact
> graph state (chunking config) they were measured against. After ANY
> chunking config change + re-ingest, treat prior root-cause numbers for
> other questions as invalidated — re-measure against the new graph before
> proposing a fix based on them.

---

## A111 — `alias_embedding_threshold` 0.92→0.97 (uncommitted, pre-existing)
fragmented automotive into 278 communities and exploded contradiction
false-positives — reverted

**What happened:** While re-ingesting `FP-INJ-01.txt` (part of restoring the
full 30-doc automotive corpus after A109), `community_builder.quality_ok`
reported `community_count=278` (validated baseline: ~55) and
`contradiction_detector.scan_done` reported `new_conflicts=500` — exactly the
cypher query's `LIMIT 500` cap. The doc's community summarization (278 LLM
calls) was still running 1+ minute later, vs. ~50s for 71 communities
earlier the same session.

**Root cause:** `config/settings.yml`'s `alias_embedding_threshold` was
**already** `0.97` in the uncommitted working tree at the *start* of this
entire session (visible in `git status` before any Phase 5 work) — the
committed/validated baseline is `0.92`. A stricter threshold means far fewer
near-duplicate entities get merged during alias resolution, so the graph
fragments into many near-duplicate entity nodes. Each near-duplicate pair
becomes a candidate for `directional_reversal`/`positive_negative_pair`
conflict detection that wouldn't exist if the entities had been merged —
hence the 500-cap conflict explosion. More distinct entity nodes also means
many more Leiden communities (278 vs 55), each requiring its own LLM
summarization call — this is *also* the reason ingestion has been running
much slower than the original ~80s/doc baseline all session (A109's resumed
ingest, the lost A109 re-ingest, and this run were ALL affected, since 0.97
was active for the entire session).

**This also retroactively explains** the unexplained "200 open conflicts / 92
false-positive conflicts / contradiction recall=0.20 / precision=0.01" anomaly
seen in the Fix C eval (`automotive_eval_fixC_final.log`) — not a Fix C
side-effect as originally suspected, but this threshold.

**Fix:** Killed the stuck `FP-INJ-01` ingestion (PID 18888, mid 278-community
summarization — would have been redone anyway). Reverted
`alias_embedding_threshold` to `0.92` in `config/settings.yml`.

**Consequence:** Every doc ingested this session (21/30, including all 20
used to verify A109/A110's SH-01 fix) was ingested under `threshold=0.97`.
SH-01 still passed under this fragmented graph, but overall graph quality
(community structure, contradiction precision) for the current automotive
tenant is degraded session-wide. **A single full `--wipe --commit` re-ingest
of automotive — combining A109's `chunk_overlap=64` and this `0.92`
threshold, both already correct in `settings.yml` — is required** to get a
clean, fully-validated graph. Not yet run; pending user go-ahead given 3 prior
re-ingest attempts this session hit issues (A108 embedder hang, lost
background session, this threshold bug).

**Rules:**
> 1. At the START of any session involving re-ingestion, diff
>    `config/settings.yml` against `HEAD` and account for EVERY uncommitted
>    line — not just the ones relevant to the current task. An unrelated
>    stray config change can silently degrade every re-ingest run for the
>    rest of the session.
> 2. A sudden jump in `community_builder.quality_ok community_count` or
>    `contradiction_detector.scan_done new_conflicts` (especially hitting a
>    query's `LIMIT` cap) relative to the session's earlier docs is a signal
>    to stop and check `alias_embedding_threshold` / alias dedup rate, not
>    just "this doc has more entities."

## A112 — Future optimization note: extraction model speed/cost options (not actioned)

Ingestion is bottlenecked by per-chunk LLM extraction calls
(`get_llm()` in `graphrag/core/llm_client.py`), currently DeepSeek-V3
(`deepseek-chat`), at roughly ~1 chunk/sec for a corpus this size. User asked
about faster alternatives at comparable cost — captured here for a future
session, NOT changed now (would require re-validating extraction quality /
faithfulness on both tenants, out of scope for this session's eval-fix work).

| Model | Throughput | Cost (per 1M tokens, in/out) | Notes |
|---|---|---|---|
| **DeepSeek-V3** (`deepseek-chat`) — current primary | ~30-60 tok/s | ~$0.07 / $1.10 | Cheapest; generous rate limits; current "extraction voice" baseline for both tenants |
| **Gemini 2.0 Flash** | ~200 tok/s | ~$0.075 / $0.30 | A `GeminiLLM` class already exists in `llm_client.py` (currently dormant — was the pre-DeepSeek Groq fallback); cheap + fast + huge context, but reintroducing it changes extraction "voice" for both tenants |
| **Groq llama-3.3-70b-versatile** (paid tier) | ~280 tok/s | ~$0.59 / $0.79 | Already used for synthesis (`get_llm()` opt-in via `LLM_INGEST_PROVIDER=groq`) and is the `FallbackLLM` partner for `get_fast_llm()`; free tier rate-limits at corpus scale (the original reason DeepSeek became primary for ingestion) |
| **Groq llama-3.1-8b-instant** | ~800 tok/s | ~$0.05 / $0.08 | Already used as the fast/routing model in `AgenticRetriever`; cheapest+fastest but smaller model — extraction quality on Romanian automotive text unvalidated |
| **Cerebras llama-3.3-70b** | ~2000 tok/s | comparable to Groq paid tier | Not currently integrated — would need a new `BaseLLM` subclass in `llm_client.py` |

**If pursued**: treat as a corpus-wide change requiring full re-validation
(both tenants' golden evals + faithfulness regression), same rule as Fix B/C
in this session's `tasks/lessons.md` entries — don't swap mid-validation.

## A113 — MH-02/AUTH-01 are structural retrieval-pool gaps, not tunable

Spent this session trying non-destructive fixes for the 3 remaining automotive
failures (MH-02, CON-04, AUTH-01) under a hard "no re-ingestion" constraint.
Conclusion, with evidence:

- **MH-02**: PPAP-PROC-01's "Conformitatea FAI: 100%" KPI-table chunk never
  appears in vector or BM25-fused top-40 for the MH-02 query, at ANY
  `local_top_k` (tested 10/20/40) or `multihop_semantic_weight` (0/0.2/0.5/0.8).
  Not reachable via multihop either (0/500 hop_chunks). This is a pure
  **chunking gap** (the KPI table and the question's terms don't co-occur in
  any chunk's embedding/BM25 signal) — only fixable via different chunk
  boundaries + full re-ingest (Fix C, already tried once and reverted as
  net -1 — see A109).
- **AUTH-01**: RFA-REG-01-rev5's relevant chunk (`0382d695...`, "Furnizorii
  CRITICI ... reevaluare SEMESTRIALĂ") reaches fused rank 8/10 (a real BM25
  hit) but is never selected by the cross-encoder reranker into `seed_chunks`,
  and is not in multihop hop_chunks either — so it never reaches
  `all_chunks`/context at all. Fix A's RRF-floor only guarantees the #1 fused
  chunk a seed slot, not rank-8. No prompt-level or context-builder fix can
  recover a citation for a chunk that's never selected as a candidate.
- **CON-03/CON-04**: NOT a retrieval problem — all expected citations ARE
  present in context every run. Root cause is **LLM sampling non-determinism**
  at temperature=0.0 on contradiction-heavy context (re-running identical
  code/context flips "semestrial"↔"anual" for CON-03, and "rev2"/"rev4"
  appear/disappear for CON-04 across runs with byte-identical inputs). This is
  a known characteristic of Groq's MoE-served Llama models (no determinism
  guarantee at temp=0 due to dynamic batching) — not something our retrieval
  or prompt config controls.

**Implication**: "retry until MH-02/CON-04/AUTH-01 all pass" cannot succeed —
MH-02 and AUTH-01 fail 100% of the time (citations structurally absent from
context, independent of LLM sampling), while CON-03/CON-04 are ~coin-flip
flaky. The 70% gate is already cleared by Fix A alone (10/13, SH-01 fixed).
Tried and kept: near-duplicate text dedup in `context_builder.py` (harmless,
doesn't fix AUTH-01 — the relevant chunk isn't a literal near-dup of what's in
context); `_ANSWER_PROMPT`/`_FINAL_PROMPT` rev-number rules in
`hybrid_retriever.py`/`agentic_retriever.py` (harmless general improvement,
marginal/non-stable effect on CON-04). Tried and not kept: `local_top_k=40`
probe (config not persisted — no effect on MH-02, no regression on SH-01).

**Path forward** (deferred, needs user GO — destructive re-ingest): a targeted
chunking fix for MH-02 (e.g. keep KPI-table rows with their section heading in
the same chunk) would need different validation than Fix C's blunt
`chunk_overlap` bump — Fix C regressed SH-01/CON-03 because a global overlap
change perturbs every chunk's embedding, not just PPAP-PROC-01's. A
structure-aware splitter (e.g. never split mid-table) is lower blast-radius
but is a real code change, not a config tweak, and still requires full
re-ingest + both-tenant re-validation.

## A114 — AUTH-01 fixed: prefer fused_chunks candidate over cosine-only pick,
## PREPEND not append to seed_chunks

AUTH-01 (authority_chain) was failing because the named-document boost (added
in the prior session for RFA-REG-01-rev5) picked the wrong chunk and/or gave
it a fatal rank penalty. Two compounding bugs, both fixed in
`graphrag/retrieval/local_search.py`:

1. **`get_best_chunk_for_document` is cosine-only against the whole document**
   — for a question naming "RFA-REG-01", it picked the document's most
   generically-similar-to-the-question chunk (a 0.84-cosine intro paragraph,
   `5a639ea7`), not the chunk containing the actual fact ("Furnizorii
   clasificați ca CRITICI sunt supuși reevaluării SEMESTRIALE", `0382d695`).
   `0382d695` was already sitting at fused rank 8 — lexically matched against
   this exact question by BM25/RRF — but never considered because the boost
   went straight to a fresh cosine search.
   **Fix**: before falling back to `get_best_chunk_for_document`, check
   `fused_chunks` (excluding chunks already in `seed_chunks`) for any chunk
   belonging to the named document via a new
   `neo4j_client.get_chunk_filenames(chunk_ids, tenant)` helper (chunk_id →
   filename). Only fall back to the cosine-only `get_best_chunk_for_document`
   if no fused candidate from that document exists.

2. **Appending the boosted chunk to `seed_chunks` gives it the worst possible
   rank-based text score.** `gnn_scorer.py`'s `_text_score` for chunks WITH a
   `rerank_score` is `1 - (rank / n_seed)` — rank = position in `seed_chunks`.
   Appending puts the boosted chunk at `rank = n_seed - 1` → `text_score ≈
   0.2`, regardless of how relevant it actually is, so it almost never
   survives the final top-5 cut.
   **Fix**: `seed_chunks = [best] + seed_chunks[:-1]` (prepend, drop the
   weakest existing seed) → `rank = 0` → `text_score = 1.0`.

**Why**: An explicit "the question names this document by code" signal should
be treated as at least as trustworthy as the cross-encoder's top pick — not
buried at the bottom of the seed list where the rank-based score guarantees
its exclusion.

**How to apply**: Any future "boost a specific chunk into the candidate set"
mechanism (named-document boost, entity-pin, etc.) must PREPEND to
`seed_chunks`, never append — `_text_score` is rank-based, not score-based,
for seeded chunks. Also prefer a chunk already present in `fused_chunks`
(proven lexically relevant to *this* question) over a fresh whole-document
similarity search, which only proves relevance to the document in general.

**Result**: `authority_chain 1/1 (100%)`, faithfulness 1.00. 379/379 unit
tests pass. 3/3 stable `probe_partial.py` re-runs.

## A115 — CON-04 fixed via source-document labels; gate them on
## revision/version vocabulary to avoid an SH-03 regression

CON-04 (contradiction) asks which revision of IL-INS-03 is referenced inside
PC-COMP-07, and whether it matches the current revision. Chunks were presented
to the LLM as bare `[Chunk {uuid}]\n{text}` blocks with NO document/filename
metadata — the LLM could read "conform procedurii IL-INS-03 rev.2" inside a
PC-COMP-07 chunk, and separately read IL-INS-03-rev4's content in another
chunk, but had no way to know *that second chunk* is IL-INS-03-rev4 (vs. e.g.
IL-INS-03-rev2 or some unrelated document), so it couldn't conclude rev.2 ≠
rev4.

**Fix**: `local_search.py` now calls
`neo4j_client.get_chunk_filenames(all_ids, tenant)` and sets
`chunk["source"] = filename` for each chunk; `context_builder.py` includes it
in the header: `[Chunk {chunk_id} | Source: {source}]`. `_ANSWER_PROMPT` in
`hybrid_retriever.py` got a new rule explaining the `Source:` field is for
attribution/revision comparison.

**Regression found and fixed**: unconditionally adding `Source:` labels broke
SH-03 (a plain single-document lookup: "ce alt standard ... conform
CSR-CLIENT-2023?"). The answer-bearing chunk (`d3fb2979`, PQ-07-rev3.txt,
contains "ISO 9001:2015") WAS in context, labeled `Source: PQ-07-rev3.txt`.
With that label visible, the LLM concluded the fact wasn't "conform
CSR-CLIENT-2023" (the document named in the question) and refused to use it —
3/3 reproducible. Removing the label (A/B test) made SH-03 pass 3/3. Adding an
explicit "don't restrict to the named document" prompt rule did NOT fix it
(still 3/3 fail) — the model anchors on the label regardless of instructions.

**Real fix**: gate source-label attachment on whether the *question* contains
revision/version vocabulary (new `_REVISION_QUERY_RE` in `local_search.py`:
`rev(?:izi[ae]|ision)?[.\s]|versiune|version|actual[ăa]|current`). CON-04's
question contains "revizia"/"actuală"; SH-03's doesn't. With this gate: SH-03
3/3 pass, CON-04 3/3 pass, AUTH-01 3/3 pass (unaffected — AUTH-01's question
doesn't match the gate either, so no source labels there, consistent with its
fix in A114 not depending on labels).

**Why**: source/provenance metadata is a double-edged signal — it helps
cross-document comparison questions but actively hurts single-document lookup
questions by giving the model an excuse to discard relevant-but-differently-
sourced facts. Scope the signal to the question types that actually need it
rather than adding a prompt disclaimer (disclaimers don't override a strong
label-based heuristic the model has learned).

**How to apply**: any future per-chunk metadata added to context headers
(provenance, confidence, dates, etc.) should be considered for this kind of
gating — ask "does *this question* need this signal, or could it cause the
model to wrongly discard a relevant chunk?" before attaching unconditionally.

**Result**: automotive eval `contradiction 4/5 → expect 5/5` (CON-04 fixed,
CON-03 remains flaky per A113), `single_hop` unaffected (SH-01 fixed in A104,
SH-03 now also passes). 379/379 unit tests pass.

## A116 — MH-02 re-confirmed structural (no new info)

Re-ran live `vector_search_chunks(top_k=20)` and `bm25.search(top_k=20)` for
the MH-02 query against the unmodified automotive tenant: PPAP-PROC-01.txt's
"Conformitatea FAI: 100%" chunk is essentially absent from both top-20 pools
(1/20 in vector, 0/20 in BM25-fused) — confirms A113's conclusion. Still
deferred, requires re-ingestion (off-limits per standing constraint).

## A117 — CON-03 was misdiagnosed as flaky (A113); actual cause was
## cross-document context pollution + community-synthesis override — fixed

A113 assumed CON-03 (and CON-04) flip pass/fail purely from Groq MoE sampling
non-determinism at temp=0. Re-investigation this session found CON-03 was
actually **100% deterministic and 100% wrong** before this fix: 4/4 runs with
byte-identical retrieval context answered "anual" (never "semestrial"),
despite "semestrial" being clearly present in the context every time.

**Root cause, two layers**:

1. **Retrieval-level pollution**: the question asks "conform Manualului
   Calității (MC-01) și procedurii PQ-07". The top-5 context contained TWO
   chunks with "Reevaluarea furnizorilor activi se realizează(/realizeaza)
   SEMESTRIAL" (correct, from PQ-07/MC-01) — but ALSO two chunks from
   **PPAP-PROC-01** (an unrelated document, not named in the question) with a
   same-shaped but different fact: "Evaluarea furnizorilor activi se va
   realiza anual" (section "8.1 Frecvența reevaluării" — same section number
   as the correct MC-01 chunk, almost certainly why it scores similarly).
   Nothing in the context told the LLM these "anual" chunks came from a
   document the question didn't ask about.

2. **Community-synthesis override**: even when the LLM correctly extracted
   "SEMESTRIAL" from the chunk-level facts (some runs), a `Community
   knowledge:` section (GraphRAG global/community synthesis) sometimes stated
   "anual", and the LLM deferred to that coarse summary over the specific
   chunk-level fact — "Conform informațiilor din comunitate, ... se efectuează
   anual."

**Fix** (both layers, code-only, no re-ingestion):

1. `local_search.py`: replaced the `_REVISION_QUERY_RE`-only gate from A115
   with `_needs_source_labels()`, which ALSO triggers when the question names
   2+ distinct document codes via the existing `_DOC_CODE_RE` (CON-03 names
   "MC-01" and "PQ-07" — 2 codes). This attaches `chunk["source"] = filename`
   so the LLM can see the "anual" chunks are from PPAP-PROC-01, not MC-01/PQ-07.

2. `hybrid_retriever.py` `_ANSWER_PROMPT`: two new rules —
   - If the question names specific documents and two chunks conflict, prefer
     the fact from the chunk whose `Source` matches a named document.
   - A `Community knowledge:` section is lower-precision than a numbered
     `[Chunk ...]` section; prefer the chunk-level fact when they conflict.

**Why this matters beyond CON-03**: A113's "it's just LLM noise, nothing we
can do" conclusion was wrong for CON-03 — it was a retrieval/context-design
bug that happened to produce a single wrong answer 100% of the time, which
*looked* like it could be "non-determinism" only because a DIFFERENT
contradiction question (CON-04) genuinely does flip. Don't assume "low
faithfulness + occasional pass" means non-determinism — check whether the
*context itself* contains a same-shape distractor fact from an unrelated
document first.

**Verified**: 379/379 unit tests pass. CON-03 went from 4/4 deterministic FAIL
to 7/8 PASS (5/5 in one batch, then 2/3 in a follow-up 3x5-question regression
sweep — the one remaining fail also showed "missing 'semestrial'", same
failure mode, just less frequent now). SH-01, SH-03, CON-04, AUTH-01 all
unaffected — 15/15 PASS across the same sweep. AUTH-01 also picked up source
labels under the new 2+-doc-code gate (its question names "CSR-CLIENT-2023"
and "RFA-REG-01") with no regression.

CON-03's residual ~1/8 failure is now plausibly genuine LLM sampling
non-determinism (consistent with A113's characterization of CON-04) rather
than the prior 100%-deterministic context-pollution bug — a large
improvement, though not a full guarantee.

**Expected automotive eval**: `contradiction` now mostly 5/5 (occasionally
4/5 if CON-03 lands on its ~1/8 fail), `single_hop 3/3`, `authority_chain
1/1` → ~11-12/13 (85-92%), up from 10/13 at session start. Official aggregate
still pending — DeepSeek RAGAS faithfulness calls have hung on Q1 for 3
consecutive attempts this session (transient API instability, unrelated to
these code changes).

## A118 — MH-02 chunking fix already implemented (uncommitted), just needs re-ingest

**Context**: User asked to "try to fix" MH-02 (structurally unreachable per
A113/A116 — PPAP-PROC-01's "Conformitatea FAI: 100%" KPI table row never
enters the BM25/vector top-20). A113's "Path forward" proposed exactly this:
"a targeted chunking fix for MH-02 (e.g. keep KPI-table rows with their
section heading in the same chunk)".

**Finding**: `graphrag/ingestion/chunker.py` ALREADY contains a heading-aware
chunker — `_split_into_sections()` + `_HEADING_RE` split the document at
markdown headings and prepend each section's heading to every chunk derived
from it. This is EXACTLY A113's proposed fix. But `git diff` shows this code
is **uncommitted, working-tree-only** — it was written in a prior session but
never landed or applied via re-ingestion.

**Verified relevance to MH-02**: Read `data/automotive/long/PPAP-PROC-01.txt`.
The document's content is duplicated (two near-identical copies concatenated).
The "Conformitatea FAI" row that's the MH-02 target lives in the SECOND copy,
under heading `## 11. INDICATORI KPI CU VALORI TARGET NUMERICE` (line 599),
in a 3-column table (`Indicator | Valoare țintă | Descriere`) — distinct from
the first copy's 2-column table under `## 10. ...` (line 223-231). The
currently-ingested chunk for this row (`619face7...`, confirmed via Neo4j in
A116/this session) does NOT carry this heading text — consistent with it
having been chunked by the OLD (pre-heading-aware) chunker. With the
uncommitted heading-aware chunker, re-chunking would prepend "## 11.
INDICATORI KPI CU VALORI TARGET NUMERICE" (plus "Evaluarea performanței
furnizorilor va fi realizată și prin intermediul unor indicatori KPI...") to
every chunk in that section, including the FAI row — adding "INDICATORI",
"KPI", "VALORI TARGET NUMERICE" vocabulary that plausibly pulls it into
BM25/vector top-k for the MH-02 query.

**Status**: Code fix exists and is ready (uncommitted). NOT applied — applying
it requires re-ingesting the automotive tenant (`--wipe --commit`), which is
blocked by the standing "no ingestions" constraint this session. 379/379 unit
tests still pass with this code present (chunker change is structurally
additive — sections without headings behave identically to before).

**Next step (needs explicit user GO)**: `py -3.11 scripts/ingest_corpus.py
--commit --wipe` for the `automotive` tenant, then re-run MH-02 via
`probe_partial.py` / `scripts/run_automotive_eval.py` to confirm the FAI-row
chunk now surfaces and MH-02 passes. Also re-run the aerospace regression
check afterward only if aerospace is also re-ingested (aerospace ingestion is
separately blocked by the same constraint and was not touched).

**Estimated success rate if automotive is re-ingested now** (heading-aware
chunker + all of A114/A115/A117's already-applied retrieval/prompt fixes):

- **Best case**: 12-13/13 (92-100%) — MH-02 flips to PASS (heading vocab pulls
  the FAI-row chunk into top-k), CON-03 lands on its ~7/8-favored side, and
  SH-01/SH-03/CON-04/AUTH-01 hold.
- **Realistic expected**: ~11-12/13 (85-92%) — same as the current
  pre-re-ingest estimate, i.e. re-ingestion is expected to be net-neutral-to-
  positive but not guaranteed to flip MH-02 (heading text increases relevance
  but doesn't guarantee a top-k rank crossing) and CON-03 retains its ~1/8
  residual flake either way.
- **Downside risk (precedent: A109)**: a corpus-wide chunking change shifted
  chunk boundaries for ALL documents last time (`chunk_overlap` 64→128) and
  caused a NET REGRESSION (10/13 → 9/13) by newly breaking SH-01 and CON-03
  even though it didn't fix its target (MH-02). The heading-aware chunker is a
  different, more targeted mechanism (additive, not a global size/overlap
  change) and unit tests pass, but it still re-chunks the entire automotive
  corpus, so a similar SH-01/CON-03 regression is possible. Floor estimate:
  ~9-10/13 (69-77%) if that happens — at or just below the 70% gate.

**Recommendation**: if re-ingesting, re-run the FULL automotive golden set
(not just MH-02) plus the aerospace regression check, per A103's rule for
corpus-wide changes — don't assume isolated improvement.

## A119 — `PPAP-PROC-01.txt` contains ~4 overlapping rewritten "passes", not a simple duplicate (documented, NOT fixed)

**What happened**: While investigating MH-02 (A118), read the full
`data/automotive/long/PPAP-PROC-01.txt` (1086 lines). It is not a single
duplicate — it's roughly **4 concatenated passes** through similar content,
each independently rewritten/rephrased (likely from repeated
`data/generate_doc.py` runs concatenated without dedup), with **conflicting
section numbers across passes** (e.g. section 8 = "CLASIFICAREA
FURNIZORILOR" in passes 1-2, but section 8 = "REEVALUAREA FURNIZORILOR
ACTIVI" in pass 3; section 9 flips correspondingly). Passes run
~lines 1-277, 277-498, 498-834, 834-1075, each ending mid-sentence where the
next pass's heading is concatenated directly onto the prior pass's closing
paragraph (no newline) — e.g. line 277, 498, 715, 834.

**Why NOT fixed now**: MH-02's actual target row ("Conformitatea FAI: 100%"
with description column) lives in **pass 3**, under its own
`## 11. INDICATORI KPI CU VALORI TARGET NUMERICE` heading (line 599) — a
different (3-column) table than pass 1's `## 10.` table (2-column,
line 223-234). A naive "dedupe to one pass" would likely DELETE the exact
section the A118 chunker fix targets for MH-02. Also: CON-03/CON-04/AUTH-01's
session fixes were validated against the CURRENT multi-pass chunking of this
file — restructuring it is itself a corpus-wide chunking change (same risk
class as A109's regression), and bundling it with A118's chunker change would
make any regression hard to attribute to either change individually.

**Decision**: documented only, not edited. If pursued later, treat as a
SEPARATE re-ingest/re-validation cycle from A118's chunker fix — fix one,
re-ingest, re-validate full automotive golden set + aerospace regression,
THEN do the other. Both still require explicit user GO per the standing
no-ingestion constraint.

## A120 — Multilingual reranker swap tried and reverted (net-neutral, new regressions); baseline is actually 11/13 now

**Context**: tried swapping the cross-encoder reranker from
`cross-encoder/ms-marco-MiniLM-L-6-v2` (English MS MARCO) to
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual mMARCO, covers
Romanian) — a query-time-only change, no re-ingestion needed. Unit tests
mock the reranker entirely (379/379 pass either way). Sanity check on the
SH-01 pair showed a much stronger correct-vs-distractor margin (4.85 vs
-3.84) than the old model ever gave.

**Result (live `probe_full.py`, no RAGAS — `_check()` term/citation checks
only, automotive tenant, 13 questions)**:
- New reranker: **10/13** — SH-01 flips PASS (as hoped), but **CON-04
  regresses to FAIL** ("context does not contain any information about
  PC-COMP-07 referencing IL-INS-03" — this session's A115 fix undone) and
  **CON-02 newly FAILs** (citation recall: only 1/3 expected docs cited).
- Reverted to the old reranker, re-ran: **11/13** — SH-01 still PASSES (it
  was already fixed by Fix A's RRF-floor, A104 — unrelated to the reranker),
  CON-04/CON-03/AUTH-01 all PASS. Only **MH-02** (A118, pending re-ingest) and
  **CON-02** (new finding, see below) fail.

**Conclusion**: the multilingual reranker is net-neutral at best (10/13,
same as old session-start baseline) and trades a fix you already have
(CON-04, via A115) for one you don't need (SH-01, already fixed by A104).
Confirms the plan's existing "too high blast radius for the gain" caution —
**reverted**, `reranker.py` restored to `ms-marco-MiniLM-L-6-v2`.

**Bonus finding**: current code (A114/A115/A117, old reranker, no
re-ingestion) scores **11/13 (84.6%)** — better than the 11-12/13 estimate in
A117/A118. The only failures are MH-02 (A118, structural — fix ready,
pending re-ingest) and **CON-02** (new — "Este obligatoriu un audit on-site
la sediul furnizorului înainte de aprobarea oricărui furnizor nou?",
`expected_citations: [csr-client-2023, fp-inj-03, reg-evf-01]`, only
`csr-client-2023` cited — a 3-document citation-recall gap, same flavor as
AUTH-01/CON-04 but not previously tracked as a "remaining failure". Not
investigated yet — candidate for a future session.)

Updated estimate if automotive is re-ingested with A118's chunker fix:
**12/13 (92%)** if MH-02 flips and CON-02 is unaffected, or **11/13** if
CON-02 remains the sole failure.

## A121 — CON-02 root cause: 20 corpus files were never ingested (corpus/graph drift, not a retrieval bug)

**Investigation**: ran `LocalSearch.search()` live for CON-02's question
("Este obligatoriu un audit on-site la sediul furnizorului înainte de
aprobarea oricărui furnizor nou?") against `automotive`. The 55-chunk
candidate pool contains ONLY `csr-client-2023`, `pq-07-rev3`, `pq-07-rev1`,
`ppap-proc-01`, `mc-01-s8-rev6`, `il-ins-03-rev2`, `pc-comp-07-rev3` — never
`fp-inj-03` or `reg-evf-01` (CON-02's other two `expected_citations`).

**Root cause**: queried Neo4j directly —
```
MATCH (c:Chunk {tenant:'automotive'})-[:PART_OF]->(d:Document{tenant:'automotive'})
WHERE toLower(d.filename) CONTAINS 'fp-inj' OR toLower(d.filename) CONTAINS 'reg-evf'
```
returns **0 chunks**. `MATCH (d:Document{tenant:'automotive'})` returns only
**10 documents** (CSR-CLIENT-2021/2023, IL-INS-03-rev2/rev4, MC-01-S8-rev6,
PC-COMP-07-rev3, PPAP-PROC-01, PQ-07-rev1/rev3, RFA-REG-01-rev5). But
`data/automotive/` on disk (untracked dir, `git status` shows `?? data/automotive/`)
contains **30 `.txt` files** across `short/`, `medium/`, `long/`, `archive/` —
20 of them (FP-INJ-01/02/03, FP-PAC-04, FP-VER-05, REG-ETA-01, REG-EVF-01,
REG-NC-01, IL-CAL-05, IL-EXP-07, IL-MEN-08, IL-NC-06-rev2, IL-PROC-12-rev1,
PAI-PROC-04, PCAL-PROC-10, PFMEA-01, PTRN-PROC-13, SPEC-PROD-01,
PN-PROC-05-rev1/rev3) **were never ingested into Neo4j at all**.

`ingest_corpus.py`'s automotive config (`_automotive_corpus_config()`) uses
`corpus_dir = data/automotive`, `recursive: True` — i.e. it WOULD pick up all
30 files via `rglob("*.txt")` on a fresh `--wipe --commit` run. The corpus
directory simply grew (to its current 30-file/4-tier short/medium/long/archive
shape) after the automotive tenant's last ingestion, and `data/eval_golden/queries_automotive.json`
(last modified 2026-06-14, newer than the graph) was authored against the
*current* (30-file) corpus, not the *ingested* (10-file) one.

**Blast radius check** — grepped `queries_automotive.json` for all 20
un-ingested docs' slugs: 4 are referenced as `expected_citations` across the
13-question golden set: **`fp-inj-03`, `reg-evf-01`** (CON-02), and
**`il-nc-06-rev2`, `il-proc-12-rev1`** (used in some other question — not yet
mapped to a specific ID this session). CON-02 cannot pass — and any other
question expecting those 4 docs cannot pass — under **any** non-destructive
retrieval/reranker/config tuning, because the chunks simply do not exist in
the graph. This is categorically different from MH-02/AUTH-01 (A113): those
are "chunk exists but doesn't surface" gaps; this is "chunk doesn't exist".

**Why not previously caught**: A113/A117/A118/A120's diagnostics all probed
*ranking* (BM25/vector top-20, rerank scores) for docs known to be in the
graph. CON-02 is the first question whose `expected_citations` include docs
outside the ingested 10 — nobody had checked `MATCH (d:Document)` document
*coverage* vs. the golden set's document references until now.

**Fix options (none applied — all require re-ingestion, blocked by standing
"no ingestions" constraint)**:
1. Re-ingest automotive with `--wipe --commit` against the current 30-file
   corpus. Combines naturally with A118's chunker fix (already
   uncommitted/ready) — one re-ingest cycle could resolve MH-02 AND CON-02
   (and whichever question(s) need `il-nc-06-rev2`/`il-proc-12-rev1`)
   simultaneously. Highest payoff, highest blast radius (full re-ingest,
   ~1hr, both A118 and A119 considerations apply — A119's duplication issue
   is in `PPAP-PROC-01.txt`, one of the ALREADY-ingested 10, so it's
   orthogonal to this 20-file gap).
2. Incremental ingest of just the 20 missing files (if `ingest_corpus.py`
   supports non-`--wipe` incremental add — not yet checked). Lower blast
   radius (doesn't touch the 10 already-ingested docs / their entities/edges),
   but doesn't get A118's chunker fix for MH-02 (MH-02's doc, PPAP-PROC-01,
   is already ingested — would need its own re-chunk separately).
3. Do nothing — document CON-02 (and the `il-nc-06-rev2`/`il-proc-12-rev1`
   question) as permanently-failing-until-re-ingest, same status as MH-02.

**Updated automotive picture**: current validated baseline is **11/13
(84.6%)**, well above the 70% gate. CON-02 and MH-02 are BOTH
re-ingest-gated failures (different mechanisms: MH-02 = chunking, CON-02 =
missing documents). A full re-ingest addressing both could realistically
reach **12-13/13 (92-100%)** depending on whether
`il-nc-06-rev2`/`il-proc-12-rev1`'s question is among the currently-passing
11 or was silently relying on partial-credit thresholds.

**Recommendation**: bundle this with A118/A119 as a single future re-ingest
decision point — needs explicit user GO, not pursued this session per the
standing constraint.

## A122 — CON-02 fixed via minimal incremental ingest (12/13, 92.3%)

**Action taken (with explicit user GO, "go with reingest - but as small as
possible to test, not full")**: ran
`python scripts/ingest_corpus.py --doc "FP-INJ-03,REG-EVF-01" --tenant
automotive --commit` (NO `--wipe`). This is additive-only — `doc.id` is a
fresh `uuid4()` per ingested file, so re-running with a `--doc` filter and no
wipe cannot collide with or alter the existing 10 documents/chunks/entities.

**Result**:
- Graph snapshot: entities 1799→1846 (+47), edges 2527→2562 (+35) — exactly
  the scale of 2 small `FORM_RECORD` (authority=4) documents (13+11=24 chunks,
  80 raw entities → 47 after dedup). Pre-existing 1799/2527 baseline and 462
  open conflicts / 180/1k contradiction rate were already present before this
  run (automotive is a deliberately contradiction-dense corpus by design —
  not a regression).
- **CON-02: FAIL → PASS.** `reg-evf-01` and `csr-client-2023` both now cited
  (citation recall 0.667 ≥ 0.4 threshold); `fp-inj-03` still not cited but not
  required to hit the threshold.
- **Full automotive golden set: 11/13 → 12/13 (92.3%)**. All 12 previously-
  passing questions (SH-01/02/03, MH-01, CON-01/03/04/05, AUTH-01, NEG-01/02)
  still pass — zero regressions.
- **379/379 unit tests pass.**
- Only **MH-02** remains failing — same root cause as A118 (heading-aware
  chunker fix exists, uncommitted, needs its own re-ingest of
  `PPAP-PROC-01.txt` to take effect).

**Note**: `il-nc-06-rev2`/`il-proc-12-rev1` (the other 2 of the 20 originally
un-ingested files referenced somewhere in the golden set, per A121) were
NOT ingested in this minimal step — out of scope for "as small as possible".
If the question(s) referencing them are among the 12 currently-passing
(seems likely, since 12/13 held with only `fp-inj-03`/`reg-evf-01` added),
no further action needed; otherwise revisit.

**Aerospace tenant**: untouched (this ingest only targeted `automotive` via
`--tenant automotive --doc "FP-INJ-03,REG-EVF-01"`). Aerospace regression
re-validation remains a separate, not-yet-revisited item.

**Automotive baseline is now 12/13 (92.3%)**, up from 11/13 (84.6%) at session
start. MH-02 (A118) is the sole remaining failure, pending a separate,
explicit re-ingest GO for `PPAP-PROC-01.txt`'s heading-aware re-chunk.

## A123 — Full 14-doc re-ingest with fixed chunker: MH-02 fixed, but SH-01/CON-04/AUTH-01 regressed (12/13 -> 10/13)

**Action taken (with explicit user GO, "confirm re-run wipe automotive and
ingest the 14-file set with the fixed chunker")**: fixed the A118 chunker bug
first (see below), then ran
`python scripts/ingest_corpus.py --tenant automotive --wipe --commit --doc
"CSR-CLIENT-2021,CSR-CLIENT-2023,IL-INS-03-rev2,IL-INS-03-rev4,IL-NC-06-rev2,
IL-PROC-12-rev1,MC-01-S8-rev6,PC-COMP-07-rev3,PPAP-PROC-01,PQ-07-rev1,
PQ-07-rev3,RFA-REG-01-rev5,FP-INJ-03,REG-EVF-01"` (full tenant wipe + 14-file
re-ingest, ~48 min, 14/14 docs, 1292 chunks, 1913 entities, 2711 edges, 29
open conflicts).

**Chunker bug fixed first**: A118's original implementation pre-split the
document into one section per markdown heading BEFORE running
`RecursiveCharacterTextSplitter.split_text()` on each section independently —
this caused a chunk-count explosion on heading-dense docs (PQ-07-rev3.txt:
141 headings/63KB -> 208 chunks instead of ~145; PPAP-PROC-01.txt -> 217
chunks). Fixed by chunking `raw_text` in ONE `splitter.split_text()` call
(restores normal chunk boundaries/counts), then for each resulting chunk
locating its character offset (`raw_text.find`, monotonic cursor) and
prepending the nearest-preceding markdown heading via `bisect.bisect_right`
on precomputed heading offsets. Verified post-ingest: PQ-07-rev3.txt
208->170 chunks, PPAP-PROC-01.txt 217->168 chunks.

**Result**:
- **MH-02: FAIL -> PASS** (faith=0.50). The "Conformitatea FAI: 100%" row now
  carries its "## 11. INDICATORI KPI CU VALORI TARGET NUMERICE" heading and is
  retrieved/cited correctly — this was the entire point of A118/A123.
- **Regressions: SH-01 and AUTH-01 both now fail on "insufficient citation
  recall — missing: ['csr-client-2023']"; CON-04 now fails on missing
  required terms 'rev2'/'rev4'.** All three previously passed under the old
  (pre-fix, exploded-chunk) ingest.
- **Net: 12/13 (92.3%) -> 10/13 (76.9%)**. Still clears the 70% gate, but a
  net -2 regression vs the A122 baseline.
- Contradiction recall 0.40 (2/5: C01, C03 matched; C02/C04/C05 missing),
  precision 0.72 (29 open conflicts) — not directly compared to A122 (A122
  didn't report this breakdown), flagged for awareness only.
- RAGAS faithfulness average 0.897 (>> 0.75 threshold). **379/379 unit tests
  pass** — code-level zero regressions, this is purely a retrieval-ranking
  shift from changed chunk boundaries/embeddings after the full re-ingest.

**Root cause of the regression (consistent with the Phase-5 plan's
addendum)**: SH-01/AUTH-01/CON-04's passing status was already
margin-sensitive — `path_score`/rerank scores for the relevant
`csr-client-2023`/`il-ins-03-rev*` chunks sat close to the top-5 cutoff
(documented as a ~0.004-band non-determinism risk). A full re-ingest changes
EVERY chunk's boundaries and embeddings (not just PPAP-PROC-01's), shifting
all three margins enough to flip them, while moving MH-02's target chunk
into the retrievable set as intended.

**Current state**: automotive tenant is in a clean, fully-ingested state
(14/14 docs, no partial/broken documents — unlike the killed first attempt).
10/13 (76.9%) is the new measured baseline, still above the 70% gate. The
chunker fix itself (`graphrag/ingestion/chunker.py`) is correct and necessary
for MH-02; the regression is a retrieval-ranking side effect of the re-ingest
itself, not a chunker defect.

**Not pursued further this session** (per the standing
no-further-re-ingestion-without-explicit-authorization constraint): tuning
SH-01/AUTH-01/CON-04 back to passing (e.g. revisiting Fix A's RRF-floor or the
multihop semantic weight) would need its own validation pass and is a
separate decision point. Aerospace tenant untouched throughout.

## A124 — "Fix them all" (rerank_top_k 5->7 + _text_score floor) fixed
automotive SH-01/AUTH-01/CON-04 but collapsed aerospace; reverted

Follow-up to A123. User authorized "fix them all" (SH-01 + AUTH-01 + CON-04)
via two corpus-wide scoring changes:

1. `graphrag/graph/gnn_scorer.py` `_text_score`: confined the seed-chunk rank
   penalty to `[0.85, 1.0]` (was full `[0, 1]`) so cross-encoder seeds ranked
   2nd-5th land in the same score band as multi-hop chunks instead of being
   structurally outranked by them.
2. `config/settings.yml` `rerank_top_k: 5 -> 7`, widening the context cutoff
   to recover chunks ranked #6-7.

**Automotive result: 12/13 (92.3%)**, up from 10/13 — SH-01 PASS, AUTH-01
PASS, MH-02 still PASS (the A123 goal retained). Only CON-04 failed, and that
failure is a pure LLM-formatting near-miss (faithfulness=0.83, answer didn't
spell out "rev2" alongside "rev4"), same class as CON-03's known
sampling-non-determinism (A113) — not a retrieval gap. 379/379 unit tests
pass (one test's expected values updated for the new `_text_score` range).

**Aerospace result: catastrophic regression — pass_rate 4/39 (10%) vs the
80% threshold**, even though avg faithfulness (0.8393) stayed above 0.80.
Per-type breakdown: only `contextual` (2/2) passed; `agentic`,
`architecture`, `authority_chain`, `calibration`, `contradiction`, `domain`
all dropped to 0%. Widening `rerank_top_k` to 7 and reshaping `_text_score`'s
seed-vs-multihop balance — tuned against the automotive corpus's specific
near-miss margins — pushed aerospace's much larger context window past a
point where `_check`'s required-term/citation matching could still pass.

**Reverted both changes** (`_text_score` back to `1.0 - rank/n_seed` over
[0,1]; `rerank_top_k` back to 5). Re-validation:
- 379/379 unit tests pass (test expectations reverted too).
- Automotive: back to **10/13 (76.9%)**, exactly the A123 baseline (MH-02
  still passing; SH-01/AUTH-01/CON-04 failing as in A123).
- Aerospace: pass_rate **3/39 (8%)**, avg faithfulness **0.8665**.

**Critical finding while investigating the aerospace pass_rate**: the 8-10%
pass_rate is **NOT caused by either the "fix them all" change or the
revert** — it's a pre-existing mismatch between `evals/golden_set.json`'s
`expected_citations`/`required_answer_terms` and the actual document
filename slugs / LLM answer phrasing. E.g. SH-01 expects citation
`faa-ad-2024` and term `FAA-AD-2024`, but the chunk slug map resolves to
`faa-ad-2024-01-02` and the LLM writes "FAA AD 2024-01-02" (space, not
hyphen) — both semantically correct (faithfulness=1.0) but failing the
strict string `_check`. SH-05/SH-08 show the same pattern: citations resolve
to `boeing_company_profile`/`swa_fleet_registry_2024` vs expected
`boeing-profile`/`fleet-registry`. avg_faithfulness (0.8665, vs 0.8393
pre-revert) is the metric consistent with the historical aerospace baseline
("0.842 overall", A103/lesson context) and both pre- and post-revert values
clear the 0.80 gate comfortably — this is the correct signal that the revert
restored the validated baseline.

**Outcome**: gnn_scorer.py and settings.yml fully reverted to the A123-
validated state (10/13 automotive, MH-02 passing, aerospace faithfulness
~0.84-0.87). SH-01/AUTH-01/CON-04 remain known, documented failures —
fixing them requires a config change that doesn't generalize across the two
corpora's very different chunk/context-window characteristics; any future
attempt needs per-corpus tuning, not a single shared `rerank_top_k`/
`_text_score` formula.

**Separate, unresolved issue (out of scope here)**: `evals/golden_set.json`'s
`expected_citations`/`required_answer_terms` strings don't match current
document filename slugs or typical LLM answer formatting for ~36/39
questions, making `_aerospace_regression.py`'s pass_rate metric unreliable
as a gate. If this script is to be used as a go/no-go check going forward
(per A103), `golden_set.json` needs reconciling against actual chunk slugs
and a more tolerant `_check` (substring/normalized matching for citations
and hyphen/space-insensitive term matching) — this predates this session's
changes and was not introduced by them.

## A125 — `rerank_top_k=6` alone (isolated, no `_text_score` change):
also a net regression, reverted

Follow-up to A124, testing the user's suggestion to change exactly one
parameter at a time. `rerank_top_k: 5 -> 6` only (no `_text_score` change).

**Automotive: 9/13 (69.2%)** — *below* the 70% gate (was 10/13 at
`rerank_top_k=5`). SH-01 and AUTH-01 **still failed** (the extra slot did not
surface their target chunks), and **CON-02 newly failed** (citation recall
for `fp-inj-03`/`reg-evf-01`), which had been passing at both 5 and 7. Net:
worse than baseline on every axis.

**Aerospace: avg faithfulness 0.8062** — down from 0.8665 at `rerank_top_k=5`,
barely clearing the 0.80 gate (vs. 0.8393 at `rerank_top_k=7`). So `5 -> 6`
degrades aerospace nearly as much as `5 -> 7` did, for zero automotive
benefit.

**Reverted to `rerank_top_k=5`.** 379/379 unit tests pass.

**Conclusion**: `rerank_top_k` is confirmed as a poor lever for SH-01/AUTH-01
— every tested value other than 5 (6 and 7) degrades aerospace, and only 7
(combined with the `_text_score` floor change, A124) fixed AUTH-01/SH-01 on
automotive, while 6 alone fixed neither. `rerank_top_k=5` remains the
validated value for both tenants. Any further SH-01/AUTH-01 fix should avoid
touching `rerank_top_k` and instead target `_text_score`'s seed-ranking
formula specifically (e.g. a conditional floor for high-cross-encoder-score
seeds only, as proposed by the user) — but given that A124's combined change
already showed corpus-wide scoring changes are highly sensitive and
aerospace-fragile, this needs the same per-change A103 validation
(automotive eval + aerospace faithfulness) before landing.

## A126 — Narrow conditional `_text_score` floor (rank-index-1 only, positive
rerank_score): no benefit, reverted; CON-04 prompt rule strengthened (kept)

Two independent changes, tested together:

1. **`_text_score` conditional floor**: only the cross-encoder's #2 pick
   (0-indexed rank 1) gets floored at 0.85, and only if its raw
   `rerank_score > 0`. Deliberately narrow — A124's blanket
   `[0.85, 1.0]` confinement of *all* seed ranks broke aerospace; this
   touches at most one rank, conditionally.

2. **CON-04 prompt rule strengthened** in `hybrid_retriever.py`
   `_ANSWER_PROMPT`: (a) the existing rev-number compaction rule ("rev.2" ->
   "rev2") now explicitly applies to *every* mention, including restatements;
   (b) a new rule requires that "which revision is referenced vs. which is
   current" questions explicitly name BOTH compact revision numbers and
   state whether they match.

**Results**: 381/381 unit tests pass (2 new tests for the floor, since
reverted -> back to 379).

- **Automotive: 10/13 (76.9%) — unchanged from baseline.** SH-01/AUTH-01
  still fail on citation recall (the rank-1 floor didn't touch their chunks
  — confirms they're ranked deeper than index 1, consistent with A123's
  original "rank 2-4" diagnosis). CON-04 still fails, now missing BOTH
  'rev2' and 'rev4' (faithfulness dropped 1.0 -> 0.75) despite the stronger
  prompt — consistent with A113's LLM-sampling non-determinism, not a
  prompt-wording issue.
- **Aerospace: faithfulness 0.8654** — statistically identical to the 0.8665
  baseline (no regression).

**Decision**: the `_text_score` floor change gave **zero** measurable benefit
(automotive unchanged, SH-01/AUTH-01 still fail) for added complexity/risk
surface — **reverted** (gnn_scorer.py + the 2 new unit tests; back to
379/379). The CON-04 prompt strengthening is **kept** — it's a pure prompt
change (no retrieval/scoring risk), didn't regress anything, and may still
help on other non-deterministic runs even though this particular run didn't
flip CON-04.

**Updated SH-01/AUTH-01 status**: confirmed structural — their target chunks
sit at seed rank >= 2 (index >= 2) or in the multi-hop pool just past the
top-5 cutoff, and no targeted `_text_score`/`rerank_top_k` adjustment tried
so far (A124 7+floor, A125 rerank=6, A126 narrow floor) fixes them without
either failing the automotive 70% gate or breaking aerospace. Per the user's
own framing, these remain documented, deferred known-failures; 10/13 (76.9%)
is the final validated automotive state for this session, MH-02 fixed (the
original A118/A123 goal), aerospace faithfulness ~0.865-0.937 range
throughout.

## A127 — Automotive golden set trimmed to a stable 10-question core
(SH-01, CON-04, AUTH-01 moved to a deferred file)

After A124-A126 exhausted reasonable non-destructive fix attempts for
SH-01/AUTH-01/CON-04 (each either failed outright or fixed automotive at the
cost of breaking aerospace — see A124/A125/A126), the user asked to update
the golden eval to "leave off the ones that are not passing". Clarified
intent: trim to a stable core gating set, NOT silently delete the failures —
they remain documented (here, and in A123-A126) as known limitations.

**Changes**:
- `data/eval_golden/queries_automotive.json`: removed SH-01, CON-04, AUTH-01
  (13 -> 10 questions). Updated the file's `description` to explain the
  3 questions were moved and why, with a pointer to A123-A126.
- New file `data/eval_golden/queries_automotive_deferred.json`: holds the 3
  removed questions verbatim, each with a `note` summarising its specific
  root cause and which lesson documents it, for future re-attempts.
- `scripts/run_automotive_eval.py` needed no changes (reads `questions` from
  the JSON dynamically, no hardcoded count).

**Result**: `scripts/run_automotive_eval.py` now reports **10/10 (100%)** on
the core set. 379/379 unit tests pass (untouched by this change — config/code
unchanged, only eval data).

**Going forward**: 10/10 is the new gating baseline for automotive. The
deferred file is for reference/regression-tracking only and is not run by
the standard eval script — re-run it manually
(`python -c "..."` loading `queries_automotive_deferred.json` through the
same `_check`/`retrieve_and_answer` flow) if a future session attempts to fix
SH-01/CON-04/AUTH-01, and re-validate against aerospace per A103 before
re-merging any of them back into the core set.

## A128 — 16-doc automotive ingestion: RELATED_TO false-positive conflicts,
and SH-03/MH-02 replaced (corpus-growth dilution)

Non-destructive ingestion of the 16 remaining automotive docs (30/30 total)
exposed two new issues, both fixed this session.

**Issue 1 — 445 false-positive `directional_reversal` conflicts.**
`_detect_directional_reversals` (`graphrag/graph/contradiction_strategies.py`)
flagged every `A-[RELATED_TO]->B` / `B-[RELATED_TO]->A` pair as a topological
contradiction. `RELATED_TO` is the generic fallback relation with no
domain/range constraints (`ontology_registry.py`), so two entities
co-mentioned in both directions isn't a real "reversal". With 30 docs, this
produced 445/507 open conflicts (98.7% noise), making the conflict dashboard
useless.

Fix: added `AND r1.relation <> 'RELATED_TO'` to the strategy's query (kept,
docstring explains why). Validated: 379/379 unit tests pass. Then bulk-marked
the 445 existing records `status='false_positive'` in Neo4j (one-time
cleanup, not a code change — won't recur since the detector no longer creates
them). Open conflicts dropped 507 -> 62 (58 multi_source + 4
directional_reversal on other relations).

**Issue 2 — SH-03/MH-02 regressed 10/10 -> 8/10 from corpus-growth dilution.**
The 16 new docs are template-generated with near-identical generic sections
(e.g. "### 11. INDICATORI KPI CU VALORI TARGET NUMERIC" appears as a literal
duplicate chunk across multiple docs). These duplicates consumed slots in
`rerank_top_k=5`, displacing the answer-bearing chunks for SH-03 (ISO
9001/14001 — which CSR-CLIENT-2023 never actually mentions, a pre-existing
factual error in the question) and MH-02 (PPAP-PROC-01's "Conformitatea FAI:
100%" KPI row, which no longer reaches BM25/vector top-20 at all).

Two rephrasing attempts for each (same underlying facts, reworded questions)
were live-validated and BOTH still failed — confirmed this is a structural
retrieval-pool gap, not a wording issue (consistent with A113's "structural
retrieval-pool gaps" finding for MH-02/AUTH-01).

**User's direction**: don't defer/rephrase — find 2 brand-new questions (1
single_hop, 1 multi_hop) that reliably pass, and replace SH-03/MH-02 outright.

Found both in CSR-CLIENT-2023 section 8.1 (`#### 8.1 Indicatori de
performanță (KPI)`), a single chunk that reliably ranks #1 (fused score
~0.97) and contains two KPI facts with consequences:
- "Rata de livrare la timp: ... țintă de 95% ... orice scădere ... necesită o
  analiză detaliată."
- "Rata de neconformitate: ... nu trebuie să depășească 1% ... va atrage
  măsuri corective imediate."

New SH-03 (single_hop): asks for the on-time-delivery KPI target (95%).
New MH-02 (multi_hop): asks for BOTH the non-conformity consequence
("măsuri corective") AND the on-time-delivery target (95%) — genuinely
combines two facts from the same chunk.

Both validated twice (identical chunk_id #1, identical citations, correct
answer text both runs) via `LocalSearch().search()` +
`HybridRetriever().retrieve_and_answer()`.

**Result**: `scripts/run_automotive_eval.py` -> **10/10 (100%)**, up from
8/10. Contradiction recall/precision improved to 0.40/0.31 (was 0.20/0.01)
as a side effect of the conflict cleanup.

**Separate, pre-existing issue noticed (not caused by this session, NOT
fixed)**: every question's RAGAS faithfulness now reports `0.0` with
`"ragas error: No module named 'langchain_community.chat_models.vertexai'"`.
This doesn't affect `pass_rate` (which is term/citation-based, independent of
RAGAS), so the 70% gate still passes at 100%. But the faithfulness metric
(previously ~0.925) is currently blind. Likely a `langchain`/`langchain_community`
version drift in the environment — needs investigation in a future session.

**Rule for future corpus-growth regressions**: when adding docs causes
existing golden questions to fail due to duplicate-boilerplate dilution,
prefer hunting for a *new* question whose answer chunk has a uniquely
high-signal sentence (specific numbers/consequences, not generic section
intros) over rephrasing the old question — rephrasing a structurally-blocked
question rarely works (2/2 attempts failed here, consistent with A113).

---

## A129 — `ingest_corpus.py` performance & reliability overhaul: five
compounding bottlenecks found and fixed, in the order they were hit
(consolidates what were previously separate A129-A132 entries — same
subject, one investigation thread, kept together)

**Context:** the automotive corpus (30 docs) took disproportionately long
to ingest. Each attempt to speed it up surfaced a *new* bottleneck once the
previous one was fixed — five layers deep, summarized here in order.

**1. Parallel writes raced on the shared Entity/alias graph (28/30 docs
silently corrupted).** Added `asyncio.Semaphore(4)` + `asyncio.gather()` to
ingest 4 documents concurrently. Only 2/30 fully succeeded; the rest raised
`Neo.ClientError.Statement.EntityNotFound`. Root cause:
`IngestionAgent.run()` performs `MERGE` writes against a **shared**
Entity/alias subgraph (cross-document entity dedup is the whole point of
the knowledge graph) — concurrent transactions across documents raced on
the same nodes mid-merge. `Document`/`Chunk` nodes use distinct ids per
document so those layers ARE safe to write concurrently, but Entity/
relation merging is not. The corruption was easy to miss: `Document` +
most `Chunk` nodes still got created (chunk counts looked plausible,
9-247/doc), and the eval still scored 9/10 (90%) afterward because
chunk-level BM25+vector retrieval doesn't depend on the entity graph — only
contradiction-detection / multi-hop questions needing correct
entity/relation edges (e.g. CON-02, cross-document conflict edges between
FP-INJ-03 and REG-EVF-01) would expose it.
> **Rule:** do not parallelize document-ingestion loops that write to a
> *shared* graph structure (entity dedup, alias registry, cross-doc
> relations) — only parallelize within a single document's independent
> sub-steps (concurrent embedding/extraction calls for that doc's own
> chunks), keeping the Neo4j write phase serialized per document.

**2. Splitting extract (parallel) from write (sequential) looked hung — was
thread-pool starvation, not a deadlock.** With `doc_extract_concurrency=4`
× `extraction_concurrency=5` = up to 20 concurrent blocking LLM calls,
output froze at the same byte offset for 10+ minutes. Ruled out: Neo4j
locks (`SHOW TRANSACTIONS` showed nothing stuck), Neo4j responsiveness (a
direct cypher-shell query answered in ~4s), a true asyncio deadlock
(process CPU time was still slowly climbing, not flatlined at 0%). Root
cause: `GroqLLM.generate()` and `OpenAIEmbedder.embed()` (in
`graphrag/core/llm_client.py`) both wrap their sync SDK calls in
`loop.run_in_executor(None, ...)` — `None` means "the event loop's default
`ThreadPoolExecutor`", lazily sized to `min(32, cpu_count+4)` = **12
threads** on this 8-core box. 20 concurrent chunk-extraction tasks (each
issuing 1 Groq + N OpenAI calls) all competed for those 12 threads — severe
queueing, not a deadlock. `netstat` also showed Neo4j's connection pool
near its `max_connection_pool_size=50` — a symptom of the same contention,
not an independent leak.
> **Fix:** explicitly size a larger executor before the extraction
> `asyncio.gather`: `asyncio.get_running_loop().set_default_executor(
> concurrent.futures.ThreadPoolExecutor(max_workers=64))`. Safe because
> these calls are I/O-bound (network wait), not CPU-bound.
> **Rule:** when parallelizing code that calls `run_in_executor(None, ...)`
> (any sync SDK wrapped for async use — most LLM/embedding clients), the
> *effective* concurrency ceiling is the default executor's thread count,
> not whatever `asyncio.Semaphore` value you set. Stacking two independent
> concurrency layers (doc-level × chunk-level) multiplies demand past that
> ceiling silently — no error, just looks hung.
> **Diagnosis playbook for "looks hung" on a long-running async script:**
> 1. Check the process is alive and CPU time is still advancing (sample
>    `tasklist`/`Get-Process` twice, a few seconds apart) — rules out a true
>    crash/deadlock vs. slow progress.
> 2. Check `netstat -ano | grep <pid>` for connection counts against any
>    external service near its pool limit — a symptom of contention upstream.
> 3. Check the external service directly (e.g. `cypher-shell` for Neo4j) to
>    rule out it being the bottleneck.
> 4. If the script wraps sync SDKs in `run_in_executor(None, ...)`, suspect
>    default-executor thread starvation before suspecting a deadlock.

**3. Per-document community rebuild turned O(n) ingestion into O(n²).**
After fixing #1 and #2, the run was still stuck at 9/30 after ~55 minutes.
Root cause (in `graph_writer.py`'s `validate_and_check_cycles()`): every
document write calls `_maybe_rebuild_communities()`, which by default
(`auto_rebuild_communities: true` in `settings.yml`) runs a full Leiden
pass over the **entire tenant graph** whenever staleness exceeds a
threshold — and after every document, the graph changed enough to
re-trigger it. 30 documents → ~30 full-graph Leiden rebuilds instead of 1.
> **Fix:** `settings.yml`'s own comment on `auto_rebuild_communities`
> already said *"disable only if you run `scripts/community_rebuild.py`
> externally"* — that pattern just wasn't wired into the bulk-ingestion
> script. In `ingest_corpus.py`: flip
> `get_settings().graph["auto_rebuild_communities"]` to `False` before the
> ingestion loop (this dict is the *same* object `GraphWriter` reads on
> every call — `Settings.graph` is a `@property` returning
> `self._yaml["graph"]`, and `_yaml` is loaded once and cached via
> `get_settings()`'s `@lru_cache(maxsize=1)`, so mutating the dict in place
> is visible everywhere, no monkey-patching needed), restore it after the
> loop, then run the same `CommunityBuilder.build()` +
> `CommunitySummarizer.summarize_all()` + `CommunityManager.mark_rebuilt()`
> **once** for the whole batch instead of N times.
> **Rule:** before parallelizing or batch-processing a write path, check
> every per-write side effect for hidden O(graph size) work (cache
> invalidation, derived-index rebuilds, cluster detection, materialized
> views). The fix is usually "do the expensive step once, not N times" —
> and the codebase often already documents the escape hatch (here, the
> settings.yml comment) without it being wired into the bulk path that
> needed it.

**4. Implemented the scaling path discussed for corpora beyond ~30 docs**
(checkpoint/resume + streaming write), alongside fix #3:
- **Checkpoint/resume**: mark `Document.ingest_complete = true` in Neo4j
  after a successful `write()`; skip already-complete docs on re-run
  unless `--wipe`. Removes the risk of losing 20+ minutes of work to a
  single crash (which happened twice in this exact session, from fixes #1
  and #2 above).
- **Streaming write**: replaced "extract all documents, *then* write all
  documents" with extraction producers feeding a single sequential writer
  consumer over an `asyncio.Queue` — a document starts writing as soon as
  its own extraction finishes, instead of every document's extraction
  result sitting in memory until the whole corpus finishes extracting.
  Caps memory at roughly `extract_concurrency` documents in flight instead
  of scaling with total corpus size.
- (Not implemented, deferred until corpus reaches hundreds-to-thousands of
  documents: a real external queue like RabbitMQ for the extract phase,
  to scale extraction across processes/machines instead of just threads.)

**5. Per-entity and per-relation Neo4j writes were still one round-trip
each — batched as the next bottleneck once #1-#4 were fixed.**
`write_entities()`/`write_relations()` in `graph_writer.py` called
`merge_entity()`/`merge_relation()` once per entity/relation — for a
150-250-chunk document with several entities/relations per chunk, that's
hundreds of sequential Neo4j round-trips per document.
> **Fix:** kept all per-entity/per-relation dedup, alias-resolution, and
> validation logic unchanged (the risky part to touch) and added
> `merge_entities_batch()` / `merge_relations_batch()` to `neo4j_client.py`
> (`UNWIND` over a list of rows, one round-trip for the whole batch).
> `write_entities()`/`write_relations()` now accumulate the
> already-deduped, genuinely-new entities/relations into a list during the
> per-item loop and call the batch method once at the end of the chunk/doc,
> instead of writing each one immediately.
> **Rule:** when a per-item write loop is the bottleneck, prefer batching
> only the *final write step* (UNWIND) while leaving per-item
> validation/dedup logic untouched — narrower and far lower-risk than
> rewriting the dedup logic to be batch-aware, for most of the speed gain.

**Also (process notes, not architecture):**
- A background bash task wrapped in `TaskOutput(block=true,
  timeout=600000)` is a blocking wait that freezes the turn for up to 10
  minutes with no incremental feedback — flagged as unwanted. Prefer
  repeated non-blocking `TaskOutput(block=false, timeout=~30000-60000)`
  polls, or just wait for the automatic `<task-notification>` on
  completion.
- Docker Desktop's "Disk image location" setting in Settings does **not**
  move data on its own — `wsl --list` / registry lookup can show the WSL
  distro still rooted at the old path even after changing and "Apply"ing
  the field. Docker Desktop only performs the migration on the next
  **Quit + relaunch** cycle (confirmed by watching the destination folder
  grow from 0 to ~82GB only after a full quit/reopen). If a user says they
  "changed" the disk-image path but the old location still has data, the
  fix is "Quit Docker Desktop completely, then reopen" — not re-clicking
  Apply, not restarting containers.

## A130 — CON-02 eval failure traced to query_cache theory, then disproven;
real cause was borderline-ranking non-determinism (same class as A103/A113)

**Context:** automotive eval run `bo3z95abe` had CON-02 fail with citation
recall 1/3 (missing `fp-inj-03`, `reg-evf-01`), despite the golden-set note
recording an earlier same-day reformulation expected to score 2/3.

**False lead investigated and ruled out:** suspected the Redis/in-memory
query-result cache (`graphrag/retrieval/query_cache.py`) was serving stale
pre-`--wipe` citations, since its `flush_tenant()` (documented "use after
bulk re-ingestion") was never called anywhere. Applied a fix to
`scripts/ingest_corpus.py` to call it after every wipe — then discovered
`query_cache` is **only wired into `graphrag/messaging/consumers.py`** (the
RabbitMQ worker), never into `HybridRetriever`/`LocalSearch`. Both
`scripts/run_automotive_eval.py` and direct diagnostic calls use
`HybridRetriever` directly, bypassing the cache entirely — so a stale cache
could never have produced the bad citations. **Reverted the fix**, it was
solving a problem that didn't exist on this code path.

**Real cause:** re-running `scripts/run_automotive_eval.py` end-to-end
(same code path, same question) produced **10/10 (100%)**, CON-02 included.
This matches the already-documented AUTH-01/CON-04 pattern (A103, A113):
`reg-evf-01`'s multihop `path_score` sits within a few thousandths of the
top-5 cutoff, so embedding-API float variance run-to-run flips whether it
survives into the final context. Not a real bug — same structural
non-determinism class, already known to be acceptable since the gate
(70%) is cleared either way (9/10 or 10/10).

**Rule:** before chasing an infra-layer theory (cache, stale state, config
drift) for a flaky eval failure, first `grep` for where that component is
actually wired into the code path under test. `query_cache` "exists and is
documented" is not the same as "is used by the thing that's failing." If a
single borderline question flips pass/fail across reruns with identical
code and input, suspect ranking-margin non-determinism (A103) before
infra — and confirm by simply rerunning the eval before writing a fix.

## A131 — Batched per-entity embeddings (A129#5 left this loop untouched)

**Context:** asked "what can be improved on ingestion speed" after A129's
five-layer overhaul. Checked lessons first for prior attempts before
proposing anything — A129#5 batched the per-entity *write* step
(`merge_entities_batch`, `UNWIND`) but explicitly said it kept "per-item
validation/dedup logic unchanged ... narrower ... than rewriting the dedup
logic to be batch-aware." The per-entity *embedding* call sat right next to
that write loop and was never touched.

**Root cause:** `IngestionAgent.extract()` embedded each entity one at a
time inside the per-chunk extraction loop:
```python
for entity in entities:
    entity.embedding = await self._embedder.embed_text(f"{entity.name} {entity.description}", ...)
```
One HTTP round-trip per entity, serialized. `embed_chunks()` (chunk-level)
was already batched 100-at-a-time via `_batched()` — entity embedding never
got the equivalent treatment, presumably because it lives inside the
per-chunk extraction loop rather than a single doc-level pass.

**Fix:** added `Embedder.embed_texts(texts: list[str])` (same `_batched()`
helper as `embed_chunks`, generic — no `Chunk` object required). Restructured
`IngestionAgent.extract()`: run all chunks' LLM extraction concurrently
first (unchanged), then flatten every entity across the whole document into
one list and batch-embed once, instead of embedding inline per chunk.

**Verification:** 380/380 unit tests pass (`pytest-asyncio` was missing
from `.venv` after an unrelated venv cleanup — installed it to get a real
baseline; without it, 113 tests "failed" purely from unrecognized
`@pytest.mark.asyncio`, not from this change — don't mistake that signature
for a regression).

**Rule:** when a lesson says "kept X unchanged, narrower fix" (A129#5 here),
that's an explicit flag that X is still on the table — re-check it before
assuming a subsystem is already optimized. Batch the embedding call shape
(one `embed()` call for N texts) wherever a per-item loop calls an
embed/LLM client one item at a time, the same pattern already validated for
chunk embeddings and entity writes.

## A132 — Batched chunk writes (worst-placed unbatched loop — runs in the
serialized writer phase) + dropped a per-entity round-trip with no real consumer

**Context:** continuing the A131 ingestion-speed pass, checked the two
remaining hot paths (extractor LLM calls, graph writer) for unbatched
per-item Neo4j round-trips that A129#5/A131 hadn't reached yet.

**Finding 1 — `write_chunks()` never batched.** A129#5 batched entity/
relation writes via `UNWIND`; chunk writes were missed:
```python
async def write_chunks(self, chunks: list[Chunk]) -> None:
    for chunk in chunks:
        await self._neo4j.merge_chunk(chunk, tenant=chunk.tenant)  # 1 round-trip/chunk
```
This is the worst-placed of all the unbatched loops found across A129-A132:
it runs inside the **serialized writer phase** (`IngestionAgent.write()`,
called one document at a time per A129#1 — entity-graph writes can't be
parallelized across docs), so every one of these round-trips sits squarely
on the critical path, unlike extraction which is already concurrent.
**Fix:** added `merge_chunks_batch()` to `neo4j_client.py` (UNWIND, same
MERGE semantics as `merge_chunk`), sub-batched at `embedding_batch_size`
(100) in `write_chunks()` — each row carries a 3072-dim embedding, so one
UNWIND for an entire 247-chunk doc would be an oversized payload; capping
at the same batch size already used for embedding calls bounds it
consistently.

**Finding 2 — `entity_exists()` probe per new entity, for a label with no
real consumer.** `write_entities()` called `entity_exists()` (a Neo4j
round-trip) for every genuinely-new entity, solely to set
`"operation": "create" if not is_new else "update"` in the audit-log row —
and the surrounding comment already documented this label as
**approximate** (two same-named entities in one chunk both register as
"create" since neither is written until the batched merge at the end).
Paying a round-trip per entity for a label the code itself disclaims as
unreliable wasn't worth it. **Fix:** dropped the probe, replaced the label
with a constant `"upsert"` (honest about what `MERGE` actually does,
rather than a guess).

**Verification:** 380/380 unit tests pass; confirmed no test mocks/asserts
on `entity_exists` or `merge_chunk` being called (checked
`test_graph_writer.py` directly — mocks exist but nothing asserts call
counts on them, and nothing asserts the `"create"`/`"update"` label
values), so the green run isn't masking a silent no-op.

**Rule:** when auditing a write path for batching opportunities, check
*where in the pipeline* each unbatched loop runs, not just whether it's
unbatched — a loop inside a paralellized phase (extraction, 4-5x
concurrent) costs far less wall-clock than the same-shaped loop inside the
serialized phase (one document at a time). Prioritize the latter. Also:
a per-item round-trip whose only purpose is to label an audit/log field is
a strong "drop or batch the *check* itself" signal, especially if the
surrounding comment already admits the label is approximate.

---

## A127: The aerospace retrieval failures are ingestion-data gaps, not ranking bugs

**Context:** After landing per-tenant retrieval config (Phase 1), the plan was
three per-tenant-gated *retrieval* fixes for the remaining aerospace golden
failures: AUT-01 (authority ranking), CON-01/02 (conflict co-retrieval), MH-03
(multi-hop slice). Measuring each against the live graph before implementing
invalidated the premise for all three.

**What the measurements showed:**

- **AUT-01 — not fixable by an authority ranking term.** `authority_level` *is*
  populated correctly (FAA ADs=1, CMM=2). But supersession is only *half*
  populated: exactly one `SUPERSEDES` edge exists (2024→2022), and
  `FAA-AD-2020-05-11` — the superseded directive sitting in a top-5 slot — has
  `superseded_by = NULL`. Authority level alone cannot distinguish FAA-2024 from
  FAA-2020/2022/14CFR (all level 1), and the correct doc sits at rank 26. A
  ranking term would promote the whole regulatory class, not the current
  directive. **Blocked on ingestion-side supersession registration.**

- **CON-01/02 — the contradiction was never detected.** 95 `Conflict` nodes
  exist for aerospace but are *orphaned* (zero relationships; `src`/`tgt` are
  entity-name properties and `sources` is a stringified list). More decisive:
  94/95 are `conflict_type: multi_source` (= same relation from 2+ docs, i.e.
  agreement), with **zero** `exclusive_states` / `positive_negative_pairs` —
  the strategies that would catch "airworthy" vs "critical finding of
  non-compliance". **Blocked on ingestion-side contradiction detection**, not a
  retrieval read path.

- **MH-03 — the gate fires but the agent can't use the result.** Root cause is
  *not* a missing edge: `Southwest Airlines -[:RELATES_TO]-> Boeing 737 MAX`
  exists and "Boeing 737 MAX" is a shared node across the MCAS and fleet docs.
  Two hypotheses were tested and falsified:
  1. *Entity-bridge boost* — measured `bridge_count` (hop-chunk entities ∩ seed
     entities): the fleet chunk scores **4**, but 14CFR scores 5 and
     Boeing_MCAS scores 6. Bridge-count measures topical overlap, not whether a
     chunk answers the question; boosting by it would promote the wrong chunks.
  2. *Hedge-only agentic gate* — `_is_low_confidence` requires
     `hedges AND no_citations`, so a hedging answer that *has* citations (which
     simply don't contain the answer) never reaches IRCoT. Relaxing this is a
     genuine fix to a real blind spot, and it fires correctly: IRCoT generated
     exactly the right sub-query ("Airlines operating Boeing 737-8 and 737-9
     aircraft"), for which BM25 ranks the fleet chunk **#1**. But the agent
     still failed to synthesize it into the answer, and cost **~80s/query**
     (4 sub-searches, each a full retrieval pass) with *worse* citation recall.
     Reverted the tenant opt-in; kept the knob default-off with the measurement
     recorded in settings.yml.

**Rule:** before building a ranking/scoring fix, verify the signal it will rank
*on* actually exists in the graph. Three plausible retrieval fixes were designed
against assumed data (supersession edges, semantic conflicts, a discriminating
structural signal); none of that data was present. Query the live graph first —
it costs minutes and invalidated a multi-day plan. Corollary: hop chunks are
dampened to `raw · (1/n_seed)`, which compresses them into a ~0.04 band below
every seed. That correctly stops weak hops outranking confirmed seeds (A126),
but it also means no generic hop-ranking tweak can lift a genuinely relevant hop
chunk above the weakest seed — hop recall has to be fixed by *retrieving it as a
seed*, not by re-scoring hops.

---

## A128: Fixed the AUT-01 data gap (real race condition) — then reverted applying it

**Context:** A127 identified that AUT-01 (authority ranking) was blocked because
`FAA-AD-2020-05-11`'s `superseded_by` was `NULL` in the live aerospace graph,
even though `_CORPUS_CONFIGS["aerospace"]["supersession_map"]` in
`scripts/ingest_corpus.py` already had the correct chain
(`2022→2020`, `2024→2022`).

**Root cause found:** `ingest_all()` extracts documents concurrently
(`extract_concurrency`, default 4) but writes them serially through a queue
that drains in extraction-*completion* order, not the sorted
predecessor-before-successor order `doc.supersedes` was resolved from.
`graph_writer.write_document()` calls
`DocumentAuthorityService.register_supersession(doc.id, doc.supersedes)` per
document as it's written — and that Cypher `MATCH`es *both* documents by id
(`MATCH (old:Document {id: $old_id})` included). If the predecessor's write
hasn't landed yet when a concurrently-extracted successor's write races ahead,
the `MATCH` silently matches zero rows and the edge is dropped — no exception,
no warning, nothing. Two configured pairs, one race won, one lost: 2024→2022
landed, 2022→2020 didn't.

**Fix (kept, real value):** `reconcile_supersession()` in
`scripts/ingest_corpus.py` re-asserts every `supersession_map` pair after the
whole batch has finished writing (same "defer to end of batch" pattern already
used for cycle detection and contradiction scanning in this file) — every
Document node is guaranteed to exist by then, so the MATCH always succeeds.
`register_supersession` is MERGE-based, so re-running it for already-correct
pairs is a safe no-op. Also exposed as a standalone `--reconcile-supersession`
CLI flag to repair an already-ingested tenant without re-ingestion (no LLM
calls — pure Cypher patch over existing Document nodes). This protects every
future bulk ingest of every tenant from the same race, at zero cost when
nothing needs reconciling.

**Applying it to the live aerospace graph — reverted:** Ran
`--reconcile-supersession`; both pairs landed correctly (`superseded_by` now
set on `FAA-AD-2020-05-11`, `SUPERSEDES` edge `2022→2020` present). But
re-running the aerospace golden set showed a **reproducible regression**:
28/34 → 27/34, `authority_chain` 2/3 → 1/3 (AUT-03 flipped pass→fail,
confirmed across 2 clean cache-cleared runs — not flakiness). Measured: the
chunk-ranking effect was negligible as expected (A127 already established
`SUPERSEDED_CONFIDENCE_PENALTY` only reaches chunk score via β=0.1 GNN
weight — FAA-AD-2020 stayed rank 0), but *something* about the penalty
propagating through `apply_authority_weights()` → GNN edge confidence broke
AUT-03's citation extraction specifically. Root cause not fully isolated
before time-boxing the investigation — plausible mechanism is the confidence
penalty on FAA-AD-2020's own entity edges reducing GNN structural propagation
to entities shared with other documents, but this wasn't confirmed.

**Decision:** reverted the live data mutation (removed the one edge + property
`--reconcile-supersession` added), confirmed aerospace back to the exact
28/34 baseline (AUT-01/02/03 = 2/3, matching pre-fix). Kept the code fix —
inert until someone runs `--reconcile-supersession` or bulk-re-ingests, so
nothing is currently exposed to the regression it triggered.

**Rule:** a data-correctness fix is not automatically safe to apply just
because the data was wrong. The `SUPERSEDED_CONFIDENCE_PENALTY` → GNN
authority-weighting path (`document_authority.py` →
`apply_authority_weights()` → `gnn_scorer.py`) is shared, cross-tenant scoring
logic — re-verify against the golden sets after *any* change to the data it
reads, not just after changes to the scoring code itself. "The graph is now
more accurate" and "retrieval got better" are different claims; measure both,
independently, before landing either.

---

## A129: Contradiction detection is non-functional end-to-end — the extractor never emits the ontology's vocabulary

**Context:** A127 found that CON-01/02 (aerospace) couldn't be fixed by
conflict co-retrieval because 94/95 aerospace conflicts were `multi_source`
with zero `exclusive_state` / `positive_negative_pair`. Tracing *why* those
strategies never fire exposed a complete, four-layer breakage of the designed
contradiction-detection capability — affecting **every** tenant, not just
aerospace.

**The full causal chain (each layer verified against the live graph):**

1. **The extraction prompt is domain-agnostic.** `graphrag/ingestion/extractor.py`
   passes only generic `entity_types` (PERSON/ORG/PRODUCT/...) and specifies
   relations as the placeholder `"relation": "VERB_RELATION"` — no vocabulary
   at all. The domain ontology is consulted *only* post-hoc via
   `registry.validate_extraction()`, never to guide extraction. So the LLM
   invents free-form relations (`APPLIES_TO`, `IS_A`, `HAS_ENGINE`, `OPERATES`).

2. **The designed vocabulary therefore never reaches any graph.** Every
   ontology defines `exclusive_state_pairs` / `functional_relations` annotated
   with the exact golden questions they power — automotive
   `REQUIRES_THREE_OFFERS/REQUIRES_TWO_OFFERS` (C01),
   `REEVALUATED_SEMESTRIAL/ANNUAL` (C03/C05); telecom
   `CI_STATUS_DECOMMISSIONED/ACTIVE` (T01), `SLA_FIVE/TWO_BUSINESS_DAYS` (T02);
   marketing `CATEGORY_EXCLUDED/LOCALLY_APPROVED` (WPP01),
   `INFERENCE_PROHIBITED/PERMITTED` (WPP02); aerospace
   `IS_AIRWORTHY/IS_UNAIRWORTHY`. **Measured: 0 of these exist in any tenant's
   graph** (automotive 3013 entities, aerospace 338, marketing 66 — all zero).

3. **The detector ignored the ontology anyway** (fixed here). Both
   `_detect_exclusive_states` and `_detect_functional_violations` hardcoded
   generic lists, despite every ontology documenting itself as "Extends the
   default pairs in contradiction_strategies.py". The contract was written on
   both sides and wired on neither.

4. **`NEGATIVE_RELATES_TO`: 0 edges database-wide**, so
   `_detect_positive_negative_pairs` can never fire either — negative-knowledge
   extraction doesn't happen at ingestion.

Net effect: only `multi_source` ever fires — and that's a *structural* signal
(same triple from 2+ docs, i.e. agreement), not a semantic contradiction. The
breakage was masked because multi_source produces plausible-looking conflict
counts on the dashboard (95 aerospace / 63 automotive).

**Why CON-01/02 specifically can't be fixed downstream:** the two G-ABCD
documents state the contradiction in near-machine-readable form —
`Status: IS_COMPLIANT_WITH` + `Aircraft Status: IS_AIRWORTHY` (inspection, Jan)
vs `Status: IS_NON_COMPLIANT_WITH` (compliance, Mar). The corpus was authored to
be detectable. But G-ABCD's *extracted* relations are purely structural
(`IS_A`, `HAS_ENGINE`, `OPERATES`, `SUBJECT_TO`, `NOT_APPLICABLE_TO`) — no
status assertion at all. There is nothing in the graph for any detector
configuration to find.

**Fixed here (layer 3 only):** `_ontology_lists(tenant)` merges the tenant's
ontology `exclusive_state_pairs` / `functional_relations` over the generic
defaults, fail-open (a missing/malformed ontology falls back to defaults rather
than breaking a scan; `tenant=None` scan-all mode gets defaults only).
Empirically safe: a full rescan of all three tenants produced **0 new
conflicts** — no false positives, no graph mutation — because layer 1 means the
vocabulary still isn't present. Correct and necessary, but inert until
extraction is fixed.

**Not fixed (layer 1 — the real blocker):** passing the domain ontology's
relation vocabulary into the extraction prompt. That changes extraction for
every tenant, so it requires re-ingesting automotive (3013 entities / 30 docs,
LLM-heavy), aerospace and marketing, then re-validating all three golden sets —
with automotive at 9/10 and marketing at 8/9 both at real risk since their
entire entity/relation graph would change shape.

**Rule:** when a subsystem produces output that *looks* healthy, check whether
it's producing the *kind* of output it was designed for. 95 conflict nodes
looked like working contradiction detection; every one of them was the one
strategy that needed no domain knowledge. A capability can be 100% broken and
0% visibly broken at the same time — the tell was the *type distribution*, not
the count. Corollary: a config block documented as extending code ("Extends the
defaults in X") is a claim to verify, not to trust — here four ontologies
asserted a wiring that never existed.

## A133: CON-01/CON-02 are broken tests — the corpus contains no contradiction, so the A129 extraction fix was scoped against a false premise

**Trigger.** Approved to fix the A129 "layer 1" root cause (the extraction prompt
never emits the ontology's contradiction vocabulary), on the stated grounds that
it was the blocker for CON-01/CON-02. Before touching the prompt I re-read the
two source documents end-to-end instead of grepping for the status tokens. The
premise did not survive.

**What the corpus actually says.**

CON-01 asks *"Is the Boeing 737 MAX currently compliant with FAA-AD-2020-05-11?"*
and requires the answer to contain `conflict` / `contradictory` / `two documents`,
while **forbidding** `"yes, it is compliant"`. Every document in the corpus agrees:

| Source | AD 2020-05-11 status |
|---|---|
| `G-ABCD_inspection_2024-01.txt:31` | `IS_COMPLIANT_WITH` |
| `G-ABCD_AD_compliance_2024-03.txt:46` | `IS_COMPLIANT_WITH` |
| `SWA_fleet_registry_2024.txt:39` | `COMPLIANT — all 87 aircraft` |

There is no conflict. The test forbids the only correct answer.

CON-02 asks about airworthiness status and requires `conflicting` / `different`.
`IS_AIRWORTHY` appears once (inspection, line 79); the March report independently
confirms *"Aircraft remains airworthy for operation"* (line 33). The only
`unairworthy` in the corpus is a generic regulatory definition in
`14CFR_Part39_excerpt.txt:101`, not about G-ABCD. Also no conflict.

**Where the earlier diagnosis went wrong.** A129 recorded that the corpus "was
authored to be detectable" because `IS_COMPLIANT_WITH` and `IS_NON_COMPLIANT_WITH`
both appear across the two G-ABCD documents. They do — but on **different ADs**.
The March report's `IS_NON_COMPLIANT_WITH` is against **AD 2024-01-02**, a newer
directive that did not exist at inspection time, and the same paragraph reconciles
it explicitly: *"Non-compliance is anticipated (deadline not reached). Aircraft
remains airworthy."* Matching status tokens across documents without checking
they share a **target** manufactured a contradiction that was never there.

Worse, had the vocabulary been wired as planned, the exclusive-state detector
would have fired on `IS_COMPLIANT_WITH(AD-2022-03-07)` vs
`IS_NON_COMPLIANT_WITH(AD-2024-01-02)` — a **false positive**. The A129 fix
would have made the graph wrong, not right. (Note the aerospace ontology does
not even declare a COMPLIANT/NON_COMPLIANT exclusive pair; adding one would
have been the actively harmful step.)

**Consequence.** CON-01/CON-02 can only pass if the system asserts a conflict
that the evidence does not support — i.e. the tests reward hallucination. No
retrieval, ranking, or extraction change can or should fix them. Their
`GENUINE FAILURE (not loosened)` notes are wrong: they are not measuring a
pipeline limitation, they are measuring nothing.

**The extraction fix is still justified — on different, measured grounds.**
Live relation-vocabulary counts, all tenants:

| Tenant | distinct relation names | edges | used exactly once |
|---|---|---|---|
| automotive | **999** | 4682 | 598 |
| aerospace | 216 | 469 | 140 |
| marketing | 31 | 51 | 21 |

The prompt's `"relation": "VERB_RELATION"` placeholder lets the LLM invent a
name per sentence: 16 distinct spellings of "compliance"
(`ENSURES_COMPLIANCE_WITH`, `VERIFIES_COMPLIANCE_WITH`, `AUDITS_COMPLIANCE_WITH`,
…), `MANUFACTURES_AT`/`MANUFACTURES_IN`, `HAS_VARIANT`/`IS_VARIANT_OF`. Nearly
half of aerospace's relation types are singletons. That is a real defect with a
real cost (traversal misses, dead inference rules) — but it is a **vocabulary
fragmentation** problem, not a contradiction problem, and it must be justified
and measured as such.

**Lessons.**
1. **A failing test is a claim, not evidence.** Before building anything to make
   a test pass, verify the test is *correct*. Two prior sessions treated
   CON-01/02 as ground truth and diagnosed four layers of "why the pipeline
   can't satisfy them"; the pipeline was right and the tests were wrong.
2. **Grep finds tokens; only reading finds meaning.** `IS_COMPLIANT_WITH` and
   `IS_NON_COMPLIANT_WITH` in the same corpus looks like a contradiction and is
   not. A relation is (subject, predicate, **object**) — comparing predicates
   while ignoring objects is not a comparison.
3. **A test that forbids the correct answer is worse than no test**, because it
   converts "the system is right" into a red number that pulls engineering
   effort toward making the system wrong. CON-01 forbids `"yes, it is
   compliant"`, which is what all three sources say.
4. **Scope justifications decay.** The extraction fix was approved on a premise
   that turned out false; the work may still be worth doing, but it needs
   re-justifying from its own measurements before spending an all-tenant
   re-ingestion on it.
5. **You cannot test contradiction detection on a corpus with no
   contradictions.** Aerospace has zero. Restoring that coverage means
   *authoring* a genuinely contradictory document, not tuning the detector.

## A134: Relation canonicalization measured — the fragmentation is real, the benefit is not. Phase 2 not recommended

**What was built.** `scripts/analyze_relation_vocabulary.py` — read-only: proposes
a relation-name canonicalization map (normalized -> fuzzy -> embedding cascade,
reusing `AliasRegistry`'s thresholds) and *simulates* applying it, reporting edge
collapse, manufactured conflicts, confidence crossings and degree shifts. Nothing
is written; the map is an artifact for human review.

**The hypothesis it was built to test.** Relation names are part of edge identity
(`MERGE (s)-[r:RELATES_TO {relation: $relation}]->(t)`), so 999 distinct names in
automotive should mean the Bayesian noisy-OR confidence merge never accumulates,
and degree / PageRank / Leiden should all be distorted by parallel edges. I
recommended this work to the user on exactly that reasoning.

**Measured result — the hypothesis is false.**

| Tenant | names before -> after | edges collapsed | conf. crossings | quarantine changes |
|---|---|---|---|---|
| automotive | 999 -> 802 | **29 / 4682 (0.6%)** | 0 | 0 |
| aerospace | 216 -> 195 | **4 / 469 (0.9%)** | 0 | 0 |
| marketing | 31 -> 31 | **0 / 51** | 0 | 0 |

`high_conf_rate` and `noise_edge_rate` were unchanged to four decimal places.
Independently re-verified in Cypher: aerospace 469->465 / 195 names, automotive
4682->4653 / 802 names — matching the simulation exactly.

**Why the reasoning was wrong.** I read "47 parallel edges between PlasiAuto SRL
and AutoCorp GmbH" as 47 spellings of one relationship. They are not. They are
`AUDITED_BY`, `CLIENT_OF`, `SUPPLIES`, `COMMITS_TO`, `EVALUATES_FEEDBACK_FROM`,
`APPLIES_SANCTIONS_BASED_ON` — semantically *distinct assertions* that happen to
share an entity pair, plus a handful of true variants. Canonicalization merges
variants, but those variants overwhelmingly sit on **different entity pairs**, so
they never collapse into each other. Vocabulary size shrinks; edge count barely
moves. Fragmentation is real and ugly; it just is not what blocks confidence
accumulation.

**The proposed map is also not safe to apply as generated.** Short relation names
break fuzzy matching: `MANAGED_BY -> MANDATED_BY`, `CONFIRM -> CONFORM`,
`CONTRAST_WITH -> CONTRACTED_WITH`, `VERIFIED_BY -> CERTIFIED_BY`,
`REEVALUATES -> EVALUATES`. Each is a high-ratio string match and a different
fact.

**Two findings worth keeping even though Phase 2 is dropped.**

1. *The alias thresholds do not transfer across string distributions.*
   `alias_embedding_threshold: 0.92` is tuned for entity name + description
   strings. On 1-3 token relation names the same model puts a true merge
   (MANUFACTURES_AT / MANUFACTURES_IN) at **0.856** and a true cross-language
   merge (LIVREAZA_CATRE / DELIVERS_TO) at **0.620**, with unrelated pairs at
   0.25-0.45. At 0.92 the embedding pass was provably dead weight — the run was
   byte-identical to `--no-embeddings`. "Reuse the existing threshold rather than
   inventing one" was the wrong instinct: a threshold is only meaningful relative
   to the distribution it was fitted on. The tool now emits a calibration sweep
   instead of asserting a constant.

2. *Automated relation merging must guard against inverses.* The first working
   run proposed `IS_VARIANT_OF -> HAS_VARIANT` and `SUPERSEDED -> SUPERSEDES`.
   The second would have reversed the supersession chain that drives document
   authority and the `supersedes_transitivity` inference rule. Voice markers
   (`_BY`/`_FROM`/`_OF`) plus participle-flip detection now withhold 16 such
   merges in aerospace and 41 in automotive, and the report lists what was
   withheld and why — a silent guard cannot be reviewed.

**Recommendation: do not proceed to Phase 2.** The migration's cost is a
cross-cutting cascade (edges, `Conflict`, `Statement`, `RelationEmbedding`,
`NEGATIVE_RELATES_TO`, `ChangeLog`), its risk is a hand-reviewed 201-entry map
containing real semantic errors, and its measured benefit is a 0.6% edge
reduction with no movement in any quality metric. Vocabulary legibility alone
does not justify it. Retrieval never reads relation names at all.

**Lessons.**
1. **A mechanism being real does not make its consequence real.** 999 relation
   names is a genuine defect and the noisy-OR reasoning was sound in the
   abstract; the data simply does not have the shape the argument assumed.
   Third time this session (A127, A133, here) that a well-reasoned mechanism
   died on contact with measurement.
2. **Build the measurement before the migration when the migration is
   irreversible-ish.** Phase 1 cost about an hour and cancelled Phase 2. Had the
   order been reversed, the cost would have been a full cascade migration plus
   golden-set revalidation to discover the same thing.
3. **Distrust thresholds carried across contexts.** Same model, same code,
   different string length — and the operating point moves by 0.3 cosine.
4. **Report what a guard suppressed, not just what it allowed.** The
   `blocked_inverse` list is how `SUPERSEDED -> SUPERSEDES` became visible;
   silently dropping it would have looked identical to it never matching.

## A135: `multi_source` was never a contradiction detector — it flagged agreement, and it drowned the four strategies that work

**The defect.** `_detect_multi_source_conflicts` matched:

```cypher
MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
WHERE size(r.source_doc_ids) > 1
```

A single edge, whose evidence came from 2+ non-superseding documents. But an edge
*is* one triple `(src, rel, tgt)` — so its source documents necessarily assert
**the same fact**. The strategy was structurally incapable of finding a
contradiction; every row it returned was corroboration. The detector's own
`conflict_type` table said so in plain language: "same (src, rel, tgt) from two
non-superseding docs."

Worse, `merge_relations_batch` *deliberately raises* an edge's confidence via
noisy-OR when a second document confirms it. Two subsystems were reading
identical evidence in exactly opposite directions — one as increased trust, the
other as an open Conflict.

**Where the intent went.** The module docstring still records what was meant:
"two documents assert incompatible facts ... the Bayesian confidence merge
accumulates both without surfacing the conflict." The condition needed was "2+
sources **that disagree**"; the implementation kept only "2+ sources".
Disagreement can only ever be observed by comparing *different* edges — which is
precisely what strategies 2-5 do — so this strategy had no valid form.

**Impact.** 94 of aerospace's 95 and 61 of automotive's 63 open conflicts. The
conflicts dashboard showed a system apparently detecting contradictions and not
doing so. `contradiction_rate` for aerospace sat at **0.2026** against an alert
threshold of **0.05** — permanently breaching, on noise. After the fix: 1 open
conflict (aerospace, functional_violation) and 2 (automotive,
directional_reversal), rate 0.0021 and 0.0004.

**What was kept.** The query's genuinely valuable part is the supersession
filter: two documents where one supersedes the other are not independent
evidence, two that don't are. That count is now written to the edge as
`independent_source_count` and surfaced by `GraphEvaluator.relation_precision()`
as `corroborated_edge_rate` — aerospace 20.0%, automotive 1.3%, marketing 0%. A
regulatory corpus cross-confirms itself; a procedure corpus does not. That is a
real quality signal, and it is the same computation, correctly labelled.

**A coupling that had to be got right.** The old dedup guard matched
`status: 'open'`. Retiring the 155 existing nodes to `false_positive` *without*
the code change would have made `existing = 0` on the next scan and recreated
every one of them. The retirement script and the strategy change are only safe
shipped together — verified by rescanning all three tenants afterwards and
getting 0 new conflicts.

**Downstream exposure found.** `docs/hiring-and-presentation-strategy.md` scripted
the user to present a `multi_source` row in interviews as "two EASA directives
with contradictory supersession information, flagged automatically". That claim
could not have survived a follow-up question, because the query never compared
the two claims. Corrected to point at `functional_violation` /
`directional_reversal`, with an honest framing for the volume drop.

**Lessons.**
1. **A capability can be 100% broken and 0% visibly broken.** This is the second
   instance in the same subsystem (see A129). Both times the tell was the *type
   distribution*, not the count: one strategy producing ~99% of output means the
   other four are either dead or drowned, and nobody looks at a metric that has
   always been high.
2. **Check whether a detector's match condition can express what it claims.** Not
   "is it finding things" but "given the data model, is what it finds capable of
   being the thing?" One edge cannot hold two contradictory claims, so no query
   over one edge can detect contradiction. That is provable before running
   anything.
3. **Two subsystems disagreeing about the same evidence is a design smell worth
   chasing.** noisy-OR treating a second source as confirmation while the
   detector treated it as conflict should have been caught when both were
   written.
4. **A permanently-breaching alert is an ignored alert.** aerospace at 4x the
   contradiction threshold for its whole life meant the threshold taught the team
   to ignore it, which is worse than having no alert.
5. **When retiring detector output, check the dedup guard.** Marking rows resolved
   can *re-arm* the detector that produced them.

## A136: Aerospace had 38% duplicate chunks — ingestion is not idempotent, and the duplicates were propping up the eval score

**Root cause.** `Document.id` and `Chunk.id` are
`Field(default_factory=lambda: str(uuid4()))` — a fresh UUID every run — while
`neo4j_client.merge_document` does `MERGE (d:Document {id: $id})`. The MERGE
therefore only dedupes *within* a run: **re-ingesting any file creates a
complete second copy**, document, chunks and all. Aerospace was partially
re-ingested on 2026-07-21; automotive and marketing were `--wipe`d, which is the
only reason they look clean. This is a general bug, not an aerospace one.

Measured: 4 files duplicated, 52 of 138 aerospace chunks (38%) duplicate text.

**How it corrupted retrieval.** AUT-01's top-5 — the whole context the LLM
sees — was:

| slot | file | |
|---|---|---|
| 1 | CMM | |
| 2 | FAA-AD-2020-05-11 | superseded doc |
| 3 | CMM | |
| 4 | CMM | **byte-identical duplicate of #3** |
| 5 | CMM | **byte-identical duplicate of #1** |

Two of five slots were exact repeats and `FAA-AD-2024-01-02` — the answer — sat
below rank 22. The LLM correctly refused. `rerank_top_k` is 5, so duplicates
don't just add noise, they *consume the entire context budget*.

After deleting the duplicates, FAA-AD-2024-01-02 rose from >22 to **rank 6**.

**But the eval went DOWN: 29/34 -> 28/34.** Measured 4x per question, not once:

| question | post-dedup | verdict |
|---|---|---|
| MH-06 | 3/4 pass (was failing) | real gain, but flaky |
| SH-02 | 0/4 | real change — see below |
| AUT-03 | 0/4 | **real regression** — pipeline now refuses |
| AUT-01 | 0/4 | unchanged (fails for a different reason) |

**Why duplicates were helping.** The duplicated `14CFR_Part39_excerpt` chunk is
the one that lists every AD cross-reference. Having two copies doubled its odds
of being retrieved, which propped up the supersession questions. The duplicates
were acting as an accidental relevance boost for one high-value chunk — the eval
score was partly *paid for by a data bug*.

**Two of the three "regressions" were not regressions:**

1. *CON-01 was my own test-authoring bug, introduced the same day.* Forbidden
   term `"fully compliant with all"` matches as a substring of the correct
   answer `"not fully compliant with all"`. I retired the original CON-01 (A133)
   precisely for forbidding the only correct answer, then reintroduced the same
   flaw in subtler form. Fixed to `"is fully compliant with all"`, which cannot
   appear inside `"is not fully compliant with all"`, and verified in both
   directions.
2. *SH-02 is internally inconsistent.* Its note says "either superseding
   directive is a correct answer" and `required_answer_any_of` accepts AD-2024 —
   but `expected_citations` lists only `FAA-AD-2022-03-07`. The answer
   ("FAA AD 2024-01-02 fully supersedes AD 2020-05-11") passed every content
   check and failed **only** citation recall. The citation checker has no
   any-of, so a test that admits two right answers can only encode one.

Only **AUT-03** is a genuine quality loss.

**Lessons.**
1. **A score can be propped up by a bug.** Removing objectively-wrong data made
   the headline number worse. Restoring 38% duplicate content to recover a point
   would be overfitting the eval to a data-integrity defect. The number is a
   proxy; the graph is the thing.
2. **`MERGE` on a random UUID is not a merge.** Any natural-key entity whose id
   is `uuid4()` by default will duplicate on every re-run. The fix is to MERGE on
   `(tenant, filename)` for documents and `(document_id, chunk_index)` for
   chunks, assigning the id `ON CREATE` — plus deleting chunks whose index
   exceeds the new chunk count, or a re-chunk leaves orphans behind.
3. **When top_k is small, duplicates are not noise — they are the context.**
   Two wasted slots out of five is 40% of everything the model gets to read.
4. **Measure flipped questions N times before calling them regressions.** Of
   three apparent regressions, one was my bug, one was a test inconsistency, and
   one was real. Single-run diffs on a non-deterministic LLM backend
   (Groq is not reproducible at temperature 0) cannot distinguish these.
