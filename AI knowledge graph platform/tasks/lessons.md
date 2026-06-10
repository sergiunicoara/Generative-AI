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

## A54 — Demo scripts: count every mock call before writing side_effect

**What happened:**
`demo_regulatory.py` step 4 (ForwardChainingEngine) and step 5 (ContradictionDetector)
both raised `StopIteration` because the `side_effect` list had fewer slots than
actual `await db.run()` calls. Step 4 needed 9 slots (4 rule queries per iteration ×
2 iterations + 1 MERGE write); step 5 needed 11 slots.

**Root cause:**
The mock was written from the outside ("query then create") rather than tracing the
actual code path. ForwardChainingEngine runs all rules in each fixpoint iteration, not
just the ones with matches.

**Rule:**
> Before writing `AsyncMock(side_effect=[...])` for a multi-step engine:
> 1. Read the source of the method under test and count every `await db.run()` call.
> 2. Note which calls are conditional (inside `for row in rows:`) — zero rows = zero
>    conditional calls.
> 3. Label each slot in a comment: `# iter1: supersedes_transitivity → 1 candidate`.
> 4. For fixpoint loops: slots = rules_per_iter × num_iterations + conditional_merges.
> StopIteration from a mock means you undercounted by exactly one call — add one slot
> at a time until the demo runs.

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

**What happened:**
Four tests in `test_safety_paths.py` had `side_effect` lists with phantom "CREATE" call
slots inserted after queries that returned `[]`.  When a query returns an empty list,
the `for row in rows:` loop body is never entered, so no CREATE is issued.  The tests
assumed a CREATE always follows a query, which drifted the mock's slot mapping —
the actual conflict row landed in the wrong slot, causing `KeyError` on unexpected
field names.

**Root cause:**
The mock sequence was designed assuming "query → create" pairs, but the real code
pattern is "query → *conditional* create per row".  Zero rows = zero creates.

**Rule:**
> When writing `AsyncMock(side_effect=[...])`, trace through the production code path
> explicitly:
> 1. List every `await db.run()` call and mark which ones are conditional (inside a
>    for-loop over rows).
> 2. Only include CREATE/UPDATE mocks when your mock rows will actually trigger that
>    code path.
> 3. Name each slot in a comment: `# 3: directional CREATE (only if row returned)`.
> Zero-row queries produce zero subsequent CREATE calls.  Every phantom slot shifts
> all subsequent slots by one, causing cascade KeyError failures on the next test run.


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

## A77 — `NOT property = true` is a null trap in Cypher

**What happened:**
`WHERE NOT e.quarantined = true` excluded all entities where `e.quarantined` was NULL
(property not set). In Cypher, `NULL = true` → NULL, `NOT NULL` → NULL, which is
falsy in WHERE — so all 374 entities were excluded and `entity_count=0` in snapshots.

**Root cause:**
Standard boolean negation doesn't work as expected when the property may be absent.

**Rule:**
> Never write `NOT e.property = true` in Neo4j Cypher when the property may be absent.
> Always write `coalesce(e.property, false) = false`.
> Same applies to `e.property = false` — use `coalesce(e.property, false) = false`.

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

## A81 — `NOT x.quarantined = true` is a null-trap in Cypher; use `coalesce`

**What happened:**
The Communities tab (incremental drift monitor) showed "0% — 0 changed entities" even though
all 374 entities were ingested after the last (non-existent) rebuild point. Root cause: three
Cypher queries in `incremental_community.py` used `WHERE NOT e.quarantined = true`. When the
`quarantined` property is absent (the common case — most entities are never quarantined), the
comparison evaluates to `NULL`, which is falsy in a `WHERE` clause, so the entire entity is
excluded. Result: 0 entities returned → 0% drift.

**Root cause:**
Neo4j Cypher null semantics: any comparison involving `NULL` returns `NULL`, not `false` and
not `true`. `NOT NULL = true` → `NOT NULL` → `NULL` (falsy). The fix is `coalesce(e.quarantined,
false) = false`.

**Rule:**
> **Every** Cypher `WHERE` clause that filters on an optional boolean property (`quarantined`,
> `deleted`, `archived`, etc.) must use `coalesce(prop, default) = expected`, not
> `NOT prop = true` or `prop = false`. Audit all new Cypher for this pattern before
> committing. The null-trap is invisible unless you query a graph where the property is absent.

**Scope of fix applied:**
- `incremental_community.py` — 3 occurrences (get_changed_entities × 2, should_full_rebuild total count)
- Previously fixed: `neo4j_client.py` (6 occurrences), `graph_snapshots.py` (2), `graph_evaluator.py` (2), `community_builder.py` (query in build_semantic_communities)

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
