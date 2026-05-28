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
