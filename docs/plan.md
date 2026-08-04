# Sales Context Graph — plan review + Codex prompt

## Part 1 — Verdict on the spec

**The architecture is right. The scope is not an MVP.**

What the spec gets genuinely right, and should not be negotiated away:

- **A Claim layer instead of asserting extracted triples as facts.** This is the single most important decision in the document. Most "KG from transcripts" projects write `(Account)-[:HAS_BLOCKER]->(Blocker)` directly and become unauditable within a week.
- **CRM authority > transcript claims, without discarding transcript claims.** Correct. Conflicts are a feature to expose, not a bug to suppress.
- **A Context Graph that is a *query-specific subgraph*, distinct from the persistent KG.** This is the actual product differentiator and the thing most competitors do not have.
- **Human review band for ambiguous mentions.** Correct — silent auto-linking is how CRMs rot.
- **LLM behind typed provider interfaces, deterministic logic stays deterministic.** Correct.

But: 21 node types × 17 relationship types × 13 endpoints × 10 use cases × 8 eval metrics × 12 deliverables is a multi-month platform, handed over as a single prompt. Given to Codex as written, the predictable outcome is broad, shallow, and partly fabricated — stub functions that satisfy the file list, tests that assert `True`, and a README claiming features that do not run.

The fix is not to reduce ambition. It is to **build one vertical slice that actually works end-to-end, then widen**.

---

## Part 2 — Twelve defects to fix *before* handing this to any code generator

### D1. The resolver is specified as two incompatible architectures at once
Stages 1–7 read as a **sequential cascade** (try exact ID, then email, then name…), but "component scores / final score" implies a **weighted blend**. These are different systems with different failure modes.

**Fix — state it explicitly:** Stages 1–5 (external ID, email, domain, canonical name, known alias) are *deterministic short-circuits*: first hit returns immediately with a fixed score and `auto_linked`, no embeddings, no LLM. Stages 6–7 (fuzzy, embedding, relational) only run when every deterministic stage misses, and only those produce a blended score.

### D2. The threshold table silently defeats the demo requirement — this is the most important defect
The spec demands the Volkswagen case resolve "using multiple signals, **not only fuzzy string similarity**". But `token_set_ratio("Volks Wagen", "Volkswagen Group")` lands around 0.90 — so under the stated `≥0.90 → auto-link` rule, it auto-links **on fuzzy alone**. The flagship demo would pass while demonstrating the exact opposite of its thesis.

**Fix — lexical/semantic similarity alone must never auto-link:**
```
auto_linked   iff  deterministic_hit
              or  (score >= 0.90 AND relational_signals >= 1)
pending_review     score >= 0.90 with ZERO relational signals   ← the mimicry guard
pending_review     0.55 <= score < 0.90
unresolved         score < 0.55
```
And the fixtures must contain a **distractor account** (`Volkswagen Financial Services`) so fuzzy matching is genuinely ambiguous. Then add the negative test that makes the claim honest: *same mention, relational evidence removed → must land in `pending_review`, not `auto_linked`*. Without that test the demo is theatre.

### D3. The Claim model is missing fields its own requirements depend on
- **`claim_id`** — `GET /claims/{id}/evidence` is in the endpoint list, but the model has no id.
- **`workspace_id`** — the security section mandates every graph query be workspace-scoped; the Claim has no workspace.
- **`polarity` / negation** — "we do **not** have security sign-off" vs "we have security sign-off". Extraction that drops negation produces confidently inverted facts. This is the highest-risk correctness bug in the whole system and the spec does not mention it.
- **`speaker_role`** (buyer / seller / unknown) — a *buyer* saying "budget is approved" is evidence; a *seller* saying it is a hypothesis. The model stores `speaker` but the authority rules never use it.
- **transaction time** — only `valid_from`/`valid_to` are specified. Sales facts get corrected retroactively; without transaction time you cannot answer "what did we believe on the day we sent that proposal", which is exactly what a disputed deal review asks.
- **`supersedes` / `contradicts`** edges — conflict detection is required but has no representation.

### D4. `Mention` + `UnresolvedMention` as two labels is a modeling error
Resolution changes a mention's *status*, not its *identity*. Two labels forces relabeling nodes on resolve, which breaks relationships already pointing at them and wrecks idempotency.

**Fix:** one `Mention` node with `resolution_status ∈ {auto_linked, pending_review, unresolved, rejected}`. `UnresolvedMention` becomes a query, not a label. `GET /unresolved-mentions` is a filter.

### D5. Idempotency is asserted, never designed
"Restart-safe and idempotent" appears with no idempotency key anywhere. This exact failure has a scar in your current platform: `Document.id = uuid4()` combined with `MERGE` on that id meant every re-ingest created a **complete second copy** — 38% duplicate chunks in the aerospace tenant, silently propping up eval scores until measured (lesson A136).

Worth being precise about how that was actually fixed there, since it is the cheaper of two valid options: the ids stayed `uuid4()`, and the **MERGE keys** moved to natural keys — `merge_document` on `(tenant, filename)`, `merge_chunk` on `(document_id, chunk_index)`, with `ON CREATE SET c.id`. That works. But for a new build I'd go further and make the ids themselves deterministic, because a sales graph needs *stable cross-references* — a `Claim` pointing at a segment, an evidence link surviving re-ingest, a `mention_id` that encodes its own provenance span. Natural MERGE keys give you idempotent writes; deterministic ids give you idempotent writes **and** stable references.

**Fix — every node ID is deterministic and content-derived:**
```
CRM node          {workspace}:{source_system}:{object_type}:{external_id}
Conversation      {workspace}:{source_system}:{call_id}
TranscriptSegment {conversation_id}:{segment_index}
Mention           {segment_id}:{char_start}:{char_end}      ← provenance for free
Claim             sha256(workspace|subject|predicate|object|source_segment|extractor_version)
```
Claim IDs including extractor version is deliberate: re-running the *same* extractor is a no-op; a *new* extractor version produces new claims that the conflict layer adjudicates against the old ones. And there must be a test asserting node/relationship counts are **byte-identical** after a second identical ingest — not "roughly the same".

### D6. Speaker identity is itself an unsolved resolution problem
The spec assumes transcripts arrive with named speakers. Gong's API returns `transcript[] → {speakerId, topic, sentences[{start, end, text}]}` — `speakerId` is an **opaque numeric ID**, not a name. Mapping speaker → Contact/Seller requires its own resolution pass over the participant list, invitee emails, self-introductions, and talk-time heuristics. Every claim's authority depends on getting this right (see D3), so it cannot be an afterthought.

**Fix:** `resolve_speakers` is an explicit workflow stage, before claim construction, reusing the same resolver and the same confidence bands.

### D7. The context budget will produce redundant context
Selecting top-N nodes by score returns ten near-identical segments about the same objection, burning the token budget on one fact.

This is measured, not theoretical: your current platform has a document-coverage diversity step for exactly this reason — and when MMR was benchmarked *on top of* it, quality got **worse** (coverage 0.929 → 0.843, MRR 0.781 → 0.772 across 33 golden questions), because two diversity mechanisms optimizing different notions of "redundant" fight each other.

**Fix:** greedy selection by `score / token_cost` with **hard per-source caps** (max K segments per conversation, max M claims per predicate, max 1 claim per subject-predicate unless they conflict — conflicts are exactly what you *do* want both sides of). Do not add a similarity-based re-ranker on top.

### D8. LangGraph is over-specified for the workflow described
`validate → normalize → segment → extract → candidates → resolve → validate_claims → persist → report` is a straight line. No branches, no cycles, no agent loop. LangGraph earns its place for exactly one thing here: the **human-review interrupt** on unresolved mentions, plus resume-from-checkpoint.

**Fix:** keep LangGraph, but only if the checkpointer is genuinely used for the review interrupt. Otherwise it is ceremony. Either way it is *plumbing* — the resolver and the claim layer are the architecture.

### D9. The evaluation metrics are named but undefined
"Hallucination rate" is unmeasurable as written.

**Fix — define them operationally:**
- *Grounded-answer rate* = fraction of factual sentences in the answer that cite ≥1 `claim_id` present in the served context graph.
- *Hallucination* = any factual sentence whose cited claim set is empty, or cites a `claim_id` **not** in the served context.
- The golden set must carry **expected `claim_id`s per question**, not just expected prose — otherwise "context recall" has nothing to measure against.

### D10. The highest-value Showpad-native use case is missing
All ten listed use cases are read-only briefings. Showpad's *existing* content recommendation is driven by attribute matching — industry, deal stage, product interest, customer segment, pushed from CRM. A knowledge graph does not need to exist to do that.

**What the graph unlocks that attribute matching cannot:**
> *Recommend the asset that addresses the specific objection raised by this named stakeholder in the last call, on this opportunity, that this buyer has not already viewed.*

That is `Objection ←[:ADDRESSES]— ContentAsset`, filtered by engagement history, traced to the exact transcript segment where the objection was raised. Make this **use case #1**, not an afterthought — it is the demo that justifies the entire build.

### D11. Salesforce lead conversion is unmodeled
Converting a Lead creates Account + Contact + Opportunity. With no conversion edge you get Lead "Elena Popescu" and Contact "Elena Popescu" as two unrelated people — a guaranteed data-quality bug on any real Salesforce export.

**Fix:** `Lead -[:CONVERTED_TO]-> Contact|Account|Opportunity`, and the resolver must treat a converted Lead's identifiers as aliases of the resulting Contact.

### D12. `HAS_STAKEHOLDER` needs to carry the role
Salesforce models this as `OpportunityContactRole` — a junction carrying *Decision Maker / Economic Buyer / Influencer / Champion*. A stakeholder map without roles is just a contact list.

**Fix:** put `role`, `influence`, and `sentiment` on the relationship, sourced from CRM where available and from transcript claims where not — with the claim layer keeping both when they disagree.

---

## Part 3 — Research grounding (use these, not invented names)

**Showpad Content API objects** (developer.showpad.com):
| Object | Key fields |
|---|---|
| `Asset` | `id`, `name`, `division{id}`, `externalId`, `description`, `permissions{isShareable}`, `isSensitive`, `isArchived`, `tags[]`, `languages[].code`, `countries[].code`, `authors[]`, `createdAt`, `updatedAt` |
| `Tag` | `id`, `name`, `division{id}` |
| `Division` | `id` — Showpad's own org-segmentation dimension; maps naturally onto `workspace_id` |
| `User` | `id`, `userName`, `email`, `myUploadsDivisionId` |

**Showpad ↔ Salesforce behaviour:** email shares, link shares, Shared Space creation, and participant invites are logged against the corresponding Contact / Lead / Opportunity / Account. Shared Spaces resolve recommendations via *the most recent opportunity the space was logged to, falling back to the most recent account*. Content recommendation today is driven by industry, deal stage, product interest, and customer segment.

**Showpad 2026 positioning** — "AI-native revenue effectiveness platform", unifying Content Management, Sales Readiness, Buyer Engagement, and Revenue Intelligence into a "System of Execution". Genie Assistant, Roleplay AI, **Field Meeting AI** (pre-meeting: summarize docs, generate agendas and talking points, compile account history + content recommendations), plus Summer '26 Genie Agents, Agent Studio, and MCP Server integrations. *Field Meeting AI is almost exactly the spec's "pre-meeting preparation" use case — align the vocabulary with theirs.*

**Salesforce standard objects:** `Account`, `Contact`, `Lead`, `Opportunity`, `User` (= seller), `Event` (meeting), `Task` (activity), `OpportunityContactRole` (junction, carries the stakeholder role).

**Gong transcript shape:** `POST /v2/calls/transcript` → `transcript[] → {speakerId, topic, sentences[{start, end, text}]}`, plus internal/external participants, topics, trackers, scorecards. Note `speakerId` is opaque — see D6.

---

## Part 4 — Revised phase plan

The spec's P0→P4 ordering is correct. The problem is that P0–P4 as scoped produces a graph with **no query path**, so nothing is demonstrable until P5+. Fix by threading one thin read path through from the start.

| Phase | Scope | Done when |
|---|---|---|
| **P0** Contracts | Pydantic v2 models for CRM objects, transcript envelope, extraction output, `Claim`, `Mention`, resolution result. Deterministic ID functions. No DB, no LLM. | ID functions are pure + property-tested; every model round-trips through JSON |
| **P1** Graph schema | Constraints, indexes, `GraphSession` wrapper that **refuses any Cypher without `$workspace_id`**, repository interfaces. | A test greps every repository query and fails on any missing workspace param |
| **P2** CRM ingestion | Salesforce-shaped fixtures → MERGE-based repos, incl. `OpportunityContactRole` and `Lead CONVERTED_TO`. | Second identical ingest changes zero node/rel counts |
| **P3** Transcript extraction | Segmentation, fake deterministic extractor, real LLM adapter behind the same interface, **negation/polarity handling**, speaker resolution (D6). | Fake extractor gives byte-identical output across runs; polarity test passes |
| **P4** Entity resolution | Deterministic cascade + probabilistic blend + relational corroboration, per D1/D2. | **VW case auto-links with signals; same case *without* relational evidence lands in review** |
| **P4.5** Thin read path | `POST /context/build` + one use case: *content recommendation for the objection raised in the last call*. | Returns selected nodes, claims, evidence refs, per-element selection reasons |

Everything else — the other nine use cases, the full endpoint list, the eight eval metrics — is P5+. Ship P0–P4.5 working before widening.

---

## Part 5 — Reuse vs greenfield (grounded in an actual inventory of your platform)

I inventoried the existing repo rather than guessing. What you already have:

| Capability | Where | Reusable? |
|---|---|---|
| Multi-stage alias resolution | `graphrag/graph/alias_registry.py` — `AliasRegistry.resolve()`, exact → normalized → rapidfuzz → embedding-cosine via Neo4j vector index, config-driven thresholds, `AmbiguousMatch` band | **~85% as-is.** Strip the `_normalize_regulatory` EASA/FAA hook. **But:** fuzzy is an O(n) linear scan over all stored keys — fine for 3k entities, will not survive a real CRM account list. Needs a blocking/candidate-generation step first. |
| Human review queue | `graphrag/graph/review_queue.py` — `enqueue/approve/reject/list_pending`, `:ReviewQueueItem` nodes, approve auto-registers the alias | **Directly reusable.** This is `/unresolved-mentions/{id}/resolve` already built. |
| Claim / provenance | `Statement` in `core/models.py` + `graph/reification.py` — reified triples as first-class nodes that other statements can endorse/contradict | **Reusable.** This *is* the Claim layer, just named differently. |
| Bitemporal | `graph/bitemporal.py` — `as_of_*`, `transaction_diff`, `get_entity_history`, indexed on all four time fields | **Reusable, and better than the spec asks for.** |
| Conflict detection | `graph/contradiction_detector.py` + pluggable `contradiction_strategies.py` (~513 lines), first-class `:Conflict` nodes | **Reusable**, needs sales-specific strategy classes. |
| Ontology | `graph/ontology_registry.py` + six YAMLs in `config/ontologies/` declaring `type_hierarchy` and `relation_rules{domain,target}` | **Fully config-driven — adding a `sales` ontology is a YAML file.** |
| **ContextGraphBuilder** | **Does not exist.** `graphrag/context_graph/` is purely decision-trace/audit (`Case → AgentRun → ToolCall → Decision`). Zero hits for `token_budget`, `node_budget`, `selection_score`. | **Greenfield.** The thing the product is named after is the thing you have to build. |
| Multi-tenancy | `tenant` property + a `WHERE` predicate repeated across hundreds of query sites; enforcement is a *post-hoc audit script* (`scripts/verify_tenant_isolation.py`) | **Weakest area.** Convention, not a choke point — exactly why P1 mandates `GraphSession`. |
| LLM extraction | `ingestion/extractor.py` returns typed Pydantic, but the boundary is a raw JSON prompt and failures are **swallowed** (`return [], []` on `JSONDecodeError`). No retry, no repair loop, no fake extractor — determinism comes from an LLM cache. | **Do not port this part.** The spec's strict-validation + retry + fake-extractor design is a genuine improvement; build it properly. |

**Recommendation: fork into a new repo.**

Three reasons, in order of weight:
1. The `ContextGraphBuilder` — the actual differentiator — is greenfield under *either* option, so "reuse" saves less than it appears to.
2. The two weakest parts of the existing platform (tenancy-by-convention, extraction with swallowed errors) are precisely the two the new spec fixes. Inheriting them means inheriting the debt you set out to avoid.
3. A clean sales-domain repo demos far better than a compliance platform wearing a sales hat.

What to carry across: the alias registry (minus the regulatory hook, plus a blocking step), the review queue, reification, bitemporal, and the ontology-as-YAML pattern. That is a substantial head start on P4 in particular — you are porting a working resolver, not inventing one.

---

## Part 6 — Scalability and operational concerns

Being direct: Parts 1–5 cover **correctness** well, **scalability** thinly, and **operations** not at all. That is partly deliberate — an MVP that ships beats an MVP that scales to a load it does not yet have. But "deliberate" only holds if the ceilings are *chosen* rather than discovered later, and if the decisions that are cheap now and expensive later are made now.

### 6a. The scale ceilings you are accepting, and where each one bites

| Ceiling | Bites at roughly | Symptom when it does |
|---|---|---|
| Fuzzy resolution is O(n) over all candidates | ~5–10k accounts | Resolution latency grows linearly per mention; ingestion stalls. *(Blocking step, already in the prompt, is the fix.)* |
| Vector index post-filters by workspace | 2+ workspaces of comparable size | **Silent wrongness, not slowness** — see 6b#3. |
| Single serialized graph writer | ~10s of transcripts/min | Ingestion backlog; no backpressure signal. |
| Per-mention embedding calls | 1k+ mentions/batch | Embedding API cost and rate limits dominate ingest time. |
| Per-segment LLM extraction | any real call volume | Cost is the binding constraint, not latency. A 60-min call ≈ 300+ segments. |
| Neo4j Community, one DB, `workspace_id` property | multi-customer deployment | No database-level isolation; a query-shaped bug is a cross-tenant leak. |
| Claims grow monotonically, never compacted | ~6 months of ingest | Traversals slow; "latest claim" queries scan history. |
| Account → everything traversal | ~500 conversations on one account | **Supernode.** Any traversal from a large Account explodes. |

### 6b. Cheap now, expensive later — these belong in the MVP

These are the ones I would not defer, because each is a few hours now and a migration later.

1. **Do not hang Claims and Mentions off `Account` directly.** Route them through `Conversation` and `Opportunity`. An enterprise account accumulates thousands of conversations; a direct `Account -[:HAS_CLAIM]-> Claim` fan-out makes every account-scoped traversal a supernode scan. Changing this after data exists is a full re-model.

2. **Batch-shaped embedding and extraction interfaces from day one.** `embed(texts: list[str]) -> list[Vector]`, not `embed(text) -> Vector`. The single-item signature leaks into every call site and retrofitting batching means touching all of them. Same for the extractor.

3. **Pre-filter the vector index by workspace, do not post-filter.** Your current platform has this exact bug documented (A146): `db.index.vector.queryNodes` returns the *global* top-k across all tenants and the `WHERE tenant = $x` runs after — so a workspace gets starved out of its own results by another workspace's higher-scoring nodes. It produced a live `no_communities` failure. The mitigation there was over-fetching (`fetch_k = max(top_k*20, 100)`); the real fix is a pre-filtered index. Either way, decide deliberately — this fails *silently and wrongly*, which is worse than failing slowly.

4. **Design erasure into provenance from the start.** Transcripts are personal data. GDPR erasure against a claim graph is genuinely hard: you must delete the segment text, invalidate derived claims, and *retain the audit record that erasure happened* — all without breaking the evidence links that make the rest of the graph trustworthy. Add `retention_class` and `erasure_status` to segments and claims now. Retrofitting this is close to impossible.

5. **Claim supersession needs a compaction policy, not just a relationship.** Decide now whether "current truth" is a query over the claim chain (simple, slow later) or a materialized `is_current` flag maintained on write (slightly more work now, orders of magnitude faster later). Recommend the flag, with the chain retained for audit.

6. **Salesforce account merges must propagate.** Merging two accounts is a routine CRM event. Every resolved mention, claim, and alias pointing at the losing account must follow. Model it as an explicit `MERGED_INTO` edge with the resolver treating the loser's identifiers as aliases of the winner — not as a destructive rewrite.

7. **Write concurrency model, decided explicitly.** Your existing platform found that parallel Neo4j writes caused `EntityNotFound` and settled on concurrent extraction feeding a *single serialized writer*. That is a real, earned answer — adopt it deliberately rather than rediscovering it under load.

### 6c. Deferred to P5+, with the trigger that should pull each one forward

Matching the "when to add X" convention already in your roadmap, so these are threshold-triggered rather than vague:

- **Ingestion worker pool + queue** — when transcript backlog exceeds the freshness SLA (a pre-meeting brief is worthless if it lands after the meeting).
- **Neo4j Enterprise / database-per-workspace** — when a second paying customer's data lands in the same instance.
- **Read-path caching** — when p95 on `/context/build` exceeds the interactive budget; measure before caching, since traversal cost is usually dominated by one bad query, not by breadth.
- **Claim compaction / archival** — when "current truth" queries start scanning more history than present state.
- **Cost governance per workspace** — before the first customer whose call volume is 10× the others.
- **Deal-level ACLs beyond workspace** — when "can seller A see seller B's calls" is first asked, which it will be. Showpad's own `Division` is the natural boundary to mirror.

### 6d. Extraction cost — window first, filter second, and measure the filter

Per-segment LLM extraction is the binding constraint on this system. A 60-minute call is 300–500 segments, and almost none of them carry a claim. Two levers, in strict order of value.

**Lever 1 — windowing (pure win, no recall tradeoff).** Do not extract per segment. Group contiguous segments into windows by topic boundary / speaker turn / ~60–90s, and extract once per window: ~300 segments → ~25 windows, roughly 10× cheaper.

The reason this costs nothing in fidelity: **extraction granularity and provenance granularity are separable.** The extraction schema already returns `supporting_segment_indices`, so you extract over a window and still cite the exact segment. Do this before any filtering — it is the larger win *and* the safe one.

**Lever 2 — filtering (real recall tradeoff; earn it with measurement).** Cheapest signals first:

| Tier | Signal | Marginal cost |
|---|---|---|
| 0 | Structural: drop sub-N-word backchannel ("yeah", "mhm"), screen-share/mute logistics | free |
| 1 | **Gong's own `topic` per segment + `trackers`** — its keyword detectors, already computed and already paid for. Consume these as a prior instead of building an objection/pricing/competitor lexicon from scratch. | free |
| 2 | Embedding proximity to per-claim-type prototype vectors — you are embedding segments anyway for the KG | ≈ 0 |
| 3 | Small-model binary gate ("does this window contain an extractable claim?") before the expensive model | ~10% of full extraction; adopt only if measured recall holds |

**Do not trim the tail of a call.** An "opening and closing pleasantries" heuristic is the obvious first filter and it is wrong: commitments, action items, and deadlines cluster in the final two minutes. Trim the opening cautiously; leave the close alone.

**Filter the extraction, never the storage.** Persist every `TranscriptSegment` regardless of whether it was extracted from. Text is cheap; LLM calls are not. This keeps full-text search complete, keeps provenance intact, and lets you re-run extraction under a better filter later without re-ingesting. Mark skipped segments with `extraction_status = skipped_by_filter` and the filter version that skipped them.

**Never apply these heuristics on the read path.** Extraction-time filtering optimizes cost against a corpus; read-time selection optimizes relevance against a single question. Different tradeoffs, different failure modes — do not share the code.

**The measurement protocol is not optional.** Build the golden set with *unfiltered* extraction as ground truth first, then report every tier as **cost reduction % vs. claim recall %**. A filter saving 60% at 99% claim recall ships; one saving 70% at 85% does not. A dropped blocker is a wrong graph, and unlike a slow query it never announces itself — which is exactly why this has to be measured rather than reasoned about.

**Do the arithmetic before building any of this.** Windowing alone is ~10×; tiers 0–2 plausibly another 2–4× on top. That is 20–40× off the naive per-segment baseline, which may well move extraction from "binding constraint" to "not worth optimizing further" — in which case tier 3 (the small-model gate) should never be built. A back-of-envelope against real expected call volume costs an hour and can delete a work item. See Part 7, item 1.

### 6e. Observability — genuinely missing, and cheap

The original spec says "OpenTelemetry-ready" and stops there. The minimum that earns its keep in an MVP: a span per workflow stage with the ingestion ID, structured resolution decisions (mention, chosen entity, score, signals fired, status) as queryable events rather than log prose, and per-workspace counters for claims written / mentions unresolved / conflicts opened. That last triple is your data-quality dashboard, and it costs almost nothing to emit at the point the decisions are already being made.

---

## Part 7 — Open decisions: resolve these *before* running the prompt

Every one of these is currently unstated. A code generator does not ask — it invents an answer, commits to it across dozens of files, and you discover the choice later as a refactor. Each is a five-minute decision now.

**1. Cost model — the one that can delete work.** How many calls/month, average duration? Windowing (~10×) plus filter tiers 0–2 (~2–4×) is plausibly 20–40× off the naive baseline. If that lands you comfortably under budget, tier 3 (small-model gate) should never be built, and §6d shrinks by a third. *Do this arithmetic first; it is the cheapest possible scope cut.*

**2. Which CRM?** The prompt assumes Salesforce (`OpportunityContactRole`, lead conversion semantics, `Event`/`Task`). Showpad also integrates Dynamics and HubSpot, whose object models differ meaningfully — HubSpot has no direct `OpportunityContactRole` equivalent. *Recommendation: commit to Salesforce for the slice and put the CRM adapter behind an interface, so a second CRM is a new adapter rather than a re-model.*

**3. Which transcript source?** The prompt assumes Gong's shape (`speakerId`, `sentences[{start,end,text}]`, topics, trackers). Chorus and Showpad's own Field Meeting AI capture differ. This matters more than it looks — §6d's tier-1 filter is *free* only if the source supplies trackers/topics. *Recommendation: Gong, with the same adapter-interface treatment.*

**4. Fork or extend?** Part 5 recommends forking, but it is your call and it changes the first day of work. Unstated, Codex will build greenfield by default.

**5. Does `workspace_id` == Showpad `Division`?** Showpad's Division is already an org-segmentation dimension with its own permissions. If they are the same concept, say so and inherit its semantics; if not, you have two overlapping isolation dimensions and need to state how they compose. *Getting this wrong is a security-boundary bug, not a modeling preference.*

**6. Embedding provider and dimensionality.** "Pluggable" is specified; the actual default is not. It determines index configuration and cost per mention. Also unstated: *what text gets embedded* for resolution — bare entity name, or name + industry + domain? Different recall profiles. The prompt currently specifies the latter; confirm it.

**7. Keep LangGraph?** Defect D8: justified only if the checkpointer genuinely serves the human-review interrupt. If you would rather ship a plain async pipeline for the slice, say so — otherwise you get the dependency and the ceremony regardless.

**8. Is the review queue in scope for the slice?** `POST /unresolved-mentions/{id}/resolve` is in the P4.5 endpoint list, but a *usable* review workflow (queue UI, assignment, audit of overrides) is a product surface of its own. *Recommendation: API only for the slice; no UI.*

---

## Part 8 — The Codex prompt

Everything below the line is the prompt. It is deliberately narrower than the original spec and resolves every ambiguity above, because an unresolved ambiguity handed to a code generator becomes an invented answer.

**Before running it:** settle Part 7 and paste the answers into the `STACK`/`SCOPE` sections. The prompt is written to be self-contained *given those decisions*; without them it will still run, but items 2, 3, and 5 in particular will be answered by invention.

---

# TASK

Build the first working vertical slice of **Sales Context Graph**: a system that ingests Salesforce-shaped CRM data and Gong-shaped sales-call transcripts into a Neo4j knowledge graph, resolves approximate entity mentions to canonical CRM records using multiple independent signals, and serves query-specific context subgraphs to sales teams.

Inspect the repository fully before editing. Preserve anything that already works.

## SCOPE — build exactly this, nothing more

Phases P0 through P4.5. Do **not** build the remaining use cases, the full endpoint list, or the full evaluation suite. A narrow slice that runs beats a broad slice that does not.

## STACK

Python 3.12+, FastAPI, Pydantic v2, Neo4j + official driver, LangGraph, RapidFuzz, pluggable embedding provider, pytest, Docker Compose, structlog. Add nothing else without justifying it in a comment.

## NON-NEGOTIABLE RULES

1. **Never claim a feature works without a test that fails when the feature is removed.** No `assert True`. No stub that returns a hardcoded happy path.
2. **Deterministic logic stays deterministic.** The LLM extracts; it never resolves entities, never scores, never writes to the database.
3. **Every Cypher query is parameterized and workspace-scoped.** No f-string interpolation of user values into Cypher, ever.
4. **Every node ID is deterministic and content-derived** (scheme below). Re-ingesting identical input must change zero counts.
5. Type hints throughout. `async` only where it buys something real.
6. Comment only where the reasoning is non-obvious — not what the code does, why it does it that way.

## SCALE-CRITICAL CONSTRAINTS — obey these even though this is an MVP

Each of these costs a little now and is a migration later. They are not optimizations; they are shape decisions.

1. **Never attach `Claim` or `Mention` directly to `Account`.** Route through `Conversation` or `Opportunity`. A large account accumulates thousands of conversations, and a direct fan-out makes every account-scoped traversal a supernode scan.
2. **Batch-shaped interfaces from the start:** `embed(texts: list[str]) -> list[Vector]` and an extractor that accepts a list of segments. Never a single-item signature — it leaks into every call site.
3. **Pre-filter the vector index by `workspace_id`; do not post-filter.** `db.index.vector.queryNodes` returns the *global* top-k and applies `WHERE` afterwards, which starves a workspace of its own results when another has higher-scoring nodes. This fails silently and wrongly. If your Neo4j version cannot pre-filter, over-fetch (`fetch_k = max(top_k * 20, 100)`) and leave a comment saying why.
4. **Add `retention_class` and `erasure_status` to `TranscriptSegment` and `Claim` now.** Transcripts are personal data; GDPR erasure must delete segment text, invalidate derived claims, and retain the audit record that erasure occurred. This cannot be retrofitted.
5. **Maintain an `is_current` boolean on claims on write**, alongside the `SUPERSEDES` chain. The chain is the audit trail; the flag is what queries use.
6. **Model Salesforce account merges as `MERGED_INTO`**, with the resolver treating the losing account's identifiers as aliases of the winner. Never destructively rewrite.
7. **Concurrent extraction, single serialized graph writer.** Parallel Neo4j writes on an interdependent graph produce `EntityNotFound` races. Extract in parallel, funnel writes through one queue.
8. **Observability minimum:** one span per workflow stage carrying the ingestion ID; every resolution decision emitted as a structured event (mention, chosen entity, component scores, signals fired, status) rather than log prose; per-workspace counters for claims written / mentions unresolved / conflicts opened.

## P0 — DATA CONTRACTS

Pydantic v2 models, no DB and no LLM in this phase.

CRM (Salesforce-shaped): `Account`, `Contact`, `Lead`, `Opportunity`, `Seller`, `Meeting`, `Activity`, `OpportunityContactRole` (carries `role`, `influence`).
Add `Lead.converted_to_contact_id` / `converted_to_account_id` / `converted_to_opportunity_id`.

Transcript (Gong-shaped): `Conversation{id, external_id, started_at, duration_s, participants[]}`, `Participant{speaker_id, name?, email?, is_internal}`, `TranscriptSegment{index, speaker_id, start_ms, end_ms, text, topic?}`.
**`speaker_id` is opaque** — names are not guaranteed.

Showpad content: `ContentAsset{id, name, division_id, external_id, tags[], languages[], countries[], is_sensitive, is_archived}`, `Tag`, `Division`, `Share{asset_id, shared_with_contact_id, opportunity_id?, shared_at}`, `AssetView{asset_id, viewer_contact_id, viewed_at, duration_s}`.

Extraction output: `ExtractionResult` containing entity mentions (with `char_start`/`char_end`), products, features, pain points, objections, blockers, buying signals, action items, commitments, stakeholders, deadlines, each with `confidence`, `polarity ∈ {affirmed, negated, hypothetical}`, and `supporting_segment_indices[]`.

**Claim** — exactly these fields:
```
claim_id, workspace_id, subject_id, predicate, object_id | object_value,
polarity, source_type, source_id, source_segment_id, source_timestamp,
speaker_id, speaker_role,           # buyer | seller | unknown
confidence, extraction_method, extractor_version,
valid_from, valid_to,               # when the fact is true in the world
transaction_from, transaction_to,   # when we believed it
review_status, created_at
```
Plus `SUPERSEDES` and `CONTRADICTS` relationships between claims.

**Mention** — one node type only:
```
mention_id, workspace_id, segment_id, char_start, char_end, surface_text,
entity_type, resolution_status ∈ {auto_linked, pending_review, unresolved, rejected},
resolved_entity_id?, score?, explanation
```
There is no separate `UnresolvedMention` label. Unresolved is a *status*, queried by filter.

**Deterministic IDs** — pure functions, property-tested:
```
crm_node_id(workspace, source_system, object_type, external_id)
conversation_id(workspace, source_system, call_id)
segment_id(conversation_id, segment_index)
mention_id(segment_id, char_start, char_end)
claim_id(workspace, subject, predicate, object, source_segment_id, extractor_version)   # sha256
```

**P0 acceptance:** ID functions are pure and stable across processes; every model round-trips through JSON without loss.

## P1 — GRAPH SCHEMA

Uniqueness constraints on every node ID. Indexes on `workspace_id`, `Account.canonical_name`, `Contact.email`, `Conversation.started_at`. Full-text index on `TranscriptSegment.text`.

Implement a `GraphSession` wrapper that is the **only** way the application executes Cypher, and that **raises if a query does not carry a `$workspace_id` parameter**. Repository interfaces live separately from service logic. Retry transient Neo4j failures with bounded exponential backoff; do not retry constraint violations.

**P1 acceptance:** a test enumerates every query in every repository and fails on any one missing `$workspace_id`. This is a real test that reads the source, not a convention documented in a README.

## P2 — CRM INGESTION

Idempotent MERGE-based repositories. Explicit transaction boundaries. Relationships: `HAS_CONTACT`, `HAS_OPPORTUNITY`, `HAS_STAKEHOLDER` (carrying `role`, `influence`), `CONVERTED_TO`, `PARTICIPATED_IN`, `CONCERNS_PRODUCT`, `OWNS` (seller→opportunity).

**P2 acceptance:** ingest the fixture set twice; assert node counts, relationship counts, and every node ID are **identical** after the second run.

## P3 — TRANSCRIPT EXTRACTION

Deterministic segmentation (same input → same segment boundaries → same `segment_id`s).

A provider interface with two implementations: a **fake deterministic extractor** driven by fixtures (used by every test) and a **real LLM adapter**. The adapter validates strictly against the Pydantic schema, retries on invalid output with bounded attempts, raises on persistent invalid output, and **never touches the database**.

**Extract over windows, not single segments.** Group contiguous segments into windows (topic boundary / speaker turn / ~60–90s) and make one extraction call per window. Extraction granularity and provenance granularity are separable: the result still carries `supporting_segment_indices`, so claims cite exact segments. This is a ~10× cost reduction with no fidelity loss and must be the default, not an optimization added later.

**Filter which windows get extracted, in this order:** structural drops (sub-N-word backchannel, screen-share/mute logistics) → Gong's own per-segment `topic` and `trackers` where present (already computed — consume them rather than rebuilding a keyword lexicon) → embedding proximity to per-claim-type prototype vectors. Make the filter a named, versioned, injectable component so it can be swapped and A/B'd; a `NullFilter` that extracts everything must exist and must be what the golden-set baseline runs.

Three rules on filtering:
- **Do not trim the end of a call.** Commitments, action items, and deadlines cluster in the final minutes. Trim openings cautiously; leave closes alone.
- **Filter extraction, never storage.** Persist every `TranscriptSegment` regardless, marked `extraction_status ∈ {extracted, skipped_by_filter}` plus the filter version. Re-running extraction under a better filter must not require re-ingesting.
- **This filter is write-path only.** Never import it into the read path — `ContextGraphBuilder` selection is a different tradeoff.

**Handle negation.** "We do not have security sign-off" must produce `polarity=negated`, not an affirmed blocker-cleared claim. Include explicit negation test cases.

**Resolve speakers before building claims.** Map opaque `speaker_id` → Contact or Seller using participant emails, meeting invitees, and self-introductions, through the *same* resolver and the *same* confidence bands as entity resolution. An unresolved speaker yields `speaker_role=unknown`, which lowers claim authority but does not drop the claim.

**P3 acceptance:** fake extractor output is byte-identical across runs; negation tests pass; a transcript with only opaque speaker IDs still produces claims, with `speaker_role=unknown` where resolution failed; windowed extraction produces claims whose `supporting_segment_indices` point at the correct individual segments; and a script reports, for each filter tier against the `NullFilter` baseline, **extraction calls saved (%) vs. claim recall (%)** — do not enable a filter tier by default until that number exists.

## P4 — ENTITY RESOLUTION

**Stage A — deterministic short-circuit.** In order; first hit returns immediately, no embeddings, no LLM:
```
A1  exact external_id, same source system      → 1.00, auto_linked
A2  exact normalized email                     → 1.00, auto_linked
A3  email domain == account domain             → 0.95, auto_linked   (Account only)
A4  exact canonical normalized name            → 0.95, auto_linked
A5  exact known alias                          → 0.95, auto_linked
```

**Candidate generation (blocking) — required before Stage B.** Do **not** run fuzzy matching against every entity in the workspace; that is an O(n) scan that dies on a real CRM account list. Generate a bounded candidate set first, via: normalized-name prefix/trigram index, email-domain match, and the vector index — union them, cap at 50 candidates, and only then score. Blocking recall is itself testable: assert the true match is present in the candidate set for every fixture case.

**Stage B — probabilistic.** Only when every Stage A rule misses, and only over the blocked candidate set:
```
lexical  = rapidfuzz.token_set_ratio(norm(mention), norm(candidate)) / 100
semantic = cosine(embed(mention + surrounding context), embed(candidate name + industry + domain))
base     = 0.6 * lexical + 0.4 * semantic
```

**Stage C — relational corroboration.** Each signal fires at most once:
```
+0.10  a seller in this conversation owns an open opportunity on the candidate account
+0.10  another already-resolved participant is a known contact of the candidate
+0.08  a product named in the segment matches a product on the candidate's open opportunity
+0.07  conversation timestamp within ±30 days of a meeting/activity on the candidate
+0.05  the candidate's account domain appears in participant emails

rel_bonus = min(sum, 0.25);  n_signals = count of signals that fired
score     = min(base + rel_bonus, 1.0)
```

**Decision rule — implement exactly this:**
```
deterministic hit                        → auto_linked
score >= 0.90 AND n_signals >= 1         → auto_linked
score >= 0.90 AND n_signals == 0         → pending_review     # lexical-mimicry guard
0.55 <= score < 0.90                     → pending_review
score < 0.55                             → unresolved
```
Similarity alone never auto-links. Thresholds and signal weights come from config, not literals in the resolver.

Return: `canonical_entity_id`, full candidate list, per-component scores, `n_signals`, final score, status, and a human-readable explanation naming which signals fired.

**P4 acceptance — three tests, all required:**
1. `"Volks Wagen"` resolves to `Volkswagen Group` as `auto_linked`, and the explanation names ≥2 distinct relational signals.
2. **The same mention, with relational evidence stripped from the fixture, lands in `pending_review`.** This is the test that proves the system is not a fuzzy matcher wearing a graph costume.
3. With distractor `Volkswagen Financial Services` present, the correct account wins *because of* relational signals — assert the distractor's score is lower and that it is not auto-linked.

## P4.5 — THIN READ PATH

`ContextGraphBuilder`. Inputs: question, `seller_id`, optional `account_id` / `opportunity_id` / `conversation_id`, time range, `max_nodes`, `max_tokens`.

Selection: deterministic filters → full-text/vector candidates → bounded graph traversal → score by `(relevance × source_authority × confidence × recency)`, then **greedy selection by `score / token_cost`** subject to hard caps: max K segments per conversation, max M claims per predicate, max 1 claim per subject-predicate **unless the claims conflict** — conflicting claims are exactly what must survive into the context. Do **not** add a similarity-based re-ranker on top of these caps.

Response includes selected nodes, selected relationships, claims, evidence references (segment ID + character span), unresolved mentions, detected conflicts, the selection score, and **the reason each element was selected**.

Endpoints for this phase only:
```
POST /api/v1/ingest/crm
POST /api/v1/ingest/transcripts
POST /api/v1/ingest/content-assets
GET  /api/v1/unresolved-mentions
POST /api/v1/unresolved-mentions/{id}/resolve
POST /api/v1/context/build
GET  /api/v1/claims/{id}/evidence
GET  /health
GET  /ready
```

**Use case to wire end-to-end — this one, first:** *given an opportunity, recommend the content asset that addresses the specific objection raised in the most recent call, excluding assets this buyer has already viewed, with the transcript segment that raised the objection as evidence.* Attribute matching cannot do this; the graph is the reason it works.

## LANGGRAPH WORKFLOW

```
validate_input → normalize_source → segment_transcript → extract_structured_information
→ resolve_speakers → generate_candidates → resolve_entities → validate_claims
→ persist_graph → compute_ingestion_report
```
Emit a structured event per transition. Use the checkpointer for genuine resume-after-interrupt on the human-review branch — if you are not using it for that, say so in a comment rather than adding ceremony. The workflow must be restartable and idempotent (which follows from deterministic IDs, not from workflow bookkeeping).

## FIXTURES

Realistic, and specifically constructed to make the demo honest:
- CRM `Account` "Volkswagen Group"; **distractor** `Account` "Volkswagen Financial Services"
- Transcript mentioning "Volks Wagen"
- Contact "Elena Popescu" on Volkswagen Group, appearing as a conversation participant
- An open opportunity owned by the seller on the call
- Product mentions for "Showpad Genie" and "Shared Spaces"
- A security-approval blocker and a pricing objection, **each with a negated variant** to exercise polarity
- An action item with a deadline
- Content assets tagged as addressing the pricing objection, plus engagement data showing one already viewed by the buyer
- One transcript with opaque `speaker_id`s only, no names

## SECURITY

Secrets from environment only, none committed. Input size limits. Transcript sanitization. `workspace_id` on every node and every query. Authorization hooks. PII-safe logging (never log transcript text or emails at INFO). Audit events for resolution decisions and manual overrides.

## DELIVERABLES

Working code; `README.md` with a Mermaid architecture diagram and setup; `architecture.md`; `ontology.md`; `entity-resolution.md` (documenting the scoring function and thresholds as implemented); sample data; Docker Compose for API + Neo4j; tests; a Makefile; example curl commands; and `demo_volkswagen.py` printing the resolution decision with **every component score and every relational signal that fired**.

## FINISHING

Run formatting, static analysis, and the full test suite. Fix failures before reporting done. Then report, accurately:
- files added / files modified
- commands executed
- tests passing (with real counts)
- **what is not implemented** — be specific and honest; an accurate gap list is more useful than an inflated completion claim
- the next recommended milestone
