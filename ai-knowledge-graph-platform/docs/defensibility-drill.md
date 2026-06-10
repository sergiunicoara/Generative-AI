# Defensibility Drill — 15 Questions a CTO Will Ask

This document exists for one purpose: **to close the gap between what the
project demonstrates and what you can defend out loud, under follow-up.**

A polished deck sets a depth expectation. The questions below are what a
technically strong CTO at a Big 4 firm actually asks when they probe that
expectation. Each one has a model answer — not a script, but the reasoning
you should be able to reproduce unscripted from first principles.

Practice rule: close this document, pick a question at random, and answer it
to a blank wall for 90 seconds. If you can't, the project is a liability not
an asset at that depth. That's not harsh — it's the audit being honest.

---

## Category 1 — Architecture decisions (ADR territory)

### Q1. Why Neo4j instead of a pure vector database like Pinecone or Weaviate?

**What they're testing:** Do you understand the actual trade-off, or did you just pick Neo4j because it's popular?

**Model answer:**

Vector databases answer one question well: "which chunks are semantically similar to this query?" That's useful but insufficient for complex enterprise knowledge.

Three things a vector index can't do that drove the choice:

1. **Multi-hop reasoning.** A contract references a regulation which applies to a company. Finding that connection requires traversing edges, not ranking embeddings. `MATCH (doc)-[:CITES]->(reg)-[:APPLIES_TO]->(company)` is a natural graph query; it's impossible in a flat embedding space.

2. **Contradiction detection.** Two documents make contradictory claims about the same entity. Detecting this requires knowing which entities are the same entity (entity resolution) and comparing their properties across source documents. In a vector store, two similar chunks are just similar — there's no identity layer to compare against.

3. **Inference and reasoning.** The forward-chaining engine derives new facts from existing ones (AD-2024 supersedes AD-2022 supersedes AD-2020 → AD-2024 transitively supersedes AD-2020). This requires the asserted edges to exist as first-class data that a rule engine can query and extend. Vectors are distances, not facts.

The cost: higher operational complexity than a vector-only stack. The trade-off is documented in `docs/adr/0001-property-graph-over-triple-store.md`.

**The follow-up they might ask:** "What about a hybrid — vector DB for retrieval, separate graph for reasoning?"

Yes, that's a valid architecture and the platform essentially implements it: OpenAI `text-embedding-3-large` embeddings (3072d) in Neo4j's native vector index for ANN search, plus the property graph for entity resolution, inference, and contradiction detection. The decision was to keep both in the same store rather than operate two separate systems, which simplifies consistency guarantees at the cost of Neo4j's vector index being less tunable than a dedicated vector DB.

---

### Q2. Why forward-chaining inference instead of query-time backward-chaining?

**What they're testing:** Do you understand the fundamentals of inference strategies, or did you just implement something?

**Model answer:**

Forward-chaining materialises derived facts at write time — the rule fires once when the data changes and writes the inferred edge into the graph. Backward-chaining derives facts at query time — when you ask a question, the engine traces backwards through rules to find what could be true.

Three reasons forward-chaining was chosen:

1. **Query latency.** A user asks "which AD currently governs this component?" If backward-chaining must traverse the full supersession graph at query time, that's unbounded computation at read time. With forward-chaining, the `[inferred]` edge is already written — the query resolves in one hop.

2. **Explainability.** Materialised inferred edges have a `source_type=inferred` property, a `confidence` value (decayed per hop), and are first-class nodes in the graph. You can show a CTO or regulator exactly what was inferred, when, with what confidence, and from which asserted edges. Backward-chaining produces answers with no persistent provenance.

3. **Contradiction detection depends on materialised facts.** The contradiction detector scans existing edges. If inferred facts don't exist as edges, contradictions between asserted and inferred facts are invisible.

The trade-off: storage overhead (inferred edges are written to disk), and the graph must be re-inferred when source data changes. The `run_for_document(doc_id)` method scopes re-inference to the affected subgraph so full rebuilds are rare. Documented in `docs/adr/0002-forward-chaining-over-backward-chaining.md`.

---

### Q3. Explain the Bayesian confidence formula: `1 − (1−c₁)(1−c₂)`. Why not just average?

**What they're testing:** Can you justify the math, or is it cargo-culted?

**Model answer:**

This is confidence fusion across multiple independent sources. When two documents both assert the same relation, we want to combine their confidences into a single score.

Averaging assumes the two sources are measuring the same underlying quantity, which is approximately correct but has two problems:

1. **Two high-confidence independent sources should be more certain than either alone.** If doc A says relation R with confidence 0.9, and doc B independently says the same thing with confidence 0.9, the fused confidence should be higher than 0.9 — not still 0.9. Averaging gives 0.9. The Bayesian formula gives 0.99.

2. **Monotonicity.** Adding a new supporting source should never decrease confidence. Averaging is monotonic, but only if the new source's confidence is above the current average. The Bayesian formula is always monotonic — adding any source with `c > 0` increases the fused score.

The formula comes from independent probability:

```
P(at least one correct) = 1 − P(all wrong)
                        = 1 − (1−c₁)(1−c₂)
```

This assumes the sources are independent, which is an approximation — in practice, two documents from the same organization have correlated errors. The platform doesn't model correlation, which means confidence may be slightly overestimated when sources are related. That's a known limitation documented in the code.

Documented in `docs/adr/0003-bayesian-confidence-accumulation.md`.

---

## Category 2 — Retrieval pipeline

### Q4. Walk me through the 6 retrieval stages. Why each one?

**What they're testing:** Can you articulate the pipeline without notes, and do you know what each stage is buying?

**Model answer:**

1. **Vector ANN (3072d OpenAI `text-embedding-3-large`, cosine).** Finds semantically similar chunks. Fast, captures meaning but misses exact terminology. The 3072-dimension embeddings are higher quality than standard 768d models for domain-specific text.

2. **BM25 keyword search.** Finds chunks containing exact domain terms — "FAA-AD-2024-01-02", "engine mount inspection", regulation codes. Vector search misses these because sparse signals get diluted. Combined with RRF (Reciprocal Rank Fusion) to merge the two ranked lists without needing score normalization.

3. **Cross-encoder reranking.** The first two stages are cheap but approximate. `ms-marco-MiniLM-L-6-v2` takes the top-N chunks and scores each (query, chunk) pair jointly — much more accurate than bi-encoder similarity but too slow to run on all chunks. Runs on the top 20–50 from stages 1+2.

4. **Multi-hop graph traversal.** For each top-ranked chunk, find the entities it mentions, then follow their RELATES_TO edges to find chunks that mention related entities but weren't in the initial results. Answers questions like "what do we know about companies related to this regulation?" that no single chunk contains.

5. **GNN scoring.** Graph Convolutional Network / Graph Attention Network message-passing over the entity subgraph. Aggregates neighbourhood information into entity embeddings, then scores chunks by proximity of their entity mentions to the query entity in embedding space. Captures structural relevance, not just textual similarity.

6. **LLM synthesis (Groq llama-3.3-70b).** Takes the final ranked context and generates a grounded, cited answer. The model is instructed to only use context, cite by chunk ID, and signal uncertainty explicitly.

Each stage reduces a different failure mode of the previous one. No single stage is sufficient; the pipeline is the point.

---

### Q5. What's IRCoT and when does the agentic fallback trigger?

**What they're testing:** Do you understand the difference between retrieval and reasoning?

**Model answer:**

IRCoT (Interleaved Retrieval and Chain-of-Thought) is an iterative retrieval strategy: instead of a single retrieval pass, the system alternates between retrieving evidence and reasoning about what evidence it still needs.

The flow in `agentic_retriever.py`:
1. Initial retrieval on the original question
2. Ask a **fast 8B model** (llama-3.1-8b-instant): "Can you answer? If not, what do you need?"
3. If it says `ANSWER:`, return it (still fast — sub-second reasoning step)
4. If it says `SEARCH: <sub-query>`, retrieve on that sub-query, add new chunks to context
5. Repeat up to 2 steps, then synthesize with the **full 70B model** for quality

The trigger in `hybrid_retriever.py`: fires when the initial answer contains hedge language
**and** has zero citations. Both signals must be present — hedge-only or zero-citations-only
are not sufficient (on sparse corpora many confident answers have no citation IDs yet).

Measured trigger rate: ~9% of queries. Agentic p95: **3.4s**. The two-model design
(8B for routing, 70B for synthesis) keeps this well below the previous 6.8s.
Hybrid p95 is unchanged at **2.2s**. Combined p95: **2.7s**.

The key follow-up defence: "We don't alert on agentic latency — we alert on agentic
**rate**. If >20% of queries trigger the fallback, the threshold is too loose, not the
corpus is hard. At 9% we're in the right zone."

---

## Category 3 — Knowledge graph depth

### Q6. What is entity resolution and why is it hard?

**What they're testing:** Whether you understand this is a core unsolved problem, not a checkbox.

**Model answer:**

Entity resolution is the problem of recognising that "SpaceX", "Space Exploration Technologies Corp.", and "Space Exploration Technologies" all refer to the same real-world entity, and merging them into a single canonical node rather than creating three separate nodes.

Why it matters: if each variant creates a separate node, the graph is fragmented. Multi-hop traversal fails (you follow edges from "SpaceX" but the acquisition document references "Space Exploration Technologies Corp."). Community detection splits one company into three clusters. Contradiction detection misses conflicts between documents that use different name variants.

Why it's hard: there's a precision-recall trade-off with no perfect threshold. Over-merge (too aggressive) and you collapse genuinely different entities — "Apple" the company vs "Apple" the fruit. Under-merge and the graph stays fragmented.

The platform implements 4 stages with increasing cost:

1. **Exact match** — same normalized string, same type → definitely the same entity
2. **Normalized match** — strip punctuation, lowercase, collapse whitespace → catches formatting variants
3. **Embedding similarity** — cosine > 0.92 → catches semantic variants, blocked by type mismatch
4. **Ambiguous queue** — cases that score 0.80–0.92 are queued for human review rather than auto-merged

The 0.92 threshold is calibrated conservatively — false merges are worse than false separations because merges are hard to undo without corrupting provenance.

---

### Q7. What is an ontology and why does it need to be in a YAML file, not hardcoded?

**What they're testing:** Do you understand domain-agnosticism as an engineering property?

**Model answer:**

An ontology is a formal definition of the concepts (entity types), relationships (relation types with domain/range constraints), and inference rules in a domain. It's the schema of the knowledge domain, distinct from the database schema.

Why YAML instead of hardcoded: the platform is meant to be domain-agnostic. The aerospace regulatory ontology is a demonstration domain. A PwC client might need a banking regulatory ontology (instruments, institutions, regulations, directives) or a healthcare ontology (drugs, conditions, procedures, guidelines). The architecture separates domain knowledge (YAML) from platform logic (Python) so that onboarding a new domain is a configuration change, not a code change.

Concretely: `config/ontologies/aerospace_regulatory.yml` defines 28 type hierarchy pairs and 12 relation rules. Changing to `banking_regulatory.yml` with different types and relations requires zero Python changes. The `OntologyRegistry` and `TypeTaxonomy` load whatever the YAML contains.

The inference rules in the YAML (`supersedes_transitivity`, `mandated_by_inverse`) are also domain-specific — different domains have different reasoning patterns. Making them configurable means the forward-chaining engine is domain-agnostic too.

---

### Q8. What's a contradiction and how does the detector find it? What types exist?

**What they're testing:** Whether you understand the nuance, not just the demo scenario.

**Model answer:**

A contradiction in the knowledge graph is when two or more sources make mutually exclusive factual claims about the same entity. Five types are detected:

1. **Multi-source conflict** — the same (source, relation, target) triple appears with different properties (e.g., confidence, validity period) from two different documents. Subtle inconsistency, not outright contradiction.

2. **Directional reversal** — doc A says X SUPERVISES Y, doc B says Y SUPERVISES X. Logically impossible for a strict hierarchy.

3. **Exclusive state** — the same entity is tagged with two mutually exclusive states in the YAML ontology: `IS_AIRWORTHY` / `IS_UNAIRWORTHY`, `IS_CERTIFIED` / `IS_DECERTIFIED`, `MANDATORY` / `ADVISORY`. This is the demo scenario.

4. **Functional violation** — a functional relation (one-to-one by domain constraint) appears with multiple targets. For example, `CEO_OF` should have exactly one source entity per target company. If two documents name different CEOs, that's a violation.

5. **Positive/negative pair** — one document asserts a relation; another explicitly negates it with a `NOT_` variant (e.g., `COMPLIANT_WITH` vs `NOT_COMPLIANT_WITH`).

The detector runs post-ingestion scoped to the newly-ingested document's entities, so it doesn't require a full graph scan on every ingest. Full scans run on a maintenance schedule.

---

## Category 4 — Production credibility

### Q9. The demo uses mocked Neo4j. Can you run it against a real database?

**What they're testing:** Whether you know your demo's vulnerability and have an answer.

**Model answer:**

Yes — `python scripts/demo_regulatory.py --live` runs the full demo against a real Neo4j instance. It:
1. Verifies the connection
2. Initialises the schema (idempotent)
3. Ingests two real conflicting documents
4. Runs the forward-chaining engine — the `SUPERSEDES [inferred]` edge is a real Neo4j write, not a replay
5. Runs the contradiction detector — the IS_AIRWORTHY / IS_UNAIRWORTHY conflict is found by a real Cypher query

The mock version exists so the demo runs in any environment without setup. The live flag requires Neo4j on `bolt://localhost:7687`. Two ways to get it:
- `docker compose -f compose.dev.yaml up neo4j` — starts just Neo4j (~30s)
- `docker compose -f compose.dev.yaml up` — starts the full stack (API + workers + dashboards)

The mock is not a weakness — it's a test strategy. Every production test suite uses mocks; the question is whether the production path also works. It does. Open Neo4j Browser after running `--live` and query `MATCH (n {tenant:'aerospace'}) RETURN n` to see the persisted graph.

---

### Q10. You have 362 passing tests. What's actually being tested?

**What they're testing:** Whether you know the test coverage or just the number.

**Model answer:**

Unit tests cover the most failure-prone logic in isolation:

- **Confidence clamping** — extractor confidence values outside [0,1] are clamped before Bayesian merge (otherwise `1−(1−c₁)(1−c₂)` overflows)
- **Embedder count mismatch** — if the embedding API returns fewer vectors than chunks, raise ValueError immediately rather than silently shifting embeddings off by one
- **Result store** — Redis SETEX/GET round-trip for cross-process query result sharing
- **SPARQL bridge** — SPARQL SELECT over Turtle export returns correct results
- **Type taxonomy** — LCA, subtype expansion, loading from hierarchy pairs
- **Retry logic** — exponential backoff fires correctly, respects max attempts
- **Ontology registry** — domain/range validation, type correction, migration map

Integration tests require live Docker services (Neo4j, RabbitMQ, Redis) and test the full ingest → query → result round-trip. They auto-skip if Docker is unavailable.

The 362 number matters less than what it covers. The critical paths — confidence arithmetic, embedding alignment, cross-process result sharing, and agent tool safety — all have regression tests from real bugs that were found and fixed.

---

### Q11. How does multi-tenancy work and where could it break?

**What they're testing:** Whether you understand isolation boundaries in a shared data store.

**Model answer:**

Every node and edge in the graph carries a `tenant` property. Every MATCH and MERGE in production Cypher includes `{..., tenant: $tenant}` as a filter. The composite entity key is `(name, type, tenant)` — "Apple" in tenant A and "Apple" in tenant B are different nodes.

Three places isolation could fail:

1. **Missing tenant filter** — a Cypher query that forgets `tenant: $tenant` leaks data across tenants. Prevented by a code review rule in `CONTRIBUTING.md` and the test suite. Every graph query in `neo4j_client.py` is audited for this.

2. **Community detection** — Leiden runs per-tenant (`WHERE e.tenant = $tenant` before building the adjacency matrix). If this filter were missing, communities would be mixed across tenants.

3. **Alias registry** — scoped per tenant. After `load()`, the alias table is pushed to a Redis hash (`graphrag:aliases:{tenant}`, 24h TTL). A second worker starting up calls `load_alias_registry()`, which tries Redis first — if another worker already pushed the table, it skips the full Neo4j MATCH entirely. Workers share deduplication state without race conditions on entity identity.

---

## Category 5 — Scaling and trade-offs

### Q12. What are the scale limits and what breaks first?

**What they're testing:** Engineering honesty and whether you've thought about production.

**Model answer:**

Three clear limits, in order of what breaks first:

1. **Ingestion throughput: ~20 docs/minute on a single worker.** Bottleneck is the Groq API rate limit (1500 requests/day on free tier) and entity resolution embedding comparison. Fix: run N parallel workers (RabbitMQ `prefetch_count=1` already set; `compose.dev.yaml` starts one — add replicas) and a paid Groq tier.

2. **Alias resolution at ~500k entities.** The in-memory alias dictionary fits in RAM up to this point. Beyond this, `load()` takes too long. **Already mitigated:** alias tables are now pushed to Redis after load; subsequent workers warm from Redis without a full Neo4j scan. Hard RAM limit deferred to >500k entities.

3. **Community rebuild at ~100k entities.** Full Leiden over the complete graph is expensive. `IncrementalCommunityDetector` is wired to the API and dashboard — the rebuild button triggers incremental, not full Leiden. Full rebuild is still available for schema migrations.

Beyond these, Neo4j handles tens of millions of nodes/edges without concern assuming appropriate hardware and indexes. The 6 indexes initialised by `init_neo4j.py` cover the hot query paths.

---

### Q13. Why Groq for text generation, DeepSeek as fallback, and OpenAI for embeddings? Why not one provider?

**What they're testing:** Whether the multi-provider architecture was deliberate, and whether it's production-hardened.

**Model answer:**

Three providers, three distinct roles — and the splits were made deliberately:

1. **Embedding: OpenAI `text-embedding-3-large` (3072d).** The Neo4j vector index is created at 3072 dimensions. Switching providers requires re-embedding and recreating the index — expensive once you have real data. OpenAI's 3072d embeddings have strong multilingual and domain-specific performance and are reliably available. Gemini was the original choice but was replaced after Gemini's prepayment credits were exhausted mid-ingestion (blocking all embedding calls). The dimension stayed the same — zero schema migration.

2. **Synthesis: Groq `llama-3.3-70b-versatile` — primary, with DeepSeek-V3 instant fallback.** Groq's free tier runs at ~150 tok/s which is fast enough for interactive queries. The fallback to DeepSeek-V3 (via OpenAI-compatible SDK, fail-fast — no sleep) activates the moment Groq raises a `RateLimitError`. This prevents ingestion from stalling overnight when the 100k token daily quota is exhausted. In production both would be replaced with client-internal models.

3. **Routing: Groq `llama-3.1-8b-instant` — latency optimisation.** The agentic IRCoT reasoning steps (trivial SEARCH/ANSWER decisions) use the 8B model at ~800 tok/s. Each step costs ~0.2s instead of ~1.5s for 70B. This is why agentic p95 is 3.4s not 6.8s — a 50% latency reduction with no quality loss on routing decisions.

The architecture separates concerns cleanly: `get_embedder()`, `get_llm()` (returns `FallbackLLM`), `get_fast_llm()`. Swapping any provider is a one-function change in `llm_client.py`. Provider selection is documented in ADR-0004 and ADR-0006.

---

### Q14. How does the platform handle a situation where the LLM extracts a wrong entity or relation?

**What they're testing:** Whether you've thought about error modes in the extraction pipeline.

**Model answer:**

Four mechanisms, in the order they fire:

1. **Ontology enforcement.** The `OntologyRegistry` validates every extraction output. Unknown entity types are corrected to `CONCEPT`. Relations with invalid domain/range combinations are downgraded (logged as `ontology_registry.drift_detected`). This catches systematic extractor failures — if the LLM starts producing `EXEC` instead of `PERSON`, the migration map corrects it automatically.

2. **Confidence threshold.** Relations extracted with confidence below the configured floor (default 0.5) are not written to the graph. The Bayesian merge means that a single high-confidence source can overcome a low-confidence initial extraction over time.

3. **Contradiction detection.** If a wrong extraction contradicts an existing fact, it surfaces as a Conflict node within the next scan cycle. The entity remains in the graph but the conflict is flagged for review.

4. **Human correction API.** `POST /graph/entities/{id}/correct` allows manual correction of entity type, name, or properties. Corrections feed into the confidence calibration service as negative samples, improving future extraction model calibration.

The platform doesn't silently drop errors — it logs them, flags them, and routes them to reviewable queues. This is the right approach for regulated domains where auditability matters.

---

### Q15 (new). If the agent can execute tools, how do you control it?

**What they're testing:** Whether you've thought about agent safety, not just agent capability.

**Model answer:**

Every tool call passes through a `ToolPolicy` gate before execution. The policy enforces several independent safety properties simultaneously:

**Allowlist.** Only tools in the registered set can be called. An unknown tool name returns `DeniedAction(reason="not_allowed")` — never an exception, always a structured response the caller can reason about.

**Risk levels.** Each tool declares a risk level at registration time:
- `low` — read-only, no side effects (`local_search`, `global_search`, `get_neighbors`)
- `medium` — reads graph internals, explicit scope required (`search_graph`, `get_community`)
- `high` — writes to the graph, needs `write+ingest` or `write+quarantine` (`ingest_document`, `quarantine_entity`)
- `restricted` — irreversible; needs `write+admin+gdpr_officer` (`erase_entity` — GDPR Art.17)

**Scope enforcement.** Each tool declares required scopes. A caller without the full set is denied. The denial carries which scopes were missing. `erase_entity` requires three scopes simultaneously — no single role can accidentally trigger a deletion.

**Argument validation.** Arguments are checked against a schema before execution: type, required fields, enum allowed values, numeric bounds. A malformed argument produces `DeniedAction(reason="invalid_arg")` before the tool ever fires.

**Cross-tenant guard.** If a caller holds `tenant:aerospace` scope and passes `tenant=banking` in an argument, the policy rejects it. Cross-tenant reads are impossible even if the caller holds sufficient scope for the tool itself.

**Dry-run mode.** `ToolPolicy(dry_run=True)` — every tool returns a `DeniedAction` without executing. Useful for untrusted or guest sessions that can preview tool behaviour but not trigger it.

**Timeout.** Every tool call is wrapped in `asyncio.wait_for`. A hung tool produces `DeniedAction(reason="timeout")`, not a hanging coroutine.

**Audit trail.** Every call — allowed, denied, timed out — is appended to an in-memory log with tool name, args, tenant, outcome, reason, and latency. `audit_summary()` gives counts by outcome.

The guardrail tests in `tests/unit/test_tool_safety.py` (49 tests, 8 classes) prove each property works independently and doesn't interfere with the others.

In code: `graphrag/agents/tool_policy.py`.

---

### Q16. You haven't worked commercially with this. Why should I believe you can deliver on a real project?

**What they're testing:** Honesty and self-awareness under pressure.

**Model answer (not a pitch — a conversation):**

That's a fair challenge. Here's what the project proves and what it doesn't.

It proves: I can independently design a non-trivial architecture, make defensible trade-off decisions, implement them correctly (the tests catch real bugs), and document the reasoning. Six ADRs covering the major decisions — the ADRs exist because I genuinely thought through the alternatives, not because they make a good impression.

What it doesn't prove: I've navigated ambiguous requirements from a client, worked in a team where someone else's code broke mine, or operated something under SLA pressure at 2am. Those are real skills and I'm not pretending the project covers them.

The honest argument for hiring is this: the things that are hardest to learn on the job — graph modeling, inference engines, entity resolution, retrieval pipeline design — I've already built and debugged. The things the project doesn't prove — team collaboration, client communication, delivery under constraints — are learnable faster with the technical foundation than without it.

If there's a structured way to demonstrate those skills — a paid short-term project, a trial engagement, a supervised first client — I'd prefer that to claims I can't substantiate yet.

---

## Preparation checklist

Before any meeting, verify you can do all of these **cold, without notes:**

- [ ] Whiteboard the 6 retrieval stages and explain what each one adds
- [ ] Derive `1−(1−c₁)(1−c₂)` from first principles and explain why not average
- [ ] Explain forward vs. backward chaining and give the three reasons for the choice
- [ ] Name the 5 contradiction types and give an example of each
- [ ] Explain the entity resolution 4-stage pipeline and the 0.92 threshold rationale
- [ ] State the three scale limits in order and what breaks first at each
- [ ] Open `docs/adr/0001-property-graph-over-triple-store.md` and talk through it cold
- [ ] Be ready to explain ADR-0004 (Groq vs Gemini), ADR-0005 (Redis result store), ADR-0006 (8B/70B split)
- [ ] Run `make smoke-test` — confirm green before the call
- [ ] Run `python scripts/demo_regulatory.py` and narrate each step without reading the output
- [ ] Run `python scripts/demo_regulatory.py --live` (requires Neo4j) and show the persisted graph
- [ ] State the real RAGAS numbers cold: faithfulness 0.937 (answerable) / 0.842 overall, precision 0.907, recall 0.867
- [ ] State the real latency numbers: hybrid p95 2.2s, agentic p95 3.4s, combined 2.7s
- [ ] Explain why p95 must be reported per mode, not combined
- [ ] Explain the two-model design: 8B for routing (~0.2s/step), 70B for synthesis (~1.5s)
- [ ] Answer Q15 (agent control) from memory: allowlist, risk levels, scopes, cross-tenant, dry-run, timeout, audit
- [ ] Know the test count cold: 362 passing (49 are tool safety guardrail tests)
- [ ] Open `docs/pwc-jd-mapping.md` — know the Gap column entries honestly
- [ ] Answer Q16 conversationally, honestly, without sounding defensive
