# Hiring & Presentation Strategy — GraphRAG Knowledge Platform

> **Purpose:** Everything you need to get the meeting, deliver the 15-minute pitch, survive the technical Q&A, and close. Includes the honest risks, the JD compliance matrix, the learning path, and step-by-step presentation script.

---

## Part 0 — JD Compliance Matrix (verified by running the code)

| JD requirement | Evidence in codebase | Grade |
|---|---|---|
| **Neo4j + Cypher in production** | 39 KG modules, 572-line `neo4j_client.py`; vector ANN, BM25 fulltext, `UNWIND`×22, `EXISTS {}`×12, APOC-with-fallback, bitemporal `as_of` queries | ✅ Strong |
| **Ontology / taxonomy modeling** | Versioned `OntologyRegistry` w/ domain/range rules + migration map; `type_taxonomy.py` (`SUBCLASS_OF`, LCA for merges); config-driven domain overlays (aerospace YAML) | ✅ Strong |
| **Python engineering** | 22,650 LOC, async throughout, 353 passing tests, CI, Docker multi-stage, `make smoke-test`, retry/backoff, structured logging | ✅ Strong |
| **KG × LLM / RAG / vector** | 6-stage pipeline: vector→BM25+RRF→cross-encoder→multi-hop→GAT GNN→LLM; agentic IRCoT fallback (8B routing + 70B synthesis) | ✅ Exceptional |
| **Formal semantics ↔ pragmatic** | OWL-RL reasoner (`owlrl`), SPARQL bridge, RDF/Turtle export — *plus* 6 ADRs documenting every major architectural decision | ✅ Strong |
| **Lead technically while hands-on** | 6 ADRs, 73 documented lessons, phased `todo.md`, `CONTRIBUTING.md`, `runbook.md`, `roadmap.md` | ✅ Good |
| **RDF/OWL** (bonus) | `owl_reasoner.py`, `sparql_bridge.py`, rdflib | ✅ |
| **Inference engines** (bonus) | Datalog forward-chaining: transitivity/symmetry/inverse/composition, fixpoint, confidence decay per hop | ✅ |
| **Entity resolution** (bonus) | 4-stage (exact/normalized/fuzzy/embedding ≥0.92) + splitter + Wikidata linker | ✅ |
| **Legal/regulatory/ESG** (bonus) | Aerospace regulatory ontology + runnable demo, document authority hierarchy, GDPR Article 17, PII guard, contradiction detection | ✅ |

**Bottom line:** every must-have bullet is covered with real, executing code, and 4 of the "highly valuable" bonus areas are present. This is a genuine production KG platform, not conceptual ontology work.

---

## Part 1 — The Three Risks (read this before anything else)

### Risk 1 — "How much of this did AI write?"

In 2026, a CTO seeing 22,650 lines, 39 modules, 6 ADRs, OWL-RL reasoning, and a polished deck — from a candidate whose stated problem is *no commercial experience* — will immediately wonder if it's largely AI-generated. If they conclude it is, the project flips from asset to liability.

**The project only works if you can defend every architectural decision from memory, fluently, under follow-up questioning.**

If you can't explain *why* `1−(1−c₁)(1−c₂)` and not just averaging — out loud, unscripted — the deck actively hurts you. It sets a depth expectation the conversation then fails to meet.

> **Rule:** The real preparation is not rehearsing the script. It is being able to whiteboard the inference engine and argue the trade-offs in any of the 6 ADRs without notes.

---

### Risk 2 — The demo is mocked (and they will probe it)

The demo runs against mocked Neo4j. The "conflict detected" is a scripted replay — legitimate for a zero-services demo, but a Big 4 technical audience will ask: *"Run it against two real documents and show me the contradiction surface."*

**Two fixes:**
- **(a) Volunteer it first:** "This runs against mocked Neo4j so it needs zero setup — I can show it on a live instance as a follow-up." Say this before they ask.
- **(b) Make it real:** Stand up a live Neo4j and ingest two genuinely conflicting documents before the meeting. The demo now uses real Neo4j (run `py -3.11 scripts/demo_regulatory.py --live`). This removes the vulnerability entirely.

---

### Risk 3 — The project closes the technical gap, not the full experience gap

"Experience" is a proxy for: team collaboration, ambiguous requirements, maintaining code others depend on, estimating, stakeholder management, surviving a security review. A solo project demonstrates *technical competence* and almost none of the *collaboration/delivery* bundle.

> **Know this going in.** The inevitable question: *"Tell me about a time you disagreed with a teammate on a design."* Have an answer that doesn't require the project.

---

## Part 2 — Hiring Strategy

### The approach that works

1. **Don't send a CV — show the work.** Link to the GitHub and the live demo output. The application is the artifact.
2. **Target decision-makers, not ATS.** Find the CTO / practice lead on LinkedIn. Ask for 15–20 minutes, not a job.
3. **Don't abandon the referral path.** If you have an internal contact (HR or otherwise), ask for both: a referral *and* the hiring manager's name. A warm flag on your application measurably increases callback rates.
4. **Run parallel targets.** Cold outreach to senior people converts at 5–20% reply. One perfect arrow is not a strategy. Keep 10–15 live targets in flight.
5. **Soften the deck's two or three biggest claims** to match what you can defend. Humbler is more credible to someone who knows the domain.

### What to say in the outreach message

> "I saw PwC has been building GraphRAG capabilities in the region. I built a production-grade GraphRAG + knowledge graph platform from scratch over the past few months — it runs a regulatory compliance demo live, has 353 passing tests, and the codebase is public. I'm not asking for a job offer — just 15 minutes to show you what it can do. Would that be useful?"

### Pre-empting the AI-authorship question

Be ready to narrate:
- The dead-ends and rewrites — the things only the actual builder knows
- Why property graph over triple store (ADR-0001)
- Why Bayesian confidence accumulation over last-write-wins (ADR-0002)
- Why forward-chaining over backward-chaining (ADR-0003)
- Why Groq for generation, Gemini for embeddings (ADR-0004)
- Why Redis as cross-process result store (ADR-0005)
- Why 8B routing + 70B synthesis in the agentic path (ADR-0006)
- What broke first when you ran it against real Neo4j, and how you fixed it

That separates "I built this" from "I generated this."

---

## Part 3 — The 15-Minute Presentation Script

### Before the meeting — prepare these

1. **Demo ready to run:** Terminal open in project folder, `py -3.11 scripts/demo_regulatory.py --live` ready to paste
2. **GitHub open in browser:** `github.com/sergiunicoara/Generative-AI`
3. **API running (optional):** `uvicorn api.main:app --port 8000 --reload`
4. **One sentence about PwC's practice:** "I saw PwC has been building GraphRAG capabilities in the region" — shows you've done homework
5. **Know the real numbers cold:**
   - Faithfulness: 0.840 | Context precision: 0.907 | Recall: 0.867
   - Hybrid p95: 2.2s | Agentic p95: 3.4s | Combined: 2.7s
   - Faithfulness: 0.840 | Precision: 0.907 | Recall: 0.867 (104 real query runs, 23 RAGAS-sampled)
   - Hybrid p95: 2.2s | Agentic p95: 3.4s | Agentic trigger rate: ~10%
   - Seed graph: 20 entities, 12 relations (10-doc aerospace corpus); pipeline targets ~2k entities at scale
   - Calibration pipeline wired: isotonic regression targets Brier < 0.20 on production corpus

---

### Slide 1 — Title *(30 seconds)*

**Say:**
> "Thanks for the time. I'm going to show you something concrete — a production-grade GraphRAG platform I built from scratch. Not a tutorial project, not a PoC — a platform with 39 knowledge graph modules, 353 passing tests, and a regulatory compliance demo that runs live against real Neo4j. I'll show you the code and run it. Let me start with why this matters."

**Don't linger.** Move immediately to slide 2.

---

### Slide 2 — The Problem *(1 minute)*

**Say:**
> "Standard RAG — just throwing documents into a vector index and asking an LLM — breaks in three ways for enterprise clients.
>
> First: hallucination. When the knowledge is complex and spread across hundreds of documents, the LLM starts filling gaps with invented facts. You can't audit that.
>
> Second: no reasoning chain. If a contract references a regulation which constrains a company, a vector search finds documents that mention those words — it can't trace the logic across the connections.
>
> Third: no audit trail. Your banking and regulatory clients can't just accept an answer. They need to say: 'This answer came from document X, section Y, authored by Z, which supersedes document W from 2022.' A black box doesn't give them that.
>
> That's the problem this platform solves."

**Anticipate:** *"Why not just fine-tune an LLM?"*
> "Fine-tuning bakes knowledge in statically. It can't be updated when regulations change. And it still can't trace where the answer came from."

---

### Slide 3 — Architecture *(1.5 minutes)*

**Say:**
> "The platform has three layers.
>
> The first is a knowledge graph in Neo4j — not a document store. A formally modeled knowledge base with entities, typed relations, an ontology, and contradiction detection. When two documents disagree about a fact, the system flags it automatically.
>
> The second layer is the retrieval pipeline — six stages. Vector search fused with BM25 keyword search via RRF, cross-encoder reranking, multi-hop graph traversal that follows entity connections across documents, then a graph neural network that re-scores results based on graph structure, not just text similarity.
>
> The third layer is the agentic fallback — IRCoT iterative retrieval. When the hybrid answer has low confidence and zero citations, it fires. It uses a fast 8B model for routing decisions — SEARCH or ANSWER, 0.2 seconds each — and the full 70B model only for final synthesis. That's why the agentic p95 is 3.4 seconds, not 7."

**Anticipate:** *"What's the difference between this and LangChain or LlamaIndex?"*
> "Those are orchestration frameworks. This is a domain-specific platform with a real Neo4j schema, ontology enforcement, forward-chaining inference, contradiction detection, and a 6-stage retrieval pipeline. LangChain gives you building blocks. This is built."

---

### Slide 4 — Capabilities / JD mapping *(1 minute — scan fast)*

**Say:**
> "I won't go through every row — you can read it. The point is: every requirement in the job description maps directly to a specific module in the codebase. This isn't conceptual alignment — I can open any of these files right now and show you the code.
>
> The one I'll call out: 'Design, implement, and validate MVP solutions' — 353 passing tests, GitHub Actions CI, Docker multi-stage build, a production runbook. That's the operational proof."

---

### Slide 5 — LIVE DEMO *(5 minutes — most important)*

**Say before switching:**
> "Let me show you the regulatory intelligence scenario — the most directly relevant to what PwC does."

**Run:**
```bash
py -3.11 scripts/demo_regulatory.py --live
```

**Narrate each step as it prints:**

**Step 1 — Ontology loads:**
> "It's reading the aerospace regulatory ontology from a YAML file in `config/ontologies/`. No code change required to switch domains. `AIRWORTHINESS_DIRECTIVE` is defined as a subtype of `REGULATION` which is a subtype of `CONCEPT`. This hierarchy drives query expansion and merge decisions automatically. To switch to banking compliance, you load a different YAML file."

**Step 2 — Forward-chaining fires:**
> "This is the inference engine. AD-2024 supersedes AD-2022, AD-2022 supersedes AD-2020. The system applies the transitivity rule and derives that AD-2024 transitively supersedes AD-2020 — even though no document explicitly states that. The derived edge is tagged `source_type=inferred` with a confidence score of 0.95², because there are two hops. This is Datalog-style reasoning — no LLM guessing, pure logic."

```
FAA-AD-2024-01-02
  └─ SUPERSEDES → FAA-AD-2022-03-07  (confidence: 0.95, asserted)
      └─ SUPERSEDES → FAA-AD-2020-05-11  (confidence: 0.95, asserted)
┌─ SUPERSEDES [inferred] ──────────────────────────┐
│  2024 → 2020  (confidence: 0.857)                │
└──────────────────────────────────────────────────┘
```

**Step 3 — Contradiction detection (the winning moment):**
> "Same aircraft, two independent documents — one says IS_AIRWORTHY, the other says IS_UNAIRWORTHY. The system detects this as a `positive_negative_pair` conflict and surfaces it for resolution. In a banking context: same company, one document says 'in compliance', another says 'under investigation'. You want to catch that before it reaches a client report. Most RAG systems would return one answer or the other randomly. This platform surfaces the conflict automatically."

**Step 4 — Authority chain:**
> "It follows the SUPERSEDES chain and tells you which document is the current governing authority. In a regulatory practice, this is exactly what an associate spends hours doing manually."

**After demo — switch to Neo4j Browser tab:**
> "And that just persisted to real Neo4j — here are the entities and edges from what we just ran."
Run: `MATCH (n {tenant:'aerospace'}) RETURN n`

---

### Slide 6 — Metrics / Observability *(1.5 minutes)*

**Say:**
> "If you can't measure it, you can't run it in production. These numbers come from 104 real query runs against the pipeline.
>
> Faithfulness is 0.840 — 84% of answers fully grounded in retrieved context. Context precision is 0.907 — almost everything we retrieve is relevant. Hybrid p95 latency is 2.2 seconds. The agentic path — which fires on about 10% of queries for hard multi-hop questions — runs at 3.4 seconds by design.
>
> The knowledge graph is seeded from 10 aerospace regulatory documents — FAA/EASA airworthiness directives and manufacturer records. The pipeline is fully wired: entity extraction, contradiction detection, calibration, community detection. Scale the corpus and all the health metrics scale with it."

**Important framing:** *"We alert on agentic rate, not agentic latency. If more than 20% of queries trigger the fallback, the threshold is too loose — not the corpus is hard."*

---

### Slide 7 — Technical Foundation *(1 minute)*

**Say:**
> "22,650 lines of production Python. 353 tests. 39 knowledge graph modules. Six architecture decision records — every major choice documented: property graph vs triple store, forward-chaining vs backward-chaining, Bayesian confidence, Groq vs Gemini, Redis result store, two-model agentic design.
>
> The two-model agentic design — 8B for routing, 70B for synthesis — cut agentic p95 from 6.8 seconds to 3.4 seconds. That's an engineering decision, not a configuration option."

---

### Slide 8 — Client Scenarios *(1.5 minutes)*

**Say:**
> "Three scenarios where I'd expect to apply this at PwC.
>
> Regulatory intelligence for banking and insurance — ingest the full regulatory corpus, detect supersessions automatically, surface contradictions before audit. That's what we just saw.
>
> Audit knowledge bases — auditors review hundreds of evidence documents. Multi-hop traversal finds connections between entities across documents that human reviewers miss. The agentic retriever runs iterative sub-searches until it grounds the answer in actual source material.
>
> Compliance monitoring — temporal knowledge modeling tracks when facts changed. GDPR erasure is built in — when a client is decommissioned, the graph wipes all their entities and generates an audit log."

---

### Slide 9 — Close *(1 minute + conversation)*

**Say:**
> "Three things I want to leave you with.
>
> Day-one delivery — I'm not proposing to learn this on your projects. The platform is built, it runs against real data, and the numbers are real.
>
> The domain expertise transfers. Aerospace is the demo domain because I know the ontology. Banking, audit, insurance — the architecture is identical. The YAML ontology file is what changes.
>
> Everything is verifiable. There's no claim here that can't be checked in 30 seconds. The GitHub link is at the bottom.
>
> What's the next step from your side?"

**Then stop talking.** Let them respond.

---

## Part 4 — Q&A Defence

**"Have you done this in a commercial context?"**
> "I haven't had the opportunity — which is why I built this instead of waiting. Every company I've approached has pointed at experience requirements. So I built the production-grade platform, documented the architecture decisions, and wrote the tests. The work is real; the only thing missing is an invoice number."

**"How does this scale to hundreds of thousands of documents?"**
> "Three paths: parallel ingestion workers — `compose.dev.yaml` starts one instance, add replicas freely since `prefetch_count=1` is already set. The alias registry is now Redis-backed so workers share entity deduplication state without each doing a full Neo4j scan on startup. And the incremental community detector is wired to the API dashboard — only communities containing changed entities get rebuilt, not the full Leiden run."

**"Why Neo4j over Pinecone or Weaviate?"**
> "Vector search answers 'which chunks are similar to my query?' It can't answer 'which entities are connected to the entity in my query, and what do documents about those entities say?' Multi-hop reasoning, forward-chaining inference, contradiction detection — none of those exist in a flat vector index. ADR-0001 in the repo explains the full tradeoff."

**"If the agent can execute tools, how do you control it?"**
> "Every tool call passes through a `ToolPolicy` gate before execution. The policy enforces an allowlist — only registered tools can be called. Each tool declares required scopes, so `erase_entity` requires `write + admin + gdpr_officer` and is denied for any other caller. Arguments are validated against a schema: type, required fields, enum values, min/max. There's a cross-tenant guard — a caller scoped to `aerospace` can't reference `banking` in an argument. Dry-run mode lets untrusted sessions preview what would happen without executing anything. Every call — allowed, denied, or timed out — is written to an audit log. `docs/pwc-jd-mapping.md` shows exactly where each of these is in the code."

**"Can this use PwC's internal LLM?"**
> "Yes — the LLM is routed through a single module, `llm_client.py`. Three functions: `get_llm()` for synthesis, `get_fast_llm()` for routing, `get_embedder()` for embeddings. Swapping from Groq to any OpenAI-compatible API is a single-file change."

**"Why is the demo using mocked Neo4j?" (if they find it)**
> "The mock version runs with zero services for a clean demo environment. The `--live` flag runs against real Neo4j — I showed that version just now. The entities and edges you saw in Neo4j Browser came from this demo script in real-time."

**"What's your availability?"**
> "Immediate. Looking to contribute to PwC's Graph RAG practice — either as a full-time engineer or on a project basis to start."

---

## Part 5 — Learning Path & Reading List

### What you must be able to whiteboard cold (highest priority)

These are the topics a CTO will probe. Know them without notes.

**1. Bayesian confidence accumulation**
- Why `1−(1−c₁)(1−c₂)` and not averaging
- Answer: averaging treats confidences as independent and additive. The Bayesian formula treats them as independent evidence — if one source says 0.9 and another says 0.9, the combined confidence is 0.99 (both would have to be wrong simultaneously). Averaging gives 0.9.
- In code: `graphrag/graph/confidence_model.py`

**2. Forward-chaining inference vs backward-chaining**
- Forward: start from known facts, derive everything reachable (materialise derived edges into the graph)
- Backward: start from a query goal, work backwards to find supporting facts (not materialised — computed at query time)
- Why forward for regulatory supersession: you want transitive supersession chains pre-computed so queries are fast; the chain is static between updates
- In code: `graphrag/inference/forward_chaining.py`

**3. Property graph vs triple store (ADR-0001)**
- Triple store: every fact is (subject, predicate, object) — pure RDF; great for federated queries and OWL reasoning
- Property graph: nodes and edges have arbitrary properties; great for traversal, real-world entity modelling, and operational queries
- Why we chose property graph: regulatory/enterprise data is inherently property-rich (confidence, source, timestamp, authority level); traversal performance matters more than SPARQL federation; OWL-RL bridge added later via rdflib export
- In code: `docs/adr/0001-property-graph-over-triple-store.md`

**4. Entity resolution — why the 4-stage pipeline**
- Stage 1: exact match (normalised string) — fast, zero false positives
- Stage 2: fuzzy (Levenshtein ≥ 85) — catches typos and abbreviations
- Stage 3: embedding cosine (≥ 0.92 threshold) — catches semantic variants ("SpaceX" vs "Space Exploration Technologies")
- Stage 4: new entity — nothing matched; create a new node
- Why threshold 0.92 for embedding stage: lower → over-merge (Apple the company vs Apple the fruit); higher → under-merge (fragmented graph). 0.92 is calibrated for curated domain data; lower for noisy corpora.
- In code: `graphrag/graph/entity_resolver.py`

**5. The 6-stage retrieval pipeline — why each stage**
1. Vector ANN — semantic similarity
2. BM25 + RRF — exact term matching (regulation codes, entity names that get diluted in 3072d embeddings)
3. Cross-encoder reranking — deep pairwise scoring of top-N candidates
4. Multi-hop graph traversal — find chunks mentioning related entities, not just the query entity
5. GNN scoring — re-score by graph-structural proximity, not just textual
6. LLM synthesis — grounded, cited answer from the final context

**6. Contradiction detection — 5 conflict types**
- `multi_source`: same claim, different confidence levels across sources
- `directional_reversal`: A→B and B→A on the same relation
- `exclusive_state`: entity in two mutually exclusive states (airworthy AND unairworthy)
- `functional_violation`: entity has two values for a single-valued property
- `positive_negative_pair`: RELATES_TO and NEGATIVE_RELATES_TO coexist
- In code: `graphrag/graph/contradiction_detector.py`

**7. Two-model agentic design (ADR-0006)**
- Why: intermediate SEARCH/ANSWER routing decisions are ~15 tokens; using 70B for this is wasteful
- 8B (llama-3.1-8b-instant): ~800 tok/s on Groq → 0.2s per step
- 70B (llama-3.3-70b-versatile): ~150 tok/s → 1.5s per step
- With max_steps=2: 2 × 0.2s routing + 1.5s synthesis + retrieval = ~3.4s vs 6.8s
- In code: `graphrag/retrieval/agentic_retriever.py` — `_reason()` and `_synthesize()`

**8. Provider split: Groq for generation, Gemini for embeddings (ADR-0004)**
- Embeddings: Gemini `gemini-embedding-001`, 3072d — hard constraint (changing requires full re-embed)
- Generation: Groq `llama-3.3-70b` — quota (1,500 RPD free), speed (~150 tok/s)
- Routing: Groq `llama-3.1-8b-instant` — ~800 tok/s, trivial structured output
- Client swap: change `get_llm()` / `get_fast_llm()` / `get_embedder()` in `llm_client.py` — nothing else

**9. Redis as cross-process result store (ADR-0005)**
- Problem: API process and query worker run in separate containers; in-process dict silently broke all multi-container deploys
- Why Redis over PostgreSQL: query results are ephemeral (1h TTL); latency matters for the polling path (~0.5ms vs ~5ms)
- Why Redis over RabbitMQ reply-to: polling model is simpler; push model requires long-lived consumer with correlation IDs
- In code: `graphrag/retrieval/result_store.py` (also used by query cache)

**10. Agent tool safety — how do you control the agent?**
- Single gate: `ToolPolicy.call(tool_name, args, tenant)` — every invocation passes through
- Four risk levels: low (read-only) / medium (graph internals) / high (write, e.g. `ingest_document`) / restricted (irreversible, e.g. `erase_entity` — needs `write+admin+gdpr_officer`)
- Dry-run mode: `ToolPolicy(dry_run=True)` — every call returns `DeniedAction` without executing; useful for untrusted sessions
- Cross-tenant guard: if caller holds `tenant:aerospace` scope and passes `tenant=banking`, the policy rejects before the tool fires
- Audit trail: every call logged with outcome, reason, tenant, latency
- In code: `graphrag/agents/tool_policy.py` — `ToolSpec`, `DeniedAction`, `AuditEntry`, `ToolPolicy.from_defaults()`
- Tests: 49 guardrail tests across 8 classes — `tests/unit/test_tool_safety.py`

---

### Papers and references worth reading

**Knowledge Graphs & Ontologies**
- Ehrlinger & Wöß (2016): "Towards a Definition of Knowledge Graphs" — the foundational framing
- Hogan et al. (2021): "Knowledge Graphs" (ACM Computing Surveys) — the comprehensive survey; read sections 1, 2, and 6
- OWL 2 Web Ontology Language Primer — W3C spec; read sections 1–4

**Retrieval-Augmented Generation**
- Lewis et al. (2020): "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — the original RAG paper
- Edge et al. (2024): "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft GraphRAG) — the closest to what you built
- Trivedi et al. (2022): "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions" — the IRCoT paper; the agentic retriever implements this

**Confidence Calibration**
- Guo et al. (2017): "On Calibration of Modern Neural Networks" — calibration curves and temperature scaling
- Platt (1999): "Probabilistic Outputs for Support Vector Machines" — isotonic regression calibration background

**Entity Resolution**
- Papadakis et al. (2021): "Blocking and Filtering Techniques for Entity Resolution: A Survey" — survey of the problem space
- In practice: read `graphrag/graph/entity_resolver.py` directly — the 4-stage pipeline is documented inline

**Knowledge Graph Embeddings**
- Bordes et al. (2013): "Translating Embeddings for Modeling Multi-Relational Data" — the TransE paper (link prediction module uses this)
- In code: `graphrag/ml/link_predictor.py`

**Community Detection**
- Traag et al. (2019): "From Louvain to Leiden: guaranteeing well-connected communities" — the Leiden algorithm used for community detection
- In code: `graphrag/graph/community_detector.py`

---

### Concepts to be comfortable explaining in conversation

| Concept | One-line answer |
|---|---|
| RRF (Reciprocal Rank Fusion) | Merges two ranked lists by `1/(k + rank)` — avoids score normalization across different retrieval systems |
| Bitemporal modeling | Two time axes: valid time (when the fact was true in the world) and transaction time (when we recorded it) |
| Leiden vs Louvain | Leiden guarantees well-connected communities; Louvain can produce internally disconnected ones |
| Brier score | Mean squared error between predicted probability and binary outcome; 0=perfect, 0.25=random |
| Isotonic regression calibration | Non-parametric monotone transformation that maps raw model confidence to empirical frequency |
| Cross-encoder vs bi-encoder | Bi-encoder: embed query and doc separately (fast, approximate). Cross-encoder: score (query, doc) jointly (slow, accurate). Use bi-encoder for ANN, cross-encoder for reranking top-N. |
| GNN message passing | Each node aggregates features from its neighbours; graph structure encodes entity relationships |
| Datalog transitivity rule | If A→B and B→C then A→C; applied to fixpoint (until no new edges can be derived) |

---

## Part 6 — The Demo Visual Guide

### What to show during Slide 5

**Show terminal in real-time.** Highlight these 4 sections as they scroll:

**1. Ontology hierarchy (Step 1)**
- Point at type pairs: `AIRWORTHINESS_DIRECTIVE ⊂ REGULATION ⊂ CONCEPT`
- Say: "No code changes needed — the YAML file defines what types and relations are valid. `config/ontologies/aerospace_regulatory.yml` is what drives this."

**2. Forward-chaining authority chain (Step 4)**
- Point at the ASCII tree showing the transitive SUPERSEDES derivation
- Say: "The system sees two explicit links and derives the transitive link automatically. Datalog-style reasoning — no LLM guessing, pure logic. Confidence decays 0.95 per hop."

**3. Contradiction detection (Step 5 — the winning moment)**
- Point at: `G-ABCD | IS_AIRWORTHY vs IS_UNAIRWORTHY | status: open`
- Say: "Two independent documents disagree on the same aircraft. The system flags this automatically so an auditor catches it before it goes into a client report."

**4. Confidence propagation (summary)**
- Point at: `Inferred edge confidence: 0.95² = 0.857 for 2-hop path`
- Say: "Confidence decays predictably per hop. You can trace exactly how confident we are in any derived fact."

### What NOT to show

- Don't dwell on mock vs live — run `--live` and it's real
- Don't read line-by-line; let the visual structure speak
- Don't explain `LCA()` or `expand_type()` unless they ask — it's proof of depth, not the story

### The YAML side-by-side (optional strong move)

Open `config/ontologies/aerospace_regulatory.yml` in VS Code alongside the terminal.

> "See how the type pairs in the YAML file match the output? This YAML file, loaded at startup, defines the entire type system and relation constraints for the domain. 28 type pairs, 12 relation rules. To scale to banking or audit domains, we just swap the YAML file."

---

## Part 7 — Code Reference (for pointing at files)

| What they ask about | File to open | Lines |
|---|---|---|
| Type hierarchy definition | `config/ontologies/aerospace_regulatory.yml` | 18–45 |
| Relation rules | `config/ontologies/aerospace_regulatory.yml` | 61–80 |
| Demo step 1 (ontology) | `scripts/demo_regulatory.py` | 100–132 |
| Demo step 4 (inference) | `scripts/demo_regulatory.py` | 228–281 |
| Demo step 5 (contradiction) | `scripts/demo_regulatory.py` | 284–325 |
| Forward-chaining engine | `graphrag/inference/forward_chaining.py` | — |
| Neo4j client | `graphrag/graph/neo4j_client.py` | — |
| 6-stage retrieval | `graphrag/retrieval/hybrid_retriever.py` | — |
| Agentic IRCoT (two-model) | `graphrag/retrieval/agentic_retriever.py` | `_reason()` / `_synthesize()` |
| Confidence accumulation | `graphrag/graph/confidence_model.py` | — |
| Entity resolver | `graphrag/graph/entity_resolver.py` | — |
| Contradiction detector | `graphrag/graph/contradiction_detector.py` | — |
| LLM router (swap here) | `graphrag/core/llm_client.py` | `get_llm()` / `get_fast_llm()` |
| Architecture decisions (6 ADRs) | `docs/adr/0001-*.md` → `0006-*.md` | — |
| Worker health server | `graphrag/workers/health_server.py` | `GET /ready`, `GET /live` |
| Full dev stack | `compose.dev.yaml` | `docker compose -f compose.dev.yaml up` |
| Seed demo data | `scripts/seed_demo_data.py` | `--commit --tenant aerospace` |
| Alias registry (Redis-backed) | `graphrag/graph/alias_registry.py` | `load_alias_registry()` |
| Wikidata linking | `graphrag/graph/entity_linker.py` | `WIKIDATA_LINKING=1` |
| Agent tool safety layer | `graphrag/agents/tool_policy.py` | `ToolPolicy.from_defaults()` |
| Tool guardrail tests (49) | `tests/unit/test_tool_safety.py` | `pytest tests/unit/test_tool_safety.py` |
| Golden eval set (40 Qs) | `evals/golden_set.json` | — |
| Golden eval runner | `scripts/run_golden_eval.py` | `py -3.11 scripts/run_golden_eval.py` |
| JD mapping (with Gap column) | `docs/pwc-jd-mapping.md` | — |

---

## Part 8 — Preparation Checklist (before the meeting)

### Technical fluency (the real bottleneck)
- [ ] Whiteboard the Bayesian confidence formula from memory — derive it, explain why not averaging
- [ ] Explain the 4-stage entity resolution pipeline: what fails at each stage, what the thresholds mean
- [ ] Run `make smoke-test` — confirm it exits 0 before the meeting (353 tests)
- [ ] Know `docs/pwc-jd-mapping.md` — open it when any JD question comes up
- [ ] Know the answer to "how do you control the agent?" cold (ToolPolicy, 4 risk levels, audit log)
- [ ] Explain ADR-0001 (property graph vs triple store) conversationally — what you considered and why you decided
- [ ] Explain ADR-0004 (Groq vs Gemini) — quota, speed, two-model design, client swap path
- [ ] Explain ADR-0005 (Redis result store) — why not PostgreSQL, why not RabbitMQ reply-to
- [ ] Explain ADR-0006 (8B routing + 70B synthesis) — the latency table, why not 8B for synthesis
- [ ] Explain the 6 retrieval stages in order, what failure mode each one fixes
- [ ] Explain the two-model agentic design: which model does what, why, latency numbers
- [ ] Explain contradiction detection: name all 5 types, explain `positive_negative_pair` with an example
- [ ] State the real RAGAS numbers cold: faithfulness 0.840, precision 0.907, recall 0.867
- [ ] State the real latency numbers: hybrid p95 2.2s, agentic p95 3.4s, why reported per mode

### Demo preparation
- [ ] Run `docker compose -f compose.dev.yaml up neo4j` (or full stack) — confirm healthy
- [ ] Run `py -3.11 scripts/seed_demo_data.py --commit` — populates graph with 20 entities + conflicts
- [ ] Run `py -3.11 scripts/demo_regulatory.py --live` — confirm all 6 steps complete
- [ ] Have Neo4j Browser open, confirm `MATCH (n {tenant:'aerospace'}) RETURN n` shows data
- [ ] Practise narrating steps 1, 4, 5 out loud without reading the output
- [ ] Know the one-liner for the full stack: `docker compose -f compose.dev.yaml up`

### Materials
- [ ] Open `C:\Users\Sergiu\Desktop\GraphRAG_PwC_Pitch.pptx` and click through all slides
- [ ] Have `config/ontologies/aerospace_regulatory.yml` open in VS Code (ready to show)
- [ ] GitHub repo open in browser: `github.com/sergiunicoara/Generative-AI`

### Soft preparation
- [ ] Prepare one honest answer for "no commercial experience" — rehearse it until it sounds natural, not defensive
- [ ] Prepare the "AI-authorship" pre-empt: one or two things only the builder would know (a dead-end you hit, a decision you reversed)
- [ ] Prepare for "Tell me about a time you disagreed with a teammate" — something that doesn't require the project
