# Hiring & Presentation Strategy — GraphRAG Knowledge Platform

> **Purpose:** Everything you need to get the meeting, deliver the 15-minute pitch, survive the technical Q&A, and close. Includes the honest risks, the JD compliance matrix, the learning path, and step-by-step presentation script.

---

## Part 0 — JD Compliance Matrix (verified by running the code)

| JD requirement | Evidence in codebase | Grade |
|---|---|---|
| **Neo4j + Cypher in production** | 38 KG modules, 572-line `neo4j_client.py`; vector ANN, BM25 fulltext, `UNWIND`×22, `EXISTS {}`×12, APOC-with-fallback, bitemporal `as_of` queries | ✅ Strong |
| **Ontology / taxonomy modeling** | Versioned `OntologyRegistry` w/ domain/range rules + migration map; `type_taxonomy.py` (`SUBCLASS_OF`, LCA for merges); config-driven domain overlays (aerospace YAML) | ✅ Strong |
| **Python engineering** | 26,600 LOC, async throughout, 362 passing tests, CI, Docker multi-stage, `make smoke-test`, retry/backoff, structured logging | ✅ Strong |
| **KG × LLM / RAG / vector** | 6-stage pipeline: vector→BM25+RRF→cross-encoder→multi-hop→GAT GNN→LLM; agentic IRCoT fallback (8B routing + 70B synthesis) | ✅ Exceptional |
| **Formal semantics ↔ pragmatic** | OWL-RL reasoner (`owlrl`), SPARQL bridge, RDF/Turtle export — *plus* 6 ADRs documenting every major architectural decision | ✅ Strong |
| **Lead technically while hands-on** | 6 ADRs, 90 documented lessons, phased `todo.md`, `CONTRIBUTING.md`, `runbook.md`, `roadmap.md` | ✅ Good |
| **RDF/OWL** (bonus) | `owl_reasoner.py`, `sparql_bridge.py`, rdflib | ✅ |
| **Inference engines** (bonus) | Datalog forward-chaining: transitivity/symmetry/inverse/composition, fixpoint, confidence decay per hop | ✅ |
| **Entity resolution** (bonus) | 4-stage (exact/normalized/fuzzy/embedding ≥0.92) + splitter + Wikidata linker | ✅ |
| **Legal/regulatory/ESG** (bonus) | Aerospace regulatory ontology + runnable demo, document authority hierarchy, GDPR Article 17, PII guard, contradiction detection | ✅ |

**Bottom line:** every must-have bullet is covered with real, executing code, and 4 of the "highly valuable" bonus areas are present. This is a genuine production KG platform, not conceptual ontology work.

---

## Part 1 — The Three Risks (read this before anything else)

### Risk 1 — "How much of this did AI write?"

In 2026, a CTO seeing 26,600 lines, 38 modules, 6 ADRs, OWL-RL reasoning, and a polished deck — from a candidate whose stated problem is *no commercial experience* — will immediately wonder if it's largely AI-generated. If they conclude it is, the project flips from asset to liability.

**The project only works if you can defend every architectural decision from memory, fluently, under follow-up questioning.**

If you can't explain *why* `1−(1−c₁)(1−c₂)` and not just averaging — out loud, unscripted — the deck actively hurts you. It sets a depth expectation the conversation then fails to meet.

> **Rule:** The real preparation is not rehearsing the script. It is being able to whiteboard the inference engine and argue the trade-offs in any of the 6 ADRs without notes.

---

### Risk 2 — The demo has a mocked mode — always use `--live`

`demo_regulatory.py` without flags uses `unittest.mock.AsyncMock` in place of Neo4j. Every output is a hardcoded scripted return — nothing touches a real database. A technical interviewer who asks *"show me the Neo4j Browser with those entities"* will immediately know the difference.

**This is resolved — but only if you use the flag:**

```bash
python scripts/demo_regulatory.py --live   # real Neo4j, no mocks
```

`--live` runs the full `run_live_demo()` path: real ingestion, real graph queries, real contradiction detection. Whatever entity/edge/conflict counts are in Neo4j right now (⚠ run `MATCH (e:Entity {tenant:'aerospace'}) RETURN count(e)` etc. to get the current figures — they drift on every fresh ingestion, see A96/A98) came from this path. After the demo you can open Neo4j Browser and run `MATCH (n {tenant:'aerospace'}) RETURN n` — the data is there.

**The only remaining rule:** always confirm Neo4j is running and seeded before the meeting. If you start the demo and Neo4j is down, you fall back to a visible error — not a clean mocked output. Confirm with `python scripts/demo_regulatory.py --live` the night before.

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

> "I saw PwC has been building GraphRAG capabilities in the region. I built a production-grade GraphRAG + knowledge graph platform from scratch over the past few months — it runs a regulatory compliance demo live, has 362 passing tests, and the codebase is public. I'm not asking for a job offer — just 15 minutes to show you what it can do. Would that be useful?"

### Pre-empting the AI-authorship question

Be ready to narrate:
- The dead-ends and rewrites — the things only the actual builder knows
- Why property graph over triple store (ADR-0001)
- Why forward-chaining over backward-chaining (ADR-0002)
- Why Bayesian confidence accumulation over last-write-wins (ADR-0003)
- Why Groq primary + DeepSeek fallback for generation, OpenAI `text-embedding-3-large` for embeddings (ADR-0004)
- Why Redis as cross-process result store (ADR-0005)
- Why 8B routing + 70B synthesis in the agentic path (ADR-0006)
- What broke first when you ran it against real Neo4j, and how you fixed it

That separates "I built this" from "I generated this."

---

## Part 3 — The 15-Minute Presentation Script

### Before the meeting — prepare these

1. **Docker Desktop running:** open Docker Desktop and wait for the whale icon to stop animating before proceeding
2. **Neo4j running:** `docker compose -f compose.dev.yaml up -d neo4j` — wait ~15s — script will tell you immediately if unreachable
2. **Demo ready to run:** Terminal open in project folder, `python scripts/demo_regulatory.py --live` ready to paste
3. **GitHub open in browser:** `https://github.com/sergiunicoara/Generative-AI/tree/main/AI%20knowledge%20graph%20platform`
3. **API running (optional):** `uvicorn api.main:app --port 8000 --reload`
4. **One sentence about PwC's practice:** "I saw PwC has been building GraphRAG capabilities in the region" — shows you've done homework
5. **Know the real numbers cold:**
   - Faithfulness (full 39-question golden set, measured 2026-06-10): **0.785** answerable (25/39 scored, 14 refusals) | baseline 0.840 | delta -0.055. ⚠ **Do not cite 0.937** — that figure was measured on a 10-question subset and is not comparable to the full set (see `tasks/lessons.md` A99). 0.785 is the honest, full-set, current number.
   - Two golden categories (`architecture`, `domain` — "what retrieval stages does the platform use", "is the ontology hard-coded") score 0.0/refused by design: the corpus contains only aerospace regulatory documents, so the system correctly refuses to answer questions about its own architecture rather than hallucinating. If asked about the 0.785 figure, lead with this — it shows the refusal behavior is working, which is part of the pitch.
   - Latency by mode: hybrid p95 2.2s | agentic p95 3.4s | combined p95 2.7s — ⚠ measured before the 2026-06-10 supersession fix and re-ingest; re-verify if quoting precisely.
   - Agentic trigger rate: ~9-10% current; alert if >20%
   - Why trigger rate matters: it is a direct read on retrieval health. If it climbs, the agent is covering for weak retrieval even if combined p95 still looks acceptable.
   - ⚠⚠ **STOP — these graph-health numbers are NOT stable. Re-run the live queries
     in Part 6 immediately before you present, every single time.** Proof: this doc
     was audited and "verified live" at 364 entities/380 edges/11 conflicts/92.12%
     coherence on the morning of 2026-06-07 — by that same afternoon, after one more
     `--wipe --commit` of the *identical* 12-doc corpus, the live graph showed
     368 entities/422 edges/7 conflicts/90.27% coherence. After a third
     `--wipe --commit` on 2026-06-10 (with a supersession-chain fix — see A99), the
     live graph showed **368 entities, 422 edges, 4 open conflicts, 9.48/1k
     contradiction rate, 53 communities**. Same corpus, different LLM extraction
     each time (non-deterministic at temperature=0 — see A96/A98). The numbers
     below **will be wrong again by the time you read this. Run the Part 6 queries
     live, in front of the room.**
   - Real corpus (verified live 2026-06-10, after supersession fix — re-verify before use): 368 entities, 422 edges (412 document + 10 inferred, 12-doc aerospace corpus, LLM-extracted); 4 open conflicts; 9.48/1k contradiction rate; 53 Leiden communities (re-verify coherence % live)
   - Pipeline: raw extraction → alias-deduplicated entity count varies run-to-run — don't quote a specific raw count or dedup % from memory; query `MATCH (a:Alias {tenant:'aerospace'}) RETURN count(a)` and the entity count live, then describe the *mechanism* ("4-stage alias resolution collapses duplicate mentions into canonical entities") rather than a number that will be stale within hours
   - SUPERSEDES chain (fixed 2026-06-10): `FAA-AD-2024-01-02 → FAA-AD-2022-03-07 → FAA-AD-2020-05-11`, with `superseded_by` set on the older docs — verify with `MATCH (a:Document)-[:SUPERSEDES]->(b:Document) RETURN a.filename, b.filename`
   - Calibration: the pipeline (`confidence_calibration.py`, Brier score + isotonic regression) now has **real live data**: 25 samples written during the 2026-06-10 faithfulness eval, **Brier score 0.700, verdict "under-confident"** (snapshot `961f2a4b`). This is an honest first measurement, not a good one — 0.700 is poor calibration (closer to 0 is better). If asked: "The calibration pipeline is wired end-to-end and has its first live sample — 25 points, Brier 0.70, currently under-confident. That's expected for a first batch; the next step is accumulating more samples and checking whether isotonic regression tightens it." Do not say "Brier score 0.052" — that number was never real.

---

### Slide 1 — Title *(30 seconds)*

**Say:**
> "Thanks for the time. I'm going to show you something concrete — a production-grade GraphRAG platform I built from scratch. Not a tutorial project, not a PoC — a platform with 38 knowledge graph modules, 362 passing tests, and a regulatory compliance demo that runs live against real Neo4j. I'll show you the code and run it. Let me start with why this matters."

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
> The third layer is the agentic fallback — IRCoT iterative retrieval. It only fires when the hybrid answer both hedges and has zero citations. That AND condition matters: the first version used OR logic and fired far too often on sparse corpora. The current trigger rate is around 9-10%, which is the metric I watch hardest because it tells me whether retrieval is healthy or whether the agent is compensating for it.
>
> When the fallback does fire, it uses a fast 8B model for routing decisions — SEARCH or ANSWER, 0.2 seconds each — and the full 70B model only for final synthesis. That's why the agentic p95 is 3.4 seconds, not 7."

**Anticipate:** *"What's the difference between this and LangChain or LlamaIndex?"*
> "Those are orchestration frameworks. This is a domain-specific platform with a real Neo4j schema, ontology enforcement, forward-chaining inference, contradiction detection, and a 6-stage retrieval pipeline. LangChain gives you building blocks. This is built."

---

### Slide 4 — Capabilities / JD mapping *(1 minute — scan fast)*

**Say:**
> "I won't go through every row — you can read it. The point is: every requirement in the job description maps directly to a specific module in the codebase. This isn't conceptual alignment — I can open any of these files right now and show you the code.
>
> The one I'll call out: 'Design, implement, and validate MVP solutions' — 362 passing tests, GitHub Actions CI, Docker multi-stage build, a production runbook. That's the operational proof."

---

### Slide 5 — LIVE DEMO *(5 minutes — most important)*

**Say before switching:**
> "Let me show you the regulatory intelligence scenario — the most directly relevant to what PwC does."

**Run** (Docker Desktop running + `docker compose -f compose.dev.yaml up -d neo4j` first):
```bash
python scripts/demo_regulatory.py --live
```

**Narrate each step as it prints:**

**Step 1 — Ontology loads:**
> "It's reading the aerospace regulatory ontology from a YAML file in `config/ontologies/`. No code change required to switch domains. `AIRWORTHINESS_DIRECTIVE` is defined as a subtype of `REGULATION` which is a subtype of `CONCEPT`. This hierarchy drives query expansion and merge decisions automatically. To switch to banking compliance, you load a different YAML file."

**Step 2 — Corpus summary:**
> "This is the live state of the graph — 12 real documents, [N] entities extracted and deduplicated by the LLM pipeline, [N] edges, [N] open conflicts. Every number is queryable in Neo4j right now." *(⚠ literally read [N] off the screen as the live demo prints it — don't pre-memorize "364/380/11"; verified live 2026-06-07 PM the actual counts were 368/422/7, having been 364/380/11 that same morning. The line works precisely because you let the terminal say the number, not you.)*

**Step 3 — Forward-chaining inference:**
> "These are the edges the inference engine derived from the live corpus — they don't exist in any document. Top of the list: 'MAX 8 fleet' related to tail number N8700L at confidence 0.90 — derived by the symmetry rule from an asserted relationship stated in the opposite direction. The rest follow the same pattern across Boeing's supply chain (Renton, WA → Boeing Factory → BOEING), facility geography (France → 50668 Cologne → Konrad-Adenauer-Ufer 3), and the regulatory hierarchy (Code of Federal Regulations → Airworthiness Directives), all at confidence 0.855. Every one is tagged source_type=inferred so it's distinguishable from what the LLM extracted directly. This is Datalog-style reasoning, not LLM guessing — and because LLM extraction is non-deterministic, the exact set of derived edges shifts slightly between ingestion runs; the reasoning method is what stays constant."

**Actual output you will see (Step 3 — live, ordered by confidence):**
```
  • MAX 8 fleet —[RELATED_TO]→ N8700L  (confidence: 0.900)
  • France —[RELATED_TO]→ 50668 Cologne  (confidence: 0.855)
  • Code of Federal Regulations —[RELATED_TO]→ Airworthiness Directives  (confidence: 0.855)
  • Boeing Factory —[RELATED_TO]→ BOEING  (confidence: 0.855)
  • Renton, WA —[RELATED_TO]→ Boeing Factory  (confidence: 0.855)
  • 50668 Cologne —[RELATED_TO]→ Konrad-Adenauer-Ufer 3  (confidence: 0.855)
  • Switzerland —[RELATED_TO]→ Zurich  (confidence: 0.855)
  • ATA 27 — Flight Controls —[RELATED_TO]→ Right aileron trim tab rod end bearing  (confidence: 0.855)
```
**Point at the top line** (MAX 8 fleet → N8700L, 0.900) — that's the symmetry rule deriving a relationship in the reverse direction from what the corpus states. The rest are the same rule firing across supply-chain, geography, and regulatory-hierarchy entities. (11 inferred edges total this run; the console sample is capped at 8 — the remaining 3, including an inverse-rule CERTIFIES edge, surface in the Neo4j query in Step 5 below.)

**Step 4 — Contradiction detection (the winning moment):**

**⚠ Anchor on conflict TYPES, not exact entity-name strings.** Like the inferred edges in Step 3, the exact conflicts shift between ingestion runs because LLM extraction is non-deterministic — entity names, aliasing, and which document pairs get flagged all vary slightly. What's *constant* across every run is the two conflict types and what they mean. Verify the actual top rows in Neo4j Browser right before you present (query below) and narrate whichever rows of each type are on screen — the mechanism and the story are identical either way.

> "Two conflict types here. `functional_violation` — the same real-world entity got extracted under multiple names or with contradictory property values across documents (here: Airbus / 'A320neo' / 'Airbus A320neo' all referring to the same aircraft program, or CFM International's engine listed as both 'LEAP-1B engine' and 'CFM LEAP-1B Engine'). `multi_source` — the same relationship asserted differently by two different source documents (here: EASA AD 2024-0072 versus EASA AD 2022-0201 — two directives in conflicting supersession order). In a banking context: same counterparty, one document says 'in compliance', another says 'under investigation'. You want to catch this before it reaches a client report. Most RAG systems would return one answer at random. This platform surfaces the conflict automatically and holds it open until a human or the authority-resolution chain settles it."

**Actual output as of the 2026-06-07 ingestion (top 10 — confirm live before presenting):**
```
  1. CFM International ⊕ ['LEAP-1B engine', 'CFM LEAP-1B Engine']               | Type: functional_violation
  2. Airbus ⊕ ['A320neo', 'Airbus A320neo']                                     | Type: functional_violation
  3. Federal Aviation Administration ⊕ United States                           | Type: multi_source
  4. BOEING ⊕ ['737 family', 'BOEING 737 MAX']                                 | Type: functional_violation
  5. Boeing Commercial Airplanes ⊕ ['BOEING 737 MAX', 'commercial jet aircraft'] | Type: functional_violation
  6. EASA AD 2024-0072 ⊕ EASA AD 2022-0201                                     | Type: multi_source
  7. Federal Aviation Administration ⊕ DOT                                      | Type: multi_source
  8. Federal Aviation Administration ⊕ FAA-2020-0481                            | Type: multi_source
  9. British Horizon Airways ⊕ G-ABCD                                           | Type: multi_source
```
**Row 6 is the one constant** — `EASA AD 2024-0072 ⊕ EASA AD 2022-0201` has survived two consecutive re-ingestions with the same `multi_source` type, so it's the safest single example to memorize if you want one fixed talking point. Everything else: point at whatever's on screen and narrate the *type*.

**To verify live before presenting:**
```cypher
MATCH (c:Conflict {tenant: 'aerospace', status: 'open'})
RETURN c.src AS src, c.tgt AS tgt, c.conflict_type AS conflict_type
ORDER BY c.conflict_type, c.src
LIMIT 10
```

**Step 5 — Authority resolution (have a fallback ready — see note below):**
> "This queries for transitive SUPERSEDES chains the inference engine derived — the kind of cross-reference an associate spends hours reconstructing manually across stacks of regulatory PDFs. [If a chain appears:] Here it surfaces 'X supersedes Y', confidence decayed from the asserted hops that produced it — derived, not stated, and pre-computed so a regulator gets a direct lookup instead of a graph traversal."

**⚠ What you will actually see this run — plan for it:**
```
  ℹ  No inferred supersession chains found
```
This happens because LLM extraction is non-deterministic — this ingestion produced 6 SUPERSEDES edges, all *asserted* (`source_type=document`), and none of them share a middle entity, so the `supersedes_transitivity` rule has nothing to chain this time. **That's a feature, not a bug, to narrate**: "The engine only derives what the corpus structure actually supports — it doesn't fabricate a chain just because the relation type exists. Watch what it *does* derive instead," then pivot directly into the Neo4j query below, which surfaces an inverse-rule edge instead: "EASA certifies the Airbus A320neo" (0.81), derived from the asserted "Airbus A320neo certified_by EASA" (0.90) — a relationship no document states in this direction. Same Datalog reasoning, different rule firing.

**After demo — switch to Neo4j Browser tab** (`http://localhost:7474`):
> "And that just persisted to real Neo4j — here are the entities and edges from what we just ran."
Run: `MATCH (n {tenant:'aerospace'}) RETURN n`

Then switch to a targeted query that shows the SUPERSEDES chain (asserted + inferred):

```cypher
MATCH (s:Entity {tenant:'aerospace'})-[r:RELATES_TO {relation: 'SUPERSEDES'}]->(t:Entity)
RETURN DISTINCT s.name AS source, r.relation AS relation_type, t.name AS target,
       r.confidence AS confidence, r.source_type AS source_type, r.source_doc_id AS document
ORDER BY r.source_type ASC, r.confidence DESC
```

⚠ **The table below is a snapshot from one specific run (verified live 2026-06-07
PM) — it is already the THIRD different version of this table this doc has carried
in one week, because LLM extraction reshuffles entity names and edge sets on every
`--wipe --commit` (see `tasks/lessons.md` A96/A98). Re-run the query above immediately
before you present and use whatever rows it returns — row count, names, and
confidences will all differ from what's printed here.** What stays true across every
run: these are all asserted directly by a source document (`source_type='document'`,
not the literal string `'asserted'` — see the terminology note above), and whether
`supersedes_transitivity` finds a chain to derive depends entirely on whether any
two of them happen to share a middle entity *this run*.

**Live snapshot, 2026-06-07 PM (7 rows — re-run to get today's actual rows):**

| source | relation_type | target | confidence | source_type |
|--------|---------------|--------|------------|------------|
| EASA AD 2024-0072 | SUPERSEDES | EASA AD 2022-0201 | 1.0 | document |
| 2024-0072 | SUPERSEDES | EASA AD 2022-0201 | 1.0 | document |
| Federal Aviation Administration | SUPERSEDES | AD 2020-05-11 | 0.95 | document |
| MCAS Software v2.0 | SUPERSEDES | MCAS Software v1.7.3 | 0.95 | document |
| MCAS Software Version 2.0 | SUPERSEDES | MCAS Software v1.7.3 | 0.95 | document |
| AD 2022-03-25 | SUPERSEDES | AD 2020-05-11 | 0.95 | document |
| AD 2024-02-15 | SUPERSEDES | AD 2024-01-02 | 0.95 | document |

*(Compare to the version this doc carried that morning: 6 rows, including
"FAA SUPERSEDES AD 2020-05-11" and "MCAS Software Version 2.0 SUPERSEDES MCAS
Software v1.7.2" at confidence 0.997/0.95 — both gone from this afternoon's
extraction, replaced by "Federal Aviation Administration" / "MCAS Software v2.0"
phrasings and a new duplicate-looking "2024-0072" entity. This is the alias-dedup
pipeline's raw material varying run to run — exactly why you read the table live.)*

To show all inferred edges (composition, symmetry, inverse rules):

```cypher
MATCH (s:Entity {tenant:'aerospace'})-[r:RELATES_TO {source_type: 'inferred'}]->(t:Entity)
RETURN DISTINCT s.name AS source, r.relation AS relation_type, t.name AS target,
       r.confidence AS confidence, r.inferred_by
ORDER BY r.confidence DESC
LIMIT 11
```

**Narrate the SUPERSEDES table (first query):**

> "[N] asserted SUPERSEDES edges this run, [zero / M] inferred — and that's worth pointing out explicitly either way. Look at the chain structure: [if none chain] none of these share a middle entity, so `supersedes_transitivity` correctly finds nothing to chain. [if one does] — here, X supersedes Y and Y supersedes Z, so the engine derived 'X supersedes Z' directly. The engine isn't pattern-matching on relation names; it's checking whether the graph *actually* contains a two-hop path before deriving a shortcut. That restraint is what makes the inferred edges trustworthy — when it does derive something, like the CERTIFIES edge below, it's because the structure genuinely supports it." *(⚠ read [N]/[M] off the query result on screen — it was 7 asserted / 0 inferred at the 2026-06-07 PM check, was 6/0 that same morning, and could be 6/1 next time two rows happen to chain. Never recite "six… zero…" from memory — that's the exact sentence that breaks under "wait, I count seven.")*

Then run the second query (all inferred edges) and narrate:

**All inferred edges — second query result (11 rows, ordered by confidence DESC):**

| row | source | relation_type | target | confidence | inferred_by |
|-----|--------|---------------|--------|------------|-------------|
| 1 | MAX 8 fleet | RELATED_TO | N8700L | 0.900 | related_to_symmetry |
| 2 | Code of Federal Regulations | RELATED_TO | Airworthiness Directives | 0.855 | related_to_symmetry |
| 3 | France | RELATED_TO | 50668 Cologne | 0.855 | related_to_symmetry |
| 4 | Boeing Factory | RELATED_TO | BOEING | 0.855 | related_to_symmetry |
| 5 | Renton, WA | RELATED_TO | Boeing Factory | 0.855 | related_to_symmetry |
| 6 | 50668 Cologne | RELATED_TO | Konrad-Adenauer-Ufer 3 | 0.855 | related_to_symmetry |
| 7 | Switzerland | RELATED_TO | Zurich | 0.855 | related_to_symmetry |
| 8 | ATA 27 — Flight Controls | RELATED_TO | Right aileron trim tab rod end bearing | 0.855 | related_to_symmetry |
| 9 | EASA | CERTIFIES | Airbus A320neo | 0.810 | certified_by_inverse |
| 10 | AOA monitoring | RELATED_TO | AOA Disagree alert | 0.810 | related_to_symmetry |
| 11 | BOEING 737 MAX | RELATED_TO | angle-of-attack | 0.765 | related_to_symmetry |

> "Row 9 is the headline: EASA certifies the Airbus A320neo, confidence 0.81 — derived by `certified_by_inverse` from the asserted edge 'Airbus A320neo certified_by EASA' at 0.90 (0.90 × 0.9 decay = 0.81). No document states the relationship in this direction; the engine derived it because the ontology defines CERTIFIES and CERTIFIED_BY as inverses. Rows 1 and 9 use two different rule types — `related_to_symmetry` and `certified_by_inverse` — both visible in the `inferred_by` column, both tagged source_type=inferred so they're distinguishable from anything the LLM extracted directly. [N] entities, [N] edges — raw LLM extractions deduplicated through 4-stage entity resolution into those canonical entities. All queryable directly, no recursion needed at query time." *(⚠ don't recite "364/380/665/45%" — query live first; verified counts shifted from 364/380 to 368/422 within the same day. State the mechanism — "4-stage alias resolution collapses duplicate mentions" — and let the live query supply the number.)*

---

### Slide 6 — Metrics / Observability *(1.5 minutes)*

**Say:**
> "If you can't measure it, you can't run it in production. These numbers come from real query runs against the live pipeline.
>
> Faithfulness is 0.937 on answerable questions — but the framing matters. Overall it's 0.842. The difference is correct refusals: when the corpus doesn't contain the answer, the system says so explicitly rather than hallucinating. RAGAS scores those refusals as zero, which is right — a system that declines is better than one that invents. I keep both numbers because 0.937 measures answer quality; 0.842 measures how often we answer at all. Alert threshold is 0.80 — we're comfortably above both. Context precision is 0.907 — almost everything retrieved is relevant.
>
> The latency number has to be split by mode. Hybrid p95 is 2.2 seconds. Agentic fallback p95 is 3.4 seconds. Reporting a single combined number hides the more important signal: agentic trigger rate. Mine is around 9-10%. If that climbs to 25%, combined p95 may still look fine while the system quietly relies on the agent to cover retrieval misses. That's the metric I watch hardest.
>
> On the right: [N] entities, [N] edges, from 12 aerospace regulatory documents — read the live numbers off the dashboard, don't recite memorized ones (they drift run-to-run; 364/380 in the morning became 368/422 by the afternoon of the *same day* — see `tasks/lessons.md` A96/A98). Raw extractions get deduplicated through the alias-resolution pipeline into those canonical entities. Contradiction density is [N] per 1,000 edges (16.59 at last live check; target < 0.85). Orphan rate is [N]% (10.9% at last live check — every extracted entity should be grounded in at least one source chunk; a nonzero rate here is honest, not a flaw, and worth acknowledging if asked). The calibration pipeline — Brier score, isotonic regression, drift trend — is built and unit-tested; it just hasn't accumulated enough live samples yet to show a populated trend, so the dashboard tab currently renders an honest empty state rather than a number I can't back up live."

**Important framing:** *"Report latency per mode, alert on trigger rate. Agentic p95 is the expected cost of the safety net; trigger rate is the early warning that the net is being called too often."*

---

### Slide 7 — The Dashboard, Live *(2 minutes)*

*This is proof moment #2.* Switch to `http://localhost:8000/admin/` — the admin dashboard is mounted inside the FastAPI app you started in step 3 of the prep checklist (`uvicorn api.main:app --port 8000`), at `/admin`. (It is *not* a standalone Dash server on its own port — `:8001` only resolves if you separately start the `dev_admin_dashboard` container from `compose.dev.yaml`, which this script doesn't ask you to do. Confirm `:8000/admin/` loads the night before.)

> "This is the operator dashboard — running against the live corpus we just ingested. Real-time observability across all 4 measurement layers."

**Click through the tabs:**

- **Graph Health** — "[N] entities, [N] edges, 12 documents ingested. Contradiction density: [N] per 1,000 edges. The live graph validates itself — any schema violation surfaces here as a health alert." *(read [N] off the gauges on screen — last live check 2026-06-10, after the supersession-chain fix: 368 entities, 422 edges, 9.48/1k density; was 16.59/1k and 28.95/1k on earlier runs that same week)*

- **Conflicts** — "[N] open conflicts detected on the real corpus — exclusive state, directional reversal, functional violation. One-click resolution with audit trail. Every resolved conflict records who resolved it, when, and why — compliance audit log." *(was 4 at last live check (2026-06-10), was 7 a few hours earlier, was 11/18 on older runs — click the tab and read the actual count)*

- **Communities** — "[N] Leiden communities with [N]% coherence. Entity drift monitor triggers a rebuild recommendation at 20% — right now it's [N]%, meaning the graph is stable and the communities are fresh." *(53 communities present at last live check (2026-06-10) — re-verify coherence % live, click the tab and read the gauge)*

- **GDPR** — "GDPR Article 17 right-to-be-forgotten. Click 'erase' and the system atomically removes that entity and all its edges, generates an audit log, keeps the graph consistent."

- **Calibration** — *(this tab now has real data — 25 samples from the 2026-06-10 faithfulness eval)* "This is the calibration pipeline — Brier score, isotonic regression, drift trend over time. As of the last eval run it has its first live snapshot: 25 samples, Brier score 0.70, currently flagged 'under-confident'. That's an honest first measurement, not a polished one — same discipline as the refusal handling in faithfulness scoring: I show you the real number and what it means, not a number I can't reproduce on demand."

---

### Slide 8 — Client Scenarios *(1.5 minutes)*

**Say:**
> "Three scenarios where I'd expect to apply this at PwC.
>
> Regulatory intelligence for banking and insurance — ingest the full regulatory corpus, detect supersessions automatically, surface contradictions before audit. That's what we just saw.
>
> Audit knowledge bases — the structured evidence — GL data, bank feeds, trial balance — is already handled by TeamMate and ACL. What those tools can't do is reason over the unstructured layer: contracts, board minutes, emails, management representations. Ask 'are there undisclosed related party transactions?' and a keyword search misses aliases and indirect relationships. A graph traversal finds: Acme Corp → director → John Smith → shareholder → Client Co — three hops across three documents that were never on the same page. The agentic retriever then iterates until every hop is grounded in a source document, not inferred.
>
> Compliance monitoring — temporal knowledge modeling tracks when facts changed. GDPR erasure is built in — when a client is decommissioned, the graph wipes all their entities and generates an audit log."

---

### Slide 9 — Technical Foundation *(1 minute)*

**Say:**
> "26,600 lines of production Python. 362 tests. 38 knowledge graph modules. Six architecture decision records — every major choice documented: property graph vs triple store, forward-chaining vs backward-chaining, Bayesian confidence accumulation, LLM provider selection, Redis result store, two-model agentic design.
>
> The two-model agentic design — 8B for routing decisions, 70B for synthesis — cut agentic p95 from 6.8 seconds to 3.4 seconds. The LLM stack uses Groq as the primary, with DeepSeek-V3 as an instant fallback on rate-limit — no sleep, no queuing. Embeddings are OpenAI `text-embedding-3-large`, 3072 dimensions. Every one of these choices has a documented ADR explaining what I considered and why I decided. That's the difference between a platform and a prototype."

---

### Slide 10 — Close *(1 minute + conversation)*

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

**"Audit firms already have TeamMate, ACL, CaseWare — why a knowledge graph?"**
> "Those tools are excellent at structured evidence: GL data, bank feeds, trial balance. They can't reason over the unstructured layer — contracts, board minutes, emails, management representations — because they're built for tabular data, not document-to-document entity relationships. The graph sits alongside them, not instead of them. It extracts entities and relationships from unstructured documents and answers questions those tools can't: 'are there undisclosed related party transactions?' requires traversing director → shareholder → client relationships across documents that have no shared row or column. That's a graph traversal, not a spreadsheet filter."

**"Wouldn't RDF/OWL be more suitable for regulatory intelligence?"**
> "For regulatory data that's already published as Linked Data — EUR-Lex, FIBO, SEC XBRL — a pure triple store with OWL-DL reasoning is the natural fit. But PwC's clients don't live there. Their regulatory knowledge is in PDFs, Word documents, internal memos, audit workpapers. There's no SPARQL endpoint for those. This platform bridges the gap: the LLM extraction pipeline turns unstructured documents into a property graph, then the OWL-RL reasoner runs on top — transitivity, symmetry, inverse rules, supersession chains are all pre-materialised. The SPARQL bridge is there too when a client has structured RDF assets to integrate. It's not either/or — it's the extraction layer that pure triple stores don't have."

**"If the agent can execute tools, how do you control it?"**
> "Every tool call passes through a `ToolPolicy` gate before execution. The policy enforces an allowlist — only registered tools can be called. Each tool declares required scopes, so `erase_entity` requires `write + admin + gdpr_officer` and is denied for any other caller. Arguments are validated against a schema: type, required fields, enum values, min/max. There's a cross-tenant guard — a caller scoped to `aerospace` can't reference `banking` in an argument. Dry-run mode lets untrusted sessions preview what would happen without executing anything. Every call — allowed, denied, or timed out — is written to an audit log. `docs/pwc-jd-mapping.md` shows exactly where each of these is in the code."

**"Can this use PwC's internal LLM?"**
> "Yes — the LLM is routed through a single module, `llm_client.py`. Three functions: `get_llm()` for synthesis, `get_fast_llm()` for routing, `get_embedder()` for embeddings. The current stack uses Groq as primary with DeepSeek as a fallback — both are already wired. Adding PwC's internal endpoint is a single-file change, because the client interface is already provider-agnostic."

**"Does the demo use real data?"**
> "Yes — I'm running with `--live`, which means real Neo4j, real ingestion, real graph queries. The mock mode exists for CI and zero-services environments. What you saw in Neo4j Browser just now came from this demo script in real time. The script has both modes; the flag makes it unambiguous."

**"What's your availability?"**
> "Immediate. Looking to contribute to PwC's Graph RAG practice — either as a full-time engineer or on a project basis to start."

---

## Part 5 — Learning Path & Reading List

### What you must be able to whiteboard cold (highest priority)

These are the topics a CTO will probe. Know them without notes.

**1. Bayesian confidence accumulation**
- Why `1−(1−c₁)(1−c₂)` and not averaging
- Answer: averaging treats confidences as independent and additive. The Bayesian formula treats them as independent evidence — if one source says 0.9 and another says 0.9, the combined confidence is 0.99 (both would have to be wrong simultaneously). Averaging gives 0.9.
- ⚠ In code: there is **no standalone `confidence_model.py`** — the formula `1.0 - (1.0 - r.confidence) * (1.0 - $confidence)` is inline Cypher in `graphrag/graph/neo4j_client.py` (merge-edge query, ~line 203) and `graphrag/graph/negative_knowledge.py` (~line 109). ADR-0003 (`docs/adr/0003-bayesian-confidence-accumulation.md`) is where the reasoning is documented.

**2. Forward-chaining inference vs backward-chaining**
- Forward: start from known facts, derive everything reachable (materialise derived edges into the graph)
- Backward: start from a query goal, work backwards to find supporting facts (not materialised — computed at query time)
- Why forward for regulatory supersession: you want transitive supersession chains pre-computed so queries are fast; the chain is static between updates
- ⚠ In code: `graphrag/graph/inference_engine.py` — **not** `graphrag/inference/forward_chaining.py` (that path doesn't exist; there is no `graphrag/inference/` package)

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
- ⚠ In code: `graphrag/graph/alias_registry.py` — **not** `entity_resolver.py` (that file doesn't exist; the docstring there describes the same exact/normalized/embedding(0.92)/queue-for-review pipeline)

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
- Trigger logic: fallback only when the answer hedges AND has zero citations. The earlier OR condition over-fired on sparse corpora, pushing agentic usage toward ~40%. The AND condition brought trigger rate back to ~9-10% without a sampled quality drop.
- Operational rule: watch agentic trigger rate more closely than agentic p95. Agentic p95 tells you the safety net is expensive; trigger rate tells you how often retrieval needs rescuing.
- In code: `graphrag/retrieval/agentic_retriever.py` — `_reason()` and `_synthesize()`

**8. Provider split: Groq primary + DeepSeek fallback, OpenAI embeddings (ADR-0004)**
- Embeddings: OpenAI `text-embedding-3-large`, 3072d — switched from Gemini; same dimensionality, same schema, no re-indexing needed after migration
- Generation: Groq `llama-3.3-70b-versatile` — primary synthesis; ~150 tok/s
- Routing: Groq `llama-3.1-8b-instant` — ~800 tok/s, trivial structured output; ~0.2s per routing step
- Fallback: DeepSeek-V3 (`deepseek-chat`) — fires immediately on Groq rate-limit, no sleep; same OpenAI-compatible API
- Why DeepSeek over queuing or retry: a rate-limited generation request needs an answer now, not in 60 seconds; DeepSeek is instant and free-tier
- Client swap: change `get_llm()` / `get_fast_llm()` / `get_embedder()` in `llm_client.py` — one file, nothing else changes

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
- In practice: read `graphrag/graph/alias_registry.py` directly — the 4-stage pipeline (exact / normalized / embedding-cosine ≥0.92 / queue-for-review) is documented inline in its module docstring

**Knowledge Graph Embeddings**
- Bordes et al. (2013): "Translating Embeddings for Modeling Multi-Relational Data" — the TransE paper (link prediction module uses this)
- ⚠ In code: `graphrag/graph/link_predictor.py` — **not** `graphrag/ml/link_predictor.py` (there is no `graphrag/ml/` package; the module lives under `graphrag/graph/`)

**Community Detection**
- Traag et al. (2019): "From Louvain to Leiden: guaranteeing well-connected communities" — the Leiden algorithm used for community detection
- ⚠ In code: `graphrag/graph/community_builder.py` — **not** `community_detector.py` (that file doesn't exist; coherence scoring itself is a separate method, `GraphEvaluator.community_coherence()`, in `graphrag/graph/graph_evaluator.py`)

---

### Concepts to be comfortable explaining in conversation

| Concept | One-line answer |
|---|---|
| RRF (Reciprocal Rank Fusion) | Merges two ranked lists by `1/(k + rank)` — avoids score normalization across different retrieval systems |
| Bitemporal modeling | Two time axes: valid time (when the fact was true in the world) and transaction time (when we recorded it) |
| Leiden vs Louvain | Leiden guarantees well-connected communities; Louvain can produce internally disconnected ones |
| Brier score | Mean squared error between predicted probability and binary outcome; 0=perfect, 0.25=random |
| Isotonic regression calibration | Non-parametric monotone transformation that maps raw model confidence to empirical frequency |
| Agentic trigger rate | Fraction of queries that fall from hybrid retrieval into agentic fallback; leading indicator of retrieval health |
| Cross-encoder vs bi-encoder | Bi-encoder: embed query and doc separately (fast, approximate). Cross-encoder: score (query, doc) jointly (slow, accurate). Use bi-encoder for ANN, cross-encoder for reranking top-N. |
| GNN message passing | Each node aggregates features from its neighbours; graph structure encodes entity relationships |
| Datalog transitivity rule | If A→B and B→C then A→C; applied to fixpoint (until no new edges can be derived) |

---

## Part 6 — The Demo Visual Guide

### What to show during Slide 5

**Show terminal in real-time.** Highlight these 4 sections as they scroll:

**1. Corpus summary (Step 2)**
- Point at: `Entities: 364 | Relations: 380 | Documents: 12 | Open conflicts: 11`
- Say: "This is the live state of the knowledge graph — 12 real documents, [N] entities extracted and deduplicated by the LLM pipeline, [N] detected conflicts. Every number is queryable." *(run the query immediately below first and read [N] from its output — was 368/7 at last live check, was 364/11 hours earlier; see A96/A98)*
- **To verify in Neo4j Browser:**
  ```cypher
  MATCH (e:Entity {tenant: 'aerospace'}) WITH count(e) AS entity_count
  MATCH (d:Document {tenant: 'aerospace'}) WITH entity_count, count(d) AS doc_count
  MATCH (:Entity {tenant: 'aerospace'})-[r:RELATES_TO]->(:Entity)
  WITH entity_count, doc_count, count(r) AS edge_count
  MATCH (c:Conflict {tenant: 'aerospace', status: 'open'})
  RETURN entity_count, doc_count, edge_count, count(c) AS conflict_count
  ```

**2. Inferred edges (Step 3)**
- Point at: `MAX 8 fleet —[RELATED_TO]→ N8700L  (confidence: 0.900)`
- Say: "The inference engine derived this edge — and 10 others — from asserted relationships stated in the opposite direction in the corpus. Confidence 0.90 is the symmetry rule's decay from the 1.0 asserted source. None of these are stated by any document directly; all are tagged source_type=inferred. Pure Datalog reasoning, not LLM guessing — and because LLM extraction is non-deterministic, the exact set shifts slightly between ingestion runs."
- **To verify in Neo4j Browser:**
  ```cypher
  MATCH (s:Entity {tenant: 'aerospace'})-[r:RELATES_TO {source_type: 'inferred'}]->(t:Entity)
  RETURN DISTINCT s.name AS src, t.name AS tgt, r.relation AS rel, r.confidence, r.inferred_by
  ORDER BY r.confidence DESC
  ```

**3. Contradiction detection (Step 4 — the winning moment)**
- Point at: `EASA AD 2024-0072 ⊕ EASA AD 2022-0201 | Type: multi_source` — this pair has survived two consecutive re-ingestions, the most reliable single example to anchor on.
- Say: "Two EASA directives with contradictory supersession information on the same subject, flagged automatically as multi_source. And [point at whichever functional_violation row is on screen, e.g. Airbus vs '[A320neo, Airbus A320neo]' or CFM International vs '[LEAP-1B engine, CFM LEAP-1B Engine]'] — a functional violation, the same real-world entity extracted under multiple names or with contradictory attributes across documents. This is what an auditor needs to catch before a report goes out."
- ⚠ Don't memorize exact entity-name strings beyond the EASA AD pair — LLM extraction is non-deterministic, so the *specific* functional_violation rows shift between ingestions (confirm live with the query in Step 4 above before presenting). The conflict *types* and what they mean never change.
- **To verify in Neo4j Browser:**
  ```cypher
  MATCH (c:Conflict {tenant: 'aerospace', status: 'open'})
  RETURN c.src, c.tgt, c.conflict_type, c.doc_a, c.doc_b
  ORDER BY c.conflict_type, c.src
  LIMIT 20
  ```

**4. Authority resolution (Step 5) — has two possible outcomes; be ready for both**
- The live query filters for `SUPERSEDES` edges with `source_type: 'inferred'`. **This run returns none** — the script will print "No inferred supersession chains found" — because the 6 asserted SUPERSEDES edges this round don't share a middle entity, so `supersedes_transitivity` has nothing to chain.
- Say (when that happens): "No transitive chain surfaced this run — and that's worth calling out: the engine only derives a shortcut when the graph structure actually supports a two-hop path. It doesn't fabricate connections just because the relation type exists. Watch what it *does* derive instead" — then pivot straight into the inverse-rule edge: "EASA certifies the Airbus A320neo, confidence 0.81 — derived from the asserted 'Airbus A320neo certified_by EASA' at 0.90. No document states it in this direction; the ontology's CERTIFIES/CERTIFIED_BY inverse mapping derived it."
- If a future run *does* produce a transitive SUPERSEDES chain, narrate it the same way the old script did: "The engine follows the transitive chain and identifies the current governing authority — confidence decayed from the asserted hops that produced it. In a regulatory practice, this is exactly what an associate spends hours tracing manually."
- **To verify in Neo4j Browser:**
  ```cypher
  MATCH (a:Entity {tenant: 'aerospace'})-[r:RELATES_TO {relation: 'SUPERSEDES', source_type: 'inferred'}]->(b:Entity)
  RETURN DISTINCT a.name AS newer, b.name AS older, r.confidence
  ORDER BY r.confidence DESC
  ```
  ```cypher
  -- fallback: the inverse-rule edge that IS live this run
  MATCH (s:Entity {tenant: 'aerospace'})-[r:RELATES_TO {relation: 'CERTIFIES', source_type: 'inferred'}]->(t:Entity)
  RETURN s.name AS source, t.name AS target, r.confidence, r.inferred_by
  ```

### What NOT to show

- Don't dwell on mock vs live — run `--live` and it's real
- Don't read line-by-line; let the visual structure speak
- Don't explain `LCA()` or `expand_type()` unless they ask — it's proof of depth, not the story

### The YAML side-by-side (optional strong move)

Open `config/ontologies/aerospace_regulatory.yml` in VS Code alongside the terminal.

> "See how the type pairs in the YAML file match the output? This YAML file, loaded at startup, defines the entire type system and relation constraints for the domain. 28 type pairs, 12 relation rules. To scale to banking or audit domains, we just swap the YAML file."

### Quick reference: Neo4j Browser queries (copy/paste ready)

After the demo, open Neo4j Browser at `http://localhost:7474` and run these queries to verify the live data:

**All entities and relationships:**
```cypher
MATCH (n {tenant: 'aerospace'}) RETURN n LIMIT 50
```

**Only inferred edges (derived by forward-chaining):**
```cypher
MATCH (s:Entity {tenant: 'aerospace'})-[r:RELATES_TO {source_type: 'inferred'}]->(t:Entity)
RETURN s.name AS src, t.name AS tgt, r.relation, r.confidence, r.inferred_by
ORDER BY r.confidence DESC
```

**Only asserted edges (LLM-extracted, no inference):**
⚠ The live property value is `'document'`, not `'asserted'` — the query below as filtered on `'asserted'` returns **zero rows** (verified live). Use `'document'`:
```cypher
MATCH (s:Entity {tenant: 'aerospace'})-[r:RELATES_TO {source_type: 'document'}]->(t:Entity)
RETURN s.name AS src, t.name AS tgt, r.relation, r.confidence
ORDER BY r.confidence DESC
LIMIT 20
```

**All open conflicts ([N] total — was 7 at last live check 2026-06-07 PM, was 11 that same morning, was 18 on an earlier run; run the query below to get today's true count):**
```cypher
MATCH (c:Conflict {tenant: 'aerospace', status: 'open'})
RETURN c.src, c.tgt, c.conflict_type, c.doc_a, c.doc_b
ORDER BY c.conflict_type
```

**Breakdown: edge source types:**
```cypher
MATCH (s:Entity {tenant: 'aerospace'})-[r:RELATES_TO]->(t:Entity)
RETURN r.source_type, count(r) AS count
ORDER BY count DESC
```

---

## Part 7 — Code Reference (for pointing at files)

| What they ask about | File to open | Lines |
|---|---|---|
| Type hierarchy definition | `config/ontologies/aerospace_regulatory.yml` | 18–45 |
| Relation rules | `config/ontologies/aerospace_regulatory.yml` | 61–80 |
| ⚠ Live demo Step 1 (ontology) — what `--live` actually runs | `scripts/demo_regulatory.py` | `run_live_demo()`, ~461–475 |
| ⚠ Live demo Step 3 (inference) — what `--live` actually runs | `scripts/demo_regulatory.py` | `run_live_demo()`, ~497–531 |
| ⚠ Live demo Step 4 (contradiction) — what `--live` actually runs | `scripts/demo_regulatory.py` | `run_live_demo()`, ~532–562 |
| ⚠ Live demo Step 5 (authority resolution) — what `--live` actually runs | `scripts/demo_regulatory.py` | `run_live_demo()`, ~563–589 |
| (Conceptual/mocked walkthrough — NOT what `--live` runs; `step1_load_ontology`…`step5_contradiction`, used by plain `run_demo()`) | `scripts/demo_regulatory.py` | lines 101–328 |
| Forward-chaining engine | `graphrag/graph/inference_engine.py` | — |
| Neo4j client | `graphrag/graph/neo4j_client.py` | — |
| 6-stage retrieval | `graphrag/retrieval/hybrid_retriever.py` | — |
| Agentic IRCoT (two-model) | `graphrag/retrieval/agentic_retriever.py` | `_reason()` / `_synthesize()` |
| Confidence accumulation (Bayesian formula, inline Cypher) | `graphrag/graph/neo4j_client.py` (~203) / `negative_knowledge.py` (~109) | — |
| Entity resolver / alias dedup (4-stage pipeline) | `graphrag/graph/alias_registry.py` | — |
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
| Golden eval runner | `scripts/run_golden_eval.py` | `python scripts/run_golden_eval.py` |
| JD mapping (with Gap column) | `docs/pwc-jd-mapping.md` | — |

---

## Part 8 — Preparation Checklist (before the meeting)

### Technical fluency (the real bottleneck)
- [ ] Whiteboard the Bayesian confidence formula from memory — derive it, explain why not averaging
- [ ] Explain the 4-stage entity resolution pipeline: what fails at each stage, what the thresholds mean
- [ ] Run `make smoke-test` — confirm it exits 0 before the meeting (362 tests)
- [ ] Know `docs/pwc-jd-mapping.md` — open it when any JD question comes up
- [ ] Know the answer to "how do you control the agent?" cold (ToolPolicy, 4 risk levels, audit log)
- [ ] Explain ADR-0001 (property graph vs triple store) conversationally — what you considered and why you decided
- [ ] Explain ADR-0002 (forward-chaining over backward-chaining) — why materialise derived edges, why not compute at query time
- [ ] Explain ADR-0003 (Bayesian confidence accumulation) — derive `1−(1−c₁)(1−c₂)` from memory, why not averaging
- [ ] Explain ADR-0004 (Groq + DeepSeek fallback, OpenAI embeddings) — why DeepSeek over retry, why OpenAI over Gemini, client swap path
- [ ] Explain ADR-0005 (Redis result store) — why not PostgreSQL, why not RabbitMQ reply-to
- [ ] Explain ADR-0006 (8B routing + 70B synthesis) — the latency table, why not 8B for synthesis
- [ ] Explain the 6 retrieval stages in order, what failure mode each one fixes
- [ ] Explain the two-model agentic design: which model does what, why, latency numbers
- [ ] Explain agentic trigger rate: why ~9-10% is healthy, why >20% is an alert, and why combined p95 hides the signal
- [ ] Explain the OR-to-AND fallback fix: hedged answer OR zero citations over-fired; hedged answer AND zero citations reduced unnecessary agentic calls
- [ ] Explain contradiction detection: name all 5 types, explain `positive_negative_pair` with an example
- [ ] State the real RAGAS numbers cold: faithfulness 0.785 on the full 39-question golden set (25 scored, 14 refusals), baseline 0.840 — and explain why the refusal split matters, and why `architecture`/`domain` questions correctly score 0/refused (no info-leak about the system's own internals from an aerospace corpus)
- [ ] State the real latency numbers: hybrid p95 2.2s, agentic p95 3.4s, combined p95 2.7s, trigger rate ~9-10%

### Demo preparation
- [ ] Start Docker Desktop — wait for whale icon to stop animating
- [ ] Run `docker compose -f compose.dev.yaml up -d neo4j` — wait ~20 seconds for Neo4j to be ready
- [ ] Confirm corpus is loaded: `docker exec dev_neo4j cypher-shell -u neo4j -p graphrag_dev "MATCH (e:Entity {tenant:'aerospace'}) RETURN count(e)"` — should return **a non-zero count in the low-to-mid hundreds** (⚠ don't expect a fixed number — it was 364 on the morning of 2026-06-07 and 368 that same afternoon after one re-ingestion; see A96/A98. Just confirm it's non-zero and roughly in that range — that's "corpus is loaded", not "matches some memorized figure"). The running container is named `dev_neo4j`, not `graphrag_neo4j` — verify with `docker ps` if it ever changes.
- [ ] If corpus is missing (returns 0): run `python scripts/ingest_corpus.py --commit --wipe` (takes 20–30 min — do this the night before)
- [ ] Run `python scripts/demo_regulatory.py --live` — confirm all 5 steps complete cleanly
- [ ] Have Neo4j Browser open at `http://localhost:7474`, confirm query returns data
- [ ] Practise narrating Steps 2, 3, 4 out loud without reading the output

### Materials
- [ ] Open `C:\Users\Sergiu\Desktop\GraphRAG_PwC_Pitch.pptx` and click through all slides
- [ ] Have `config/ontologies/aerospace_regulatory.yml` open in VS Code (ready to show)
- [ ] GitHub repo open in browser: `https://github.com/sergiunicoara/Generative-AI/tree/main/AI%20knowledge%20graph%20platform`

### Soft preparation
- [ ] Prepare one honest answer for "no commercial experience" — rehearse it until it sounds natural, not defensive
- [ ] Prepare the "AI-authorship" pre-empt: one or two things only the builder would know (a dead-end you hit, a decision you reversed)
- [ ] Prepare for "Tell me about a time you disagreed with a teammate" — something that doesn't require the project
