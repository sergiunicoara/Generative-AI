# Presentation Playbook — GraphRAG for Enterprise Intelligence

Everything you need to deliver the pitch as one coherent performance: the deck,
the live demo, the live dashboards, and the Q&A defense — sequenced, scripted,
and with fallbacks for when something doesn't cooperate.

> **The one-line story you're selling:** *"I built a production-grade GraphRAG
> platform that reasons over a knowledge graph, detects contradictions, and is
> fully observable — and every claim is verifiable in code, live."*

---

## 1. Asset map — what you're integrating

| Asset | File / URL | Role in the talk |
|---|---|---|
| **Slide deck (10 slides)** | `C:\Users\Sergiu\Desktop\GraphRAG_PwC_Pitch.pptx` | The spine of the talk |
| **Live regulatory demo** | `scripts/demo_regulatory.py` (`--live` for real Neo4j) | Proof moment #1 |
| **Admin dashboard** | `http://localhost:8001/admin/` (demo mode) | Proof moment #2 |
| **Business KPI dashboard** | `http://localhost:8050/dashboard/` | Proof moment #2 (metrics) |
| **Defensibility drill** | `docs/defensibility-drill.md` | Q&A defense (15 CTO questions) |
| **Terminology reference** | `docs/graphrag-terminology.md` | Q&A vocabulary backup |
| **Metrics inventory** | `docs/performance-metrics-inventory.md` | Q&A metrics backup |

The deck already *references* both proof moments (slide 6 = metrics framework,
slide 7 = the dashboard view). The playbook is what turns those references into
live, in-the-room demonstrations.

---

## 2. One-time setup (do this once, days before)

```powershell
# From the project root
cd "C:\Users\Sergiu\Desktop\Projects\Generative-AI\AI knowledge graph platform"

# 1. Dependencies (the dashboard mount needs a2wsgi)
py -3.11 -m pip install -r requirements.txt
py -3.11 -c "import a2wsgi; print('a2wsgi OK')"

# 2. Smoke-test the deck regenerates (only if you edit it)
cd "C:\Users\Sergiu\Desktop"; node graphrag_pitch.js   # → GraphRAG_PwC_Pitch.pptx

# 3. Smoke-test the mock demo runs with zero backend
cd "C:\Users\Sergiu\Desktop\Projects\Generative-AI\AI knowledge graph platform"
py -3.11 scripts/demo_regulatory.py        # should print 6 steps, no errors

# 4. Confirm the dashboard launches in demo mode
$env:GRAPHRAG_DASHBOARD_DEMO = "1"
py -3.11 -m uvicorn api.main:app --port 8001
#   → open http://localhost:8001/admin/  → all 5 tabs populated → Ctrl-C
```

**Decision to make up front — live or mock demo?**

| | Mock demo (default) | Live demo (`--live`) |
|---|---|---|
| Needs | nothing | Docker + Neo4j running |
| Risk | ~zero | Neo4j must be healthy |
| Credibility | high | highest ("real database") |

Recommendation: **rehearse both**. Default to `--live` if you control the
environment and Docker is reliable; keep mock as the instant fallback.

---

## 3. Pre-flight checklist (T-15 minutes, in the room)

Run these so everything is warm before you start. Use **three terminals** and a
browser with tabs pre-opened.

**Terminal 1 — (only if doing the live demo) start Neo4j:**
```powershell
cd "C:\Users\Sergiu\Desktop\Projects\Generative-AI\AI knowledge graph platform"
# Quick: just Neo4j
docker compose -f compose.dev.yaml up -d neo4j
# OR full stack (API + workers + dashboards auto-start):
# docker compose -f compose.dev.yaml up -d
# wait ~30s, then verify:
curl http://localhost:7474     # Neo4j Browser responds
# Seed demo data (if graph is empty):
py -3.11 scripts/seed_demo_data.py --commit
```

**Terminal 2 — start the admin dashboard in demo mode:**
```powershell
cd "C:\Users\Sergiu\Desktop\Projects\Generative-AI\AI knowledge graph platform"
$env:GRAPHRAG_DASHBOARD_DEMO = "1"
py -3.11 -m uvicorn api.main:app --port 8001
# leave running → http://localhost:8001/admin/
```

**Terminal 3 — start the business KPI dashboard (optional):**
```powershell
cd "C:\Users\Sergiu\Desktop\Projects\Generative-AI\AI knowledge graph platform"
py -3.11 graphrag/business_matrix/dashboard_server.py
# leave running → http://localhost:8050/dashboard/
```

**Terminal 4 — keep this one empty and ready** to run the demo on cue:
```powershell
cd "C:\Users\Sergiu\Desktop\Projects\Generative-AI\AI knowledge graph platform"
# (you'll type the demo command live during slide 5)
```

**Browser — pre-open these tabs (in order):**
1. `http://localhost:8001/admin/` (Graph Health tab showing)
2. `http://localhost:7474` (Neo4j Browser — only if live demo)
3. `https://github.com/sergiunicoara/Generative-AI` (the repo, for "it's all real")

**Pre-flight checklist:**
- [ ] Deck open in PowerPoint, presenter view, slide 1
- [ ] Terminal 4 cd'd into the project, font size bumped (Ctrl+Scroll) so the room can read it
- [ ] `http://localhost:8001/admin/` loads with gauges visible
- [ ] `docs/defensibility-drill.md` open in a side window (your Q&A safety net)
- [ ] Phone/laptop on Do-Not-Disturb; notifications off; screen-share confirmed

---

## 4. Run of show (≈ 20 minutes)

Slide-by-slide. **Bold** = the single sentence that must land. *Italic* = stage
direction.

### Slide 1 — Title (30s)
> "I'm going to show you a GraphRAG platform I built end-to-end — and rather than
> talk about it abstractly, **I'll run it live and you can challenge any claim.**"

### Slide 2 — The Problem (1m)
Name the three failures: hallucination, no reasoning chain, no audit trail.
> "Standard RAG retrieves similar text. **It can't reason across connected facts,
> and it can't tell you when two documents contradict each other** — which is
> exactly what a regulated client cannot tolerate."

### Slide 3 — Architecture (2m)
Walk the three layers left-to-right (Knowledge Graph → Retrieval → Agent).
> "Three layers, one system. The graph stores facts and inference; the 6-stage
> pipeline retrieves them; the agent reasons and answers with citations."
*If asked to go deeper here, defer: "I'll show each of these running in two minutes."*

### Slide 4 — Capabilities / JD mapping (1.5m)
> "Every requirement on the role maps to something already built — **and the
> right-hand column is the file you can grep for.**"

### Slide 5 — Live Demo cue (then SWITCH to Terminal 4) (4–5m)
*This is proof moment #1.* Switch to the terminal and run:

```powershell
# Live (real Neo4j) — preferred:
py -3.11 scripts/demo_regulatory.py --live

# OR mock (instant, zero backend) — fallback:
py -3.11 scripts/demo_regulatory.py
```

Narrate each of the 6 steps as it prints (don't read the screen silently):
1. **Ontology loads from YAML** — "domain knowledge is config, not code."
2. **Domain/range validation** — "the graph rejects nonsense relations."
3. **Forward-chaining inference** — "**AD-2024 transitively supersedes AD-2020 —
   the system derived that, no one wrote it.**"
4. **Contradiction detection** — "**same aircraft flagged airworthy AND
   unairworthy — caught automatically.**"
5. Authority chain query — "which document currently governs?"
6. (live only) "and that just persisted to a real Neo4j — here it is:"
   *switch to Neo4j Browser tab, run* `MATCH (n {tenant:'aerospace'}) RETURN n`.

### Slide 6 — Observability / metrics framework (1.5m)
> "If you can't measure it, you can't run it in production. **16 metrics across 4
> layers.** The real numbers from 104 query runs: faithfulness 0.840, context precision
> 0.907, hybrid p95 **2.2s**. The real corpus (12 aerospace regulatory documents,
> LLM-extracted): **374 entities, 456 relations, 70 open conflicts, 99.6% high-confidence
> edges, 0% orphans** — all verified live in Neo4j.
> The calibration pipeline corrects LLM confidence from raw to isotonic-adjusted —
> Brier score is the target metric once the corpus scales."

### Slide 7 — The Dashboard, Live (then SWITCH to browser tab 1) (3m)
*This is proof moment #2.* Switch to `http://localhost:8001/admin/`.
> "This is the operator dashboard — running against the seed data we just ingested."
*Click through the tabs:*
- **Conflicts** — "70 open conflicts detected on the real corpus — exclusive state, directional
  reversal, functional violation. One-click resolution with audit trail."
- **Communities** — "0% entity drift since the last Leiden rebuild — communities are fresh. The drift monitor triggers a rebuild recommendation at 20%."
- **GDPR** — "GDPR Article 17 right-to-be-forgotten — with an audit log per request."
- **Calibration** — "The calibration pipeline is wired — isotonic regression corrects
  raw LLM confidence. On a production corpus this targets Brier < 0.20."
*(Optional: switch to `:8050/dashboard/` for the latency/RAGAS time-series.)*

### Slide 8 — Client Scenarios (2m)
Tie it to their world: regulatory intelligence, audit KB, compliance monitoring.
> "Aerospace was the demo. **The same engine maps to banking, audit, and
> insurance — swap the YAML ontology, nothing else changes.**"

### Slide 9 — Technical Foundation (1.5m)
> "22,650 lines, 353 passing tests (49 are agent safety guardrails), 39 KG modules, 6 ADRs, 82 documented lessons. **This is a product,
> not a prototype.**"

### Slide 10 — Close (1m)
> "Everything I showed is in the repo. **Grep for the function, run the test, see
> it work.** What's the next step?"

---

## 5. Q&A playbook — map questions to your prepared answers

Keep `docs/defensibility-drill.md` open. The 15 questions there are pre-answered.
Fast lookup:

| If they ask about… | Go to drill question |
|---|---|
| Why Neo4j over a vector DB | Q1 |
| Forward vs backward chaining | Q2 |
| The Bayesian confidence math | Q3 |
| The 6 retrieval stages | Q4 |
| Agentic / IRCoT fallback | Q5 |
| Entity resolution | Q6 |
| Ontology / YAML design | Q7 |
| Contradiction types | Q8 |
| "Is the demo mocked?" | Q9 — *answer: run `--live` if you haven't* |
| Test coverage | Q10 |
| Multi-tenancy / isolation | Q11 |
| Scale limits | Q12 |
| Why Groq + Gemini | Q13 |
| Handling wrong extractions | Q14 |
| "No commercial experience" | Q15 — *be honest, lean on the foundation* |

Vocabulary curveball (a term you blank on)? `docs/graphrag-terminology.md`.
Metric definition challenge? `docs/performance-metrics-inventory.md`.

**Golden rule:** if challenged on whether something is real, **don't argue —
show it.** Switch to the terminal, grep the function, run the test.

---

## 6. Timing variants

| Slot | What to cut |
|---|---|
| **5 min** (elevator) | Slides 1 → 5 (mock demo, steps 3+4 only) → 10. Skip dashboards. |
| **20 min** (standard) | Full run of show above. |
| **45 min** (deep dive) | Full + live `--live` demo + Neo4j Browser + all dashboard tabs + open an ADR on screen + walk one test in the repo. |

---

## 7. Failure fallbacks (rehearse these)

| If… | Then… |
|---|---|
| Neo4j won't start | Drop to mock demo: `py -3.11 scripts/demo_regulatory.py`. Say "running the in-process test harness" — it's true. |
| Dashboard won't load | **Slide 7 IS the dashboard** (native gauges) — present it as the screen. You lose nothing. |
| `/admin` 404s | You forgot `a2wsgi` or the env var. Fallback to slide 7. (Root cause: `grep admin_dashboard_unavailable` in the API log.) |
| Internet down | Everything is local except the GitHub tab — skip it. The demo and dashboards are offline-capable. |
| Projector mangles colors | The deck was built bright for projectors; the dashboards less so — lean on the deck + terminal. |
| You blank on a question | "Great question — let me show you rather than tell you," then grep/run. Buys time and proves the point. |

---

## 8. Teardown

```powershell
# Stop the dashboards / API (Ctrl-C in each terminal), then:
docker compose -f compose.dev.yaml down        # stops all dev services
# Add -v to also wipe volumes (fresh start next time):
# docker compose -f compose.dev.yaml down -v
```

The live demo writes to the `aerospace` tenant; it self-clears on the next
`--live` run, so no manual cleanup is needed between rehearsals.

---

## 9. The 60-second version (if you only remember one thing)

1. Open deck. 2. Run `py -3.11 scripts/demo_regulatory.py` — narrate the
contradiction being caught. 3. Open `http://localhost:8001/admin/` (demo mode) —
point at the green gauges. 4. "It's all in the repo — grep it, run it." Done.
