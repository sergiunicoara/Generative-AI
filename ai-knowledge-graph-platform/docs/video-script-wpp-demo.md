# Demo Script — GraphRAG for WPP Open (AdTech/MarTech)

> Numbers measured on the WPP demo corpus (4 documents, 2 engineered contradictions):
> **51 entities · 49 edges · 2/2 contradictions detected**, **RAGAS faithfulness = 0.95**, **marketing tenant isolated**.

---

## Quick Start — 3 commands

```powershell
# 1. Start services
docker-compose up -d neo4j rabbitmq
docker start graphrag_redis   # container already exists; first run: docker run -d -p 6379:6379 --name graphrag_redis redis:7-alpine

# 2. Start API (Terminal 1)
$env:GRAPHRAG_DEFAULT_TENANT = "marketing"
$env:PYTHONUTF8 = "1"
python -m uvicorn api.main:app --port 8000

# 3. Start worker (Terminal 2)
$env:GRAPHRAG_DEFAULT_TENANT = "marketing"
$env:PYTHONUTF8 = "1"
python workers/query_worker.py
```

**Then open:** `http://localhost:8000/demo`

---

## Preparation before the interview / recording

### A. API keys (once, in `.env`)

```env
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=sk-...
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphrag_dev
RABBITMQ_URL=amqp://graphrag:graphrag_dev@localhost:5672/
REDIS_URL=redis://127.0.0.1:6379/0
ENV=development
```

### B. Start services

Start **Docker Desktop** first. Wait for the green icon (~30-60s).

```powershell
docker-compose up -d neo4j rabbitmq
docker start graphrag_redis   # container already exists; first run: docker run -d -p 6379:6379 --name graphrag_redis redis:7-alpine
```

Verify: `docker ps` → 3 containers running (`graphrag_neo4j`, `graphrag_rabbitmq`, `graphrag_redis`).

### C. Ingest the WPP corpus (if not already done)

**Check first:**
```powershell
python scripts/check_counts.py --tenant marketing
```
Or in Neo4j Browser (`http://localhost:7474`):
```cypher
MATCH (e:Entity {tenant: "marketing"}) RETURN count(e) AS entities
MATCH ()-[r {tenant: "marketing"}]->() RETURN count(r) AS edges
```
Expected: **51 entities, 49 edges, 2 open conflicts**.

**If empty, run ingestion:**
```powershell
$env:GRAPHRAG_DEFAULT_TENANT="marketing"
python scripts/ingest_corpus.py --tenant marketing --commit --wipe
```
Duration: ~5-10 min (4 documents). Should finish with: `51 entities · 49 edges`. Then seed conflicts:
```powershell
python -m scripts.seed_marketing_conflicts   # or re-run the seed script manually
```

### D. Verify PageRank scores are persisted

Run once during prep to confirm GDS is working (you'll run it live again during the demo):

```powershell
python -m scripts.pagerank_compute --tenant marketing
```

Expected output (top 3):
```
=== PageRank top entities — tenant: marketing ===
   1. Brand Guideline                              [PolicyDocument]  score=1.40290
   2. Statement of Work                            [Contract]        score=1.33...
   3. ...
```

### E. Tabs to have open before the interview

| Tab | Content |
|-----|---------|
| 1 | `http://localhost:8000/demo` (chat UI, auto-authenticated) |
| 2 | File Explorer at `data/wpp_demo/` (4 .txt files) |
| 3 | Neo4j Browser at `http://localhost:7474` |
| 4 | `data/wpp_demo/graphify-out/graph.html` (knowledge graph visual) |

---

## DEMO SCRIPT — English (interview or recording)

### [0:00-0:10] — Open with the question, not the slides

*On screen: `http://localhost:8000/demo` already open.*

Say:
> "I built something before this call that maps directly to what WPP Open works on.
> Can I show you one scenario?"

---

### [0:10-0:30] — Show the corpus

*Switch to File Explorer → `data/wpp_demo/` — 4 files visible.*

Say:
> "Four documents. A Statement of Work, a Brand Guideline, a Campaign Brief, and a Data Privacy Policy.
> This is exactly the document set that governs a global paid media campaign.
> The question is: can your AI tell you when a campaign brief contradicts a binding contract?"

---

### [0:30-1:00] — Question 1: Simple compliance lookup

*In the chat UI, type or click:*
```
What ad categories are excluded under the Nova Beverages SOW?
```

*Wait for answer (~3-5s).*

Say:
> "Simple lookup. The answer comes back with the exact source — SOW Section 2, three excluded categories.
> Document cited, section cited. If it can't cite, it refuses to answer."

---

### [1:00-1:45] — Question 2: The contradiction

*Type:*
```
Can we run sports-betting placements in Germany for the EU Q3 SummerRush campaign?
```

*Wait for answer.*

Say:
> "This is where it gets interesting. The Campaign Brief approved sports-betting companion apps
> as an adjacency vertical — the EU Desk Regional Director signed off on it.
>
> But the system flags two independent violations:
> First — the SOW Section 2 strictly excludes gambling and sports-betting placements.
> SOW Section 4 says the SOW prevails over any Campaign Brief. That's a material breach.
>
> Second — the Data Privacy Policy Section 3 prohibits gambling-adjacent behavioral inference
> independent of any consent mechanism. Section 4 says the DPP is legally binding and supersedes
> any campaign-level approval.
>
> The Campaign Brief had no valid path to approval. The graph found this by traversing four documents,
> resolving the authority chain, and detecting the conflict. A human reviewer looking only at the
> Brief and the Brand Guideline would have missed it."

---

### [1:45-2:15] — Question 3: Authority resolution

*Type:*
```
Which document has the highest authority over campaign decisions?
```

*Wait for answer.*

Say:
> "The graph encodes the authority hierarchy explicitly — not just as metadata,
> but as GOVERNS and SUPERSEDES edges between document nodes.
> SOW outranks the Brand Guideline. The DPP supersedes campaign-level approvals.
> Every answer is resolution-aware."

---

### [2:15-2:45] — Show the graph visualization

*Switch to Tab 4: `data/wpp_demo/graphify-out/graph.html`.*

Say:
> "This is the knowledge graph for these four documents. Four communities, one per document.
> The cross-community edges you see here — those are the contradiction paths.
> SOW PROHIBITS sports-betting-placements. Campaign Brief PERMITS it. Same target node, two opposing
> edges. That's how the contradiction is detected: it's structural, not keyword matching."

---

### [2:45-3:10] — Show Neo4j Browser (optional, if interviewer is technical)

*Switch to Neo4j Browser. Run:*
```cypher
MATCH (c:Conflict {tenant: "marketing"})
RETURN c.description, c.status, c.severity
```

Say:
> "Two open Conflict nodes. Each linked to the contradicting edges via HAS_CONFLICT relationships.
> They're traversable, auditable, and surfaced in every query that touches related entities.
> Nothing is silent, nothing is auto-resolved. An analyst has to explicitly close a conflict."

---

### [3:10-3:30] — PageRank

*In terminal, scroll up to the `python -m scripts.pagerank_compute --tenant marketing` output from prep — or re-run it now.*

Say:
> "Before any user asks a question, PageRank already knows the most important entities
> in the knowledge base. The Brand Guideline scores highest — it's the most cross-referenced
> concept across all four documents. Statement of Work comes in second. That ranking feeds
> directly into retrieval scoring — more central entities surface higher in answers.
> It runs natively via Neo4j GDS, tenant-isolated, and persists on each entity node."

---

### [3:30-4:00] — Architecture in 30 seconds

*In terminal, run live:*
```powershell
python -m scripts.pagerank_compute --tenant marketing
```

*While output prints, say:*
> "Six-stage retrieval pipeline — GDS PageRank runs natively in Neo4j,
> scores persist on Entity nodes, top entity back in under 2 seconds."

*Switch to Tab 4: `data/wpp_demo/graphify-out/graph.html`.*

Say:
> "Leiden community detection partitions the graph into 9 clusters.
> The retrieval pipeline uses both scores — PageRank for authority,
> communities for context expansion.
>
> Under 5 seconds end-to-end. RAGAS faithfulness of 0.95.
> 380 passing tests across three isolated tenants.
> The domain config is a YAML file — adding a tenant means writing a new ontology YAML,
> not touching the pipeline code."

---

### [4:00-4:15] — Close

Say:
> "That's the platform. It's live, it's on your domain, and I can walk through
> any part of the architecture or the Cypher queries underneath."

Then ask one of the questions from `docs/wpp-jd-mapping.md`.

---

## Key questions to ask after the demo

1. "How does WPP Open handle tenant isolation today — property-level filtering or separate graph per client?"
2. "Is cross-document contradiction detection a solved problem on the platform, or still an open area?"
3. "What's your latency budget for graph-augmented retrieval at scale?"

---

## Cypher queries to have ready (show if they ask about Neo4j expertise)

```cypher
-- Most central entities by PageRank
MATCH (e:Entity {tenant: "marketing"})
WHERE e.pagerank IS NOT NULL
RETURN e.name, e.type, e.pagerank
ORDER BY e.pagerank DESC LIMIT 10

-- Open contradiction nodes
MATCH (c:Conflict {tenant: "marketing", status: "open"})
RETURN c.description, c.severity, c.resolution

-- Authority chain traversal
MATCH path = (sow:Entity {id: "sow-nova-2024-q3"})-[:GOVERNS|SUPERSEDES*1..3]->(doc:Entity)
RETURN [n IN nodes(path) | n.name] AS chain

-- PROHIBITS vs PERMITS on same target (contradiction pattern)
MATCH (a:Entity)-[:PROHIBITS]->(target:Entity)<-[:PERMITS]-(b:Entity)
WHERE a.tenant = "marketing"
RETURN a.name AS prohibitor, b.name AS permitter, target.name AS contested_entity
```

---

## If they ask about the IATF automotive demo

The same platform runs on a 30-document IATF 16949 corpus:
- **10/10 audit scenarios passed**
- **RAGAS faithfulness 0.95**
- **5 cross-document contradictions detected** (supplier reevaluation frequency, audit requirements, procedure revision references)

Say: *"The same pipeline, different domain YAML. That's the point — the graph logic is domain-agnostic."*
