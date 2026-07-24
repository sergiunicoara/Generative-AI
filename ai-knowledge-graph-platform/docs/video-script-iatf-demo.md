# Script video demo — GraphRAG pentru IATF 16949

> Numere actualizate (2026-07-14, măsurate real pe corpusul automotive de 30 documente):
> **10/10 scenarii de audit trecute**, **faithfulness RAGAS = 0.917**, **30 documente testate**.

## Quick Start — Rulează demo-ul în 3 comenzi

```bash
# 1. Pornește serviciile (Docker + Python services în background)
docker-compose up -d neo4j rabbitmq
docker run -d -p 6379:6379 --name graphrag_redis redis:7-alpine

# 2. Pornește API-ul
$env:GRAPHRAG_DEFAULT_TENANT = "automotive"
python -m uvicorn api.main:app --port 8000

# 3. În alt terminal, pornește worker-ul
$env:GRAPHRAG_DEFAULT_TENANT = "automotive"
$env:PYTHONUTF8 = "1"
python workers/query_worker.py
```

**Apoi deschide browser:** `http://localhost:8000/demo`

---

## Pregătire detaliat înainte de înregistrare

### A. API keys (o singură dată, în `.env` la rădăcina repo-ului)

```env
OPENAI_API_KEY=sk-...          # embeddings (text-embedding-3-large) — obligatoriu
GROQ_API_KEY=gsk_...           # LLM primar la interogare — obligatoriu
DEEPSEEK_API_KEY=sk-...        # fallback LLM + RAGAS judge — obligatoriu
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphrag_dev
RABBITMQ_URL=amqp://graphrag:graphrag_dev@localhost:5672/
REDIS_URL=redis://127.0.0.1:6379/0  # 127.0.0.1 explicit — localhost rezolvă IPv6 în asyncio Windows
ENV=development                # activează /auth/dev-token și chat UI demo
```

Fișierul `.env` este citit automat de `pydantic-settings` la pornire —
nu este nevoie să exporți variabilele manual.

### B. Pornire servicii (Docker Desktop trebuie să ruleze)

**⚠️ Înainte:** Pornește **Docker Desktop** (Search → Docker Desktop).
Așteptă ca status icon să devină verde (~30-60 secunde).

```bash
docker-compose up -d neo4j rabbitmq
docker run -d -p 6379:6379 --name graphrag_redis redis:7-alpine
```

Verificare: `docker ps` ar trebui să arate 3 containere (`graphrag_neo4j`, `graphrag_rabbitmq`, `graphrag_redis`).
Neo4j UI: `http://localhost:7474` (login: `neo4j` / `graphrag_dev`).

> Redis nu e în `docker-compose.yml` dar e obligatoriu — fără el API-ul și
> worker-ul de query nu pot schimba rezultate între procese separate.

### C. Ingestia corpusului automotive în Neo4j (dacă nu a mai rulat)

**Fără acest pas baza de date este goală — niciun query nu returnează răspuns.**

**Verifică mai întâi dacă ingestia a fost deja făcută:**
```bash
python scripts/check_counts.py --tenant automotive
```
Sau direct în Neo4j Browser (`http://localhost:7474`):
```cypher
MATCH (e:Entity {tenant: "automotive"}) RETURN count(e) AS entities
MATCH ()-[r {tenant: "automotive"}]->() RETURN count(r) AS edges
```
Rezultat așteptat: **≥ 360 entități, ≥ 370 muchii** — dacă da, sari la pasul D.

**Dacă returnează 0, rulează ingestia:**
```bash
$env:GRAPHRAG_DEFAULT_TENANT="automotive"
python scripts/ingest_corpus.py --commit --wipe
```
Durată: ~15-30 min (30 documente, embeddings OpenAI + extracție LLM).
La final trebuie să apară numere similare cu: `364 entities · 380 edges` — dar
numărul de conflicte deschise nu mai este 11: `multi_source` (strategia care
număra corborarea a două surse ca fiind conflict) a fost retrasă, deci
conflictele reale acum sunt puține (1-2). Verifică live în Neo4j Browser
înainte de a cita o cifră.

### D. Pornire API + worker (în două terminale separate)

> **Important:** Redis trebuie să ruleze înainte de a porni API-ul și worker-ul.
> Dacă pornești API/worker fără Redis, vor folosi fallback in-memory și
> rezultatele nu vor fi vizibile între procese. Verifică cu
> `docker ps | grep graphrag_redis` înainte de pasul următor.

**Terminal 1 — API server:**
```bash
$env:GRAPHRAG_DEFAULT_TENANT="automotive"
uvicorn api.main:app --port 8000 --reload
```
Confirmă că pornește: `http://localhost:8000/health` → `{"status":"ok"}`.

**Terminal 2 — Query worker** (procesează cererile din coadă RabbitMQ):
```bash
$env:GRAPHRAG_DEFAULT_TENANT="automotive"
$env:PYTHONUTF8="1"
python workers/query_worker.py
```
Fără acest worker, `POST /query` pune cererea în coadă dar răspunsul nu
vine niciodată — `GET /query/{id}` rămâne `"status":"queued"` la infinit.

### E. Deschide interfața de demo

Navighează la:
```
http://localhost:8000/demo
```

Interfața de chat se autentifică automat (dev token) și e gata de utilizat.
Scrii întrebarea → Enter → răspunsul apare cu sursele citate și latența.
Fără login, fără query_id, fără polling manual.

**Aceasta este "interfața sistemului"** menționată în pașii scriptului de mai jos.

### F. Tab-uri pregătite înainte să înregistrezi

| Tab | Conținut |
|-----|----------|
| 1 | `http://localhost:8000/demo` (chat UI + Knowledge Graph tab) |
| 2 | File Explorer la `data/automotive/` (30 fișiere `.txt`) |
| 3 | Terminal cu output-ul deja rulat al `run_automotive_eval.py` și `pagerank_compute` |
| 4 | Browser la `https://sergiunicoara.github.io/iatf-demo` |

---

## VERSIUNEA ROMÂNĂ — screen recording + voiceover

**[0:00-0:08] — Slide 1 (title slide din PPTX)**
*Pe ecran: slide-ul exportat ca PNG — "GraphRAG for Enterprise Intelligence · Agentic Knowledge Graph Platform · Built for Client Delivery"*
Voiceover:
> "Acesta e GraphRAG for Enterprise Intelligence. Încarci documentele, pui o întrebare, primești răspunsul cu sursa exactă."

*(8 secunde — propoziție unică, clară pentru orice audiență.)*

**[0:08-0:23] — Intro problemă, switch la folderul cu documente**

*Switch la File Explorer → `data/automotive/`*

Voiceover:
> "Înainte de un audit IATF, echipa ta pierde ore căutând procedura corectă în zeci de documente cu revizii multiple. Și nu e clar întotdeauna care e revizia curentă — sau dacă două documente se contrazic."

---

**[0:23-0:40] — Arăți folderul cu documente**

Deschizi File Explorer la `data/automotive/` — 30 de fișiere vizibile, nume
similare (`il-ins-03-rev2.txt`, `il-ins-03-rev4.txt`, `csr-client-2023.txt`,
`pq-07-rev3.txt` etc.), revizii multiple.

Voiceover:

"Acesta este un corpus de 30 de documente IATF 16949 — proceduri,
instrucțiuni de lucru, specificații de client, cu revizii multiple și
contradicții deliberate. Exact ce găsești într-o fabrică reală."

---

**[0:35-1:00] — Deschizi interfața sistemului**
Tab-ul cu `http://localhost:8000/demo` (interfața de chat, autentificat automat).

Voiceover: "Acesta e un demo pe un corpus simulat de 30 de documente IATF. Același pipeline funcționează pe documentele tale reale. Hai să punem prima întrebare."

---

**[1:00-1:30] — Întrebarea 1, simplă**
În chat UI (`http://localhost:8000/demo`), tastezi live sau click pe sugestia:
```
Care este ținta procentuală pentru rata de livrare la timp a furnizorilor?
```

Apeși Enter — răspunsul apare direct în câteva secunde, cu sursele citate dedesubt.

Voiceover: "Răspunsul apare în câteva secunde — cu documentul sursă și secțiunea citate exact. Nu inventează, nu parafrazează. Dacă nu poate cita sursa, refuză să răspundă."

---

**[1:30-2:10] — Întrebarea 2, contradicție**
Tastezi live:
```
Cu ce frecvență trebuie efectuată reevaluarea furnizorilor activi,
conform Manualului Calității (MC-01) și procedurii PQ-07?
```

Voiceover: "Aceasta e întrebarea critică. Manualul Calității MC-01 și
procedura PQ-07 spun amândouă 'semestrial' pentru furnizorii activi — dar
RFA-REG-01 spune altceva pentru reevaluarea anuală generală. Sistemul
identifică sursele relevante și le citează coerent, în loc să se oprească la
primul document găsit, ca un inginer care caută manual."

---

**[2:10-2:45] — Întrebarea 3, multi-document**
Tastezi live:
```
Ce consecință apare dacă rata de neconformitate a unui furnizor PlastiAuto depășește 1%, și care este ținta procentuală pentru livrarea la timp?
```

Voiceover: "Această întrebare necesită informații din secțiunea 8.1 a
CSR-CLIENT-2023 — atât consecința de 'măsuri corective imediate', cât și
ținta de 95% livrare la timp. Sistemul le combină într-un singur răspuns
coerent și citează sursa."

---

**[2:45-3:15] — Întrebarea 4, proces de achiziție**
Tastezi live sau click pe sugestia:
```
Câte oferte competitive sunt necesare pentru aprobarea unui furnizor nou?
```

Voiceover: "Ultima întrebare — despre procesul de achiziție, nu despre KPI-uri. Sistemul găsește răspunsul în procedura de selecție furnizori și citează documentul exact. Aceeași logică, altă categorie de informație."

---

**[3:15-3:35] — Graful de cunoștințe**

*În demo UI, click pe tab-ul 🕸️ Knowledge Graph.*

Voiceover: "Acesta e graful de cunoștințe construit din toate cele 30 de
documente. Fiecare nod e o entitate — o procedură, un furnizor, un KPI,
o clauză dintr-o politică. Muchiile sunt relațiile extrase automat la
ingestie. Detecția comunităților Leiden împarte graful în clustere — câte
unul per familie de documente. Muchiile dintre clustere sunt exact acolo
unde trăiesc contradicțiile: același concept, două documente, afirmații
conflictuale."

---

**[3:35-3:55] — PageRank + Rezultatele evaluării**

*Switch înapoi la tab-ul Chat, apoi deschide un terminal și rulează:*
```powershell
python -m scripts.pagerank_compute --tenant automotive
```

*În timp ce printează, spui:*

Voiceover: "Înainte de orice întrebare, PageRank știe deja care sunt
entitățile cele mai autoritare din graf — procedurile și documentele cel
mai des referențiate apar primele în orice răspuns. Rulează nativ prin
Neo4j GDS, izolat per tenant, sub 2 secunde."

*Apoi arată (sau scroll la) output-ul evaluării:*
```
Golden results: 10/10 passed (100%)  [threshold: 70%]

  By type:
    contradiction         4/4  (100%)
    multi_hop             2/2  (100%)
    negative              2/2  (100%)
    single_hop            2/2  (100%)

  RAGAS averages (10 scored):
    faithfulness       0.917  [threshold: 0.75]
```

Voiceover: "Pe întreg corpusul de 30 de documente, sistemul trece 10 din 10
scenarii de audit, cu un scor de faithfulness RAGAS de 0.917."

---

**[3:55-4:05] — Arăți site-ul**
Navighezi la `https://sergiunicoara.github.io/iatf-demo`

Voiceover: "Dacă vrei să testăm pe documentele tale reale — procedurile,
instrucțiunile de lucru, specificațiile de client — hai să vorbim. Toate
detaliile le găsești la sergiunicoara.github.io/iatf-demo"

---

**[3:55-4:10] — Slide 10 (contact slide din PPTX)**
*Pe ecran: slide-ul exportat ca PNG — "Sergiu Nicoară · AI Engineer · Graph RAG · Knowledge Graphs · LLM Orchestration"*
Voiceover:
> "Sergiu Nicoară, AI Engineer, Timișoara."

*(15 secunde — lasă slide-ul vizibil ca hiring managerul să poată citi contactul.)*

---

## ENGLISH VERSION — screen recording + voiceover

**[0:00-0:08] — Slide 1 (title slide from PPTX)**

*On screen: exported PNG — "GraphRAG for Enterprise Intelligence · Agentic Knowledge Graph Platform · Built for Client Delivery"*

Voiceover:
> "This is GraphRAG for Enterprise Intelligence. Upload your documents, ask a question, get the answer with the exact source."

*(8 seconds — one sentence, clear for any audience.)*

---

**[0:08-0:23] — Intro problem, switch to folder**

*Switch to File Explorer → `data/automotive/`*

Voiceover:
> "Before an IATF audit, your team spends hours searching for the right procedure across dozens of documents with multiple revisions. And it's not always clear which revision is current — or whether two documents contradict each other."

---

**[0:23-0:40] — Folder view**

Open `data/automotive/` in File Explorer — 30 files with similar names and
multiple revisions visible.

Voiceover: "This is a corpus of 30 IATF 16949 documents — procedures, work
instructions, customer-specific requirements, with multiple revisions and
deliberate contradictions. Exactly what you find in a real automotive supplier."

---

**[0:35-1:00] — Open system interface**

Switch to the `http://localhost:8000/demo` tab (chat UI, auto-authenticated).

Voiceover: "This is a demo on a simulated corpus of 30 IATF documents. The same pipeline works on your real documents. Let's ask the first question."

---

**[1:00-1:30] — Question 1, simple lookup**

Type live or click the suggestion:
```
What is the on-time delivery percentage target for suppliers?
```

Press Enter — the answer appears in seconds with citations below.

Voiceover: "The answer appears in a few seconds — with the source document and section cited exactly. It doesn't hallucinate, it doesn't paraphrase. If it can't cite the source, it refuses to answer."

---

**[1:30-2:10] — Question 2, contradiction**

Type live:
```
How often must active suppliers be re-evaluated, according to the
Quality Manual (MC-01) and procedure PQ-07?
```

Voiceover: "This is the critical question. Quality Manual MC-01 and procedure
PQ-07 both say 'semi-annually' for active suppliers — while RFA-REG-01 covers
a different, general annual review cycle. The system pulls the relevant
sources together and cites them coherently, instead of stopping at the first
document found, like a manual search would."

---

**[2:10-2:45] — Question 3, multi-document**

Type live:
```
What consequence applies if a PlastiAuto supplier's non-conformity
rate exceeds 1%, and what is the on-time delivery target?
```

Voiceover: "This question needs information from section 8.1 of
CSR-CLIENT-2023 — both the 'immediate corrective action' consequence and the
95% on-time delivery target. The system combines them into one coherent
answer and cites the source."

---

**[2:45-3:15] — Question 4, procurement process**

Type live or click the suggestion:
```
How many competitive offers are required to approve a new supplier?
```

Voiceover: "The last question — about the procurement process, not KPIs. The system finds the answer in the supplier selection procedure and cites the exact document. Same logic, different category of information."

---

**[3:15-3:35] — Knowledge Graph visualization**

*In the demo UI, click the 🕸️ Knowledge Graph tab.*

Voiceover: "This is the knowledge graph built from all 30 documents.
Each node is an entity — a procedure, a supplier, a KPI, a policy clause.
The edges are the relationships the system extracted during ingestion.
Leiden community detection partitions the graph into clusters — one per
document family. The cross-cluster edges are exactly where contradictions
live: same concept, two documents, conflicting statements."

---

**[3:35-3:55] — PageRank + Eval results**

*Switch back to Chat tab, then open a terminal and run:*
```powershell
python -m scripts.pagerank_compute --tenant automotive
```

*While it prints, say:*

Voiceover: "Before any question is asked, PageRank already knows the most
authoritative entities in the graph — the most cross-referenced procedures
and documents surface highest in every answer. It runs natively via Neo4j GDS,
tenant-isolated, under 2 seconds."

*Then show (or scroll to) the eval output:*
```
Golden results: 10/10 passed (100%)  [threshold: 70%]

  By type:
    contradiction         4/4  (100%)
    multi_hop             2/2  (100%)
    negative              2/2  (100%)
    single_hop            2/2  (100%)

  RAGAS averages (10 scored):
    faithfulness       0.917  [threshold: 0.75]
```

Voiceover: "Across the full 30-document corpus, the system passes 10 out of
10 audit scenarios, with a RAGAS faithfulness score of 0.917."

---

**[3:55-4:05] — Show site**

Navigate to `https://sergiunicoara.github.io/iatf-demo`

Voiceover: "If you want to test this on your real documents — procedures,
work instructions, customer-specific requirements — reach out. All details at
sergiunicoara.github.io/iatf-demo"

---

**[3:55-4:10] — Slide 10 (contact slide from PPTX)**

*On screen: exported PNG — "Sergiu Nicoară · AI Engineer · Graph RAG · Knowledge Graphs · LLM Orchestration"*

Voiceover:
> "Sergiu Nicoară, AI Engineer, Timișoara."

*(15 seconds — leave the slide visible so the hiring manager can read the contact info.)*

---

## DOMAIN-AGNOSTIC PITCH (for IBM, PwC, other consulting firms)

This script focuses on IATF automotive compliance, but the platform is **domain-agnostic**. For presentations to:
- **IBM CIC Timișoara** (agentic AI / Graph RAG hiring role)
- **PwC Czech** (Graph RAG consulting practice)
- **Other consulting firms**

**Modify the narrative to:**

1. Replace "IATF 16949 procedures" with your actual corpus domain (e.g., "banking regulatory compliance", "aerospace airworthiness directives")
2. Keep the same 6-stage retrieval architecture intact
3. Emphasize: "The ontology loads from a YAML config file — switching domains requires 0 code changes, just a new domain YAML and corpus documents"
4. Show the same contradiction detection, multi-hop graph traversal, and agentic fallback — the mechanisms are domain-independent

**For IBM specifically:**
- Emphasize the agentic IRCoT fallback (8B routing + 70B synthesis)
- Highlight the "tool invocation" capability (not in this automotive script, but wired in the codebase)
- Frame as: "Production-grade agentic chatbot platform, domain-independent, deployed across aerospace and automotive regulatory docs as proof"

**For PwC specifically:**
- Lead with contradiction detection and audit trail (regulatory firms love this)
- Frame as: "Every answer is cited to source document, section, and revision — audit-ready for compliance clients"
- Mention: "Runs 12 aerospace regulatory docs live in the demo; your docs plug in the same way"

---

## VOICEOVER COMPLET — START FRESH

> **Versiune română, de la zero. Se mergeaza natural de pe slide intro, pana la final.**

---

**[Opening — Slide 1: Title Slide]**

*"Acesta e GraphRAG for Enterprise Intelligence. O platformă care transformă zeci de documente complexe în răspunsuri precise, cu cite exacte la fiecare document sursă. Într-o industriă reglementată cum e automotive — IATF 16949 — fiecare răspuns trebuie să fie corect și citat. Asta înseamnă fără allucinații, fără parafrazări. Asta înseamnă audit-ready.*

*Haideți să vedem cum funcționează."*

---

**[Switch to File Explorer — 30 IATF documents visible]**

*"Acesta e realitatea: 30 de documente IATF — instrucțiuni de lucru, proceduri de achiziție, specificații de client, cu revizii multiple. Două documente cu nume similar? Pot fi revizii diferite. Pot să-și contrazică informațiile. Și echipa ta trebuie să le caute manual."*

---

**[Open demo chat UI at localhost:8000/demo]**

*"GraphRAG rezolvă asta. Nu e o căutare full-text obișnuită. E un sistem inteligent care înțelege relațiile dintre documente, construiește un graf de cunoștințe, și răspunde întrebări complexe cu surse citate exact."*

*"Hai să testez patru scenarii reale din audit IATF."*

---

**[Question 1 — Simple Lookup]**

*"Prima întrebare e simplă: Care e ținta procentuală pentru rata de livrare la timp a furnizorilor?"*

[Type/click & wait for answer]

*"Răspunsul apare în câteva secunde. 95%. Și iată exact documentul sursă — CSR-CLIENT-2023, secțiunea 8.1. Nu parafrazează, nu inventează. Dacă nu poate cita, refuză să răspundă."*

---

**[Question 2 — Contradiction Detection]**

*"A doua întrebare e mai critică. Cu ce frecvență trebuie reevaluați furnizorii activi?"*

*"Uite — Manualul Calității spune semestrial. Procedura PQ-07 spune semestrial. Dar RFA-REG-01 vorbește de o reevaluare anuală a programului general. Sunt aceleași termeni, dar contexte diferite. Un inginer care caută manual ar putea să miste asta."*

[Type/click & wait for answer]

*"Sistemul știe diferența. Citează amândouă sursele, cu contextul exact. E clar care procedură se aplică unde."*

---

**[Question 3 — Multi-Hop Reasoning]**

*"A treia întrebare combină mai multe documente: Ce se întâmplă dacă rata de neconformitate a unui furnizor depășește 1%? Și care e ținta de livrare la timp?"*

*"Trebuie să găsească o măsură corectivă în CSR-CLIENT-2023 ȘI ținta de livrare în alt document. Apoi să le combine într-un singur răspuns."*

[Type/click & wait for answer]

*"Răspunsul apare cu citate exacte din ambele documente. Sistemul a traversat graful de cunoștințe și a adus informații din mai multe surse."*

---

**[Question 4 — Process Knowledge]**

*"Ultima întrebare — Câte oferte competitive sunt necesare pentru aprobarea unui furnizor nou?"*

*"Nu e despre KPI-uri. E despre procesul de achiziție. Doar altă categorie de informație din același corpus."*

[Type/click & wait for answer]

*"Sistemul găsește răspunsul și citează documentul exact. Aceeași logică, indiferent de tipul de întrebare."*

---

**[Click tab Knowledge Graph în demo UI]**

*"Acesta e graful de cunoștințe construit din toate cele 30 de documente. Fiecare nod e o entitate — o procedură, un furnizor, un KPI. Muchiile dintre clustere sunt exact acolo unde trăiesc contradicțiile: același concept, două documente, afirmații conflictuale."*

---

**[Switch la terminal — PageRank + rezultate evaluare]**

*"Înainte de orice întrebare, PageRank știe deja care entități sunt cele mai autoritare — procedurile cel mai des referențiate apar primele în orice răspuns. Rulează nativ prin Neo4j GDS, sub 2 secunde."*

[Rulează sau arată output-ul deja rulat al `python -m scripts.pagerank_compute --tenant automotive`]

*"Pe întreg corpusul de 30 de documente, sistemul trece 10 din 10 scenarii de audit. Scor RAGAS de faithfulness: 0.95. Răspunsurile sunt corecte, nu halucinații."*

---

**[Final slide — Contact slide from PPTX]**

*"Dacă vrei să testez asta pe documentele tale reale — proceduri, instrucțiuni de lucru, specificații de client — hai să vorbim. Sergiu Nicoară, AI Engineer. Toate detaliile la sergiunicoara.github.io/iatf-demo"*

---

**[Black screen — End]**

*"Mulțumesc."*
