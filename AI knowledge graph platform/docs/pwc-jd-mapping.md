# PwC JD Mapping — Graph RAG / Knowledge Engineering Role

> **How to use this in an interview:**
> When a question maps to this table, open the file immediately.
> "I can show you the code for that right now" beats any verbal answer.

The **Gap** column is honest. No gap means you can demonstrate it live.
A gap means you know the limit and can explain the next step.

---

## Must-Have Requirements

### Neo4j + Cypher expertise in production environments

| JD Requirement | Project Evidence | File / Endpoint | Demo Step | Gap |
|---|---|---|---|---|
| Strong Neo4j expertise | 39 KG modules, 572-line client | `graphrag/graph/neo4j_client.py` | — | None |
| Production Cypher | Vector ANN, BM25, UNWIND×22, EXISTS{}×12, APOC fallback, bitemporal as_of | `neo4j_client.py` | — | None |
| Schema design | Indexes, constraints, vector index (3072d) | `graphrag/graph/schema.cypher` | — | None |
| Multi-hop traversal | `Chunk→Entity→RELATES_TO*→Entity→Chunk` depth 2 | `hybrid_retriever.py → _multihop_expand()` | Demo step 4 | None |
| 6 production Cypher patterns | Multi-hop, bitemporal, transitive supersession, contradiction scan, community ANN, entity audit | `docs/cypher-patterns.md` | — | None |
| Live Neo4j demo | Persisted entities, edges, inferred SUPERSEDES edge | `py -3.11 scripts/demo_regulatory.py --live` | Demo step 4–6 | None |

---

### Deep ontology / taxonomy modeling experience

| JD Requirement | Project Evidence | File / Endpoint | Demo Step | Gap |
|---|---|---|---|---|
| Ontology modeling | Versioned type taxonomy, SUBCLASS_OF hierarchy, LCA for merge decisions | `graphrag/graph/type_taxonomy.py` | Demo step 1–2 | None |
| Domain/range constraints | Every relation write validated against domain/range rules | `graphrag/graph/ontology_registry.py` | Demo step 2 | None |
| Schema migration | Deprecated relation names auto-migrated on ingestion | `ontology_registry.py → migrate_relation()` | — | None |
| Config-driven domains | Aerospace regulatory ontology in YAML; swap file → new domain, no Python changes | `config/ontologies/aerospace_regulatory.yml` | Demo step 1 | None |
| OWL-RL reasoning | Subclass, symmetric, inverse properties; consistency check | `graphrag/graph/owl_reasoner.py` | — | Owlrl not wired to live ingestion; runs over RDF export |
| SPARQL 1.1 | In-process over Turtle export; pre-built queries | `graphrag/graph/sparql_bridge.py`, `POST /kg/sparql` | — | None |
| RDF/Turtle export | owl:NamedIndividual, owl:ObjectProperty, rdfs:subClassOf, reified confidence | `scripts/export_rdf.py` | — | None |

---

### Python engineering skills

| JD Requirement | Project Evidence | File / Endpoint | Demo Step | Gap |
|---|---|---|---|---|
| Production Python | 22,650 LOC, fully async, retry/backoff, structured logging | `graphrag/` package | — | None |
| Test suite | 325 passing unit tests (21 new guardrail tests) | `tests/unit/` → `make test` | — | Integration test coverage is lighter than unit |
| CI/CD | GitHub Actions: pytest matrix + ruff lint | `.github/workflows/` | — | No deploy step in CI (Fly.io is manual) |
| Containerised | Docker multi-stage build; one-command dev stack | `Dockerfile`, `compose.dev.yaml` | — | None |
| Health probes | GET /ready + GET /live on all workers | `graphrag/workers/health_server.py` | — | None |
| Documentation | 6 ADRs, 73 lessons, CONTRIBUTING.md, runbook.md, roadmap.md | `docs/adr/`, `docs/` | — | None |

---

### KG integration with LLMs, RAG, or vector search

| JD Requirement | Project Evidence | File / Endpoint | Demo Step | Gap |
|---|---|---|---|---|
| Vector search | Gemini 3072d embeddings, cosine ANN | `graphrag/ingestion/embedder.py` | — | None |
| Hybrid retrieval | BM25 + Vector fused via RRF (k=60) | `hybrid_retriever.py → _rrf_merge()` | — | None |
| Reranking | ms-marco-MiniLM-L-6-v2 cross-encoder | `graphrag/retrieval/cross_encoder_reranker.py` | — | None |
| GNN scoring | GAT message-passing over entity subgraph | `graphrag/retrieval/gnn_scorer.py` | — | GNN not pre-trained on domain data; uses random init |
| Agentic retrieval | IRCoT: 8B routing + 70B synthesis, max 2 steps | `graphrag/retrieval/agentic_retriever.py` | — | None |
| Agent tool safety | Allowlist, scopes, arg validation, timeout, dry-run, audit | `graphrag/agents/tool_policy.py` | — | None |
| RAGAS evaluation | Faithfulness 0.840, precision 0.907, recall 0.867 | `graphrag/evaluation/ragas_evaluator.py` | — | Evaluation is sampled 20%, not every query |
| Golden eval set | 40 questions, citation recall, pass/fail thresholds | `evals/golden_set.json`, `scripts/run_golden_eval.py` | — | Golden set requires live API; not in CI yet |

---

### Formal semantics ↔ pragmatic balance

| JD Requirement | Project Evidence | File / Endpoint | Demo Step | Gap |
|---|---|---|---|---|
| Chose property graph over RDF | ADR-0001: 5 alternatives evaluated | `docs/adr/0001-property-graph-over-triple-store.md` | — | None |
| Inference design | ADR-0002: forward-chaining over backward-chaining | `docs/adr/0002-forward-chaining-over-backward-chaining.md` | Demo step 4 | None |
| Confidence model | ADR-0003: Bayesian 1-(1-c₁)(1-c₂) over last-write-wins | `docs/adr/0003-bayesian-confidence-accumulation.md` | — | None |
| Provider choice | ADR-0004: Groq for generation, Gemini for embeddings | `docs/adr/0004-groq-over-gemini-for-text-generation.md` | — | None |
| Architecture | ADR-0005: Redis result store; ADR-0006: dual-LLM design | `docs/adr/0005-*.md`, `docs/adr/0006-*.md` | — | None |

---

### Lead technically while remaining hands-on

| JD Requirement | Project Evidence | File / Endpoint | Demo Step | Gap |
|---|---|---|---|---|
| Team documentation | CONTRIBUTING.md with ADR process, PR checklist, coding standards | `CONTRIBUTING.md` | — | No team has actually used it; it's a solo project |
| Runbook | Startup order, 9 health checks, 8 failure patterns, backup/restore | `docs/runbook.md` | — | None |
| Scaling roadmap | Scale limits, decision thresholds, roadmap with completed items | `docs/roadmap.md` | — | None |
| Self-directed learning | 73 documented lessons from 6 months of development | `tasks/lessons.md` | — | None |

---

## Bonus / Highly Valuable Areas

### RDF / OWL

| JD Requirement | Project Evidence | File / Endpoint | Gap |
|---|---|---|---|
| OWL-RL reasoning | owlrl materialises subClassOf, symmetric, inverse; consistency check | `graphrag/graph/owl_reasoner.py` | Runs over exported Turtle, not live graph |
| SPARQL | SPARQL 1.1 SELECT in-process; pre-built queries for entity relations, hierarchy | `graphrag/graph/sparql_bridge.py` | — |
| RDF export | Turtle with owl:NamedIndividual, reified confidence annotations | `scripts/export_rdf.py` | — |

---

### Inference engines

| JD Requirement | Project Evidence | File / Endpoint | Demo Step | Gap |
|---|---|---|---|---|
| Forward-chaining | Transitivity, symmetry, inverse, composition; fixpoint; confidence decay 0.95/hop | `graphrag/inference/forward_chaining.py` | Demo step 4 | None |
| Derived edge provenance | `source_type=inferred`, confidence stamped | `neo4j_client.py → write_inferred_edge()` | Demo step 4 | None |
| TransE link prediction | entity embeddings, predict_tail(), predict_relation() | `graphrag/ml/link_predictor.py`, `POST /kg/predict-links` | — | Model not trained on real domain data |

---

### Entity resolution

| JD Requirement | Project Evidence | File / Endpoint | Gap |
|---|---|---|---|
| 4-stage pipeline | Exact → fuzzy (Levenshtein ≥85) → embedding (cosine ≥0.92) → queue | `graphrag/graph/alias_registry.py` | None |
| Cross-worker sharing | Redis-backed alias table (24h TTL); warm-load skips full Neo4j scan | `alias_registry.py → load_alias_registry()` | None |
| External grounding | Wikidata QID linking (opt-in via WIKIDATA_LINKING=1) | `graphrag/graph/entity_linker.py` | Rate-limited; not proven on real production traffic |

---

### Legal / regulatory data

| JD Requirement | Project Evidence | File / Endpoint | Demo Step | Gap |
|---|---|---|---|---|
| Regulatory ontology | Aerospace: 28 type pairs, 12 relation rules, 4 inference rules | `config/ontologies/aerospace_regulatory.yml` | Demo step 1 | Aerospace only; banking/healthcare not yet built |
| Authority hierarchy | 4-level: Regulatory → Manufacturer → Internal → Informal | `graphrag/graph/document_authority.py` | — | None |
| SUPERSEDES chains | Forward-chaining derives transitive supersession automatically | Demo step 4 | — | None |
| GDPR Art. 17 | Forget-entity: cascade delete + audit log | `POST /kg/gdpr/forget-entity` | Dashboard GDPR tab | None |
| PII guard | Redacts phone, email, NI numbers at extraction time | `graphrag/ingestion/pii_guard.py` | — | None |
| Contradiction detection | 5 conflict types; surfaced per tenant | `graphrag/graph/contradiction_detector.py` | Demo step 5 | None |

---

## Operational / Production Evidence

| Claim | How to verify |
|---|---|
| "Not a tutorial project" | `make test` → 325 passing; `make smoke-test` → exits 0 |
| "Observable" | Admin dashboard `/admin/` — 7 health metrics, Brier score, alerts |
| "Deployable" | `docker compose -f compose.dev.yaml up` → full stack in one command |
| "Controlled agent" | `ToolPolicy.from_defaults()` — allowlist, scopes, audit, dry-run |
| "Scalable" | Redis alias registry, incremental community, RabbitMQ parallel workers |
| "Regulated-client ready" | GDPR tab, multi-tenant isolation, contradiction detection, audit trail |

---

## Quick-Reference Numbers

| Metric | Value | How to verify |
|---|---|---|
| Faithfulness | 0.840 | `GET /evaluation/summary` |
| Context precision | 0.907 | same |
| Context recall | 0.867 | same |
| Hybrid p95 | 2.2s | `GET /kpis/summary` |
| Agentic p95 | 3.4s | same |
| Agentic trigger rate | ~9% | same |
| Entities (seed corpus) | 20 (12-doc aerospace seed; pipeline targets ~2k at scale) | Neo4j Browser or `/kg/health/alerts` |
| Relations (seed corpus) | 12 (pipeline targets ~7k at scale) | same |
| Contradiction detection | wired & verified end-to-end | Admin dashboard → Conflicts tab |
| Alias coverage | pipeline wired; target > 90% at scale | same |
| Brier score (calibration pipeline) | target < 0.20 on production corpus | Admin dashboard → Calibration tab |
| Passing tests | 325 | `py -3.11 -m pytest tests/unit -q` |
| KG modules | 39 | `ls graphrag/graph/*.py \| wc -l` |
| ADRs | 6 | `ls docs/adr/*.md` |
| Lines of code | 22,650 | — |
