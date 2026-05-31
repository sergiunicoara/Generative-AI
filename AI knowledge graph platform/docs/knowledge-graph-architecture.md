# Knowledge Graph Architecture

This document describes the architectural decisions, data model, and operational
design of the knowledge graph layer. It is intended for engineers and architects
evaluating the system for integration or extension.

---

## 1. Core Data Model

```
(Document)-[:PART_OF]-(Chunk)-[:MENTIONS]->(Entity)
                                               │
                              (Entity)-[:RELATES_TO {
                                  relation,          ← UPPER_SNAKE_CASE, ontology-validated
                                  confidence,        ← Bayesian-accumulated [0,1]
                                  weight,
                                  source_doc_ids,    ← list, all contributing documents
                                  source_type,       ← document | inferred | manual
                                  valid_from,        ← valid time start (nullable)
                                  valid_to,          ← valid time end (nullable)
                                  recorded_at,       ← transaction time (immutable)
                                  tenant             ← strict per-tenant isolation
                              }]->(Entity)
                                               │
                              (Entity)-[:MEMBER_OF]->(Community)
                              (Entity)-[:SUBCLASS_OF]->(EntityType)
                              (Entity)-[:ALIAS_OF inverse ALIAS_OF]-(Alias)
```

Every node and edge carries `tenant` for strict multi-tenant isolation.
The composite key `(name, type, tenant)` is the canonical entity identifier.

---

## 2. Graph Layer Responsibilities

The knowledge graph layer is responsible for **six distinct concerns**, each
implemented as a focused module:

| Concern | Module | Responsibility |
|---|---|---|
| Schema enforcement | `ontology_registry.py` | Type constraints, relation domain/range, migration |
| Type hierarchy | `type_taxonomy.py` | SUBCLASS_OF hierarchy, subtype expansion for queries |
| Entity resolution | `alias_registry.py` | 4-stage deduplication before every MERGE |
| Temporal modeling | `bitemporal.py` | Valid time + transaction time; time-travel queries |
| Inference | `inference_engine.py` | Forward-chaining rules; derived edge materialisation |
| Conflict tracking | `contradiction_detector.py` | 5 conflict types; resolution workflow |
| Property validation | `property_schema.py` | Per-type attribute cardinality rules |
| Graph health | `graph_evaluator.py` | 6 semantic metrics; trend snapshots |
| Community structure | `community_builder.py` | Leiden communities; global search summaries |
| Calibration | `confidence_calibration.py` | Brier score; isotonic confidence correction |

---

## 3. Graph Integrity Guards

Every ingestion write triggers a cascade of integrity checks:

```
write_document() → write_chunks() → write_entities() → write_relations()
                                                              │
                                          validate_and_check_cycles()
                                                              │
                                          ┌───────────────────┤
                                          │                   │
                              IngestionValidator    CycleDetector
                              (degree anomalies,    (APOC or pure
                               self-loop removal)    Cypher DFS)
                                          │
                                   QuarantineService
                                   (auto-quarantine
                                    anomalous entities)
                                          │
                                ContradictionDetector
                                (scan new doc scope,
                                 persist Conflict nodes)
                                          │
                                 CommunityManager
                                 (staleness check;
                                  conditional rebuild)
```

The quarantine system flags entities for human review without deleting them —
they are excluded from retrieval but remain in the graph for audit purposes.

---

## 4. Negative Knowledge

The graph explicitly models **asserted non-relationships** via `NEGATIVE_RELATES_TO`
edges with the same provenance model as positive edges:

```cypher
(A:Entity)-[:NEGATIVE_RELATES_TO {relation: "USES", confidence: 0.9, ...}]->(B:Entity)
```

This prevents the closed-world assumption problem: when a domain expert asserts that
"A does NOT use B", that fact should survive future ingestion of documents that only
mention A and B without commenting on their relationship.

When a `RELATES_TO` and a `NEGATIVE_RELATES_TO` edge coexist for the same triple,
the contradiction detector raises a `positive_negative_pair` conflict for resolution.

---

## 5. Document Authority System

Source documents carry an authority level (lower = higher authority):

| Level | Name | Examples |
|---|---|---|
| 1 | REGULATORY | Airworthiness directives, ITAR regulations, FAA rules |
| 2 | MANUFACTURER_SPEC | OEM design specifications, approved data |
| 3 | INTERNAL_PROCEDURE | Company SOPs, work instructions |
| 4 | INFORMAL | Emails, meeting notes, wiki pages |

When Document A `SUPERSEDES` Document B (modelled as a `SUPERSEDES` edge), edges
from B receive a confidence penalty (`superseded_confidence_penalty: 0.5` by default).
The authority system answers: "Which document's version of this fact should we trust?"

This is foundational for regulatory compliance graphs where an Airworthiness Directive
(AD) supersedes a previous AD for the same aircraft component.

---

## 6. Multi-Tenant Architecture

Tenant isolation is enforced at **every layer** of the stack:

- **Graph:** all MATCH/MERGE operations include `tenant: $tenant` in node patterns
- **Entity identity:** `MERGE (e:Entity {name: $name, type: $type, tenant: $tenant})`
- **Alias registry:** one registry instance per tenant in a per-process pool
- **Community detection:** Leiden runs per-tenant; communities carry `tenant`
- **Health metrics:** `GraphHealthSnapshot` nodes carry `tenant`; all 6 metrics
  are scoped by tenant in their Neo4j queries
- **Contradiction detection:** scan always filters by `tenant` to prevent
  cross-tenant edge comparison
- **Session store:** `graphrag:session:<session_id>` keys in Redis are not
  tenant-namespaced (sessions are user-scoped, not tenant-scoped)

---

## 7. Reification — Statements About Statements

For domains requiring meta-assertions (regulatory compliance, legal reasoning),
the graph supports **reification** via `Statement` nodes:

```
(A:Entity)-[:SUBJECT_OF]->(s:Statement {
    relation:       "CEO_OF",
    confidence:     0.95,
    source_doc_ids: [...],
    tenant:         "default"
})-[:OBJECT_OF]->(B:Entity)
```

A `Statement` node can then be the target of further assertions:
- Endorsements: `(expert)-[:ENDORSES]->(s)`
- Contradictions: `(s1:Statement)-[:CONTRADICTS]->(s2:Statement)`
- Meta-properties: `(s)-[:HAS_EVIDENCE]->(doc)`

This avoids the property-limit problem of attaching arbitrary metadata to edges
and enables first-class reasoning about provenance and epistemic status.

**Implementation:** `graphrag/graph/reification.py`

---

## 8. RDF / Interoperability

The graph can be serialised to **Turtle (RDF)** for interoperability with OWL
tooling, SPARQL consumers, and linked-data systems:

```bash
python scripts/export_rdf.py --tenant default --output graph_export.ttl
```

The export maps:
- `Entity` nodes → `owl:NamedIndividual` with `rdf:type` from entity type
- `EntityType` nodes → `owl:Class` with `rdfs:subClassOf` hierarchy
- `RELATES_TO` edges → `owl:ObjectProperty` instances
- `NEGATIVE_RELATES_TO` edges → annotated with `owl:complementOf` semantics
- `SUBCLASS_OF` edges → `rdfs:subClassOf`

This allows the ontology to be consumed by Protégé, reasoners (HermiT, Pellet),
and SPARQL endpoints without requiring a full migration to a triple store.

---

## 9. LLM Routing — Groq for Generation, Gemini for Embeddings

All LLM calls are centralised through `graphrag/core/llm_client.py`. This module
routes text generation to Groq and embeddings to Gemini, with a clean singleton
interface used across all pipeline stages.

```
                ┌─────────────────────────────┐
                │       llm_client.py          │
                │                              │
                │  get_llm()    → GroqLLM      │
                │  get_embedder() → GeminiEmbedder │
                └───────────┬─────────┬────────┘
                            │         │
               ┌────────────▼─┐   ┌───▼────────────────┐
               │ Groq API     │   │ Gemini API          │
               │ llama-3.3-   │   │ gemini-embedding-   │
               │ 70b-versatile│   │ 001 (3072d vectors) │
               └──────────────┘   └────────────────────┘
```

### Why this split?

| Concern | Groq | Gemini |
|---|---|---|
| Text generation | llama-3.3-70b-versatile, free tier, 1500+ RPD | quota-limited, rate-throttled for free keys |
| Embedding | — | `gemini-embedding-001` (3072d), high-dimensional, cosine-compatible |
| Cost | Free tier sufficient for dev/testing | Free tier for embeddings only |

### What uses Groq

- `graphrag/ingestion/extractor.py` — entity + relation extraction from chunks
- `graphrag/retrieval/local_search.py` — answer synthesis from retrieved context
- `graphrag/retrieval/global_search.py` — map-reduce community summarisation
- `graphrag/retrieval/agentic_retriever.py` — IRCoT sub-queries
- `graphrag/graph/community_summarizer.py` — LLM community summaries

### What uses Gemini (embeddings only)

- `graphrag/ingestion/embedder.py` — chunk embedding batches
- `graphrag/retrieval/local_search.py` — query embedding for vector ANN

### Cross-process result store

Query results are written by the worker and read by the API. These are separate
OS processes, so in-process dicts do not work. Both processes connect to Redis
independently through `graphrag/retrieval/result_store.py`:

```
Query Worker                         API Process
─────────────                        ───────────
QueryAgent.run(query_id)
 → answer computed
 → ResultStore.set(query_id, result)
     ↓ Redis SETEX (1h TTL)
                                     GET /query/{query_id}
                                      → ResultStore.get(query_id)
                                          ↑ Redis GET
                                      → 200 {status: "completed", answer: ...}
```

**Without Redis**, the worker writes to its own in-process memory and the API
always returns `{"status": "queued"}`. Set `REDIS_URL` in `.env` and ensure
Redis is running before starting workers.

---

## 10. Scalability Considerations

| Concern | Current design | Scale path |
|---|---|---|
| Write throughput | Sequential per-document; RabbitMQ decouples producers | Parallel workers per tenant |
| Read latency | Vector ANN + BM25 in Neo4j; Redis result cache | Read replicas; query result TTL tuning |
| Community rebuild | Leiden on full entity graph per tenant | Incremental rebuild (changed entities only) via `IncrementalCommunityDetector` |
| Alias resolution | In-memory dict per process | Redis-backed for multi-replica deployments |
| Inference | Post-ingestion forward-chaining; bounded by MAX_RETRIES | Scoped to affected document's entity subgraph via `run_for_document()` |
| KPI metrics | SQLite per-process | TimescaleDB hypertable with continuous aggregates |

---

## 11. Key Files

```
graphrag/graph/
├── neo4j_client.py         — async driver, MERGE helpers, vector/BM25 search
├── ontology_registry.py    — versioned schema, domain/range enforcement, migration
├── type_taxonomy.py        — SUBCLASS_OF hierarchy, transitive expansion
├── alias_registry.py       — 4-stage entity resolution, per-tenant pool
├── bitemporal.py           — valid time + transaction time queries
├── inference_engine.py     — Datalog forward-chaining rules
├── contradiction_detector.py  — 5 conflict types, resolution workflow
├── contradiction_strategies.py — detection method implementations (mixin)
├── negative_knowledge.py   — NEGATIVE_RELATES_TO edges
├── reification.py          — Statement nodes for meta-assertions
├── property_schema.py      — per-type attribute cardinality validation
├── graph_evaluator.py      — 6 semantic health metrics, trend snapshots
├── community_builder.py    — Leiden communities, semantic communities (HDBSCAN)
├── community_manager.py    — staleness scoring, snapshot, rebuild gating
├── incremental_community.py — changed-entity-only community rebuild
├── confidence_calibration.py — Brier score, isotonic correction curves
├── graph_snapshots.py      — before/after snapshot diffing
└── edge_embeddings.py      — TransE triple embeddings, link prediction
```
