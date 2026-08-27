# Knowledge Graph Architecture

This document describes the architectural decisions, data model, and operational
design of the knowledge graph layer. It is intended for engineers and architects
evaluating the system for integration or extension.

---

## 1. Core Data Model

```
(Document)-[:PART_OF]-(Chunk)-[:MENTIONS]->(Entity)
(:Document)-[:LINKS_TO {tenant, provenance, ACL snapshot, observed_at}]->(:Document)
(:Entity)-[:HAS_SYSTEM_REPRESENTATION]->(:SystemRepresentation)
(:Chunk)-[:ASSERTS_IN_CONTEXT]->(:ContextualAssertion)
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
Explicit document links are source-observed only; similarity never creates
`LINKS_TO` edges. System representations preserve source-system context beneath
the tenant-scoped canonical entity.

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
| Document topology | `document_loader.py`, `neo4j_client.py` | Explicit HTML/Markdown/SharePoint links, ACL-aware bounded traversal, late-target reconciliation |
| Contextual identity | `graph_writer.py`, `neo4j_client.py` | Tenant/source-system representations and chunk-backed contextual assertions |

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

Conflict detection is no longer write-side only: `HybridRetriever` looks up open
`Conflict` nodes for entities in the retrieved result set and the answer prompt
is warned when context includes a disputed fact, gated by
`retrieval.conflict_annotation_enabled` (default on).

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

The same export also supplies a SKOS browsing projection: each tenant has a
`skos:ConceptScheme`; entity types and entities are `skos:Concept`s with
`skos:prefLabel`, and type/entity navigation is represented by
`skos:broader`. This leaves the OWL/RDFS semantics intact while making exports
directly usable by vocabulary browsers and SKOS-aware RDF tools.

This allows the ontology to be consumed by Protégé, reasoners (HermiT, Pellet),
and SPARQL endpoints without requiring a full migration to a triple store.

**Confidence and provenance are reified, not just attached.** Every exported
edge with a confidence score or source document is wrapped in an `owl:Axiom`
(`export_rdf.py`) carrying `owl:annotatedSource` / `annotatedProperty` /
`annotatedTarget` plus `:confidence` (`xsd:float`) and `:sourceDoc`
annotations — the standard OWL pattern for making statements *about*
statements, matching the reification already used internally (§7).

**SHACL validation is real and CI-verified, not just present.** `shacl_validator.py`
defines actual `sh:NodeShape` shapes (every entity needs an `rdfs:label` and a
domain type; every `owl:Axiom` needs a complete source/property/target triple;
confidence must be `xsd:float` in `[0,1]`) and runs them via `pyshacl.validate()`.
`tests/unit/test_export_rdf.py::TestExportProducesConformantGraph` asserts the
*real* `export()` pipeline output — not just hand-built test graphs — conforms,
which runs in `pytest tests/unit/` on every push (`.github/workflows/ci.yml`).
A change that breaks the export's shape guarantees fails CI, not just a manual
`--validate` run.

**Community structure gets an independent, standard cross-check.**
`graph_evaluator.py`'s `community_coherence()` is a hand-rolled intra-community
edge-density ratio computed in Cypher. `community_modularity()` computes
standard Newman-Girvan modularity via **NetworkX** on the same subgraph —
a different formula (it accounts for expected edge density under a random
graph with the same degree distribution, not just raw intra/total edges),
so a community that looks coherent by the simple ratio can still score low
modularity if it's dominated by high-degree hub entities.

**SPARQL is real and network-exposed, but bounded — precise framing
matters here.** `POST /kg/sparql` (`api/routes/kg/knowledge.py`) runs real
SPARQL 1.1 SELECT queries (`SPARQLBridge`, `graphrag/graph/sparql_bridge.py`,
wrapping rdflib's built-in engine) against the last Turtle export on disk.
This is a genuine, tested, callable SPARQL capability — not a stub. What it
is *not*: a persistent triple-store service (no GraphDB/Stardog/Virtuoso),
and not live against current graph state — it queries a snapshot file that
only updates when `export_rdf.py` is re-run, so it can be stale relative to
Neo4j. The live, continuously-updated system is Neo4j as a labeled property
graph; RDF/OWL/SHACL/SPARQL is a real, tested interoperability layer
exported from it, not a second production database running in parallel.

---

## 9. LLM Routing — Groq for Generation and Fast Routing, OpenAI for Embeddings

All LLM calls are centralised through `graphrag/core/llm_client.py`. This module
routes text generation (`get_llm()`) to a `FallbackLLM` chain by default:
Groq → DeepSeek. `LLM_INGEST_PROVIDER=deepseek` makes DeepSeek primary with
Groq fallback; `LLM_INGEST_PROVIDER=cerebras` opts into the Cerebras → DeepSeek
→ Groq chain. `LLM_INGEST_PROVIDER=groq` explicitly selects the default route.
The agentic retriever's intermediate SEARCH/ANSWER routing decisions
(`get_fast_llm()`) default to the configured Groq fast model (DeepSeek
fallback), since that path is latency-bound. Embeddings go to OpenAI
`text-embedding-3-large`, with a clean singleton interface used across all
pipeline stages.

```
                ┌─────────────────────────────────────┐
                │          llm_client.py               │
                │                                      │
                │  get_llm()      → FallbackLLM        │
                │                    (Groq primary)    │
                │  get_fast_llm() → FallbackLLM        │
                │                    (Groq fast primary)│
                │  get_embedder() → OpenAIEmbedder     │
                └───────────┬──────────────┬───────────┘
                            │              │
               ┌────────────▼──┐   ┌───────▼──────────────┐
               │ Groq API      │   │ OpenAI API            │
               │ configured    │   │ text-embedding-3-     │
               │ large model   │   │ large (3072d vectors) │
               │ (default)     │   └──────────────────────┘
               └──────┬────────┘
                      │ on failure
               ┌──────▼────────┐
               │ DeepSeek API  │
               │ deepseek-v4-  │
               │ flash         │
               └───────────────┘
```

### Why this split?

| Concern | Groq + DeepSeek (+ Cerebras opt-in) | OpenAI |
|---|---|---|
| Text generation (synthesis + extraction) | Groq `groq_model` (default via `get_llm()`); falls to DeepSeek `deepseek-v4-flash`; Cerebras is an explicit opt-in route | — |
| Routing steps | `groq_fast_model` via Groq (default); DeepSeek fallback | — |
| Embedding | — | `text-embedding-3-large` (3072d), cosine-compatible, same schema as prior Gemini index |
| Cost | Groq is the default generation route; DeepSeek is used on fallback or explicit override | ~$0.13/1M tokens |

### What uses Cerebras / DeepSeek / Groq

- `graphrag/ingestion/extractor.py` — entity + relation extraction from chunks (Groq default)
- `graphrag/retrieval/local_search.py` — answer synthesis from retrieved context (Groq default)
- `graphrag/retrieval/global_search.py` — direct community-summary retrieval and bounded synthesis context; legacy map-reduce is configuration-gated
- `graphrag/retrieval/agentic_retriever.py` — IRCoT routing (Groq fast model) and final synthesis (Groq default via `get_llm()`)
- `graphrag/graph/community_summarizer.py` — LLM community summaries (Groq default)
- `graphrag/evaluation/ragas_evaluator.py` — RAGAS judge LLM (DeepSeek first, Groq fallback, Gemini only as a final compatibility fallback; intentionally independent of the generation and fast-routing tiers)

### What uses OpenAI (embeddings only)

- `graphrag/ingestion/embedder.py` — chunk embedding batches
- `graphrag/retrieval/local_search.py` — query embedding for vector ANN
- `graphrag/agents/ingestion_agent.py` — entity name+description embedding

> **RAGAS evaluator note:** The judge LLM for RAGAS metrics is resolved in priority
> order: DeepSeek → Groq (`langchain-groq`) → Gemini compatibility fallback → None. This ordering is specific to
> the evaluation judge and is separate from `get_llm()`'s generation-primary
> choice.
> Install with `pip install langchain-groq`.

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

**Without Redis**, `ResultStore` no longer silently falls back to its own
in-process memory — it logs an ERROR and drops the write/read, so the API
returns `{"status": "queued"}` visibly rather than masking a cross-process
split-brain. Set `REDIS_URL` in `.env` and ensure Redis is running before
starting workers.

### Governed answer cache

`ResultStore` is transport state keyed by the new `query_id`; it is not an
answer cache. Stateless worker-path queries also pass through
`graphrag/retrieval/query_cache.py` inside `HybridRetriever`. Its canonical
SHA-256 key covers the tenant, normalized question, requested and effective
retrieval modes, output-affecting retrieval configuration, the full
primary/fallback model route, prompt version, ontology version, and the
tenant's durable `KGCorpusState.revision`.

Every retrieval-visible graph mutation marks `KGCorpusState.updating=true`
before its graph writes. Cache reads are disabled while the tenant has active
updates, and the final concurrent completion atomically increments the
revision and clears the flag. Old Redis entries may
remain until TTL expiry, but cannot match the new key. A cache entry is written
only after the corresponding Context Graph decision trace persists. Hits keep
the new request's `query_id` and expose `source_query_id`, `source_trace_id`,
and `cache_key`, so the saved time does not erase provenance. Queries with a
`session_id` deliberately bypass this cache until conversation state can be
included in the canonical key.

### Adaptive retrieval routing

`query_planner.py` remains the deterministic cold-start policy, while
`adaptive_router.py` persists tenant- and query-class-scoped route statistics
in `KGRetrievalRouteStat`. Once at least two modes have enough samples, the
router chooses local, hybrid, or global retrieval using an EWMA quality signal
minus a bounded, log-scaled latency penalty. Deterministic 5% exploration keeps
untried routes measurable. Explicit non-hybrid modes and vector-only ablations
remain authoritative, and any router storage failure falls back to the planner.

### Versioned community summaries

Leiden community IDs are deterministic over tenant, level, algorithm, and
member IDs. Every non-empty generated summary appends a
`CommunitySummarySnapshot` with valid-time and transaction-time boundaries,
the summary embedding, a canonical content hash, exact chunk/document IDs and
versions, and `SUPPORTED_BY`/`DERIVED_FROM` evidence links. Temporal global
retrieval searches these snapshots instead of the mutable current `Community`
projection. Neo4j 2026 performs the tenant predicate inside the vector index;
Neo4j 5.20 uses the bounded over-fetch fallback.

### Source catalog and connector boundary

`KGSource` owns source identity, type, owner, classification, refresh SLA, and
status. Immutable `KGSourceMapping` versions store canonical mapping JSON and
its SHA-256 digest, but reject credential-shaped fields. Connectors implement
the provider-neutral `SourceConnector.records()` protocol and emit
`SourceEnvelope` records; credentials remain in deployment secret stores.
Cataloged documents link to their source through `INGESTED_FROM`.

### Local relational-to-graph ingestion

The repository includes a provider-neutral local reference path in
`graphrag/ingestion/relational.py`. `SQLiteSourceConnector` reads a local
read-only SQLite database, while `RelationalGraphMapping` declares which tables
become tenant-scoped entities and relations. `RelationalGraphIngestor` validates
required identifiers and relationship endpoints before writing anything, then
reuses `GraphWriter` for ontology checks, entity resolution, audit logging and
Neo4j persistence.

Each import is represented by a deterministic relational source `Document` and
`Chunk`. The source ID, mapping version and ontology version are retained as
metadata, so imported facts remain attributable and repeatable. The local
synthetic sustainability example is available through
`scripts/demo_sustainability_relational.py`; it demonstrates supplier,
material and facility data without claiming a live ERP, ESG provider or cloud
deployment.

`PostgreSQLSourceConnector` supports the identical mapping contract through a
read-only local PostgreSQL/TimescaleDB-compatible SQLAlchemy URL. Both source
adapters pass an in-memory SHACL candidate-batch gate before the first Neo4j
write; violations cannot leave a partial relational import behind.

The reproducible PostgreSQL vertical slice is
`scripts/demo_sustainability_e2e.py`. Given an explicitly supplied local
SQLAlchemy/asyncpg URL, `--seed` creates only named synthetic demo tables,
imports them to Neo4j, and asks the MCP-backed controlled question `Which
suppliers lack verified emissions evidence?`. The fixed query template returns
only tenant-scoped suppliers that have no `REPORTED` emissions record linked to
`HAS_EVIDENCE`. The Docker-backed integration test
`tests/e2e/test_relational_postgres_neo4j.py` independently verifies this
PostgreSQL-to-Neo4j-to-MCP path with isolated containers.

This is not yet an RML/R2RML engine, an OBDA federation layer, or a live
GLEIF/Copernicus connector.

### Controlled agent graph facts

The MCP server exposes `query_graph_facts_tool` for a narrow class of direct
graph facts, such as `What does Northwind Components supply?`, `List
suppliers`, and `Which suppliers lack verified emissions evidence?`. It is a
deterministic intent parser, not arbitrary LLM-to-Cypher:
only fixed, read-only Cypher templates are executable; values are parameters;
every template has a tenant predicate and a maximum result limit of 100.
Unsupported questions are returned to normal cited GraphRAG retrieval.

### PROV-O interoperability

The Turtle export now binds the standard `prov:` namespace. Entities and
reified relationship assertions with a source document emit
`prov:wasDerivedFrom` links to tenant-scoped source-artifact resources; relation
assertions also export `prov:generatedAtTime` when extraction time is present.
The existing annotation vocabulary remains available for platform-specific
confidence and temporal fields.

### Correlation and telemetry

FastAPI accepts or creates `X-Correlation-ID`, returns it to the caller, and
propagates it in the RabbitMQ payload, AMQP properties, worker result, cost
events, and `CGAgentRun`. RabbitMQ also injects/extracts W3C trace context so
optional OTLP spans remain one distributed trace across API and worker
processes. Correlation IDs are deliberately excluded from Prometheus labels to
avoid unbounded metric cardinality. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to enable
export; Prometheus remains available at `/metrics` independently.

---

## 10. Scalability Considerations

| Concern | Current design | Scale path |
|---|---|---|
| Write throughput | Sequential per-document; RabbitMQ decouples producers | Parallel workers per tenant |
| Read latency | Vector ANN + BM25 in Neo4j; governed Redis answer cache; Redis result transport | Read replicas; cache TTL and corpus-revision monitoring |
| Community rebuild | Leiden on full entity graph per tenant | Incremental rebuild (changed entities only) via `IncrementalCommunityDetector` |
| Alias resolution | In-memory dict per process | Redis-backed for multi-replica deployments |
| Inference | Post-ingestion forward-chaining; bounded by MAX_RETRIES | Scoped to affected document's entity subgraph via `run_for_document()` |
| KPI metrics | SQLite by default; optional TimescaleDB backend via `TIMESCALE_DB_URL` and `KPI_BACKEND=timescale` | TimescaleDB hypertable with continuous aggregates when volume/SLOs justify it |

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
├── contradiction_detector.py  — 4 conflict types, resolution workflow (multi_source retired 2026-07-24, see A135)
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
├── pagerank.py             — GDS centrality + staleness-triggered recompute (see §12)
├── gnn_scorer.py           — GCN/GAT retrieval re-scoring (see §12)
└── edge_embeddings.py      — TransE triple embeddings, link prediction
```

## 12. Retrieval Scoring — GNN and PageRank, and why they're not combined

Two structural-signal mechanisms exist in retrieval, deliberately kept
separate rather than merged, after investigating whether they should be
(see `tasks/lessons.md` A139 for the full reasoning):

**GNN scoring** (`graphrag/graph/gnn_scorer.py`) — query-scoped, recomputed
fresh on every query, never persisted. Runs 2 layers of GCN or GAT
message-passing over the ~50 entities relevant to *this* query's retrieved
chunks, blending the result with cross-encoder/text score into
`final_score`. Hub-dampening penalizes high-fan-out entities using
graph-level `degree` (not PageRank — degree directly measures the dilution
risk dampening exists to suppress; PageRank measures something else).

**PageRank** (`graphrag/graph/pagerank.py`) — corpus-wide, computed once per
tenant via Neo4j GDS, persisted onto `Entity.pagerank`, recomputed only when
`GraphWriter._maybe_recompute_pagerank()` detects staleness after an
ingestion (growth drift, document re-ingestion, or a decay-conditional time
ceiling — see A139). Consumed only as a narrow low-confidence-retrieval
tiebreak in `local_search.py`, never as a general relevance boost: global
importance anti-correlates with correctness on precise lookups (a specific
document ID is rarely the corpus's most central entity), so it only nudges
rankings when neither text nor GNN scoring produced a confident result.

**Why not wired together**: GNN's own message-passing already partially
captures "nearness to structurally important entities" through ordinary
2-hop aggregation — explicitly injecting PageRank into GNN's attention
weights would risk double-counting that signal, on top of touching tested
propagation math for an uncertain gain. Kept as two independent,
interpretable signals instead.

---

## 13. MCP Server — Exposing Retrieval as Agent Tools

`mcp_server/` exposes versioned, entitlement-filtered Model Context Protocol
capabilities callable by MCP-compatible clients, not just the platform's own
FastAPI/RabbitMQ stack:

| Capability | Wraps | Returns |
|---|---|---|
| `kg.answer.query@1.0.0` | `HybridRetriever.retrieve_and_answer()` — the same hybrid retrieval and cited synthesis used by the API | `QueryResult.model_dump()`: answer, citations, contexts, latency, mode |
| `kg.entity.lookup@1.0.0` | `AliasRegistry.resolve()` + tenant-scoped Neo4j evidence lookup + PageRank | resolved canonical name/type, relations, PageRank importance (nullable — never coerced to 0) |
| `kg.facts.query@1.0.0` | fixed, parameterized fact-query templates | bounded graph facts; never raw Cypher/SPARQL |
| `cg.precedent.find@1.0.0` | outcome-backed Context Graph precedent lookup | policy-compatible decisions with outcome/feedback score components |
| `biz.workorder.create@1.0.0` | typed, idempotent WorkOrder command service | execution, stale-version, or approval-required receipt |

**Transport is deliberate:** local stdio is bound to the scoped
`GRAPHRAG_MCP_TOKEN` supplied by the launcher. `mcp_server.remote` exposes
the same FastMCP server over authenticated Streamable HTTP at `/mcp`; each
Bearer token is verified and bound to that request only, and must carry an
`aud` claim naming this MCP resource — a token minted for the REST API is
rejected (RFC 8707; see `docs/adr/0010-audience-bound-access-tokens.md`). The
signed tenant is the authority; client-supplied tenant values are assertions
that must match. Remote `/metrics` is also authenticated; `/health` and the
RFC 9728 document at `/.well-known/oauth-protected-resource/<path>` are the
public surfaces, the latter because a client with no usable token has to be
able to discover where to get one. See
`docs/adr/0009-agent-platform-trust-boundaries.md` and
`docs/mcp-operations.md` for the deployment contract.

**A design constraint worth naming explicitly**: stdout is the stdio MCP
protocol's JSON-RPC channel. `mcp_server/server.py` configures structlog to
stderr before importing GraphRAG modules, so diagnostics cannot corrupt the
protocol stream. Capability invocation, router choice, evaluation outcome,
cost, and latency events carry correlation IDs; tenant identity remains a
structured field rather than a high-cardinality Prometheus label.

**What it deliberately doesn't expose (yet)**: `get_pagerank_by_entity_names`
isn't a standalone tool — folded into `lookup_entity_tool`'s
`importance_pagerank` field instead, since a bare "importance score" lookup
has no use once entity lookup already surfaces it. `SPARQLBridge` (§8)
is excluded too — it queries a Turtle export that only refreshes when
`scripts/export_rdf.py` is manually re-run, so it can silently drift stale
relative to what the other two tools see live in Neo4j. A future
`sparql_query` tool would need an explicit "as of export timestamp X"
caveat in its response to be honest about that gap.
