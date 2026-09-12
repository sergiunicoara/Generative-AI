# Knowledge Graph Engineer — Interview Refresh

Use this as a quick preparation sheet for the Energy-sector role. The safest
positioning is: **I build evidence-backed semantic systems, and I know clearly
which parts of a local POC are production extension work.**

## 30-second introduction

> I work on enterprise knowledge graphs and GraphRAG systems in Python. My focus
> is turning heterogeneous source data into provenance-aware, tenant-scoped
> graph facts that can be queried, validated and explained. For the Energy POC,
> I materialised SAP-shaped assets/work orders with R2RML and telemetry with
> RML, published conformant RDF through SHACL, used a committed SPARQL query
> for the maintenance advisory, and exposed the result through a user-friendly
> Graph API and operations UI. The same published RDF can be projected one-way
> to Neo4j for GraphRAG traversal. The answer carries its evidence and
> uncertainty.

## JD terminology in plain English

| Term | Meaning | How to discuss it in this project |
|---|---|---|
| **Knowledge graph** | A connected model of entities, relationships, attributes, provenance and time. | Neo4j stores Documents, Chunks, Entities, Claims, Relations, Communities and Context Graph traces. |
| **RDF (Resource Description Framework)** | W3C graph data model based on subject–predicate–object triples. | The Energy POC materialises source-shaped records into RDF and exports Turtle. |
| **RDF triple** | One statement: `(subject, predicate, object)`. | `WT-01 energy:hasTemperatureObservation obs-1001`. |
| **IRI (Internationalized Resource Identifier)** | Globally unique identifier for an RDF resource. | R2RML templates produce stable asset IRIs such as `/energy/asset/WT-01`. |
| **RDFS (RDF Schema) / OWL (Web Ontology Language)** | Vocabulary and ontology languages for types, classes, properties and constraints. | The platform has type taxonomy, domain/range checks, subclass relationships and OWL-RL export/reasoning paths. |
| **SKOS (Simple Knowledge Organization System)** | W3C model for controlled vocabularies, concepts and taxonomies. | Useful for asset types, maintenance categories, failure codes and domain navigation. |
| **SPARQL (SPARQL Protocol and RDF Query Language)** | Query language and protocol for RDF graphs; comparable to SQL for triples. | `maintenance_review.rq` joins asset, telemetry, bulletin and work-order evidence. |
| **R2RML (RDB to RDF Mapping Language)** | W3C mapping language from relational databases to RDF. | `ontology/mappings/energy-assets.r2rml.ttl` maps source tables and joins into the Energy graph. |
| **RML (RDF Mapping Language)** | Extension of R2RML for mapping non-relational sources such as JSON, CSV and XML into RDF. | `ontology/mappings/energy-observations.rml.ttl` describes the JSON telemetry mapping contract. |
| **Semantic modelling** | Designing formal concepts, relations, constraints and identifiers so data has shared meaning. | The ontology separates Asset, Observation, Bulletin, WorkOrder, source identity and effective dates. |
| **GraphRAG** | Retrieval augmented generation where graph structure improves evidence selection and reasoning. | The platform combines vector ANN, BM25, reranking, graph traversal, GNN scoring and grounded synthesis. |
| **Hybrid retrieval** | Combining semantic vector retrieval with lexical and graph retrieval. | Vector ANN finds meaning; BM25 catches exact IDs; RRF and reranking combine candidates. |
| **ANN / vector search** | Fast approximate similarity search over embeddings. | Neo4j HNSW vector indexes retrieve semantically related chunks. |
| **BM25** | Term-based ranking using TF/IDF. | Important for exact policy numbers, asset IDs and technical codes. |
| **RRF** | Reciprocal Rank Fusion; combines rankings from different retrieval methods. | Merges vector and BM25 result lists before deeper reranking. |
| **Cross-encoder** | Scores a query and candidate document jointly; accurate but slower. | Used after broad retrieval to improve relevance ordering. |
| **Entity resolution** | Deciding when different names refer to the same real-world entity. | Exact/normalised → fuzzy → embedding → new entity, with ambiguous cases routed for review. |
| **Provenance** | Metadata showing where a fact came from and when. | Source document, chunk, character span, source system, confidence and timestamps travel with claims. |
| **Bitemporal data** | Valid time: when a fact is true; transaction time: when the system recorded it. | Supports historical “as-of” answers and prevents current data rewriting history. |
| **Inference** | Deriving new graph facts from rules or ontology semantics. | Forward-chaining rules create inferred edges with source type and confidence decay. |
| **Contradiction detection** | Finding facts that cannot all be true together. | Handles exclusive states, directional reversals, functional violations and positive/negative pairs. |
| **SHACL (Shapes Constraint Language)** | W3C constraint language for validating RDF graphs. | Incomplete Energy observations are rejected instead of becoming confident recommendations. |
| **PROV-O (PROV Ontology)** | W3C ontology for provenance activities, agents and entities. | The platform can project lineage into standards-aligned provenance RDF. |
| **Semantic search** | Search using meaning, ontology, relations and context—not only keywords. | `/search` provides retrieval; GraphRAG adds graph-aware evidence and synthesis. |
| **Graph API** | API exposing graph queries, retrieval or graph operations to applications. | FastAPI exposes query, search, SPARQL, RDF export, corrections, sources, Context Graph and MCP routes. |
| **Tenant isolation** | Ensuring one customer/workspace cannot access another’s data. | Tenant is taken from signed identity, not request input; graph, cache and traces are tenant-scoped. |
| **Triplestore** | Database optimised for RDF triples and SPARQL. | The POC uses an in-process RDF bridge for the demo; Stardog, GraphDB, Neptune RDF, RDFox or Virtuoso are production options. |
| **Property graph** | Graph database with nodes, relationships and properties. | Neo4j is the current operational graph runtime; RDF/OWL/SPARQL provide standards-based projections. |
| **Canonical semantic model** | A storage-independent statement of domain meaning from which target schemas are generated. | `ontology/models/energy-asset-intelligence.yaml` compiles to OWL/RDFS, SHACL, Neo4j constraints, and capability diagnostics. |
| **Capability-loss diagnostic** | A machine-readable warning that a target cannot preserve or enforce a semantic rule exactly. | Neo4j cannot natively enforce abstract types or relationship cardinality, so the compiler identifies the exact rule and required runtime control. |

## Concrete examples you can show from this project

### One model, RDF and property-graph projections

[`ontology/models/energy-asset-intelligence.yaml`](../ontology/models/energy-asset-intelligence.yaml)
is the authoritative Energy model. For example, `WindTurbine` extends `Asset`,
`Auditable` is a mixin rather than a false superclass, and `hasComponent` is
declared once on `Asset` and therefore applies to its descendants. The compiler
produces four reviewable artifacts:

```text
canonical YAML
  ├─ OWL/RDFS ontology       domain meaning and inference vocabulary
  ├─ SHACL shapes            closed-world publication validation
  ├─ Neo4j constraints       enforceable LPG keys
  └─ diagnostics.json        every approximation or unenforceable rule
```

Interview explanation: “RDF versus property graph is a deployment choice, not
the first modelling decision. I preserve one domain contract and compile it to
each target. I also report semantic loss explicitly: OWL minimum cardinality
does not reject missing data under the open-world assumption, while Neo4j
cannot enforce abstract types or relationship counts in portable DDL. SHACL
and the shared mutation validator provide the corresponding operational
controls.”

### RDFS and OWL — Energy ontology

[`ontology/energy/energy-asset-intelligence.ttl`](../ontology/energy/energy-asset-intelligence.ttl)
defines the Energy vocabulary. This is a real example of OWL classes,
object/datatype properties and RDFS hierarchy/ranges:

```turtle
energy:Observation a owl:Class ; rdfs:subClassOf prov:Entity .
energy:WorkOrder a owl:Class ; rdfs:subClassOf prov:Entity .
energy:DocumentRevision a owl:Class ;
  rdfs:subClassOf energy:TechnicalDocument .

energy:concernsAsset a owl:ObjectProperty .
energy:value a owl:DatatypeProperty ; rdfs:range xsd:decimal .
energy:observedAt a owl:DatatypeProperty ; rdfs:range xsd:dateTime .
```

Interview explanation: “`rdfs:subClassOf` means a `DocumentRevision` is also a
`TechnicalDocument`. `owl:ObjectProperty` connects resources, for example a
work order to an asset. `owl:DatatypeProperty` connects a resource to a scalar
value, and the RDFS range makes the expected datatype explicit.”

### R2RML — relational assets and work orders into RDF

[`ontology/mappings/energy-assets.r2rml.ttl`](../ontology/mappings/energy-assets.r2rml.ttl)
maps the SAP-shaped SQLite source:

```turtle
energy:AssetMap a rr:TriplesMap;
  rr:logicalTable [ rr:tableName "sap_assets" ];
  rr:subjectMap [
    rr:template "https://example.energy.demo/asset/{asset_id}";
    rr:class energy:Asset
  ];
  rr:predicateObjectMap [ rr:predicate rdfs:label;
    rr:objectMap [ rr:column "asset_name" ] ].

energy:WorkOrderMap a rr:TriplesMap;
  rr:logicalTable [ rr:tableName "sap_work_orders" ];
  rr:subjectMap [
    rr:template "https://example.energy.demo/record/{work_order_id}";
    rr:class energy:WorkOrder
  ];
  rr:predicateObjectMap [ rr:predicate energy:status;
    rr:objectMap [ rr:column "status" ] ];
  rr:predicateObjectMap [ rr:predicate energy:concernsAsset;
    rr:objectMap [ rr:parentTriplesMap energy:AssetMap;
      rr:joinCondition [ rr:child "asset_id"; rr:parent "asset_id" ] ] ].
```

Interview explanation: “The asset ID becomes a stable IRI. The parent triples
map turns the relational work-order-to-asset join into the semantic
`energy:concernsAsset` relationship. The mapping is version-controlled rather
than hidden in Python transformation code.”

### RML — observation data from a non-relational source shape

[`ontology/mappings/energy-observations.rml.ttl`](../ontology/mappings/energy-observations.rml.ttl)
contains an RML-style `ObservationMap` with an IRI template based on
`{observation_id}` and class `energy:Observation`. Use this distinction in an
interview: **R2RML is the relational mapping standard; RML generalises the same
approach to JSON, CSV, XML and API payloads.**

The repository also has [`ontology/mappings/supply-chain.r2rml.ttl`](../ontology/mappings/supply-chain.r2rml.ttl),
which is useful evidence that the mapping pattern is not Energy-only.

### SPARQL — the actual maintenance query

[`evals/energy_demo/sparql/maintenance_review.rq`](../evals/energy_demo/sparql/maintenance_review.rq)
executes the core Energy decision:

```sparql
PREFIX energy: <https://example.energy.demo/ontology#>

SELECT ?asset ?temperature ?threshold ?bulletin ?workOrder WHERE {
  VALUES ?bulletinId { "{{BULLETIN_ID}}" }
  ?observation a energy:Observation ;
    energy:observedAsset ?asset ;
    energy:metric "temperature_c" ;
    energy:value ?temperature .
  ?bulletin a energy:DocumentRevision ;
    energy:documentId ?bulletinId ;
    energy:temperatureReviewThreshold ?threshold .
  ?workOrder a energy:WorkOrder ;
    energy:concernsAsset ?asset ;
    energy:status "open" .
  FILTER(?temperature > ?threshold)
}
```

Interview explanation: “This query does not merely retrieve similar text. It
joins the observed asset, its measurement, the currently selected engineering
bulletin and an open work order. The response can therefore identify exactly
which facts produced the recommendation.”

### SHACL — reject incomplete RDF before it drives a decision

[`ontology/shapes/energy-asset-intelligence.shapes.ttl`](../ontology/shapes/energy-asset-intelligence.shapes.ttl)
defines minimum fields and datatypes:

```turtle
energy:ObservationShape a sh:NodeShape ; sh:targetClass energy:Observation ;
  sh:property [ sh:path energy:value ; sh:minCount 1 ; sh:datatype xsd:decimal ] ;
  sh:property [ sh:path energy:unit ; sh:minCount 1 ; sh:datatype xsd:string ] ;
  sh:property [ sh:path energy:observedAt ; sh:minCount 1 ; sh:datatype xsd:dateTime ] .
```

The E2E demonstration deliberately validates an incomplete observation and
reports `conforms: false`. The interview message is: “Validation is a gate, not
a dashboard decoration. A temperature without a value or unit cannot support a
maintenance recommendation.”

### RDF, SPARQL and Neo4j — complementary serving paths

[`graphrag/graph/sparql_bridge.py`](../graphrag/graph/sparql_bridge.py) loads a
Turtle export into `SPARQLBridge`, and [`api/routes/kg/knowledge.py`](../api/routes/kg/knowledge.py)
exposes `POST /kg/sparql`. In the Energy POC, the direction is deliberately
RDF-first: published RDF is the semantic source of truth, and
`graphrag/domains/energy/lpg_projection.py` can create a one-way, rebuildable
Neo4j traversal/GraphRAG read model. This supports an interview answer such as:
“RDF/Turtle plus SPARQL provide the governed interoperability surface; Neo4j
optimizes operational traversal and GraphRAG without becoming a competing
source of truth.”

### GraphRAG — concrete retrieval path

[`graphrag/retrieval/hybrid_retriever.py`](../graphrag/retrieval/hybrid_retriever.py)
is the main retrieval orchestrator. The documented pipeline is:

```text
query embedding / vector ANN
          + BM25 full-text
          ↓
       RRF fusion
          ↓
    cross-encoder reranking
          ↓
  bounded multi-hop graph expansion
          ↓
       GNN/GAT scoring
          ↓
 evidence-gated answer synthesis
```

[`graphrag/retrieval/agentic_retriever.py`](../graphrag/retrieval/agentic_retriever.py)
is a bounded fallback when the hybrid answer is low-confidence—not the default
for every question. That distinction signals practical cost and latency
awareness.

### Provenance, temporal state and governed actions

- [`docs/knowledge-graph-architecture.md`](knowledge-graph-architecture.md) documents relation provenance: source documents, confidence, `valid_from`, `valid_to`, `recorded_at` and tenant.
- [`graphrag/context_graph/`](../graphrag/context_graph/) stores cases, agent runs, observations, manifests, options, decisions and policy evaluations; manifests have integrity hashes.
- [`graphrag/agents/tool_policy.py`](../graphrag/agents/tool_policy.py) is the gate for allowed tools, risk, caller scopes, argument validation, dry-run, timeouts and audit.

## End-to-end worked example — WT-01

Use this as a compact whiteboard walkthrough. It is deliberately a deterministic
decision path: an LLM may explain the result, but it does not decide whether the
threshold is exceeded.

1. **Source records.** The SAP-shaped SQLite fixture has `sap_assets(WT-01, …)`
   and `sap_work_orders(WO-9001, WT-01, open)`. The Energy service materialises
   the matching temperature observation and the selected engineering bulletin.
2. **Mapping and identity.** The versioned R2RML mapping specifies the stable
   asset IRI `https://example.energy.demo/asset/WT-01`; its parent-triples-map
   join specifies `WO-9001 energy:concernsAsset …/WT-01` when the mapping runs.
3. **RDF facts.** The graph represents an `energy:Observation` with metric
   `temperature_c`, a numeric value, its observed asset, an `energy:DocumentRevision`
   with a review threshold, and an open `energy:WorkOrder`.
4. **SPARQL decision.** `maintenance_review.rq` joins those facts and applies
   `FILTER(?temperature > ?threshold)`. For the demo fixture, the returned
   evidence is **WT-01 | 96°C | 85°C | MFG-GBX-17-R2 | WO-9001**.
5. **Recommendation.** “WT-01 needs maintenance review: its temperature is
   above the currently selected bulletin threshold and an open work order gives
   an operational path to act.” The response carries source/evidence IDs; it
   does not present a probabilistic guess as a fact.
6. **Operational projection.** The published RDF can become tenant-scoped Neo4j
   `Entity` and `RELATES_TO` rows. That enables turbine → gearbox → observation
   → work order → bulletin traversal for GraphRAG, while preserving RDF IRI and
   provenance and allowing the read model to be rebuilt.

An easy follow-up is WT-04: the service returns **insufficient evidence** rather
than inventing a recommendation when a required fact is absent.

## Semantic reasoning and validation traps

| Topic | Accurate interview answer |
|---|---|
| **RDFS domain and range** | They are semantic statements. A reasoner may infer a type from a property's domain or range; they are not an input-validation rule that rejects bad data. |
| **OWL's open-world assumption** | Missing information is unknown, not false. The absence of an open work order cannot prove that no work order exists. Use explicit status, closed-world source checks, or SHACL where the business process needs completeness. |
| **SHACL** | SHACL validates explicit graph constraints—required fields, datatypes and cardinality—against a chosen data graph. In this POC, it is the quality gate before a finding is relied upon. |
| **Labels versus identity** | `rdfs:label` is for people; stable IRIs built from source business keys are for joins, deduplication and durable links. Two assets may share a label without being the same resource. |
| **Inference scope** | The Energy POC demonstrates RDF materialisation, SPARQL and SHACL. Do not claim that its maintenance result depends on rich OWL inference. The wider platform separately contains OWL-RL/forward-reasoning paths. |

Useful phrase: “I keep ontology semantics, data-quality validation, and the
business rule separate, so each is testable and explainable.”

## Be precise about what is implemented

| Capability | What the repository demonstrates | What to frame as a production extension |
|---|---|---|
| Enterprise sources | SAP/Snowflake/SharePoint-shaped fixtures and mapping patterns. | Live authenticated connectors, CDC scheduling, source contracts and operational monitoring. |
| RDF and semantics | Versioned R2RML/RML mappings, Turtle materialisation, SPARQL, SHACL, vocabulary and RDF/SPARQL bridge. | A managed triplestore and chosen reasoner configuration such as Stardog, GraphDB, Neptune RDF, RDFox or Virtuoso. |
| Energy maintenance answer | Deterministic, evidence-backed threshold and open-work-order decision with tenant checks, SHACL publication and a repeatable E2E scenario. | Client thresholds, asset taxonomy, workflow integration and human approval. |
| Energy Neo4j projection | One-way, tenant-scoped RDF-to-Neo4j batch projection, with fail-closed handling for unsupported RDF. | Live sizing, indexes, projection scheduling, reconciliation and retrieval evaluation against client data. |
| GraphRAG | Hybrid retrieval, bounded graph expansion, reranking and evidence-gated synthesis in the broader platform. | Grounding it in the client's approved corpus, evaluation set, security model and latency budget. |

This distinction is a strength: it shows a credible executable POC without
pretending that synthetic source shapes are already the client's production
systems.

## Production scenarios to talk through

| Scenario | Engineering response |
|---|---|
| Duplicate asset IDs across sources | Preserve source-system identity and provenance, reconcile to a canonical asset only with explicit matching rules, and retain links rather than silently merging records. |
| Late telemetry | Store event time separately from processing/recorded time. Recompute affected findings or publish a new corpus revision so “as-of” answers remain reproducible. |
| Changed engineering bulletin | Model immutable document revisions with validity dates; select the applicable revision at query time and show which revision was used. |
| Deleted SharePoint document | Preserve a tombstone and provenance, mark dependent evidence stale or superseded, and prevent the deleted revision from being selected for a new answer. |
| Permission change | Carry ACL/tenant metadata through ingestion and retrieval, invalidate affected indexes/caches, and enforce authorization again at query and tool-execution time. |
| Failed incremental load | Use idempotent source keys/content hashes, quarantine failed records, retain the last good corpus revision, and publish only after validation and reconciliation complete. |

## Mock interview drills

### Why use R2RML instead of a Python-only transformation?

“Python is still useful for extraction and orchestration, but R2RML makes the
relational-to-RDF semantics explicit, portable, reviewable and versioned. A
reviewer can see the IRI policy, classes and joins without reverse-engineering
application code. I test the mapping output just as I test application code.”

### How would you make a SPARQL maintenance query performant?

“Start with selective patterns—tenant, bulletin revision, asset or time range—
and avoid unbounded graph walks. Inspect the triplestore query plan, index the
common predicates and literal filters, paginate operational views, and keep
expensive full-text/vector retrieval outside a deterministic compliance query.
Then benchmark representative graph sizes, not only toy data.”

### How do RDF and a property graph coexist here?

“They serve complementary needs. The operational platform uses Neo4j for
traversal and application workflows. RDF/Turtle and SPARQL provide a standards
surface for semantic interoperability, mappings and external consumers. I keep
the identity and provenance model aligned across both projections.”

### How do you prevent a GraphRAG answer from hallucinating?

“I retrieve candidates with hybrid search, rerank and expand only a bounded
neighbourhood. The answer is evidence-gated: it cites source chunks/graph facts
and returns insufficient evidence when required facts are missing. For decisions
such as maintenance thresholds, I use a deterministic query or rule rather than
asking the model to infer the outcome.”

### How would you handle evolving vocabulary from several business teams?

“I would version a small core ontology, use SKOS for locally managed controlled
terms, map source-specific concepts to the core rather than forcing immediate
standardisation, and establish change review with competency questions and
regression data.”

### What would you test before a production release?

“Mapping fixtures, SHACL conformance, SPARQL result snapshots, identity and
tenant isolation, temporal/as-of behaviour, ACL filtering, retrieval relevance,
negative cases, and a complete ingest-to-answer regression run. I would also
monitor freshness, validation failure rate, answer evidence coverage and query
latency.”

## Five-minute final refresh

- Start with the business outcome: evidence-backed maintenance prioritisation,
  not “a chatbot over documents.”
- Explain the chain: source record → stable IRI and RDF → SHACL gate → SPARQL
  evidence → governed API/UI answer.
- R2RML maps relational data; RML extends that pattern to JSON, CSV, XML and APIs.
- RDF is an interchange model; Neo4j is the operational property-graph runtime
  in this platform; they are complementary.
- Domain/range can infer types. SHACL validates completeness. OWL is open-world.
- Never say missing evidence proves a negative; return insufficient evidence.
- Separate deterministic operational decisions from probabilistic GraphRAG
  explanations and discovery.
- Treat provenance, tenant isolation, time and ACLs as first-class graph data.
- For production, mention idempotency, corpus revisions, reconciliation,
  observability and human approval for actions.
- Close with the repeatable proof: `scripts/run_energy_demo_e2e.py` and the
  Energy unit tests exercise the POC path end to end.

Questions worth asking the client: “Which asset and document identifiers are
authoritative?”, “What decision needs to be explainable to an operator or
auditor?”, and “Which latency, freshness and access-control constraints are
non-negotiable?”

## The Energy POC flow

```text
SAP-shaped assets/work orders
        + Snowflake-shaped telemetry
        + SharePoint-shaped engineering bulletins
                     |
              R2RML mapping
                     |
              RDF materialisation
                     |
       SHACL / semantic validation
                     |
       version-controlled SPARQL
                     |
    evidence-backed maintenance answer
                     |
    tenant-scoped API + operations UI
```

The current scenario is deliberately explainable: WT-01 has a 96 °C gearbox
observation, the effective manufacturer bulletin sets an 85 °C threshold, and
open work order WO-9001 provides operational context. When evidence is missing,
the answer says more telemetry is required.

## Likely interview questions

### Why use RDF instead of only a property graph?

RDF gives standardised semantics, globally identifiable resources, linked-data
interoperability and SPARQL. A property graph is often excellent for operational
traversal, application performance and developer ergonomics. I would not make it
a religious choice: use a property graph where it is strongest, expose or
project RDF where interoperability and formal semantics matter, and keep mapping
and provenance explicit.

### Why use R2RML/RML?

Mappings separate source-system shape from canonical meaning. They are versioned,
reviewable and repeatable. A source column can change without changing the
ontology, and a new source can map into the same semantic model. I would test
mapping coverage, key uniqueness, null handling, datatype conversion, joins and
incremental behaviour.

### How would you integrate SAP, Snowflake and SharePoint?

First profile each source and identify business keys, update timestamps, deletes,
authority and ACL metadata. Then build adapters or CDC/batch extracts into a
canonical source-record envelope. Apply RML/R2RML mappings, reconcile versions
idempotently, preserve source IDs and provenance, validate the resulting graph,
and publish a corpus revision only after the write completes successfully.

### How do you prevent hallucinated GraphRAG answers?

Use retrieval and answer gates: tenant/ACL filtering, source authority, temporal
validity, contradiction annotation, evidence coverage, confidence thresholds and
citations. The model should only synthesise from bounded retrieved evidence. If
evidence is insufficient, return an explicit uncertainty response or ask for more
information. Evaluation must test groundedness, not just fluency.

### How do you handle conflicting sources?

Do not silently overwrite one fact with another. Preserve both assertions with
provenance and timestamps, detect the conflict type, apply authority and
supersession rules, route unresolved cases for human review, and expose the
conflict to retrieval and answer policy.

### How do you evaluate a knowledge graph?

Evaluate several layers: mapping completeness, entity-resolution precision and
recall, relation/schema validity, contradiction rate, provenance coverage,
retrieval recall/precision, answer faithfulness, citation correctness, latency,
cost and tenant-isolation/security regression. A graph can be syntactically valid
and still be semantically wrong or operationally unusable.

### How would you choose Stardog, GraphDB, Neptune RDF, RDFox or Virtuoso?

Compare SPARQL 1.1 coverage, reasoning semantics, SHACL support, federation,
transactions, security/ACL integration, vector or hybrid search, operational
model, cloud fit, performance, licensing and team capability. I would benchmark
the actual workload—mapping load, incremental updates, SPARQL patterns,
reasoning, concurrent retrieval and recovery—rather than choose from feature
lists alone.

### What would you productionise next?

Replace fixtures with authenticated SAP/Snowflake/SharePoint connectors, choose a
managed RDF/runtime topology, formalise ontology governance, add CDC and replay,
connect enterprise ACLs, harden secrets and observability, run performance/load
benchmarks, and establish data-quality and incident runbooks. The local POC proves
the semantic workflow; it is not presented as a finished production deployment.

## Strong trade-offs to mention

- **RDF vs Neo4j:** standards/interoperability versus operational traversal and ecosystem fit.
- **Materialised inference vs query-time reasoning:** faster predictable reads versus more storage and re-materialisation work.
- **Vector vs lexical retrieval:** semantic recall versus exact identifiers and terminology.
- **Broad retrieval vs bounded retrieval:** recall versus latency, cost and answer noise.
- **Automatic entity resolution vs human review:** throughput versus false-link risk.
- **LLM extraction vs deterministic mappings:** flexibility versus repeatability, cost and auditability.
- **Cache vs freshness:** lower latency/cost versus corpus-revision invalidation complexity.
- **Autonomous agent writes vs governed tools:** convenience versus safety, audit and compensation requirements.

## Project evidence to point to

- `docs/architecture.html` — full platform architecture
- `docs/energy-architecture.html` — Energy-specific architecture and demo story
- `ontology/mappings/energy-assets.r2rml.ttl` — R2RML mapping
- `evals/energy_demo/sparql/maintenance_review.rq` — semantic maintenance query
- `graphrag/domains/energy/demo.py` — source materialisation and evidence-backed answer service
- `scripts/run_energy_demo_e2e.py` — repeatable E2E workflow
- `tests/unit/test_energy_demo.py` — mapping, answer, isolation, RDF and validation tests
- `mcp_server/capabilities/` — governed agent tool capabilities
- `graphrag/context_graph/` — decision traces, manifests and policy evaluation
- `graphrag/retrieval/` — hybrid, adaptive and agentic retrieval

## Phrases worth using

- “The graph is an evidence system, not just a RAG index.”
- “I separate source observation, semantic assertion, inference and business decision.”
- “Unknown is a valid result; the system should not convert missing evidence into certainty.”
- “Tenant and ACL boundaries are retrieval constraints, not UI conventions.”
- “Mappings, ontology versions, query versions and corpus revisions need to be observable.”
- “I would benchmark the target workload and recovery behaviour before choosing the RDF platform.”

## Avoid overclaiming

Say **SAP-shaped**, **Snowflake-shaped** and **SharePoint-shaped** for the current
POC unless you have actually connected to those systems. Say the architecture is
ready for Stardog/GraphDB/Neptune RDF/RDFox/Virtuoso evaluation rather than saying
the project already runs on all of them. Be explicit that the Energy data is
synthetic and the recommendation is advisory.
