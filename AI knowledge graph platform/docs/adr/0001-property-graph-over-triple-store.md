# ADR 0001 — Property Graph (Neo4j) over Triple Store (RDF Store)

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024-Q1 |
| **Deciders** | Platform architect, AI/ML lead |

---

## Context

The platform needs a graph database to store entities, relations, and document provenance
for a RAG pipeline. Two viable categories exist:

**Triple stores (RDF stores):** Virtuoso, Apache Jena, Stardog, Oxigraph — native RDF/OWL,
SPARQL query language, open-world semantics, strong ontology tooling.

**Property graphs:** Neo4j, Amazon Neptune (property graph mode), ArangoDB — flexible
property schemas on nodes and edges, Cypher query language, vector index support,
closed-world semantics option.

---

## Decision

**Use Neo4j with the property graph model.** Provide RDF/OWL interoperability via
`scripts/export_rdf.py` (rdflib Turtle serialisation) for toolchain integration.

---

## Reasons

### 1. Edge properties are first-class requirements

Every `RELATES_TO` edge needs 10+ properties:

```
confidence, weight, source_doc_id, source_doc_ids (list),
source_type, constraint_type, valid_from, valid_to,
recorded_at, tenant, extraction_model, chunk_span_start/end
```

In an RDF triple store, attaching properties to predicates requires **reification**
— wrapping each triple in a `rdf:Statement` node with 3+ additional triples. A 1M-edge
graph becomes 5M+ triples before any data is added. Neo4j's property graph model
natively supports arbitrary key-value properties on edges at no structural cost.

### 2. Vector indexes on entity nodes

The retrieval pipeline requires ANN (approximate nearest-neighbour) search over 3072d
entity embeddings. Neo4j 5.x provides native vector indexes (`db.index.vector`) with
HNSW graph structure. Triple stores have no native vector index — they would require
a separate Qdrant/Pinecone sidecar, adding operational complexity and a cross-service
join on every retrieval query.

### 3. Cypher expressiveness for graph traversal patterns

The multi-hop retrieval query:
```cypher
MATCH path = (e)-[:RELATES_TO*1..2 {tenant: $t}]->(e2)
WHERE ALL(r IN relationships(path) WHERE r.confidence >= 0.7)
```

requires filtering relationship properties along a variable-length path. SPARQL
property paths (`?a :rel* ?b`) have no equivalent mechanism for filtering intermediate
edge properties — you must use SPARQL 1.1 `FILTER` with nested sub-patterns,
which is verbose and poorly optimised in most stores.

### 4. Closed-world semantics align with the use case

The knowledge graph is built from ingested documents — facts not in the graph are
unknown, not false. This is *technically* open-world, but in practice retrieval
quality improves when the system treats the ingested corpus as authoritative for
the retrieval window. Neo4j's closed-world orientation matches this operational model.
We explicitly model negative knowledge via `NEGATIVE_RELATES_TO` edges where needed,
avoiding reliance on the open-world assumption.

### 5. Transaction time stamping

`recorded_at = datetime() ON CREATE` on every MERGE — immutable transaction time
for bitemporal queries. Triple stores support named graphs for versioning, but
attaching immutable transaction time to individual triples requires additional
infrastructure that Neo4j provides natively via ON CREATE SET semantics.

---

## Consequences

**Positive:**
- Single database for graph storage, vector search, and full-text (BM25) — no sidecars
- Property-rich edges at zero structural overhead
- Cypher multi-hop traversal with edge property filters
- Native vector ANN indexes

**Negative:**
- No native SPARQL support — queries require Cypher rewrite for SPARQL consumers
- No native OWL reasoning — forward-chaining inference is implemented in Python
  (`ForwardChainingEngine`) and materialised as edges rather than evaluated at query time
- Ontology tooling (Protégé, HermiT) requires the Turtle export step

**Mitigation:** The `export_rdf.py` script provides a standards-compliant Turtle export
on demand, enabling interoperability with OWL reasoners and SPARQL endpoints without
coupling the primary workload to RDF semantics.
