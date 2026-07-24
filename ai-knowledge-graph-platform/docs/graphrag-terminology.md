# GraphRAG Terminology Reference

Every term used in the platform, the literature, and interviews — with a
short plain-English definition, a concrete example where possible, and the
file where you'll find it implemented.

---

## A

### Alias
An alternative name for an entity that refers to the same real-world thing.

**Example:** "SpaceX", "Space Exploration Technologies", and "Space Exploration Corp." are all aliases of the same organisation entity.

**In this project:** `graphrag/graph/alias_registry.py` — the registry stores aliases in Neo4j as `(:Alias {value})-[:ALIAS_OF]->(:Entity)` nodes. Resolution runs in 4 stages: exact match → normalized match → embedding similarity → human review queue.

---

### Agentic RAG / Agentic Retriever
A retrieval strategy where an LLM decides what to search for next, rather than running a single retrieval pass. The agent iterates: retrieve → reason → retrieve more → answer.

**Example:** Question: "What regulations govern the engine mount on aircraft built before 2018?" Stage 1 retrieves general engine mount documents. LLM decides it needs the specific FAA directive. Stage 2 retrieves on "FAA-AD-2024-01-02". Now it can answer.

**In this project:** `graphrag/retrieval/agentic_retriever.py` — implements IRCoT. Triggers when the hybrid retriever returns a low-confidence answer. 4-step max, then synthesizes from accumulated context.

---

### ANN (Approximate Nearest Neighbor)
A search algorithm that finds the vectors most similar to a query vector — not the exact nearest neighbors (which requires comparing to all vectors) but a fast approximation with tunable precision/recall trade-off.

**Example:** Given a query embedding for "engine airworthiness requirements", ANN finds the 20 most semantically similar chunk embeddings from a corpus of 50,000 chunks in milliseconds.

**In this project:** Neo4j's native vector index using HNSW (Hierarchical Navigable Small World). Called via `db.index.vector.queryNodes`. See `graphrag/graph/neo4j_client.py: vector_search_chunks()`.

---

### Asserted edge / Asserted fact
A fact explicitly stated in a source document, as opposed to an *inferred* fact derived by a reasoning engine.

**Example:** Document states "FAA-AD-2024-01-02 supersedes FAA-AD-2022-03-07." That creates an asserted `SUPERSEDES` edge — recorded with `source_type='document'` (see note below on the literal property value).

**In this project:** All edges written by the extractor carry `source_type='document'` — ⚠ **not** the literal string `'asserted'` (verified live: `MATCH ()-[r:RELATES_TO {tenant:'aerospace'}]->() RETURN r.source_type, count(r)` returns only `'document'` and `'inferred'`). "Asserted" is the human-readable concept — "stated in a source document, as opposed to derived by inference" — but a query filtered on `{source_type: 'asserted'}` returns **zero rows**. Use `'document'`. The inference engine writes edges with `source_type='inferred'`. Contradiction detection uses this distinction to trace the origin of any conflict.

---

## B

### Bayesian confidence accumulation
A formula for combining confidence scores from multiple independent sources into a single fused score.

**Formula:** `fused = 1 − (1−c₁)(1−c₂)...`

**Example:** Doc A asserts relation R with confidence 0.8. Doc B independently asserts the same relation with confidence 0.7. Fused confidence = `1 − (1−0.8)(1−0.7) = 1 − 0.06 = 0.94`. Two independent sources are more certain than either alone. Averaging would give 0.75, which is wrong — it doesn't capture the independent confirmation.

**In this project:** `graphrag/graph/neo4j_client.py: merge_relation()` — the Cypher `SET r.confidence = ...` block implements this as a Cypher expression. See `docs/adr/0003-bayesian-confidence-accumulation.md`.

---

### Bi-encoder
A retrieval model that encodes the query and the document independently into separate vectors, then scores them by vector similarity (e.g., cosine). Fast because document embeddings can be pre-computed.

**Example:** `text-embedding-3-large` encodes both queries and documents into 3072d vectors. At query time, only the query needs encoding; all document embeddings are pre-computed and stored in the vector index.

**Contrast with:** Cross-encoder — encodes query and document *jointly*, which is more accurate but much slower (can't pre-compute).

**In this project:** Stages 1 (vector ANN) and 2 (BM25 fusion). `graphrag/ingestion/embedder.py` for documents, `graphrag/retrieval/local_search.py` for query embedding.

---

### BM25 (Best Match 25)
A classical information retrieval ranking algorithm based on term frequency (TF) and inverse document frequency (IDF). Finds documents that contain the exact query terms, weighted by their rarity in the corpus.

**Example:** Query "FAA-AD-2024-01-02 engine mount". BM25 will score chunks highly if they contain the exact AD number — something a semantic embedding search might miss if the embedding space conflates "AD-2024" with "AD-2023".

**In this project:** Neo4j full-text index via `db.index.fulltext.queryNodes`. Stage 2 of the retrieval pipeline. Combined with vector ANN results via RRF. See `graphrag/graph/neo4j_client.py: bm25_search_chunks()`.

---

## C

### Chunk
A contiguous segment of a document that fits within an LLM's context window. Documents are split into chunks for embedding and retrieval.

**Example:** A 20-page regulatory document is split into ~40 chunks of 512 tokens each. Each chunk is embedded and stored separately, with a link back to the parent document.

**In this project:** `graphrag/core/models.py: Chunk` — has fields: `id`, `document_id`, `text`, `chunk_index`, `embedding`, `token_count`. Created by `graphrag/ingestion/chunker.py`.

---

### Community (graph community)
A dense cluster of entities in the graph — nodes that are more connected to each other than to the rest of the graph. Found by community detection algorithms (Leiden, Louvain, etc.).

**Example:** In a regulatory corpus, one community might contain all entities related to "Boeing 737 MAX airworthiness" — the relevant ADs, the aircraft types, the responsible regulators, and the maintenance procedures.

**In this project:** `graphrag/graph/community_builder.py` — runs multi-resolution Leiden via `graspologic`. Communities are used by `graphrag/retrieval/global_search.py` for map-reduce summarisation across the full corpus.

---

### Community summary
An LLM-generated natural language summary of a community's entities and their relationships. Used as a high-level context source in global search.

**In this project:** `graphrag/graph/community_summarizer.py` — generates summaries using Groq. Stored as `Community.summary` in Neo4j.

---

### Confidence (edge confidence)
A float [0.0, 1.0] representing how certain the system is that a relation is correct. 0.9+ = explicit statement in source, 0.6–0.9 = strong implication, below 0.6 = weak inference.

**Example:** "FAA-AD-2024-01-02 supersedes FAA-AD-2022-03-07" — explicit statement → confidence 0.95. "The 2024 directive appears to update the 2022 guidance" — implication → confidence 0.75.

**In this project:** Extracted by Groq in the JSON output. Clamped to [0,1] in `graphrag/ingestion/extractor.py`. Merged with Bayesian accumulation in `merge_relation`. Decays per hop for inferred edges (default decay 0.95).

---

### Contradiction detection
The process of finding pairs of facts in the knowledge graph that cannot both be true simultaneously.

**Four types in this project:**
1. Directional reversal
2. Exclusive state (e.g., IS_AIRWORTHY and IS_UNAIRWORTHY)
3. Functional violation
4. Positive/negative pair

A former fifth type, "multi-source conflict" (same triple asserted by two
non-superseding documents), was retired — agreement between two independent
documents on the same triple is corroboration, not a contradiction. It's now
tracked as a trust signal (`independent_source_count` / `corroborated_edge_rate`)
instead of an open `Conflict`.

**In this project:** `graphrag/graph/contradiction_detector.py: scan()`. Runs post-ingestion scoped to the new document's entities. Results stored as `(:Conflict)` nodes. Retrieval checks for open conflicts on entities in the result set and warns the LLM in the answer prompt when context includes a disputed fact (gated by `retrieval.conflict_annotation_enabled`, default on).

---

### Cross-encoder
A reranking model that takes a (query, document) pair as a single input and scores their relevance jointly. More accurate than bi-encoders but cannot pre-compute document representations — must run at query time.

**Example:** After bi-encoder retrieval returns 20 chunks, `ms-marco-MiniLM-L-6-v2` scores each (query, chunk) pair jointly, producing a more accurate ranking than cosine similarity alone.

**In this project:** Stage 3 of the retrieval pipeline. `graphrag/retrieval/reranker.py`. The cold-start latency (~200ms for first query) is documented in the runbook.

---

### Cypher
Neo4j's declarative graph query language. Analogous to SQL for relational databases but designed for graph traversal.

**Example:**
```cypher
MATCH (ad:Entity {type: 'AIRWORTHINESS_DIRECTIVE', tenant: 'aerospace'})
-[:RELATES_TO {relation: 'SUPERSEDES'}]->
(prev:Entity {type: 'AIRWORTHINESS_DIRECTIVE'})
RETURN ad.name, prev.name, COUNT{(ad)-[:RELATES_TO]->()} AS degree
```

**In this project:** Used throughout `graphrag/graph/neo4j_client.py`. Key patterns documented in `docs/cypher-patterns.md`.

---

## D

### Datalog
A declarative logic programming language designed for querying relational data. Used here informally to describe the style of the forward-chaining inference rules: "if A supersedes B and B supersedes C, then A supersedes C."

**In this project:** The `InferenceRule` dataclass in `graphrag/graph/inference_engine.py` encodes Datalog-style rules as transitivity/symmetry/inverse/composition patterns. Not literally Datalog — implemented as Cypher queries.

---

### Dead letter queue (DLQ)
A message queue that receives messages that failed processing after the maximum number of retries. Used to prevent bad messages from blocking healthy ones.

**Example:** A malformed document crashes the ingestion worker on 3 attempts. Rather than blocking the queue, the message moves to `graphrag.ingest.queue.dlq` for manual investigation.

**In this project:** RabbitMQ DLQ per queue, configured in `workers/ingestion_worker.py`. Check via RabbitMQ management UI at `localhost:15672`.

---

### Dense retrieval
Retrieval based on dense vector representations (embeddings), as opposed to *sparse retrieval* (keyword-based like BM25). Dense retrieval captures semantic meaning; sparse retrieval captures exact term matches.

**In this project:** Stage 1 (vector ANN). Combined with sparse retrieval (BM25, stage 2) in a *hybrid* approach.

---

### Domain/range constraint
An ontology rule that restricts which entity types can participate in a relation. The *domain* is the constraint on the source entity type; the *range* is the constraint on the target.

**Example:** `SUPERSEDES` domain = `[REGULATION, AIRWORTHINESS_DIRECTIVE, SERVICE_BULLETIN]`, range = same. This means a `PERSON` cannot SUPERSEDE a `REGULATION` — the relation would fail domain validation.

**In this project:** `graphrag/graph/ontology_registry.py: validate_relation_triplet()`. Rules loaded from `config/ontologies/*.yml`.

---

## E

### Embedding
A dense vector representation of text (or other data) that captures its semantic meaning. Similar texts have similar embeddings (high cosine similarity).

**Example:** "engine mount inspection" and "nacelle attachment inspection" have high cosine similarity because they describe the same concept in different words. "engine mount inspection" and "quarterly earnings report" have low similarity.

**In this project:** 3072-dimensional OpenAI embeddings (`text-embedding-3-large`). Stored on `(:Chunk).embedding` and `(:Entity).embedding` in Neo4j. Used for ANN search and entity resolution.

---

### Entity
A discrete, named concept extracted from text — a person, organization, product, location, concept, or event.

**Example:** From "FAA issued directive AD-2024-01-02 requiring inspection of all Boeing 737-800 engine mounts": entities are `FAA` (REGULATOR), `AD-2024-01-02` (AIRWORTHINESS_DIRECTIVE), `Boeing 737-800` (AIRCRAFT_TYPE).

**In this project:** `graphrag/core/models.py: Entity`. Written to Neo4j via `merge_entity()`. Key: `(name, type, tenant)`.

---

### Entity resolution
The process of determining that multiple textual mentions refer to the same real-world entity, and merging them into a canonical node. Also called entity disambiguation or record linkage.

**In this project:** `graphrag/graph/alias_registry.py` — 4-stage pipeline. See full explanation in the defensibility drill, Q6.

---

## F

### Faithfulness (RAGAS metric)
Measures whether the generated answer is factually grounded in the retrieved context. A faithful answer makes only claims that can be traced to the provided context. Score [0,1].

**Example:** Context contains "G-ABCD failed the AD-2024-01-02 compliance check." Answer says "G-ABCD is airworthy." Faithfulness ≈ 0 (answer contradicts context). Answer says "G-ABCD has not completed the required inspection per AD-2024-01-02." Faithfulness ≈ 1.

**In this project:** Computed by `graphrag/evaluation/ragas_evaluator.py`. Logged in `KPIEvent.faithfulness`. Threshold: 0.8 (alert if below). Measured: **0.937** on answerable questions; **0.842** overall (correct refusals score 0 and are excluded from the answerable denominator).

---

### Forward-chaining inference
An inference strategy that derives new facts from existing ones at *write time*, materialising the derived facts into the graph. Contrast with backward-chaining (query-time derivation).

**Example:** Rule: "if A supersedes B and B supersedes C, then A transitively supersedes C." When a new SUPERSEDES edge is written, the engine runs and writes any derivable transitive edges. Subsequent queries find the transitive fact as a normal edge.

**In this project:** `graphrag/graph/inference_engine.py: ForwardChainingEngine`. Runs to fixpoint (iterates until no new edges are derived). Post-ingestion scoped via `run_for_document(doc_id)`. See `docs/adr/0002-forward-chaining-over-backward-chaining.md`.

---

### Fusion (RRF — Reciprocal Rank Fusion)
A technique for combining ranked lists from different retrieval methods (e.g., vector ANN and BM25) into a single ranking without needing to normalize scores.

**Formula:** `RRF_score(d) = Σ 1/(k + rank(d))` where k=60 (default).

**Example:** A chunk ranked 3rd by vector ANN and 1st by BM25 gets a higher fused score than one ranked 1st by vector only. This rewards results that appear consistently across multiple methods.

**In this project:** `graphrag/retrieval/hybrid_retriever.py`. Stage 1+2 output is fused before cross-encoder reranking.

---

## G

### GAT (Graph Attention Network)
A type of graph neural network that assigns different attention weights to different neighbors when aggregating information. More expressive than GCN.

**Example:** Entity "G-ABCD" has edges to "FAA-AD-2024-01-02" (high attention — it's directly affected) and "Boeing 737 History" (lower attention — only loosely related). The GAT selectively amplifies the more relevant neighbors.

**In this project:** `graphrag/graph/gnn_scorer.py` — implements both GCN and GAT. Selectable via config `gnn_type: gat`.

---

### GCN (Graph Convolutional Network)
A type of graph neural network that aggregates information from all neighbors with equal weight, using symmetric normalization.

**In this project:** Default GNN type in `graphrag/graph/gnn_scorer.py`. Lighter than GAT; use GAT when neighbor selectivity matters.

---

### Global search
A retrieval mode that answers broad, thematic questions by summarising across all communities in the graph, rather than retrieving specific chunks. Uses map-reduce: summarise each community, then synthesise.

**Example:** "What are the main airworthiness concerns for Boeing 737 aircraft?" — this requires synthesizing across many documents. Local search (specific chunk retrieval) would return fragments. Global search summarises the entire relevant sub-graph.

**In this project:** `graphrag/retrieval/global_search.py`. Requires communities to be built first (`scripts/community_rebuild.py`).

---

### GNN (Graph Neural Network)
A neural network architecture that operates on graphs by passing messages between connected nodes. Each node aggregates information from its neighbors to update its representation.

**In this project:** Stage 5 of the retrieval pipeline. Re-scores chunks by their structural position in the entity subgraph relative to the query entity. See `graphrag/graph/gnn_scorer.py`.

---

### GraphRAG
Retrieval-Augmented Generation enhanced with a knowledge graph. Instead of (or in addition to) retrieving raw text chunks by semantic similarity, the system retrieves structured facts, entity relationships, and graph-derived context from a knowledge graph.

**Standard RAG:** Query → vector search → top-k chunks → LLM generates answer

**GraphRAG:** Query → vector search + graph traversal + inference + community summaries → structured context → LLM generates cited, reasoned answer

**In this project:** The entire platform is a GraphRAG implementation. The 6-stage retrieval pipeline is the GraphRAG pipeline.

---

## H

### Hybrid retrieval
A retrieval strategy that combines dense (vector/semantic) and sparse (keyword/BM25) retrieval, exploiting the complementary strengths of each method.

**Example:** "FAA-AD-2024-01-02 applicability" — BM25 finds the exact AD number; vector search finds semantically related airworthiness content. Hybrid fusion promotes results that appear in both.

**In this project:** `graphrag/retrieval/hybrid_retriever.py`. The primary retrieval mode.

---

### Hybrid search
Same as hybrid retrieval. Sometimes used specifically to mean the combination of full-text and vector search within a single query (as opposed to combining the results of two separate queries).

---

### HNSW (Hierarchical Navigable Small World)
The graph-based algorithm used by most modern vector indexes (including Neo4j's) to perform fast approximate nearest neighbor search. Builds a multi-layer graph where each layer is progressively sparser.

**In this project:** Neo4j's native vector index uses HNSW internally. Configured at index creation time (see `graphrag/graph/schema.cypher`).

---

## I

### Inferred edge
A relation derived by the inference engine from existing asserted edges, rather than extracted from a source document. Stored in the graph with `source_type='inferred'` and a decayed confidence score.

**Example:** Asserted: AD-2024 supersedes AD-2022 (confidence 0.95), AD-2022 supersedes AD-2020 (confidence 0.95). Inferred: AD-2024 supersedes AD-2020 (confidence 0.95² = 0.857).

**In this project:** Written by `graphrag/graph/inference_engine.py`. Distinguished from asserted edges by `source_type` property.

---

### IRCoT (Interleaved Retrieval and Chain-of-Thought)
A multi-step reasoning strategy where retrieval and reasoning are interleaved: the model retrieves evidence, reasons about it, determines what additional evidence is needed, retrieves again, and so on.

**In this project:** Implemented in `graphrag/retrieval/agentic_retriever.py`. Used as the fallback when hybrid retrieval produces a low-confidence answer.

---

## K

### Knowledge graph (KG)
A structured representation of entities and the relationships between them, stored as a graph (nodes and edges). Enables machine reasoning over complex, interconnected knowledge.

**Example:**
```
(FAA-AD-2024-01-02) -[SUPERSEDES]-> (FAA-AD-2022-03-07)
(FAA-AD-2024-01-02) -[APPLIES_TO]-> (Boeing 737-800)
(Boeing 737-800)    -[MANUFACTURED_BY]-> (Boeing)
```

**In this project:** Stored in Neo4j. Entity nodes with typed relations. 39 modules in `graphrag/graph/` operate on the graph.

---

## L

### LCA (Least Common Ancestor)
In a type taxonomy tree, the LCA of two types is the most specific common parent type.

**Example:** LCA(AIRWORTHINESS_DIRECTIVE, SERVICE_BULLETIN) = CONCEPT. Both are subtypes of CONCEPT; CONCEPT is the most specific common parent.

**In this project:** `graphrag/graph/type_taxonomy.py: least_common_ancestor()`. Used for entity merge decisions — if two entities' types have a close LCA, they might be the same entity expressed differently.

---

### Leiden algorithm
A community detection algorithm that improves on the Louvain algorithm. Finds communities by optimising modularity while maintaining connectedness of communities.

**In this project:** Run via the `graspologic` library. `graphrag/graph/community_builder.py`. Multi-resolution: runs at multiple gamma values to find communities at different granularities.

---

### Local search
A retrieval mode that finds specific, entity-level answers by searching for relevant chunks near specific entities in the graph. Contrast with global search (thematic, corpus-wide).

**Example:** "Which documents reference FAA-AD-2024-01-02?" — specific lookup near a known entity. Use local search, not global.

**In this project:** `graphrag/retrieval/local_search.py`. The primary path through `hybrid_retriever.py`.

---

## M

### Materialization
The act of computing and storing derived facts explicitly, rather than computing them at query time. Forward-chaining inference *materialises* inferred edges into the graph.

**Trade-off:** Storage overhead and maintenance cost at write time vs. fast, simple reads at query time.

**In this project:** All inferred edges are materialized. `scripts/community_rebuild.py` materialises community structure. RAGAS scores are materialized as `KPIEvent` rows.

---

### Mentions (entity mentions)
The specific occurrences of an entity in a text chunk. A chunk can mention multiple entities, and an entity can be mentioned in multiple chunks.

**In this project:** `MENTIONS` relationship: `(:Chunk)-[:MENTIONS]->(:Entity)`. Written by `neo4j_client.merge_mentions()`. Used in multi-hop traversal: start from query-relevant chunks, follow MENTIONS edges to find entities, then follow RELATES_TO edges to find related entities.

---

### Multi-hop traversal
Graph traversal that follows multiple edges in sequence to find entities or facts connected through intermediate nodes — entities that no single chunk directly links together.

**Example:** Query: "What compliance requirements apply to the engine on G-ABCD?" 
- Chunk mentions G-ABCD (AIRCRAFT_TYPE)
- G-ABCD RELATES_TO Boeing 737-800 (AIRCRAFT_TYPE)
- Boeing 737-800 SUBJECT_OF FAA-AD-2024-01-02 (AIRWORTHINESS_DIRECTIVE)
- FAA-AD-2024-01-02 REQUIRES engine mount inspection (MAINTENANCE_PROCEDURE)

**In this project:** Stage 4. `graphrag/graph/neo4j_client.py: get_multihop_chunks()`. Default depth: 2 hops.

---

## O

### Ontology
A formal specification of the concepts (entity types), relationships (relation types), and rules in a knowledge domain. Defines *what kinds of things exist* and *how they relate*.

**Example:** The aerospace ontology defines that `AIRWORTHINESS_DIRECTIVE` is a subtype of `REGULATION`, and that `SUPERSEDES` can only relate regulatory documents to regulatory documents.

**In this project:** `config/ontologies/aerospace_regulatory.yml`. Loaded at startup by `OntologyRegistry` and `TypeTaxonomy`. Domain-agnostic: swap the YAML to change domains.

---

### OWL (Web Ontology Language)
A W3C standard for encoding ontologies using formal description logic. Enables machine-readable ontology definitions with well-defined semantics.

**In this project:** `graphrag/graph/owl_reasoner.py` — applies OWL 2 RL entailment rules to an rdflib graph. Used for offline reasoning over RDF exports. `scripts/export_rdf.py --infer` triggers it.

---

### OWL-RL (OWL 2 RL profile)
A tractable subset of OWL 2 that can be implemented using forward-chaining rules in polynomial time. Supports subclass propagation, symmetric/inverse properties, property chains.

**In this project:** Applied by `OWLRLReasoner` using the `owlrl` Python library. Generates inferred triples in RDF exported graphs.

---

## P

### Provenance
The record of where a fact came from — which source document, which extraction model, which prompt version, when it was extracted.

**Example:** Edge `SUPERSEDES` between two ADs has properties: `source_doc_id="ad-compliance-check-2024-03"`, `extraction_model="llama-3.3-70b"`, `extracted_at=2026-05-31T12:00:00Z`.

**In this project:** Every `RELATES_TO` edge stores `source_doc_id`, `source_doc_ids` (list for Bayesian merges), `extraction_model`, `prompt_version`, `extracted_at`. Queried via `GET /graph/entities/{id}/provenance`.

---

## R

### RAG (Retrieval-Augmented Generation)
A technique for grounding LLM answers in external knowledge. Instead of relying only on the LLM's training data, RAG retrieves relevant documents and includes them in the LLM's context window.

**Weakness that GraphRAG addresses:** Standard RAG retrieves by semantic similarity only — it can't reason across connections, detect contradictions, or trace facts through multi-hop paths.

---

### RAGAS
A framework for evaluating RAG pipelines on four metrics: faithfulness, answer relevancy, context precision, context recall. Uses an LLM as the judge.

**In this project:** `graphrag/evaluation/ragas_evaluator.py`. Samples 20% of queries automatically. Judge LLM: Groq (DeepSeek-V3 fallback).

---

### RDF (Resource Description Framework)
A W3C standard for representing knowledge as subject-predicate-object triples. The basis for semantic web technologies.

**Example:** `(FAA-AD-2024-01-02) -- (supersedes) --> (FAA-AD-2022-03-07)` as an RDF triple.

**In this project:** `scripts/export_rdf.py` exports the graph to Turtle format. `graphrag/graph/sparql_bridge.py` enables SPARQL queries over the export.

---

### Reranking
The process of re-scoring an initial set of retrieved documents with a more expensive but more accurate model. Trades query-time cost for precision.

**In this project:** Stage 3 — cross-encoder `ms-marco-MiniLM-L-6-v2`. Takes top 50 chunks from stages 1+2, re-ranks to top 20 for subsequent stages.

---

### Relation
A typed, directed connection between two entities in the knowledge graph. Extracted from text by the LLM.

**Example:** `(FAA) -[ISSUED]-> (FAA-AD-2024-01-02)`, `(FAA-AD-2024-01-02) -[APPLIES_TO]-> (Boeing 737-800)`.

**In this project:** `graphrag/core/models.py: Relation`. Stored as `(:Entity)-[:RELATES_TO {relation: "SUPERSEDES", ...}]->(:Entity)` in Neo4j. The `relation` property holds the semantic type; the `RELATES_TO` label is the physical label.

---

### RRF (Reciprocal Rank Fusion)
See **Fusion**.

---

## S

### Session context
Conversation history that influences subsequent queries in a multi-turn interaction. Allows the system to answer follow-up questions that reference prior turns.

**Example:** Turn 1: "What does FAA-AD-2024-01-02 require?" Turn 2: "When was it issued?" Without session context, Turn 2 has no referent for "it".

**In this project:** `graphrag/retrieval/session_context.py`. Stored in Redis with TTL. Enriches query prompts with relevant prior turns.

---

### SPARQL
A query language for RDF data, analogous to SQL for relational data or Cypher for property graphs.

**Example:**
```sparql
SELECT ?ad ?superseded
WHERE {
  ?ad a :AIRWORTHINESS_DIRECTIVE .
  ?ad :supersedes ?superseded .
}
```

**In this project:** `graphrag/graph/sparql_bridge.py` — runs SPARQL SELECT over Turtle-exported RDF using rdflib. Useful for semantic web integrations or OWL-RL reasoning output.

---

## T

### Taxonomy (type taxonomy)
A hierarchical classification of entity types — a tree where each type is a subtype of another. Enables query expansion and type-safe entity resolution.

**Example:**
```
CONCEPT
  └─ REGULATION
       └─ AIRWORTHINESS_DIRECTIVE
       └─ TYPE_CERTIFICATE
  └─ MAINTENANCE_PROCEDURE
       └─ SERVICE_BULLETIN
```

**In this project:** `graphrag/graph/type_taxonomy.py: TypeTaxonomy`. Loaded from base types + domain ontology pairs. Enables `expand_type("REGULATION")` to return all subtypes for inclusive queries.

---

### Temporal modeling (bitemporal)
Tracking two time dimensions for every fact: *valid time* (when the fact was true in the real world) and *transaction time* (when the system recorded the fact).

**Example:** An aircraft was IS_AIRWORTHY from 2024-01-15 to 2024-03-19 (valid time). This was recorded in the graph on 2024-01-16 (transaction time). When the compliance check on 2024-03-20 revokes airworthiness, the valid_to is set — but the transaction record of the original assertion remains.

**In this project:** `RELATES_TO` edges have `valid_from`, `valid_to`, `extracted_at` (valid time) and `recorded_at` (transaction time, set once on CREATE, never updated). Enables "as-of" queries: "what was the airworthiness status as of 2024-02-01?"

---

### Tenant
A logical isolation boundary in a multi-tenant deployment. All entities, relations, and queries are scoped to a tenant identifier.

**In this project:** Every Neo4j node and edge has a `tenant` property. The composite entity key is `(name, type, tenant)`. Default tenant: "default". Demo tenant: "aerospace".

---

### TransE
A knowledge graph embedding model that learns entity and relation embeddings such that `head + relation ≈ tail`. Used for link prediction — predicting likely but unobserved relations.

**Example:** Given many examples of `(country) -[CAPITAL_OF]-> (city)`, TransE learns that `embedding(country) + embedding(CAPITAL_OF) ≈ embedding(city)`. New countries can then have their capitals predicted.

**In this project:** `graphrag/graph/link_predictor.py`. Operates on entity embeddings from the graph. Used as an additional signal for entity resolution and relation completion.

---

## V

### Vector search
Retrieval by similarity in embedding space. The query is encoded into a vector, and documents are ranked by cosine similarity (or other distance metrics) to the query vector.

**In this project:** Stage 1. 3072d OpenAI embeddings (`text-embedding-3-large`), Neo4j HNSW index. See `graphrag/graph/neo4j_client.py: vector_search_chunks()`.

---

## W

### Wikidata linking
The process of connecting extracted entities to their corresponding entries in Wikidata (a structured knowledge base), enriching them with canonical identifiers and additional properties.

**Example:** Extracted entity "Boeing" (ORG) → Wikidata entity Q66 (The Boeing Company), enriched with headquarters location, founding date, industry classification.

**In this project:** `graphrag/graph/entity_linker.py`. Runs ad-hoc (not yet wired into the production ingestion pipeline — see `docs/roadmap.md`).

---

## Platform-specific terms

### GraphHealthSnapshot
A point-in-time record of all graph quality metrics — entity resolution quality, relation precision, contradiction rate, orphan rate, community coherence. Persisted as a node in Neo4j for trend tracking.

**In this project:** `graphrag/graph/graph_evaluator.py: persist_snapshot()`. Queried via `GET /kg/health/snapshot`.

---

### Conflict node
A Neo4j node `(:Conflict)` created when the contradiction detector finds a contradiction. Stores the type, involved entities, source documents, and resolution status.

**In this project:** Created by `graphrag/graph/contradiction_detector.py`. Viewed via `GET /kg/conflicts`.

---

### OntologyVersion node
A Neo4j node that snapshots the ontology schema at a point in time, including a hash of the type/relation definitions. Enables detecting schema drift across deployments.

**In this project:** Created by `graphrag/graph/ontology_registry.py: load()`.

---

### KPIEvent
A business metric record capturing query-level performance: latency, RAGAS scores, retrieval mode, model version.

**In this project:** `graphrag/core/models.py: KPIEvent`. Written by `graphrag/business_matrix/kpi_tracker.py`. Viewed via `GET /kpis/summary` and the dashboard.

---

## Performance metrics in this project

Beyond RAGAS, the platform tracks:

| Metric | Source | API |
|---|---|---|
| `latency_ms` | Every query | `GET /kpis/timeseries?metric=latency_ms` |
| `faithfulness` | RAGAS (20% sample) | `GET /kpis/summary` |
| `answer_relevancy` | RAGAS | `GET /kpis/summary` |
| `context_precision` | RAGAS | `GET /kpis/summary` |
| `context_recall` | RAGAS | `GET /kpis/summary` |
| `entity_resolution_quality` | GraphEvaluator | `GET /kg/health/snapshot` |
| `relation_precision` | GraphEvaluator | `GET /kg/health/snapshot` |
| `contradiction_rate` | GraphEvaluator | `GET /kg/health/snapshot` |
| `orphan_growth_rate` | GraphEvaluator | `GET /kg/health/snapshot` |
| `merge_split_error_proxy` | GraphEvaluator | `GET /kg/health/snapshot` |
| `community_coherence` | GraphEvaluator | `GET /kg/health/snapshot` |
| `brier_score` | CalibrationService | `graphrag/graph/confidence_calibration.py` |
| `confidence_distribution` | CalibrationService | Calibration curve (isotonic bins) |
| `gnn_score` | GNNScorer | Per-chunk in retrieval results |
| `final_score` | HybridRetriever | α·cross_encoder + β·gnn_score |
| `p95_latency` | Alert threshold | Dashboard / alerts |

The RAGAS metrics measure **answer quality**. The GraphEvaluator metrics measure **graph health** — independent dimensions that RAGAS cannot see. A graph with 40% orphan entities produces poor retrieval even if the LLM writes polished answers from the fragments it finds.
