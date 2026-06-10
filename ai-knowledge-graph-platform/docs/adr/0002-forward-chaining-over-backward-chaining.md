# ADR 0002 — Forward-chaining Inference over Backward-chaining (Prolog-style)

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024-Q2 |
| **Deciders** | Platform architect |

---

## Context

The knowledge graph requires inference to derive implicit facts — transitivity
(A `SUBSIDIARY_OF` B, B `SUBSIDIARY_OF` C → A `SUBSIDIARY_OF` C), inverses
(A `WORKS_AT` B → B `EMPLOYS` A), and compositions (A `LOCATED_IN` B, B `PART_OF` C
→ A `LOCATED_IN` C).

Two principal inference strategies exist:

**Backward-chaining (query-time):** Prolog-style — rules are evaluated lazily at
query time. No pre-materialisation; derived facts are computed on demand. Used in:
Prolog, Datalog engines (Soufflé), OWL reasoners (HermiT, Pellet).

**Forward-chaining (materialisation):** Rules fire eagerly post-ingestion; derived
facts are written as first-class edges to the graph. Used in: production rule
engines (Drools), graph reasoning pipelines.

---

## Decision

**Use forward-chaining (materialisation).** Derived edges are written to Neo4j as
`RELATES_TO` with `source_type = "inferred"` and a `rule_name` tag. The
`ForwardChainingEngine` runs to fixpoint post-ingestion, scoped to the affected
document's entity subgraph for efficiency.

---

## Reasons

### 1. Query latency SLO

The system targets P95 query latency < 3 seconds. Backward-chaining evaluates rules
at query time — for a graph with 1M edges and 20 inference rules, this adds O(depth ×
rules) to every query. Forward-chaining amortises the compute over ingestion time
(where latency tolerance is higher) and keeps query paths flat.

### 2. Inferred facts are graph-searchable

Pre-materialised inferred edges participate in vector ANN searches, BM25 index,
multi-hop traversal, and GNN scoring exactly like asserted edges. Backward-chaining
produces virtual facts that exist only in the reasoner — they cannot be indexed,
cannot appear in GNN adjacency matrices, and cannot be returned by Neo4j pattern
queries.

### 3. Auditable provenance

Each inferred edge carries `source_type = "inferred"`, `rule_name`, and `confidence`
(product of premise confidences × decay per hop). This makes inferred facts
**distinguishable from asserted facts** at query time and **auditable** for regulatory
domains that require explainability of derived conclusions.

### 4. Stale inference is detectable and repairable

When an asserted edge is retracted or corrected, downstream inferred edges become
stale. Because inferred edges are first-class graph objects tagged with `rule_name`,
they can be identified and recomputed:

```cypher
MATCH ()-[r:RELATES_TO {source_type: 'inferred', rule_name: $rule}]->()
DELETE r
```

Backward-chaining has no persistent state — stale inferences are invisible.

### 5. Rules are domain-configurable without code changes

Inference rules are defined in YAML (`config/ontologies/*.yml → inference_rules`),
loaded by `domain_ontology.build_inference_rules_from_ontology()`, and passed to
`ForwardChainingEngine`. Adding a domain-specific transitivity rule for `SUPERSEDES`
in the aerospace ontology requires no Python changes.

---

## Consequences

**Positive:**
- Query latency unaffected by rule count or graph size
- Inferred facts participate in all retrieval stages (vector, BM25, GNN, multi-hop)
- Auditable, deletable, recomputable
- Domain rules via config YAML

**Negative:**
- Inferred facts become stale after retraction of premises — requires recomputation
- Materialisation cost at ingestion time (proportional to affected subgraph × rule count)
- Open-world inference (facts not in graph are unknown, not provably false) is
  approximated rather than guaranteed — OWL DL completeness is not claimed

**Mitigation:** `ForwardChainingEngine.run_for_document(doc_id)` scopes inference to
the entity subgraph of the just-ingested document, limiting materialisation cost to
the affected neighbourhood rather than the full graph.
