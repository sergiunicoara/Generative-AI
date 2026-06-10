# Ontology Model

This document describes the formal knowledge model underpinning the platform: entity type
hierarchy, relation schema, domain/range constraints, inference rules, and the design
decisions that balance formal semantics with pragmatic LLM-extracted knowledge.

---

## 1. Entity Type Hierarchy

Entities follow a strict type taxonomy stored as `EntityType` nodes connected by
`SUBCLASS_OF` edges in Neo4j. This enables **type-aware query expansion** — a search
for `AGENT` automatically includes all subtypes without the caller enumerating them.

```
Thing
├── Agent
│   ├── PERSON          — individual human actors
│   └── ORG             — companies, agencies, regulatory bodies
│       └── REGULATOR   — can be added at runtime per domain
│
├── Artifact
│   └── PRODUCT         — physical or digital products, systems, components
│
├── Place
│   └── LOCATION        — geographic entities (city, country, region)
│
├── Temporal
│   └── EVENT           — time-bounded occurrences
│
└── Abstract
    └── CONCEPT         — ideas, standards, regulations, processes
```

The hierarchy is **extensible at runtime** without code changes:

```python
await taxonomy.register_subclass("AIRWORTHINESS_DIRECTIVE", parent="CONCEPT")
await taxonomy.register_subclass("AIRCRAFT_TYPE",           parent="PRODUCT")
await taxonomy.register_subclass("REGULATOR",               parent="ORG")
```

Domain-specific type trees are defined in `config/ontologies/` and loaded at startup.
See `config/ontologies/aerospace_regulatory.yml` for an aerospace example.

**Implementation:** `graphrag/graph/type_taxonomy.py`  
Provides `expand_type()`, `get_ancestors()`, `least_common_ancestor()`, and
`query_by_type()` — the last of which injects the full subtype set into Cypher
`WHERE e.type IN $types` filters.

---

## 2. Relation Schema — Domain/Range Constraints

Every `RELATES_TO` edge carries a `relation` property (UPPER_SNAKE_CASE). The
`OntologyRegistry` enforces **domain/range constraints** on every write:

| Relation | Domain | Range | Notes |
|---|---|---|---|
| `CEO_OF` | PERSON | ORG | Functional: one target per source (per doc scope) |
| `FOUNDED` | PERSON, ORG | ORG, PRODUCT | |
| `FOUNDED_BY` | ORG, PRODUCT | PERSON | Inverse of FOUNDED (auto-inferred) |
| `OWNS` | PERSON, ORG | ORG, PRODUCT | |
| `ACQUIRED` | ORG | ORG, PRODUCT | |
| `MANUFACTURES` | ORG | PRODUCT | Functional |
| `LAUNCHED` | ORG, PERSON | PRODUCT | |
| `WORKS_AT` | PERSON | ORG | Inverse `EMPLOYS` auto-inferred |
| `PART_OF` | PRODUCT, ORG, EVENT | ORG | |
| `USES` | ORG, PRODUCT, PERSON | PRODUCT | |
| `LOCATED_IN` | ORG, PERSON, EVENT | LOCATION | Composition rule: `A LOCATED_IN B, B PART_OF C → A LOCATED_IN C` |
| `RELATED_TO` | any | any | Open fallback; symmetric |
| `SUPERSEDES` | CONCEPT | CONCEPT | Domain-specific; used in regulatory graphs |
| `MANDATED_BY` | CONCEPT | ORG | Regulatory authority chain |
| `APPLIES_TO` | CONCEPT | PRODUCT, ORG | Compliance scope |

**Constraint enforcement flow:**

```
Extractor output (LLM JSON)
      │
      ▼
OntologyRegistry.validate_relation_triplet(src_type, relation, tgt_type)
      │
      ├─ Unknown relation name → normalised to UPPER_SNAKE_CASE or mapped via migration_map
      ├─ Valid domain/range → pass through
      ├─ Invalid domain/range → downgrade to RELATED_TO + record schema event in Neo4j
      └─ Deprecated name (e.g. IS_CEO) → silently migrated to canonical (CEO_OF)
```

Constraint violations are **non-blocking** (downgraded, not dropped) because LLM
extractors produce inconsistent type attribution. The schema event log lets you track
drift over time and tighten constraints as the extraction pipeline matures.

**Implementation:** `graphrag/graph/ontology_registry.py`

---

## 3. Relation Migration Map

Deprecated or renamed relations are auto-migrated at ingestion time without requiring
a schema migration of existing edges:

```yaml
# config/settings.yml → ontology.migration_map
migration_map:
  IS_CEO:            CEO_OF
  FOUNDED_BY_PERSON: FOUNDED_BY
  PART_OF_ORG:       PART_OF
  HAS_LOCATION:      LOCATED_IN
```

This allows the extraction prompt to evolve without breaking graph queries that
rely on canonical relation names.

---

## 4. Inference Rules

The `ForwardChainingEngine` applies **Datalog-style forward-chaining rules** to derive
implicit edges. Rules run post-ingestion and write `RELATES_TO` edges tagged with
`source_type = "inferred"` and a reference to the rule that fired.

### Built-in rules

| Rule | Pattern | Derived |
|---|---|---|
| `subsidiary_transitivity` | A `SUBSIDIARY_OF` B, B `SUBSIDIARY_OF` C | A `SUBSIDIARY_OF` C (depth ≤ 3, decay 0.85/hop) |
| `related_to_symmetry` | A `RELATED_TO` B | B `RELATED_TO` A |
| `works_at_inverse` | A `WORKS_AT` B | B `EMPLOYS` A |
| `founded_inverse` | A `FOUNDED` B | B `FOUNDED_BY` A |
| `located_in_part_of` | A `LOCATED_IN` B, B `PART_OF` C | A `LOCATED_IN` C (decay 0.8) |

### Domain-specific rules (config)

Additional rules are defined in `config/settings.yml` under `inference.rules`:

```yaml
inference:
  rules:
    - name: supersedes_transitivity
      relation: SUPERSEDES
      rule_type: transitivity
      max_depth: 5
      confidence_decay: 0.95

    - name: mandated_by_inverse
      relation: MANDATED_BY
      rule_type: inverse
      derived_relation: MANDATES
```

**Confidence propagation:** inferred edge confidence = product of premise confidences
× decay^depth. This ensures inferred facts carry lower confidence than directly
asserted ones and are ranked accordingly by the GNN scorer.

**Implementation:** `graphrag/graph/inference_engine.py`

---

## 5. Temporal Modeling (Bitemporal)

The knowledge graph models **two independent time axes**:

| Dimension | Fields | Meaning |
|---|---|---|
| Valid time (VT) | `valid_from`, `valid_to` on Entity/Relation | When the fact was true in the real world |
| Transaction time (TT) | `recorded_at` (immutable ON CREATE) | When the fact was recorded in the database |

This enables the key bitemporal query: **"What did the system know on date TT about
the world as it was at time VT?"** — essential for regulatory compliance where
retroactive corrections must not silently overwrite historical snapshots.

```cypher
-- What entities were valid in Jan 2024, as known before the March 2024 re-ingestion?
MATCH (e:Entity)
WHERE e.valid_from  <= datetime("2024-01-31")
  AND (e.valid_to IS NULL OR e.valid_to >= datetime("2024-01-01"))
  AND e.recorded_at <= datetime("2024-03-01")
RETURN e.name, e.type
```

**Implementation:** `graphrag/graph/bitemporal.py` — `BitemporalStore` with
`as_of_entities()`, `as_of_edges()`, `transaction_diff()`, `time_travel_report()`.

---

## 6. Confidence Model

Edge confidence is **Bayesian-accumulated**, not last-write-wins:

```
new_confidence = 1 − (1 − existing_confidence) × (1 − incoming_confidence)
```

When two documents independently assert the same relation, their confidences
**reinforce** rather than overwrite. A relation seen in 3 independent sources
with confidence 0.8 each yields:
```
1 − (0.2)(0.2)(0.2) = 0.992
```

Confidence degrades along inferred paths (decay per hop) and via the document
authority system (superseded documents receive a configurable penalty). The
calibration service tracks whether model-reported confidence matches empirical
accuracy and provides isotonic correction curves.

**Sources of confidence:**
- LLM extractor: `[0, 1]` (clamped; raw values from LLM JSON)
- Bayesian accumulation across multiple documents
- Authority penalty from superseded document edges
- Temporal decay: `confidence × exp(−ln2 / half_life_days × age_days)`
- Propagated path confidence: product along multi-hop chain

---

## 7. Design Decisions

### Why Neo4j over a triple store (RDF store)?

Triple stores optimise for `(s, p, o)` pattern matching under open-world semantics.
This graph requires **property-rich edges** (confidence, source_doc_ids, authority,
valid_from/valid_to, recorded_at, chunk_span) and **vector indexes** on entity nodes
for ANN retrieval. Neo4j's property graph model handles both naturally; attaching
first-class properties to RDF predicates requires reification, which introduces
significant query complexity.

RDF serialisation (Turtle export) is available via `scripts/export_rdf.py` for
interoperability with OWL tooling and SPARQL consumers.

### Why Leiden communities over modularity maximisation?

Leiden produces communities with **resolution guarantees** — no internally
disconnected communities, bounded community sizes. Modularity maximisation
(Louvain) can produce arbitrarily disconnected communities that mislead global
search. The staleness system rebuilds communities only when entity drift exceeds
a threshold, avoiding unnecessary compute on stable subgraphs.

### Why forward-chaining over backward-chaining inference?

Backward-chaining (Prolog-style) evaluates rules at query time — expensive for
a pipeline that serves user queries with a 3-second P95 SLO. Forward-chaining
pre-materialises derived facts post-ingestion when compute budget is available.
The trade-off: inferred facts can become stale if underlying asserted edges are
retracted. A `source_type = "inferred"` tag lets the system identify and recompute
stale inferences after contradiction resolution.

### Formal semantics vs LLM pragmatics

LLM extractors are non-deterministic and produce noisy type attributions. The
ontology enforces constraints **non-blockingly**: violations are downgraded and
logged rather than rejected. This prevents ingestion failure cascades while
building a precise schema violation audit that feeds back into prompt improvement.
The tension is intentional: rigid schema enforcement optimises for graph quality
over throughput; the current balance favours throughput with a clear upgrade path
to stricter enforcement as prompt quality matures.
