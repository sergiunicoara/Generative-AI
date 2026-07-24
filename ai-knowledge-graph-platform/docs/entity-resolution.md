# Entity Resolution

Entity resolution (ER) is the process of determining that multiple surface forms
refer to the same real-world entity, and merging them into a single canonical node.

Without ER, the graph fragments: "SpaceX", "Space Exploration Technologies Corp.",
and "Space Exploration Corp." become three separate nodes with no edges between them.
Retrieval, GNN propagation, and community detection all degrade because the graph
topology no longer reflects real-world relationships.

---

## The 4-Stage Pipeline

Every entity extracted from a document passes through four resolution stages before
a Neo4j MERGE is issued. Stages are applied in order; the first match wins.

```
Extracted entity name (raw LLM output)
            │
            ▼
   ┌─────────────────────────────┐
   │  Stage 1: Exact/Normalized  │  O(1) hash lookup
   │  Match                      │
   └──────────────┬──────────────┘
                  │ no match
                  ▼
   ┌─────────────────────────────┐
   │  Stage 2: Fuzzy Match       │  rapidfuzz ratio ≥ 85
   │  (Levenshtein-based)        │
   └──────────────┬──────────────┘
                  │ no match
                  ▼
   ┌─────────────────────────────┐
   │  Stage 3: Embedding         │  cosine similarity ≥ 0.92
   │  Similarity                 │  on entity.embedding (3072d)
   └──────────────┬──────────────┘
                  │ no match
                  ▼
   ┌─────────────────────────────┐
   │  Stage 4: New Entity        │  MERGE into Neo4j
   │  Creation                   │  register in alias cache
   └─────────────────────────────┘
```

---

## Stage 1: Exact / Normalized Match

The alias registry maintains an in-memory hash map:

```
normalized_alias → (canonical_name, canonical_type)
```

Normalization is deterministic: lowercase → strip punctuation → collapse whitespace.

```python
_normalize("Apple, Inc.")           # → "apple inc"
_normalize("Space Exploration Corp") # → "space exploration corp"
_normalize("Elon  Musk")            # → "elon musk"
```

The same normalization is applied to both the stored key (at load time) and the
lookup key (at resolution time) — guaranteeing that write and read keys always match.

**Aliases registered in Neo4j persist across restarts.** The registry is loaded
from the `(Entity)<-[:ALIAS_OF]-(Alias)` subgraph on worker startup and refreshed
after each ingestion batch.

**Example:**

```
Input:  "Space Exploration Technologies"
Key:    "space exploration technologies"
Match:  → ("SpaceX", "ORG")   ✓  resolved in O(1)
```

---

## Stage 2: Fuzzy Match

When no exact normalized match exists, the registry computes a
`rapidfuzz.fuzz.ratio` (Levenshtein edit distance score, 0–100) between the
normalized input and every stored alias key.

A match is accepted when `score ≥ 85` (configurable via `alias_fuzzy_threshold`
in `settings.yml`).

**Example:**

```
Input:  "Spacex Inc"    → normalized: "spacex inc"
Stored: "spacex"        → score: 86   ✓  match
```

Fuzzy matching catches common OCR errors, abbreviation variants, and minor
transcription differences that normalized matching misses.

**Trade-off:** O(n) over all stored aliases. Acceptable because the alias
registry is an in-process dict (not a DB query) and typical tenants have fewer
than 50,000 distinct entities.

---

## Stage 3: Embedding Similarity

When fuzzy matching also fails, the pipeline checks whether the incoming entity's
3072-dimensional OpenAI embedding (text-embedding-3-large) is very close to an existing entity's embedding.

```cypher
CALL db.index.vector.queryNodes('entity_embeddings', 5, $embedding)
YIELD node AS e, score
WHERE e.type = $type
  AND e.tenant = $tenant
  AND e.name <> $exclude_name
  AND score >= 0.92
RETURN e.name, e.type, score
ORDER BY score DESC
LIMIT 1
```

Cosine similarity ≥ 0.92 indicates near-identical semantic content. This catches
aliases that share no surface-level string similarity:
- "Musk" ↔ "Elon Musk" (when both appear in the same tenant context)
- "FDA" ↔ "Food and Drug Administration"
- "§ 21 CFR Part 820" ↔ "21 CFR 820 Quality System Regulation"

When a match is found, the incoming name is **registered as an alias** of the
canonical entity and persisted to Neo4j for future exact-match lookup.

---

## Stage 4: New Entity Creation

If all three stages fail, the entity is genuinely new. A Neo4j `MERGE` is issued
on `(name, type, tenant)` composite key, and the canonical name is registered in
the in-process alias cache for future lookups within the same ingestion batch.

**Collision guard:** because the merge key is `(name, type, tenant)`, two
same-named entities with different real-world meanings (e.g. "Apple" the
company vs. "Apple" the fruit) could previously merge into one node whenever
the LLM assigned them the same generic type. This is now mitigated two ways:
(a) the extraction prompt is tenant-aware and pulls entity types from each
tenant's domain ontology (`config/ontologies/*.yml`) instead of one shared
generic type list, so same-named entities in different domains are more
likely to get distinct types; (b) `merge_entity()` computes embedding cosine
similarity between the existing and incoming entity data and logs a warning
on likely collision.

---

## Multi-Tenant Isolation

The alias registry is **per-tenant**. An entity named "Acme Corp" in tenant
`aerospace` and in tenant `finance` are independent canonical nodes with independent
alias sets. Resolution never crosses tenant boundaries.

```
Tenant: aerospace    SpaceX → ("SpaceX", "ORG")
Tenant: finance      SpaceX → (not registered — new entity)
```

---

## Confidence and Provenance

When an alias match is found, the new variant is persisted to Neo4j with:
- `confidence`: the match quality score (fuzzy ratio / cosine similarity)
- `source_doc`: the document that introduced the variant
- `normalized`: the normalized form used as the lookup key

This provides a full audit trail of alias decisions, which is critical for
regulated domains where data lineage must be explainable.

---

## Bayesian Confidence Accumulation

When the same `(source, relation, target)` triple is seen in multiple documents,
confidence **accumulates** rather than overwriting:

```
new_confidence = 1 − (1 − c1) × (1 − c2)
```

A relation corroborated across three independent documents with confidence 0.8
each converges to `1 − 0.2³ = 0.992`. This is correct Bayesian belief fusion
under the assumption of independent evidence.

---

## Contradiction Detection as ER Complement

After entity resolution, contradiction detection scans for cases where the **same
entities** are linked by semantically incompatible assertions from different source
documents:

- **Directional reversal:** A→B and B→A for the same relation type
- **Exclusive states:** entity simultaneously `IS_ACTIVE` and `IS_DEPRECATED`
- **Functional violation:** `CEO_OF` with two different targets (one person can't
  lead two organisations simultaneously in the same time window)
- **Positive/negative pair:** RELATES_TO and NEGATIVE_RELATES_TO coexist for the
  same triple

The former `multi_source` strategy (same `(A, rel, B)` triple asserted by two
non-superseding documents) has been retired as a conflict type — an edge is a
single triple, so two documents agreeing on it is corroboration, not a
contradiction. It's now tracked as a trust signal (`independent_source_count`
/ `corroborated_edge_rate`) rather than an open conflict.

Conflicts are persisted as `Conflict` nodes with `status: "open"` and surfaced via
the `/corrections/list-conflicts` API for manual or authority-based resolution.
Retrieval also checks for open conflicts on entities in the result set
(`ContradictionDetector.get_open_conflicts_for_entities`) and the answer prompt
is warned when context includes a disputed fact, gated by
`retrieval.conflict_annotation_enabled` (default on).

**Implementation:** `graphrag/graph/contradiction_detector.py`,
`graphrag/graph/contradiction_strategies.py`
