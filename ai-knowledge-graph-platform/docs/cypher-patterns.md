# Cypher Patterns

**These are illustrative simplifications, not verbatim production queries.**
Every pattern below is faithful to the *shape* of the reasoning it explains,
but the real implementations add caps, quarantine filters, bitemporal
boundaries, tenant scoping, and semantic-ranking branches this doc omits for
readability. Where a pattern diverges from the real query in a way that
matters operationally, the function pointer below says so explicitly — read
the cited function for the exact current Cypher before copy-pasting into a
production incident.

Each pattern addresses a specific retrieval, reasoning, or compliance
challenge that arises in multi-document enterprise knowledge graphs.

---

## 1. Multi-hop graph traversal (bridging cross-document facts)

The standard retrieval path traces `Chunk → Entity → RELATES_TO* → Entity → Chunk`
to find evidence distributed across separate documents.

```cypher
-- Find all chunks reachable from seed chunks up to `hops` RELATES_TO steps,
-- weighted by path confidence and path length.
-- Used in: graphrag/graph/neo4j_client.py → get_multihop_chunks()
-- Real query differs: traversal is UNDIRECTED (-[:RELATES_TO*1..hops]-, no
-- arrow), has a per-seed LIMIT plus an outer total LIMIT, quarantine-filters
-- every node on the path (not just the far entity), and optionally blends
-- path score with vector.similarity.cosine when a query embedding is given.

MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {tenant: $tenant})
WHERE c.id IN $seed_chunk_ids

MATCH path = (e)-[:RELATES_TO*1..{hops}]->(e2:Entity {tenant: $tenant})
WHERE ALL(r IN relationships(path) WHERE r.tenant = $tenant)
  AND NOT e2.quarantined = true

WITH c, e2,
     length(path)                                           AS path_length,
     reduce(conf = 1.0, r IN relationships(path) |
         conf * coalesce(r.confidence, 1.0))                AS path_confidence

MATCH (c2:Chunk)-[:MENTIONS]->(e2)
WHERE c2.id NOT IN $seed_chunk_ids

RETURN DISTINCT
    c2.id                                                   AS chunk_id,
    c2.text                                                 AS text,
    e2.name                                                 AS via_entity,
    path_length,
    path_confidence,
    (path_confidence / toFloat(path_length))                AS path_score
ORDER BY path_score DESC
LIMIT $top_k
```

**Why this works:** confidence decays multiplicatively along each hop (product of edge
confidences). Dividing by path length penalises indirect evidence, so a high-confidence
1-hop result ranks above a weak 3-hop result.

---

## 2. Bitemporal as-of query (point-in-time reconstruction)

Answer: *"What did the system know on date TT about the world as it was at time VT?"*
Essential for regulatory compliance where retroactive corrections must not silently
overwrite the state of knowledge at a prior date.

```cypher
-- Reconstruct entity graph as-of valid_time VT, as recorded before transaction_time TT.
-- VT  = when the fact was true in the real world (user-specified)
-- TT  = when the fact was recorded in the database (audit cut-off)
-- Used in: graphrag/graph/bitemporal.py → as_of_entities()
-- Real query differs: valid_to bound is exclusive (e.valid_to > $vt, not
-- >=), and both this and the edge query below filter out quarantined
-- entities/edges (NOT e.quarantined = true) — not shown here.

MATCH (e:Entity)
WHERE e.tenant = $tenant
  AND (e.valid_from  IS NULL OR e.valid_from  <= $valid_time)
  AND (e.valid_to    IS NULL OR e.valid_to    >= $valid_time)
  AND  e.recorded_at <= $transaction_time
RETURN e.name AS name, e.type AS type,
       e.valid_from AS valid_from, e.valid_to AS valid_to,
       e.recorded_at AS recorded_at
ORDER BY e.name
```

## 2a. Explicit document-link traversal

Use only source-observed links for link-dependent multi-hop retrieval. The
application additionally injects the shared ACL predicate for `source`, `link`,
and `target` before returning target chunk text.

Simplified from `get_linked_document_chunks()`
(`graphrag/graph/neo4j_client.py`) — the real query also applies
`valid_at`/`transaction_at` bitemporal filtering, ranks semantically via
`query_embedding`/`vector.similarity.cosine` when supplied, and orders by
`semantic_score DESC, link.observed_at DESC` (not just `observed_at DESC`,
and not a hardcoded `<= datetime()`).

```cypher
UNWIND $seed_chunk_ids AS seed_id
MATCH (seed:Chunk {id: seed_id, tenant: $tenant})-[:PART_OF]->
      (source:Document {tenant: $tenant})
      -[link:LINKS_TO {tenant: $tenant}]->
      (target:Document {tenant: $tenant})
MATCH (target_chunk:Chunk {tenant: $tenant})-[:PART_OF]->(target)
WHERE coalesce(source.is_deleted, false) = false
  AND coalesce(target.is_deleted, false) = false
  AND link.recorded_at <= datetime()
RETURN target_chunk.id AS chunk_id, target_chunk.text AS text,
       source.filename AS link_source, target.filename AS link_target,
       link.target_url AS target_url, link.anchor_text AS anchor_text,
       link.observed_at AS observed_at
ORDER BY link.observed_at DESC
LIMIT $top_k
```

Implemented by `Neo4jClient.get_linked_document_chunks()`. Unresolved source
observations live in `DocumentLinkReference` until a same-tenant target arrives.

```cypher
-- Corresponding edge query — same bitemporal filter applied to RELATES_TO.
-- Used in: graphrag/graph/bitemporal.py → as_of_edges()

MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
WHERE r.tenant = $tenant
  AND (r.valid_from  IS NULL OR r.valid_from  <= $valid_time)
  AND (r.valid_to    IS NULL OR r.valid_to    >= $valid_time)
  AND  r.recorded_at <= $transaction_time
RETURN s.name AS src, t.name AS tgt,
       r.relation AS relation, r.confidence AS confidence,
       r.recorded_at AS recorded_at
```

---

## 3. Transitive supersession closure (regulatory authority resolution)

Answer: *"Which document is the current authority on this component / regulation?"*
Uses the `SUPERSEDES` chain to find the most recent governing document, following
transitivity materialised by the `supersedes_transitivity` inference rule.

```cypher
-- Find the current governing document for a given subject (component, part number, etc.)
-- by following SUPERSEDES chains to their root (no superseding document exists above them).

MATCH (current_doc:Document {id: $doc_id})
OPTIONAL MATCH path = (current_doc)-[:SUPERSEDES*1..10]->(older_doc:Document)
WITH current_doc, collect(older_doc.id) AS superseded_docs

-- Verify nothing supersedes current_doc (it is the current authority)
OPTIONAL MATCH (newer:Document)-[:SUPERSEDES*1..1]->(current_doc)
WITH current_doc, superseded_docs, count(newer) AS n_superseding
WHERE n_superseding = 0

RETURN current_doc.id       AS current_authority,
       current_doc.filename  AS filename,
       superseded_docs        AS supersedes_history,
       current_doc.authority_level AS authority_level
```

```cypher
-- Transitive closure query (used after inference materialises SUPERSEDES*):
-- Find the effective confidence of a fact, accounting for document supersession penalty.

MATCH (s:Entity {tenant: $tenant})-[r:RELATES_TO {relation: $relation}]->(t:Entity {tenant: $tenant})
MATCH (d:Document {id: r.source_doc_id})
OPTIONAL MATCH (newer:Document)-[:SUPERSEDES*1..]->(d)
WITH r, d, count(newer) AS n_superseding_docs
RETURN r.confidence * CASE WHEN n_superseding_docs > 0
                            THEN $superseded_penalty   -- e.g. 0.5
                            ELSE 1.0 END               AS effective_confidence,
       d.id AS source_doc,
       n_superseding_docs
ORDER BY effective_confidence DESC
```

---

## 4. Contradiction scan (functional-violation detection)

Detect entities that violate functional (many-to-one) relation constraints — e.g.
a part number with two different manufacturers, or an AD with two conflicting compliance
requirements. Used by `ContradictionDetector._detect_functional_violations()`.

```cypher
-- Find entities with multiple distinct targets for a functional relation.
-- "Functional" means: exactly one target per source per time window.
-- APOC-free: uses reduce() to flatten list-of-lists without external plugins.

MATCH (s:Entity)-[r:RELATES_TO {relation: $rel}]->(t:Entity)
WHERE r.source_doc_ids IS NOT NULL
  AND s.tenant = $tenant
  -- Real query in _detect_functional_violations() always filters tenant
  -- unconditionally — there is no 'default' bypass; shown correctly here.

WITH s.name AS src,
     collect(DISTINCT t.name)  AS targets,
     reduce(acc = [], lst IN collect(r.source_doc_ids) | acc + lst) AS all_docs
WHERE size(targets) > 1   -- functional violation: multiple targets

OPTIONAL MATCH (c:Conflict {
    src: src, relation: $rel,
    conflict_type: 'functional_violation',
    status: 'open'
})
WITH src, targets, all_docs, count(c) AS existing_conflicts
WHERE existing_conflicts = 0   -- not already flagged

RETURN src, targets, all_docs AS docs
LIMIT $scan_limit
```

---

## 5. Community-based global search (Leiden hierarchy query)

Retrieve the most semantically relevant community summaries for a question, used
in `GlobalSearch` for broad "what does the corpus know about X?" queries that
require synthesising across multiple documents.

```cypher
-- Vector ANN search over community summary embeddings, filtered by tenant and
-- excluding fallback communities (connected-components quality signal).
-- Used in: graphrag/graph/neo4j_client.py → vector_search_communities()

CALL db.index.vector.queryNodes(
    'community_embeddings',
    $top_k,
    $query_embedding
) YIELD node AS c, score

WHERE c.tenant = $tenant
-- Fallback-community exclusion ('[fallback:' summaries) is NOT in Cypher —
-- it happens in Python, in global_search.py, after this query returns.
-- The real vector_search_communities() also has four separate query
-- branches (filtered-vector-index vs. over-fetch, bitemporal vs. plain) to
-- work around a tenant-starvation bug (tasks/lessons.md A146) — this is the
-- simplest of the four.

RETURN c.id        AS community_id,
       c.summary   AS summary,
       c.level     AS level,
       c.member_count AS member_count,
       score        AS similarity
ORDER BY score DESC
```

The Leiden algorithm produces hierarchical communities at configurable resolution.
Level 0 = fine-grained clusters; higher levels = broader thematic groupings.
The default global search path retrieves top-k community summaries and places a
bounded context directly into final synthesis. The legacy map-reduce pattern,
which runs one LLM call per community, remains available only for controlled
ablation.

---

## 6. Entity resolution audit trail

Trace why two entity names were merged — useful for debugging alias resolution
decisions and regulatory data lineage requirements.

```cypher
-- Show the full alias chain for a canonical entity:
-- which raw document names were resolved to this canonical entity, by whom, and
-- with what confidence?

MATCH (e:Entity {name: $canonical_name, tenant: $tenant})
OPTIONAL MATCH (a:Alias)-[:ALIAS_OF]->(e)
RETURN e.name         AS canonical_name,
       e.type         AS canonical_type,
       collect({
           alias:       a.value,
           normalised:  a.normalized,
           source_doc:  a.source_doc,
           confidence:  a.confidence,
           created_at:  a.created_at
       })              AS aliases
ORDER BY e.name
```

```cypher
-- Find all entities that were NOT resolved to a canonical — i.e. potential
-- duplicates that escaped the 4-stage resolution pipeline.
-- Useful for data quality review and alias registry warm-up.

MATCH (e1:Entity {tenant: $tenant}), (e2:Entity {tenant: $tenant})
WHERE e1.id < e2.id
  AND e1.type = e2.type
  AND e1.name <> e2.name
  AND NOT (e1)<-[:ALIAS_OF]-()   -- e1 has no aliases (not a canonical with known variants)

WITH e1, e2,
     vector.similarity.cosine(e1.embedding, e2.embedding) AS sim
WHERE sim >= $similarity_threshold   -- e.g. 0.90
-- Uses the built-in vector.similarity.cosine, not the Graph Data Science
-- plugin's gds.similarity.cosine — every actual cosine-similarity call site
-- in this codebase (neo4j_client.py, alias_registry.py) uses the built-in,
-- consistent with this doc's own APOC/plugin-free philosophy (Pattern 4).

RETURN e1.name AS name_a, e2.name AS name_b,
       e1.type AS type, round(sim, 4) AS cosine_similarity
ORDER BY sim DESC
LIMIT 50
```
