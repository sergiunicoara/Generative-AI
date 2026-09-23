# ADR 0012 — GraphBackend Protocol and a Gremlin (Neptune/Cosmos DB) Backend

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-09-23 |
| **Deciders** | Platform architect |

---

## Context

ADR-0001 chose Neo4j as the property-graph store. That decision stands
unchanged here — `Neo4jClient` (`graphrag/graph/neo4j_client.py`) remains
the platform's only runtime backend, with ~60 async methods and no
abstraction layer between it and its ~100 direct callers (`get_neo4j()`).

Separately, Gremlin (Apache TinkerPop) is the query interface for two
property-graph stores Neo4j does not speak: **AWS Neptune** and **Azure
Cosmos DB's Gremlin API**. `docs/roadmap.md` already scoped Gremlin as an
"interoperability target, not an additional source of truth" — the same
posture ADR-0001's SPARQL addenda take toward Blazegraph/GraphDB/Stardog.
This ADR is that posture applied to the property-graph side.

A full swap-in replacement for `Neo4jClient` — translating all ~60 methods
and migrating ~100 call sites — was explicitly rejected as out of scope
for a single session; see `graph_backend.py`'s module docstring for the
concrete method count this ADR covers instead.

---

## Decision

**Add `GraphBackend`** (`graphrag/graph/graph_backend.py`) — a
`@runtime_checkable` `Protocol` covering 11 methods (`close`,
`entity_exists`, `merge_entity`, `merge_mentions`, `merge_relation`,
`merge_entities_batch`, `merge_mentions_batch`, `get_entity_neighbors`,
`get_relations_for_entity`, `get_all_entities`, `get_all_relations`) —
entity/relation CRUD, 1-hop retrieval, and the two batch-write methods
that matter for real ingestion volume. `Neo4jClient` satisfies this
structurally, with no code changes to it.

**Implement `GremlinBackend`** (`graphrag/graph/gremlin_client.py`)
against that Protocol, targeting Neptune/Cosmos DB's Gremlin API via
`gremlinpython`. Wired into a real caller via `scripts/query_gremlin.py`
(a smoke-check CLI, `make gremlin-query`) and `gremlin_source_from_env()`
(`GREMLIN_URL`/`GREMLIN_USERNAME`/`GREMLIN_PASSWORD`, mirroring
`triplestore.py`'s `remote_sparql_source_from_env()`).

No `get_graph_backend()` accessor and no migration of any `get_neo4j()`
call site — this does not change what the running platform actually reads
from or writes to.

---

## What's deliberately NOT covered

`Neo4jClient` has ~48 other methods with no Gremlin equivalent: schema/
corpus-lifecycle admin, document/chunk/structured-data ingestion,
`merge_relations_batch`, community detection, vector/BM25 search,
multi-hop traversal (`get_multihop_chunks`), and PageRank/GDS-dependent
analytics. None of these are part of `GraphBackend`. This is the same
honesty convention `triplestore.py` uses for Neptune's SPARQL `load()`
gap: documented, not silently omitted.

Within the 9 original methods: `get_entity_neighbors`/
`get_relations_for_entity` reject (`NotImplementedError`) rather than
silently ignore `as_of`/`transaction_at` temporal filters — no verified
Gremlin translation exists for them. `merge_entities_batch` never computes
`prior_similarity` (Neo4j's version uses `vector.similarity.cosine`, with
no portable TinkerPop equivalent verified here). Both batch methods are
Python loops over the single-item methods, not a single server-side
`UNWIND` the way `Neo4jClient`'s are.

---

## Verification status

Every `GremlinBackend` traversal is unit-tested against a mocked
low-level boundary (`tests/unit/test_gremlin_client.py`) and additionally
**live-verified (2026-09-23) against a real `tinkerpop/gremlin-server:3.7.2`
container** (`tests/e2e/test_live_gremlin.py`, skips cleanly without
Docker) — full entity/relation lifecycle including batch writes, Bayesian
confidence accumulation checked numerically (0.8, 0.5 → 0.9), and the
`NotImplementedError` rejections.

**Not verified**: a real Neptune or Cosmos DB endpoint. Both are
TinkerPop-compliant but each has its own documented deviations from the
reference server (step support gaps, property-cardinality differences) —
no such infrastructure was available in this environment. Treat this the
same way ADR-0001's addendum treats Stardog/RDFox/Virtuoso: a documented,
carefully-reasoned starting point, not a verified claim against those two
specific vendors.

---

## Consequences

**Positive:**
- A real, tested, live-verified path to Neptune/Cosmos DB exists for the
  retrieval-critical subset, without touching the platform's runtime
  backend at all.
- `GraphBackend` gives future backend work (or a genuine migration
  decision, if one is ever made) a real interface to extend instead of a
  blank slate.

**Negative:**
- `GraphBackend` is a second, smaller surface to keep in sync with
  `Neo4jClient`'s core CRUD/retrieval shape if that shape changes.
- `GremlinBackend`'s two-step (read-then-write) upsert for
  `merge_entity`/`merge_relation` is not atomic the way Cypher's single
  `MERGE ... ON CREATE ... ON MATCH` is — a documented, small race window
  against a concurrent writer that does not exist on the Neo4j side.

**Mitigation:** `GremlinBackend` is additive and inert by default —
nothing imports it unless `GREMLIN_URL` is configured or
`scripts/query_gremlin.py` is invoked directly. A deployment that never
configures Gremlin is completely unaffected, the same posture
`remote_sparql_source_from_env()` takes for the SPARQL side.
