# LinkedIn Knowledge Graph Group Research — 2026-08-26

## Executive decision

The accessible group corpus mostly confirms capabilities already present in this
platform. The one high-confidence gap worth implementing immediately was
**structural retrieval-trajectory evaluation**: measuring whether a query used
the right retrieval surfaces, found the expected evidence and graph edges, and
did so within a tool-call budget. That capability is now captured in every
non-cached retrieval result (when enabled), exposed through the evaluation API,
and consumed by the golden-set runner when a case declares structural
expectations.

No experimental database, ontology redesign, or reinforcement-learning loop was
introduced. Those proposals have higher operational risk or weaker independent
evidence than the deterministic capability implemented here.

## Research scope and completeness

- Group: `linkedin.com/groups/8659061/`, accessed through the authenticated UI.
- Accessible top-level posts recorded: **109**.
- Feed labels covered: **1 day to 3 months** old.
- Posts with technical-keyword matches: **102**.
- Posts containing external links: **83**.
- Posts with a collapsed-text affordance: **94**; their complete text was
  already present in LinkedIn's accessibility DOM and was captured there.
- Explicit crawl/inspection cycles: **17**.
- `Show more results` activations with no new top-level records: **5**.
- Primary sources inspected for shortlisted claims: **8**.
- Corpus status: **PARTIAL**.

The status is intentionally not “complete.” LinkedIn continued to expose an
enabled pagination control, but repeated activations returned no additional
records. Therefore 109 is the reproducible accessible corpus for this session,
not a claim about the group's entire history.

## Evidence policy

LinkedIn posts were treated as discovery leads, not as proof. Architecture or
benchmark decisions required a linked primary source and were discounted when
the evidence was a self-benchmark, preprint, illustrative validation, or a
platform-specific implementation. Duplicate themes were consolidated into one
finding.

Primary sources used in the decision include:

- [WorkSurface-Bench](https://github.com/haolpku/WorkSurface-Bench) for
  route/evidence/answer/efficiency evaluation across RAG, table, graph, and
  cross-surface tasks.
- [Graph-R1](https://arxiv.org/abs/2507.21892) for observable multi-turn graph
  retrieval and structural reward concepts; the work reports ICML 2026
  acceptance, but its RL training loop was not adopted.
- [AgentGL](https://arxiv.org/abs/2604.05846) for graph-native agent tools and
  topology-aware navigation; retained as research because it is preliminary.
- [K12-KGraph](https://github.com/haolpku/K12-KGraph) for schema-derived,
  evidence-carrying graph benchmark cases; its domain is specialized and its
  NeurIPS 2026 submission status is not production validation.
- [OntoBricks](https://github.com/databrickslabs/ontobricks) for governed
  ontology lifecycle patterns. Its useful controls already have equivalents in
  this platform.
- [Declarative Epistemic Contexts](https://arxiv.org/abs/2606.15246) for
  attributed claims and epistemic stance; kept as a prototype candidate.
- [Trait Nodes / 5GNF](https://arxiv.org/abs/2606.18297) for a graph-modelling
  workflow; evidence is illustrative, so no remodel was justified.
- [Slater](https://github.com/Hikari-Systems/slater) for a low-memory,
  Bolt-compatible graph-backend candidate; vendor self-benchmarks are not
  sufficient to change the production backend.

## Capability comparison

| Theme from corpus | Existing project evidence | Decision |
|---|---|---|
| Hybrid vector/text/graph retrieval and reranking | `local_search.py`, `hybrid_retriever.py`, GAT/GCN scorer, RRF, cross-encoder | Already implemented |
| Agentic multi-hop retrieval | `agentic_retriever.py`, `query_planner.py`, `adaptive_router.py` | Already implemented; add measurable trace |
| Identity resolution | alias registry, review queue, entity-resolution evaluator | Already implemented |
| OWL/RDF/SKOS/SHACL/SPARQL and inference | ontology registry, RDF bridge, SKOS projection, SHACL gate, Datalog-style forward chaining | Already implemented |
| Governed ontology lifecycle | proposals, migration, versioning, review and publication controls | Implemented differently; no OntoBricks dependency |
| Claims, findings and provenance | intelligence extraction plus claim-to-evidence graph | Already implemented |
| Temporal/geographic expansion | temporal query expansion, bitemporal graph, hierarchy support | Already implemented |
| Tables as first-class JSON-LD objects | semantic interchange and intelligence ingestion | Already implemented |
| Persistent agent/context memory | Context Graph decisions, episodes, observations, tool calls and policy versions | Already implemented |
| Structural route/evidence evaluation | Existing evals measured answers/citations but not the chosen multi-surface trajectory | **Implemented now** |
| Epistemic worlds/stances | Confidence, contradiction and source provenance exist; no general cognitive-world model | Prototype only |
| Alternative low-memory graph backend | Backend benchmark harness exists; no independent production proof for Slater | Benchmark only |
| RL-trained graph retrieval policy | Deterministic/adaptive routing exists; no stable reward dataset or training operations | Research only |

## Implemented now

1. `QueryResult` can carry a bounded `RetrievalTrajectory` with ordered steps,
   selected surfaces, evidence IDs, canonical graph edges, tool-call count,
   route reason, query class, and completion path.
2. Standard local/global/hybrid retrieval records its actual search result.
   Agentic retrieval records seed search, sub-searches, newly discovered
   evidence, and answer/synthesis completion. The hybrid-to-agentic path merges
   both parts into one ordered trace.
3. `POST /evaluation/retrieval-trajectory/score` provides deterministic scoring.
4. Golden cases may add `expected_surfaces`, `expected_evidence_ids`,
   `expected_graph_edges`, `tool_budget`, and `min_trajectory_score`. Cases
   without those fields preserve their old behavior.
5. The aggregate is an adapted structural score:

   `0.35 × answer + 0.30 × structural evidence + 0.25 × route F1 + 0.10 × efficiency`

   Structural evidence is evidence F1, averaged with graph-edge recall when a
   case declares expected graph edges. This is an adaptation of the published
   WorkSurface-Bench category weighting, not a claim of byte-for-byte benchmark
   compatibility.
6. The feature is controlled by `retrieval.trajectory_capture_enabled` and does
   not alter ranking or answer generation.

## Correctness defects found during baseline

- `Neo4jClient.get_relations_for_entity()` passed a literal
  `{temporal_filter}` to Cypher, so `as_of` did not apply the validity interval.
  The query is now interpolated and the existing regression test passes.
- The same latent interpolation defect existed in entity-neighbor expansion,
  multi-hop traversal, and relation-subgraph retrieval. In those paths it
  disabled combinations of valid-time, transaction-time, tenant-edge, hop-depth,
  and semantic-score fragments. All are now interpolated and protected by
  direct Cypher-shape regression tests.
- Metadata-schema registration silently accepted the body tenant `default` and
  overwrote it with the token tenant. It now uses the mandatory
  `assert_request_tenant` denial path, preserving fail-closed tenant isolation.
- The active Python environment lacked `openpyxl`, although it is already
  declared in `requirements/ingestion.txt`; this is an environment restoration
  issue, not an undeclared project dependency.

## Deferred and rejected work

- **Prototype:** attributed epistemic contexts layered onto the existing Claim
  model. Gate on a real query set where source stance changes the answer.
- **Benchmark:** Slater or another Bolt-compatible backend using the existing
  backend benchmark harness. Require independent correctness, backup/restore,
  bitemporal, tenancy, and failure-recovery evidence before adoption.
- **Research:** Graph-R1/AgentGL-style learned routing. First accumulate stable,
  expert-labelled trajectories; do not train on self-generated reward alone.
- **Rejected now:** remodel the production graph around trait nodes. The current
  typed Claim/Artifact/Source design is more explicit for provenance-bound RAG,
  and the cited work does not demonstrate migration value for this workload.

## Reproduction and limitations

The machine-readable finding set is in `research/linkedin_findings.json` and the
priorities are summarized in the root `ROADMAP.md`. LinkedIn may reorder or hide
posts, relative date labels are not immutable timestamps, and inaccessible post
comments were not treated as evidence. Primary-source status is recorded as of
2026-08-26 and should be rechecked before adopting deferred technologies.

## Verification

- Pre-change suite excluding the unavailable spreadsheet dependency:
  **1,691 passed, 8 skipped, 2 failed**. The two failures exposed the temporal
  relation and tenant-denial defects described above.
- Environment restored with the already-declared `openpyxl 3.1.5` dependency.
- Focused retrieval/Cypher guards: **23 passed**.
- Focused implementation and regression set: **40 passed**.
- Final deterministic suite: **1,712 passed, 7 skipped, 0 failed** in 600.39s.
- Changed-file Ruff check: **passed**.
- Graphify AST graph: refreshed with **7,970 nodes and 18,074 edges** using
  `--no-cluster`; community reclustering was skipped after the default update
  remained CPU-bound for more than 20 minutes. The code graph itself is current.

The seven skips are live-service tests that require external Neo4j/PostgreSQL
services; they are not failed or uncollected unit tests.
