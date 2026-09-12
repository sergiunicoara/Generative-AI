# Energy Asset & Maintenance Intelligence POC

This is a synthetic wind-farm demonstration. It shows how SAP-shaped asset and
work-order exports, Snowflake-shaped telemetry, and SharePoint-shaped technical
guidance can become RDF evidence for an advisory maintenance review.

## Scope and data ownership

The synthetic demo module owns its local RDF projection only. Source-shaped
records are fixtures, not live SAP, Snowflake, or SharePoint integrations.
The optional GraphDB store is an RDF serving copy loaded from the Turtle export;
it is not a second source of truth. Neo4j is optional and receives only a
governed, rebuildable RDF-derived read model; it never writes back to RDF.

## Run locally

```powershell
python scripts/create_energy_demo_sqlite.py
python scripts/ingest_r2rml.py --mapping ontology/mappings/energy-assets.r2rml.ttl --sqlite artifacts/energy-demo-sap.sqlite --tenant energy-demo --source-id synthetic-sap --validate-only
python scripts/run_energy_demo.py --export-turtle artifacts/energy-demo.ttl
python scripts/run_energy_demo.py --as-of 2026-05-01T00:00:00Z
python scripts/evaluate_energy_demo.py
python scripts/build_energy_evaluation_report.py
python scripts/project_energy_rdf_to_neo4j.py  # requires configured Neo4j
python scripts/run_energy_demo_e2e.py --output artifacts/energy-demo-e2e-report.json
python -m pytest tests/unit/test_energy_demo.py -q
```

`build_energy_evaluation_report.py` writes a versioned local evidence artifact
covering fixed-answer correctness, maintenance-evidence coverage, abstention,
tenant isolation, publication freshness, p95 local answer latency and RDF
materialisation throughput. Its claim policy explicitly limits those timing and
throughput observations to the small synthetic fixture.

The API is tenant-scoped and requires an authenticated token with tenant
`energy-demo` and scope `read`:

```text
GET /energy-demo/questions
GET /energy-demo/answer/maintenance_review
GET /energy-demo/answer/historical_state?as_of=2026-05-01T00:00:00Z
GET /energy-demo/rdf
GET /energy-demo/validation
GET /energy-demo/publication
GET /energy-demo/quarantine
POST /energy-demo/rollback?version_id=<id>   (requires scope `write`)
```

Use only the five fixed question IDs. This POC does not expose arbitrary client
SPARQL. The `maintenance_review.rq` query is version controlled under
`evals/energy_demo/sparql/` for technical inspection.

## SHACL as a publication gate

The vocabulary and validation contract are not maintained as two competing
models. `ontology/models/energy-asset-intelligence.yaml` is the canonical
semantic model; `python -m graphrag.semantic_model compile ...` deterministically
projects it into the OWL/RDFS ontology, SHACL shapes, Neo4j key constraints,
and machine-readable capability diagnostics. `make semantic-model-check`
detects hand edits or stale generated files.

This keeps the modelling decision independent of storage. OWL states domain
meaning under open-world semantics; SHACL rejects incomplete candidate RDF;
Neo4j DDL enforces the subset its schema supports. Required non-key properties,
abstract types, and relationship cardinalities that Neo4j cannot express are
reported rather than silently dropped. `SemanticMutationValidator`, injected
into the shared `GraphWriter` for compiled domains, rejects invalid nodes and
relationship endpoints before mutation. Maximum relationship counts still
need a same-transaction database check in a production connector to avoid a
check-then-write race.

The migration was introduced behind parity tests before the generated files
replaced their manually maintained predecessors. The checked tests parse the
generated Turtle with RDFLib, validate representative valid and invalid graphs
through the project's actual SHACL service, compare the established Energy
classes/properties, exercise compiler drift and source-location diagnostics,
and rerun the existing Energy publication and answer flow. Generated Neo4j
Cypher is tested structurally; it is not presented as proof of live database
enforcement.

The RDF graph is not served the moment it's built. Every time
`EnergyDemoService` builds a candidate graph (R2RML/RML mapping execution
plus the hand-written topology/bulletins), `graphrag/domains/energy/
publication.py`'s `DatasetPublisher` stages it, validates it against
`ontology/shapes/energy-asset-intelligence.shapes.ttl` (via the platform's
real `SHACLValidator`, not a separate ad hoc check), quarantines any record
that violates a shape (its own triples excluded, with the SHACL violation
messages recorded as the reason), and publishes the conformant remainder
as a new version. `GET /rdf` and every `/answer/*` question only ever see
that published graph. `GET /publication` reports the current version;
`GET /quarantine` lists whatever is currently quarantined (normally
empty — item 2's mapping-completion work, plus a decimal-literal-typing
fix this gate found live, made the real pipeline fully conformant).
`POST /rollback` restores an earlier version (the one just before current,
or a named `version_id`) as a new version — history is append-only and is
never rewritten or deleted.

`GET /validation` is a separate, older capability probe: it demonstrates
SHACL rejection against one fixed synthetic invalid record, not the live
published graph — kept for that narrower purpose, now also routed through
the real `SHACLValidator` instead of a bypassing `pyshacl.validate()` call.

## Optional Neo4j GraphRAG read model

`scripts/project_energy_rdf_to_neo4j.py` projects the **published Energy RDF
graph** into the platform's Neo4j `Entity` / `RELATES_TO` model for traversal
and GraphRAG retrieval. It preserves each resource's RDF IRI, selected Energy
type, literals and datatype metadata, source provenance, and trusted tenant.
The direction is deliberately one way: RDF remains the evidence source of
truth, while Neo4j can be dropped and rebuilt from the published RDF version.

The projector fails closed if it sees a blank node, an unsupported vocabulary
term, a repeated literal that needs a collection mapping, or an Energy resource
without a supported concrete type. This prevents a convenient but lossy RDF ↔
property-graph synchronization claim. It is a read-model projection, not a
bidirectional sync engine and not a replacement for RDF/SPARQL governance.

## GraphDB path

Start the optional RDF service with:

```powershell
docker compose -f compose.energy-demo.yaml up -d
```

GraphDB is exposed only on `127.0.0.1:7200`. The repository no longer needs to
be created by hand through the GraphDB UI first —
`graphrag.graph.triplestore.TripleStoreTarget.load()` now auto-provisions it
via GraphDB's `/rest/repositories` REST API (`ensure_namespace()`) before
loading, the same way it already did for Blazegraph namespaces:

```python
from graphrag.graph.triplestore import TripleStoreTarget

target = TripleStoreTarget("graphdb", "http://localhost:7200", repository="energy-demo")
await target.load(Path("artifacts/energy-demo.ttl").read_bytes())
```

Verified live end to end (2026-09) against `ontotext/graphdb:10.8.1` — the
tag `compose.energy-demo.yaml` now pins — including repository
auto-creation, load, query, and that data survives a container restart; see
`tests/e2e/test_live_graphdb.py`. That image runs unlicensed ("Product:
GRAPHDB_LITE ... Licensee: Freeware ... Expiry date: none") — **GraphDB
11.0+ requires a registered license file to start at all**, which is why an
earlier pin to `11.2.0` here would not have booted. No GraphDB credentials
are committed.

The Docker-backed recovery test exports the Energy dataset as Turtle through
a read-only SPARQL `CONSTRUCT`, loads it into a fresh GraphDB repository, and
reruns the same committed `evals/energy_demo/sparql/maintenance_review.rq`
query that `EnergyDemoService.answer()` uses. This is portable RDF *dataset*
recovery, not a GraphDB binary backup: repository configuration, users,
inference caches, and operational state remain the responsibility of the
selected GraphDB backup procedure.

## Demo narrative

For one repeatable proof of the complete local workflow, run
`python scripts/run_energy_demo_e2e.py`. It writes the source-shaped SQLite
fixture and Turtle export, then asserts R2RML/RML materialisation, explicit
`WindTurbine` typing, SHACL publication, the committed SPARQL answer, evidence
provenance, historical revision selection, the no-evidence boundary, tenant
isolation, and the RDF-to-Neo4j batch projection contract. Add
`--live-neo4j` only when a configured Neo4j instance should receive the
rebuildable read model. The JSON report is suitable for attaching to the demo
or using as a regression artifact.

1. Ask which assets need review. WT-01 is identified using a 96 C measurement,
   open work order WO-9001, and bulletin MFG-GBX-17-R2.
2. Show the evidence records and the source/revision timestamps.
3. Ask about the revision: R2 supersedes R1 and changes the threshold from 90 C
   to 85 C.
4. Query 2026-05-01 to recover R1 as the authoritative historical guidance.
5. Ask for incomplete assessments and show that WT-04 through WT-10 receive no
   maintenance conclusion.
6. Use a different tenant token and show that the POC returns no evidence.

## Governed maintenance workflow

The POC also exposes a small operational lifecycle for `WO-9001`. Source RDF
continues to describe the SAP-shaped work-order evidence; workflow changes are
separate, append-only transition records: `review_required` → `approved` →
`completed`. Each transition records its actor, timestamp and reason, is
tenant-scoped, and requires the API's `write` scope. The lifecycle is available
at `GET /energy-demo/work-orders/WO-9001/lifecycle`; a transition is requested
through `POST /energy-demo/work-orders/WO-9001/transition`.

This is a local, in-memory demonstration of governed graph operations. It does
not update SAP or control equipment.

## Limitations

Both mapping files are genuinely executed, not just parsed as contracts.
`ontology/mappings/energy-assets.r2rml.ttl` runs via
`graphrag/ingestion/r2rml_rdf.py`'s `materialize_r2rml()` (also independently
verified live against GraphDB/Blazegraph, see `tests/e2e/`).
`ontology/mappings/energy-observations.rml.ttl` runs via a small,
self-built, deliberately narrow RML executor,
`graphrag/ingestion/rml_rdf.py`'s `materialize_rml()` -- narrow because no
RML processor or JSONPath library is installed anywhere in this repo, and
this mapping's shape (a top-level JSON array, direct-key references) needs
neither. Both raise on any construct outside their documented supported
subset rather than silently approximating it.

What's still hand-written, and stays that way deliberately:
`graphrag/domains/energy/demo.py`'s turbine/gearbox topology (structural
scaffolding, not sourced data) and the two manufacturer-bulletin document
revisions (narrative document content, not naturally a mapping target).

The sample is deliberately small. It demonstrates contracts, provenance,
revision handling, and permission behavior; it makes no enterprise-scale claim.
Its running publication path remains RDF-native. Neo4j is an optional,
rebuildable GraphRAG read model, governed by the same canonical semantic model.
See [the production-readiness preflight](../energy-production-readiness-preflight.md)
for the deployment evidence still required outside this repository.
