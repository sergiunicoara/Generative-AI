# Energy Asset & Maintenance Intelligence POC

This is a synthetic wind-farm demonstration. It shows how SAP-shaped asset and
work-order exports, Snowflake-shaped telemetry, and SharePoint-shaped technical
guidance can become RDF evidence for an advisory maintenance review.

## Scope and data ownership

The synthetic demo module owns its local RDF projection only. Source-shaped
records are fixtures, not live SAP, Snowflake, or SharePoint integrations.
Neo4j remains unchanged and receives no POC writes. The optional GraphDB store
is an RDF serving copy loaded from the Turtle export; it is not a second source
of truth.

## Run locally

```powershell
python scripts/create_energy_demo_sqlite.py
python scripts/ingest_r2rml.py --mapping ontology/mappings/energy-assets.r2rml.ttl --sqlite artifacts/energy-demo-sap.sqlite --tenant energy-demo --source-id synthetic-sap --validate-only
python scripts/run_energy_demo.py --export-turtle artifacts/energy-demo.ttl
python scripts/run_energy_demo.py --as-of 2026-05-01T00:00:00Z
python scripts/evaluate_energy_demo.py
python -m pytest tests/unit/test_energy_demo.py -q
```

The API is tenant-scoped and requires an authenticated token with tenant
`energy-demo` and scope `read`:

```text
GET /energy-demo/questions
GET /energy-demo/answer/maintenance_review
GET /energy-demo/answer/historical_state?as_of=2026-05-01T00:00:00Z
GET /energy-demo/rdf
GET /energy-demo/validation
```

Use only the five fixed question IDs. This POC does not expose arbitrary client
SPARQL. The `maintenance_review.rq` query is version controlled under
`evals/energy_demo/sparql/` for technical inspection.

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

## Demo narrative

1. Ask which assets need review. WT-01 is identified using a 96 C measurement,
   open work order WO-9001, and bulletin MFG-GBX-17-R2.
2. Show the evidence records and the source/revision timestamps.
3. Ask about the revision: R2 supersedes R1 and changes the threshold from 90 C
   to 85 C.
4. Query 2026-05-01 to recover R1 as the authoritative historical guidance.
5. Ask for incomplete assessments and show that WT-04 through WT-10 receive no
   maintenance conclusion.
6. Use a different tenant token and show that the POC returns no evidence.

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
