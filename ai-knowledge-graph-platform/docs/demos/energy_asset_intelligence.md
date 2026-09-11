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

GraphDB is exposed only on `127.0.0.1:7200`. Create a local repository named
`energy-demo` through the GraphDB UI, then load `artifacts/energy-demo.ttl`
using the repository's Graph Store HTTP endpoint. The repository contains a
generic GraphDB URL builder in `graphrag.graph.triplestore.TripleStoreTarget`.

The GraphDB image and live repository path have not been verified in this
environment. Check GraphDB image availability and license terms before client
use. No GraphDB credentials are committed.

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

The R2RML mapping is executable with the repository's supported mapping subset.
The RML mapping is a version-controlled contract, but no maintained RML
processor is bundled or executed by this POC. Add an approved processor and a
live integration test before representing RML processing as implemented.

The sample is deliberately small. It demonstrates contracts, provenance,
revision handling, and permission behavior; it makes no enterprise-scale claim.
