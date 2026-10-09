# Graph validation and the publication gate

Status: **implemented and unit-verified** (2026-10-09). The Neo4j-side pieces
(quarantine store Cypher, `run_read` sessions) are exercised with mocks locally;
live-database verification is CI e2e only.

## Lifecycle

Every ingestion batch moves through:

```
EXTRACTED -> STAGED -> VALIDATED -> PUBLISHED
                    \-> REJECTED   (a document-level BLOCKING rule failed)
```

The state, plus counts of BLOCKING / WARNING / INFORMATIONAL findings and the
number of quarantined records, is stored on the run manifest under
`stage_metrics.publication` (`:IngestionRunManifest.stage_metrics_json`).

Design (plan decision D1): validation runs **in memory, before any graph
write**. STAGED is a property of the in-flight batch, not of graph nodes, so
no retrieval query needs a publication filter and an invalid record is never
written, not merely hidden.

| Path | Behaviour on a BLOCKING record |
|---|---|
| Document ingestion (`IngestionAgent.write`) | The record is removed from the batch and quarantined. Valid records publish. Relations whose endpoint was rejected are cascaded (`REL-REF-002`). A BLOCKING *document* rule rejects the batch: nothing is written, the manifest is `failed`, and the document is quarantined. |
| Relational ingestion (`RelationalGraphIngestor.ingest` / `ingest_incremental`) | All-or-nothing, as before (SHACL already raised): the run raises `ValueError` naming the rule ids, after quarantining the offending records. |
| After alias resolution (`GraphWriter.write_relations`) | Relations whose *resolved* endpoint types violate domain/range (`REL-DOMAIN-002`) or whose endpoint is missing (`REL-REF-001`) were previously dropped with only a log line. They are now collected and quarantined by both paths. |

Toggle: `config/settings.yml` → `ingestion.publication_gate_enabled` (default
`true`). `false` restores the legacy write-then-validate path. Post-write
`validate_and_check_cycles` still runs either way.

## Severities

- **BLOCKING** — the record is quarantined and never published.
- **WARNING** — published and reported (metrics, manifest, report).
- **INFORMATIONAL** — published and reported.

New rules with no prior enforcement ship as WARNING so the gate does not
reject data that ingested before (ontology type, property values, extracted
domain/range, relation casing that the writer normalises).

## Rule catalogue

Source of truth: `graphrag/graph/validation/rules.py`. `shacl_ref` maps a rule
to the equivalent constraint in `ontology/shapes/ingestion.shapes.ttl`.

| Rule | Severity | Checks |
|---|---|---|
| DOC-TENANT-001 | BLOCKING | document tenant non-empty |
| DOC-TENANT-002 | BLOCKING | every chunk has its document's tenant |
| DOC-PROV-001 | BLOCKING | document has a source identifier (filename / source_path / content_hash) |
| DOC-TEMPORAL-001 | BLOCKING | document `valid_from <= valid_to` |
| DOC-SUPERSEDES-001 | WARNING | every SUPERSEDES target exists **in the same tenant** (other tenants are never queried, so the report cannot leak them) |
| ENT-REQ-001 / 002 | BLOCKING | name / type non-empty (`ing:MappedEntityShape`) |
| ENT-TENANT-001 | BLOCKING | entity tenant equals batch tenant (`"default"` = unset; the writer stamps it) |
| ENT-CONF-001 | BLOCKING | confidence finite, in [0, 1] |
| ENT-ID-001 | BLOCKING | one id is not used for two different (name, type) identities |
| ENT-ER-001 | BLOCKING | resolution status is known; canonical name/type set together |
| ENT-TYPE-001 | WARNING | type is in the tenant ontology (only when the registry is loaded) |
| ENT-PROP-001 / 002 | WARNING | required semantic property / allowed values (`PropertySchemaValidator` rules) |
| ENT-ORPHAN-001 | INFORMATIONAL | entity takes part in no relation in its chunk |
| REL-REQ-001 | BLOCKING | relation type non-empty |
| REL-REQ-002 | WARNING | relation type is UPPER_SNAKE_CASE |
| REL-REF-001 | BLOCKING | both endpoints are entities of the batch (dangling reference) |
| REL-REF-002 | BLOCKING | neither endpoint was rejected (cascade) |
| REL-SELF-001 | BLOCKING | not a self-loop (same id, or same name+type) |
| REL-CONF-001 | BLOCKING | confidence in [0, 1], weight finite |
| REL-TEMPORAL-001 | BLOCKING | `valid_from <= valid_to` |
| REL-STATE-001 | BLOCKING | `confidence_state` ∈ ASSERTED, INFERRED, DISPUTED, RETRACTED, APPROVED |
| REL-DOMAIN-001 | WARNING | extracted endpoint types fit the relation's domain/range |
| REL-DOMAIN-002 | BLOCKING | *resolved* endpoint types fit domain/range (enforced by `GraphWriter`) |
| SEM-VIOLATION-001 | BLOCKING | `SemanticMutationValidator` (datatypes, cardinality) — only when one is injected |

Cardinality: expressible only through an injected `SemanticMutationValidator`
(`PublicationGate(semantic_validator=...)`); production does not inject one yet.

## Quarantine

`:QuarantinedRecord {tenant, id}` (unique), linked from its
`:IngestionRunManifest` via `[:QUARANTINED]`. It keeps `rule_ids`, `messages`,
`payload_json` (the full record; relations also carry their endpoint names and
types), `source`, `document_id`, `document_key`, `chunk_id`, `status`
(`QUARANTINED` | `RESOLVED`), `attempts`, timestamps. It is not an `:Entity`
and has no edges into the published graph, so no retrieval path can see it.
Ids are deterministic per (tenant, document, record, chunk): re-ingesting the
same bad record updates the existing row.

API (tenant from the token, never from the request):

| Endpoint | Scope | |
|---|---|---|
| `GET /corrections/quarantine/records?status=&rule_id=&limit=` | read | list |
| `GET /corrections/quarantine/summary` | read | counts by status, source, open rule |
| `POST /corrections/quarantine/records/{id}/retry` `{"payload": {...}}` | write | re-validate the corrected (or stored) payload; if it passes, publish through `GraphWriter` (alias resolution and domain/range still apply) and mark `RESOLVED` |

Retry refuses documents (re-ingest the corrected source) and records that are
not `QUARANTINED`. A corrected payload cannot move a record into another
tenant (`ENT-TENANT-001`). A relation retry also requires both endpoints to be
published entities (read-only check).

## Read-only guarantee

Validation and quarantine-read queries go through `run_read_only`, which
rejects any query containing a write or procedure clause (`CREATE`, `MERGE`,
`SET`, `DELETE`, `REMOVE`, `DROP`, `FOREACH`, `LOAD CSV`, `CALL {}`,
`CALL db.|dbms.|apoc.`) before sending it, and runs it via
`Neo4jClient.run_read` (a `READ_ACCESS` session the server enforces).

## Reports and CI

`ValidationReport` gives `to_dict()` (counts by rule, severity, tenant,
source) and `to_junit_xml()` (one testcase per rule; BLOCKING hits are
failures). CI writes pytest JUnit XML (`reports/junit-*.xml`) and uploads it
as the `junit-reports` artifact. SARIF is not produced: nothing in CI consumes it.

## Metrics

Prometheus labels are bounded enums only:
`graphrag_validation_violations_total{rule_id,severity}`,
`graphrag_quarantined_records_total{record_kind}`,
`graphrag_publication_batches_total{outcome}`,
`graphrag_quarantine_retries_total{outcome}`. Tenant, document and entity
identifiers go to structured logs (`publication_gate.*`).

## Limitations

- The extractor's own SHACL check (when the ontology registry is loaded) still
  drops a whole chunk on violation before the gate sees it, recording only an
  `OntologyEvent`. Moving that into per-record quarantine is future work.
- Schema version on quarantine rows is `null` until Phase 2 (schema registry).
- Quarantine Cypher has not been run against a live Neo4j locally (no Docker
  daemon); it is covered by mocks here and needs the CI e2e job.
