# Versioned schema registry

Status: **implemented and unit-verified** (2026-10-09). The registry's Cypher
(single active version, `CONFORMS_TO` repointing, retained history, legacy-row
handling) is proven by `tests/e2e/test_live_schema_registry.py`, which runs in
CI only: no Docker daemon was available locally.

## Model

The existing `:OntologyVersion` node is the schema version. It now also carries
the `:SchemaVersion` label, so `OntologyProposal -[:PROPOSED_FOR]->` links,
`OntologyEvent` history and every stored id keep working.

```
(:Dataset {tenant, id, name})-[:CONFORMS_TO]->(:OntologyVersion:SchemaVersion)
(:SchemaVersion)-[:USES_VALIDATION_PROFILE]->(:ValidationProfile {tenant, id, name, version, content_hash})
(:SchemaVersion)-[:IMPORTS]->(:SchemaVersion)
```

`SchemaVersion` properties: `tenant, dataset_id, id, name, version,
content_hash, schema_hash (legacy 16-char prefix), source_uri, active,
created_at, loaded_at, activated_at, deactivated_at, entity_types`.

Everything carries `tenant` and every query filters on it. Unique keys:
`(tenant, dataset_id, content_hash)` on versions, `(tenant, id)` on datasets,
`(tenant, content_hash)` on profiles.

**Dataset.** There was no dataset concept; the default is one dataset per
tenant (`dataset_id = tenant`). `ontology.schema_registry.datasets:
{tenant: dataset}` maps a tenant to a different dataset. Tenants are never
merged: two tenants with a byte-identical ontology have separate versions.

## Content hash

`sha256` (full 64 hex) of canonical JSON over everything that changes what is
valid or what a term means: allowed entity types; domain/range rules; the
built-in relation rules; vocabulary; the migration map; the ontology YAML's
`ontology` block (id, version, status, ...) and its `type_hierarchy`,
`relation_rules`, `inference_rules`, `exclusive_state_pairs`,
`functional_relations`, `vocabulary` sections; and the validation-profile hash.
Unrelated YAML keys (e.g. `authority_levels`) are deliberately excluded. The
**validation profile** hash covers `ontology/shapes/ingestion.shapes.ttl` and
`export.shapes.ttl`, so a shape change is a new schema version.

*Behaviour change:* the previous 16-char hash covered only types and domain
rules, so edits to vocabulary, inference rules, the YAML version or shapes were
invisible. The new hash is wider, so the first load after upgrade registers a
new version for every tenant (see Migration).

## Lifecycle

- At most one version per (tenant, dataset) is `active`. Activating a version
  deactivates the others (retained for reproducibility) and repoints
  `CONFORMS_TO`, in one statement.
- `OntologyRegistry.load()` computes the identity from the files and calls
  `SchemaRegistry.register_and_activate`: idempotent for the active hash; a new
  hash is a **version change**; a previously known hash is a **rollback**
  (reactivation). Either records a `schema_drift` `OntologyEvent`, a warning
  log and `graphrag_schema_drift_events_total{outcome}`.
- `mode: enforce` (below) instead refuses to load a schema that is not the
  active registered version.
- Operations (all tenant-scoped): `SchemaRegistry.activate / deactivate /
  rollback / add_import / versions / get_active / check`, exposed by
  `scripts/schema_registry.py show|drift|activate|rollback|deactivate --tenant T`.
  Rollback reactivates the most recently deactivated version. `activate` refuses
  a version id that does not belong to the caller's tenant and dataset.

## Drift detection and unknown schemas

| Where | What happens |
|---|---|
| Ingestion (`OntologyRegistry.load`, once per tenant per process) | `auto`: drift recorded, new version activated (legacy behaviour). `enforce`: `UnknownSchemaError` (hash never registered) or `SchemaDriftError` (registered but inactive) before anything is written. A dataset with no active version bootstraps. |
| API startup (`api/main.py`) and ingestion/combined workers | `startup_drift_check()`: for each tenant with an active version, compare the files to it; log `schema_registry.startup_drift`, count `startup_drift`. Read-only, never fatal. |
| CLI | `scripts/schema_registry.py drift --tenant T` exits 1 on drift (usable in CI/deploy gates). |

Configuration (`config/settings.yml`, `ontology.schema_registry`):
`mode: auto | enforce`, `datasets: {}`. Default is `auto`, which keeps today's
"edit the YAML, restart" workflow. Use `enforce` where a schema change must be
an explicit, reviewed activation.

## Provenance

The active version label `name@version#hash12` is recorded on:

- `QueryResult.schema_version`, the context-graph trace manifest
  (`ontology_version`, previously the constant `platform/v1`), and the **answer
  cache key** (so a schema change cannot serve an answer produced under another
  schema). Lookup is cached for 30 s, time-boxed to 2 s, and backs off for 30 s
  after a failure; it degrades to `platform/v1`, never failing a query.
- `Document.schema_version` / `schema_content_hash`, stamped at ingestion.
- Validation reports (`ValidationReport.schema_version`), the run manifest
  (`stage_metrics.publication.schema_version`) and `QuarantinedRecord.schema_version`.

## Migration

1. Deploy. `schema.cypher` is idempotent and adds three constraints and one
   index (`schema_version_key`, `dataset_key`, `validation_profile_key`,
   `schema_version_active`); workers apply them at startup, or run
   `python scripts/init_neo4j.py`.
2. On first load per tenant a new-style version is registered and activated;
   the pre-existing `OntologyVersion` rows (no `content_hash`/`dataset_id`) are
   deactivated and retained, and keep their proposals and events. No backfill
   is needed. A `schema_drift` event with outcome `version_change` is expected
   exactly once per tenant after upgrade.
3. Answer caches keyed on `platform/v1` miss once, then repopulate.

## Limitations

- Not yet re-hashed on a YAML edit without a restart: `ensure_ontology_schema`
  loads once per tenant per process; `startup_drift_check` and
  `scripts/schema_registry.py drift` report the difference.
- `apply_ontology_migration` still mutates the in-memory migration map without
  creating a version; the change is captured on the next load's hash.
- `Entity` and `Relation` nodes are not stamped; the Document, manifest and
  answer are.
- `IMPORTS` has an API but nothing in the ontology YAML declares imports yet.
- `startup_drift_check` lists tenants with an active version (a system-level
  read); a tenant with no registered version is not reported until its first load.
