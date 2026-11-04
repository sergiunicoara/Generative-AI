# Ingestion gaps and graph completeness

Two read-only checks for operating the graph as a data asset. Neither writes, calls an LLM, or adds load on a source
system. Both are implemented and tested (unit tests, plus live Neo4j tests in `tests/e2e/test_live_gaps_and_completeness.py`).
Neither has been run against production-scale data, and no throughput figure is claimed.

## Ingestion-gap check (missed events)

`ContentSyncService.reconcile` covers one direction: items that vanished from the source are tombstoned. A missed
webhook or a failed ingest causes the other: the source has an item the graph lacks, holds an older revision of, or
holds without chunks. `graphrag/enterprise/gaps.py` finds those.

* **Input:** an *inventory* of `{external_id: source_version}` (or a bare id list). This is the listing a connector
  already produces for `reconcile`: ids and versions, no content, so no extra load on the source.
* **Graph side:** one read-only Cypher query over the tenant's documents for that `source_id`.
* **Output:** counts and capped samples per kind, a `coverage` share, and the complete `repair_ids` list to re-fetch.

| kind | meaning | in `repair_ids` |
| --- | --- | --- |
| `missing` | source lists it, graph has no document | yes |
| `stale` | graph holds an older `source_version` | yes |
| `restored_in_source` | graph tombstoned it, source lists it again | yes |
| `not_ingested` | document exists but status is not `done` or it has no chunks | yes |
| `duplicate_in_graph` | more than one graph document for one external id | no (investigate) |
| `unverifiable_version` | graph has no `source_version`, so a changed revision cannot be detected | no |
| `would_tombstone` | live in graph, absent from the source: what `reconcile` would soft-delete | no |

`coverage` is the share of inventory items present, current and fully ingested. With a bare id list, staleness is not
judged. The check never repairs anything; the connector decides what to re-fetch.

```bash
python scripts/ingestion_gaps.py --tenant acme --source-id sharepoint --inventory inventory.json
```

API: `POST /sync/{source_id}/gaps` (read scope) with `{"inventory": {"id": "version"}}`.

## Completeness report

`graphrag/graph/completeness.py` answers "is the stored graph populated enough to serve retrieval?", separate from
`correctness.py` (is what is stored valid?) and from answer-quality evals. Each measure is a share with numerator and
denominator, `None` when the denominator is zero. Quarantined entities, deleted documents and non-current relations
are excluded; scope is one tenant, optionally one `source_id`.

* Documents: ingested, chunked, linked to entities, with external id, with source version.
* Chunks: embedded, with text, linked to entities.
* Entities: described, embedded, typed, with chunk evidence, connected to another entity.
* Relations (current only): named, with confidence, sourced, dated.

`incomplete` lists measures under 0.95, lowest first. A low share is a prompt to look: some fields are legitimately
absent for some sources (for example a source with no versions).

```bash
python scripts/graph_completeness.py --tenant aerospace [--source-id S]
```

API: `GET /governance/completeness?source_id=...` (read scope).

## Not covered

* No measurement on a large graph, and no ingestion throughput or load test.
* The gap check depends on the connector supplying a faithful inventory; a wrong listing gives a wrong report.
* Completeness is structural. It does not say whether an extracted fact is true (see the correctness report and the
  golden set for that).
* No scheduled run or alert is wired; run the scripts or call the routes.

## Data expectations and build lineage (2026-11-04)

Relational mappings can declare expectations checked on the rows read, before
SHACL and before any write; a failed expectation blocks the build and the
previous publication stays in place:

```python
RelationalGraphMapping(..., expectations=[
    Expectation(id="suppliers-present", table="suppliers", kind="row_count", min_rows=1),
    Expectation(id="supplier-ids", table="suppliers", kind="unique", column="id"),
    Expectation(id="fresh-prices", table="prices", kind="freshness", column="updated_at", max_age_hours=24),
])
```

Kinds: `schema` (columns present), `not_null`, `unique`, `freshness` (newest
timestamp within `max_age_hours`) and `row_count` (`min_rows` / `max_rows`).
With a `BuildLedger` (`scripts/ingest_r2rml.py` uses one), every build is an
`IngestionBuild` node with its input tables (rows, SHA-256), code version,
mapping version, expectation results, outcome and output counts; a published
build is linked `PUBLISHED` to the source document, so
`BuildLedger.lineage(tenant, entity_id)` resolves an entity to the latest build
that published it. `scripts/export_catalog.py openlineage` turns builds into
OpenLineage run events and `scripts/export_catalog.py atlas` turns published
business definitions into Apache Atlas glossary terms (dry run unless `--push`).
Verified offline only (SQLite fixtures, fakes, mocked HTTP).
