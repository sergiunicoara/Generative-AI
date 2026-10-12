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

API: `POST /enterprise/sync/{source_id}/gaps` (read scope) with `{"inventory": {"id": "version"}}`. The route prefix
follows how the enterprise router is mounted.

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

API: `GET /enterprise/governance/completeness?source_id=...` (read scope).

## Not covered

* No measurement on a large graph, and no ingestion throughput or load test.
* The gap check depends on the connector supplying a faithful inventory; a wrong listing gives a wrong report.
* Completeness is structural. It does not say whether an extracted fact is true (see the correctness report and the
  golden set for that).
* No scheduled run or alert is wired; run the scripts or call the routes.
