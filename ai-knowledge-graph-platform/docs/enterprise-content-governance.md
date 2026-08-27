# Enterprise content governance

This platform has a provider-neutral content plane for enterprise repositories.
The built-in SharePoint connector adapts Microsoft Graph delta responses into
the API contracts below, while ingestion, extraction, graph writes and
retrieval remain identical for every source.

## Permission-aware retrieval

Each `Document` stores a normalised access policy:

- `tenant`: every authenticated caller in the document tenant may retrieve it;
- `restricted`: `allow_principals` is checked against `user:<subject>` and
  `group:<group>` identities from signed claims;
- explicit `deny_principals` always win;
- a restricted policy with unknown ACL state, or a policy requiring groups when
  the token has no resolved `groups` claim, is denied.

Set `access_control.enabled: true` only once ACL metadata has been loaded for a
source. At that point document/chunk vector and BM25 retrieval are filtered in
Neo4j, and answer-cache keys include the entitlement fingerprint. Community
summaries, graph expansion, and direct MCP graph fact/entity tools are withheld
until derived graph artifacts have their own ACL materialisation. This is
intentional denial on uncertainty, not a partial filtering claim.

## Metadata governance

`metadata_envelope` is the three-tier source contract:

1. Universal envelope: collection, source system, source/external ID and URL,
   version, content type, classification, effective-from/effective-to dates.
2. Governed collection tier: `collection_metadata`, validated against a
   versioned `CollectionMetadataSchema`.
3. Open discovery tier: bounded `discovery_metadata` for fields that have not
   yet been promoted into a collection schema.

Register a schema through `POST /governance/schemas`, then monitor field and ACL
coverage through `GET /governance/coverage`. Set
`metadata_governance.require_active_collection_schema: true` to reject content
without an active collection/version contract.

## Synchronisation plane

Use `POST /sync/{source_id}/changes` for webhook or delta batches. An upsert is
published through the normal ingestion queue; a delete tombstones the matching
`(tenant, source_id, external_id)` document without physically erasing evidence.
The service persists the latest delta cursor and a `ContentSyncRun` audit node.

Call `POST /sync/{source_id}/reconcile` with external IDs from a periodic full
source scan. Missing IDs are tombstoned, and the source receives its next review
time. `GET /sync/due-full-reviews` exposes sources due for an external scheduler
or connector worker. This supports Graph API delta polling and scheduled full
reconciliation without coupling the knowledge graph to a single vendor.

### SharePoint / Microsoft Graph

Configure `content_sync.sharepoint_sources.<source_id>` with the Microsoft
Entra directory/client IDs, site/drive IDs, tenant, and the **name** of an
environment variable holding the client secret. Run
`POST /sync/sharepoint/{source_id}/run` with a tenant-scoped write token. The
connector follows Graph `nextLink`/`deltaLink` pagination, downloads changed
file content, persists the opaque delta cursor, maps deletes to tombstones, and
normalises user/group permissions into document ACLs. Link-based or unresolvable
 permissions are stored as unknown and therefore denied when ACL enforcement is
 enabled.

### Explicit document links

HTML and Markdown links, including HTML synchronized from SharePoint, are
treated as source-observed document topology. When a target's canonical URL is
present in the same tenant, ingestion materializes `Document A -[:LINKS_TO]->
Document B`. The edge retains URL, anchor/locator, source system and version,
observation/recording timestamps, tenant, and the source ACL snapshot.

Missing targets are reconciled later; re-ingestion removes links deleted from
the source revision. Retrieval follows only explicit links and checks the
source document, link snapshot, and target document permissions before returning
target chunks. Similarity never creates `LINKS_TO` edges. The bounded traversal
is controlled by `retrieval.document_link_traversal_enabled`.

## Semantic interchange and spreadsheets

`scripts/export_rdf.py --format json-ld` emits JSON-LD; Turtle remains the
default for the in-process SPARQL bridge. External ontology linking accepts
Turtle and JSON-LD. Excel workbooks are supported through
`ExcelWorkbookConnector`: each worksheet is mapped as a table with the same
declarative entity/relation mapping, source lineage, and pre-write SHACL gate
used by relational imports. The first worksheet row must contain unique,
identifier-safe column names.

## Lineage and obligations

An ingestion request may carry `lineage_assertions` (`SUPERSEDES` or `AMENDS`)
and `obligation_drafts`. Every assertion must name its source chunk and include
the exact evidence quote. Ingestion writes a pending `LineageReview` or
`ObligationReview`; it does not create a live supersession/amendment edge or
active obligation.

Review via `/lineage/reviews/{id}/approve` or `/reject`. Approval creates the
provenance-carrying relation or an active bitemporal `Obligation`. Query the
approved register with `GET /obligations?as_of=<ISO-8601>`.
