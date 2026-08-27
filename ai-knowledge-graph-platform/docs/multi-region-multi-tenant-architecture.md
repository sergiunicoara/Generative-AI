# Multi-region and multi-tenant architecture

This is the target operating model, not evidence that the local development
stack already has multi-region availability. The current graph is tenant-scoped
by a required `tenant` property; production deployment adds regional placement,
data residency, and recovery boundaries around that authorization invariant.

```
Client → global DNS / WAF → regional API + MCP gateway → regional queues
                                      │                    │
                                      ├── tenant policy     ├── regional workers
                                      │                    │
                                      └── regional Neo4j ◄──┘
                                           │
                    encrypted, approved backup / replication
                                           │
                                  recovery region
```

## Tenant isolation

Every request is bound to one tenant from a verified token. A request body,
MCP argument, cache key, retrieval query, source catalog entry, and KPI event
must never select a different tenant. Tenant IDs are logical isolation; they
are not a substitute for data-residency placement.

| Tier | Graph placement | When to use |
|---|---|---|
| Shared regional graph | One Neo4j deployment per region, every node and relationship tenant-filtered | Small or medium tenants with compatible residency and risk requirements |
| Dedicated tenant graph | Separate database/cluster and keys per tenant | Regulated, high-volume, contractual-isolation, or incompatible-residency tenants |

No cross-tenant graph query is supported. Aggregate observability uses
de-identified scalar metrics exported from each region; it does not read
document, entity, or chunk payloads into a global graph.

## Regional routing and writes

Maintain a signed tenant-placement registry outside the request payload:
`tenant → home_region, tier, residency_policy, key_reference, failover_mode`.
The API and MCP gateway resolve that registry before opening a graph or queue
connection. Cache and idempotency keys include both tenant and home region.

Writes are accepted only in the tenant home region. Each ingestion command has
a stable document/command ID and is queued regionally. Do not use active-active
graph writes until deterministic conflict handling and cross-region graph
ordering have been demonstrated. For a planned failover, stop writes, capture
the regional recovery point, promote the recovery region, and update the
placement registry atomically.

## Read continuity and recovery

Read replicas may serve retrieval only when their corpus revision is at least
the request's required revision. A stale replica returns a bounded
"updating"/retry response rather than synthesizing an answer from silently
outdated evidence. Backups are encrypted, tenant-restorable, and tested using
the same tenant scope and source-lineage validation as primary ingestion.

| Control | Shared tier baseline | Dedicated tier baseline |
|---|---:|---:|
| RPO | ≤ 15 min | contract-specific |
| RTO | ≤ 4 h | contract-specific |
| Restore test | quarterly sampled tenant | quarterly per tenant |
| Residency breach | fail closed | fail closed |

These are planning targets, not achieved SLOs. Record measured RPO/RTO from a
restore drill in the production evidence artifact before representing them as
operational guarantees.

## MCP and identity federation

Each regional MCP gateway advertises a regional protected-resource URI and
accepts only tokens whose `aud` names that exact URI. Federation trusts named
issuers and explicit audiences; tokens are never forwarded between regional
gateways. MCP 2026-07-28 calls are stateless and do not need load-balancer
affinity; retain affinity only for legacy session clients during migration.

## Operational rollout

1. Put a tenant in a single home region and verify API/MCP/cache/queue tenant
   boundaries under load.
2. Enable encrypted regional backups and complete a restore drill into an
   isolated recovery graph.
3. Add a read replica with corpus-revision gating; compare retrieval and
   citations against the home region.
4. Add placement-registry changes behind dual control, audit logging, and a
   tested rollback.
5. Offer dedicated graphs only after key rotation, quota, backup, and alert
   ownership are demonstrably tenant-specific.

## Relational/OBDA ingestion

R2RML mapping files are versioned source artifacts. The mapping adapter accepts
only safe table/column mappings and materializes source-local data through the
existing SHACL gate. `FederatedOBDAIngestor` validates every source in a single
tenant federation before it writes any source. Cross-region joins are not
performed at query time: replicate an approved, versioned source snapshot to
the tenant home region or expose a separately governed data product.
