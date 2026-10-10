# Guarded graph operations (MCP capabilities and agent tools)

Status: **implemented and unit-verified** (2026-10-10), including the
adversarial suite `tests/unit/test_guarded_operations.py`. The server-side
enforcement of READ sessions is proven live by
`tests/e2e/test_live_publication_gate.py::test_run_read_refuses_writes_server_side`
(CI only).

## No generated Cypher

No MCP capability, agent tool or API route accepts Cypher, labels,
relationship types, property names or query clauses. Every graph operation is
a fixed, versioned, allowlisted operation (`capability_id@version`, e.g.
`kg.entity.lookup@1.0.0`) implemented as tested application code with
parameterized queries. A test asserts that no declared argument is named like
a query-language or schema element.

## What every MCP capability call goes through

`mcp_server/registry.py` `CapabilityRegistry.call`:

1. **Resolve** the operation (qualified name, bare id or legacy alias).
   Fabricated names -> `not_found`; the metric label is `unknown` (bounded
   cardinality; previously the raw client string).
2. **Authenticate**; the identity must be bound to a tenant (`tenant_required`).
3. **Tenant**: always the identity's tenant. A caller-supplied `tenant` that
   differs -> `tenant_mismatch`; it is never an authority.
4. **Validate arguments** (`validate_args`, shared with agent tools): declared
   types, enums, numeric bounds, string length (`max_length`, default 4000),
   `pattern` (e.g. ISO dates, identifiers), and **undeclared argument names are
   rejected** (no smuggled `scopes`, `identity`, `cypher`, `label`, `depth` ...).
5. **Dry run** previews without executing.
6. **Approval for mutations**: a non-read capability executes only if the
   service it calls enforces approval itself (`approval_enforced_by`, e.g. the
   governed work-order commands) or the registry's `approval_hook` approves the
   call. Otherwise `approval_required` / `approval_denied`.
7. **Execution scope** (`graphrag/graph/execution_scope.py`): read capabilities
   run in a **READ-access Neo4j session** (the server refuses writes), every
   query carries a **server-side transaction timeout**, and reading stops with
   `result_too_large` beyond `max_rows` (default 1000). The whole call is also
   bounded by `timeout_s` (default 30 s) -> `timeout`, and the serialized result
   by `max_result_bytes` (default 1 MB). `kg.answer.query` reads the corpus but
   records traces and caches, so it opts out of the READ session explicitly
   (`read_only_session=False`).
8. **Provenance receipt** on dict results: `operation_id`, `operation`, `kind`,
   `read_only_session`, `tenant`, `subject`, `args_sha256`, `result_sha256`,
   `executed_at`.
9. **Audit**: every call, allowed or denied (including fabricated names), is
   recorded as `(:CapabilityAuditEvent)` with operation id, operation, requested
   name, outcome, subject, tenant, a hash of the arguments (raw arguments are
   never stored) and duration. Persistence is time-boxed (2 s) and best effort;
   a failure is logged and never changes the result.

Denials are structured (`DeniedCapabilityCall{capability, reason, detail,
operation_id}`), never exceptions.

## REST agent tools (`POST /agent/tool`)

On top of `ToolPolicy` (allowlist, scopes, argument validation, quotas,
timeouts):

- **Mutating tools** (`quarantine_entity`, `erase_entity`, `ingest_document`)
  act only on the caller's own tenant, even if the token carries other
  `tenant:<x>` scopes, and require `confirm: true` (a call without it returns
  `confirmation_required`).
- The **actor is the authenticated subject**; a body `requested_by` is ignored.
- `quarantine_entity` uses `QuarantineService` (audit log) with targeted
  invalidation; `erase_entity` uses `GDPRService.forget_entity` (audit record,
  revision bump, cache eviction). Both were raw Cypher writes before.
- Graph read tools that do not carry the caller's document ACL context
  (`local_search`, `global_search`, `get_neighbors`, `search_graph`,
  `get_community`) are refused while access control is enabled
  (`acl_unsupported`); use `/query`.
- `GET /agent/audit` returns the tenant's durable audit events (it previously
  returned a hard-coded empty list).

## RDF export and SPARQL

- `scripts/export_rdf.py --tenant` is mandatory and exports exactly one tenant.
  Previously `--tenant default` (the default) exported every tenant's data to
  `exports/default/`. Retracted edges and quarantined entities are no longer
  exported.
- `POST /kg/sparql/update` with `persist=true` (overwrites the tenant's
  published export) requires the `admin` scope.

## Limitations

- The default MCP `approval_hook` is none: a new non-read capability is
  inert until it names a service-enforced approval or a hook is configured.
- The receipt is attached to dict results only; list or scalar results get an
  audit record but no inline receipt.
- Arguments are audited as a hash; correlating a denied call to its text needs
  the structured log of the caller.
