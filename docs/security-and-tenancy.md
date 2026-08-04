# Security and Tenancy

## Tenant isolation mechanism

Every stored node carries `workspace_id`. `src/graph/execution.py`'s
`GraphExecutor.tenant_query()` is the only execution mode repositories use for
sales-domain data, and it **structurally rejects** Cypher that doesn't scope a
matched node/relationship by `workspace_id` — checked before any query reaches
the driver:

```python
_WORKSPACE_PROP_MAP_PATTERN = re.compile(r"\{[^{}]*\bworkspace_id\s*:\s*\$workspace_id\b[^{}]*\}")
_WORKSPACE_WHERE_EQUALITY_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.workspace_id\s*=\s*\$workspace_id\b")
```

Two accepted forms:
1. `{workspace_id: $workspace_id, ...}` inside a node/relationship pattern —
   what `scoped_match()` produces and every repository uses for MATCH/MERGE.
2. `x.workspace_id = $workspace_id` — needed for full-text/vector procedure
   calls (`CALL db.index.*.queryNodes(...) YIELD node`), which have no
   property-map MATCH pattern to scope at all; the equality must appear in a
   `WHERE` clause **before** any `ORDER BY`/`LIMIT`, never applied to an
   already-truncated result set.

A query with neither form raises `TenantScopingError` — proven in
`tests/unit/graph/test_execution.py`, including that a bare `$workspace_id`
parameter that's passed but never matched against is rejected.

`schema_query()` and `operational_query()` bypass this guard — allowlisted by
call-site convention (`src/graph/schema.py`/`src/graph/migrations/*` for the
former, `/health`+`/ready`+`SHOW INDEXES` for the latter). This vertical slice
has no separate caller-identity/ACL layer to enforce that mechanically; see
"Not production-authorized" below.

### Adversarial proof, not just unit-level

`tests/integration/test_tenant_isolation.py` runs two workspaces with
**identical** Account names, Claim subjects, and Mention statuses against the
live database and proves:
- a shared-attribute lookup (e.g. `find_accounts_by_name`) in workspace A never
  returns workspace B's row, even though both exist with the exact same name;
- reading a **real, valid** id from workspace B while scoped to workspace A
  returns nothing — proving the `MATCH` pattern's `workspace_id` property
  actually gates access, not merely that ids happen not to collide (they
  don't, by construction — every id is `crm_entity_id(workspace, ...)`, a hash
  that already includes `workspace`, so this test deliberately targets the one
  class of query that *doesn't* go through a unique hash: name/status lookups).

## Workspace vs. division

`workspace_id` is the tenant/security boundary. Showpad-derived nodes
(`ContentAsset`) additionally carry `division_id` — Showpad's own
organizational/permission dimension *inside* a workspace, not itself a tenant
boundary. They are not interchangeable; nothing in this repo authorizes access
based on `division_id` alone.

## Authentication — not implemented

`api/dependencies.py::get_workspace_id` reads a trusted `X-Workspace-Id`
header — **not** real authentication. It exists so every route depends on one
function (never reads a header or, worse, a request-body field directly),
which is what makes `workspace_id` "come from trusted request/authentication
context, not a user-controlled body field" even before real auth exists:
swapping this function's body for JWT/session-derived extraction later changes
nothing about any route's signature.

**This repo is not production-authorized.** There is no identity provider, no
authorization-policy interface beyond the workspace boundary, and no
division/team/opportunity-level authorization hooks. Anyone who can set an
`X-Workspace-Id` header can read/write that workspace. This mirrors the
plan's own stated scope boundary (§13): "not described as production-
authorized until a real identity provider and policy implementation exist."

## PII and secrets

- Secrets (`NEO4J_PASSWORD`, `LLM_API_KEY`, `EMBEDDING_API_KEY`) come from
  environment variables only (`src/core/config.py`, `pydantic-settings`,
  `.env` — never committed; see `.env.example`).
- `src/core/config.py`'s production-secrets validator fails fast if
  `env=production` and `neo4j_password` is still the insecure default.
- Extraction prompts (`src/extraction/prompt.py`) delimit the transcript as
  data, explicitly instruct the model to treat embedded text as content to
  extract *from*, never instructions to follow, and grant no tool access — see
  `tests/security/test_prompt_injection_fixture.py`.
- No transcript text or email appears in `structlog` INFO-level log calls
  anywhere in this codebase (all logging is IDs/counts/enums — spot-check any
  `log.info(...)` call in `src/`).
- `Claim.retention_class` and `Claim.erasure_status`
  (`src/domain/enums.py::ErasureStatus`) exist on the model; `ErasureEvent`
  exists as an audit-record type. **No erasure-propagation implementation
  exists yet** (no code walks embeddings/search-indexes/caches/derived
  summaries on an erasure request) — the fields are the contract a future
  phase implements against, not a working feature today.
- No legal-hold state is modeled.

## What an adversarial reviewer should check next

1. Confirm every new repository method added after this document was written
   still routes through `tenant_query()` (grep for `operational_query(` /
   `schema_query(` outside `src/graph/schema.py`, `src/graph/migrations/`, and
   `/health`+`/ready` — any other call site is a policy violation).
2. Confirm no route reads `workspace_id` from `request.json()`/a Pydantic
   request body field — every `ContextBuildRequest`/`CrmIngestionRequest`/etc.
   deliberately has no `workspace_id` field (proven for one route in
   `tests/integration/test_context_api.py::
   test_context_build_workspace_id_comes_from_header_not_body`).
3. Real auth: replace `get_workspace_id`'s header read with a verified
   JWT/session claim before any non-demo deployment.
