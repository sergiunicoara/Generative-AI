# MCP Operations Runbook

This runbook covers the authenticated MCP gateway. Its security contract is
defined in [ADR 0009](adr/0009-agent-platform-trust-boundaries.md) and
[ADR 0010](adr/0010-audience-bound-access-tokens.md).

## Token audience (read this before issuing an MCP token)

The gateway accepts **only** tokens minted for its own resource identifier.
A REST API token is rejected with 401, by design — see ADR 0010. Two variables
decide this and must agree between the API process, the gateway process, and
the client:

| Variable | Meaning | Local default |
|---|---|---|
| `GRAPHRAG_API_RESOURCE` | Canonical URI of the REST API resource server | `http://localhost:8000` |
| `GRAPHRAG_MCP_RESOURCE` | Canonical URI of this gateway | `http://localhost:8002/mcp` |

Canonical means absolute URI, no query, no fragment, no trailing slash. A
non-canonical value is rejected at startup rather than producing tokens that
silently never match.

Obtain an MCP token by naming the resource in the token request (RFC 8707):

```powershell
$body = @{
  grant_type    = "client_credentials"
  client_id     = $env:GRAPHRAG_CLIENT_ID
  client_secret = $env:GRAPHRAG_CLIENT_SECRET
  scope         = "read"
  resource      = $env:GRAPHRAG_MCP_RESOURCE
} | ConvertTo-Json
$token = (Invoke-RestMethod http://localhost:8000/auth/token -Method Post -Body $body -ContentType "application/json").access_token
```

The response echoes the bound `resource`; verify it matches before wiring the
token into a client. A client that does not know where to ask can read the
gateway's unauthenticated discovery document, which every 401 also points at
in its `WWW-Authenticate` header:

```powershell
Invoke-RestMethod http://localhost:8002/.well-known/oauth-protected-resource/mcp
```

Local stdio clients are deliberately exempt from audience validation: the MCP
specification directs stdio servers to take credentials from their launcher's
environment, so `GRAPHRAG_MCP_TOKEN` for `python mcp_server/server.py` does not
need an MCP audience.

## Local verification

Start the normal infrastructure and API first so JWT issuance, Neo4j, and
Redis are available. Then start the remote gateway:

```powershell
$env:GRAPHRAG_MCP_PORT = "8002"
python -m mcp_server.remote
```

`GET http://localhost:8002/health` is intentionally public for an orchestrator
probe. `/mcp` and `/metrics` require `Authorization: Bearer <scoped JWT>`.
The gateway accepts legacy session-oriented Streamable HTTP during the
migration window and the stateless MCP 2026-07-28 core. Modern requests use
`MCP-Protocol-Version: 2026-07-28`, `Mcp-Method`, and, for `tools/call`, a
matching `Mcp-Name`; the gateway rejects header/body mismatches before a tool
is resolved. Modern calls do not use `initialize` or `Mcp-Session-Id`.
Use a real MCP Streamable HTTP client for protocol calls; do not hand-craft a
write JSON-RPC request as an operational test.

The standalone command binds to `127.0.0.1` by default. Container definitions
set `GRAPHRAG_MCP_HOST=0.0.0.0` because their Service/network policy is the
network boundary. If a browser-based client sends an `Origin` header, list its
exact origin in `GRAPHRAG_MCP_ALLOWED_ORIGINS`; an unlisted origin is rejected
with 403. Non-browser clients normally omit `Origin` and are unaffected.

```powershell
# A protected observability smoke test; never print the token in a ticket.
Invoke-WebRequest http://localhost:8002/metrics -Headers @{ Authorization = "Bearer $env:GRAPHRAG_MCP_TOKEN" }

# Deterministic capability and router safety gates (no external services).
python scripts/run_capability_eval.py
python -m pytest tests/unit/test_mcp_identity.py tests/unit/test_mcp_remote.py tests/unit/test_mcp_contract_compat.py tests/unit/test_workorder_compensation.py -q
```

For local stdio clients, launch `python mcp_server/server.py` with a scoped
`GRAPHRAG_MCP_TOKEN`. Stdout is protocol-only; diagnostics go to stderr.

## Capability contract

The capability registry is the client integration boundary. Each tool has a
stable dotted identifier, semantic version, argument keys, risk class,
required scopes, dry-run support, approval requirement, and optional legacy
aliases. The committed snapshot is compatibility-tested before release.

Export the client-consumable contract with:

```bash
python scripts/export_mcp_contract.py --output artifacts/mcp-capabilities-v1.json
```

Consumers should bind to a fully qualified name such as
`biz.workorder.compensate@1.0.0`. A bare capability ID resolves to the newest
registered version only when a client intentionally accepts that upgrade
policy. Legacy aliases remain recorded and compatibility-tested. The exported
JSON can feed an SDK generator or separate contract package, but publishing one
requires an explicit registry, licence, version, and support decision.

## Deployment

`deploy/kubernetes/mcp.yaml` ships a non-root, read-only-root-filesystem MCP
gateway on an internal `graphrag-mcp` service. It has a PDB and ClientIP
affinity because Streamable HTTP may keep a long-lived session response. Do
not expose this Service with a public LoadBalancer.

Before exposing `/mcp` through an ingress:

1. Terminate TLS at the approved gateway and forward the `Authorization` and
   `X-Correlation-ID` headers unchanged.
2. Replace the placeholders in
   `deploy/kubernetes/network-policy-production.example.yaml` with the real
   ingress-controller namespace and explicit DNS, Neo4j, Redis, JWKS/IdP,
   OTLP, and provider egress destinations. Apply it as a reviewed production
   overlay.
3. Configure Prometheus with a least-privilege Bearer token for `/metrics`.
4. Confirm an unscoped token sees neither `biz.workorder.create` nor any other
   withheld capability via `discover_capabilities`.
5. Keep replicas client-affine until legacy MCP session state is backed by a
   shared, tested session store. Modern 2026-07-28 calls are stateless; remove
   affinity only after legacy-client migration is complete.

Render the base manifest before applying it:

```bash
kubectl kustomize deploy/kubernetes
```

## Incident checks

| Symptom | Check | Response |
|---|---|---|
| 401 from `/mcp` | JWT expiry, subject, tenant claim, **and `aud` vs `GRAPHRAG_MCP_RESOURCE`** | Reissue a token with `resource=<GRAPHRAG_MCP_RESOURCE>`; do not relax the gateway. The `WWW-Authenticate` header on the 401 names the metadata document and required scope |
| Every client 401s after a deploy | `GRAPHRAG_MCP_RESOURCE` changed, or differs between the API and gateway processes | Restore agreement between the two processes and the client's `resource` parameter; audience is an exact string comparison |
| 413 from `/mcp` | `GRAPHRAG_MCP_MAX_REQUEST_BYTES`, client payload | Reduce/chunk the client request; increase only after a capacity review |
| 403 `Origin is not allowed` | Browser origin absent from `GRAPHRAG_MCP_ALLOWED_ORIGINS` | Add the exact trusted HTTPS origin; never use a wildcard |
| Structured `tenant_mismatch` denial | Client-provided tenant vs signed claim | Correct the client configuration; never override the claim |
| Missing write capability in discovery | `biz:write` entitlement | Grant through the identity provider approval process, not application config |
| Long-lived session disconnects after scale | Service affinity / rollout events | Drain client sessions; do not remove affinity without shared-session validation |

## Evidence collection

The gateway already emits protected Prometheus metrics for versioned MCP
capability calls and governed-write receipt outcomes. To prepare a truthful
portfolio or operational report, save an authenticated `/metrics` response,
copy `docs/templates/production-evidence-template.json`, fill only measured fields, and
generate a source-linked report:

```powershell
python scripts/export_operational_evidence.py `
  --metrics artifacts/mcp.prom `
  --metadata artifacts/production-evidence.json `
  --output artifacts/operational-evidence.json
```

`null` means “not measured.” It must never become an implied production,
customer, availability, or business-impact claim.

## Context Graph outcome demo

The outcome loop requires an existing decision and policy version in the same
tenant. The script creates an append-only action, observed outcome, and human
feedback, then retrieves precedents. It validates the new `ASSESSES` link:

```bash
python scripts/demo_context_graph_outcomes.py \
  --tenant marketing \
  --decision-id <existing-decision-id> \
  --policy-version-id <existing-policy-version-id>
```

The script is deliberately not an MCP write capability. It is a controlled
operator/demo action; agent-visible Context Graph access is read-only
`cg.precedent.find@1.0.0`.
