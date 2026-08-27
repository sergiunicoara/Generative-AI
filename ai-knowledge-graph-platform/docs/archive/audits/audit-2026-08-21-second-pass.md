# Second-Pass Audit — 2026-08-21

A follow-up review of the tree left by
[audit-2026-08-21.md](audit-2026-08-21.md). That pass closed nine cross-cutting
gaps; this one re-examined the areas it did not reach and the areas it listed as
planned. It is deliberately narrow: rather than re-litigating settled decisions,
it looks for defects that survive a careful reading and are provable offline.

## What was checked and found clean

Recording the negative results matters as much as the findings — several
plausible-sounding defects turned out not to exist, and re-"fixing" them would
have been churn:

| Hypothesis | Result |
|---|---|
| Cypher injection via interpolated labels/relations/fields | **Clean.** Every f-string Cypher site interpolates a hardcoded literal or a type-dispatched constant; all caller data goes through parameters |
| Blocking I/O inside `async def` (requests, `time.sleep`, subprocess) | **Clean.** An AST sweep over `graphrag/` found none |
| Unsalted hashing of M2M client secrets | **Not a defect.** The secret is `secrets.token_urlsafe(40)` — 320 bits. A KDF protects low-entropy passwords; it buys nothing here |
| Missing trace-context propagation across RabbitMQ | **Already implemented.** `publish` injects and `consume` extracts W3C context |
| Async singleton cold-start races generally | **Only one.** Every other singleton initialises synchronously and cannot interleave |
| Tenant leaking through cache keys | **Clean.** `build_cache_key` includes the tenant in both the digest and the key prefix |

## Findings and remediation

| Priority | Finding | Risk | Resolution |
|---|---|---|---|
| P1 | Access tokens carried no `aud` claim, so a REST API token was accepted verbatim by the remote MCP transport | Token passthrough / confused deputy. A token leaked from any API client immediately reached the governed MCP write and approval surface. Violates three MUSTs in the MCP 2026-07-28 authorization specification | Canonical resource identifiers (RFC 8707); `aud` on every issued token; strict audience validation at the MCP transport; `resource` parameter on `POST /auth/token` with `invalid_target` rejection. See [ADR 0010](adr/0010-audience-bound-access-tokens.md) |
| P1 | The API accepted MCP-audience tokens too, once audience binding existed | Closing only the API-token-reaches-MCP direction would leave the identical confused-deputy shape running the other way | The REST API verifies its own audience as well, non-strictly: a token naming the MCP resource is rejected, a token with no `aud` at all is still accepted for the one-hour rollout window |
| P1 | No RFC 9728 protected resource metadata, and 401s carried no `WWW-Authenticate` header | An MCP client that lacks a usable token has no way to discover where to get one; the specification makes RFC 9728 a MUST | Unauthenticated metadata document at `/.well-known/oauth-protected-resource/<path>`; `WWW-Authenticate: Bearer` on every 401 naming that document, the error, and the minimum scope |
| P1 | `decode_access_token` did not require `exp` | A token minted without an expiry claim never expires. Only this codebase mints tokens today, so exposure required a second issuance path or a leaked key — but the absence of a shape check is what makes such a path silent | `exp` is now required on every decode path, independent of audience |
| P1 | `get_query_cache()` awaited `connect()` inside an unguarded `if _cache is None` | Two coroutines racing on the first query each open a Redis connection pool; one leaks for the life of the process. The same defect was fixed for `get_rabbitmq()` in the previous pass and missed here | Lazy `asyncio.Lock` with an inner double-check; a failed connect is not cached, so a transient outage cannot permanently pin the process to the fallback |
| P1 | The answer cache's in-process fallback was unbounded, and entries expired only when something read that exact key again | A Redis outage turns an unbounded stream of distinct queries into unbounded process memory (OWASP API4). The provenance index was a second unbounded structure holding keys for entries that no longer existed | Bounded LRU (`semantic_answer_cache_max_memory_entries`, default 2048); a full expiry sweep throttled to once per TTL/10 with a per-entry freshness check on every read, so the sweep cost stays off the query path without ever serving stale data; and a reverse provenance index so removing an entry is O(entities it cited) rather than O(whole index) |
| P2 | A Redis outage silently degraded the answer cache to per-process storage | `invalidate_for_entities` then cannot reach a sibling replica: after a correction, that replica keeps serving the superseded answer for a full TTL. This is a correctness failure presented as a warning log | `semantic_answer_cache_strict` refuses to start in that state, matching the existing `session_store_strict` trade-off. Default remains off for single-process local work |
| P2 | The answer cache's Redis pool had no closer | The one shared client absent from `close_shared_resources()`; a restarted API process leaked connections until Redis timed them out | `close_query_cache()` added to the shared shutdown path |
| P2 | Rate-limit counters were per-process, and keyed only on IP address | "20/minute" was really 20/minute/replica/restart — a limit nobody reasoned about. IP-only keying also makes every caller behind one NAT share a bucket while one credential driven from many addresses is never throttled | Shared Redis storage when `REDIS_URL` is set, with an explicit in-memory fallback on outage; buckets keyed on the verified `sub` claim when present, namespaced so a subject cannot collide with an address |
| P2 | Non-dev startup validation rejected only the *literal* default signing key | Any other short secret passed. This is how a 23-byte key reached this repository's own `.env`, producing a PyJWT `InsecureKeyLengthWarning` nobody was grepping for | Minimum 32 bytes enforced for `jwt_secret_key` and `session_secret_key` outside `DEV_ENVS`, per RFC 7518 §3.2 |
| P3 | Spans interrupted by `asyncio.CancelledError` exported as UNSET | The SDK records `Exception` but not `BaseException`. Request timeouts, budget aborts, and worker shutdown all unwind through cancellation, so the spans an operator most wants during an incident looked healthy | `trace_span` records and marks only the cases the SDK does not, leaving its richer status description intact for ordinary errors |
| P3 | Redis fallbacks in the M2M client registry and user-provisioning table were completely silent | A client registered during an outage exists on one replica and 401s on every other — indistinguishable from a bad secret at the caller | One structured warning per failed operation, naming the operation and the impact |

## Scope discipline

Four things were deliberately **not** changed:

- **The unsalted client-secret hash.** Defensible as-is (see above). Changing it
  would be motion, not improvement.
- **`client_id` enumeration through distinct 401 messages** on `/auth/token`.
  Real, but `client_id` is 128 bits of `secrets.token_urlsafe`; the oracle
  reveals nothing an attacker can act on, and the clearer error is worth more
  operationally.
- **An async rate-limit backend.** slowapi 0.1.10 has no async `Limiter`, so
  Redis-backed limiting blocks the event loop for one round trip per limited
  request. Only six endpoints are limited and all are dominated by LLM or
  Neo4j work, so the hop is noise; replacing slowapi to avoid it would be a
  larger change than the problem justifies today. The constraint is documented
  at the top of `api/limiter.py` so the next person to decorate a hot endpoint
  sees it first.
- **The corpus-specific answer prompt.** `_ANSWER_PROMPT` in
  `hybrid_retriever.py` hardcodes aerospace-corpus rules (revision-number
  formatting, `doc_id` metadata conventions, specific airworthiness phrasing).
  This is genuine architectural debt for a platform sold as domain-general, but
  changing it without being able to run the golden eval would be trading a
  measured pass rate for an unmeasured one. Moved to the roadmap as a
  prompt-per-ontology item, gated on an eval run.

## Standards and specifications applied

| Source | Requirement used | Where |
|---|---|---|
| [MCP 2026-07-28 Authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) | Resource servers MUST validate token audience; MUST implement RFC 9728; MUST NOT accept or transit other tokens; SHOULD include `scope` in `WWW-Authenticate` | `mcp_server/remote.py`, `mcp_server/oauth_metadata.py` |
| [RFC 8707](https://www.rfc-editor.org/rfc/rfc8707.html) | Canonical resource URI form; `resource` request parameter; `invalid_target` | `graphrag/core/resource_identifiers.py`, `POST /auth/token` |
| [RFC 9728 §3.1](https://datatracker.ietf.org/doc/html/rfc9728) | Well-known prefix inserted between host and path | `mcp_server/oauth_metadata.metadata_path` |
| [RFC 6750 §3](https://datatracker.ietf.org/doc/html/rfc6750#section-3) | `WWW-Authenticate` challenge parameters and quoting | `oauth_metadata.challenge_header` |
| [RFC 7518 §3.2](https://datatracker.ietf.org/doc/html/rfc7518#section-3.2) | HMAC key at least as long as the hash output | `graphrag/core/config.py` |
| [OWASP API4:2023](https://owasp.org/API-Security/editions/2023/en/0xa4-unrestricted-resource-consumption/) | Bounded caches and shared rate-limit state | `query_cache.py`, `api/limiter.py` |

## Note on the MCP specification version

The previous audit targeted MCP 2025-06-18. The current specification is
**2026-07-28**, which adds a stateless protocol core, multi-round-trip
requests, header-based routing, cacheable list results, and a formal extensions
framework alongside the authorization hardening implemented here. At the time
of this audit only authorization was adopted; the current implementation also
provides a compatibility adapter for stateless `tools/list` and `tools/call`
in `mcp_server/transport_20260728.py`. The legacy SDK session path remains
during migration, and full client interoperability still needs live coverage.

## Verification

| Gate | Result |
|---|---|
| `pytest` (full suite, before changes) | 1,293 passed, 1 skipped, 328s |
| `pytest` (full suite, after changes) | **1,354 passed, 1 skipped**, 259s — +61 tests, 0 failures |
| `ruff check .` (repository-wide) | passed |
| `compileall` for `api/ graphrag/ workers/ mcp_server/ scripts/` | passed |
| `docker compose -f docker-compose.yml config` | passed |
| `docker compose -f compose.dev.yaml config` | passed |
| Kubernetes manifests (`yaml.safe_load_all`) | all 12 parse |
| `pip-audit -r requirements.lock` | 2 advisories, both the previously-documented no-fix RAGAS-stack ones (`ragas` SSRF, `diskcache` pickle). No dependencies were added or changed in this pass |
| Secret-pattern scan over the diff | clean |

New test files:

| File | Covers |
|---|---|
| `tests/unit/test_mcp_oauth_resource.py` (27) | Canonical URI rules, cross-resource rejection in both directions, missing/wrong/multi-valued/malformed `aud`, missing `exp`, both `WWW-Authenticate` shapes, RFC 9728 path placement and document contents, and the `resource` parameter round-trip through the real `/auth/token` route |
| `tests/unit/test_query_cache_resilience.py` (10) | LRU bound and eviction order, full expiry sweep, throttled-sweep-never-serves-stale, provenance index lifetime, strict mode, concurrent cold start, failed connect not cached |
| `tests/unit/test_rate_limit_identity.py` (12) | Subject-vs-address bucketing, namespace collision, unverified-header rejection, `X-Forwarded-For` hop depth, storage selection and fallback |
| `tests/unit/test_tracing_error_status.py` (4) | SDK behaviour for `Exception` preserved; cancellation newly marked ERROR |

Not run: live Docker-backed integration tests, the golden retrieval eval, and
any load or chaos exercise. All three need services and provider credentials
this pass did not have. Nothing here claims a measured quality, latency, or
capacity result.

## Breaking change

Remote MCP clients must now request a token bound to the MCP resource:

```
POST /auth/token
{"grant_type": "client_credentials", "client_id": "...", "client_secret": "...",
 "scope": "read", "resource": "<GRAPHRAG_MCP_RESOURCE>"}
```

Pre-existing MCP tokens are rejected with 401 and a `WWW-Authenticate` header
pointing at the discovery document. Local stdio clients are unaffected. See
[docs/mcp-operations.md](mcp-operations.md) for the operator procedure and
[ADR 0010](adr/0010-audience-bound-access-tokens.md) for the rationale.
