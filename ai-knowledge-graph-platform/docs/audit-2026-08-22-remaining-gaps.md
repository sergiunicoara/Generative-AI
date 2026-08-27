# Remaining Gaps to 10/10 — Status as of 2026-08-22

> **Current implementation addendum (2026-08-27):** this historical snapshot
> is reconciled with commit `597a9b5`. The aerospace answer policy is now
> tenant-configured outside the retriever; controlled GraphRAG-Benchmark
> routes, the stateless MCP 2026-07-28 adapter, R2RML/same-tenant OBDA, and
> fuzz/property plus opt-in mutation targets are implemented. Remaining gaps
> are live-evidence items: golden retrieval quality, external IdP integration,
> DR/load drills, branch protection, and a measured mutation score.

This is a status check against the audit scorecard in
[audit-2026-08-21-second-pass.md](audit-2026-08-21-second-pass.md), recording
what has closed since that pass and what is honestly still open. It is a
snapshot, not a new audit pass — see that document and
[audit-2026-08-21.md](audit-2026-08-21.md) for the full methodology.

## Closed since the last audit (verified, not asserted)

| Row | What closed it |
|---|---|
| Security (partial) | Audience-bound tokens, RS256/JWKS/rotation, revocation, RFC 9728 discovery, prompt-injection corpus, dead GenAI token-usage telemetry fixed, multi-issuer OAuth federation code + tests (this session — see item 3) |
| Reliability | Chaos/retry-storm tests, concurrency tests, property-based invariants, failure-exercise harness rewired to actually execute (previously hardcoded `True` with nothing run) |
| Observability | Real metrics wired to real code paths, alert rules cross-checked against actual metric names, Grafana dashboard, SLO doc |
| Knowledge representation | Exact-fidelity RDF round-trip tests, OWL/SHACL/SKOS interoperability verified against real third-party engines (`owlrl`, `pyshacl`), not just triple-count checks. The `(partial)` qualifier this row carried was itself stale: all 103 RDF/OWL/SHACL/SKOS/SPARQL-bridge/cross-ontology-linker unit tests pass with zero skips, and no TODO/FIXME/NotImplemented remains anywhere in the KR source or test suite — there was no actual remaining gap to qualify. |
| Testing (item 5, CI gate integration) | Corrected a root-vs-subdirectory search mistake in this same audit: a real, comprehensive CI workflow already exists at the monorepo root and is pushed to `origin/main` — see item 5 below |

## Still open, and why

### 1. Scale/availability evidence — 0%

Nothing here is a code fix. It needs 10x/100x/1000x load tests, a Neo4j
cluster/read-replica setup, autoscaling on queue age, and a capacity report.
This cannot be produced from a coding session — it needs real infrastructure
under real, sustained load.

### 2. Complete retrieval-quality evaluation — near 0%

A full golden-set run including the RAGAS unscorable/refusal cases,
DRIFT/PageRank/GNN ablations, and a GraphRAG-Bench comparison.

**Currently blocked**: Docker is not running in this environment, so Neo4j,
Redis, and RabbitMQ are unavailable. This was checked directly before
starting the most recent work session — it is also why the aerospace-prompt
decoupling below was not attempted; its own stated prerequisite is "a
runnable golden eval."

### 3. Federated MCP + tested disaster recovery — partially closed this session

OAuth 2.1 is done for the *local* case — audience binding, protected-resource
metadata, revocation.

**Closed this session**: multi-issuer trust dispatch is now real, working
code, not a paper design. `graphrag/core/issuer_trust.py` (new) holds a
configured allow-list of trusted external issuers (`jwt_trusted_issuers` in
`graphrag/core/config.py`), each scoped to specific audiences. Every token
now carries an `iss` claim; `api/auth/jwt.py`'s `decode_access_token`
dispatches on it — a self-issued token (including one minted before `iss`
existed) verifies exactly as before, an externally-issued token is verified
only against that issuer's own fetched-and-cached RS256 keys, and **an
unrecognized issuer is rejected outright, never falling through to try local
keys**. A trusted issuer's tokens are further restricted to the audience(s)
it was explicitly configured for, even if the caller only asked for a
generic decode. HS256 is excluded from federation structurally (a JWKS has
no symmetric-key concept), not by a runtime check. Fully unit-tested in
`tests/unit/test_issuer_federation.py` (8 tests) and
`tests/unit/test_config.py` (settings validation) with an injected fake HTTP
client — no Docker, no real network, no live IdP required. The critical
negative test (a token signed with this deployment's own real key but
claiming an unconfigured foreign issuer) is rejected, proving there is no
key-fallback path that would defeat the whole point of naming an issuer.

**Still open**: an integration test against a *real* external IdP (Auth0,
Okta, etc. actually issuing a token this code verifies end-to-end), external
IdP trust-establishment tooling (the config today assumes a human hand-enters
the trust anchor), and an actual DR restore drill with measured RTO/RPO. All
three still need live infrastructure this session cannot provide. Unit-testing
the federation *code* does not by itself constitute "tested disaster
recovery" — that claim is not being made here.

### 4. Architecture — aerospace-prompt separation (closed 2026-08-27)

The historical audit deliberately deferred this because
`hybrid_retriever.py`'s answer-synthesis prompt hardcoded aerospace-specific
rules. Commit `597a9b5` moved synthesis into a tenant-configured answer policy,
kept aerospace formatting opt-in, and documented the multi-region/tenant
target architecture. A live golden comparison is still useful evidence, but it
is no longer a missing implementation.

### 5. Testing — CI gate integration — closed (this session)

Previously marked "unverified" because `.github/workflows/` was inspected
only inside `ai-knowledge-graph-platform/`, not at the actual git root —
this is a monorepo (`Generative-AI/`), and GitHub Actions only reads
workflows from the repository root, not a subdirectory. Correcting that:

`Generative-AI/.github/workflows/ai-knowledge-graph-platform-ci.yml` exists,
is correctly placed, and is pushed to `origin/main` (added 2026-08-20,
`e159a41`). It runs unit, integration, load, and e2e (real Neo4j/Redis via
testcontainers, with a hard failure if e2e silently skips instead of a false
green), lint, and Terraform/Kubernetes manifest validation, triggered on
push/PR to `main`/`develop` plus a nightly unattended cron run. Its own
commit message documents that an earlier copy lived at
`ai-knowledge-graph-platform/.github/workflows/ci.yml` and never executed —
not once — because of the same root-vs-subdirectory mistake this audit
initially repeated.

**Still not verifiable from this environment**: whether GitHub's
branch-protection rule on `main` actually requires this workflow to pass
before merge. That's a repo-settings toggle on GitHub, not a file in the
repo — checking it needs authenticated `gh`/GitHub API access, which this
session does not have. Fuzz/property checks now run through `make test-fuzz`;
`make mutation` provides the opt-in Mutmut campaign. A complete mutation score
still needs a CI run.

## Bottom line

Everything verifiable offline, without live infrastructure or LLM API spend,
is either closed or blocked on a real prerequisite that is being respected
rather than worked around. What remains for 10/10 is almost entirely
**evidence that requires running things unavailable in this environment**:
Docker services, a live LLM budget for the golden eval, a real load-testing
environment, and an actual DR drill.

## Highest-leverage next step

Get Docker Desktop running and provide a `.env` with live LLM keys. That
alone unlocks:

- running the golden eval (item 2),
- then measuring the already-implemented answer-policy separation with a real
  before/after golden comparison (item 4).

Item 5 (CI gate integration) is now closed as of this session — see above.
The one loose end left on it, branch-protection enforcement, needs
authenticated GitHub access (`gh auth login`, or connecting the GitHub MCP
connector) rather than live infrastructure — a cheap follow-up whenever
that's available.
