# Implementation Audit

**Date:** 2026-09-21
**Method:** Four parallel code-and-test audits (tenant/opportunity isolation; MCP/agent security + prompt injection; provenance/temporal/contradiction/entity-resolution; retrieval/eval/observability/cost), each verifying claims against actual source and test files with `file:line` evidence — never against README, roadmap, or comments. Baseline tests were not re-run as part of this audit pass; existing automotive/aerospace eval baselines from this session (`tasks/automotive-retrieval-diagnosis-report.md`) remain the most recent recorded results.

Status definitions: **Implemented** (verified in code AND covered by a meaningful test) · **Partial** (code exists but enforcement/integration/testing is incomplete) · **Missing** (no functional implementation found) · **Documented only** (described in docs, not in executable code).

---

## 1. Tenant and opportunity isolation

| Capability | Status | Evidence | Problems | Required action |
|---|---|---|---|---|
| Tenant derivation from auth | Implemented | `api/auth/dependencies.py:101-117` (`get_tenant` reads JWT only); `assert_request_tenant` (`:120-144`) rejects a client-supplied tenant that disagrees with the token | None significant | Keep |
| Cypher tenant-scoping | Implemented | `graphrag/graph/neo4j_client.py` — hundreds of MATCH/MERGE calls carry `{tenant: $tenant}`; `graphrag/core/tenancy.py:18-29` `require_tenant` fails loud on falsy tenant | Many functions still default `tenant: str = "default"` in signatures — a footgun for a future caller that forgets to pass tenant, even though every current caller does | Fix: make `tenant` a required (no-default) kwarg in `neo4j_client.py` |
| Vector/lexical retrieval tenant filtering | Partial | `neo4j_client.py:1735-1854` — `db.index.vector.queryNodes` runs ANN globally across all tenants first, then filters `WHERE c.tenant = $tenant` before any row leaves the DB layer — no cross-tenant leak, but code comments admit a tenant "can be starved out of a small top-k by other tenants' higher-scoring nodes" | Recall correctness (not security) risk; a `_filtered_vector_search` native pre-filter path exists but is conditionally used | Fix: confirm the native filtered-index path is enabled in production |
| Multi-hop graph traversal | Implemented | `neo4j_client.py:1944-1993` — `tenant_filter` applied unconditionally to every relationship and neighbor in the expansion path (previously skippable, now fixed per in-code comment) | No dedicated unit test asserts this specific function's filter | Add: direct regression test on `expand()`'s tenant filter |
| Cache tenant scoping | Implemented | `graphrag/retrieval/query_cache.py:78-89` hashes tenant into the cache key; cross-tenant cache-flush vulnerability (flush by path param) already fixed — tenant now derived from token (`tests/unit/test_tenant_isolation.py:802-858`) | None found | Keep |
| Ingestion tenant requirement | Partial | `scripts/ingest_corpus.py` always stamps an explicit `--tenant`; API ingest path forces tenant from JWT (`api/routes/ingest.py:61`) | The CLI script is a trusted-operator tool with no tenant-allowlist validation — acceptable only if it stays operator-only | Keep, document as trusted-operator-only |
| Opportunity/deal-level authorization | **Missing** | Zero matches for "opportunity" anywhere in code, tests, or scripts | This platform has no opportunity/deal/CRM-record-level ACL concept at all — not partially built, entirely absent | Add: must be designed and built from scratch if this domain concept is actually required |
| Authz at tool-execution layer (not just HTTP route) | Implemented | `graphrag/agents/tool_policy.py:88-138,170-252` — `validate_args` enforces `tenant:<name>` scope on any tool arg named `tenant`; `ToolPolicy.call` re-checks scopes at execution time | Scope model is coarse (`read`/`write`/`admin`/`tenant:<x>`) — no per-resource granularity, consistent with #7 (no opportunity concept) | Keep |
| Tenant-isolation test quality | Partial | `tests/unit/test_tenant_isolation.py` — most tests assert Cypher **string shape** against a mocked Neo4j client, not real cross-tenant denial against live data. Only one test (`test_two_tenants_read_different_export_files`) proves true end-to-end denial | `scripts/verify_tenant_isolation.py` does real live-graph cross-tenant checks but is a manual ops script, **not wired into CI/pytest** | Add: integration test (live/testcontainer Neo4j) proving `local_search`/`vector_search_chunks`/`expand` never return another tenant's data; wire `verify_tenant_isolation.py` into CI |
| Neo4j schema-level tenant enforcement | Partial | `graphrag/graph/schema.cypher:7-30,71-95` — composite uniqueness constraints on `(key, tenant)` plus tenant indexes | No `tenant IS NOT NULL` constraint exists on any label — a node with `tenant = NULL` can still be created at the DB level; enforcement is 100% application-layer | Add: `IS NOT NULL` property-existence constraints on `tenant` for Entity/Chunk/Document/Community |

---

## 2. MCP and agent tool security

| Capability | Status | Evidence | Problems | Required action |
|---|---|---|---|---|
| MCP server existence | Implemented | `mcp_server/server.py`, `tools.py`, `registry.py`, `capabilities/*.py`; versioned capabilities (graph_stats, entity_lookup, facts_query, workorder_create, workorder_compensate, discover) | None — real, not a stub | Keep |
| Tool allowlist + scope + arg validation | Implemented | `graphrag/agents/tool_policy.py:143-291` — allowlist (186-188), scope check (198-201), arg validation (204-206,88-138), dry-run gate (192-195), `asyncio.wait_for` timeout (222-230) | Tenant-cross-access guard only fires for tools whose schema declares a `tenant` arg; tools without one get tenant force-injected from caller context — correct by design but fragile if that overwrite logic regresses | Add: regression test proving a tool with no `tenant` in its schema can't have effective tenant overridden via a smuggled kwarg |
| Tool risk classification | Implemented | `ToolSpec.risk` (safe/moderate/destructive, `tool_policy.py:59`); `CapabilitySpec.kind`/`risk` (`mcp_server/registry.py:36`); scope-gated by risk tier (`tool_policy.py:340-438`) | None | Keep |
| Argument validation mechanism | Partial | Hand-rolled typed schema validator (`tool_policy.py:88-138`), not Pydantic/JSON Schema per-tool; Pydantic is used elsewhere for `CommandEnvelope`/`CommandReceipt` (`graphrag/business/commands.py:31-82`) | Hand-rolled validator doesn't reject unknown/extra keys (acknowledged in-code) | Fix: consider per-capability Pydantic models |
| HITL / human approval | Partial | Implemented for business writes: `graphrag/business/policy.py:29-64`, `service.py:100-139,259-287` — critical/high-severity or any `actor.type=="agent"` command escalates to approval | The generic `ToolPolicy`/`CapabilityRegistry` gate has **no approval step of its own** — high-risk tools like `quarantine_entity`/`erase_entity` execute immediately once scopes pass; `requires_approval` exists as a `CapabilitySpec` field but isn't confirmed wired | Add: route restricted-risk MCP capabilities through the business approval flow, or confirm and test the existing wiring |
| Prompt-injection structural containment | Implemented (explicitly not behavioral) | `graphrag/core/prompt_security.py:8-10` (`escape_prompt_data`), used in `extractor.py:100-103` and `answer_policy.py:19-21`; delimited `<source_text>`/`<retrieved_context>` boundaries with explicit "never follow instructions" language | No output-side scan for leaked instructions/tool calls; the test suite itself documents this limit ("does not prove the model ignores the instruction... needs a live model and a graded run") | Add: defense-in-depth output check; mocked-LLM behavioral test against the injection corpus |
| Runtime tool/permission escalation | Missing (i.e., correctly absent) | No dynamic tool-registration code found; only static `.register()` at module load | None — this is the desired state | Keep |
| Adversarial security tests | Implemented | `tests/unit/test_prompt_injection_corpus.py` (20-payload OWASP-LLM01-derived corpus), `test_capability_registry.py:143-166` (tenant-mismatch denial), `test_tool_safety.py` | Corpus proves structural containment (delimiters can't escape), not that a live model actually refuses an injected instruction | Add: mocked/live-LLM behavioral spot-check against the corpus |
| Network egress restriction | Partial | `graphrag/core/connector_url_safety.py:38,45-71` — scheme/host allowlist only, explicitly **not** an SSRF/private-IP blocklist (documented in-code); `sparql_bridge.py:44-65` blocks SSRF-prone SPARQL forms; K8s NetworkPolicy exists but is explicitly excluded from the base kustomization | No runtime private-IP/CIDR blocklist if any connector URL ever becomes tenant-supplied; production network policy is opt-in, not default | Fix: promote NetworkPolicy to a required prod overlay; add IP blocklist if URLs become user-influenced |
| Audit logging | Partial | Durable: `graphrag/graph/audit_trail.py:27-131` (append-only `ChangeLog` for graph mutations). MCP capability calls recorded via `agent_telemetry.py` metrics. `ToolPolicy._audit` is **in-memory only** per its own docstring | Tool-call audit trail does not survive a process restart, unlike graph-mutation audit | Fix: implement persistent flush of `ToolPolicy` audit entries |

---

## 3. Evidence provenance and temporal correctness

| Capability | Status | Evidence | Problems | Required action |
|---|---|---|---|---|
| Claim/evidence model | Partial | `graphrag/core/models.py:186-251` — Entity/Relation carry `source_chunk_id(s)`, `source_doc_id`, `confidence`, `extraction_model`; no distinct Claim/Assertion node, no TranscriptSegment/CRMRecord model | `extraction_model` is a **static config label** (`cfg.groq_model`, `extractor.py:60,124,161`), not real per-call provider/response metadata — confirmed independently earlier this session against the automotive corpus | Add: real Claim/Assertion linking node; capture true per-call provider info |
| Structured answer evidence | Partial/Missing | `context_builder.py:105,403-408` — `QueryResult.citations` is a flat `list[str]`, no graph path, no per-citation confidence or timestamp | Answers cannot currently surface "claim + source + timestamp + path + confidence" as the spec requires | Add: structured evidence object replacing the flat citation list |
| Bitemporal fields | Partial | `neo4j_client.py:1050-1230` sets `recorded_at`/`valid_from`/`valid_to`; `local_search.py:1467-1729` accepts `valid_at`/`transaction_at` | Default "current state" queries do **not** filter out expired (`valid_to` passed) facts unless a caller explicitly opts into point-in-time mode — this inverts the required default | Fix: make current-state queries filter `valid_to IS NULL OR valid_to > now()` by default |
| Supersession handling | Partial | `document_authority.py:32-33,93-150,235-306` — `SUPERSEDED_CONFIDENCE_PENALTY=0.5` downweights superseded docs | Superseded documents are never removed from retrieval, only downweighted — by design, but means "exclude superseded facts by default" (spec requirement) is not actually true today | Fix: align default current-state behavior with the bitemporal fix above |
| Contradiction detection | Partial | `contradiction_strategies.py:183,264,344,422` — 4 active strategies (directional_reversals, exclusive_states, functional_violations, positive_negative_pairs); `contradiction_detector.py:31,44-53,141-164` — `resolve()` preserves provenance via conflict node | At the relation-merge layer (`neo4j_client.py:1192-1230`), same-edge merges are **last-write-wins** on `source_chunk_id`/`extraction_model`/`valid_from`/`valid_to`/`weight` — only `confidence` (Bayesian) and `source_doc_ids` (list) actually accumulate; older single-chunk provenance is silently overwritten | Fix: version relation provenance per contributing document instead of scalar overwrite |
| Entity resolution ordered pipeline | Implemented (partial contextual step) | `alias_registry.py:293-357` (exact/normalized/regulatory/stem → fuzzy rapidfuzz, threshold 85, review band 70); `graph_writer.py:358-435` (→ embedding auto-merge ≥0.92 → review band 0.85) | **Confirmed**: embedding auto-merge fires on cosine similarity alone with no corroborating signal — in-code comment literally says "fail open." No distinct contextual-validation (tenant/company/role/meeting) stage exists | Fix: add a contextual/co-occurrence check before auto-merging on embedding similarity alone — directly violates the spec's "never merge solely on embedding threshold" requirement |
| Resolution status/candidate persistence | Partial | `review_queue.py:24,68,99-166` — `ReviewQueueItem.status` (pending/approved/rejected) exists for ambiguous-band candidates only | Auto-resolved entities (exact/fuzzy≥85/embedding≥0.92) get **no** `resolution_status`/`resolution_method` field on the Entity node itself, and competing candidates aren't persisted for those cases | Add: `resolution_status`/`resolution_method` fields on Entity; persist runner-up candidates generally |
| Bitemporal/resolution test quality | Partial | `test_bitemporal.py`, `test_document_authority.py`, `test_contradiction_detector.py`, `test_alias_registry.py`, `test_entity_resolution_benchmark.py`, `test_graph_writer.py:88-95,470-517` | Bitemporal tests are fully mocked (assert query-string shape, not real behavior). **Zero** out-of-order-ingestion tests found anywhere. Entity-resolution benchmark tests a metrics harness with hardcoded outcomes, not the real pipeline. No misspelled-name or same-name-different-company test through the real resolution path | Add: real out-of-order-ingestion integration test; real same-name/different-org disambiguation test |

---

## 4. GraphRAG retrieval reliability

| Capability | Status | Evidence | Problems | Required action |
|---|---|---|---|---|
| Pipeline shape (BM25 → vector → RRF → rerank → multi-hop → GNN) | Implemented | `local_search.py:212-638`; `hybrid_retriever.py:361-927` orchestrates local+global; agentic/IRCoT-style fallback at `:803-871` | None structural | Keep |
| Insufficient-context / abstention path | Partial | `sufficiency.py:22-73`, wired at `hybrid_retriever.py:698-725` — computes `sufficient`/`reason_code` | **The actual abstention message only fires if `retrieval_sufficiency_abstain_enabled=true`, which defaults to `false`** (`config/settings.yml:95`) — by default, the LLM still answers even when evidence is judged insufficient. This directly contradicts the spec's "explicit insufficient-context behavior" requirement | Fix: flip default to `true` for production tenants, or gate on `policy_result=ESCALATE` at the API layer |
| Traversal-depth limits | Implemented | Query-class policy (`traversal_policy.py:29-52`) plus hard Cypher clamp `hops = min(max(hops,1),8)` (`neo4j_client.py:1927`) | None | Keep |
| Relationship-type allowlist | **Missing** | Schema has a single generic `RELATES_TO` type with a free-text `relation` property; `get_multihop_chunks` traverses `RELATES_TO*1..{hops}` with no filter on `relation` value | Any semantic relation, including low-confidence/inferred ones, is traversable during multi-hop expansion | Add: allowlisted-`relation` WHERE clause per query class where relevant |
| Stale-fact / ambiguity penalties | Partial | Time-based confidence half-life decay exists (`gnn_scorer.py:236-245`) | **Defaults to disabled** (`confidence_half_life_days=0`, `local_search.py:117`). Conflict handling is binary escalate/allow, not a graded ranking penalty | Fix: enable half-life decay with a tuned per-tenant value if staleness should affect ranking |
| Evidence-coverage measurement | **Missing** | `evidence_bundle.py:24-59` only counts IDs, not a coverage ratio; `claim_verifier.py:80-122` strips ungrounded sentences but only logs a count, is opt-in, and doesn't persist a ratio | No standalone "% of answer backed by evidence" score exists anywhere; faithfulness is delegated entirely to RAGAS in eval scripts, separate from production answers | Add: `n_removed / total_sentences` coverage score persisted on `QueryResult` |

---

## 5. Evaluation and release gates

| Capability | Status | Evidence | Problems | Required action |
|---|---|---|---|---|
| Retrieval IR metrics (Recall@k, MRR, nDCG) | Implemented (but disconnected from gates) | `graphrag/evaluation/ir_metrics.py:21-115` — precision@k, recall@k, reciprocal_rank, average_precision, ndcg_at_k; consumed by `scripts/run_retrieval_quality_eval.py`, `scripts/eval_hop_ranking.py`, `graphrag/context_graph/evaluation.py` | The "release-gate" scripts (`run_golden_eval.py`, `_aerospace_regression.py`, `run_automotive_eval.py`) compute **only pass/fail + faithfulness/citation-recall**, never ranked-list IR metrics — a recall@k/nDCG regression would not fail any gate | Fix: fold `ir_metrics.py` scores into the golden/aerospace/automotive gate scripts with their own thresholds |
| Generation metrics (faithfulness, citation correctness, unsupported-claim rate) | Partial | Faithfulness via RAGAS in eval scripts; citation-recall checked in `run_golden_eval.py`/`run_automotive_eval.py` | No standalone unsupported-claim-rate metric found separate from RAGAS faithfulness | Add: unsupported-claim-rate metric |
| Graph correctness metrics (path validity, temporal correctness, superseded-fact usage, evidence coverage) | Missing | Not found as tracked metrics anywhere | None of these four specific metrics exist | Add |
| Entity-resolution metrics (precision, recall, incorrect-merge rate, unresolved rate, review rate) | Documented only | `test_entity_resolution_benchmark.py:7-25` exercises a metrics harness, but with hardcoded mocked outcomes — not measured against the real pipeline | No real, current incorrect-merge-rate or review-rate number exists | Add: run the harness against real resolution output, not mocks |
| Agent-trajectory metrics | Missing | No tool-selection-correctness, argument-validity, or escalation-rate metrics found | — | Add |
| Security eval metrics (leakage, injection success, bypass rates) | Partial | Adversarial tests exist (§2) but are pass/fail unit tests, not tracked rate metrics feeding a gate | No dashboard/gate tracks these as ongoing rates | Add: wire security test results into a tracked gate, not just CI pass/fail |
| Operations metrics (p50/p95 latency, error rate, retry rate, DLQ rate, token/cost) | Partial | Latency spans exist (§ below); DLQ/retry counters real (`rabbitmq_client.py`) | Cost is not real (§7) | Fix: see cost section |
| Release gates in config, not code | Implemented | `evals/golden_set.json` → `min_context_precision`, `min_faithfulness`, `min_relevancy`, `min_citation_recall`, `pass_rate_min`, read at `run_golden_eval.py:275-306`; scripts exit non-zero on breach | None | Keep |
| CI enforcement of gates | **Missing** | No `.github/workflows/` directory exists in this repo at all | Gates only run if a human manually invokes the scripts — nothing blocks a merge automatically, including for security-relevant regressions. This is the single most consequential gap against the spec's "a security regression must fail CI regardless of quality improvements" requirement, because there is no CI to fail | Add: GitHub Actions workflow running `run_golden_eval.py` / `_aerospace_regression.py` / security tests on PR, failing the build on breach |

---

## 6. Observability and failure recovery

| Capability | Status | Evidence | Problems | Required action |
|---|---|---|---|---|
| OpenTelemetry spans | Partial | Real spans: `observability/tracing.py:46-80` (`trace_span`), `genai_telemetry.py` (`llm_call_span`); wired at `api/main.py:133` (HTTP), `messaging/consumers.py:74` (queue), `llm_client.py:670,689` (every LLM call), `evaluation_agent.py:174` | Tracing is **entirely no-op unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set**, with nothing enforcing that in prod config. No span wraps individual retrieval sub-stages (BM25/vector/GNN/rerank) inside `local_search.py` — only outer HTTP/consume/LLM-call spans exist | Add: stage-level spans inside `LocalSearch.search()`; confirm OTEL env var is set in deployment config |
| Prometheus metrics | Implemented | Real counters: `cost_attribution.py:24-29`, `access_control_metrics.py` | None | Keep |
| Grafana / Langfuse | Missing | Zero code hits outside docs/deps | Not implemented despite possible doc references | Documented only → no action unless required |
| Idempotency | Implemented | Query decision traces keyed by `sha256(tenant:query_id)` with existence check (`hybrid_retriever.py:191-195`) | None | Keep |
| Retry + backoff | Implemented | `rabbitmq_client.py:254-306` — exponential backoff `min(2**retries,30)`, `x-retry-count` header | None | Keep |
| DLQ | Partial | Real DLQ envelope with retry/DLQ metrics (`rabbitmq_client.py:234-290`) | DLQ envelope stores only a **truncated payload summary** (8 keys / 80 chars each) — genuine poison-message replay from the DLQ is not possible without the original payload | Fix: store the full original message body (or a pointer) alongside the DLQ envelope |
| Liveness endpoint (`/health`) | Implemented | `api/main.py:230-233` — pure liveness stub, always 200 | Correctly minimal | Keep |
| Readiness endpoint (`/health/ready`) | Partial | `api/main.py:236-299+` — genuinely checks Neo4j (`RETURN 1`), Redis (`ping()`), and the LLM provider fallback chain, returns 503 on failure | **RabbitMQ is not checked** despite being a hard dependency for the async query path | Fix: add RabbitMQ connectivity check to `/health/ready` |

---

## 7. Cost and operational reporting

| Capability | Status | Evidence | Problems | Required action |
|---|---|---|---|---|
| Token tracking | Implemented | `genai_telemetry.py:108-163` — `record_token_usage`/`record_llm_usage` tracked per call | None | Keep |
| Cost-per-request tracking | **Missing (real values)** | `cost_attribution.py:43-59` `record_cost_event`/`CostEvent`, called from `hybrid_retriever.py:492-496,749-752` — **both call sites hardcode `cost_usd=0.0`** | Cost is a permanent placeholder even during real paid DeepSeek/Groq/Cerebras calls (independently reproduced this session — `observability.cost_event` logged `cost_usd=0.0` during genuinely billed DeepSeek activity). No pricing table exists anywhere in the repo — `llm_client.py:344-346` has only a comment citing public pricing, never used in code. Cost dashboards/budgets (`observability/budgets.py`) are cosmetic since dollar cost is never actually computed | Add: versioned per-model price table (YAML), multiply by `record_token_usage`'s input/output counts before calling `record_cost_event` |
| Cost by tenant / model / strategy | Missing | Depends entirely on the above being real first | — | Add, after pricing table exists |
| Budgets/warnings | Documented only | `observability/budgets.py` exists but operates on the always-zero cost value | Effectively non-functional today | Fix as part of the pricing-table task |

---

## Summary

**Genuinely solid, not overstated in docs:** tenant derivation and query-time scoping, tool-execution-layer authorization, MCP tool allowlisting/scoping, adversarial prompt-injection test corpus, entity-resolution's fuzzy-match tier, retry/backoff/idempotency, readiness checks for Neo4j/Redis/LLM.

**Real, specific gaps found — not hypothetical, all evidence-backed:**
1. **No CI at all** (`.github/workflows/` doesn't exist) — every release gate in this repo is manually invoked, never automatically enforced.
2. **Cost tracking is entirely placeholder** (`cost_usd=0.0` hardcoded at both call sites) despite real per-call token tracking existing.
3. **Entity resolution auto-merges on embedding similarity alone**, explicitly "fail open" per its own code comment — violates the spec's core entity-resolution safety requirement.
4. **Bitemporal "current state" queries don't exclude expired facts by default** — inverted from the required default.
5. **Insufficient-context abstention is implemented but disabled by default** (`retrieval_sufficiency_abstain_enabled=false`).
6. **No opportunity/deal-level authorization concept exists at all** in this codebase — would need to be built from scratch if actually required by the domain.
7. Most "tenant isolation" and "bitemporal" tests assert **query-string shape against mocks**, not real behavior against live data — a live cross-tenant/temporal test suite mostly doesn't exist yet, despite a real ops script (`verify_tenant_isolation.py`) that could be wired into CI to close this gap directly.

This audit does not evaluate deployment infrastructure, real traffic, or external review — only what is verifiable in this repository's code and tests as of 2026-09-21.
