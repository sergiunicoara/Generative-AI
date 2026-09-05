"""Prometheus metrics for the 9 named observability targets in
docs/plan.md Section 14 ("Metrics:" bullet list). Every metric below is
instantiated once at import time; call sites increment/observe it directly
rather than going through an indirection layer, matching how
`src/graph/alias_registry.py` and friends already reach for module-level
singletons rather than a DI container.

Cardinality note (docs/plan.md Section 14, verbatim): "Avoid
unbounded-cardinality metric labels. Workspace-level operational detail
should be available through controlled logs/traces or bounded reporting,
not arbitrary metric labels at large tenant counts." None of the metrics
below carry `workspace_id` (or any other high-cardinality field) as a
label for exactly that reason -- per-tenant detail belongs in structured
logs (src/core/logging.py), not here. Every label used has a small, fixed
set of values (job kind, decision status, truncation reason, etc).

Serving: GET /metrics (api/main.py) renders the default
`prometheus_client` registry via `generate_latest()` -- no collector
service, no new deployed infrastructure, a plain scrape target.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- 1. Ingestion count and duration by status -----------------------------
INGESTION_JOBS_TOTAL = Counter(
    "scg_ingestion_jobs_total",
    "Ingestion jobs that reached a terminal state, by kind and status.",
    ["kind", "status"],
)
INGESTION_JOB_DURATION_SECONDS = Histogram(
    "scg_ingestion_job_duration_seconds",
    "Ingestion job processing duration in seconds, by kind.",
    ["kind"],
)

# --- 2/3. Extraction windows, provider calls, failures and retries ---------
EXTRACTION_WINDOWS_TOTAL = Counter(
    "scg_extraction_windows_total",
    "Extraction windows built from a transcript (src/extraction/windowing.py).",
)
EXTRACTION_PROVIDER_CALLS_TOTAL = Counter(
    "scg_extraction_provider_calls_total",
    "LLM extraction provider call attempts, by outcome.",
    ["outcome"],  # success | retry | permanent_failure
)

# --- 4. Candidate-generation latency ----------------------------------------
CANDIDATE_GENERATION_DURATION_SECONDS = Histogram(
    "scg_candidate_generation_duration_seconds",
    "Entity-resolution candidate generation latency: DB fetch plus "
    "blocking/narrowing (src/resolution/pipeline.py's union_candidates call).",
)

# --- 5. Blocking recall (evaluation runs only) ------------------------------
BLOCKING_RECALL = Gauge(
    "scg_blocking_recall",
    "Fraction of gold-standard entities retained after blocking, from the "
    "most recently scored evaluation run. There is no ground truth in live "
    "traffic to measure recall against -- this is set by an eval harness "
    "(tests/eval/*), not sampled from request handling. See "
    "record_blocking_recall() below.",
)

# --- 6. Auto-link, review, unresolved, and rejection counts ----------------
RESOLUTION_DECISIONS_TOTAL = Counter(
    "scg_resolution_decisions_total",
    "Mention resolution decisions, by outcome (src/resolution/policy.py::decide).",
    ["status"],  # auto_linked | pending_review | unresolved | rejected
)

# --- 7. Claims created, superseded, conflicted, erased, and adjudicated ----
CLAIMS_TOTAL = Counter(
    "scg_claims_total",
    "Claim lifecycle events, by event type.",
    ["event"],  # created | superseded | conflicted | erased | adjudicated
)

# --- 8. Context Graph latency, result count, and budget truncation ---------
CONTEXT_GRAPH_BUILD_DURATION_SECONDS = Histogram(
    "scg_context_graph_build_duration_seconds",
    "src/context_graph/builder.py ContextGraphBuilder.build() latency.",
)
CONTEXT_GRAPH_RESULT_COUNT = Histogram(
    "scg_context_graph_result_count",
    "Claims selected into a built Context Graph.",
    buckets=(1, 2, 5, 10, 20, 50, 100),
)
CONTEXT_GRAPH_TRUNCATED_TOTAL = Counter(
    "scg_context_graph_truncated_total",
    "Context Graph builds that hit a budget cap before exhausting the "
    "scored candidate list, by which cap was hit.",
    ["reason"],  # max_nodes | max_tokens
)

# --- 9. Queue depth and oldest-job age --------------------------------------
INGESTION_QUEUE_DEPTH = Gauge(
    "scg_ingestion_queue_depth",
    "Jobs currently waiting in the ingestion queue (src/ingestion/queue.py).",
)
INGESTION_QUEUE_OLDEST_JOB_AGE_SECONDS = Gauge(
    "scg_ingestion_queue_oldest_job_age_seconds",
    "Age in seconds of the oldest job still waiting in the ingestion "
    "queue. 0 when the queue is empty.",
)
# Added after a review of this repo's own reliability posture (2026-08-08):
# queue.py::queue_health() has computed this count since Phase 4, but only
# ever returned it in GET /ready's JSON body -- nothing pushed it to a
# metric, so a job permanently failing and landing in the DLQ was
# observable only by someone manually polling /ready. Populated by
# sample_queue_metrics() alongside the two gauges above (same Redis round
# trip cadence), and checked by src/core/alerting.py.
INGESTION_DLQ_DEPTH = Gauge(
    "scg_ingestion_dlq_depth",
    "Jobs currently sitting in the ingestion dead-letter list "
    "(src/ingestion/queue.py) -- each one failed permanently or exhausted "
    "its retry budget and needs a human to look at it.",
)

# --- Phase 6 addition: prompt-injection guardrail flags ---------------------
# Not one of docs/plan.md §14's original 9 -- added when the guardrail
# itself was (src/extraction/guardrail.py), same "count every real signal
# this system produces" discipline as the rest of this file.
GUARDRAIL_FLAG_TOTAL = Counter(
    "scg_guardrail_flag_total",
    "Extraction windows flagged by the prompt-injection heuristic guardrail, "
    "regardless of enforcement mode (log_only vs block).",
)

# --- Phase 8 addition: LLM gateway fallback events --------------------------
# Not one of docs/plan.md §14's original 9 -- added alongside
# src/llm/gateway.py. A fallback chain is a silent-degradation risk this
# codebase has otherwise refused (src/llm/chat.py's LlmNotConfiguredError
# fails loud rather than degrading); this counter is the mitigation --
# every fallback is loud, never silent. See docs/adr-0005-llm-gateway-
# fallback.md.
LLM_FALLBACK_TOTAL = Counter(
    "scg_llm_fallback_total",
    "Times the LLM gateway fell back from the primary provider to the "
    "configured secondary provider, by provider pair and reason.",
    ["from_provider", "to_provider", "reason"],
)

# --- Rate limiting (docs/evaluation.md's Showpad engineering-rigor
# assessment, 2026-08-08, Band 2) -- deliberately unlabeled: a per-
# workspace_id label here would violate this file's own cardinality rule
# (see the module docstring above); a workspace hitting its limit
# repeatedly is visible in structured logs (src/core/rate_limit.py),
# which is the right place for that unbounded-cardinality detail.
RATE_LIMIT_REJECTED_TOTAL = Counter(
    "scg_rate_limit_rejected_total",
    "Requests rejected for exceeding the per-workspace rate limit.",
)

# --- Async transcript ingestion: per-window visibility -----------------------
# INGESTION_JOB_DURATION_SECONDS above measures a whole job; a transcript job
# is a fan-out over N extraction windows, so a slow job gives no clue whether
# one window stalled or every window is uniformly slow. These three close that
# gap. Deliberately unlabeled by workspace_id (this file's cardinality rule).
EXTRACTION_WINDOW_DURATION_SECONDS = Histogram(
    "scg_extraction_window_duration_seconds",
    "Wall-clock duration of a single extraction window's provider call, "
    "including any in-provider retries.",
)
EXTRACTION_WINDOWS_FAILED_TOTAL = Counter(
    "scg_extraction_windows_failed_total",
    "Extraction windows that exhausted their retries and were skipped. The "
    "job continues with the remaining windows -- one bad window must not "
    "discard an entire transcript -- so this is the only signal that "
    "partial extraction occurred.",
)
RESOLUTION_DURATION_SECONDS = Histogram(
    "scg_resolution_duration_seconds",
    "Wall-clock duration of one entity-resolution decision during ingestion "
    "(candidate generation, scoring and policy), by outcome status.",
    ["status"],
)
RESOLUTION_CANDIDATES_CONSIDERED = Histogram(
    "scg_resolution_candidates_considered",
    "How many candidates a single resolution decision scored. A persistent 0 "
    "means blocking is surfacing nothing and every mention is failing safe to "
    "UNRESOLVED -- indistinguishable from 'no matches exist' in the decision "
    "counter alone.",
    buckets=(0, 1, 2, 5, 10, 25, 50),
)
INGESTION_QUEUE_RETRIES_TOTAL = Counter(
    "scg_ingestion_queue_retries_total",
    "Jobs requeued after a retryable failure, by kind. Distinct from "
    "scg_extraction_provider_calls_total{outcome=\"retry\"}, which counts "
    "in-provider JSON-repair attempts rather than whole-job redeliveries.",
    ["kind"],
)

# --- Sales MCP and local CRM command surface --------------------------------
# No tenant identifier is used as a label: per-tenant diagnosis stays in the
# structured audit logs, while metrics remain safe to operate at tenant scale.
MCP_REQUESTS_TOTAL = Counter(
    "scg_mcp_requests_total",
    "MCP requests by method, bounded capability and outcome.",
    ["method", "capability", "outcome"],
)
MCP_REQUEST_DURATION_SECONDS = Histogram(
    "scg_mcp_request_duration_seconds",
    "MCP request latency by bounded capability.",
    ["capability"],
)
CRM_COMMANDS_TOTAL = Counter(
    "scg_crm_commands_total",
    "Synthetic/local CRM command outcomes by operation.",
    ["operation", "outcome"],
)
GROUNDED_RECOMMENDATIONS_TOTAL = Counter(
    "scg_grounded_recommendations_total",
    "Grounded sales recommendation versus abstention outcomes.",
    ["outcome"],
)

# --- Diagnostics invariant checks (src/diagnostics/invariants.py) ----------
INVARIANT_CHECKS_TOTAL = Counter(
    "scg_invariant_checks_total",
    "Structured invariant checks, by stage, check name, and outcome.",
    ["stage", "name", "outcome"],
)


def record_blocking_recall(value: float) -> None:
    """Called by tests/eval/* after scoring a labeled run's blocking output
    against ground truth. Not invoked from any request-handling path."""
    BLOCKING_RECALL.set(value)
