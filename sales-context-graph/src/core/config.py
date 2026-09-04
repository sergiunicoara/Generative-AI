# Forked from ai-knowledge-graph-platform (graphrag/core/config.py) — trimmed to
# sales-context-graph's actual env surface (see .env.example). Dropped fields with
# no analog here: google/openai/deepseek/groq-specific knobs, wikidata_linking_enabled,
# llm_cache_enabled, llm_ingest_provider, rabbitmq_url, session/oauth/cors settings
# (a real IdP is still deferred per docs/plan.md §13 — workspace_api_keys below is
# an MVP API-key-per-workspace stand-in, not that).
#
# The forked config/settings.yml loader and its Settings.ontology /
# Settings.ingestion accessors were removed once their only readers (the legacy
# src/graph/* cluster: ontology_registry, alias_registry, contradiction_strategies)
# were deleted. config/settings.yml has never existed in this repo, so the loader
# only ever returned {}. config/ontologies/sales.yml is unaffected -- it is read
# directly by the live src/graph/sales_ontology.py, not through Settings.

"""Load .env into a typed Settings object."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import structlog
from pydantic import model_validator
from pydantic_settings import BaseSettings

log = structlog.get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]  # repo root


class Settings(BaseSettings):
    # ── LLM provider for structured extraction (P3) + the Increment 15 LLM layer ──
    # llm_provider="anthropic" + a non-empty llm_api_key is what src/llm/chat.py
    # requires to build a real ChatFn. Left blank, every LLM-backed route returns
    # 503 rather than a fabricated answer (see src/llm/chat.py).
    llm_provider: str = ""
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-5"
    llm_max_output_tokens: int = 2048
    # Empty means "use the real vendor endpoint" (the SDK's own default) --
    # only set for loadtest/'s LLM-call-concurrency layer (Phase 10,
    # docs/evaluation.md's B6), which points this at a local mock server
    # (loadtest/mock_llm_server.py) instead of spending real API calls
    # under load. See src/llm/chat.py::build_chat_fn().
    llm_base_url: str = ""

    # Optional voice output for the sales-assistant answer. Disabled by
    # default so text answers never depend on an external audio vendor.
    # `tts_provider=openai` enables the OpenAI speech endpoint in
    # src/tts/provider.py; callers still receive the answer text immediately.
    tts_provider: str = ""
    tts_api_key: str = ""
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    tts_base_url: str = "https://api.openai.com/v1"
    tts_timeout_seconds: float = 2.0

    # ── Embedding provider (candidate generation / vector retrieval) ─────────────
    embedding_provider: str = ""
    embedding_api_key: str = ""

    # ── Neo4j ─────────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "scg_dev_local"  # noqa: S105 -- a dev default, not a real secret; _validate_production_secrets below refuses to boot in production with it unchanged

    # ── Auth (MVP: API key per workspace, see api/dependencies.py::verify_api_key) ─
    # JSON map workspace_id -> secret key, e.g. WORKSPACE_API_KEYS='{"ws-demo":"..."}'.
    # pydantic-settings JSON-decodes dict-typed fields from a single env var.
    workspace_api_keys: dict[str, str] = {}

    # Temporary, explicitly opt-in access for a read-only public product demo.
    # Never enable this in production; the demo key is intentionally separate
    # from the real workspace key map and is scoped to one synthetic workspace.
    demo_public_access_enabled: bool = False
    demo_public_workspace_id: str = "ws-demo"
    demo_public_api_key: str = ""
    demo_public_tts_enabled: bool = False

    # ── OIDC/JWT SSO (docs/evaluation.md's Showpad engineering-rigor ────────────
    # assessment, Band 2: "no SSO/SAML/OIDC/SCIM") -- see src/auth/sso.py.
    # Off by default; when enabled, api.dependencies.verify_api_key switches
    # standard API routes to validated JWT/JWKS claims. A real external IdP account
    # (Okta/Auth0/Azure AD/...) is outside what this repo can stand up itself;
    # what's real here is the validation logic (real JWT/JWKS signature,
    # issuer, audience, and expiry checks), tested against a locally-generated
    # keypair -- connecting a real IdP later is 3 env vars, not new code.
    sso_enabled: bool = False
    sso_issuer: str = ""
    sso_audience: str = ""
    sso_jwks_url: str = ""
    # Which JWT claim carries the tenant identity this app maps to
    # workspace_id -- IdPs vary (Okta/Auth0 commonly use a custom claim
    # namespaced to your app; Azure AD AAD B2C might use "tid"). Configurable
    # rather than hardcoded to one vendor's convention.
    sso_workspace_claim: str = "workspace_id"
    # When enabled, API routes require actor claims supplied by the verified
    # gateway and resource policies become deny-by-default. Keep false for
    # local API-key demos; production deployments should turn it on together
    # with SSO/IdP claim mapping.
    authz_enforcement_enabled: bool = False
    # When claims are forwarded by an ingress/gateway rather than validated in
    # this process, require an explicit deployment declaration. This prevents
    # a client from self-asserting X-User-Roles on an accidentally enabled API.
    authz_trusted_gateway_enabled: bool = False
    # Production may legitimately run with authz_enforcement_enabled=False --
    # §13's identity provider is still deferred, and get_access_context() 503s
    # every request if enforcement is on without SSO or a trusted gateway, so
    # forcing it true here would take the deployment down rather than secure
    # it. What must not happen is *silently* landing there: every other
    # production check below refuses to boot on an unsafe default, and before
    # this flag existed a production deploy that simply never mentioned
    # authorization booted with it off and said nothing. Setting this to true
    # is the operator writing down "yes, this deployment runs without
    # application-level authorization" on purpose.
    authz_enforcement_disabled_ack: bool = False

    # Authenticated Streamable-HTTP MCP.  It uses the same per-request API
    # key/JWT boundary as the REST API; the local CRM emulator is opt-in and
    # remains synthetic even when this endpoint is enabled.
    mcp_enabled: bool = False
    mcp_request_max_bytes: int = 65536
    local_crm_emulator_path: str = "data/runtime/local_crm_emulator.json"
    # Empty is allowed only for local Compose convenience. Production startup
    # requires this token and api.main protects /metrics with it.
    metrics_api_key: str = ""

    # ── Redis (durable ingestion job store, see api/state.py::get_ingestion_store) ─
    # Empty means "no Redis configured" -> falls back to InMemoryIngestionStore.
    redis_url: str = ""

    # Durable ingestion execution is opt-in for local development and must be
    # enabled in production together with the worker service.
    ingestion_queue_enabled: bool = False
    ingestion_queue_max_attempts: int = 3
    ingestion_worker_heartbeat_seconds: int = 60
    # Number of independent Redis claims executed by one worker process.
    # Each slot has its own processing list, so a slow transcript cannot hold
    # the only claim slot hostage. Horizontal replicas remain supported; this
    # setting is a bounded local concurrency knob, not a fairness guarantee.
    ingestion_worker_concurrency: int = 1
    # Which ExtractionProvider the API route and the worker construct. Default
    # stays "fixture" -- the deterministic regex provider this repo has always
    # run in-process, so this setting changes nothing until an operator opts
    # in. "llm" selects LlmExtractionProvider, which is what actually routes
    # window text through redact_pii() and the injection guardrail (both are
    # implemented inside that provider, so on the fixture path they are inert
    # by construction, not by accident).
    extraction_provider: Literal["fixture", "llm"] = "fixture"
    # Bounds how many extraction windows of a single transcript may have an
    # in-flight provider call at once. 1 preserves the previous strictly
    # sequential behaviour. Kept small by default: a transcript fans out to
    # many windows, and an unbounded gather would hand the LLM vendor (and
    # Neo4j, via the claim writes that follow) a burst sized by transcript
    # length rather than by anything this system chose.
    extraction_max_concurrency: int = 4
    # Phase 4 (docs/evaluation.md's ingestion-reliability item, ADR-0001's
    # addendum) -- SQS-style visibility timeout: how long a job may sit
    # claimed-but-unfinished in a worker's own processing list before the
    # reaper assumes that worker crashed and puts it back on the main
    # queue. Must comfortably exceed the slowest real job (transcript
    # extraction, the only LLM-backed ingestion kind) -- 5 minutes default.
    ingestion_visibility_timeout_seconds: int = 300

    # ── Kafka transport (Phase 8, feature-flagged, off by default) ─────────────
    # Added per explicit stakeholder direction -- docs/adr-0003-kafka-event-bus.md
    # documents why this was originally judged premature and what changed the
    # decision. "redis" (default) is the recommended path at this system's
    # current scale; src/ingestion/queue.py's reliable-queue pattern (Phase 4)
    # is unaffected either way.
    ingestion_transport: Literal["redis", "kafka"] = "redis"
    kafka_bootstrap_servers: str = "localhost:9095"

    # ── Qdrant secondary vector store (Phase 8, feature-flagged) ────────────────
    # Added per explicit stakeholder direction -- docs/adr-0004-qdrant-
    # secondary-vector-store.md. Neo4j's native vector index
    # (contact_embeddings_v1) remains primary; this is a standalone, optional
    # capability (src/embedding/qdrant_backend.py), not wired into the main
    # entity-resolution candidate pipeline.
    vector_backend: Literal["neo4j", "qdrant"] = "neo4j"
    qdrant_url: str = "http://localhost:6335"

    # ── LLM gateway fallback (Phase 8, feature-flagged) ─────────────────────────
    # Added per explicit stakeholder direction -- docs/adr-0005-llm-gateway-
    # fallback.md documents why this was originally judged premature (a
    # fallback chain adds a silent-degradation path, exactly the failure mode
    # src/llm/chat.py's LlmNotConfiguredError exists to avoid) and how
    # src/llm/gateway.py mitigates it: fallback only on transient/
    # availability errors, never on a validation/schema failure, and every
    # fallback event is logged at warning + counted, never silent. Off by
    # default -- disabled, the primary provider's existing "fail loud with
    # 503" behavior is completely unchanged.
    llm_fallback_enabled: bool = False
    llm_fallback_provider: str = ""
    llm_fallback_api_key: str = ""
    llm_fallback_model: str = ""

    # ── Query result cache (Phase 5, docs/evaluation.md's semantic/result-cache ──
    # item) -- exact-match, workspace-scoped, see src/core/cache/query_cache.py.
    # On by default; still a no-op wherever REDIS_URL is unset, same fail-open
    # shape as everything else optional-Redis in this codebase.
    query_cache_enabled: bool = True
    query_cache_ttl_seconds: int = 300

    # ── Per-tenant rate limiting (docs/evaluation.md's Showpad engineering-
    # rigor assessment, 2026-08-08, Band 2) -- see src/core/rate_limit.py.
    # Closes a real, previously-undocumented gap ("no rate limiting, quotas,
    # or request-size limits -- anywhere"), so on by default, unlike the
    # Phase 8 items that stayed off pending a measured need. 120/minute is a
    # generous placeholder for a single small-team workspace, not a
    # measured capacity figure -- same honesty standard applied to every
    # other unmeasured default in this codebase (see docs/evaluation.md's
    # "Vendor SLO targets" note).
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 120

    # ── PII redaction at egress (Phase 6, docs/evaluation.md's PII item) ────────
    # See src/redaction/pii.py's module docstring for the locked-in design:
    # raw text stays verbatim at rest (evidence-model requirement), redacted
    # only at LLM-prompt and log egress points. On by default.
    pii_redaction_enabled: bool = True

    # ── Prompt-injection guardrail (Phase 6, additive to the existing ───────────
    # structural defenses in src/extraction/prompt.py) — see
    # src/extraction/guardrail.py. log_only (default) flags and metrics a
    # suspected injection attempt but never blocks extraction; "block" rejects
    # the window outright. log_only is the locked-in default: a probabilistic
    # heuristic classifier becoming a new hard-failure mode on top of the
    # existing deterministic defenses is a worse trade than staying
    # observability-only until real data justifies blocking.
    guardrail_enforcement_mode: Literal["log_only", "block"] = "log_only"

    # ── Vector index population + reranker (Phase 7, docs/evaluation.md's ──────
    # B5 item) — off by default until both the tenant-filter fix (Phase 1,
    # already shipped) and a real backfill (src/embedding/backfill.py) are
    # verified live in an environment. See src/context_graph/reranker.py.
    reranker_enabled: bool = False

    # ── Proactive digest (Increment 17, see src/usecases/digest.py) ──────────────
    # Empty slack_webhook_url means POST /api/v1/digest/deliver returns 503 rather
    # than posting nowhere; GET /api/v1/digest (JSON, no delivery) works regardless.
    slack_webhook_url: str = ""
    digest_stale_share_days: int = 7
    digest_stalled_deal_days: int = 21

    # ── Threshold alerting (docs/evaluation.md's Showpad engineering-rigor ──────
    # assessment, Band 4: "metrics without alerts") -- see
    # src/core/alerting.py. Reuses slack_webhook_url above, same "JSON
    # always available, Slack delivery only if configured" split as the
    # digest feature. Thresholds are unmeasured placeholders, same honesty
    # standard as rate_limit_requests_per_minute above -- not a claim of a
    # measured capacity figure.
    alert_max_queue_depth: int = 100
    alert_max_oldest_job_age_seconds: int = 900
    # A DLQ is meant to be near-empty at all times -- unlike the other two
    # thresholds, any nonzero steady-state count here usually means "go
    # look," not "getting busy." Still configurable rather than hardcoded
    # to 0, in case a deployment wants to batch-review a small backlog.
    alert_max_dlq_depth: int = 5

    # ── Embeddable panel (Increment 20, see api/routes/viz.py's /viz/panel) ──────
    # Space-separated list of origins allowed to iframe /viz/panel (sets
    # Content-Security-Policy: frame-ancestors). Empty means no origin is
    # allowed to embed it — the panel still works when opened directly, but
    # embedding is denied by default rather than left open.
    embed_allowed_origins: str = ""

    # HMAC secret for the panel token (src/viz/panel_tokens.py) that replaced
    # putting the real workspace API key in /viz/panel's URL (docs/
    # evaluation.md's Showpad-compatibility analysis, item 3). Empty means
    # /viz/panel-token (the minting endpoint) and /viz/panel itself both
    # return 503 rather than issuing/accepting an unsigned or weakly-signed
    # token — same "fail loud, never silently degrade" posture as
    # src/llm/chat.py's LlmNotConfiguredError.
    panel_token_secret: str = ""
    # ~1 year: the panel is a static, admin-configured iframe src with no
    # OAuth flow to refresh it (see README.md), so this deliberately isn't a
    # short-lived session token — revocation (bumping the stored version in
    # Redis, see panel_tokens.py) is the mechanism for invalidating one
    # early, not a short TTL forcing every embed to be re-minted constantly.
    panel_token_ttl_seconds: int = 60 * 60 * 24 * 365

    # Buyer-space binary uploads are written outside Neo4j and are never
    # accepted unscanned.  ``buyer_upload_scanner=clamav`` enables the clamd
    # INSTREAM protocol; the endpoint returns 503 when disabled/unavailable
    # instead of quietly persisting an unscanned file.
    buyer_upload_storage_dir: str = "data/buyer_uploads"
    buyer_upload_max_bytes: int = 10 * 1024 * 1024
    buyer_upload_scanner: Literal["disabled", "clamav"] = "disabled"
    buyer_upload_clamav_host: str = "localhost"
    buyer_upload_clamav_port: int = 3310
    buyer_upload_retention_days: int = 90

    # ── App ───────────────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    env: str = "development"

    model_config = {"env_file": str(ROOT / ".env"), "extra": "ignore"}

    @model_validator(mode="after")
    def _validate_production_secrets(self) -> "Settings":
        """Fail fast if production is running with insecure defaults."""
        if self.env == "production" and self.neo4j_password == "scg_dev_local":  # noqa: S105 -- comparing against the known dev default, not a hardcoded credential
            raise ValueError(
                "neo4j_password must be changed from the default 'scg_dev_local' in production."
            )
        if self.env == "production" and not self.workspace_api_keys:
            raise ValueError(
                "workspace_api_keys must be configured in production (WORKSPACE_API_KEYS)."
            )
        if self.env == "production" and not self.panel_token_secret:
            raise ValueError(
                "panel_token_secret must be configured in production (PANEL_TOKEN_SECRET) "
                "for /viz/panel to mint or verify tokens."
            )
        if self.env == "production" and self.demo_public_access_enabled:
            raise ValueError("demo_public_access_enabled must remain disabled in production.")
        if self.env == "production" and not self.metrics_api_key:
            raise ValueError("metrics_api_key must be configured in production (METRICS_API_KEY).")
        if self.env == "production" and not self.authz_enforcement_enabled and not self.authz_enforcement_disabled_ack:
            raise ValueError(
                "production must make an explicit choice about application authorization: set "
                "AUTHZ_ENFORCEMENT_ENABLED=true (requires SSO_ENABLED or "
                "AUTHZ_TRUSTED_GATEWAY_ENABLED), or set "
                "AUTHZ_ENFORCEMENT_DISABLED_ACK=true to run deliberately without it."
            )
        # Catch at boot what api.dependencies.get_access_context() would
        # otherwise only reveal as a 503 on the first real request.
        if self.authz_enforcement_enabled and not (self.sso_enabled or self.authz_trusted_gateway_enabled):
            raise ValueError(
                "authz_enforcement_enabled requires SSO_ENABLED or AUTHZ_TRUSTED_GATEWAY_ENABLED; "
                "without one of them every authorized route returns 503."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
