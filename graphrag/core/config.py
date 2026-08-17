"""Load settings.yml + .env into a typed Settings object."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings


ROOT = Path(__file__).resolve().parents[2]  # repo root

# Only these environments may run with development defaults. The secrets check
# in Settings is deliberately an allow-list rather than `env == "production"`:
# keyed on production alone, an unset or misspelled ENV ("prod", "Production ",
# "") silently fell through to the default HS256 signing key, which is written
# in plaintext in this file and would let anyone forge a valid token.
DEV_ENVS = ("development", "dev", "test", "testing", "local")


def is_dev_env(env: str) -> bool:
    """True if `env` (case/whitespace-normalized) is in the dev allow-list.

    Single source of truth for the "am I in a development environment"
    check, shared by config.py's own secret validation and every other
    exact-match site (dashboard, demo routes, dev-only auth endpoints) that
    previously each did `env == "development"` independently — an unnamed
    or differently-spelled dev env would agree with none of them, and a
    misspelled *production* env ("Production ", "prod") would agree with
    all of them, both in the fail-open direction.
    """
    return env.strip().lower() in DEV_ENVS


def _load_yaml() -> dict:
    path = ROOT / "config" / "settings.yml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_tenant_config(base: dict, tenant: str = "default") -> dict:
    """Merge a config block's ``tenant_overrides.<tenant>`` over its defaults.

    Shared by ``Settings.retrieval_for`` and the retrievers, which each already
    hold the global retrieval dict (with its ``tenant_overrides`` sub-block) and
    resolve per-tenant at query time. Mirrors ``AlertService._effective_thresholds``:
    global dict, then the tenant's overrides win.

    No override for the tenant ⇒ the base dict is returned unchanged (identity
    when it has no ``tenant_overrides`` key at all), so an empty overrides block
    is a guaranteed no-op. The ``tenant_overrides`` key is stripped from the
    result so consumers never see it as a retrieval knob.
    """
    overrides = base.get("tenant_overrides", {}).get(tenant, {})
    if not overrides:
        if "tenant_overrides" not in base:
            return base
        return {k: v for k, v in base.items() if k != "tenant_overrides"}
    merged = {k: v for k, v in base.items() if k != "tenant_overrides"}
    merged.update(overrides)  # tenant wins
    return merged


class Settings(BaseSettings):
    # ── Google AI (last-resort fallback for RAGAS judge only; embeddings use OpenAI) ──
    google_api_key: str = ""
    gemini_ingest_model: str = "gemini-2.0-flash"
    gemini_query_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "gemini-embedding-001"

    # ── OpenAI (primary embeddings + DeepSeek fallback path) ────────────────────
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-large"   # 3072d — matches schema

    # ── DeepSeek (text generation fallback) ─────────────────────────────────────
    deepseek_api_key: str = ""

    # ── Cerebras (text generation primary — free tier: 1M tokens/day, no card) ──
    # Model verified live against GET /v1/models on 2026-08-17 — Cerebras only
    # serves {gpt-oss-120b, gemma-4-31b, zai-glm-4.7}, not "llama-3.3-70b" (a
    # Groq/Meta model name that never existed on Cerebras; the original guess
    # at implementation time was wrong and every call was silently failing
    # over to paid DeepSeek ever since, defeating the whole point of this
    # provider chain — see docs/audit-2026-08-13.md).
    cerebras_api_key: str = ""
    cerebras_model: str = "gpt-oss-120b"

    # ── Groq (text generation) ───────────────────────────────────────────────────
    groq_api_key: str = ""
    # Both model names below verified live against GET /v1/models on
    # 2026-08-17 — Groq deprecated "llama-3.3-70b-versatile" and
    # "llama-3.1-8b-instant" (both 404 model_not_found now); replaced with the
    # closest available equivalents in Groq's current catalog.
    groq_model: str = "openai/gpt-oss-120b"
    # Fast model used for cheap intermediate reasoning steps (IRCoT SEARCH/ANSWER
    # decisions). No small instant-class model remains in Groq's catalog as of
    # 2026-08-17 (the whole 8B-class tier was retired) — gpt-oss-20b is the
    # smallest/fastest general chat model currently available.
    groq_fast_model: str = "openai/gpt-oss-20b"

    # ── Optional features ───────────────────────────────────────────────────────
    # Wikidata linking: ground high-confidence entities to canonical QIDs.
    # Off by default to keep ingestion fast and avoid Wikidata rate limits.
    # Enable with WIKIDATA_LINKING=1 or wikidata_linking_enabled: true in settings.yml.
    wikidata_linking_enabled: bool = False

    # Deterministic-ingestion cache: Groq/DeepSeek are NOT reproducible at
    # temperature=0 (batched GPU/LPU inference + mid-run rate-limit fallback
    # between model families), so re-ingesting the same corpus produces a
    # different graph shape every run. This memoizes raw extraction responses
    # so repeated `--wipe --commit` runs replay identical results — essential
    # for demo scripts that reference specific entity names / confidence values.
    # Off by default — production ingestion of new documents must hit the live LLM.
    # Enable with LLM_CACHE_ENABLED=1 or llm_cache_enabled: true in settings.yml.
    llm_cache_enabled: bool = False

    # Selects which provider is PRIMARY for get_llm() — the other is always
    # available as an automatic fallback (FallbackLLM), never fully bypassed.
    # Also affects LLM_CACHE_ENABLED=1 runs: a one-time cache-populating
    # ingestion run should use ONE provider's extraction "voice" for the whole
    # corpus — mixing providers mid-run (which happens naturally whenever the
    # primary fails over) pollutes the cache with two different models'
    # outputs for what should be one deterministic baseline (see llm_cache.py).
    # "" = default: get_llm() uses FallbackLLM.cerebras_primary() — Cerebras
    #      primary (free tier: 1M tokens/day, no card, LPU-fast) falling over
    #      to DeepSeek, then Groq, on failure/quota. Keeps token spend on the
    #      paid DeepSeek key to the cases where Cerebras itself is down or
    #      over its daily cap, instead of every call.
    # "deepseek" = get_llm() uses FallbackLLM.deepseek_primary() — skips
    #          Cerebras entirely, DeepSeek-V4 primary with Groq fallback.
    #          This was the default before 2026-08-17; use it if Cerebras
    #          quality/latency doesn't hold up for a given workload.
    # "groq" = get_llm() uses FallbackLLM.groq_primary() instead — Groq
    #          primary (~280 tok/s) with instant DeepSeek fallback. Useful
    #          for quick/low-volume dev runs. Enable with
    #          LLM_INGEST_PROVIDER=groq; remove/unset afterwards — this is a
    #          one-shot knob, not a permanent provider switch.
    llm_ingest_provider: str = ""

    # ── Neo4j ───────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphrag_dev"

    # ── RabbitMQ ────────────────────────────────────────────────────────────────
    rabbitmq_url: str = "amqp://graphrag:graphrag_dev@localhost:5672/"

    # ── OAuth / JWT ──────────────────────────────────────────────────────────────
    jwt_secret_key: str = "change-me-in-production"
    # Separate secret for SessionMiddleware cookie signing.
    # When empty, main.py derives one from jwt_secret_key + ":session".
    # Set explicitly in production to allow independent rotation.
    session_secret_key: str = ""
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    cors_origins: list[str] = ["http://localhost:8000", "http://localhost:8050"]

    # Tenant stamped into every JWT issued by the browser/dev flows. Requests
    # are scoped to the tenant in their token — never to a tenant supplied in
    # the request body — so this is the only place a browser session's tenant
    # is decided. M2M clients pick their tenant at registration time.
    default_tenant: str = "default"

    # ── App ─────────────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    # Deliberately NOT "development". An unset ENV var previously defaulted
    # into the dev allow-list, silently permitting every insecure default
    # below (JWT secret, Neo4j password, dev CORS) in a deployment that never
    # named its environment. "" is not in DEV_ENVS, so an unset ENV now falls
    # through to the same strict validation as a real production deployment —
    # fail closed on missing config, not just misspelled config.
    env: str = ""

    # ── YAML config (loaded separately, merged at property access) ──────────────
    _yaml: dict = {}

    model_config = {"env_file": str(ROOT / ".env"), "extra": "ignore"}

    @model_validator(mode="after")
    def _validate_production_secrets(self) -> "Settings":
        """Fail fast if a non-development environment has insecure defaults."""
        env = self.env.strip().lower()
        if env not in DEV_ENVS:
            if env == "":
                raise ValueError(
                    "ENV is not set. Set ENV=development (or dev/test/testing/local) "
                    "for local work, or ENV=production for a real deployment. "
                    "Refusing to start with development defaults in an unnamed "
                    "environment."
                )
            if self.jwt_secret_key == "change-me-in-production":
                raise ValueError(
                    "jwt_secret_key must be set to a strong random secret in production. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
                )
            if self.neo4j_password == "graphrag_dev":
                raise ValueError(
                    "neo4j_password must be changed from the default 'graphrag_dev' in production."
                )
            if not self.session_secret_key:
                raise ValueError(
                    "session_secret_key must be explicitly set in production so cookie signing keys "
                    "can be rotated independently from JWT keys."
                )
            if self.session_secret_key == self.jwt_secret_key:
                raise ValueError(
                    "session_secret_key must differ from jwt_secret_key in production."
                )
            if "graphrag_dev" in self.rabbitmq_url or "localhost" in self.rabbitmq_url:
                raise ValueError(
                    "rabbitmq_url must not use local or development credentials in production."
                )
            if any(origin.startswith("http://") or "localhost" in origin for origin in self.cors_origins):
                raise ValueError(
                    "cors_origins must contain only approved HTTPS production origins."
                )
        return self

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "_yaml", _load_yaml())

    # ── Accessors ────────────────────────────────────────────────────────────────
    @property
    def ingestion(self) -> dict:
        return self._yaml.get("ingestion", {})

    @property
    def graph(self) -> dict:
        return self._yaml.get("graph", {})

    @property
    def retrieval(self) -> dict:
        return self._yaml.get("retrieval", {})

    def retrieval_for(self, tenant: str = "default") -> dict:
        """Retrieval config with this tenant's overrides merged over the global
        defaults (see ``resolve_tenant_config``). Different corpora need
        different retrieval depths, but a global change that fixes one tenant
        has historically regressed another (see the reverts in settings.yml);
        this scopes an override to one tenant. Empty overrides ⇒ global dict."""
        return resolve_tenant_config(self.retrieval, tenant)

    @property
    def evaluation(self) -> dict:
        return self._yaml.get("evaluation", {})

    @property
    def ontology(self) -> dict:
        return self._yaml.get("ontology", {})

    @property
    def business_matrix(self) -> dict:
        return self._yaml.get("business_matrix", {})

    @property
    def context_graph(self) -> dict:
        return self._yaml.get("context_graph", {})

    @property
    def maintenance(self) -> dict:
        """Maintenance thresholds from settings.yml.

        This accessor did not exist, which made the whole ``maintenance:``
        block in config/settings.yml structurally unreachable — editing
        ``orphan_flag_enabled`` or ``stale_edge_days`` there had no effect,
        because scripts/maintenance.py hardcoded its own copies of the values.
        """
        return self._yaml.get("maintenance", {})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
