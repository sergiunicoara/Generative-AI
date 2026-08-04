# Forked from ai-knowledge-graph-platform (graphrag/core/config.py) — trimmed to
# sales-context-graph's actual env surface (see .env.example). Dropped fields with
# no analog here: google/openai/deepseek/groq-specific knobs, wikidata_linking_enabled,
# llm_cache_enabled, llm_ingest_provider, rabbitmq_url, session/oauth/cors settings
# (auth is explicitly deferred per docs/plan.md §13 until a real IdP exists).
#
# _load_yaml() no longer crashes when config/settings.yml is absent (it is, until
# a later phase's ontology work adds one) — it now fails open to {} with a warning,
# instead of the original's bare open()/FileNotFoundError.

"""Load settings.yml (if present) + .env into a typed Settings object."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import structlog
import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings

log = structlog.get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]  # repo root


def _load_yaml() -> dict:
    path = ROOT / "config" / "settings.yml"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except OSError as exc:
        log.warning("config.settings_yaml_load_failed", path=str(path), error=str(exc))
        return {}


class Settings(BaseSettings):
    # ── LLM provider for structured extraction (P3) ──────────────────────────────
    llm_provider: str = ""
    llm_api_key: str = ""

    # ── Embedding provider (candidate generation / vector retrieval) ─────────────
    embedding_provider: str = ""
    embedding_api_key: str = ""

    # ── Neo4j ─────────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "scg_dev_local"

    # ── App ───────────────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    env: str = "development"

    # ── YAML config (loaded separately, merged at property access) ──────────────
    _yaml: dict = {}

    model_config = {"env_file": str(ROOT / ".env"), "extra": "ignore"}

    @model_validator(mode="after")
    def _validate_production_secrets(self) -> "Settings":
        """Fail fast if production is running with insecure defaults."""
        if self.env == "production" and self.neo4j_password == "scg_dev_local":
            raise ValueError(
                "neo4j_password must be changed from the default 'scg_dev_local' in production."
            )
        return self

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "_yaml", _load_yaml())

    # ── Accessors ─────────────────────────────────────────────────────────────────
    # Only sections the ported src/graph/*.py legacy modules actually read
    # (ontology_registry.load() -> settings.ontology; alias_registry.__init__ ->
    # settings.ingestion). Add more only once a phase's code actually calls
    # get_settings().<section>.
    @property
    def ontology(self) -> dict:
        return self._yaml.get("ontology", {})

    @property
    def ingestion(self) -> dict:
        return self._yaml.get("ingestion", {})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
