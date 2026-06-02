"""Load settings.yml + .env into a typed Settings object."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings


ROOT = Path(__file__).resolve().parents[2]  # repo root


def _load_yaml() -> dict:
    path = ROOT / "config" / "settings.yml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    # ── Google AI (embeddings only) ──────────────────────────────────────────────
    google_api_key: str = ""
    gemini_ingest_model: str = "gemini-2.0-flash"
    gemini_query_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "gemini-embedding-001"

    # ── Groq (text generation) ───────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    # Fast model used for cheap intermediate reasoning steps (IRCoT SEARCH/ANSWER
    # decisions). llama-3.1-8b-instant runs at ~800 tok/s on Groq vs ~150 tok/s
    # for 70B — cuts each reasoning step from ~1.5s to ~0.2s.
    groq_fast_model: str = "llama-3.1-8b-instant"

    # ── Optional features ───────────────────────────────────────────────────────
    # Wikidata linking: ground high-confidence entities to canonical QIDs.
    # Off by default to keep ingestion fast and avoid Wikidata rate limits.
    # Enable with WIKIDATA_LINKING=1 or wikidata_linking_enabled: true in settings.yml.
    wikidata_linking_enabled: bool = False

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

    # ── App ─────────────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    env: str = "development"

    # ── YAML config (loaded separately, merged at property access) ──────────────
    _yaml: dict = {}

    model_config = {"env_file": str(ROOT / ".env"), "extra": "ignore"}

    @model_validator(mode="after")
    def _validate_production_secrets(self) -> "Settings":
        """Fail fast if production is running with insecure defaults."""
        if self.env == "production":
            if self.jwt_secret_key == "change-me-in-production":
                raise ValueError(
                    "jwt_secret_key must be set to a strong random secret in production. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
                )
            if self.neo4j_password == "graphrag_dev":
                raise ValueError(
                    "neo4j_password must be changed from the default 'graphrag_dev' in production."
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

    @property
    def evaluation(self) -> dict:
        return self._yaml.get("evaluation", {})

    @property
    def ontology(self) -> dict:
        return self._yaml.get("ontology", {})

    @property
    def business_matrix(self) -> dict:
        return self._yaml.get("business_matrix", {})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
