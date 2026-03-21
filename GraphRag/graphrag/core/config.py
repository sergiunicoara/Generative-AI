"""Load settings.yml + .env into a typed Settings object."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


ROOT = Path(__file__).resolve().parents[2]  # repo root


def _load_yaml() -> dict:
    path = ROOT / "config" / "settings.yml"
    with open(path) as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    # ── Google AI ───────────────────────────────────────────────────────────────
    google_api_key: str = ""
    gemini_ingest_model: str = "gemini-2.0-flash"
    gemini_query_model: str = "gemini-2.0-pro-exp"
    gemini_embed_model: str = "text-embedding-004"

    # ── Neo4j ───────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphrag_dev"

    # ── RabbitMQ ────────────────────────────────────────────────────────────────
    rabbitmq_url: str = "amqp://graphrag:graphrag_dev@localhost:5672/"

    # ── TimescaleDB ─────────────────────────────────────────────────────────────
    timescale_url: str = (
        "postgresql+asyncpg://graphrag:graphrag_dev@localhost:5432/graphrag_kpis"
    )

    # ── OAuth / JWT ──────────────────────────────────────────────────────────────
    jwt_secret_key: str = "change-me-in-production"
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    cors_origins: list[str] = ["http://localhost:8000", "http://localhost:8050"]

    # ── App ─────────────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    env: str = "development"

    # ── YAML config (loaded separately, merged at property access) ──────────────
    _yaml: dict = {}

    model_config = {"env_file": str(ROOT / ".env"), "extra": "ignore"}

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
    def business_matrix(self) -> dict:
        return self._yaml.get("business_matrix", {})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
