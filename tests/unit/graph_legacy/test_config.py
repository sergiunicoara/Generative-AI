"""src/core/config.py must work with no .env and no config/settings.yml present —
neither file exists in this repo yet (settings.yml is deferred to a later phase,
see src/core/config.py's own comment), and the original graphrag/core/config.py
this was forked from crashed on a missing settings.yml. That crash must not
survive the fork.
"""

import pytest

from src.core.config import Settings, get_settings

_ENV_KEYS = (
    "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
    "LLM_PROVIDER", "LLM_API_KEY",
    "EMBEDDING_PROVIDER", "EMBEDDING_API_KEY",
    "LOG_LEVEL", "ENV",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_settings_works_with_no_env_file_and_no_settings_yaml():
    settings = get_settings()
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password == "scg_dev_local"
    assert settings.env == "development"


def test_missing_settings_yaml_degrades_to_empty_sections():
    settings = get_settings()
    assert settings.ontology == {}
    assert settings.ingestion == {}


def test_get_settings_is_a_cached_singleton():
    assert get_settings() is get_settings()


def test_production_with_default_password_raises():
    with pytest.raises(ValueError, match="neo4j_password"):
        Settings(env="production", neo4j_password="scg_dev_local")


def test_production_with_changed_password_does_not_raise():
    settings = Settings(env="production", neo4j_password="a-real-secret")
    assert settings.env == "production"
