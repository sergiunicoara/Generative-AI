"""src/core/config.py must work with no .env and no config/settings.yml present
(settings.yml is deferred to a later phase, see src/core/config.py's own
comment), and the original graphrag/core/config.py this was forked from crashed
on a missing settings.yml. That crash must not survive the fork.

These tests pass `_env_file=None` rather than relying on `.env` being absent
from the developer's checkout. `.env` is gitignored, so it does not exist in CI
— but the README tells developers to create one (`cp .env.example .env`), and
before this was made explicit, doing so broke two of these tests: `.env`'s
NEO4J_URI overrode the expected default, and its WORKSPACE_API_KEYS satisfied
the production check that the test asserts should fail. A config test must
assert against declared defaults, not against whatever the local machine holds.
"""

import pytest

from src.core.config import Settings, get_settings

_ENV_KEYS = (
    "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
    "LLM_PROVIDER", "LLM_API_KEY",
    "EMBEDDING_PROVIDER", "EMBEDDING_API_KEY",
    "LOG_LEVEL", "ENV", "METRICS_API_KEY", "PANEL_TOKEN_SECRET",
    "AUTHZ_ENFORCEMENT_ENABLED", "AUTHZ_TRUSTED_GATEWAY_ENABLED",
    "AUTHZ_ENFORCEMENT_DISABLED_ACK", "SSO_ENABLED",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_settings_works_with_no_env_file_and_no_settings_yaml():
    settings = Settings(_env_file=None)
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password == "scg_dev_local"
    assert settings.env == "development"


def test_get_settings_is_a_cached_singleton():
    assert get_settings() is get_settings()


def test_production_with_default_password_raises():
    with pytest.raises(ValueError, match="neo4j_password"):
        Settings(
            _env_file=None, env="production", neo4j_password="scg_dev_local",
            workspace_api_keys={"ws-1": "k"},
        )


def test_production_without_workspace_api_keys_raises():
    with pytest.raises(ValueError, match="workspace_api_keys"):
        Settings(
            _env_file=None, env="production", neo4j_password="a-real-secret",
            panel_token_secret="a-real-panel-secret",
        )


def test_production_without_panel_token_secret_raises():
    with pytest.raises(ValueError, match="panel_token_secret"):
        Settings(
            _env_file=None, env="production", neo4j_password="a-real-secret",
            workspace_api_keys={"ws-1": "k"},
        )


def test_production_with_changed_password_and_api_keys_does_not_raise():
    # _env_file=None is load-bearing here, not boilerplate (see this module's
    # docstring). Without it this test reads the developer's real `.env`, and
    # once that file gained DEMO_PUBLIC_ACCESS_ENABLED=true (for the public
    # demo), the production guard for *that* field fired and failed a test
    # that has nothing to do with it. CI never caught it -- CI has no `.env`
    # -- so it only broke on machines that followed the README's own
    # `cp .env.example .env` instruction. Exactly the failure mode this
    # module's docstring already warned about.
    settings = Settings(
        _env_file=None, env="production", neo4j_password="a-real-secret",
        workspace_api_keys={"ws-1": "k"}, panel_token_secret="a-real-panel-secret",
        metrics_api_key="a-real-metrics-key", authz_enforcement_disabled_ack=True,
    )
    assert settings.env == "production"


def test_production_without_metrics_api_key_raises():
    with pytest.raises(ValueError, match="metrics_api_key"):
        Settings(
            _env_file=None, env="production", neo4j_password="a-real-secret",
            workspace_api_keys={"ws-1": "k"}, panel_token_secret="a-real-panel-secret",
        )


def test_production_must_state_its_authorization_choice():
    """Production may run without application authz -- but not by accident.

    Every other check here refuses to boot on an unsafe *default*; before
    this one existed, a production deploy that never mentioned authorization
    inherited authz_enforcement_enabled=False silently.
    """
    with pytest.raises(ValueError, match="explicit choice about application authorization"):
        Settings(
            _env_file=None, env="production", neo4j_password="a-real-secret",
            workspace_api_keys={"ws-1": "k"}, panel_token_secret="a-real-panel-secret",
            metrics_api_key="a-real-metrics-key",
        )


def test_authz_enforcement_without_a_claims_source_raises_at_boot():
    """api.dependencies.get_access_context() 503s every request in this
    configuration; catching it at boot beats discovering it in traffic."""
    with pytest.raises(ValueError, match="requires SSO_ENABLED"):
        Settings(_env_file=None, authz_enforcement_enabled=True)


def test_authz_enforcement_with_a_trusted_gateway_is_accepted():
    settings = Settings(
        _env_file=None, authz_enforcement_enabled=True, authz_trusted_gateway_enabled=True
    )
    assert settings.authz_enforcement_enabled
