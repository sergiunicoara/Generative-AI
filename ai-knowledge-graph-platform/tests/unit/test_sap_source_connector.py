"""Contract tests for graphrag/ingestion/sap_source.py's SAPODataSourceConnector.

Mirrors tests/unit/test_http_source_connector.py's four concern-groups
(token caching, pagination, retry/throttling, credential/uri safety), plus
what's specific to this vendor adapter: token *renewal* on real expiry
(the generic connector caches forever) and schema-drift detection.
"""

from __future__ import annotations

import httpx
import pytest

from graphrag.ingestion.http_source import OAuth2ClientCredentialsConfig
from graphrag.ingestion.sap_source import (
    SAPODataConnectorError,
    SAPODataSourceConfig,
    SAPODataSourceConnector,
    SAPSchemaDriftError,
)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _config(**overrides) -> SAPODataSourceConfig:
    defaults: dict = dict(
        base_url="https://sap.example.test",
        sap_client="100",
        oauth=OAuth2ClientCredentialsConfig(
            token_url="https://sap.example.test/token",
            client_id="client-1",
            client_secret_env="TEST_SAP_CLIENT_SECRET",
        ),
        table_paths={"assets": "/AssetSet"},
        max_retries=3,
        backoff_seconds=0,
    )
    defaults.update(overrides)
    return SAPODataSourceConfig(**defaults)


@pytest.fixture(autouse=True)
def _client_secret(monkeypatch):
    monkeypatch.setenv("TEST_SAP_CLIENT_SECRET", "s3cr3t")


class TestOAuthTokenCachingAndRenewal:
    async def test_token_is_reused_across_calls(self):
        token_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal token_calls
            if request.url.path.endswith("/token"):
                token_calls += 1
                return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
            assert request.headers["Authorization"] == "Bearer tok"
            return httpx.Response(200, json={"d": {"results": []}})

        connector = SAPODataSourceConnector(_config(), client=_client(handler))
        await connector.read_table("assets")
        await connector.read_table("assets")
        assert token_calls == 1

    async def test_token_is_renewed_once_it_expires(self, monkeypatch):
        clock = {"now": 0.0}
        monkeypatch.setattr("graphrag.ingestion.sap_source.time.monotonic", lambda: clock["now"])
        token_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal token_calls
            if request.url.path.endswith("/token"):
                token_calls += 1
                return httpx.Response(200, json={"access_token": f"tok-{token_calls}", "expires_in": 100})
            return httpx.Response(200, json={"d": {"results": []}})

        connector = SAPODataSourceConnector(_config(), client=_client(handler))
        await connector.read_table("assets")
        assert token_calls == 1
        clock["now"] += 200  # past expiry + the 30s margin
        await connector.read_table("assets")
        assert token_calls == 2

    async def test_missing_secret_env_raises(self, monkeypatch):
        monkeypatch.delenv("TEST_SAP_CLIENT_SECRET", raising=False)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not make any HTTP call")

        connector = SAPODataSourceConnector(_config(), client=_client(handler))
        with pytest.raises(ValueError, match="TEST_SAP_CLIENT_SECRET"):
            await connector.read_table("assets")

    async def test_empty_token_in_response_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"access_token": ""})

        connector = SAPODataSourceConnector(_config(), client=_client(handler))
        with pytest.raises(ValueError, match="no access_token"):
            await connector.read_table("assets")


class TestPagination:
    async def test_follows_the_d_next_link_until_exhausted(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/token"):
                return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
            assert request.headers["sap-client"] == "100"
            if "page=2" in str(request.url):
                return httpx.Response(200, json={"d": {"results": [{"id": "2"}]}})
            return httpx.Response(200, json={
                "d": {"results": [{"id": "1"}], "__next": "https://sap.example.test/AssetSet?page=2"},
            })

        connector = SAPODataSourceConnector(_config(), client=_client(handler))
        rows = await connector.read_table("assets")
        assert rows == [{"id": "1"}, {"id": "2"}]

    async def test_unknown_table_raises_before_any_http_call(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not make any HTTP call")

        connector = SAPODataSourceConnector(_config(), client=_client(handler))
        with pytest.raises(ValueError, match="no OData entity-set path"):
            await connector.read_table("unknown_table")

    async def test_sql_injection_shaped_table_name_is_rejected_before_any_http_call(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not make any HTTP call")

        connector = SAPODataSourceConnector(_config(), client=_client(handler))
        with pytest.raises(ValueError, match="simple SQL identifier"):
            await connector.read_table("assets; DROP TABLE x--")


class TestThrottlingAndRetry:
    async def test_429_with_retry_after_eventually_succeeds(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            if request.url.path.endswith("/token"):
                return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json={"d": {"results": [{"id": "1"}]}})

        connector = SAPODataSourceConnector(_config(), client=_client(handler))
        rows = await connector.read_table("assets")
        assert rows == [{"id": "1"}]
        assert attempts == 2

    async def test_5xx_eventually_succeeds(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            if request.url.path.endswith("/token"):
                return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
            attempts += 1
            if attempts < 2:
                return httpx.Response(503)
            return httpx.Response(200, json={"d": {"results": []}})

        connector = SAPODataSourceConnector(_config(), client=_client(handler))
        await connector.read_table("assets")
        assert attempts == 2

    async def test_retries_are_bounded_then_a_clear_vendor_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/token"):
                return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
            return httpx.Response(503)

        connector = SAPODataSourceConnector(_config(max_retries=2), client=_client(handler))
        with pytest.raises(SAPODataConnectorError):
            await connector.read_table("assets")

    async def test_non_retryable_4xx_is_not_retried(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            if request.url.path.endswith("/token"):
                return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
            attempts += 1
            return httpx.Response(403)

        connector = SAPODataSourceConnector(_config(), client=_client(handler))
        with pytest.raises(httpx.HTTPStatusError):
            await connector.read_table("assets")
        assert attempts == 1


class TestSchemaChange:
    async def test_missing_required_column_fails_closed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/token"):
                return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
            return httpx.Response(200, json={"d": {"results": [{"id": "1"}]}})

        config = _config(expected_columns={"assets": frozenset({"id", "temperature_c"})})
        connector = SAPODataSourceConnector(config, client=_client(handler))
        with pytest.raises(SAPSchemaDriftError, match="temperature_c"):
            await connector.read_table("assets")

    async def test_unexpected_extra_column_is_tolerated(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/token"):
                return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
            return httpx.Response(200, json={"d": {"results": [{"id": "1", "new_field": "x"}]}})

        config = _config(expected_columns={"assets": frozenset({"id"})})
        connector = SAPODataSourceConnector(config, client=_client(handler))
        rows = await connector.read_table("assets")
        assert rows == [{"id": "1", "new_field": "x"}]


class TestUriNeverLeaksCredentials:
    def test_uri_is_just_the_base_url(self):
        connector = SAPODataSourceConnector(_config(), client=None)
        assert connector.uri == "https://sap.example.test"
        assert "s3cr3t" not in connector.uri
