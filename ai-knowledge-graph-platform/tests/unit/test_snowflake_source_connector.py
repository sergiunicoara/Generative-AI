"""Contract tests for graphrag/ingestion/snowflake_source.py's
SnowflakeSQLAPISourceConnector.

Same four concern-groups as tests/unit/test_sap_source_connector.py
(token caching/renewal, pagination, throttling, schema changes), adapted
to Snowflake's real shape: statement submission + async poll + partitioned
results, instead of a flat cursor.
"""

from __future__ import annotations

import httpx
import pytest

from graphrag.ingestion.http_source import OAuth2ClientCredentialsConfig
from graphrag.ingestion.snowflake_source import (
    SnowflakeConnectorError,
    SnowflakeSchemaDriftError,
    SnowflakeSourceConfig,
    SnowflakeSQLAPISourceConnector,
)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _config(**overrides) -> SnowflakeSourceConfig:
    defaults: dict = dict(
        base_url="https://acme.snowflakecomputing.com",
        oauth=OAuth2ClientCredentialsConfig(
            token_url="https://acme.snowflakecomputing.com/oauth/token",
            client_id="client-1",
            client_secret_env="TEST_SNOWFLAKE_CLIENT_SECRET",
        ),
        warehouse="COMPUTE_WH",
        database="ENERGY_DB",
        schema="PUBLIC",
        max_retries=3,
        backoff_seconds=0,
        max_poll_attempts=5,
        poll_interval_seconds=0,
    )
    defaults.update(overrides)
    return SnowflakeSourceConfig(**defaults)


def _is_token_request(request: httpx.Request) -> bool:
    return request.url.path.endswith("/oauth/token")


def _token_response() -> httpx.Response:
    return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})


@pytest.fixture(autouse=True)
def _client_secret(monkeypatch):
    monkeypatch.setenv("TEST_SNOWFLAKE_CLIENT_SECRET", "s3cr3t")


class TestOAuthTokenCachingAndRenewal:
    async def test_token_is_reused_across_calls(self):
        token_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal token_calls
            if _is_token_request(request):
                token_calls += 1
                return _token_response()
            assert request.headers["X-Snowflake-Authorization-Token-Type"] == "OAUTH"
            return httpx.Response(200, json={
                "resultSetMetaData": {"rowType": [{"name": "ID"}], "partitionInfo": [{"rowCount": 0}]},
                "data": [], "statementHandle": "h1",
            })

        connector = SnowflakeSQLAPISourceConnector(_config(), client=_client(handler))
        await connector.read_table("assets")
        await connector.read_table("assets")
        assert token_calls == 1

    async def test_token_is_renewed_once_it_expires(self, monkeypatch):
        clock = {"now": 0.0}
        monkeypatch.setattr("graphrag.ingestion.snowflake_source.time.monotonic", lambda: clock["now"])
        token_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal token_calls
            if _is_token_request(request):
                token_calls += 1
                return httpx.Response(200, json={"access_token": f"tok-{token_calls}", "expires_in": 100})
            return httpx.Response(200, json={
                "resultSetMetaData": {"rowType": [], "partitionInfo": [{"rowCount": 0}]},
                "data": [], "statementHandle": "h1",
            })

        connector = SnowflakeSQLAPISourceConnector(_config(), client=_client(handler))
        await connector.read_table("assets")
        assert token_calls == 1
        clock["now"] += 200
        await connector.read_table("assets")
        assert token_calls == 2

    async def test_missing_secret_env_raises(self, monkeypatch):
        monkeypatch.delenv("TEST_SNOWFLAKE_CLIENT_SECRET", raising=False)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not make any HTTP call")

        connector = SnowflakeSQLAPISourceConnector(_config(), client=_client(handler))
        with pytest.raises(ValueError, match="TEST_SNOWFLAKE_CLIENT_SECRET"):
            await connector.read_table("assets")


class TestStatementExecutionAndPagination:
    async def test_synchronous_result_is_returned_without_polling(self):
        poll_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal poll_calls
            if _is_token_request(request):
                return _token_response()
            if request.method == "POST":
                return httpx.Response(200, json={
                    "resultSetMetaData": {
                        "rowType": [{"name": "ID"}, {"name": "NAME"}],
                        "partitionInfo": [{"rowCount": 1}],
                    },
                    "data": [["1", "WT-01"]],
                    "statementHandle": "h1",
                })
            poll_calls += 1
            raise AssertionError("must not poll for a synchronous result")

        connector = SnowflakeSQLAPISourceConnector(_config(), client=_client(handler))
        rows = await connector.read_table("assets")
        assert rows == [{"ID": "1", "NAME": "WT-01"}]
        assert poll_calls == 0

    async def test_async_execution_polls_until_complete(self):
        poll_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal poll_calls
            if _is_token_request(request):
                return _token_response()
            if request.method == "POST":
                return httpx.Response(200, json={"code": "090001", "statementHandle": "h2"})
            poll_calls += 1
            if poll_calls < 2:
                return httpx.Response(200, json={"code": "090001", "statementHandle": "h2"})
            return httpx.Response(200, json={
                "code": "00000",
                "resultSetMetaData": {"rowType": [{"name": "ID"}], "partitionInfo": [{"rowCount": 1}]},
                "data": [["1"]],
            })

        connector = SnowflakeSQLAPISourceConnector(_config(), client=_client(handler))
        rows = await connector.read_table("assets")
        assert rows == [{"ID": "1"}]
        assert poll_calls == 2

    async def test_polling_forever_without_completion_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if _is_token_request(request):
                return _token_response()
            return httpx.Response(200, json={"code": "090001", "statementHandle": "h3"})

        connector = SnowflakeSQLAPISourceConnector(_config(max_poll_attempts=3), client=_client(handler))
        with pytest.raises(SnowflakeConnectorError, match="did not complete"):
            await connector.read_table("assets")

    async def test_fetches_every_partition(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if _is_token_request(request):
                return _token_response()
            if request.method == "POST":
                return httpx.Response(200, json={
                    "resultSetMetaData": {
                        "rowType": [{"name": "ID"}],
                        "partitionInfo": [{"rowCount": 1}, {"rowCount": 1}, {"rowCount": 1}],
                    },
                    "data": [["1"]],
                    "statementHandle": "h4",
                })
            partition = request.url.params.get("partition")
            return httpx.Response(200, json={"data": [[f"{partition}-row"]]})

        connector = SnowflakeSQLAPISourceConnector(_config(), client=_client(handler))
        rows = await connector.read_table("assets")
        assert rows == [{"ID": "1"}, {"ID": "1-row"}, {"ID": "2-row"}]

    async def test_sql_injection_shaped_table_name_is_rejected_before_any_http_call(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not make any HTTP call")

        connector = SnowflakeSQLAPISourceConnector(_config(), client=_client(handler))
        with pytest.raises(ValueError, match="simple SQL identifier"):
            await connector.read_table("assets; DROP TABLE x--")


class TestThrottlingAndRetry:
    async def test_429_with_retry_after_eventually_succeeds(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            if _is_token_request(request):
                return _token_response()
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json={
                "resultSetMetaData": {"rowType": [], "partitionInfo": [{"rowCount": 0}]},
                "data": [], "statementHandle": "h5",
            })

        connector = SnowflakeSQLAPISourceConnector(_config(), client=_client(handler))
        await connector.read_table("assets")
        assert attempts == 2

    async def test_retries_are_bounded_then_a_clear_vendor_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if _is_token_request(request):
                return _token_response()
            return httpx.Response(503)

        connector = SnowflakeSQLAPISourceConnector(_config(max_retries=2), client=_client(handler))
        with pytest.raises(SnowflakeConnectorError):
            await connector.read_table("assets")

    async def test_non_retryable_4xx_is_not_retried(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            if _is_token_request(request):
                return _token_response()
            attempts += 1
            return httpx.Response(403)

        connector = SnowflakeSQLAPISourceConnector(_config(), client=_client(handler))
        with pytest.raises(httpx.HTTPStatusError):
            await connector.read_table("assets")
        assert attempts == 1


class TestSchemaChange:
    async def test_missing_required_column_fails_closed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if _is_token_request(request):
                return _token_response()
            return httpx.Response(200, json={
                "resultSetMetaData": {"rowType": [{"name": "ID"}], "partitionInfo": [{"rowCount": 1}]},
                "data": [["1"]], "statementHandle": "h6",
            })

        config = _config(expected_columns={"assets": frozenset({"ID", "TEMPERATURE_C"})})
        connector = SnowflakeSQLAPISourceConnector(config, client=_client(handler))
        with pytest.raises(SnowflakeSchemaDriftError, match="TEMPERATURE_C"):
            await connector.read_table("assets")

    async def test_column_matching_is_case_insensitive(self):
        """Snowflake unquoted identifiers are uppercased by default --
        expected_columns given lowercase must still match."""
        def handler(request: httpx.Request) -> httpx.Response:
            if _is_token_request(request):
                return _token_response()
            return httpx.Response(200, json={
                "resultSetMetaData": {"rowType": [{"name": "ID"}], "partitionInfo": [{"rowCount": 1}]},
                "data": [["1"]], "statementHandle": "h7",
            })

        config = _config(expected_columns={"assets": frozenset({"id"})})
        connector = SnowflakeSQLAPISourceConnector(config, client=_client(handler))
        rows = await connector.read_table("assets")
        assert rows == [{"ID": "1"}]

    async def test_unexpected_extra_column_is_tolerated(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if _is_token_request(request):
                return _token_response()
            return httpx.Response(200, json={
                "resultSetMetaData": {
                    "rowType": [{"name": "ID"}, {"name": "NEW_FIELD"}],
                    "partitionInfo": [{"rowCount": 1}],
                },
                "data": [["1", "x"]], "statementHandle": "h8",
            })

        config = _config(expected_columns={"assets": frozenset({"ID"})})
        connector = SnowflakeSQLAPISourceConnector(config, client=_client(handler))
        rows = await connector.read_table("assets")
        assert rows == [{"ID": "1", "NEW_FIELD": "x"}]


class TestUriNeverLeaksCredentials:
    def test_uri_is_the_database_and_schema_path(self):
        connector = SnowflakeSQLAPISourceConnector(_config(), client=None)
        assert connector.uri == "https://acme.snowflakecomputing.com/ENERGY_DB/PUBLIC"
        assert "s3cr3t" not in connector.uri
