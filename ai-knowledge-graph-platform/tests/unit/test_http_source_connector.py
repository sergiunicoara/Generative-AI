"""RESTAPISourceConnector: OAuth2 token caching, cursor pagination, and
bounded retry-on-transient-failure -- all against httpx.MockTransport, no
live network or Docker.
"""
from __future__ import annotations

import httpx
import pytest

from graphrag.ingestion.http_source import (
    OAuth2ClientCredentialsConfig,
    RESTAPISourceConfig,
    RESTAPISourceConnector,
    RESTSourceConnectorError,
)

TOKEN_URL = "https://auth.example.com/oauth/token"
BASE_URL = "https://api.example.com"


def _config(**overrides) -> RESTAPISourceConfig:
    defaults = dict(
        base_url=BASE_URL,
        oauth=OAuth2ClientCredentialsConfig(
            token_url=TOKEN_URL, client_id="cid", client_secret_env="TEST_CLIENT_SECRET",
        ),
        table_paths={"assets": "/assets"},
        page_size=2,
        max_retries=3,
        backoff_seconds=0,  # no real delay in tests
    )
    defaults.update(overrides)
    return RESTAPISourceConfig(**defaults)


@pytest.fixture(autouse=True)
def _client_secret(monkeypatch):
    monkeypatch.setenv("TEST_CLIENT_SECRET", "shh")


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestOAuthTokenCaching:
    @pytest.mark.asyncio
    async def test_second_call_reuses_the_cached_token_no_second_token_request(self):
        token_requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/token"):
                token_requests.append(request)
                return httpx.Response(200, json={"access_token": "tok-1"})
            assert request.headers["Authorization"] == "Bearer tok-1"
            return httpx.Response(200, json={"results": [{"id": "a1"}], "next": None})

        connector = RESTAPISourceConnector(_config(), client=_client(handler))
        await connector.read_table("assets")
        await connector.read_table("assets")

        assert len(token_requests) == 1

    @pytest.mark.asyncio
    async def test_missing_client_secret_env_raises(self, monkeypatch):
        monkeypatch.delenv("TEST_CLIENT_SECRET", raising=False)

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not make an HTTP call with no secret")

        connector = RESTAPISourceConnector(_config(), client=_client(handler))
        with pytest.raises(ValueError, match="TEST_CLIENT_SECRET"):
            await connector.read_table("assets")

    @pytest.mark.asyncio
    async def test_empty_access_token_response_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        connector = RESTAPISourceConnector(_config(), client=_client(handler))
        with pytest.raises(ValueError, match="no access_token"):
            await connector.read_table("assets")


class TestPagination:
    @pytest.mark.asyncio
    async def test_follows_next_cursor_until_exhausted(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/token"):
                return httpx.Response(200, json={"access_token": "tok"})
            if "page2" in str(request.url):
                return httpx.Response(200, json={"results": [{"id": "a3"}], "next": None})
            return httpx.Response(
                200,
                json={"results": [{"id": "a1"}, {"id": "a2"}], "next": f"{BASE_URL}/assets?page2"},
            )

        connector = RESTAPISourceConnector(_config(), client=_client(handler))
        rows = await connector.read_table("assets")

        assert [r["id"] for r in rows] == ["a1", "a2", "a3"]

    @pytest.mark.asyncio
    async def test_unknown_table_raises_before_any_request(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not make an HTTP call for an unconfigured table")

        connector = RESTAPISourceConnector(_config(), client=_client(handler))
        with pytest.raises(ValueError, match="no endpoint configured"):
            await connector.read_table("unknown_table")

    @pytest.mark.asyncio
    async def test_malicious_table_name_is_rejected_before_any_request(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not make an HTTP call for an invalid identifier")

        connector = RESTAPISourceConnector(_config(), client=_client(handler))
        with pytest.raises(ValueError, match="simple SQL identifier"):
            await connector.read_table("assets; DROP TABLE x")


class TestRetry:
    @pytest.mark.asyncio
    async def test_transient_429_then_success_returns_data(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/token"):
                return httpx.Response(200, json={"access_token": "tok"})
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429)
            return httpx.Response(200, json={"results": [{"id": "a1"}], "next": None})

        connector = RESTAPISourceConnector(_config(), client=_client(handler))
        rows = await connector.read_table("assets")

        assert rows == [{"id": "a1"}]
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_transient_5xx_then_success_returns_data(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/token"):
                return httpx.Response(200, json={"access_token": "tok"})
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"results": [{"id": "a1"}], "next": None})

        connector = RESTAPISourceConnector(_config(), client=_client(handler))
        rows = await connector.read_table("assets")

        assert rows == [{"id": "a1"}]

    @pytest.mark.asyncio
    async def test_retries_are_bounded_then_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/token"):
                return httpx.Response(200, json={"access_token": "tok"})
            return httpx.Response(503)

        connector = RESTAPISourceConnector(_config(max_retries=2), client=_client(handler))
        with pytest.raises(RESTSourceConnectorError, match="failed after 2 attempts"):
            await connector.read_table("assets")

    @pytest.mark.asyncio
    async def test_non_retryable_4xx_raises_immediately_without_exhausting_retries(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/token"):
                return httpx.Response(200, json={"access_token": "tok"})
            calls["n"] += 1
            return httpx.Response(404)

        connector = RESTAPISourceConnector(_config(), client=_client(handler))
        with pytest.raises(httpx.HTTPStatusError):
            await connector.read_table("assets")
        assert calls["n"] == 1  # not retried


class TestUriNeverLeaksCredentials:
    def test_uri_is_just_the_base_url(self):
        connector = RESTAPISourceConnector(_config())
        assert connector.uri == BASE_URL
        assert "shh" not in connector.uri
