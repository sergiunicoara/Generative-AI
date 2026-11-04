"""Databricks SQL Statement Execution connector (offline, httpx.MockTransport)."""

from __future__ import annotations

import httpx
import pytest

from graphrag.ingestion.databricks_source import (
    DatabricksConnectorError,
    DatabricksSchemaDriftError,
    DatabricksSourceConfig,
    DatabricksSQLSourceConnector,
)
from graphrag.ingestion.http_source import OAuth2ClientCredentialsConfig
from graphrag.ingestion.relational import TabularSourceConnector

HOST = "https://acme.cloud.databricks.com"
COLUMNS = {"schema": {"columns": [{"name": "asset_id"}, {"name": "name"}]}}


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setenv("TEST_DATABRICKS_SECRET", "s3cr3t")


def _config(**overrides) -> DatabricksSourceConfig:
    base = dict(host=HOST, oauth=OAuth2ClientCredentialsConfig(token_url=f"{HOST}/oidc/v1/token", client_id="sp-1",
                                                              client_secret_env="TEST_DATABRICKS_SECRET"),
                warehouse_id="wh-1", catalog="main", schema="energy", backoff_seconds=0,
                poll_interval_seconds=0, max_poll_attempts=3)
    base.update(overrides)
    return DatabricksSourceConfig(**base)


def _connector(handler, **overrides) -> DatabricksSQLSourceConnector:
    return DatabricksSQLSourceConnector(_config(**overrides), httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _token(request):
    return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})


async def test_inline_result_with_chunks_is_read_in_order():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/oidc/v1/token":
            return _token(request)
        if request.method == "POST":
            return httpx.Response(200, json={
                "statement_id": "s1", "status": {"state": "SUCCEEDED"}, "manifest": COLUMNS,
                "result": {"data_array": [["A1", "Pump"]],
                           "next_chunk_internal_link": "/api/2.0/sql/statements/s1/result/chunks/1"}})
        return httpx.Response(200, json={"data_array": [["A2", "Valve"]]})

    rows = await _connector(handler).read_table("assets")
    assert rows == [{"asset_id": "A1", "name": "Pump"}, {"asset_id": "A2", "name": "Valve"}]
    body = seen[1].read().decode()
    assert '"warehouse_id":"wh-1"' in body.replace(" ", "") and "SELECT * FROM assets" in body
    assert seen[1].headers["authorization"] == "Bearer tok"
    assert seen[0].headers["authorization"].startswith("Basic "), "client credentials go in basic auth"


async def test_a_running_statement_is_polled_until_it_succeeds():
    states = iter(["RUNNING", "SUCCEEDED"])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oidc/v1/token":
            return _token(request)
        if request.method == "POST":
            return httpx.Response(200, json={"statement_id": "s2", "status": {"state": "PENDING"}})
        state = next(states)
        payload = {"statement_id": "s2", "status": {"state": state}}
        if state == "SUCCEEDED":
            payload |= {"manifest": COLUMNS, "result": {"data_array": [["A1", "Pump"]]}}
        return httpx.Response(200, json=payload)

    assert await _connector(handler).read_table("assets") == [{"asset_id": "A1", "name": "Pump"}]


@pytest.mark.parametrize("state", ["FAILED", "CANCELED", "CLOSED"])
async def test_terminal_failure_states_raise(state):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oidc/v1/token":
            return _token(request)
        return httpx.Response(200, json={"statement_id": "s3", "status": {
            "state": state, "error": {"error_code": "BAD", "message": "no table"}}})

    with pytest.raises(DatabricksConnectorError, match=state):
        await _connector(handler).read_table("assets")


async def test_polling_is_bounded():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oidc/v1/token":
            return _token(request)
        return httpx.Response(200, json={"statement_id": "s4", "status": {"state": "RUNNING"}})

    with pytest.raises(DatabricksConnectorError, match="poll"):
        await _connector(handler).read_table("assets")


async def test_missing_required_column_fails_closed_and_foreign_chunk_links_are_refused():
    def drift(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oidc/v1/token":
            return _token(request)
        return httpx.Response(200, json={"status": {"state": "SUCCEEDED"}, "manifest": COLUMNS, "result": {}})

    with pytest.raises(DatabricksSchemaDriftError):
        await _connector(drift, expected_columns={"assets": frozenset({"asset_id", "site"})}).read_table("assets")

    def foreign(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oidc/v1/token":
            return _token(request)
        return httpx.Response(200, json={"status": {"state": "SUCCEEDED"}, "manifest": COLUMNS,
                                         "result": {"next_chunk_internal_link": "https://evil.example/x"}})

    with pytest.raises(DatabricksConnectorError, match="next_chunk"):
        await _connector(foreign).read_table("assets")


async def test_identifiers_urls_and_secrets_are_checked():
    with pytest.raises(ValueError):
        await _connector(lambda r: httpx.Response(500)).read_table("assets; DROP TABLE x")
    with pytest.raises(ValueError):
        _config(host="file:///etc")
    with pytest.raises(ValueError):
        _config(catalog="main.evil")
    connector = _connector(lambda r: httpx.Response(500))
    assert isinstance(connector, TabularSourceConnector)
    assert "s3cr3t" not in connector.uri and connector.uri == f"{HOST}/main/energy"
