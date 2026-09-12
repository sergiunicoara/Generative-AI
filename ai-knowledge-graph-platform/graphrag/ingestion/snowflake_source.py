"""Snowflake SQL API v2 source connector.

Vendor-specific, unlike `graphrag/ingestion/http_source.py`'s generic
`RESTAPISourceConnector`: Snowflake's real SQL API v2 shape is a
statement-submission + async-poll + partitioned-result protocol, not a flat
GET-with-cursor -- materially different enough that it cannot reuse the
generic connector's pagination logic --

1. `POST /api/v2/statements` submits `SELECT * FROM <table>`. A small
   result set returns synchronously (200, with `resultSetMetaData`
   already present); a larger one returns `202`/an in-progress `code`, and
   the caller must poll `GET /api/v2/statements/{handle}` until it settles.
2. A large result set is split into partitions
   (`resultSetMetaData.partitionInfo`); partition 0 is embedded in the
   completed response, every other partition needs its own
   `GET .../statements/{handle}?partition=N`.
3. Rows arrive as `list[list[str]]`, not JSON objects -- column names/order
   come from `resultSetMetaData.rowType`, and that same schema is what
   schema-drift detection checks against `expected_columns`.
4. Real OAuth2 token expiry is tracked and the token is proactively
   re-fetched, the same gap-closing this connector shares with
   `graphrag/ingestion/sap_source.py`.

Verification status
--------------------
Contract-tested against `httpx.MockTransport` only. No live Snowflake
account or credentials exist in this environment -- treat this as a
documented, tested protocol shape, not a certified vendor integration.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from graphrag.core.connector_url_safety import assert_safe_connector_url
from graphrag.graph.source_catalog import SourceKind
from graphrag.ingestion.connector_retry import RetryExhaustedError, send_with_retry
from graphrag.ingestion.http_source import OAuth2ClientCredentialsConfig
from graphrag.ingestion.relational import _identifier

log = structlog.get_logger(__name__)

_TOKEN_EXPIRY_MARGIN_SECONDS = 30.0
_DEFAULT_TOKEN_TTL_SECONDS = 3600.0
# Snowflake's own code for "statement still executing" on a GET poll.
_STILL_EXECUTING_CODE = "090001"


class SnowflakeConnectorError(RuntimeError):
    """Raised after retries are exhausted, on a non-retryable HTTP error, or
    when statement execution never reaches a terminal state within the
    configured poll budget."""


class SnowflakeSchemaDriftError(RuntimeError):
    """Raised when a table's result-set schema is missing a required
    column -- fail closed rather than silently ingesting an incomplete row
    shape. An unlisted *extra* column is tolerated and only logged."""


@dataclass(frozen=True)
class SnowflakeSourceConfig:
    base_url: str  # e.g. https://<account>.snowflakecomputing.com
    oauth: OAuth2ClientCredentialsConfig
    warehouse: str
    database: str
    schema: str
    # Table name -> required column names, matched case-insensitively
    # against Snowflake's default-uppercased unquoted identifiers.
    expected_columns: dict[str, frozenset[str]] = field(default_factory=dict)
    max_retries: int = 3
    backoff_seconds: float = 0.5
    timeout_seconds: float = 30.0
    # Bounded poll loop for async statement execution (202/still-executing).
    max_poll_attempts: int = 10
    poll_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        assert_safe_connector_url(self.base_url, context="SnowflakeSourceConfig base_url")


class SnowflakeSQLAPISourceConnector:
    """`TabularSourceConnector`-conformant Snowflake SQL API v2 connector.

    `client` is injectable so this needs no live network or Docker to
    exercise -- same posture as `RESTAPISourceConnector`/`SAPODataSourceConnector`.
    """

    kind = SourceKind.DATABASE  # a warehouse, conceptually, even though the
    # transport is HTTP -- PostgreSQLSourceConnector sets the precedent for
    # what DATABASE means in this repo's SourceKind vocabulary.

    def __init__(self, config: SnowflakeSourceConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._client = client
        self._access_token = ""
        self._token_expires_at = 0.0

    @property
    def uri(self) -> str:
        # Never the token or secret.
        return f"{self.config.base_url}/{self.config.database}/{self.config.schema}"

    async def _ensure_token(self) -> str:
        if self._access_token and time.monotonic() < self._token_expires_at - _TOKEN_EXPIRY_MARGIN_SECONDS:
            return self._access_token
        secret = os.getenv(self.config.oauth.client_secret_env, "")
        if not secret:
            raise ValueError(
                f"environment variable {self.config.oauth.client_secret_env!r} is not set"
            )
        data = {
            "client_id": self.config.oauth.client_id,
            "client_secret": secret,
            "grant_type": "client_credentials",
        }
        if self.config.oauth.scope:
            data["scope"] = self.config.oauth.scope
        response = await self._send("POST", self.config.oauth.token_url, data=data)
        payload = response.json()
        token = str(payload.get("access_token", ""))
        if not token:
            raise ValueError("Snowflake OAuth token endpoint returned no access_token")
        self._access_token = token
        try:
            self._token_expires_at = time.monotonic() + float(payload.get("expires_in"))
        except (TypeError, ValueError):
            self._token_expires_at = time.monotonic() + _DEFAULT_TOKEN_TTL_SECONDS
        return token

    async def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        if self._client is not None:
            return await self._client.request(method, url, **kwargs)
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            return await client.request(method, url, **kwargs)

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        token = await self._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            # Real Snowflake SQL API requirement for an OAuth bearer token,
            # distinguishing it from Snowflake's own key-pair JWT auth mode.
            "X-Snowflake-Authorization-Token-Type": "OAUTH",
            **kwargs.pop("headers", {}),
        }
        try:
            return await send_with_retry(
                lambda: self._send(method, url, headers=headers, **kwargs),
                max_retries=self.config.max_retries,
                base_backoff_seconds=self.config.backoff_seconds,
            )
        except RetryExhaustedError as exc:
            raise SnowflakeConnectorError(f"{method} {url} failed: {exc}") from exc

    async def _submit_statement(self, sql: str) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/api/v2/statements"
        body = {
            "statement": sql,
            "warehouse": self.config.warehouse,
            "database": self.config.database,
            "schema": self.config.schema,
        }
        response = await self._request("POST", url, json=body)
        return response.json()

    async def _poll_until_complete(self, statement_handle: str) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/api/v2/statements/{statement_handle}"
        for _attempt in range(self.config.max_poll_attempts):
            response = await self._request("GET", url)
            payload = response.json()
            if payload.get("resultSetMetaData") is not None and str(payload.get("code", "")) != _STILL_EXECUTING_CODE:
                return payload
            await asyncio.sleep(self.config.poll_interval_seconds)
        raise SnowflakeConnectorError(
            f"statement {statement_handle!r} did not complete within "
            f"{self.config.max_poll_attempts} poll attempt(s)"
        )

    async def _fetch_partition(self, statement_handle: str, partition: int) -> list[list[Any]]:
        url = f"{self.config.base_url.rstrip('/')}/api/v2/statements/{statement_handle}"
        response = await self._request("GET", url, params={"partition": partition})
        return list(response.json().get("data", []))

    def _check_schema(self, table: str, columns: list[str]) -> None:
        required = self.config.expected_columns.get(table)
        if not required:
            return
        required_upper = {c.upper() for c in required}
        present_upper = {c.upper() for c in columns}
        missing = required_upper - present_upper
        if missing:
            raise SnowflakeSchemaDriftError(
                f"Snowflake table {table!r} is missing required column(s): {sorted(missing)}"
            )
        extra = present_upper - required_upper
        if extra:
            log.info("snowflake_source.unexpected_columns", table=table, extra=sorted(extra))

    async def read_table(self, table: str) -> list[dict[str, Any]]:
        _identifier(table, "table")
        # `table` is already validated as a bare SQL identifier above --
        # same "validate then interpolate; identifiers can't be
        # parameterized" pattern relational.py's own connectors use.
        sql = f"SELECT * FROM {table}"

        result = await self._submit_statement(sql)
        if result.get("resultSetMetaData") is None:
            handle = str(result.get("statementHandle", ""))
            if not handle:
                raise SnowflakeConnectorError("Snowflake did not return a statementHandle")
            result = await self._poll_until_complete(handle)
        else:
            handle = str(result.get("statementHandle", ""))

        metadata = result.get("resultSetMetaData", {})
        columns = [str(c.get("name", "")) for c in metadata.get("rowType", [])]
        self._check_schema(table, columns)

        partitions = metadata.get("partitionInfo") or [{}]
        raw_rows: list[list[Any]] = list(result.get("data", []))
        for partition_index in range(1, len(partitions)):
            raw_rows.extend(await self._fetch_partition(handle, partition_index))

        return [dict(zip(columns, raw_row, strict=False)) for raw_row in raw_rows]


__all__ = [
    "SnowflakeConnectorError",
    "SnowflakeSchemaDriftError",
    "SnowflakeSourceConfig",
    "SnowflakeSQLAPISourceConnector",
]
