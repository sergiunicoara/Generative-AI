"""Databricks SQL Statement Execution API 2.0 source connector.

Same shape and posture as `graphrag/ingestion/snowflake_source.py`, for the
Databricks protocol:

1. `POST /api/2.0/sql/statements` submits `SELECT * FROM <table>` against a SQL
   warehouse, with the catalog and schema in the request body. With
   `wait_timeout` the call may return the finished result directly; otherwise
   `status.state` is `PENDING` or `RUNNING` and the caller polls
   `GET /api/2.0/sql/statements/{statement_id}`.
2. `SUCCEEDED` carries `manifest.schema.columns` and the first result chunk
   (`result.data_array`, `format=JSON_ARRAY`, `disposition=INLINE`); every
   further chunk is fetched through `next_chunk_internal_link`.
3. `FAILED`, `CANCELED` and `CLOSED` are terminal errors.
4. Authentication is OAuth machine-to-machine (`/oidc/v1/token`,
   client credentials, scope `all-apis`), with proactive token refresh.

Verification status
--------------------
Contract-tested against `httpx.MockTransport` only. No Databricks workspace or
credentials exist in this environment; this is a documented, tested protocol
shape, not a certified vendor integration.
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
_RUNNING_STATES = frozenset({"PENDING", "RUNNING"})
_FAILED_STATES = frozenset({"FAILED", "CANCELED", "CLOSED"})


class DatabricksConnectorError(RuntimeError):
    """Retries exhausted, a terminal statement state, or no terminal state
    within the poll budget."""


class DatabricksSchemaDriftError(RuntimeError):
    """A table's result schema is missing a required column (fail closed)."""


@dataclass(frozen=True)
class DatabricksSourceConfig:
    host: str  # e.g. https://<workspace>.cloud.databricks.com
    oauth: OAuth2ClientCredentialsConfig
    warehouse_id: str
    catalog: str
    schema: str
    # Table name -> required column names, matched case-insensitively.
    expected_columns: dict[str, frozenset[str]] = field(default_factory=dict)
    wait_timeout: str = "10s"
    max_retries: int = 3
    backoff_seconds: float = 0.5
    timeout_seconds: float = 30.0
    max_poll_attempts: int = 10
    poll_interval_seconds: float = 1.0
    max_chunks: int = 10_000

    def __post_init__(self) -> None:
        assert_safe_connector_url(self.host, context="DatabricksSourceConfig host")
        _identifier(self.catalog, "catalog")
        _identifier(self.schema, "schema")


class DatabricksSQLSourceConnector:
    """`TabularSourceConnector`-conformant Databricks SQL connector.

    `client` is injectable so it runs without network access in tests.
    """

    kind = SourceKind.DATABASE

    def __init__(self, config: DatabricksSourceConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._client = client
        self._access_token = ""
        self._token_expires_at = 0.0

    @property
    def uri(self) -> str:
        # Never the token or secret.
        return f"{self.config.host.rstrip('/')}/{self.config.catalog}/{self.config.schema}"

    def _url(self, path: str) -> str:
        return f"{self.config.host.rstrip('/')}{path}"

    async def _ensure_token(self) -> str:
        if self._access_token and time.monotonic() < self._token_expires_at - _TOKEN_EXPIRY_MARGIN_SECONDS:
            return self._access_token
        secret = os.getenv(self.config.oauth.client_secret_env, "")
        if not secret:
            raise ValueError(f"environment variable {self.config.oauth.client_secret_env!r} is not set")
        data = {"grant_type": "client_credentials", "scope": self.config.oauth.scope or "all-apis"}
        response = await self._send("POST", self.config.oauth.token_url, data=data,
                                    auth=(self.config.oauth.client_id, secret))
        if response.status_code >= 400:
            raise DatabricksConnectorError(f"token request failed with HTTP {response.status_code}")
        payload = response.json()
        token = str(payload.get("access_token", ""))
        if not token:
            raise ValueError("Databricks OAuth token endpoint returned no access_token")
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

    async def _request(self, method: str, url: str, **kwargs) -> dict[str, Any]:
        token = await self._ensure_token()
        headers = {"Authorization": f"Bearer {token}", **kwargs.pop("headers", {})}
        try:
            response = await send_with_retry(
                lambda: self._send(method, url, headers=headers, **kwargs),
                max_retries=self.config.max_retries,
                base_backoff_seconds=self.config.backoff_seconds,
            )
        except RetryExhaustedError as exc:
            raise DatabricksConnectorError(f"{method} {url} failed: {exc}") from exc
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _state(payload: dict[str, Any]) -> str:
        return str((payload.get("status") or {}).get("state", "")).upper()

    def _raise_if_failed(self, payload: dict[str, Any]) -> None:
        state = self._state(payload)
        if state in _FAILED_STATES:
            error = (payload.get("status") or {}).get("error") or {}
            raise DatabricksConnectorError(
                f"statement {payload.get('statement_id', '')!r} ended {state}: "
                f"{error.get('error_code', '')} {error.get('message', '')}".strip()
            )

    async def _poll_until_complete(self, statement_id: str) -> dict[str, Any]:
        url = self._url(f"/api/2.0/sql/statements/{statement_id}")
        for _attempt in range(self.config.max_poll_attempts):
            payload = await self._request("GET", url)
            self._raise_if_failed(payload)
            if self._state(payload) not in _RUNNING_STATES:
                return payload
            await asyncio.sleep(self.config.poll_interval_seconds)
        raise DatabricksConnectorError(
            f"statement {statement_id!r} did not complete within {self.config.max_poll_attempts} poll attempt(s)"
        )

    def _check_schema(self, table: str, columns: list[str]) -> None:
        required = self.config.expected_columns.get(table)
        if not required:
            return
        present = {c.lower() for c in columns}
        missing = {c.lower() for c in required} - present
        if missing:
            raise DatabricksSchemaDriftError(
                f"Databricks table {table!r} is missing required column(s): {sorted(missing)}"
            )
        extra = present - {c.lower() for c in required}
        if extra:
            log.info("databricks_source.unexpected_columns", table=table, extra=sorted(extra))

    async def read_table(self, table: str) -> list[dict[str, Any]]:
        _identifier(table, "table")
        body = {
            "statement": f"SELECT * FROM {table}",  # identifier validated above
            "warehouse_id": self.config.warehouse_id,
            "catalog": self.config.catalog,
            "schema": self.config.schema,
            "wait_timeout": self.config.wait_timeout,
            "on_wait_timeout": "CONTINUE",
            "disposition": "INLINE",
            "format": "JSON_ARRAY",
        }
        payload = await self._request("POST", self._url("/api/2.0/sql/statements"), json=body)
        self._raise_if_failed(payload)
        if self._state(payload) in _RUNNING_STATES:
            statement_id = str(payload.get("statement_id", ""))
            if not statement_id:
                raise DatabricksConnectorError("Databricks did not return a statement_id")
            payload = await self._poll_until_complete(statement_id)
        if self._state(payload) != "SUCCEEDED":
            raise DatabricksConnectorError(f"unexpected statement state {self._state(payload)!r}")

        columns = [str(c.get("name", "")) for c in ((payload.get("manifest") or {}).get("schema") or {}).get("columns", [])]
        self._check_schema(table, columns)

        chunk = payload.get("result") or {}
        rows: list[list[Any]] = list(chunk.get("data_array") or [])
        for _ in range(self.config.max_chunks):
            link = chunk.get("next_chunk_internal_link")
            if not link:
                break
            if not str(link).startswith("/api/2.0/sql/statements/"):
                raise DatabricksConnectorError("unexpected next_chunk_internal_link")
            chunk = await self._request("GET", self._url(str(link)))
            rows.extend(chunk.get("data_array") or [])
        else:
            raise DatabricksConnectorError(f"more than {self.config.max_chunks} result chunks")
        return [dict(zip(columns, row, strict=False)) for row in rows]


__all__ = [
    "DatabricksConnectorError",
    "DatabricksSchemaDriftError",
    "DatabricksSourceConfig",
    "DatabricksSQLSourceConnector",
]
