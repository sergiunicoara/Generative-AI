"""SAP Gateway/S4HANA OData v2 source connector.

Vendor-specific, unlike `graphrag/ingestion/http_source.py`'s generic
`RESTAPISourceConnector`: SAP's real OData v2 read shape differs from a
flat `{"results": [...], "next": "<url>"}` cursor in three ways this
connector models explicitly --

1. Every request carries an `sap-client` header (the client/tenant number).
2. The collection envelope is `{"d": {"results": [...], "__next": "<url>"}}`,
   not a flat top-level object.
3. Real OAuth2 token expiry is tracked and the token is proactively
   re-fetched, closing a gap `RESTAPISourceConnector` has today (it caches a
   token for the connector's whole lifetime, never re-fetching).

Verification status
--------------------
Contract-tested against `httpx.MockTransport` only. No live SAP
Gateway/S4HANA instance or credentials exist in this environment -- see
`graphrag/ingestion/http_source.py`'s own module docstring for the same
honest framing it already established for the vendor-neutral connector.
Treat this as a documented, tested protocol shape, not a certified vendor
integration.
"""

from __future__ import annotations

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

# Margin subtracted from a token's reported expiry so a request in flight
# never straddles the exact expiry instant.
_TOKEN_EXPIRY_MARGIN_SECONDS = 30.0
_DEFAULT_TOKEN_TTL_SECONDS = 3600.0


class SAPODataConnectorError(RuntimeError):
    """Raised after retries are exhausted, or on a non-retryable HTTP error."""


class SAPSchemaDriftError(RuntimeError):
    """Raised when an entity set's response is missing a required column --
    fail closed rather than silently ingesting an incomplete row shape. An
    unlisted *extra* column is tolerated (additive drift is forward
    compatible) and only logged."""


@dataclass(frozen=True)
class SAPODataSourceConfig:
    base_url: str
    # SAP client/tenant number, e.g. "100" -- sent as the sap-client header
    # on every request, distinct from this platform's own `tenant` concept.
    sap_client: str
    oauth: OAuth2ClientCredentialsConfig
    # Table name (as used in an R2RML rr:tableName) -> the OData entity-set
    # path that serves it, e.g. {"assets": "/AssetSet"}.
    table_paths: dict[str, str] = field(default_factory=dict)
    # Table name -> required column names. Checked once against the first
    # non-empty page of a read_table() call.
    expected_columns: dict[str, frozenset[str]] = field(default_factory=dict)
    page_size: int = 100
    max_retries: int = 3
    # 0 in tests to avoid real delays; a real deployment should override this.
    backoff_seconds: float = 0.5
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        assert_safe_connector_url(self.base_url, context="SAPODataSourceConfig base_url")


class SAPODataSourceConnector:
    """`TabularSourceConnector`-conformant SAP OData v2 read connector.

    `client` is injectable (an `httpx.AsyncClient` or, in tests, one built
    on `httpx.MockTransport`) so this needs no live network or Docker to
    exercise -- same posture as `RESTAPISourceConnector`.
    """

    kind = SourceKind.API

    def __init__(self, config: SAPODataSourceConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._client = client
        self._access_token = ""
        self._token_expires_at = 0.0

    @property
    def uri(self) -> str:
        # Never the token or secret.
        return self.config.base_url

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
            raise ValueError("SAP OAuth token endpoint returned no access_token")
        self._access_token = token
        try:
            self._token_expires_at = time.monotonic() + float(payload.get("expires_in"))
        except (TypeError, ValueError):
            # No/unparseable expires_in -- treat as long-lived rather than
            # crash, matching RESTAPISourceConnector's tolerant handling of
            # an unexpected token response shape.
            self._token_expires_at = time.monotonic() + _DEFAULT_TOKEN_TTL_SECONDS
        return token

    async def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        """One raw HTTP call through the injected or a scratch client --
        no retry here, _request wraps this via send_with_retry."""
        if self._client is not None:
            return await self._client.request(method, url, **kwargs)
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            return await client.request(method, url, **kwargs)

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        token = await self._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "sap-client": self.config.sap_client,
            **kwargs.pop("headers", {}),
        }
        try:
            return await send_with_retry(
                lambda: self._send(method, url, headers=headers, **kwargs),
                max_retries=self.config.max_retries,
                base_backoff_seconds=self.config.backoff_seconds,
            )
        except RetryExhaustedError as exc:
            raise SAPODataConnectorError(f"{method} {url} failed: {exc}") from exc

    def _check_schema(self, table: str, rows: list[dict[str, Any]]) -> None:
        required = self.config.expected_columns.get(table)
        if not required or not rows:
            return
        present = set(rows[0].keys())
        missing = required - present
        if missing:
            raise SAPSchemaDriftError(
                f"SAP entity set {table!r} is missing required column(s): {sorted(missing)}"
            )
        extra = present - required
        if extra:
            log.info("sap_source.unexpected_columns", table=table, extra=sorted(extra))

    async def read_table(self, table: str) -> list[dict[str, Any]]:
        _identifier(table, "table")
        path = self.config.table_paths.get(table)
        if path is None:
            raise ValueError(f"no OData entity-set path configured for table {table!r}")

        url = f"{self.config.base_url.rstrip('/')}{path}"
        params: dict[str, Any] | None = {"$format": "json", "$top": self.config.page_size}
        rows: list[dict[str, Any]] = []
        schema_checked = False
        while url:
            response = await self._request("GET", url, params=params)
            envelope = response.json().get("d", {})
            page = envelope.get("results", [])
            rows.extend(page)
            if not schema_checked and page:
                self._check_schema(table, page)
                schema_checked = True
            # A cursor response's __next is a full URL carrying its own
            # query string -- params must not be re-appended on top of it.
            url = str(envelope.get("__next") or "")
            params = None
        return rows


__all__ = [
    "SAPODataConnectorError",
    "SAPODataSourceConfig",
    "SAPODataSourceConnector",
    "SAPSchemaDriftError",
]
