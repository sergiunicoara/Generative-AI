"""A genuinely production-shaped tabular source connector: OAuth2
client-credentials auth, cursor pagination, bounded retry on transient
failure.

Closes a confirmed gap, not a claimed vendor integration: "SAP-shaped" and
"Snowflake-shaped" in this repo's energy demo are naming convention on
synthetic SQLite/JSON fixtures only (`scripts/create_energy_demo_sqlite.py`,
`ontology/mappings/energy-observations.rml.ttl`'s own comment) — no auth,
pagination, or retry logic exists anywhere in the connector layer. Both SAP
OData and Snowflake's SQL API are, at the transport level, OAuth2 + paginated
JSON HTTP — this connector proves that shape works end-to-end against a
mocked transport rather than claiming a specific vendor SDK integration this
repo has no credentials or Docker-mockable service to prove.

Template: `graphrag/enterprise/sharepoint.py`'s `MicrosoftGraphClient`
(OAuth2 client-credentials + token caching) — reused as a pattern, not
copied verbatim, since that client is Graph-API-specific (delta links,
permissions) and this one is generic tabular pagination.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from graphrag.core.connector_url_safety import assert_safe_connector_url
from graphrag.graph.source_catalog import SourceKind
from graphrag.ingestion.relational import _identifier


class RESTSourceConnectorError(RuntimeError):
    """Raised after retries are exhausted, or on a non-retryable HTTP error."""


@dataclass(frozen=True)
class OAuth2ClientCredentialsConfig:
    token_url: str
    client_id: str
    # Secret is read from this environment variable at request time, never
    # stored on the config object or logged — same posture as
    # SharePointSourceConfig.client_secret_env.
    client_secret_env: str
    scope: str = ""

    def __post_init__(self) -> None:
        assert_safe_connector_url(self.token_url, context="OAuth2ClientCredentialsConfig token_url")


@dataclass(frozen=True)
class RESTAPISourceConfig:
    base_url: str
    oauth: OAuth2ClientCredentialsConfig
    # Table name (as used in an R2RML rr:tableName / EntityTableMapping.table)
    # -> the endpoint path that serves it. Validated against the same
    # SQL-identifier shape relational.py's SQLite/Postgres connectors already
    # enforce on table names, before it's used to build a URL.
    table_paths: dict[str, str] = field(default_factory=dict)
    page_size: int = 100
    max_retries: int = 3
    # 0 in tests to avoid real delays; a real deployment should override this.
    backoff_seconds: float = 0.5
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        assert_safe_connector_url(self.base_url, context="RESTAPISourceConfig base_url")


class RESTAPISourceConnector:
    """`TabularSourceConnector`-conformant OAuth2 + paginated + retrying
    HTTP source. `client` is injectable (an `httpx.AsyncClient` or, in
    tests, one built on `httpx.MockTransport`) so this needs no live
    network or Docker to exercise.
    """

    kind = SourceKind.API

    def __init__(self, config: RESTAPISourceConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._client = client
        self._access_token = ""

    @property
    def uri(self) -> str:
        # Never the token or secret — matches PostgreSQLSourceConnector's
        # hide_password posture and SourceMapping's secret-key rejection
        # (graphrag/graph/source_catalog.py) for whatever ends up in
        # SourceSystem.uri via this property.
        return self.config.base_url

    async def _ensure_token(self) -> str:
        if self._access_token:
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
        token = str(response.json().get("access_token", ""))
        if not token:
            raise ValueError("token endpoint returned no access_token")
        self._access_token = token
        return token

    async def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        """One raw HTTP call through the injected or a scratch client — no
        retry here, _request_with_retry wraps this."""
        if self._client is not None:
            return await self._client.request(method, url, **kwargs)
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            return await client.request(method, url, **kwargs)

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        token = await self._ensure_token()
        headers = {"Authorization": f"Bearer {token}", **kwargs.pop("headers", {})}
        last_error: str = ""
        for attempt in range(self.config.max_retries):
            response = await self._send(method, url, headers=headers, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                if attempt + 1 < self.config.max_retries:
                    await asyncio.sleep(self.config.backoff_seconds * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        raise RESTSourceConnectorError(
            f"{method} {url} failed after {self.config.max_retries} attempts: {last_error}"
        )

    async def read_table(self, table: str) -> list[dict[str, Any]]:
        _identifier(table, "table")
        path = self.config.table_paths.get(table)
        if path is None:
            raise ValueError(f"no endpoint configured for table {table!r}")

        url = f"{self.config.base_url.rstrip('/')}{path}"
        params: dict[str, Any] | None = {"page_size": self.config.page_size}
        rows: list[dict[str, Any]] = []
        while url:
            response = await self._request_with_retry("GET", url, params=params)
            payload = response.json()
            rows.extend(payload.get("results", []))
            # A cursor response's `next` is a full URL carrying its own query
            # string — params must not be re-appended on top of it.
            url = str(payload.get("next") or "")
            params = None
        return rows


__all__ = [
    "OAuth2ClientCredentialsConfig",
    "RESTAPISourceConfig",
    "RESTAPISourceConnector",
    "RESTSourceConnectorError",
]
