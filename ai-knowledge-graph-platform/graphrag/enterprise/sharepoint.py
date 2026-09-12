"""Microsoft Graph / SharePoint connector for the provider-neutral sync plane."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from graphrag.enterprise.models import ACLState, DocumentAccessPolicy, MetadataEnvelope, SyncChange, SyncChangeType
from graphrag.enterprise.sync import ContentSyncService
from graphrag.ingestion.connector_retry import RetryExhaustedError, send_with_retry
from graphrag.ingestion.document_loader import extract_document_links, load_document_content

log = structlog.get_logger(__name__)

_TOKEN_EXPIRY_MARGIN_SECONDS = 30.0
_DEFAULT_TOKEN_TTL_SECONDS = 3600.0
# Bounds delta()'s pagination loop -- protects against a malformed or
# maliciously long @odata.nextLink chain looping forever.
_DEFAULT_MAX_DELTA_PAGES = 500


class GraphConnectorError(RuntimeError):
    """Raised after retries are exhausted, or on a non-retryable HTTP error."""


@dataclass(frozen=True)
class SharePointSourceConfig:
    source_id: str
    tenant_id: str
    client_id: str
    client_secret_env: str
    site_id: str
    drive_id: str
    tenant: str

    @classmethod
    def from_mapping(cls, source_id: str, value: dict[str, Any]) -> "SharePointSourceConfig":
        required = ("tenant_id", "client_id", "client_secret_env", "site_id", "drive_id", "tenant")
        missing = [key for key in required if not str(value.get(key, "")).strip()]
        if missing:
            raise ValueError(f"SharePoint source {source_id!r} is missing: {', '.join(missing)}")
        return cls(source_id=source_id, **{key: str(value[key]) for key in required})


class MicrosoftGraphClient:
    """Minimal Graph client that keeps credentials in environment variables."""

    graph_base = "https://graph.microsoft.com/v1.0"

    def __init__(
        self,
        config: SharePointSourceConfig,
        client: httpx.AsyncClient | None = None,
        *,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        max_delta_pages: int = _DEFAULT_MAX_DELTA_PAGES,
    ):
        self.config = config
        self._client = client
        self._access_token = ""
        self._token_expires_at = 0.0
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._max_delta_pages = max_delta_pages

    async def _ensure_token(self) -> str:
        # Tracks real expiry and proactively re-fetches -- previously this
        # cached a token for the client's entire lifetime and never
        # refreshed it, even past real expiry.
        if self._access_token and time.monotonic() < self._token_expires_at - _TOKEN_EXPIRY_MARGIN_SECONDS:
            return self._access_token
        secret = os.getenv(self.config.client_secret_env, "")
        if not secret:
            raise ValueError(f"environment variable {self.config.client_secret_env!r} is not set")
        token_url = f"https://login.microsoftonline.com/{self.config.tenant_id}/oauth2/v2.0/token"
        # Routed through the same _send helper (injected or scratch client)
        # as every other request -- previously this always opened its own
        # throwaway httpx.AsyncClient, so a test's injected MockTransport
        # client could never observe or mock the token exchange at all.
        response = await self._send("POST", token_url, data={
            "client_id": self.config.client_id,
            "client_secret": secret,
            "grant_type": "client_credentials",
            "scope": "https://graph.microsoft.com/.default",
        })
        response.raise_for_status()
        payload = response.json()
        token = str(payload.get("access_token", ""))
        if not token:
            raise ValueError("Microsoft identity platform returned no access token")
        self._access_token = token
        try:
            self._token_expires_at = time.monotonic() + float(payload.get("expires_in"))
        except (TypeError, ValueError):
            self._token_expires_at = time.monotonic() + _DEFAULT_TOKEN_TTL_SECONDS
        return token

    async def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        """One raw HTTP call through the injected or a scratch client --
        no retry here, _request wraps this via send_with_retry."""
        if self._client is not None:
            return await self._client.request(method, url, **kwargs)
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            return await client.request(method, url, **kwargs)

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        token = await self._ensure_token()
        headers = {"Authorization": f"Bearer {token}", **kwargs.pop("headers", {})}
        try:
            return await send_with_retry(
                lambda: self._send(method, url, headers=headers, **kwargs),
                max_retries=self._max_retries,
                base_backoff_seconds=self._backoff_seconds,
            )
        except RetryExhaustedError as exc:
            raise GraphConnectorError(f"{method} {url} failed: {exc}") from exc

    async def delta(self, cursor: str = "") -> tuple[list[dict[str, Any]], str]:
        url = cursor or f"{self.graph_base}/sites/{self.config.site_id}/drives/{self.config.drive_id}/root/delta"
        items: list[dict[str, Any]] = []
        delta_link = cursor
        headers = {"Prefer": "deltashowremovedasdeleted,deltatraversepermissiongaps,deltashowsharingchanges"}
        pages = 0
        while url:
            pages += 1
            if pages > self._max_delta_pages:
                raise GraphConnectorError(
                    f"delta() exceeded {self._max_delta_pages} pages -- "
                    "a malformed @odata.nextLink chain, not real pagination"
                )
            response = await self._request("GET", url, headers=headers)
            payload = response.json()
            items.extend(payload.get("value", []))
            url = str(payload.get("@odata.nextLink") or "")
            delta_link = str(payload.get("@odata.deltaLink") or delta_link)
        return items, delta_link

    async def content(self, item_id: str) -> bytes:
        response = await self._request("GET", f"{self.graph_base}/drives/{self.config.drive_id}/items/{item_id}/content")
        return response.content

    async def permissions(self, item_id: str) -> list[dict[str, Any]]:
        response = await self._request(
            "GET", f"{self.graph_base}/drives/{self.config.drive_id}/items/{item_id}/permissions",
            headers={"Prefer": "hierarchicalsharing"},
        )
        return list(response.json().get("value", []))


class SharePointSyncConnector:
    """Translate Graph drive-item changes into durable platform sync changes."""

    def __init__(self, config: SharePointSourceConfig, graph_client=None, sync_service=None):
        self.config = config
        self._graph = graph_client or MicrosoftGraphClient(config)
        self._sync = sync_service or ContentSyncService()

    @classmethod
    def from_settings(cls, source_id: str) -> "SharePointSyncConnector":
        from graphrag.core.config import get_settings

        sources = get_settings().content_sync.get("sharepoint_sources", {})
        if not isinstance(sources, dict) or source_id not in sources:
            raise ValueError(f"SharePoint source {source_id!r} is not configured")
        return cls(SharePointSourceConfig.from_mapping(source_id, dict(sources[source_id])))

    async def sync_once(self) -> dict:
        cursor = await self._sync.current_cursor(self.config.source_id, self.config.tenant)
        items, next_cursor = await self._graph.delta(cursor)
        changes: list[SyncChange] = []
        failed_item_ids: list[str] = []
        for item in items:
            try:
                change = await self._to_change(item)
            except (httpx.HTTPError, GraphConnectorError) as exc:
                # Previously any content() fetch failure (a network blip,
                # throttling exhausting retries) propagated uncaught and
                # aborted sync_once() for every other item in the batch too
                # -- the adjacent permissions() call already failed closed
                # per-item instead of aborting; content() gets the same
                # per-item isolation now.
                item_id = str(item.get("id") or "")
                log.warning("sharepoint.item_processing_failed", item_id=item_id, error=str(exc))
                failed_item_ids.append(item_id)
                continue
            if change is not None:
                changes.append(change)
        # Only advance the cursor when every item in this batch was
        # processed cleanly. A failed item must not be silently dropped
        # forever by advancing past it -- holding the cursor back means the
        # whole batch (including already-succeeded items, safely
        # idempotent by external_id) is retried on the next sync_once()
        # until it fully succeeds.
        applied_cursor = next_cursor if not failed_item_ids else cursor
        result = await self._sync.apply_changes(
            self.config.source_id, changes, self.config.tenant, cursor=applied_cursor, trigger="delta",
        )
        return {**result, "source_id": self.config.source_id, "received": len(items),
                "failed_items": failed_item_ids}

    async def _to_change(self, item: dict[str, Any]) -> SyncChange | None:
        item_id = str(item.get("id") or "")
        if not item_id:
            return None
        if "deleted" in item or "@removed" in item:
            return SyncChange(change_type=SyncChangeType.DELETE, external_id=item_id)
        if item.get("folder"):
            return None
        filename = str(item.get("name") or "")
        if not filename:
            return None
        content = await self._graph.content(item_id)
        source_url = str(item.get("webUrl") or "")
        source_version = str(item.get("eTag") or item.get("cTag") or "")
        text = load_document_content(filename, content)
        try:
            policy = _access_policy(await self._graph.permissions(item_id))
        except (httpx.HTTPError, ValueError):
            policy = DocumentAccessPolicy(mode="restricted", state=ACLState.UNKNOWN, requires_group_resolution=True)
        return SyncChange(
            change_type=SyncChangeType.UPSERT,
            external_id=item_id,
            filename=filename,
            text=text,
            metadata=MetadataEnvelope(
                collection="sharepoint", schema_version="v1", source_system="sharepoint",
                external_id=item_id, source_url=source_url,
                source_version=source_version,
                content_type=str(item.get("file", {}).get("mimeType") or "text/plain"),
            ),
            access_policy=policy,
            document_links=extract_document_links(
                filename, content, base_url=source_url,
                source_system="sharepoint", source_version=source_version,
            ),
        )


def _access_policy(permissions: list[dict[str, Any]]) -> DocumentAccessPolicy:
    principals: set[str] = set()
    needs_groups = False
    for permission in permissions:
        subjects = [permission.get("grantedToV2"), permission.get("grantedTo")]
        subjects.extend(permission.get("grantedToIdentitiesV2") or [])
        if permission.get("link"):
            return DocumentAccessPolicy(mode="restricted", state=ACLState.UNKNOWN, requires_group_resolution=True)
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            for key, prefix in (("user", "user"), ("siteUser", "user"), ("group", "group"), ("siteGroup", "group")):
                identity = subject.get(key)
                if isinstance(identity, dict) and identity.get("id"):
                    principals.add(f"{prefix}:{identity['id']}")
                    needs_groups = needs_groups or prefix == "group"
    if not principals:
        return DocumentAccessPolicy(mode="restricted", state=ACLState.UNKNOWN, requires_group_resolution=True)
    return DocumentAccessPolicy(
        mode="restricted", state=ACLState.KNOWN,
        allow_principals=sorted(principals), requires_group_resolution=needs_groups,
    )
