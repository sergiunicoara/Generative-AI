"""Microsoft Graph / SharePoint connector for the provider-neutral sync plane."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from graphrag.enterprise.models import ACLState, DocumentAccessPolicy, MetadataEnvelope, SyncChange, SyncChangeType
from graphrag.enterprise.sync import ContentSyncService
from graphrag.ingestion.document_loader import extract_document_links, load_document_content


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

    def __init__(self, config: SharePointSourceConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._client = client
        self._access_token = ""

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        if not self._access_token:
            secret = os.getenv(self.config.client_secret_env, "")
            if not secret:
                raise ValueError(f"environment variable {self.config.client_secret_env!r} is not set")
            token_url = f"https://login.microsoftonline.com/{self.config.tenant_id}/oauth2/v2.0/token"
            async with httpx.AsyncClient(timeout=20) as token_client:
                token = await token_client.post(token_url, data={
                    "client_id": self.config.client_id,
                    "client_secret": secret,
                    "grant_type": "client_credentials",
                    "scope": "https://graph.microsoft.com/.default",
                })
                token.raise_for_status()
                self._access_token = str(token.json().get("access_token", ""))
            if not self._access_token:
                raise ValueError("Microsoft identity platform returned no access token")
        headers = {"Authorization": f"Bearer {self._access_token}", **kwargs.pop("headers", {})}
        if self._client is not None:
            response = await self._client.request(method, url, headers=headers, **kwargs)
        else:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                response = await client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response

    async def delta(self, cursor: str = "") -> tuple[list[dict[str, Any]], str]:
        url = cursor or f"{self.graph_base}/sites/{self.config.site_id}/drives/{self.config.drive_id}/root/delta"
        items: list[dict[str, Any]] = []
        delta_link = cursor
        headers = {"Prefer": "deltashowremovedasdeleted,deltatraversepermissiongaps,deltashowsharingchanges"}
        while url:
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
        changes = [change for item in items if (change := await self._to_change(item)) is not None]
        result = await self._sync.apply_changes(
            self.config.source_id, changes, self.config.tenant, cursor=next_cursor, trigger="delta",
        )
        return {**result, "source_id": self.config.source_id, "received": len(items)}

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
