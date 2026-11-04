"""Confluence Cloud connector for the provider-neutral sync plane.

Changed pages are found with a CQL search on ``lastmodified`` within the
configured spaces; the cursor is the newest page timestamp seen. Confluence
has no delete feed, so removed and trashed pages leave the graph through
``reconcile()``, which lists the pages that still exist and lets
``ContentSyncService.reconcile`` tombstone the rest.

Permissions
-----------
A Confluence reader needs space permission *and* must pass the read
restrictions of the page and of every ancestor page (restrictions are
inherited down the tree). The mapping to ``DocumentAccessPolicy`` (one allow
list, any match grants) is deliberately conservative:

* no restriction on the page or its ancestors: the space's readers, taken
  from ``space_principals`` in the source configuration; a space without that
  entry is ``UNKNOWN`` and denied, because space permissions are not read
  from the API here;
* restrictions on one or more levels: the intersection of the principals
  named at every restricted level. A reader who qualifies at two levels only
  through *different* groups is denied (false denial, never false
  permission). An empty intersection is ``UNKNOWN`` and denied;
* principals named in restrictions are assumed to hold space access, which
  Confluence itself requires before they can read the page;
* an unreadable restriction is ``UNKNOWN``, denied and audited.

Verification status: contract-tested against ``httpx.MockTransport`` only;
no Confluence site or credentials exist in this environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
import os
from typing import Any

import httpx
import structlog

from graphrag.core.connector_url_safety import assert_safe_connector_url
from graphrag.enterprise.models import ACLState, DocumentAccessPolicy, MetadataEnvelope, SyncChange, SyncChangeType
from graphrag.enterprise.sync import ContentSyncService
from graphrag.ingestion.connector_retry import RetryExhaustedError, send_with_retry

log = structlog.get_logger(__name__)

_SPACE_KEY = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_~")


class ConfluenceConnectorError(RuntimeError):
    """Retries exhausted, a non-success response, or a pagination limit hit."""


@dataclass(frozen=True)
class ConfluenceSourceConfig:
    source_id: str
    base_url: str  # https://<site>.atlassian.net
    email: str
    api_token_env: str
    space_keys: tuple[str, ...]
    tenant: str
    # Space key -> principals ("user:<accountId>", "group:<id>") with space read access.
    space_principals: dict[str, tuple[str, ...]] = field(default_factory=dict)
    page_limit: int = 50
    max_pages: int = 200
    max_retries: int = 3
    backoff_seconds: float = 0.5
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        assert_safe_connector_url(self.base_url, context="ConfluenceSourceConfig base_url")
        if not self.space_keys:
            raise ValueError("Confluence source needs at least one space key")
        for key in self.space_keys:
            if not key or not set(key) <= _SPACE_KEY:
                raise ValueError(f"invalid Confluence space key {key!r}")

    @classmethod
    def from_mapping(cls, source_id: str, mapping: dict[str, Any]) -> "ConfluenceSourceConfig":
        missing = [k for k in ("base_url", "email", "api_token_env", "space_keys", "tenant") if not mapping.get(k)]
        if missing:
            raise ValueError(f"Confluence source {source_id!r} is missing {missing}")
        return cls(
            source_id=source_id, base_url=str(mapping["base_url"]).rstrip("/"), email=str(mapping["email"]),
            api_token_env=str(mapping["api_token_env"]), space_keys=tuple(mapping["space_keys"]),
            tenant=str(mapping["tenant"]),
            space_principals={k: tuple(v) for k, v in (mapping.get("space_principals") or {}).items()},
        )


class ConfluenceClient:
    """Thin REST client: CQL search, page restrictions. ``client`` is injectable."""

    def __init__(self, config: ConfluenceSourceConfig, client: httpx.AsyncClient | None = None):
        self.config = config
        self._client = client

    def _auth(self) -> tuple[str, str]:
        token = os.getenv(self.config.api_token_env, "")
        if not token:
            raise ValueError(f"environment variable {self.config.api_token_env!r} is not set")
        return (self.config.email, token)

    async def _send(self, url: str, params: dict[str, Any]) -> httpx.Response:
        auth = self._auth()
        if self._client is not None:
            return await self._client.get(url, params=params, auth=auth)
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds, follow_redirects=False) as client:
            return await client.get(url, params=params, auth=auth)

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        try:
            response = await send_with_retry(
                lambda: self._send(url, params or {}),
                max_retries=self.config.max_retries, base_backoff_seconds=self.config.backoff_seconds,
            )
        except RetryExhaustedError as exc:
            raise ConfluenceConnectorError(f"GET {path} failed: {exc}") from exc
        if response.status_code >= 400:
            raise ConfluenceConnectorError(f"GET {path} returned HTTP {response.status_code}")
        return response.json()

    def _spaces_cql(self) -> str:
        return "space in (" + ",".join(f'"{key}"' for key in self.config.space_keys) + ")"

    async def search(self, cql: str, *, expand: str = "") -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        start = 0
        for _ in range(self.config.max_pages):
            params: dict[str, Any] = {"cql": cql, "limit": self.config.page_limit, "start": start}
            if expand:
                params["expand"] = expand
            payload = await self._get("/wiki/rest/api/content/search", params)
            batch = list(payload.get("results") or [])
            results.extend(batch)
            if not batch or not (payload.get("_links") or {}).get("next"):
                return results
            start += len(batch)
        raise ConfluenceConnectorError(f"more than {self.config.max_pages} result pages")

    async def changed_pages(self, since: str) -> list[dict[str, Any]]:
        cql = f"type=page AND {self._spaces_cql()}"
        if since:
            # CQL compares at minute precision; ">=" re-reads the boundary minute
            # (an idempotent re-upsert) instead of skipping it.
            stamp = datetime.fromisoformat(since).astimezone(timezone.utc).strftime("%Y/%m/%d %H:%M")
            cql += f' AND lastmodified >= "{stamp}"'
        return await self.search(cql + " order by lastmodified asc",
                                 expand="body.storage,version,ancestors,space")

    async def page_ids(self) -> list[str]:
        return [str(page["id"]) for page in await self.search(f"type=page AND {self._spaces_cql()}") if page.get("id")]

    async def read_restrictions(self, page_id: str) -> set[str]:
        payload = await self._get(f"/wiki/rest/api/content/{page_id}/restriction/byOperation/read",
                                  {"expand": "restrictions.user,restrictions.group"})
        restrictions = payload.get("restrictions") or {}
        principals = {f"user:{u['accountId']}" for u in (restrictions.get("user") or {}).get("results", [])
                      if u.get("accountId")}
        principals |= {f"group:{g.get('id') or g.get('name')}" for g in (restrictions.get("group") or {}).get("results", [])
                       if g.get("id") or g.get("name")}
        return principals


class _Text(HTMLParser):
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def storage_to_text(storage: str) -> str:
    parser = _Text()
    parser.feed(storage or "")
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def access_policy(levels: list[set[str]], space_readers: tuple[str, ...] | None) -> tuple[DocumentAccessPolicy, str]:
    """Map page-tree restrictions to one allow list (see the module docstring).

    ``levels`` holds the read-restriction principals of the page and each
    ancestor (an empty set means that level is unrestricted)."""
    restricted = [level for level in levels if level]
    if not restricted:
        if not space_readers:
            return DocumentAccessPolicy(mode="restricted", state=ACLState.UNKNOWN), "space_readers_unknown"
        allow = set(space_readers)
    else:
        allow = set.intersection(*restricted)
        if not allow:
            return DocumentAccessPolicy(mode="restricted", state=ACLState.UNKNOWN), "restriction_intersection_empty"
    return DocumentAccessPolicy(
        mode="restricted", state=ACLState.KNOWN, allow_principals=sorted(allow),
        requires_group_resolution=any(p.startswith("group:") for p in allow),
    ), ""


class ConfluenceSyncConnector:
    """Translate Confluence page changes into platform sync changes."""

    def __init__(self, config: ConfluenceSourceConfig, client=None, sync_service=None):
        self.config = config
        self._client = client or ConfluenceClient(config)
        self._sync = sync_service or ContentSyncService()
        self.acl_audit: list[dict] = []

    @classmethod
    def from_settings(cls, source_id: str) -> "ConfluenceSyncConnector":
        from graphrag.core.config import get_settings

        sources = get_settings().content_sync.get("confluence_sources", {})
        if not isinstance(sources, dict) or source_id not in sources:
            raise ValueError(f"Confluence source {source_id!r} is not configured")
        return cls(ConfluenceSourceConfig.from_mapping(source_id, dict(sources[source_id])))

    async def sync_once(self) -> dict:
        cursor = await self._sync.current_cursor(self.config.source_id, self.config.tenant)
        pages = await self._client.changed_pages(cursor)
        restriction_cache: dict[str, set[str] | None] = {}
        changes: list[SyncChange] = []
        failed: list[str] = []
        newest = cursor
        for page in pages:
            page_id = str(page.get("id") or "")
            try:
                change = await self._to_change(page, restriction_cache)
            except (httpx.HTTPError, ConfluenceConnectorError) as exc:
                log.warning("confluence.page_processing_failed", page_id=page_id, error=str(exc))
                failed.append(page_id)
                continue
            if change is not None:
                changes.append(change)
            when = str((page.get("version") or {}).get("when") or "")
            if when and (not newest or _ts(when) > _ts(newest)):
                newest = when
        applied = newest if not failed else cursor
        result = await self._sync.apply_changes(self.config.source_id, changes, self.config.tenant,
                                                cursor=applied, trigger="delta")
        return {**result, "source_id": self.config.source_id, "received": len(pages), "failed_items": failed}

    async def reconcile(self) -> dict:
        """Tombstone pages that no longer exist (deleted, trashed or moved out of scope)."""
        return await self._sync.reconcile(self.config.source_id, await self._client.page_ids(), self.config.tenant)

    async def _restrictions(self, page_id: str, cache: dict[str, set[str] | None]) -> set[str] | None:
        if page_id not in cache:
            try:
                cache[page_id] = await self._client.read_restrictions(page_id)
            except (httpx.HTTPError, ConfluenceConnectorError, ValueError):
                cache[page_id] = None
        return cache[page_id]

    async def _to_change(self, page: dict[str, Any], cache: dict[str, set[str] | None]) -> SyncChange | None:
        page_id = str(page.get("id") or "")
        if not page_id:
            return None
        if str(page.get("status") or "current") != "current":
            return SyncChange(change_type=SyncChangeType.DELETE, external_id=page_id)
        title = str(page.get("title") or page_id)
        text = storage_to_text(str(((page.get("body") or {}).get("storage") or {}).get("value") or ""))
        space_key = str((page.get("space") or {}).get("key") or "")
        observed = datetime.now(timezone.utc)
        levels: list[set[str]] = []
        unreadable = False
        for level_id in [page_id, *[str(a.get("id")) for a in page.get("ancestors") or [] if a.get("id")]]:
            principals = await self._restrictions(level_id, cache)
            if principals is None:
                unreadable = True
                break
            levels.append(principals)
        if unreadable:
            policy, reason = DocumentAccessPolicy(mode="restricted", state=ACLState.UNKNOWN), "restrictions_unreadable"
        else:
            policy, reason = access_policy(levels, self.config.space_principals.get(space_key))
        if reason:
            self.acl_audit.append({"item_id": page_id, "event": "acl_unresolved", "outcome": "denied",
                                   "reason": reason, "observed_at": observed.isoformat()})
        policy = policy.model_copy(update={"observed_at": observed})
        links = page.get("_links") or {}
        version = page.get("version") or {}
        return SyncChange(
            change_type=SyncChangeType.UPSERT,
            external_id=page_id,
            filename=f"{title}.txt",
            text=text or title,
            metadata=MetadataEnvelope(
                collection="confluence", schema_version="v1", source_system="confluence",
                external_id=page_id, source_url=f"{self.config.base_url}/wiki{links.get('webui', '')}",
                source_version=str(version.get("number") or ""), content_type="text/html",
                discovery_metadata={"space_key": space_key} if space_key else {},
            ),
            access_policy=policy,
        )


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


__all__ = [
    "ConfluenceClient",
    "ConfluenceConnectorError",
    "ConfluenceSourceConfig",
    "ConfluenceSyncConnector",
    "access_policy",
    "storage_to_text",
]
