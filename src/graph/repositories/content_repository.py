"""Tenant-safe repository for ContentAsset/AssetView — same pattern as
crm_repository.py."""

from __future__ import annotations

from src.domain.knowledge import AssetView, ContentAsset
from src.graph.execution import GraphExecutor, scoped_match

_ASSET_RETURN = (
    "n.content_asset_id AS content_asset_id, n.workspace_id AS workspace_id, "
    "n.division_id AS division_id, n.title AS title, n.url AS url, "
    "n.content_type AS content_type, n.tags AS tags"
)

_VIEW_RETURN = (
    "v.asset_view_id AS asset_view_id, v.workspace_id AS workspace_id, "
    "v.content_asset_id AS content_asset_id, v.viewer_contact_id AS viewer_contact_id, "
    "v.viewed_at AS viewed_at"
)


class ContentRepository:
    def __init__(self, executor: GraphExecutor | None = None):
        self._executor = executor or GraphExecutor()

    async def upsert_content_asset(self, asset: ContentAsset) -> None:
        match = scoped_match("ContentAsset", "n", content_asset_id="content_asset_id")
        await self._executor.tenant_query(
            f"""
            MERGE {match}
            ON CREATE SET n.created_at = datetime()
            SET n.division_id = $division_id,
                n.title = $title,
                n.url = $url,
                n.content_type = $content_type,
                n.tags = $tags,
                n.updated_at = datetime()
            """,
            workspace_id=asset.workspace_id,
            content_asset_id=asset.content_asset_id,
            division_id=asset.division_id,
            title=asset.title,
            url=asset.url,
            content_type=asset.content_type,
            tags=asset.tags,
        )

    async def get_content_asset(self, workspace_id: str, content_asset_id: str) -> ContentAsset | None:
        match = scoped_match("ContentAsset", "n", content_asset_id="content_asset_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_ASSET_RETURN}",
            workspace_id=workspace_id,
            content_asset_id=content_asset_id,
        )
        return ContentAsset(**rows[0]) if rows else None

    async def list_content_assets(self, workspace_id: str) -> list[ContentAsset]:
        match = scoped_match("ContentAsset", "n")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_ASSET_RETURN}",
            workspace_id=workspace_id,
        )
        return [ContentAsset(**row) for row in rows]

    async def upsert_asset_view(self, view: AssetView) -> None:
        match = scoped_match("AssetView", "v", asset_view_id="asset_view_id")
        await self._executor.tenant_query(
            f"""
            MERGE {match}
            SET v.content_asset_id = $content_asset_id,
                v.viewer_contact_id = $viewer_contact_id,
                v.viewed_at = $viewed_at
            """,
            workspace_id=view.workspace_id,
            asset_view_id=view.asset_view_id,
            content_asset_id=view.content_asset_id,
            viewer_contact_id=view.viewer_contact_id,
            viewed_at=view.viewed_at.isoformat(),
        )

    async def list_viewed_asset_ids(self, workspace_id: str, viewer_contact_id: str) -> set[str]:
        """§12 — 'exclude assets already viewed by that buyer.'"""
        match = scoped_match("AssetView", "v", viewer_contact_id="viewer_contact_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN DISTINCT v.content_asset_id AS content_asset_id",
            workspace_id=workspace_id,
            viewer_contact_id=viewer_contact_id,
        )
        return {row["content_asset_id"] for row in rows}
