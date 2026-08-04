"""Showpad-style content adapter (docs/plan.md §4 initial adapters).

Showpad-derived nodes may carry division_id in addition to workspace_id (§4) —
division is the Showpad organizational/permission dimension inside a workspace,
not itself a tenant-isolation boundary.
"""

from __future__ import annotations

from src.domain.identity import crm_entity_id
from src.domain.knowledge import AssetView, ContentAsset
from src.ingestion.adapters.base import ParsedRecord, compute_content_hash


class ShowpadAdapter:
    source_system = "showpad"
    # This fixture models a point-in-time content export with no explicit
    # deletion/archival flag — unlike Salesforce's IsDeleted, there is nothing
    # trustworthy to tombstone on here (§6).
    supports_deletion_signal = False

    def parse_content_asset(self, workspace_id: str, raw: dict, *, division_id: str | None = None) -> ParsedRecord:
        external_id = raw["id"]
        content_asset_id = crm_entity_id(workspace_id, self.source_system, "ContentAsset", external_id)
        asset = ContentAsset(
            content_asset_id=content_asset_id,
            workspace_id=workspace_id,
            division_id=division_id,
            title=raw["title"],
            url=raw["url"],
            content_type=raw.get("type"),
            tags=list(raw.get("tags", [])),
        )
        return ParsedRecord(
            entity=asset,
            external_id=external_id,
            object_type="ContentAsset",
            content_hash=compute_content_hash(raw),
        )

    def parse_asset_view(
        self, workspace_id: str, content_asset_id: str, viewer_contact_id: str, raw: dict
    ) -> AssetView:
        """AssetViews are append-only engagement events, not versioned source
        records — no reconciliation needed, each raw view maps 1:1 to one
        AssetView node keyed by its own deterministic id."""
        external_id = raw["id"]
        asset_view_id = crm_entity_id(workspace_id, self.source_system, "AssetView", external_id)
        return AssetView(
            asset_view_id=asset_view_id,
            workspace_id=workspace_id,
            content_asset_id=content_asset_id,
            viewer_contact_id=viewer_contact_id,
            viewed_at=raw["viewed_at"],
        )
