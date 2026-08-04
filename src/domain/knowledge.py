"""Sales knowledge entities (docs/plan.md §5 sales knowledge entities).

ContentAsset deliberately carries no direct Objection-mapping field here — §12
requires that mapping to have "an explicit source... a curated taxonomy or
reviewed Claim; it must not be invented at query time." That source is the sales
ontology YAML (config/ontologies/sales.yml, landing in Increment 4/P2), not a
field baked into this contract layer.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel


class Product(BaseModel):
    product_id: str
    workspace_id: str
    name: str


class Feature(BaseModel):
    feature_id: str
    workspace_id: str
    product_id: str
    name: str


class Objection(BaseModel):
    objection_id: str
    workspace_id: str
    category: str  # e.g. "PRICING", "SECURITY" — curated vocabulary, not free text
    description: str


class PainPoint(BaseModel):
    pain_point_id: str
    workspace_id: str
    description: str


class Blocker(BaseModel):
    blocker_id: str
    workspace_id: str
    description: str


class BuyingSignal(BaseModel):
    buying_signal_id: str
    workspace_id: str
    description: str


class ActionItem(BaseModel):
    action_item_id: str
    workspace_id: str
    description: str
    owner_id: str | None = None
    due_date: date | None = None


class Commitment(BaseModel):
    commitment_id: str
    workspace_id: str
    description: str
    committed_by: str | None = None
    due_date: date | None = None


class ContentAsset(BaseModel):
    content_asset_id: str
    workspace_id: str
    division_id: str | None = None  # §4 — Showpad-derived nodes may carry division_id
    title: str
    url: str
    content_type: str | None = None
    tags: list[str] = []


class Share(BaseModel):
    share_id: str
    workspace_id: str
    content_asset_id: str
    shared_with_contact_id: str
    shared_by_seller_id: str | None = None
    shared_at: datetime


class AssetView(BaseModel):
    asset_view_id: str
    workspace_id: str
    content_asset_id: str
    viewer_contact_id: str
    viewed_at: datetime
