"""§12 — Context Graph builder response shape."""

from __future__ import annotations

from pydantic import BaseModel

from src.domain.assertion import Claim, Conflict


class EvidenceReference(BaseModel):
    claim_id: str
    source_segment_id: str | None
    evidence_char_start: int
    evidence_char_end: int
    excerpt: str


class SelectedItem(BaseModel):
    claim_id: str
    score: float
    reason: str


class ContextGraphResult(BaseModel):
    workspace_id: str
    claims: list[Claim] = []
    evidence: list[EvidenceReference] = []
    unresolved_mention_ids: list[str] = []
    conflicts: list[Conflict] = []
    selected_items: list[SelectedItem] = []
    budget_max_nodes: int
    budget_max_tokens: int
    nodes_used: int
    tokens_used: int
    truncated: bool
