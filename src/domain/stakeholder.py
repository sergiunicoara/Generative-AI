"""§5 closing note — stakeholder assignments are not stored as mutable properties
on a HAS_STAKEHOLDER relationship. StakeholderAssignment is the materialized
current view; source-specific role/influence/sentiment/authority remain Claims
with provenance, and conflicting Claims can coexist. The graph shape
(Opportunity)-[:HAS_ASSIGNMENT]->(StakeholderAssignment)-[:ASSIGNS]->(Contact) is
a P1 repository concern; this is just the materialized-view record.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class StakeholderAssignment(BaseModel):
    assignment_id: str
    workspace_id: str
    opportunity_id: str
    contact_id: str
    role: str | None = None
    influence: str | None = None
    sentiment: str | None = None
    authority: str | None = None
    updated_at: datetime
