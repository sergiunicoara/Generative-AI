"""Salesforce-shaped CRM entities (docs/plan.md §5 CRM entities, §6 source versioning).

Every entity carries workspace_id (tenant boundary, §4) and source_record_id
(links back to the SourceRecord/SourceSnapshot that produced it, §6). Lead
conversion and Account merges are modeled as typed reference fields — the
resolver treats identifiers belonging to converted/merged records as aliases of
the surviving canonical entity (§5); materializing CONVERTED_TO/MERGED_INTO as
graph edges is a P1/P2 repository concern, not this package's.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator

from src.domain.enums import SourceStatus


class SourceRecord(BaseModel):
    """§6 — durable identity of one external source object across re-ingests."""
    model_config = ConfigDict(frozen=True)

    source_record_id: str
    workspace_id: str
    source_system: str
    object_type: str
    external_id: str
    source_status: SourceStatus = SourceStatus.ACTIVE
    first_seen_at: datetime
    last_seen_at: datetime
    current_snapshot_id: str | None = None


class SourceSnapshot(BaseModel):
    """§6 — one versioned content capture of a SourceRecord.

    Identical content_hash on re-ingest is a no-op (see src/domain/versioning.py);
    changed content creates a new snapshot and marks the prior one superseded.
    """
    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    source_record_id: str
    source_version: int
    content_hash: str
    ingestion_run_id: str
    captured_at: datetime
    superseded: bool = False


class Account(BaseModel):
    account_id: str
    workspace_id: str
    source_record_id: str
    name: str
    domain: str | None = None  # email-domain corroboration signal only (§8) — never sufficient alone
    merged_into_account_id: str | None = None  # §5 Account -[:MERGED_INTO]-> Account


class Contact(BaseModel):
    contact_id: str
    workspace_id: str
    source_record_id: str
    account_id: str | None = None
    name: str
    email: str | None = None

    @property
    def normalized_email(self) -> str | None:
        """Used by deterministic resolution rule A2 (§8) — exact normalized unique email.

        Plain str, not a validated email type: real CRM exports contain malformed-
        but-real addresses, and rejecting them at the contract layer would make
        ingestion fail on data P2's reconciliation should instead be able to flag
        and carry through, not silently drop.
        """
        return self.email.strip().lower() if self.email else None


class Lead(BaseModel):
    lead_id: str
    workspace_id: str
    source_record_id: str
    name: str
    email: str | None = None
    # §5 Lead -[:CONVERTED_TO]-> Contact|Account|Opportunity
    converted_to_type: str | None = None
    converted_to_id: str | None = None

    @model_validator(mode="after")
    def _converted_fields_are_paired(self) -> "Lead":
        if bool(self.converted_to_type) != bool(self.converted_to_id):
            raise ValueError("converted_to_type and converted_to_id must both be set or both be empty")
        if self.converted_to_type is not None and self.converted_to_type not in {
            "Contact", "Account", "Opportunity",
        }:
            raise ValueError("converted_to_type must be one of Contact, Account, Opportunity")
        return self


class Seller(BaseModel):
    seller_id: str
    workspace_id: str
    source_record_id: str
    name: str
    email: str | None = None


class Opportunity(BaseModel):
    opportunity_id: str
    workspace_id: str
    source_record_id: str
    account_id: str
    seller_id: str
    name: str
    stage: str
    is_open: bool = True


class OpportunityContactRole(BaseModel):
    role_id: str
    workspace_id: str
    opportunity_id: str
    contact_id: str
    role: str | None = None


class Meeting(BaseModel):
    meeting_id: str
    workspace_id: str
    source_record_id: str
    opportunity_id: str | None = None
    occurred_at: datetime
    subject: str | None = None


class Activity(BaseModel):
    activity_id: str
    workspace_id: str
    source_record_id: str
    opportunity_id: str | None = None
    activity_type: str
    occurred_at: datetime
