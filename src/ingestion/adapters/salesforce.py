"""Salesforce-shaped CRM adapter (docs/plan.md §4 initial adapters).

Parses raw Salesforce export dicts (the shapes REST/Bulk API callers actually
see: `Id`, `Name`, `AccountId`, `OwnerId`, `IsDeleted`, `MasterRecordId`,
`IsConverted`, `ConvertedContactId`, ...) into this repo's own src/domain/crm.py
models. Every adapter is behind this typed interface; adding another CRM
provider must not require changing src/domain/crm.py or the reconciliation/
repository layers downstream of it (§4).
"""

from __future__ import annotations

from src.domain.crm import Account, Contact, Lead, Opportunity
from src.domain.identity import crm_entity_id
from src.ingestion.adapters.base import ParsedRecord, compute_content_hash


class SalesforceAdapter:
    source_system = "salesforce"
    # Salesforce exposes IsDeleted on every object (recycle-bin semantics) on a
    # full/incremental export — a genuinely trustworthy deletion signal, unlike
    # an adapter that only ever sees a partial batch with no way to distinguish
    # "not in this batch" from "actually deleted" (§6: tombstone only when the
    # adapter supplies a trustworthy signal).
    supports_deletion_signal = True

    def parse_account(self, workspace_id: str, raw: dict) -> ParsedRecord:
        external_id = raw["Id"]
        account_id = crm_entity_id(workspace_id, self.source_system, "Account", external_id)
        account = Account(
            account_id=account_id,
            workspace_id=workspace_id,
            source_record_id=account_id,
            name=raw["Name"],
            domain=raw.get("Website"),
            merged_into_account_id=(
                crm_entity_id(workspace_id, self.source_system, "Account", raw["MasterRecordId"])
                if raw.get("MasterRecordId")
                else None
            ),
        )
        return ParsedRecord(
            entity=account,
            external_id=external_id,
            object_type="Account",
            content_hash=compute_content_hash(raw),
            extra={"is_deleted": bool(raw.get("IsDeleted"))},
        )

    def parse_contact(self, workspace_id: str, raw: dict) -> ParsedRecord:
        external_id = raw["Id"]
        contact_id = crm_entity_id(workspace_id, self.source_system, "Contact", external_id)
        account_id = (
            crm_entity_id(workspace_id, self.source_system, "Account", raw["AccountId"])
            if raw.get("AccountId")
            else None
        )
        contact = Contact(
            contact_id=contact_id,
            workspace_id=workspace_id,
            source_record_id=contact_id,
            account_id=account_id,
            name=raw["Name"],
            email=raw.get("Email"),
        )
        return ParsedRecord(
            entity=contact,
            external_id=external_id,
            object_type="Contact",
            content_hash=compute_content_hash(raw),
            extra={"is_deleted": bool(raw.get("IsDeleted"))},
        )

    def parse_lead(self, workspace_id: str, raw: dict) -> ParsedRecord:
        external_id = raw["Id"]
        lead_id = crm_entity_id(workspace_id, self.source_system, "Lead", external_id)

        converted_to_type: str | None = None
        converted_to_id: str | None = None
        if raw.get("IsConverted"):
            # A real Salesforce conversion creates a Contact + Account together
            # and optionally an Opportunity, but src/domain/crm.py's Lead models
            # only one converted_to_type/id pair — Contact is the most specific
            # "who the lead became" and takes priority, then Account, then
            # Opportunity, documented here since it's a real simplification.
            if raw.get("ConvertedContactId"):
                converted_to_type = "Contact"
                converted_to_id = crm_entity_id(
                    workspace_id, self.source_system, "Contact", raw["ConvertedContactId"]
                )
            elif raw.get("ConvertedAccountId"):
                converted_to_type = "Account"
                converted_to_id = crm_entity_id(
                    workspace_id, self.source_system, "Account", raw["ConvertedAccountId"]
                )
            elif raw.get("ConvertedOpportunityId"):
                converted_to_type = "Opportunity"
                converted_to_id = crm_entity_id(
                    workspace_id, self.source_system, "Opportunity", raw["ConvertedOpportunityId"]
                )

        lead = Lead(
            lead_id=lead_id,
            workspace_id=workspace_id,
            source_record_id=lead_id,
            name=raw["Name"],
            email=raw.get("Email"),
            converted_to_type=converted_to_type,
            converted_to_id=converted_to_id,
        )
        return ParsedRecord(
            entity=lead,
            external_id=external_id,
            object_type="Lead",
            content_hash=compute_content_hash(raw),
            extra={"is_deleted": bool(raw.get("IsDeleted"))},
        )

    def parse_opportunity(self, workspace_id: str, raw: dict) -> ParsedRecord:
        external_id = raw["Id"]
        opportunity_id = crm_entity_id(workspace_id, self.source_system, "Opportunity", external_id)
        opportunity = Opportunity(
            opportunity_id=opportunity_id,
            workspace_id=workspace_id,
            source_record_id=opportunity_id,
            account_id=crm_entity_id(workspace_id, self.source_system, "Account", raw["AccountId"]),
            seller_id=crm_entity_id(workspace_id, self.source_system, "Seller", raw["OwnerId"]),
            name=raw["Name"],
            stage=raw["StageName"],
            is_open=not raw.get("IsClosed", False),
        )
        return ParsedRecord(
            entity=opportunity,
            external_id=external_id,
            object_type="Opportunity",
            content_hash=compute_content_hash(raw),
            extra={"is_deleted": bool(raw.get("IsDeleted"))},
        )
