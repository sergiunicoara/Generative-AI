"""Tenant-safe repository for Account (P1 proof-of-pattern — every other
repository in this package follows this same shape: build patterns via
scoped_match(), never hand-roll a node pattern, always go through
GraphExecutor.tenant_query()).
"""

from __future__ import annotations

from src.domain.crm import Account, Contact, Lead, Opportunity
from src.graph.execution import GraphExecutor, scoped_match

_ACCOUNT_RETURN = (
    "a.account_id AS account_id, a.workspace_id AS workspace_id, "
    "a.source_record_id AS source_record_id, a.name AS name, a.domain AS domain, "
    "a.merged_into_account_id AS merged_into_account_id"
)

_CONTACT_RETURN = (
    "c.contact_id AS contact_id, c.workspace_id AS workspace_id, "
    "c.source_record_id AS source_record_id, c.account_id AS account_id, "
    "c.name AS name, c.email AS email"
)

_LEAD_RETURN = (
    "l.lead_id AS lead_id, l.workspace_id AS workspace_id, "
    "l.source_record_id AS source_record_id, l.name AS name, l.email AS email, "
    "l.converted_to_type AS converted_to_type, l.converted_to_id AS converted_to_id"
)

_OPPORTUNITY_RETURN = (
    "o.opportunity_id AS opportunity_id, o.workspace_id AS workspace_id, "
    "o.source_record_id AS source_record_id, o.account_id AS account_id, "
    "o.seller_id AS seller_id, o.name AS name, o.stage AS stage, o.is_open AS is_open"
)


class CrmRepository:
    def __init__(self, executor: GraphExecutor | None = None):
        self._executor = executor or GraphExecutor()

    async def upsert_account(self, account: Account) -> None:
        match = scoped_match("Account", "a", account_id="account_id")
        await self._executor.tenant_query(
            f"""
            MERGE {match}
            ON CREATE SET a.created_at = datetime()
            SET a.source_record_id = $source_record_id,
                a.name = $name,
                a.domain = $domain,
                a.merged_into_account_id = $merged_into_account_id,
                a.updated_at = datetime()
            """,
            workspace_id=account.workspace_id,
            account_id=account.account_id,
            source_record_id=account.source_record_id,
            name=account.name,
            domain=account.domain,
            merged_into_account_id=account.merged_into_account_id,
        )

    async def get_account(self, workspace_id: str, account_id: str) -> Account | None:
        match = scoped_match("Account", "a", account_id="account_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_ACCOUNT_RETURN}",
            workspace_id=workspace_id,
            account_id=account_id,
        )
        return Account(**rows[0]) if rows else None

    async def find_accounts_by_name(self, workspace_id: str, name: str) -> list[Account]:
        """Exact-name lookup, tenant-scoped. Used by the duplicate-exact-names
        isolation fixture — two workspaces may each have an Account named
        identically, and this must never return the other workspace's row."""
        match = scoped_match("Account", "a", name="name")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_ACCOUNT_RETURN}",
            workspace_id=workspace_id,
            name=name,
        )
        return [Account(**row) for row in rows]

    async def list_accounts(self, workspace_id: str, *, limit: int = 100) -> list[Account]:
        match = scoped_match("Account", "a")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_ACCOUNT_RETURN} ORDER BY a.name LIMIT $limit",
            workspace_id=workspace_id,
            limit=limit,
        )
        return [Account(**row) for row in rows]

    async def upsert_contact(self, contact: Contact) -> None:
        match = scoped_match("Contact", "c", contact_id="contact_id")
        await self._executor.tenant_query(
            f"""
            MERGE {match}
            ON CREATE SET c.created_at = datetime()
            SET c.source_record_id = $source_record_id,
                c.account_id = $account_id,
                c.name = $name,
                c.email = $email,
                c.normalized_email = $normalized_email,
                c.updated_at = datetime()
            """,
            workspace_id=contact.workspace_id,
            contact_id=contact.contact_id,
            source_record_id=contact.source_record_id,
            account_id=contact.account_id,
            name=contact.name,
            email=contact.email,
            normalized_email=contact.normalized_email,
        )

    async def get_contact(self, workspace_id: str, contact_id: str) -> Contact | None:
        match = scoped_match("Contact", "c", contact_id="contact_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_CONTACT_RETURN}",
            workspace_id=workspace_id,
            contact_id=contact_id,
        )
        return Contact(**rows[0]) if rows else None

    async def upsert_lead(self, lead: Lead) -> None:
        match = scoped_match("Lead", "l", lead_id="lead_id")
        await self._executor.tenant_query(
            f"""
            MERGE {match}
            ON CREATE SET l.created_at = datetime()
            SET l.source_record_id = $source_record_id,
                l.name = $name,
                l.email = $email,
                l.converted_to_type = $converted_to_type,
                l.converted_to_id = $converted_to_id,
                l.updated_at = datetime()
            """,
            workspace_id=lead.workspace_id,
            lead_id=lead.lead_id,
            source_record_id=lead.source_record_id,
            name=lead.name,
            email=lead.email,
            converted_to_type=lead.converted_to_type,
            converted_to_id=lead.converted_to_id,
        )

    async def get_lead(self, workspace_id: str, lead_id: str) -> Lead | None:
        match = scoped_match("Lead", "l", lead_id="lead_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_LEAD_RETURN}",
            workspace_id=workspace_id,
            lead_id=lead_id,
        )
        return Lead(**rows[0]) if rows else None

    async def upsert_opportunity(self, opportunity: Opportunity) -> None:
        match = scoped_match("Opportunity", "o", opportunity_id="opportunity_id")
        await self._executor.tenant_query(
            f"""
            MERGE {match}
            ON CREATE SET o.created_at = datetime()
            SET o.source_record_id = $source_record_id,
                o.account_id = $account_id,
                o.seller_id = $seller_id,
                o.name = $name,
                o.stage = $stage,
                o.is_open = $is_open,
                o.updated_at = datetime()
            """,
            workspace_id=opportunity.workspace_id,
            opportunity_id=opportunity.opportunity_id,
            source_record_id=opportunity.source_record_id,
            account_id=opportunity.account_id,
            seller_id=opportunity.seller_id,
            name=opportunity.name,
            stage=opportunity.stage,
            is_open=opportunity.is_open,
        )

    async def get_opportunity(self, workspace_id: str, opportunity_id: str) -> Opportunity | None:
        match = scoped_match("Opportunity", "o", opportunity_id="opportunity_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_OPPORTUNITY_RETURN}",
            workspace_id=workspace_id,
            opportunity_id=opportunity_id,
        )
        return Opportunity(**rows[0]) if rows else None
