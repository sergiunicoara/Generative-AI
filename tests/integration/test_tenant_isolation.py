"""§16 P1 exit criterion — 'two workspaces with intentionally identical names and
external IDs... reads, writes, relationships, full-text retrieval, vector
retrieval, review operations, and evidence lookups cannot cross the boundary.'

Every deterministic entity id already embeds workspace (src/domain/identity.py),
so two workspaces sharing an external_id never collide on id alone — that's not
what these tests are for. The real risk is a query that matches on a *shared,
non-unique* attribute (name, resolution_status, subject_id) without also scoping
by workspace_id; GraphExecutor.tenant_query()'s scoped_match() requirement exists
specifically to make that class of bug structurally hard to write. These tests
prove it holds for the real, running database — not just the regex guard.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.domain.assertion import Claim
from src.domain.conversation import Mention
from src.domain.crm import Account
from src.domain.enums import AdjudicationStatus, Polarity, ResolutionStatus, SpeakerRole
from src.domain.identity import crm_entity_id
from src.graph.repositories.claim_repository import ClaimRepository
from src.graph.repositories.crm_repository import CrmRepository
from src.graph.repositories.review_repository import ReviewRepository

pytestmark = pytest.mark.asyncio


def _workspace_pair() -> tuple[str, str]:
    suffix = uuid4().hex[:8]
    return f"ws-alpha-{suffix}", f"ws-beta-{suffix}"


async def test_duplicate_account_names_do_not_cross_workspace_boundary(executor):
    ws_alpha, ws_beta = _workspace_pair()
    repo = CrmRepository(executor)

    account_alpha = Account(
        account_id=crm_entity_id(ws_alpha, "salesforce", "Account", "001xx-shared"),
        workspace_id=ws_alpha,
        source_record_id="rec-alpha",
        name="Acme Corp",
    )
    account_beta = Account(
        account_id=crm_entity_id(ws_beta, "salesforce", "Account", "001xx-shared"),
        workspace_id=ws_beta,
        source_record_id="rec-beta",
        name="Acme Corp",  # identical name, identical source external_id — different workspace
    )
    await repo.upsert_account(account_alpha)
    await repo.upsert_account(account_beta)

    alpha_results = await repo.find_accounts_by_name(ws_alpha, "Acme Corp")
    beta_results = await repo.find_accounts_by_name(ws_beta, "Acme Corp")

    assert [a.account_id for a in alpha_results] == [account_alpha.account_id]
    assert [a.account_id for a in beta_results] == [account_beta.account_id]
    assert account_beta.account_id not in {a.account_id for a in alpha_results}
    assert account_alpha.account_id not in {a.account_id for a in beta_results}


async def test_get_account_rejects_a_valid_id_under_the_wrong_workspace(executor):
    """account_beta.account_id is a real, valid id in the database — just not
    under ws_alpha. Reading it while scoped to ws_alpha must return nothing, even
    though the id itself is genuine (proves the MATCH pattern's workspace_id
    property gates access, not just id uniqueness)."""
    ws_alpha, ws_beta = _workspace_pair()
    repo = CrmRepository(executor)

    account_beta = Account(
        account_id=crm_entity_id(ws_beta, "salesforce", "Account", "002xx"),
        workspace_id=ws_beta,
        source_record_id="rec-beta-2",
        name="Beta Only Corp",
    )
    await repo.upsert_account(account_beta)

    cross_workspace_read = await repo.get_account(ws_alpha, account_beta.account_id)
    same_workspace_read = await repo.get_account(ws_beta, account_beta.account_id)

    assert cross_workspace_read is None
    assert same_workspace_read is not None
    assert same_workspace_read.account_id == account_beta.account_id


async def test_claims_with_identical_subject_id_do_not_cross_workspace_boundary(executor):
    ws_alpha, ws_beta = _workspace_pair()
    repo = ClaimRepository(executor)
    now = datetime.now(timezone.utc)
    shared_subject_id = "contact-shared-subject"  # deliberately identical across workspaces

    def _claim(workspace_id: str, claim_id: str, object_value: str) -> Claim:
        return Claim(
            claim_id=claim_id,
            workspace_id=workspace_id,
            subject_id=shared_subject_id,
            predicate="RAISED_OBJECTION",
            object_value=object_value,
            polarity=Polarity.AFFIRMED,
            source_type="transcript",
            evidence_char_start=0,
            evidence_char_end=10,
            source_timestamp=now,
            speaker_role=SpeakerRole.BUYER,
            confidence=0.9,
            valid_from=now,
            transaction_from=now,
            adjudication_status=AdjudicationStatus.UNREVIEWED,
            retention_class="standard",
            created_at=now,
        )

    claim_alpha = _claim(ws_alpha, f"claim-alpha-{uuid4().hex[:8]}", "pricing")
    claim_beta = _claim(ws_beta, f"claim-beta-{uuid4().hex[:8]}", "security")
    await repo.create_claim(claim_alpha)
    await repo.create_claim(claim_beta)

    alpha_claims = await repo.list_claims_by_subject(ws_alpha, shared_subject_id)
    beta_claims = await repo.list_claims_by_subject(ws_beta, shared_subject_id)

    assert {c.claim_id for c in alpha_claims} == {claim_alpha.claim_id}
    assert {c.claim_id for c in beta_claims} == {claim_beta.claim_id}


async def test_mentions_with_identical_resolution_status_do_not_cross_workspace_boundary(executor):
    ws_alpha, ws_beta = _workspace_pair()
    repo = ReviewRepository(executor)

    mention_alpha = Mention(
        mention_id=f"mention-alpha-{uuid4().hex[:8]}",
        workspace_id=ws_alpha,
        segment_id="seg-alpha",
        char_start=0,
        char_end=10,
        surface_text="Volks Wagen",
        normalized_surface="volks wagen",
        entity_type="ORG",
        resolution_status=ResolutionStatus.PENDING_REVIEW,
    )
    mention_beta = Mention(
        mention_id=f"mention-beta-{uuid4().hex[:8]}",
        workspace_id=ws_beta,
        segment_id="seg-beta",
        char_start=0,
        char_end=10,
        surface_text="Volks Wagen",
        normalized_surface="volks wagen",
        entity_type="ORG",
        resolution_status=ResolutionStatus.PENDING_REVIEW,  # identical status, different workspace
    )
    await repo.upsert_mention(mention_alpha)
    await repo.upsert_mention(mention_beta)

    alpha_pending = await repo.list_mentions_by_status(ws_alpha, "PENDING_REVIEW")
    beta_pending = await repo.list_mentions_by_status(ws_beta, "PENDING_REVIEW")

    assert {m.mention_id for m in alpha_pending} == {mention_alpha.mention_id}
    assert {m.mention_id for m in beta_pending} == {mention_beta.mention_id}
