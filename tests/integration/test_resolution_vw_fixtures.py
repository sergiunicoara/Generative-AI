"""§16 P4 exit criterion — the required Volkswagen fixture suite:

- 'Volks Wagen' resolves to 'Volkswagen Group' with at least two named
  relational signals.
- the same mention without relational evidence remains pending review.
- 'Volkswagen Financial Services' is present as a distractor.
- a weak-base candidate cannot auto-link only through bonuses.
- insufficient top1/top2 margin forces review.
- domain equality alone never auto-links.
- duplicate exact names do not deterministic-link.

Threshold calibration against this exact fixture is documented in
src/resolution/policy.py's module-level comment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.domain.conversation import Conversation, Mention
from src.domain.crm import Account, Opportunity
from src.domain.enums import ResolutionStatus
from src.domain.identity import crm_entity_id, mention_id, segment_id
from src.graph.execution import GraphExecutor
from src.graph.repositories.crm_repository import CrmRepository
from src.graph.repositories.conversation_repository import ConversationRepository
from src.resolution.candidates import CandidateGenerator
from src.resolution.pipeline import gather_relational_signals, resolve_mention

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 6, 15, tzinfo=timezone.utc)


def _ws() -> str:
    return f"ws-vw-{uuid4().hex[:8]}"


def _vw_mention(workspace_id: str, segment_id_: str) -> Mention:
    return Mention(
        mention_id=mention_id(segment_id_, 0, 11, "volks wagen", "ORG"),
        workspace_id=workspace_id,
        segment_id=segment_id_,
        char_start=0,
        char_end=11,
        surface_text="Volks Wagen",
        normalized_surface="volks wagen",
        entity_type="ORG",
    )


async def _seed_vw_accounts(crm_repo: CrmRepository, workspace_id: str) -> tuple[Account, Account]:
    vw_group = Account(
        account_id=crm_entity_id(workspace_id, "salesforce", "Account", "001VWGROUP"),
        workspace_id=workspace_id, source_record_id="rec-vw-group",
        name="Volkswagen Group", domain="vw.com",
    )
    vw_financial = Account(
        account_id=crm_entity_id(workspace_id, "salesforce", "Account", "001VWFIN"),
        workspace_id=workspace_id, source_record_id="rec-vw-financial",
        name="Volkswagen Financial Services", domain="vwfs.com",
    )
    await crm_repo.upsert_account(vw_group)
    await crm_repo.upsert_account(vw_financial)
    return vw_group, vw_financial


async def test_vw_positive_autolinks_with_two_plus_relational_signals(executor):
    workspace_id = _ws()
    crm_repo = CrmRepository(executor)
    conv_repo = ConversationRepository(executor)
    candidate_generator = CandidateGenerator(executor)

    vw_group, vw_financial = await _seed_vw_accounts(crm_repo, workspace_id)

    conversation = Conversation(
        conversation_id=crm_entity_id(workspace_id, "gong", "Conversation", "call-vw"),
        workspace_id=workspace_id, source_record_id="rec-call-vw", source_system="gong",
        external_call_id="call-vw", occurred_at=_T0, account_id=vw_group.account_id,
    )
    await conv_repo.upsert_conversation(conversation)

    opportunity = Opportunity(
        opportunity_id=crm_entity_id(workspace_id, "salesforce", "Opportunity", "006VW"),
        workspace_id=workspace_id, source_record_id="rec-opp-vw", account_id=vw_group.account_id,
        seller_id="seller-1", name="VW Deal", stage="Negotiation", is_open=True,
    )
    await crm_repo.upsert_opportunity(opportunity)

    seg_id = segment_id(conversation.conversation_id, 0)
    mention = _vw_mention(workspace_id, seg_id)

    signals = await gather_relational_signals(
        candidate_generator, workspace_id=workspace_id,
        conversation_id=conversation.conversation_id, seller_id="seller-1",
        participant_email_domain="vw.com",
    )
    assert len(signals.get(vw_group.account_id, frozenset())) >= 2  # sanity: signals actually gathered

    outcome = await resolve_mention(
        workspace_id=workspace_id, mention=mention, entity_type="Account",
        candidate_generator=candidate_generator, decided_at=_T0,
        relational_signals_by_entity=signals,
    )

    assert outcome.decision.status == ResolutionStatus.AUTO_LINKED
    assert outcome.decision.resolved_entity_id == vw_group.account_id
    assert outcome.decision.resolved_entity_id != vw_financial.account_id
    assert len(outcome.decision.relational_signals) >= 2
    assert outcome.mention.resolved_entity_id == vw_group.account_id


async def test_vw_positive_autolinks_with_real_semantic_scoring(executor, embedding_provider):
    """Same fixture as the positive case above, but with a real (not None)
    semantic score computed by a local sentence-transformers model — proves
    the embedding provider is actually wired into resolve_mention, not just
    unit-tested in isolation. See src/resolution/scoring.py's
    DEFAULT_LEXICAL_WEIGHT comment for why semantic is a small (3%) blend
    contribution: measured all-MiniLM-L6-v2 cosine similarity for short
    company-name fragments is real signal, but weaker than lexical fuzzy
    matching for this task."""
    workspace_id = _ws()
    crm_repo = CrmRepository(executor)
    conv_repo = ConversationRepository(executor)
    candidate_generator = CandidateGenerator(executor)

    vw_group, vw_financial = await _seed_vw_accounts(crm_repo, workspace_id)

    conversation = Conversation(
        conversation_id=crm_entity_id(workspace_id, "gong", "Conversation", "call-vw-embed"),
        workspace_id=workspace_id, source_record_id="rec-call-vw-embed", source_system="gong",
        external_call_id="call-vw-embed", occurred_at=_T0, account_id=vw_group.account_id,
    )
    await conv_repo.upsert_conversation(conversation)

    opportunity = Opportunity(
        opportunity_id=crm_entity_id(workspace_id, "salesforce", "Opportunity", "006VWEMBED"),
        workspace_id=workspace_id, source_record_id="rec-opp-vw-embed", account_id=vw_group.account_id,
        seller_id="seller-embed-1", name="VW Deal", stage="Negotiation", is_open=True,
    )
    await crm_repo.upsert_opportunity(opportunity)

    seg_id = segment_id(conversation.conversation_id, 0)
    mention = _vw_mention(workspace_id, seg_id)

    signals = await gather_relational_signals(
        candidate_generator, workspace_id=workspace_id,
        conversation_id=conversation.conversation_id, seller_id="seller-embed-1",
        participant_email_domain="vw.com",
    )

    outcome = await resolve_mention(
        workspace_id=workspace_id, mention=mention, entity_type="Account",
        candidate_generator=candidate_generator, decided_at=_T0,
        relational_signals_by_entity=signals, embedding_provider=embedding_provider,
    )

    assert outcome.decision.semantic_score is not None  # real embedding computed, not the None fallback
    assert outcome.decision.status == ResolutionStatus.AUTO_LINKED
    assert outcome.decision.resolved_entity_id == vw_group.account_id


async def test_vw_without_relational_evidence_stays_pending_review(executor):
    workspace_id = _ws()
    crm_repo = CrmRepository(executor)
    candidate_generator = CandidateGenerator(executor)
    vw_group, _ = await _seed_vw_accounts(crm_repo, workspace_id)

    mention = _vw_mention(workspace_id, segment_id("conv-no-evidence", 0))

    outcome = await resolve_mention(
        workspace_id=workspace_id, mention=mention, entity_type="Account",
        candidate_generator=candidate_generator, decided_at=_T0,
        relational_signals_by_entity={},  # no relational evidence at all
    )

    assert outcome.decision.status == ResolutionStatus.PENDING_REVIEW
    assert outcome.decision.resolved_entity_id is None


async def test_distractor_is_present_but_never_selected(executor):
    workspace_id = _ws()
    crm_repo = CrmRepository(executor)
    candidate_generator = CandidateGenerator(executor)
    vw_group, vw_financial = await _seed_vw_accounts(crm_repo, workspace_id)

    mention = _vw_mention(workspace_id, segment_id("conv-distractor", 0))

    outcome = await resolve_mention(
        workspace_id=workspace_id, mention=mention, entity_type="Account",
        candidate_generator=candidate_generator, decided_at=_T0,
        relational_signals_by_entity={vw_group.account_id: frozenset({"a", "b", "c"})},
    )

    assert vw_financial.account_id in outcome.candidates_shown  # present...
    assert outcome.decision.resolved_entity_id != vw_financial.account_id  # ...never confused for the match


async def test_weak_base_candidate_cannot_autolink_through_relational_bonus_alone(executor):
    workspace_id = _ws()
    crm_repo = CrmRepository(executor)
    candidate_generator = CandidateGenerator(executor)

    weak_match = Account(
        account_id=crm_entity_id(workspace_id, "salesforce", "Account", "001WEAK"),
        workspace_id=workspace_id, source_record_id="rec-weak", name="Totally Unrelated Company", domain="unrelated.com",
    )
    await crm_repo.upsert_account(weak_match)

    mention = _vw_mention(workspace_id, segment_id("conv-weak", 0))

    # inject the maximum possible relational evidence directly — even so, the
    # weak lexical/base match must not be pushed over the auto-link line.
    outcome = await resolve_mention(
        workspace_id=workspace_id, mention=mention, entity_type="Account",
        candidate_generator=candidate_generator, decided_at=_T0,
        relational_signals_by_entity={weak_match.account_id: frozenset({"a", "b", "c", "d", "e"})},
    )

    assert outcome.decision.status != ResolutionStatus.AUTO_LINKED
    assert outcome.decision.base_score is not None and outcome.decision.base_score < 0.70


async def test_duplicate_exact_names_do_not_deterministic_link_and_tied_margin_forces_review(executor):
    workspace_id = _ws()
    crm_repo = CrmRepository(executor)
    candidate_generator = CandidateGenerator(executor)

    dup_a = Account(
        account_id=crm_entity_id(workspace_id, "salesforce", "Account", "001DUPA"),
        workspace_id=workspace_id, source_record_id="rec-dup-a", name="Acme Corp",
    )
    dup_b = Account(
        account_id=crm_entity_id(workspace_id, "salesforce", "Account", "001DUPB"),
        workspace_id=workspace_id, source_record_id="rec-dup-b", name="Acme Corp",
    )
    await crm_repo.upsert_account(dup_a)
    await crm_repo.upsert_account(dup_b)

    seg_id = segment_id("conv-dup", 0)
    mention = Mention(
        mention_id=mention_id(seg_id, 0, 9, "acme corp", "ORG"),
        workspace_id=workspace_id, segment_id=seg_id, char_start=0, char_end=9,
        surface_text="Acme Corp", normalized_surface="Acme Corp", entity_type="ORG",
    )

    outcome = await resolve_mention(
        workspace_id=workspace_id, mention=mention, entity_type="Account",
        candidate_generator=candidate_generator, decided_at=_T0,
        relational_signals_by_entity={},  # identical for both — tied final scores
    )

    # Stage A3 saw 2 candidates for the exact name, so it correctly refused to
    # deterministic-link (fell through to the probabilistic path instead of
    # picking one arbitrarily).
    assert outcome.decision.status != ResolutionStatus.AUTO_LINKED
    assert outcome.decision.margin is not None and outcome.decision.margin < 0.08


async def test_domain_equality_alone_never_autolinks(executor):
    workspace_id = _ws()
    crm_repo = CrmRepository(executor)
    candidate_generator = CandidateGenerator(executor)

    account = Account(
        account_id=crm_entity_id(workspace_id, "salesforce", "Account", "001ACME"),
        workspace_id=workspace_id, source_record_id="rec-acme", name="Acme Global Holdings", domain="acme.com",
    )
    await crm_repo.upsert_account(account)

    seg_id = segment_id("conv-domain", 0)
    mention = Mention(
        mention_id=mention_id(seg_id, 0, 4, "acme", "ORG"),
        workspace_id=workspace_id, segment_id=seg_id, char_start=0, char_end=4,
        surface_text="Acme", normalized_surface="acme", entity_type="ORG",
    )

    signals = await gather_relational_signals(
        candidate_generator, workspace_id=workspace_id, participant_email_domain="acme.com",
    )
    assert signals.get(account.account_id) == frozenset({"participant_email_domain_matches_account"})

    outcome = await resolve_mention(
        workspace_id=workspace_id, mention=mention, entity_type="Account",
        candidate_generator=candidate_generator, decided_at=_T0,
        relational_signals_by_entity=signals,
    )

    assert outcome.decision.status != ResolutionStatus.AUTO_LINKED
