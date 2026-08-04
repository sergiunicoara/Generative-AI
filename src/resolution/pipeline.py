"""§8 resolution orchestration for one Mention: Stage A (deterministic) first,
falling through to candidate generation -> scoring -> decision policy. Persists
nothing itself — callers (the ingestion/API layer) persist the returned Mention
and ResolutionDecision via src/graph/repositories/review_repository.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from src.domain.assertion import ResolutionDecision
from src.domain.conversation import Mention
from src.domain.enums import ResolutionStatus
from src.embedding.provider import EmbeddingProvider
from src.resolution.candidates import Candidate, CandidateGenerator, union_candidates
from src.resolution.deterministic import DeterministicRule, resolve_deterministic
from src.resolution.policy import PolicyThresholds, decide, decide_deterministic
from src.resolution.scoring import cosine_similarity, rank_candidates, score_candidate


@dataclass(frozen=True)
class ResolutionOutcome:
    mention: Mention
    decision: ResolutionDecision
    candidates_shown: list[str]


def _resolution_decision_id(mention_id: str) -> str:
    return hashlib.sha256(f"resolution\x1f{mention_id}".encode()).hexdigest()


async def gather_relational_signals(
    candidate_generator: CandidateGenerator,
    *,
    workspace_id: str,
    conversation_id: str | None = None,
    seller_id: str | None = None,
    participant_email_domain: str | None = None,
) -> dict[str, frozenset[str]]:
    """§8's relational signals, as an entity_id -> {signal names} map. Each
    source contributes at most one distinctly-named signal per entity, so a
    candidate matched by multiple sources naturally accumulates multiple
    signals (their union), never double-counted from one source."""
    signals: dict[str, set[str]] = {}
    if conversation_id:
        for c in await candidate_generator.relational_candidates_via_conversation_account(workspace_id, conversation_id):
            signals.setdefault(c.entity_id, set()).add("participant_belongs_to_account")
    if seller_id:
        for c in await candidate_generator.relational_candidates_via_seller_open_opportunity(workspace_id, seller_id):
            signals.setdefault(c.entity_id, set()).add("seller_owns_open_opportunity")
    if participant_email_domain:
        for c in await candidate_generator.relational_candidates_via_domain_match(workspace_id, participant_email_domain):
            signals.setdefault(c.entity_id, set()).add("participant_email_domain_matches_account")
    return {entity_id: frozenset(names) for entity_id, names in signals.items()}


async def resolve_mention(
    *,
    workspace_id: str,
    mention: Mention,
    entity_type: str,
    candidate_generator: CandidateGenerator,
    decided_at: datetime,
    relational_signals_by_entity: dict[str, frozenset[str]] | None = None,
    thresholds: PolicyThresholds = PolicyThresholds(),
    candidate_cap: int = 50,
    embedding_provider: EmbeddingProvider | None = None,
) -> ResolutionOutcome:
    relational_signals_by_entity = relational_signals_by_entity or {}

    exact_matches = await candidate_generator.exact_name_candidates(
        workspace_id, entity_type, mention.normalized_surface
    )
    deterministic = resolve_deterministic(
        DeterministicRule.A3_EXACT_CANONICAL_NAME, [c.entity_id for c in exact_matches]
    )

    if deterministic is not None:
        status = decide_deterministic(deterministic.entity_id)
        return _build_outcome(
            mention=mention, workspace_id=workspace_id, decided_at=decided_at,
            resolved_entity_id=deterministic.entity_id, status=status,
            top1=None, margin=None, candidates_shown=[deterministic.entity_id],
        )

    pool = await candidate_generator.all_names_in_workspace(workspace_id, entity_type)
    candidates: list[Candidate] = union_candidates(exact_matches, pool, cap=candidate_cap)

    # One batched embed() call for the mention + every candidate name — never
    # one call per candidate (same N+1-avoidance principle as the rest of
    # this repo). Absent entirely when no embedding_provider is passed;
    # score_candidate() falls back to semantic=None (lexical-only) either way.
    semantic_by_entity: dict[str, float] = {}
    if embedding_provider is not None and candidates:
        texts = [mention.surface_text] + [c.name for c in candidates]
        vectors = await embedding_provider.embed(texts)
        mention_vector, candidate_vectors = vectors[0], vectors[1:]
        semantic_by_entity = {
            c.entity_id: cosine_similarity(mention_vector, vec)
            for c, vec in zip(candidates, candidate_vectors)
        }

    scored = [
        score_candidate(
            entity_id=c.entity_id, entity_type=c.entity_type, name=c.name,
            mention_surface=mention.surface_text,
            semantic=semantic_by_entity.get(c.entity_id),
            relational_signals=relational_signals_by_entity.get(c.entity_id, frozenset()),
        )
        for c in candidates
    ]
    ranking = rank_candidates(scored)
    top1 = ranking.ranked[0] if ranking.ranked else None
    status = decide(top1, ranking.margin, thresholds=thresholds)
    resolved_entity_id = top1.entity_id if (top1 and status == ResolutionStatus.AUTO_LINKED) else None

    return _build_outcome(
        mention=mention, workspace_id=workspace_id, decided_at=decided_at,
        resolved_entity_id=resolved_entity_id, status=status,
        top1=top1, margin=ranking.margin if top1 else None,
        candidates_shown=[c.entity_id for c in candidates],
    )


def _build_outcome(
    *, mention, workspace_id, decided_at, resolved_entity_id, status, top1, margin, candidates_shown,
) -> ResolutionOutcome:
    decision = ResolutionDecision(
        resolution_decision_id=_resolution_decision_id(mention.mention_id),
        workspace_id=workspace_id,
        mention_id=mention.mention_id,
        resolved_entity_id=resolved_entity_id,
        status=status,
        lexical_score=top1.lexical if top1 else None,
        semantic_score=top1.semantic if top1 else None,
        base_score=top1.base if top1 else None,
        relational_bonus=top1.rel_bonus if top1 else None,
        final_score=top1.final if top1 else None,
        margin=margin,
        relational_signals=list(top1.relational_signals) if top1 else [],
        decided_at=decided_at,
    )
    resolved_mention = mention.model_copy(update={
        "resolved_entity_id": resolved_entity_id,
        "resolution_status": status,
    })
    return ResolutionOutcome(mention=resolved_mention, decision=decision, candidates_shown=candidates_shown)
