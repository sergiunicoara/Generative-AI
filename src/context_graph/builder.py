"""§12 Context Graph builder — bounded, scored selection over a workspace's
Claims for a given scope.

This vertical slice's builder is scoped to what it actually needs to serve:
Claims connected to a Conversation or a subject, not arbitrary open-domain
retrieval over free-text questions across the whole graph (that would need a
real full-text/vector question-ranking layer this slice doesn't build — see
'relevance' in _score_claim's docstring for the honest limitation). The 7-step
pipeline is genuinely implemented at this scope:

1. deterministic scope filters      -> ContextGraphScope
2. tenant-safe candidate retrieval  -> ClaimRepository (already tenant_query-backed)
3. bounded traversal                -> list_claims_for_conversation's fixed-depth,
                                        fixed-relationship-allowlist traversal
4. scoring                          -> _score_claim
5. greedy budget selection          -> build()'s main loop
6. diversity caps                   -> per-predicate cap
7. conflict preservation            -> response shape carries `conflicts`;
                                        no Conflict-detection wiring exists yet
                                        between Claims in this vertical slice,
                                        so it's always empty here, not silently
                                        dropped from the contract.

Every call is a single bounded repository fetch — no per-Claim follow-up query,
so this never becomes N+1 (§12: 'Avoid N+1 repository calls').
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.context_graph.models import ContextGraphResult, EvidenceReference, SelectedItem
from src.domain.assertion import Claim
from src.domain.enums import AdjudicationStatus
from src.graph.repositories.claim_repository import ClaimRepository

DEFAULT_MAX_NODES = 50
DEFAULT_MAX_TOKENS = 4000
DEFAULT_PREDICATE_DIVERSITY_CAP = 5

_ADJUDICATION_WEIGHT = {
    AdjudicationStatus.ACCEPTED: 1.0,
    AdjudicationStatus.UNREVIEWED: 0.7,
    AdjudicationStatus.DISPUTED: 0.4,
    AdjudicationStatus.REJECTED: 0.0,
}


@dataclass(frozen=True)
class ContextGraphScope:
    workspace_id: str
    conversation_id: str | None = None
    subject_id: str | None = None


def _claim_tokens(claim: Claim) -> int:
    """A rough word-count proxy, not a real tokenizer — matches the same
    documented simplification src/extraction/windowing.py already uses for
    token budgeting, for the same reason (no tokenizer is pinned yet)."""
    text = f"{claim.subject_id} {claim.predicate} {claim.object_value or claim.object_id or ''}"
    return len(text.split())


def _score_claim(claim: Claim, *, now: datetime) -> float:
    """confidence, recency, and adjudication_status are genuinely computed.
    'Relevance'/'source authority' collapse into the scope filter itself here —
    this builder answers 'what's the well-evidenced context for this specific
    conversation/subject', not 'rank all Claims in the workspace against a
    free-text question', which is a materially different (and unbuilt) ranking
    problem."""
    age_days = max((now - claim.source_timestamp).days, 0)
    recency = 1.0 / (1.0 + age_days / 30.0)
    adjudication = _ADJUDICATION_WEIGHT.get(claim.adjudication_status, 0.5)
    return round(0.5 * claim.confidence + 0.3 * recency + 0.2 * adjudication, 4)


def _explain(claim: Claim, score: float) -> str:
    return f"confidence={claim.confidence:.2f}, adjudication={claim.adjudication_status.value}, score={score:.2f}"


class ContextGraphBuilder:
    def __init__(self, claim_repo: ClaimRepository):
        self._claim_repo = claim_repo

    async def build(
        self,
        scope: ContextGraphScope,
        *,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        predicate_diversity_cap: int = DEFAULT_PREDICATE_DIVERSITY_CAP,
        now: datetime | None = None,
    ) -> ContextGraphResult:
        now = now or datetime.now(timezone.utc)

        if scope.conversation_id:
            candidates = await self._claim_repo.list_claims_for_conversation(
                scope.workspace_id, scope.conversation_id
            )
        elif scope.subject_id:
            candidates = await self._claim_repo.list_claims_by_subject(scope.workspace_id, scope.subject_id)
        else:
            candidates = []

        scored = sorted(
            ((c, _score_claim(c, now=now)) for c in candidates),
            key=lambda pair: pair[1],
            reverse=True,
        )

        selected: list[tuple[Claim, float]] = []
        tokens_used = 0
        predicate_counts: dict[str, int] = {}
        truncated = False
        for claim, score in scored:
            if len(selected) >= max_nodes:
                truncated = True
                break
            claim_tokens = _claim_tokens(claim)
            if tokens_used + claim_tokens > max_tokens:
                truncated = True
                continue
            if predicate_counts.get(claim.predicate, 0) >= predicate_diversity_cap:
                continue
            selected.append((claim, score))
            tokens_used += claim_tokens
            predicate_counts[claim.predicate] = predicate_counts.get(claim.predicate, 0) + 1

        evidence = [
            EvidenceReference(
                claim_id=c.claim_id, source_segment_id=c.source_segment_id,
                evidence_char_start=c.evidence_char_start, evidence_char_end=c.evidence_char_end,
                excerpt=f"{c.predicate}:{c.object_value or c.object_id or ''}",
            )
            for c, _ in selected
        ]
        selected_items = [SelectedItem(claim_id=c.claim_id, score=s, reason=_explain(c, s)) for c, s in selected]

        return ContextGraphResult(
            workspace_id=scope.workspace_id,
            claims=[c for c, _ in selected],
            evidence=evidence,
            unresolved_mention_ids=[],
            conflicts=[],
            selected_items=selected_items,
            budget_max_nodes=max_nodes,
            budget_max_tokens=max_tokens,
            nodes_used=len(selected),
            tokens_used=tokens_used,
            truncated=truncated,
        )
