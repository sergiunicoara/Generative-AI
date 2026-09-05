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
7. conflict preservation            -> Increment 11: detect_conflicting_claims()
                                        runs over the already-selected Claims
                                        (no extra repository fetch — every
                                        candidate is already in memory from
                                        step 5), and any detected Conflicts are
                                        both returned and persisted via
                                        ConflictRepository for later querying
                                        independent of a specific build() call.

Every call is a single bounded repository fetch — no per-Claim follow-up query,
so this never becomes N+1 (§12: 'Avoid N+1 repository calls'). Conflict
detection is a pure in-memory scan over Claims already fetched, so it doesn't
add one either — only the persistence of newly-detected Conflicts is extra
I/O, bounded by the (typically small) number of actual contradictions found.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from src.context_graph.models import ContextGraphResult, EvidenceReference, SelectedItem
from src.context_graph.reranker import rerank
from src.core.config import get_settings
from src.core.telemetry import (
    CONTEXT_GRAPH_BUILD_DURATION_SECONDS,
    CONTEXT_GRAPH_RESULT_COUNT,
    CONTEXT_GRAPH_TRUNCATED_TOTAL,
)
from src.diagnostics.invariants import InvariantCheck, run_invariants
from src.domain.assertion import Claim
from src.domain.enums import AdjudicationStatus
from src.graph.repositories.claim_repository import ClaimRepository
from src.graph.repositories.conflict_repository import ConflictRepository
from src.graph.repositories.source_repository import SourceRepository
from src.resolution.conflict_detection import detect_conflicting_claims
from src.summarization.call_summary import CallSummaryUseCase

DEFAULT_MAX_NODES = 50
DEFAULT_MAX_TOKENS = 4000
DEFAULT_PREDICATE_DIVERSITY_CAP = 5

_ADJUDICATION_WEIGHT = {
    AdjudicationStatus.ACCEPTED: 1.0,
    AdjudicationStatus.UNREVIEWED: 0.7,
    AdjudicationStatus.DISPUTED: 0.4,
    AdjudicationStatus.REJECTED: 0.0,
}

# Human-tunable starting points, not load-bearing: source_system isn't baked
# into any stored score, so these can be retuned later with no backfill.
# salesforce (structured CRM fields) ranks above gong (ASR + LLM extraction
# noise on a call transcript); an unknown/not-yet-set source_system gets the
# neutral default rather than being penalized or favored.
_SOURCE_AUTHORITY_WEIGHT: dict[str, float] = {
    "salesforce": 1.0,
    "gong": 0.7,
}
_DEFAULT_SOURCE_AUTHORITY_WEIGHT = 0.5

# How much a reranked relevance score weighs against the base
# confidence/recency/adjudication/authority score once the reranker fires.
# Also a tunable starting point.
_RERANK_RELEVANCE_WEIGHT = 0.6


def _normalize_relevance(raw_logit: float) -> float:
    """The cross-encoder returns an unbounded raw logit (see reranker.py) --
    sigmoid is the natural inverse-link to bring it into [0, 1] so it can be
    blended with the already-[0, 1] base score below."""
    return 1.0 / (1.0 + math.exp(-raw_logit))


def _blend_relevance(base_score: float, raw_relevance: float) -> float:
    """Blend, not replace. Previously the reranked score replaced the base
    confidence/recency/adjudication/authority score outright -- letting a
    highly-relevant-but-unadjudicated Claim beat a fully-reviewed one purely
    because the reranker path was taken. Blending keeps both signals live."""
    return round(
        _RERANK_RELEVANCE_WEIGHT * _normalize_relevance(raw_relevance)
        + (1 - _RERANK_RELEVANCE_WEIGHT) * base_score,
        4,
    )


@dataclass(frozen=True)
class ContextGraphScope:
    workspace_id: str
    conversation_id: str | None = None
    subject_id: str | None = None
    # Phase 7 (docs/evaluation.md's B5 item): optional free-text question to
    # rerank the scoped Claims against. Absent (default) means no reranking
    # regardless of reranker_enabled -- _score_claim's own docstring already
    # states relevance-to-a-question is "a materially different (and
    # unbuilt) ranking problem" from what it computes; this field is what
    # makes that ranking problem answerable when a caller actually has a
    # question to rank against.
    query_text: str | None = None


def _claim_tokens(claim: Claim) -> int:
    """A rough word-count proxy, not a real tokenizer — matches the same
    documented simplification src/extraction/windowing.py already uses for
    token budgeting, for the same reason (no tokenizer is pinned yet)."""
    text = f"{claim.subject_id} {claim.predicate} {claim.object_value or claim.object_id or ''}"
    return len(text.split())


def _score_claim(claim: Claim, *, now: datetime) -> float:
    """confidence, recency, adjudication_status, and source authority are
    genuinely computed. 'Relevance' (ranking against a free-text question)
    still collapses into the scope filter itself here — this builder answers
    'what's the well-evidenced context for this specific conversation/
    subject', not 'rank all Claims in the workspace against a free-text
    question'; the latter is handled separately, when scope.query_text is
    set, by the reranker blend in build()."""
    age_days = max((now - claim.source_timestamp).days, 0)
    recency = 1.0 / (1.0 + age_days / 30.0)
    adjudication = _ADJUDICATION_WEIGHT.get(claim.adjudication_status, 0.5)
    authority = (
        _SOURCE_AUTHORITY_WEIGHT.get(claim.source_system, _DEFAULT_SOURCE_AUTHORITY_WEIGHT)
        if claim.source_system is not None
        else _DEFAULT_SOURCE_AUTHORITY_WEIGHT
    )
    return round(0.45 * claim.confidence + 0.20 * recency + 0.15 * adjudication + 0.20 * authority, 4)


def _explain(claim: Claim, score: float, *, relevance: float | None = None) -> str:
    parts = [f"confidence={claim.confidence:.2f}", f"adjudication={claim.adjudication_status.value}"]
    if relevance is not None:
        parts.append(f"relevance={relevance:.2f}")
    parts.append(f"score={score:.2f}")
    return ", ".join(parts)


def _claim_rerank_text(claim: Claim) -> str:
    """The cross-encoder's "passage" side of the (query_text, passage) pair
    -- predicate plus whatever object the Claim actually carries (exactly
    one of object_value/object_id is ever set, same as _claim_tokens)."""
    return f"{claim.predicate}: {claim.object_value or claim.object_id or ''}"


class ContextGraphBuilder:
    def __init__(
        self,
        claim_repo: ClaimRepository,
        conflict_repo: ConflictRepository | None = None,
        call_summary_usecase: CallSummaryUseCase | None = None,
        source_repo: SourceRepository | None = None,
    ):
        self._claim_repo = claim_repo
        self._conflict_repo = conflict_repo or ConflictRepository()
        # Phase 3 dual-layer retrieval, optional: None (the default) means
        # every existing single-arg ContextGraphBuilder(claim_repo) call
        # site keeps working unchanged -- attaching a summary needs an LLM
        # chat_fn this builder otherwise has no reason to require.
        self._call_summary_usecase = call_summary_usecase
        # Same opt-in shape as call_summary_usecase above: None (the default)
        # skips the source-traceability invariant check entirely, so every
        # existing call site keeps working unchanged.
        self._source_repo = source_repo

    async def build(
        self,
        scope: ContextGraphScope,
        *,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        predicate_diversity_cap: int = DEFAULT_PREDICATE_DIVERSITY_CAP,
        now: datetime | None = None,
        include_summary: bool = False,
    ) -> ContextGraphResult:
        now = now or datetime.now(timezone.utc)
        started_at = time.monotonic()

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

        # Phase 7 reranker (docs/evaluation.md's B5 item): off unless both
        # reranker_enabled and scope.query_text are set -- a caller with no
        # question to rank against gets exactly the pre-Phase-7 confidence/
        # recency/adjudication/authority ordering, unchanged. Reordering
        # happens on the already-fully-in-memory `scored` list, no extra DB
        # fetch. The relevance score is BLENDED with (not a replacement for)
        # the base score -- see _blend_relevance's docstring for why a
        # straight replacement was a bug, not a design choice.
        relevance_by_claim_id: dict[str, float] = {}
        if get_settings().reranker_enabled and scope.query_text and scored:
            claims_in_order = [c for c, _ in scored]
            base_score_by_claim_id = {c.claim_id: s for c, s in scored}
            relevance_scores = await rerank(
                scope.query_text, [_claim_rerank_text(c) for c in claims_in_order]
            )
            relevance_by_claim_id = {
                c.claim_id: r for c, r in zip(claims_in_order, relevance_scores, strict=True)
            }
            scored = sorted(
                (
                    (c, _blend_relevance(base_score_by_claim_id[c.claim_id], r))
                    for c, r in zip(claims_in_order, relevance_scores, strict=True)
                ),
                key=lambda pair: pair[1],
                reverse=True,
            )

        selected: list[tuple[Claim, float]] = []
        tokens_used = 0
        predicate_counts: dict[str, int] = {}
        truncated = False
        truncated_reason: str | None = None  # first cap hit wins, for the metric label below
        for claim, score in scored:
            if len(selected) >= max_nodes:
                truncated = True
                truncated_reason = truncated_reason or "max_nodes"
                break
            claim_tokens = _claim_tokens(claim)
            if tokens_used + claim_tokens > max_tokens:
                truncated = True
                truncated_reason = truncated_reason or "max_tokens"
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
        selected_items = [
            SelectedItem(claim_id=c.claim_id, score=s, reason=_explain(c, s, relevance=relevance_by_claim_id.get(c.claim_id)))
            for c, s in selected
        ]

        conflicts = detect_conflicting_claims([c for c, _ in selected], now=now)
        for conflict in conflicts:
            await self._conflict_repo.create_conflict(conflict)

        CONTEXT_GRAPH_BUILD_DURATION_SECONDS.observe(time.monotonic() - started_at)
        CONTEXT_GRAPH_RESULT_COUNT.observe(len(selected))
        if truncated_reason is not None:
            CONTEXT_GRAPH_TRUNCATED_TOTAL.labels(reason=truncated_reason).inc()

        # Phase 3 dual-layer retrieval: additive to the Claims above, never
        # a replacement for them -- a failed/unavailable summary (no
        # call_summary_usecase wired in, no conversation_id in scope, no
        # citable Claims, a rejected hallucinated citation) degrades to
        # summary=None, not a failed build.
        summary = None
        if include_summary and self._call_summary_usecase is not None and scope.conversation_id:
            summary = await self._call_summary_usecase.get_or_generate(
                scope.workspace_id, scope.conversation_id
            )

        if self._source_repo is not None:
            await self._check_source_traceability(scope.workspace_id, [c for c, _ in selected])

        return ContextGraphResult(
            workspace_id=scope.workspace_id,
            claims=[c for c, _ in selected],
            evidence=evidence,
            unresolved_mention_ids=[],
            conflicts=conflicts,
            selected_items=selected_items,
            budget_max_nodes=max_nodes,
            budget_max_tokens=max_tokens,
            nodes_used=len(selected),
            tokens_used=tokens_used,
            truncated=truncated,
            summary=summary,
        )

    async def _check_source_traceability(self, workspace_id: str, claims: list[Claim]) -> None:
        """Invariant: every selected Claim that carries a source_record_id
        must resolve to a retrievable SourceRecord. Claims with no
        source_record_id are excluded from the check (not treated as a
        violation) -- not every Claim's provenance is a SourceRecord today.
        Opt-in via source_repo (constructor) so this never fires unless a
        caller explicitly wants it."""
        if self._source_repo is None:
            return
        source_record_ids = {c.source_record_id for c in claims if c.source_record_id}
        if not source_record_ids:
            return
        found = await self._source_repo.get_source_records(workspace_id, list(source_record_ids))
        run_invariants("context_graph.build", [
            InvariantCheck(
                name="cited_claims_trace_to_source_record",
                check_fn=lambda: source_record_ids.issubset(found.keys()),
                detail=f"missing={sorted(source_record_ids - found.keys())}",
            ),
        ])
