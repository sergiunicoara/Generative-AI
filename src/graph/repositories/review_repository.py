"""Tenant-safe repository for Mention, ResolutionDecision, and ReviewDecision —
same pattern as crm_repository.py. ResolutionDecision/ReviewDecision are routed
through Mention (never a bare fan-out), same §10 principle as everything else.
"""

from __future__ import annotations

import json

from src.domain.assertion import ResolutionDecision, ReviewDecision
from src.domain.conversation import Mention
from src.graph.execution import GraphExecutor, scoped_match

_MENTION_RETURN = (
    "m.mention_id AS mention_id, m.workspace_id AS workspace_id, m.segment_id AS segment_id, "
    "m.char_start AS char_start, m.char_end AS char_end, m.surface_text AS surface_text, "
    "m.normalized_surface AS normalized_surface, m.entity_type AS entity_type, "
    "m.resolved_entity_id AS resolved_entity_id, m.resolution_status AS resolution_status"
)

_RESOLUTION_DECISION_RETURN = (
    "rd.resolution_decision_id AS resolution_decision_id, rd.workspace_id AS workspace_id, "
    "rd.mention_id AS mention_id, rd.resolved_entity_id AS resolved_entity_id, rd.status AS status, "
    "rd.lexical_score AS lexical_score, rd.semantic_score AS semantic_score, "
    "rd.base_score AS base_score, rd.relational_bonus AS relational_bonus, "
    "rd.final_score AS final_score, rd.margin AS margin, "
    "rd.relational_signals AS relational_signals, rd.decided_at AS decided_at"
)

_REVIEW_DECISION_RETURN = (
    "rvd.review_decision_id AS review_decision_id, rvd.workspace_id AS workspace_id, "
    "rvd.mention_id AS mention_id, rvd.reviewer_id AS reviewer_id, rvd.decided_at AS decided_at, "
    "rvd.selected_entity_id AS selected_entity_id, rvd.rejected AS rejected, "
    "rvd.candidates_shown AS candidates_shown, rvd.original_scores AS original_scores, "
    "rvd.reason AS reason, rvd.affected_claim_ids AS affected_claim_ids, "
    "rvd.previous_review_decision_id AS previous_review_decision_id"
)


class ReviewRepository:
    def __init__(self, executor: GraphExecutor | None = None):
        self._executor = executor or GraphExecutor()

    async def upsert_mention(self, mention: Mention) -> None:
        match = scoped_match("Mention", "m", mention_id="mention_id")
        await self._executor.tenant_query(
            f"""
            MERGE {match}
            ON CREATE SET m.created_at = datetime()
            SET m.segment_id = $segment_id,
                m.char_start = $char_start,
                m.char_end = $char_end,
                m.surface_text = $surface_text,
                m.normalized_surface = $normalized_surface,
                m.entity_type = $entity_type,
                m.resolved_entity_id = $resolved_entity_id,
                m.resolution_status = $resolution_status
            """,
            workspace_id=mention.workspace_id,
            mention_id=mention.mention_id,
            segment_id=mention.segment_id,
            char_start=mention.char_start,
            char_end=mention.char_end,
            surface_text=mention.surface_text,
            normalized_surface=mention.normalized_surface,
            entity_type=mention.entity_type,
            resolved_entity_id=mention.resolved_entity_id,
            resolution_status=mention.resolution_status.value,
        )

    async def get_mention(self, workspace_id: str, mention_id: str) -> Mention | None:
        match = scoped_match("Mention", "m", mention_id="mention_id")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_MENTION_RETURN}",
            workspace_id=workspace_id,
            mention_id=mention_id,
        )
        return Mention(**rows[0]) if rows else None

    async def list_mentions_by_status(self, workspace_id: str, resolution_status: str) -> list[Mention]:
        match = scoped_match("Mention", "m", resolution_status="resolution_status")
        rows = await self._executor.tenant_query(
            f"MATCH {match} RETURN {_MENTION_RETURN}",
            workspace_id=workspace_id,
            resolution_status=resolution_status,
        )
        return [Mention(**row) for row in rows]

    async def upsert_resolution_decision(self, decision: ResolutionDecision) -> None:
        mention_match = scoped_match("Mention", "m", mention_id="mention_id")
        await self._executor.tenant_query(
            f"""
            MATCH {mention_match}
            MERGE (rd:ResolutionDecision {{resolution_decision_id: $resolution_decision_id}})
            SET rd.workspace_id = $workspace_id,
                rd.mention_id = $mention_id,
                rd.resolved_entity_id = $resolved_entity_id,
                rd.status = $status,
                rd.lexical_score = $lexical_score,
                rd.semantic_score = $semantic_score,
                rd.base_score = $base_score,
                rd.relational_bonus = $relational_bonus,
                rd.final_score = $final_score,
                rd.margin = $margin,
                rd.relational_signals = $relational_signals,
                rd.decided_at = $decided_at
            MERGE (m)-[:HAS_RESOLUTION_DECISION]->(rd)
            """,
            workspace_id=decision.workspace_id,
            mention_id=decision.mention_id,
            resolution_decision_id=decision.resolution_decision_id,
            resolved_entity_id=decision.resolved_entity_id,
            status=decision.status.value,
            lexical_score=decision.lexical_score,
            semantic_score=decision.semantic_score,
            base_score=decision.base_score,
            relational_bonus=decision.relational_bonus,
            final_score=decision.final_score,
            margin=decision.margin,
            relational_signals=decision.relational_signals,
            decided_at=decision.decided_at.isoformat(),
        )

    async def get_resolution_decision(
        self, workspace_id: str, resolution_decision_id: str
    ) -> ResolutionDecision | None:
        mention_match = scoped_match("Mention", "m")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {mention_match}
            MATCH (m)-[:HAS_RESOLUTION_DECISION]->(rd:ResolutionDecision {{resolution_decision_id: $resolution_decision_id}})
            RETURN {_RESOLUTION_DECISION_RETURN}
            """,
            workspace_id=workspace_id,
            resolution_decision_id=resolution_decision_id,
        )
        return ResolutionDecision(**rows[0]) if rows else None

    async def create_review_decision(self, decision: ReviewDecision) -> None:
        mention_match = scoped_match("Mention", "m", mention_id="mention_id")
        await self._executor.tenant_query(
            f"""
            MATCH {mention_match}
            MERGE (rvd:ReviewDecision {{review_decision_id: $review_decision_id}})
            SET rvd.workspace_id = $workspace_id,
                rvd.mention_id = $mention_id,
                rvd.reviewer_id = $reviewer_id,
                rvd.decided_at = $decided_at,
                rvd.selected_entity_id = $selected_entity_id,
                rvd.rejected = $rejected,
                rvd.candidates_shown = $candidates_shown,
                rvd.original_scores = $original_scores,
                rvd.reason = $reason,
                rvd.affected_claim_ids = $affected_claim_ids,
                rvd.previous_review_decision_id = $previous_review_decision_id
            MERGE (m)-[:HAS_REVIEW_DECISION]->(rvd)
            """,
            workspace_id=decision.workspace_id,
            mention_id=decision.mention_id,
            review_decision_id=decision.review_decision_id,
            reviewer_id=decision.reviewer_id,
            decided_at=decision.decided_at.isoformat(),
            selected_entity_id=decision.selected_entity_id,
            rejected=decision.rejected,
            candidates_shown=decision.candidates_shown,
            original_scores=json.dumps(decision.original_scores),
            reason=decision.reason,
            affected_claim_ids=decision.affected_claim_ids,
            previous_review_decision_id=decision.previous_review_decision_id,
        )

    async def get_review_decision(self, workspace_id: str, review_decision_id: str) -> ReviewDecision | None:
        mention_match = scoped_match("Mention", "m")
        rows = await self._executor.tenant_query(
            f"""
            MATCH {mention_match}
            MATCH (m)-[:HAS_REVIEW_DECISION]->(rvd:ReviewDecision {{review_decision_id: $review_decision_id}})
            RETURN {_REVIEW_DECISION_RETURN}
            """,
            workspace_id=workspace_id,
            review_decision_id=review_decision_id,
        )
        if not rows:
            return None
        row = dict(rows[0])
        row["original_scores"] = json.loads(row["original_scores"]) if row.get("original_scores") else {}
        return ReviewDecision(**row)
