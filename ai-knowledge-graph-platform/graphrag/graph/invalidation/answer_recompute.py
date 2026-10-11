"""Opt-in automatic re-answering of invalidated query decisions (docs/invalidation.md).

Off by default (``invalidation.reanswer_decisions: false``): re-answering calls the retrieval
pipeline and therefore an LLM, which the deterministic recompute path deliberately never does.

Scope and guards:

* Only decisions recorded for a plain retrieval query (``CGCase.case_type = retrieval_query``,
  question in the case description). Any other decision stays with human review.
* Refused while access control is enabled. The original caller's document permissions are not
  stored with the decision, so re-running the question tenant-wide could surface evidence the
  original caller could not see. The decision then goes back to human review.
* The recorded decision is never rewritten (its trace hash covers it). The new answer is stored
  next to it as ``reanswered_*`` properties, so history and the replacement are both kept.
* An answer that comes back without sufficient evidence ends INSUFFICIENT_EVIDENCE, not VALID.
"""
from __future__ import annotations

import structlog

from graphrag.graph.invalidation.models import ArtifactState
from graphrag.graph.validation.graph_checks import run_read_only

log = structlog.get_logger(__name__)

_QUESTION = """
MATCH (dec:CGDecision {tenant: $tenant, id: $id})<-[:PRODUCED_DECISION]-(run:CGAgentRun {tenant: $tenant})
      -[:ADDRESSES]->(c:CGCase {tenant: $tenant})
RETURN c.case_type AS case_type, c.description AS question
LIMIT 1
"""

_STORE = """
MATCH (dec:CGDecision {tenant: $tenant, id: $id})
SET dec.reanswered_answer = $answer,
    dec.reanswered_confidence = $confidence,
    dec.reanswered_citations = $citations,
    dec.reanswered_at = datetime()
"""


def enabled() -> bool:
    from graphrag.core.config import get_settings

    return bool((get_settings().invalidation or {}).get("reanswer_decisions", False))


def default_answer_recomputer(neo4j_client, *, retriever_factory=None):
    """The recomputer to hand to ``RecomputeWorker``, or None when the feature is off."""
    if not enabled():
        return None

    async def recompute(tenant: str, decision_id: str) -> ArtifactState:
        from graphrag.core.config import get_settings

        if get_settings().access_control.get("enabled", False):
            log.info("answer_recompute.skipped", reason="access_control_enabled", decision=decision_id)
            return ArtifactState.NEEDS_REVIEW
        rows = await run_read_only(neo4j_client, _QUESTION, tenant=tenant, id=decision_id)
        if not rows or rows[0].get("case_type") != "retrieval_query" or not rows[0].get("question"):
            return ArtifactState.NEEDS_REVIEW
        if retriever_factory is not None:
            retriever = retriever_factory()
        else:
            from graphrag.retrieval.hybrid_retriever import HybridRetriever
            retriever = HybridRetriever()
        result = await retriever.retrieve_and_answer(rows[0]["question"], tenant=tenant)
        sufficiency = result.retrieval_sufficiency or {}
        if not result.citations or not sufficiency.get("sufficient", True):
            return ArtifactState.INSUFFICIENT_EVIDENCE
        await neo4j_client.run(
            _STORE, tenant=tenant, id=decision_id, answer=result.answer,
            confidence=result.confidence, citations=list(result.citations),
        )
        return ArtifactState.VALID

    return recompute
