"""Claim-to-evidence provenance graph for evaluated query turns."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

from graphrag.core.models import EvalResult, QueryResult


@dataclass(frozen=True)
class ClaimNode:
    id: str
    tenant: str
    query_id: str
    text: str


@dataclass(frozen=True)
class ArtifactNode:
    id: str
    tenant: str
    query_id: str
    artifact_type: str
    external_id: str
    content_digest: str = ""


@dataclass(frozen=True)
class ActionNode:
    id: str
    tenant: str
    query_id: str
    action_type: str
    retrieval_mode: str
    correlation_id: str = ""
    source_trace_id: str = ""


@dataclass(frozen=True)
class CheckNode:
    id: str
    tenant: str
    query_id: str
    check_type: str
    status: str
    score: float | None = None
    version: str = ""
    reason: str = ""


@dataclass
class ClaimEvidenceGraph:
    claims: list[ClaimNode] = field(default_factory=list)
    artifacts: list[ArtifactNode] = field(default_factory=list)
    actions: list[ActionNode] = field(default_factory=list)
    checks: list[CheckNode] = field(default_factory=list)
    supported_by: list[tuple[str, str]] = field(default_factory=list)
    produced_by: list[tuple[str, str]] = field(default_factory=list)
    validated_by: list[tuple[str, str]] = field(default_factory=list)


def build_claim_evidence_graph(
    result: QueryResult, evaluation: EvalResult, *, tenant: str,
) -> ClaimEvidenceGraph:
    """Build a provenance association graph without claiming entailment per sentence.

    ``SUPPORTED_BY`` means the artifact was part of the retrieved evidence set;
    the aggregate evaluation check carries the stronger validated/unvalidated
    status. This distinction prevents a global RAGAS score from being
    misrepresented as sentence-level proof.
    """
    graph = ClaimEvidenceGraph()
    query_id = result.query_id
    action = ActionNode(
        id=f"action:{query_id}", tenant=tenant, query_id=query_id,
        action_type="query_retrieval_and_synthesis", retrieval_mode=result.retrieval_mode,
        correlation_id=result.correlation_id, source_trace_id=result.source_trace_id,
    )
    graph.actions.append(action)

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", result.answer or "") if part.strip()]
    for index, sentence in enumerate(sentences):
        claim = ClaimNode(
            id=f"claim:{query_id}:{index}", tenant=tenant, query_id=query_id, text=sentence,
        )
        graph.claims.append(claim)
        graph.produced_by.append((claim.id, action.id))

    artifacts: list[ArtifactNode] = []
    for citation in dict.fromkeys(str(item) for item in result.citations if str(item).strip()):
        artifacts.append(ArtifactNode(
            id=f"artifact:{query_id}:document:{_safe_id(citation)}",
            tenant=tenant, query_id=query_id, artifact_type="document",
            external_id=citation,
        ))
    for index, context in enumerate(result.contexts):
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
        artifacts.append(ArtifactNode(
            id=f"artifact:{query_id}:context:{digest[:16]}",
            tenant=tenant, query_id=query_id, artifact_type="retrieved_context",
            external_id=f"context-{index}", content_digest=digest,
        ))
    graph.artifacts.extend(artifacts)
    for claim in graph.claims:
        graph.supported_by.extend((claim.id, artifact.id) for artifact in artifacts)

    faithfulness = float(evaluation.faithfulness)
    score = faithfulness if math.isfinite(faithfulness) else None
    check = CheckNode(
        id=f"check:{query_id}:faithfulness", tenant=tenant, query_id=query_id,
        check_type="ragas_faithfulness", status=("passed" if score is not None and score >= 0.8 else "unscorable"),
        score=score,
    )
    graph.checks.append(check)
    graph.validated_by.extend((claim.id, check.id) for claim in graph.claims)
    judge_check = CheckNode(
        id=f"check:{query_id}:judge", tenant=tenant, query_id=query_id,
        check_type="judge_retrieve_abstain", status=evaluation.judge_decision,
        score=(evaluation.judge_confidence if math.isfinite(evaluation.judge_confidence) else None),
    )
    graph.checks.append(judge_check)
    if evaluation.judge_decision == "accept":
        graph.validated_by.extend((claim.id, judge_check.id) for claim in graph.claims)
    for rubric in evaluation.rubric_results:
        rubric_id = str(rubric.get("rubric_id", "unknown"))
        rubric_check = CheckNode(
            id=f"check:{query_id}:rubric:{_safe_id(rubric_id)}", tenant=tenant, query_id=query_id,
            check_type=rubric_id, status="passed" if rubric.get("passed") else "failed",
            score=float(rubric.get("score", 0.0)), version=str(rubric.get("version", "")),
            reason=str(rubric.get("reason", "")),
        )
        graph.checks.append(rubric_check)
        graph.validated_by.extend((claim.id, rubric_check.id) for claim in graph.claims)
    return graph


async def persist_claim_evidence_graph(neo4j: Any, graph: ClaimEvidenceGraph) -> None:
    """Persist the graph with tenant-scoped MERGE operations."""
    tenant = _tenant(graph)
    await neo4j.run(
        """
        UNWIND $items AS item
        MERGE (n:Claim {tenant: item.tenant, id: item.id})
        SET n.query_id = item.query_id, n.text = item.text
        """,
        items=[vars(item) for item in graph.claims],
    )
    await neo4j.run(
        """
        UNWIND $items AS item
        MERGE (n:Artifact {tenant: item.tenant, id: item.id})
        SET n.query_id = item.query_id, n.artifact_type = item.artifact_type,
            n.external_id = item.external_id, n.content_digest = item.content_digest
        """,
        items=[vars(item) for item in graph.artifacts],
    )
    await neo4j.run(
        """
        UNWIND $items AS item
        MERGE (n:Action {tenant: item.tenant, id: item.id})
        SET n.query_id = item.query_id, n.action_type = item.action_type,
            n.retrieval_mode = item.retrieval_mode, n.correlation_id = item.correlation_id,
            n.source_trace_id = item.source_trace_id
        """,
        items=[vars(item) for item in graph.actions],
    )
    await neo4j.run(
        """
        UNWIND $items AS item
        MERGE (n:Check {tenant: item.tenant, id: item.id})
        SET n.query_id = item.query_id, n.check_type = item.check_type,
            n.status = item.status, n.score = item.score, n.version = item.version,
            n.reason = item.reason
        """,
        items=[vars(item) for item in graph.checks],
    )
    await neo4j.run(
        """
        UNWIND $rels AS rel
        MATCH (c:Claim {tenant: $tenant, id: rel[0]})
        MATCH (a:Artifact {tenant: $tenant, id: rel[1]})
        MERGE (c)-[:SUPPORTED_BY]->(a)
        """,
        tenant=tenant, rels=[list(item) for item in graph.supported_by],
    )
    await neo4j.run(
        """
        UNWIND $rels AS rel
        MATCH (c:Claim {tenant: $tenant, id: rel[0]})
        MATCH (a:Action {tenant: $tenant, id: rel[1]})
        MERGE (c)-[:PRODUCED_BY]->(a)
        """,
        tenant=tenant, rels=[list(item) for item in graph.produced_by],
    )
    await neo4j.run(
        """
        UNWIND $rels AS rel
        MATCH (c:Claim {tenant: $tenant, id: rel[0]})
        MATCH (k:Check {tenant: $tenant, id: rel[1]})
        MERGE (c)-[:VALIDATED_BY]->(k)
        """,
        tenant=tenant, rels=[list(item) for item in graph.validated_by],
    )


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)[:120]


def _tenant(graph: ClaimEvidenceGraph) -> str:
    for collection in (graph.claims, graph.artifacts, graph.actions, graph.checks):
        if collection:
            return collection[0].tenant
    raise ValueError("claim evidence graph cannot be persisted without a tenant")


__all__ = ["ClaimEvidenceGraph", "build_claim_evidence_graph", "persist_claim_evidence_graph"]
