"""Governed review queue for ontology drift discovered during ingestion.

An LLM may suggest a new entity type or relation predicate, but that must not
silently extend the active ontology.  This module stores a deduplicated,
tenant-scoped proposal with source document/chunk provenance.  Approval is an
auditable governance decision; a separate versioned ontology migration remains
responsible for changing the active schema.
"""

from __future__ import annotations

import hashlib
import json
from uuid import uuid4

import structlog

from graphrag.core.models import Chunk, Entity, Relation

log = structlog.get_logger(__name__)


def build_ontology_proposals(
    report: dict,
    entities: list[Entity],
    relations: list[Relation],
    chunk: Chunk,
    *,
    limit: int = 8,
) -> list[dict]:
    """Turn rejected extraction output into bounded, source-grounded proposals."""
    rejected_entities = set(report.get("rejected_entity_ids", []))
    rejected_relations = set(report.get("rejected_relation_ids", []))
    new_relations = set(report.get("new_relations", []))
    proposals: list[dict] = []

    for entity in entities:
        if entity.id not in rejected_entities:
            continue
        proposals.append({
            "kind": "entity_type",
            "proposed_value": entity.type.strip().upper()[:100],
            "entity_name": entity.name[:500],
            "source_type": "",
            "target_type": "",
            "reason": "unknown_entity_type",
            "confidence": entity.confidence,
        })

    entities_by_id = {entity.id: entity for entity in entities}
    for relation in relations:
        if relation.id not in rejected_relations:
            continue
        source = entities_by_id.get(relation.source_entity_id)
        target = entities_by_id.get(relation.target_entity_id)
        source_type = source.type if source else ""
        target_type = target.type if target else ""
        if relation.relation in new_relations:
            kind = "relation"
            reason = "unknown_relation"
        else:
            kind = "relation_pair"
            reason = "invalid_domain_range"
        proposals.append({
            "kind": kind,
            "proposed_value": relation.relation.strip().upper()[:100],
            "entity_name": "",
            "source_type": source_type[:100],
            "target_type": target_type[:100],
            "reason": reason,
            "confidence": relation.confidence,
        })

    unique: dict[tuple[str, str, str, str], dict] = {}
    for proposal in proposals:
        identity = (
            proposal["kind"], proposal["proposed_value"],
            proposal["source_type"], proposal["target_type"],
        )
        unique.setdefault(identity, proposal)
    deduped = list(unique.values())[:max(0, limit)]

    # A proposed name is flagged as conflicting with itself when the batch
    # proposes the same value under more than one kind (e.g. "VENDOR" seen
    # both as an entity_type and inside a relation_pair) -- a real,
    # cheaply-computed collision, not a heuristic guess.
    value_kinds: dict[str, set[str]] = {}
    for proposal in deduped:
        value_kinds.setdefault(proposal["proposed_value"], set()).add(proposal["kind"])
    for proposal in deduped:
        siblings = value_kinds[proposal["proposed_value"]] - {proposal["kind"]}
        proposal["conflicts"] = sorted(siblings)

    return deduped


class OntologyProposalService:
    """Persist and resolve human-governed ontology-change proposals."""

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    @staticmethod
    def _fingerprint(tenant: str, proposal: dict) -> str:
        payload = {
            "tenant": tenant,
            "kind": proposal["kind"],
            "proposed_value": proposal["proposed_value"],
            "source_type": proposal.get("source_type", ""),
            "target_type": proposal.get("target_type", ""),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    async def submit(
        self,
        proposals: list[dict],
        chunk: Chunk,
        *,
        ontology_version_id: str = "",
    ) -> list[str]:
        """Upsert proposals after the source chunk has been written to Neo4j."""
        proposal_ids: list[str] = []
        for proposal in proposals:
            fingerprint = self._fingerprint(chunk.tenant, proposal)
            rows = await self._neo4j.run(
                """
                MERGE (p:OntologyProposal {tenant: $tenant, fingerprint: $fingerprint})
                ON CREATE SET p.id = $id,
                              p.kind = $kind,
                              p.proposed_value = $proposed_value,
                              p.source_type = $source_type,
                              p.target_type = $target_type,
                              p.entity_name = $entity_name,
                              p.reason = $reason,
                              p.confidence = $confidence,
                              p.conflicts = $conflicts,
                              p.affected_use_cases = $affected_use_cases,
                              p.business_impact = $business_impact,
                              p.urgency = $urgency,
                              p.effort = $effort,
                              p.risk = $risk,
                              p.dependencies = $dependencies,
                              p.requesting_team = $requesting_team,
                              p.status = 'pending',
                              p.seen_count = 0,
                              p.created_at = datetime()
                SET p.seen_count = coalesce(p.seen_count, 0) + 1,
                    p.last_seen_at = datetime(),
                    p.last_source_doc_id = $document_id,
                    p.last_source_chunk_id = $chunk_id
                WITH p
                OPTIONAL MATCH (c:Chunk {tenant: $tenant, id: $chunk_id})
                FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END |
                    MERGE (p)-[:EVIDENCED_BY]->(c))
                WITH p
                OPTIONAL MATCH (d:Document {tenant: $tenant, id: $document_id})
                FOREACH (_ IN CASE WHEN d IS NULL THEN [] ELSE [1] END |
                    MERGE (p)-[:ASSERTED_IN]->(d))
                WITH p
                OPTIONAL MATCH (o:OntologyVersion {tenant: $tenant, id: $ontology_version_id})
                FOREACH (_ IN CASE WHEN o IS NULL THEN [] ELSE [1] END |
                    MERGE (p)-[:PROPOSED_FOR]->(o))
                RETURN p.id AS id
                """,
                id=str(uuid4()),
                tenant=chunk.tenant,
                fingerprint=fingerprint,
                kind=proposal["kind"],
                proposed_value=proposal["proposed_value"],
                source_type=proposal.get("source_type", ""),
                target_type=proposal.get("target_type", ""),
                entity_name=proposal.get("entity_name", ""),
                reason=proposal.get("reason", ""),
                confidence=proposal.get("confidence", 1.0),
                conflicts=proposal.get("conflicts", []),
                affected_use_cases=proposal.get("affected_use_cases", []),
                business_impact=proposal.get("business_impact", ""),
                urgency=proposal.get("urgency", ""),
                effort=proposal.get("effort", ""),
                risk=proposal.get("risk", ""),
                dependencies=proposal.get("dependencies", []),
                requesting_team=proposal.get("requesting_team", ""),
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                ontology_version_id=ontology_version_id,
            )
            if rows:
                proposal_ids.append(rows[0]["id"])
        return proposal_ids

    async def list(self, tenant: str, *, status: str = "pending", limit: int = 100) -> list[dict]:
        return await self._neo4j.run(
            """
            MATCH (p:OntologyProposal {tenant: $tenant})
            WHERE $status = '' OR p.status = $status
            RETURN p.id AS id, p.kind AS kind, p.proposed_value AS proposed_value,
                   p.source_type AS source_type, p.target_type AS target_type,
                   p.entity_name AS entity_name, p.reason AS reason, p.status AS status,
                   p.confidence AS confidence, p.conflicts AS conflicts,
                   p.affected_use_cases AS affected_use_cases,
                   p.business_impact AS business_impact, p.urgency AS urgency,
                   p.effort AS effort, p.risk AS risk, p.dependencies AS dependencies,
                   p.requesting_team AS requesting_team,
                   p.seen_count AS seen_count, p.created_at AS created_at,
                   p.last_seen_at AS last_seen_at, p.reviewed_by AS reviewed_by,
                   p.reviewed_at AS reviewed_at, p.decision_reason AS decision_reason,
                   p.decision_model_version AS decision_model_version,
                   p.merge_target AS merge_target
            ORDER BY p.last_seen_at DESC
            LIMIT $limit
            """,
            tenant=tenant,
            status=status,
            limit=limit,
        )

    # Roadmap "P1 -- ontology curation and human-in-the-loop workbench",
    # bullet 2: "approve, edit, reject, merge, defer and quarantine
    # decisions, each with actor, reason, timestamp and model/version
    # evidence." `approve`/`reject` are unchanged for existing callers;
    # `action` is the general form.
    _ACTION_STATUS = {
        "approve": "approved",
        "reject": "rejected",
        "edit": "edited",
        "merge": "merged",
        "defer": "deferred",
        "quarantine": "quarantined",
    }

    async def decide(
        self,
        proposal_id: str,
        *,
        approve: bool | None = None,
        action: str | None = None,
        reviewed_by: str,
        tenant: str,
        reason: str = "",
        model_version: str = "",
        edited_value: str | None = None,
        merge_target: str | None = None,
    ) -> dict:
        """Record a human decision without mutating the active ontology.

        Either `action` (one of `approve`/`reject`/`edit`/`merge`/`defer`/
        `quarantine`) or the legacy `approve` boolean must be given; `action`
        wins if both are passed.
        """
        if action is None:
            if approve is None:
                return {"error": "Either 'action' or 'approve' must be given"}
            action = "approve" if approve else "reject"
        if action not in self._ACTION_STATUS:
            return {"error": f"Unknown action '{action}'"}
        status = self._ACTION_STATUS[action]

        set_clauses = [
            "p.status = $status",
            "p.reviewed_by = $reviewed_by",
            "p.reviewed_at = datetime()",
            "p.decision_reason = $reason",
            "p.decision_model_version = $model_version",
        ]
        params: dict = {
            "proposal_id": proposal_id,
            "tenant": tenant,
            "status": status,
            "reviewed_by": reviewed_by,
            "reason": reason,
            "model_version": model_version,
        }
        if action == "edit" and edited_value:
            set_clauses.append("p.proposed_value = $edited_value")
            params["edited_value"] = edited_value
        if action == "merge" and merge_target:
            set_clauses.append("p.merge_target = $merge_target")
            params["merge_target"] = merge_target

        rows = await self._neo4j.run(
            f"""
            MATCH (p:OntologyProposal {{id: $proposal_id, tenant: $tenant, status: 'pending'}})
            SET {', '.join(set_clauses)}
            RETURN p.id AS id, p.kind AS kind, p.proposed_value AS proposed_value, p.status AS status
            """,
            **params,
        )
        if not rows:
            return {"error": f"Proposal {proposal_id} not found or already resolved"}
        result = dict(rows[0])
        log.info("ontology_proposal.decided", proposal_id=proposal_id, tenant=tenant, status=status, action=action)
        return result

    # Statuses that represent a completed human decision -- the useful
    # signal for golden-set/training export. "pending" is deliberately
    # excluded: an undecided proposal is not yet evidence of anything.
    _DECIDED_STATUSES = ("approved", "rejected", "edited", "merged", "deferred", "quarantined")

    async def export_golden_set(
        self, tenant: str, *, statuses: tuple[str, ...] | None = None, limit: int = 500,
    ) -> list[dict]:
        """Roadmap "P1 -- ontology curation and human-in-the-loop workbench",
        bullet 5: "Export review decisions as golden-set/training data."

        Returns one record per decided proposal, shaped as an (input, label)
        pair plus the human's corrective evidence -- suitable for a
        supervised extraction-triage classifier or a reviewer-agreement
        golden set. Does not attempt to infer a reward signal or ranking;
        that is a modeling decision for whoever consumes the export.
        """
        rows = await self._neo4j.run(
            """
            MATCH (p:OntologyProposal {tenant: $tenant})
            WHERE p.status IN $statuses
            RETURN p.id AS id, p.kind AS kind, p.proposed_value AS proposed_value,
                   p.entity_name AS entity_name, p.source_type AS source_type,
                   p.target_type AS target_type, p.reason AS reason,
                   p.confidence AS confidence, p.status AS status,
                   p.reviewed_by AS reviewed_by, p.reviewed_at AS reviewed_at,
                   p.decision_reason AS decision_reason,
                   p.decision_model_version AS decision_model_version,
                   p.merge_target AS merge_target
            ORDER BY p.reviewed_at DESC
            LIMIT $limit
            """,
            tenant=tenant,
            statuses=list(statuses) if statuses else list(self._DECIDED_STATUSES),
            limit=limit,
        )
        return [
            {
                "id": row["id"],
                "input": {
                    "kind": row["kind"],
                    "proposed_value": row["proposed_value"],
                    "entity_name": row["entity_name"],
                    "source_type": row["source_type"],
                    "target_type": row["target_type"],
                    "extraction_reason": row["reason"],
                    "extraction_confidence": row["confidence"],
                },
                "label": row["status"],
                "corrected_value": row["proposed_value"] if row["status"] == "edited" else None,
                "merge_target": row["merge_target"] if row["status"] == "merged" else None,
                "decision_reason": row["decision_reason"],
                "decision_model_version": row["decision_model_version"],
                "reviewed_by": row["reviewed_by"],
                "reviewed_at": row["reviewed_at"],
            }
            for row in rows
        ]

    async def status_report(self, tenant: str) -> dict:
        """Roadmap bullet 5, second half: "expose status reporting for
        partner teams." A plain count-by-status/kind breakdown -- no
        derived SLA or throughput metric, since this repository has no
        agreed SLA for curator turnaround to measure against.
        """
        rows = await self._neo4j.run(
            """
            MATCH (p:OntologyProposal {tenant: $tenant})
            RETURN p.status AS status, p.kind AS kind, count(p) AS count
            """,
            tenant=tenant,
        )
        by_status: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        total = 0
        for row in rows:
            count = row["count"]
            by_status[row["status"]] = by_status.get(row["status"], 0) + count
            by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + count
            total += count
        return {
            "tenant": tenant,
            "total": total,
            "by_status": by_status,
            "by_kind": by_kind,
            "pending": by_status.get("pending", 0),
        }
