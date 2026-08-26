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
        })

    unique: dict[tuple[str, str, str, str], dict] = {}
    for proposal in proposals:
        identity = (
            proposal["kind"], proposal["proposed_value"],
            proposal["source_type"], proposal["target_type"],
        )
        unique.setdefault(identity, proposal)
    return list(unique.values())[:max(0, limit)]


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
                   p.seen_count AS seen_count, p.created_at AS created_at,
                   p.last_seen_at AS last_seen_at, p.reviewed_by AS reviewed_by,
                   p.reviewed_at AS reviewed_at
            ORDER BY p.last_seen_at DESC
            LIMIT $limit
            """,
            tenant=tenant,
            status=status,
            limit=limit,
        )

    async def decide(self, proposal_id: str, *, approve: bool, reviewed_by: str, tenant: str) -> dict:
        """Record a human decision without mutating the active ontology."""
        status = "approved" if approve else "rejected"
        rows = await self._neo4j.run(
            """
            MATCH (p:OntologyProposal {id: $proposal_id, tenant: $tenant, status: 'pending'})
            SET p.status = $status, p.reviewed_by = $reviewed_by, p.reviewed_at = datetime()
            RETURN p.id AS id, p.kind AS kind, p.proposed_value AS proposed_value, p.status AS status
            """,
            proposal_id=proposal_id,
            tenant=tenant,
            status=status,
            reviewed_by=reviewed_by,
        )
        if not rows:
            return {"error": f"Proposal {proposal_id} not found or already resolved"}
        result = dict(rows[0])
        log.info("ontology_proposal.decided", proposal_id=proposal_id, tenant=tenant, status=status)
        return result
