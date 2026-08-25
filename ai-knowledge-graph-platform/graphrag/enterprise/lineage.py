"""Text-backed lineage review and a bitemporal obligations register."""

from __future__ import annotations

from uuid import uuid4

from graphrag.enterprise.models import LineageAssertion, LineageRelation, ObligationDraft
from graphrag.graph.neo4j_client import get_neo4j


class LineageService:
    """Keeps high-stakes lineage and obligations pending until reviewed.

    A relation is never inferred merely from document similarity.  Each request
    has to reference the exact source chunk and quote that states the amendment
    or supersession.  Approval is the only path that materialises the graph edge
    or live obligation.
    """

    def __init__(self, neo4j_client=None):
        self._neo4j = neo4j_client or get_neo4j()

    async def submit_lineage(
        self, document_id: str, assertion: LineageAssertion, tenant: str,
    ) -> dict:
        review_id = str(uuid4())
        rows = await self._neo4j.run(
            """
            MATCH (source:Document {id: $document_id, tenant: $tenant})
            MATCH (target:Document {id: $target_document_id, tenant: $tenant})
            MATCH (chunk:Chunk {id: $evidence_chunk_id, tenant: $tenant})-[:PART_OF]->(source)
            CREATE (review:LineageReview {
                id: $review_id, tenant: $tenant, status: 'pending',
                relation: $relation, source_document_id: $document_id,
                target_document_id: $target_document_id,
                evidence_chunk_id: $evidence_chunk_id, evidence_quote: $evidence_quote,
                confidence: $confidence, effective_from: $effective_from,
                effective_to: $effective_to, created_at: datetime()
            })
            MERGE (review)-[:PROPOSES]->(source)
            MERGE (review)-[:TARGETS]->(target)
            MERGE (review)-[:SUPPORTED_BY]->(chunk)
            RETURN review.id AS review_id, review.status AS status
            """,
            review_id=review_id,
            tenant=tenant,
            document_id=document_id,
            target_document_id=assertion.target_document_id,
            relation=assertion.relation.value,
            evidence_chunk_id=assertion.evidence_chunk_id,
            evidence_quote=assertion.evidence_quote,
            confidence=assertion.confidence,
            effective_from=assertion.effective_from.isoformat() if assertion.effective_from else None,
            effective_to=assertion.effective_to.isoformat() if assertion.effective_to else None,
        )
        if not rows:
            raise ValueError("lineage evidence, source, or target document was not found in this tenant")
        return rows[0]

    async def submit_obligation(self, document_id: str, draft: ObligationDraft, tenant: str) -> dict:
        review_id = str(uuid4())
        rows = await self._neo4j.run(
            """
            MATCH (source:Document {id: $document_id, tenant: $tenant})
            MATCH (chunk:Chunk {id: $evidence_chunk_id, tenant: $tenant})-[:PART_OF]->(source)
            CREATE (review:ObligationReview {
                id: $review_id, tenant: $tenant, status: 'pending',
                source_document_id: $document_id, evidence_chunk_id: $evidence_chunk_id,
                obligation: $obligation, subject: $subject, beneficiary: $beneficiary,
                due_at: $due_at, effective_from: $effective_from, effective_to: $effective_to,
                evidence_quote: $evidence_quote, confidence: $confidence, created_at: datetime()
            })
            MERGE (review)-[:SUPPORTED_BY]->(chunk)
            RETURN review.id AS review_id, review.status AS status
            """,
            review_id=review_id,
            tenant=tenant,
            document_id=document_id,
            evidence_chunk_id=draft.evidence_chunk_id,
            obligation=draft.obligation,
            subject=draft.subject,
            beneficiary=draft.beneficiary,
            due_at=draft.due_at.isoformat() if draft.due_at else None,
            effective_from=draft.effective_from.isoformat() if draft.effective_from else None,
            effective_to=draft.effective_to.isoformat() if draft.effective_to else None,
            evidence_quote=draft.evidence_quote,
            confidence=draft.confidence,
        )
        if not rows:
            raise ValueError("obligation evidence or source document was not found in this tenant")
        return rows[0]

    async def approve_lineage(self, review_id: str, reviewed_by: str, tenant: str) -> dict:
        row = await self._resolve_lineage_review(review_id, reviewed_by, tenant, "approved")
        if not row:
            return {"error": f"lineage review {review_id} not found or already resolved"}
        relation = LineageRelation(row["relation"])
        # Relationship types cannot be parameters. The enum above makes the
        # formatted token closed and safe from Cypher injection.
        query = f"""
            MATCH (source:Document {{id: $source_document_id, tenant: $tenant}})
            MATCH (target:Document {{id: $target_document_id, tenant: $tenant}})
            MERGE (source)-[r:{relation.value}]->(target)
            SET r.source_review_id = $review_id, r.evidence_chunk_id = $evidence_chunk_id,
                r.evidence_quote = $evidence_quote, r.confidence = $confidence,
                r.effective_from = $effective_from, r.effective_to = $effective_to,
                r.recorded_at = datetime(), r.reviewed_by = $reviewed_by
            FOREACH (_ IN CASE WHEN $relation = 'SUPERSEDES' THEN [1] ELSE [] END |
                SET target.superseded_by = $source_document_id,
                    target.superseded_at = datetime())
            RETURN type(r) AS relation
        """
        await self._neo4j.run(query, **row, tenant=tenant, review_id=review_id, reviewed_by=reviewed_by)
        return {"review_id": review_id, "status": "approved", "relation": relation.value}

    async def reject_lineage(self, review_id: str, reviewed_by: str, tenant: str) -> dict:
        row = await self._resolve_lineage_review(review_id, reviewed_by, tenant, "rejected")
        return {"review_id": review_id, "status": "rejected"} if row else {
            "error": f"lineage review {review_id} not found or already resolved"
        }

    async def approve_obligation(self, review_id: str, reviewed_by: str, tenant: str) -> dict:
        rows = await self._neo4j.run(
            """
            MATCH (review:ObligationReview {id: $review_id, tenant: $tenant, status: 'pending'})
            SET review.status = 'approved', review.reviewed_by = $reviewed_by,
                review.reviewed_at = datetime()
            CREATE (o:Obligation {
                id: randomUUID(), tenant: $tenant, status: 'active',
                source_document_id: review.source_document_id,
                evidence_chunk_id: review.evidence_chunk_id, obligation: review.obligation,
                subject: review.subject, beneficiary: review.beneficiary, due_at: review.due_at,
                effective_from: review.effective_from, effective_to: review.effective_to,
                evidence_quote: review.evidence_quote, confidence: review.confidence,
                review_id: review.id, reviewed_by: $reviewed_by, recorded_at: datetime()
            })
            MERGE (o)-[:DERIVED_FROM]->(review)
            RETURN o.id AS obligation_id, o.status AS status
            """,
            review_id=review_id, tenant=tenant, reviewed_by=reviewed_by,
        )
        return rows[0] if rows else {"error": f"obligation review {review_id} not found or already resolved"}

    async def reject_obligation(self, review_id: str, reviewed_by: str, tenant: str) -> dict:
        rows = await self._neo4j.run(
            """
            MATCH (review:ObligationReview {id: $review_id, tenant: $tenant, status: 'pending'})
            SET review.status = 'rejected', review.reviewed_by = $reviewed_by,
                review.reviewed_at = datetime()
            RETURN review.id AS review_id
            """,
            review_id=review_id, tenant=tenant, reviewed_by=reviewed_by,
        )
        return {"review_id": review_id, "status": "rejected"} if rows else {
            "error": f"obligation review {review_id} not found or already resolved"
        }

    async def list_reviews(self, tenant: str, kind: str = "lineage", status: str = "pending") -> list[dict]:
        label = "LineageReview" if kind == "lineage" else "ObligationReview"
        return await self._neo4j.run(
            f"""
            MATCH (review:{label} {{tenant: $tenant}})
            WHERE $status = 'all' OR review.status = $status
            RETURN review {{.*}}, labels(review) AS labels
            ORDER BY review.created_at DESC
            LIMIT 250
            """,
            tenant=tenant, status=status,
        )

    async def obligations(self, tenant: str, as_of: str | None = None) -> list[dict]:
        return await self._neo4j.run(
            """
            MATCH (o:Obligation {tenant: $tenant, status: 'active'})
            WHERE $as_of IS NULL
               OR ((o.effective_from IS NULL OR o.effective_from <= datetime($as_of))
                   AND (o.effective_to IS NULL OR o.effective_to > datetime($as_of)))
            RETURN o {.*} AS obligation
            ORDER BY o.due_at ASC, o.recorded_at DESC
            LIMIT 500
            """,
            tenant=tenant, as_of=as_of,
        )

    async def _resolve_lineage_review(
        self, review_id: str, reviewed_by: str, tenant: str, status: str,
    ) -> dict | None:
        rows = await self._neo4j.run(
            """
            MATCH (review:LineageReview {id: $review_id, tenant: $tenant, status: 'pending'})
            SET review.status = $status, review.reviewed_by = $reviewed_by,
                review.reviewed_at = datetime()
            RETURN review.source_document_id AS source_document_id,
                   review.target_document_id AS target_document_id,
                   review.relation AS relation, review.evidence_chunk_id AS evidence_chunk_id,
                   review.evidence_quote AS evidence_quote, review.confidence AS confidence,
                   review.effective_from AS effective_from, review.effective_to AS effective_to
            """,
            review_id=review_id, tenant=tenant, reviewed_by=reviewed_by, status=status,
        )
        return rows[0] if rows else None
