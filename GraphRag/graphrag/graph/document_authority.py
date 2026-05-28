"""Document authority hierarchy and supersession chains.

Problems solved
---------------
1. Contradictory sources — two documents disagree on a fact.
   Resolution: lower authority_level number wins (1=regulatory beats 4=informal).

2. Inconsistent document semantics — an old spec is still in the graph
   after a newer one supersedes it.
   Resolution: SUPERSEDES edges between documents; retrieval penalizes
   superseded sources.

3. Multi-source truth conflicts — same relation extracted from two docs
   with different authority levels.
   Resolution: confidence is weighted by the source document's authority.

Authority levels (lower = higher authority):
    1  REGULATORY          airworthiness directives, FARs, EASAs
    2  MANUFACTURER_SPEC   OEM design specifications
    3  INTERNAL_PROCEDURE  company SOPs and work instructions
    4  INFORMAL            emails, meeting notes, wiki pages
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

# Authority penalty applied to edge confidence when source doc is superseded
SUPERSEDED_CONFIDENCE_PENALTY = 0.5


class DocumentAuthorityService:
    """
    Manages document authority levels and supersession chains.

    Usage::

        svc = DocumentAuthorityService(neo4j_client)
        await svc.register_supersession("doc_v2_id", supersedes=["doc_v1_id"])
        authority = await svc.get_authority("doc_id")
        conflicts = await svc.find_conflicts_for_entity("SpaceX", entity_type="ORG", tenant="default")
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def set_authority_level(self, doc_id: str, level: int) -> None:
        """Set the authority level of a document."""
        await self._neo4j.run(
            """
            MATCH (d:Document {id: $doc_id})
            SET d.authority_level = $level,
                d.authority_updated_at = datetime()
            """,
            doc_id=doc_id,
            level=level,
        )
        log.info("doc_authority.set", doc_id=doc_id, level=level)

    async def register_supersession(
        self, new_doc_id: str, supersedes: list[str]
    ) -> None:
        """
        Record that new_doc_id supersedes the listed document IDs.
        Adds SUPERSEDES edges and marks old docs with `superseded_by`.
        """
        for old_doc_id in supersedes:
            await self._neo4j.run(
                """
                MATCH (new:Document {id: $new_id})
                MATCH (old:Document {id: $old_id})
                MERGE (new)-[r:SUPERSEDES]->(old)
                ON CREATE SET r.recorded_at = datetime()
                SET old.superseded_by = $new_id,
                    old.superseded_at  = datetime()
                """,
                new_id=new_doc_id,
                old_id=old_doc_id,
            )
        log.info(
            "doc_authority.supersession_registered",
            new_doc=new_doc_id,
            supersedes=supersedes,
        )

    async def get_authority(self, doc_id: str) -> dict:
        """Return authority metadata for a document."""
        rows = await self._neo4j.run(
            """
            MATCH (d:Document {id: $doc_id})
            OPTIONAL MATCH (newer:Document)-[:SUPERSEDES]->(d)
            RETURN d.authority_level AS level,
                   d.superseded_by   AS superseded_by,
                   newer.id          AS superseded_by_id,
                   d.valid_from      AS valid_from,
                   d.valid_to        AS valid_to
            """,
            doc_id=doc_id,
        )
        if not rows:
            return {"level": 4, "superseded": False}
        r = rows[0]
        return {
            "level": r.get("level") or 4,
            "superseded": r.get("superseded_by") is not None,
            "superseded_by": r.get("superseded_by"),
            "valid_from": r.get("valid_from"),
            "valid_to": r.get("valid_to"),
        }

    async def find_conflicts_for_entity(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str = "default",
    ) -> list[dict]:
        """
        Find RELATES_TO edges originating from this entity that carry
        multiple source documents (``source_doc_ids`` list length > 1).

        Each such edge was produced by merging evidence from several ingestion
        passes; when the contributing documents have different authority levels
        the lower-numbered level (= higher authority) should be preferred.

        ``entity_type`` and ``tenant`` are required to uniquely identify the
        entity — matching on name alone is ambiguous when a tenant has two
        entities with the same name but different types.

        Returns a list of dicts, one per multi-source edge::

            {
                "target":       str,          # target entity name
                "target_type":  str,          # target entity type
                "relation":     str,
                "source_doc_ids": list[str],  # all contributing docs
                "confidence":   float,        # merged edge confidence
                "doc_levels":   list[dict],   # [{id, authority_level}, ...]
                "best_doc_id":  str,          # highest-authority doc
                "resolution":   str,          # human-readable hint
            }

        Note: this is a diagnostic / authority-resolution helper.  The primary
        contradiction-detection path is ``ContradictionDetector.scan()``, which
        runs full conflict analysis including supersession checks.
        """
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
                  -[r:RELATES_TO]->
                  (t:Entity {tenant: $tenant})
            WHERE r.source_doc_ids IS NOT NULL
              AND size(r.source_doc_ids) > 1
            WITH t.name          AS tgt,
                 t.type          AS tgt_type,
                 r.relation      AS rel,
                 r.source_doc_ids AS doc_ids,
                 r.confidence    AS confidence
            UNWIND doc_ids AS doc_id
            OPTIONAL MATCH (d:Document {id: doc_id})
            WITH tgt, tgt_type, rel, doc_ids, confidence,
                 collect({id: d.id, level: coalesce(d.authority_level, 4)}) AS doc_levels
            RETURN tgt, tgt_type, rel, doc_ids, confidence, doc_levels
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
        )

        conflicts = []
        for row in rows:
            doc_levels = row.get("doc_levels") or []
            # Sort by authority level — lower number = higher authority
            ranked = sorted(doc_levels, key=lambda d: d.get("level", 4))
            best = ranked[0] if ranked else {}
            conflicts.append(
                {
                    "target":         row["tgt"],
                    "target_type":    row["tgt_type"],
                    "relation":       row["rel"],
                    "source_doc_ids": row.get("doc_ids") or [],
                    "confidence":     row.get("confidence"),
                    "doc_levels":     ranked,
                    "best_doc_id":    best.get("id"),
                    "resolution":     (
                        f"Prefer doc {best.get('id')} "
                        f"(authority level {best.get('level', 4)})"
                        if best else "No document metadata available"
                    ),
                }
            )
        return conflicts

    async def apply_authority_weights(self, edges: list[dict]) -> list[dict]:
        """
        Adjust edge confidence based on source document authority.
        Called by GNNScorer before building the adjacency matrix.

        Rules:
        - Regulatory (1): confidence × 1.0 (unchanged)
        - Manufacturer spec (2): confidence × 0.95
        - Internal procedure (3): confidence × 0.85
        - Informal (4): confidence × 0.70
        - Superseded: additional × 0.5 penalty
        """
        AUTHORITY_MULTIPLIER = {1: 1.0, 2: 0.95, 3: 0.85, 4: 0.70}

        if not edges:
            return edges

        # Fetch authority info for all source docs in one query
        doc_ids = list({e.get("source_doc_id") for e in edges if e.get("source_doc_id")})
        if not doc_ids:
            return edges

        rows = await self._neo4j.run(
            """
            UNWIND $ids AS doc_id
            MATCH (d:Document {id: doc_id})
            RETURN d.id AS id,
                   coalesce(d.authority_level, 4) AS level,
                   d.superseded_by IS NOT NULL    AS superseded
            """,
            ids=doc_ids,
        )
        doc_meta = {r["id"]: r for r in rows}

        for edge in edges:
            doc_id = edge.get("source_doc_id")
            if not doc_id or doc_id not in doc_meta:
                continue
            meta = doc_meta[doc_id]
            multiplier = AUTHORITY_MULTIPLIER.get(meta["level"], 0.70)
            if meta.get("superseded"):
                multiplier *= SUPERSEDED_CONFIDENCE_PENALTY
            edge["confidence"] = float(edge.get("confidence", 1.0)) * multiplier

        return edges
