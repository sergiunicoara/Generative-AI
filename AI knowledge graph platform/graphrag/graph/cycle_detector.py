"""Cyclic dependency detection in the entity relation graph.

Problem solved
--------------
Circular dependencies (A → B → C → A) silently corrupt multi-hop
traversal — Cypher variable-length path queries loop forever or return
incorrect results.  Detection must run after every ingestion batch.

Implementation
--------------
Uses Neo4j's native `apoc.algo.findCycles` when APOC is available,
with a pure-Cypher fallback using path pattern matching.

Both approaches are bounded — they only check up to MAX_CYCLE_LENGTH
hops to avoid runaway queries on dense graphs.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

MAX_CYCLE_LENGTH = 6   # max hops to check for cycles


class CycleDetector:
    """
    Detects cycles in RELATES_TO and REQUIRES edges.

    Usage::

        detector = CycleDetector(neo4j_client)
        cycles = await detector.detect()
        if cycles:
            log.warning("cycles_found", count=len(cycles))
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def detect(self, relation_types: list[str] | None = None) -> list[dict]:
        """
        Find all cycles up to MAX_CYCLE_LENGTH hops.

        Returns a list of cycle dicts, each with:
            - ``path``:   list of entity names forming the cycle
            - ``length``: number of edges in the cycle
            - ``types``:  relation types involved
        """
        # Try APOC first (faster, native)
        apoc_available = await self._check_apoc()
        if apoc_available:
            return await self._detect_apoc(relation_types)
        return await self._detect_cypher(relation_types)

    async def _check_apoc(self) -> bool:
        try:
            rows = await self._neo4j.run(
                "RETURN apoc.version() AS v"
            )
            return bool(rows)
        except Exception:
            return False

    async def _detect_apoc(self, relation_types: list[str] | None) -> list[dict]:
        rel_filter = "|".join(relation_types or ["RELATES_TO", "REQUIRES"])
        rows = await self._neo4j.run(
            f"""
            MATCH (e:Entity)
            CALL apoc.algo.findCycles([e], {{relTypesAndDirections: '{rel_filter}>', maxDepth: {MAX_CYCLE_LENGTH}}})
            YIELD path
            RETURN [n IN nodes(path) | n.name] AS path,
                   length(path)                 AS length,
                   [r IN relationships(path) | type(r)] AS types
            """
        )
        cycles = [dict(r) for r in rows]
        if cycles:
            log.warning("cycle_detector.cycles_found", count=len(cycles), method="apoc")
        else:
            log.info("cycle_detector.no_cycles", method="apoc")
        return cycles

    async def _detect_cypher(self, relation_types: list[str] | None) -> list[dict]:
        """
        Pure-Cypher fallback: find paths where start == end within N hops.
        Less efficient but works without APOC.
        """
        rows = await self._neo4j.run(
            f"""
            MATCH path = (e:Entity)-[:RELATES_TO*2..{MAX_CYCLE_LENGTH}]->(e)
            RETURN [n IN nodes(path) | n.name]     AS path,
                   length(path)                     AS length,
                   [r IN relationships(path) | type(r)] AS types
            LIMIT 100
            """
        )
        cycles = [dict(r) for r in rows]
        if cycles:
            log.warning("cycle_detector.cycles_found", count=len(cycles), method="cypher")
        else:
            log.info("cycle_detector.no_cycles", method="cypher")
        return cycles

    async def flag_cycles(self, cycles: list[dict]) -> None:
        """
        Mark nodes involved in cycles with a `has_cycle = true` flag
        so downstream traversal can apply bounded depth automatically.
        """
        if not cycles:
            return
        involved: set[str] = set()
        for cycle in cycles:
            involved.update(cycle.get("path", []))

        await self._neo4j.run(
            """
            UNWIND $names AS name
            MATCH (e:Entity {name: name})
            SET e.has_cycle = true, e.cycle_flagged_at = datetime()
            """,
            names=list(involved),
        )
        log.info("cycle_detector.nodes_flagged", count=len(involved))

    async def run(self) -> list[dict]:
        """Detect cycles and flag affected nodes. Returns cycle list."""
        cycles = await self.detect()
        if cycles:
            await self.flag_cycles(cycles)
        return cycles
