"""Re-ingesting an edge must not undo manual overrides or lifecycle transitions.

Static check of the Cypher sent to the driver; the live behaviour is proven in
tests/e2e/test_live_merge_relation_preserves_state.py.
"""
from __future__ import annotations

import pytest

from graphrag.core.models import Relation
from graphrag.graph.neo4j_client import Neo4jClient


class _Capture(Neo4jClient):
    def __init__(self):  # no driver
        self.queries: list[str] = []

    async def run(self, query, **params):
        self.queries.append(query)
        return []


def _guarded(query: str, prefix: str) -> None:
    assert "locked" in query
    assert f"r.source_type      = CASE WHEN prior_type = 'manual'" in query
    assert "r.confidence_state = CASE WHEN locked" in query
    assert "r.valid_from       = CASE WHEN locked" in query
    assert "r.valid_to         = CASE WHEN locked OR" in query
    for state in ("APPROVED", "RETRACTED", "DISPUTED"):
        assert state in query
    # The snapshot must precede the SET so evaluation order cannot matter.
    assert query.index("AS locked") < query.index("SET r.weight")
    # No unconditional overwrite remains.
    for bad in (
        f"r.confidence_state = {prefix}confidence_state,",
        f"r.source_type      = {prefix}source_type,",
        f"r.valid_to         = datetime({prefix}valid_to),",
    ):
        assert bad not in query


@pytest.mark.asyncio
async def test_merge_relation_guards_manual_state():
    c = _Capture()
    await c.merge_relation(
        Relation(source_entity_id="a", target_entity_id="b", relation="OWNS"),
        "A", "ORG", "B", "ORG", tenant="t",
    )
    _guarded(c.queries[0], "$")


@pytest.mark.asyncio
async def test_merge_relations_batch_guards_manual_state():
    c = _Capture()
    await c.merge_relations_batch([{"src_name": "A"}], tenant="t")
    _guarded(c.queries[0], "row.")
