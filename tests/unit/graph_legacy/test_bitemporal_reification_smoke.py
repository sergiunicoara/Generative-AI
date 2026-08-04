"""Smoke tests for the two legacy src/graph/*.py modules that needed no import
fixes (bitemporal.py, reification.py) — proves they still work against a mocked
Neo4j client after the fork, and that reification.py's one f-string-built Cypher
fragment (`match_clause` in get_statements) never lets caller-controlled text into
the query text it sends to the driver — it only ever selects between three fixed
templates keyed off `role`, so a `role` value outside {"subject","object"} still
resolves to a safe, fully-parameterized fragment ("both") rather than smuggling
arbitrary text into the Cypher string.
"""

from unittest.mock import AsyncMock

import pytest

from src.graph.bitemporal import BitemporalStore
from src.graph.reification import ReificationService


@pytest.mark.asyncio
async def test_bitemporal_as_of_entities_sends_parameterized_query():
    mock_neo4j = AsyncMock()
    mock_neo4j.run.return_value = []
    store = BitemporalStore(mock_neo4j)

    await store.as_of_entities(
        valid_time="2024-06-01T00:00:00",
        transaction_time="2024-09-01T00:00:00",
        tenant="acme'; DROP DATABASE neo4j; --",
    )

    cypher, kwargs = mock_neo4j.run.call_args.args[0], mock_neo4j.run.call_args.kwargs
    assert "$tenant" in cypher
    assert "acme" not in cypher  # tenant value must never be interpolated into the query text
    assert kwargs["tenant"] == "acme'; DROP DATABASE neo4j; --"


@pytest.mark.asyncio
async def test_bitemporal_transaction_diff_runs_three_parameterized_queries():
    mock_neo4j = AsyncMock()
    mock_neo4j.run.return_value = [{"count": 0}]
    store = BitemporalStore(mock_neo4j)

    result = await store.transaction_diff(
        tt_from="2024-01-01T00:00:00", tt_to="2024-07-01T00:00:00", tenant="acme"
    )

    assert result["tenant"] == "acme"
    assert mock_neo4j.run.call_count == 3
    for call in mock_neo4j.run.call_args_list:
        assert "$tt_from" in call.args[0] or "$tt_to" in call.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["subject", "object", "both", "not-a-real-role"])
async def test_reification_get_statements_match_clause_is_always_fixed_template(role):
    mock_neo4j = AsyncMock()
    mock_neo4j.run.return_value = []
    svc = ReificationService(mock_neo4j)

    await svc.get_statements("Acme Corp", "ORG", tenant="default", role=role)

    cypher = mock_neo4j.run.call_args.args[0]
    # The match clause is one of exactly three fixed templates selected by `role`;
    # an unrecognized role falls through to the "both" template (else branch) —
    # it must never appear verbatim inside the Cypher text itself.
    assert role not in cypher or role in ("subject", "object", "both")
    assert "$name" in cypher and "$type" in cypher and "$tenant" in cypher


@pytest.mark.asyncio
async def test_reification_reify_relation_is_idempotent_and_parameterized():
    mock_neo4j = AsyncMock()
    mock_neo4j.run.side_effect = [
        [],  # inherited-confidence lookup (no existing edge)
        [{"id": "stmt-123"}],  # MERGE result
    ]
    svc = ReificationService(mock_neo4j)

    stmt_id = await svc.reify_relation(
        src_name="Elena Popescu", src_type="PERSON",
        relation="RAISED_OBJECTION", tgt_name="Pricing", tgt_type="OBJECTION",
        tenant="workspace-a",
    )

    assert stmt_id == "stmt-123"
    merge_call = mock_neo4j.run.call_args_list[1]
    assert "MERGE" in merge_call.args[0]
    assert merge_call.kwargs["tenant"] == "workspace-a"
