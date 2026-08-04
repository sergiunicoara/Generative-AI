"""GraphExecutor.tenant_query()'s scoping guard is checked before any Neo4j call
is made, so this is a pure unit test — no database needed."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.graph.execution import GraphExecutor, TenantScopingError, scoped_match


def test_scoped_match_always_includes_workspace_id():
    pattern = scoped_match("Account", "a")
    assert pattern == "(a:Account {workspace_id: $workspace_id})"


def test_scoped_match_with_extra_props():
    pattern = scoped_match("Account", "a", account_id="account_id")
    assert pattern == "(a:Account {workspace_id: $workspace_id, account_id: $account_id})"


@pytest.mark.asyncio
async def test_tenant_query_rejects_cypher_with_neither_scoping_form():
    mock_client = AsyncMock()
    executor = GraphExecutor(mock_client)

    with pytest.raises(TenantScopingError):
        await executor.tenant_query(
            "MATCH (a:Account) WHERE a.name = $name RETURN a",
            workspace_id="ws-1",
            name="Acme",
        )
    mock_client.run.assert_not_called()


@pytest.mark.asyncio
async def test_tenant_query_accepts_where_equality_scoping_form():
    """Needed for full-text/vector procedure calls (CALL db.index....queryNodes)
    which have no property-map MATCH pattern to scope at all — only a WHERE
    equality after YIELD."""
    mock_client = AsyncMock()
    mock_client.run.return_value = []
    executor = GraphExecutor(mock_client)

    await executor.tenant_query(
        "CALL db.index.fulltext.queryNodes('idx', $q) YIELD node, score "
        "WHERE node.workspace_id = $workspace_id RETURN node, score",
        workspace_id="ws-1",
        q="acme",
    )
    mock_client.run.assert_called_once()


@pytest.mark.asyncio
async def test_tenant_query_rejects_a_bare_parameter_with_no_property_map():
    mock_client = AsyncMock()
    executor = GraphExecutor(mock_client)

    with pytest.raises(TenantScopingError):
        await executor.tenant_query("MATCH (a:Account) RETURN a", workspace_id="ws-1")
    mock_client.run.assert_not_called()


@pytest.mark.asyncio
async def test_tenant_query_accepts_scoped_match_output():
    mock_client = AsyncMock()
    mock_client.run.return_value = []
    executor = GraphExecutor(mock_client)

    match = scoped_match("Account", "a", account_id="account_id")
    await executor.tenant_query(f"MATCH {match} RETURN a", workspace_id="ws-1", account_id="acc-1")

    mock_client.run.assert_called_once()
    call_kwargs = mock_client.run.call_args.kwargs
    assert call_kwargs["workspace_id"] == "ws-1"
    assert call_kwargs["account_id"] == "acc-1"


@pytest.mark.asyncio
async def test_schema_query_and_operational_query_bypass_the_scoping_guard():
    mock_client = AsyncMock()
    mock_client.run.return_value = []
    executor = GraphExecutor(mock_client)

    await executor.schema_query("CREATE INDEX foo IF NOT EXISTS FOR (n:Account) ON (n.name)")
    await executor.operational_query("SHOW INDEXES YIELD name RETURN name")

    assert mock_client.run.call_count == 2
