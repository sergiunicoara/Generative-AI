"""ReviewQueueService.approve() calls into AliasRegistry.register_alias() via
get_alias_registry() (review_queue.py -> alias_registry.py). Proves that cross-file
wiring survives the graphrag.* -> src.* import rewrite from Increment 1.
"""

from unittest.mock import AsyncMock

import pytest

import src.graph.alias_registry as alias_registry_module
from src.graph.review_queue import ReviewQueueService


@pytest.fixture(autouse=True)
def _reset_alias_registry_pool():
    alias_registry_module._registries.clear()
    yield
    alias_registry_module._registries.clear()


@pytest.mark.asyncio
async def test_approve_registers_alias_through_alias_registry():
    mock_neo4j = AsyncMock()
    mock_neo4j.run.side_effect = [
        [
            {
                "raw_name": "Volks Wagen",
                "raw_type": "ORG",
                "candidate_name": "Volkswagen Group",
                "candidate_type": "ORG",
                "source_doc": "conv-abc",
            }
        ],  # ReviewQueueItem MATCH/SET
        [],  # AliasRegistry.register_alias()'s MERGE (a:Alias ...)
    ]
    service = ReviewQueueService(neo4j_client=mock_neo4j)

    result = await service.approve(item_id="item-1", reviewed_by="reviewer@example.com", tenant="workspace-a")

    assert result["status"] == "approved"
    assert result["alias_registered"] == "Volks Wagen → Volkswagen Group"
    assert mock_neo4j.run.call_count == 2

    alias_merge_call = mock_neo4j.run.call_args_list[1]
    assert "MERGE (a:Alias" in alias_merge_call.args[0]
    assert alias_merge_call.kwargs["canonical_name"] == "Volkswagen Group"
    assert alias_merge_call.kwargs["raw_value"] == "Volks Wagen"

    # The registry created by get_alias_registry() inside approve() picked up
    # the new alias in its in-memory cache too.
    registry = alias_registry_module.get_alias_registry(mock_neo4j, tenant="workspace-a")
    assert registry.resolve("Volks Wagen") == ("Volkswagen Group", "ORG")


@pytest.mark.asyncio
async def test_approve_on_missing_item_does_not_touch_alias_registry():
    mock_neo4j = AsyncMock()
    mock_neo4j.run.return_value = []  # no matching pending ReviewQueueItem
    service = ReviewQueueService(neo4j_client=mock_neo4j)

    result = await service.approve(item_id="missing", reviewed_by="reviewer@example.com", tenant="workspace-a")

    assert "error" in result
    assert mock_neo4j.run.call_count == 1
