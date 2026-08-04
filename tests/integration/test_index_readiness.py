"""§10 — 'Verify that every required index is online in readiness checks.' This is
the query /ready will run once built (Increment 4) — proven here directly against
the schema bootstrap from src/graph/migrations/migration_001_init_schema.py.
"""

from __future__ import annotations

import asyncio

import pytest

from src.graph.schema import ALL_INDEX_NAMES

pytestmark = pytest.mark.asyncio


async def test_schema_bootstrap_is_idempotent(executor):
    """Running the migration twice (conftest's `executor` fixture already ran it
    once) must not raise — every statement uses IF NOT EXISTS."""
    from src.graph.migrations.migration_001_init_schema import run as run_migration

    await run_migration(executor)  # second application, same session


async def test_all_required_indexes_come_online(executor):
    rows = await executor.operational_query(
        "SHOW INDEXES YIELD name, state RETURN name, state"
    )
    states_by_name = {row["name"]: row["state"] for row in rows}

    missing = [name for name in ALL_INDEX_NAMES if name not in states_by_name]
    assert not missing, f"indexes never created: {missing}"

    # Indexes can briefly report POPULATING right after creation — poll briefly.
    not_online = {name: states_by_name[name] for name in ALL_INDEX_NAMES if states_by_name[name] != "ONLINE"}
    for _ in range(10):
        if not not_online:
            break
        await asyncio.sleep(1)
        rows = await executor.operational_query(
            "SHOW INDEXES YIELD name, state RETURN name, state"
        )
        states_by_name = {row["name"]: row["state"] for row in rows}
        not_online = {name: states_by_name[name] for name in ALL_INDEX_NAMES if states_by_name[name] != "ONLINE"}

    assert not not_online, f"indexes not ONLINE after waiting: {not_online}"
