"""Load tests — concurrency benchmarks for GraphRAG in-process components.

These tests do NOT require a live Neo4j instance or Redis server.  They use
AsyncMock for external dependencies and measure wall-clock latency for pure
Python + asyncio work.

Thresholds are intentionally generous (100 ms average) to make the suite
pass on any CI machine without hardware assumptions.  Tighten them against
your production baseline once you have profiling data.

Run with:
    pytest tests/load/ -v
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag.core.models import SessionTurn


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_turn(q: str = "question", a: str = "answer") -> SessionTurn:
    return SessionTurn(question=q, answer=a)


async def _measure(coros: list, *, label: str) -> dict:
    """Run coroutines concurrently and report timing statistics."""
    t0 = time.perf_counter()
    await asyncio.gather(*coros)
    elapsed = time.perf_counter() - t0
    n = len(coros)
    return {
        "label":    label,
        "n":        n,
        "total_s":  elapsed,
        "avg_ms":   (elapsed / n) * 1000 if n else 0,
    }


# ── 1. Session store — concurrent saves ───────────────────────────────────────

@pytest.mark.asyncio
async def test_session_store_concurrent_saves():
    """50 concurrent save_turn() calls must all succeed in < 100 ms average."""
    from graphrag.retrieval.session_store import SessionStore

    store = SessionStore(redis_url=None)
    n = 50

    coros = [
        store.save_turn(f"sess-{i % 5}", _make_turn(f"q{i}", f"a{i}"))
        for i in range(n)
    ]
    stats = await _measure(coros, label="session_store.concurrent_saves")

    assert stats["avg_ms"] < 100, (
        f"average save latency {stats['avg_ms']:.1f} ms exceeds 100 ms budget"
    )
    # Verify data integrity: each of the 5 session slots has turns
    for slot in range(5):
        turns = await store.load_turns(f"sess-{slot}")
        assert len(turns) > 0, f"sess-{slot} should have turns after concurrent saves"


# ── 2. Session store — concurrent loads ───────────────────────────────────────

@pytest.mark.asyncio
async def test_session_store_concurrent_loads():
    """50 concurrent load_turns() all return the correct turn count."""
    from graphrag.retrieval.session_store import SessionStore

    store = SessionStore(redis_url=None)
    sessions = {f"s{i}": [_make_turn(f"q{j}", f"a{j}") for j in range(3)] for i in range(5)}

    # Pre-populate sessions sequentially
    for sid, turns in sessions.items():
        for t in turns:
            await store.save_turn(sid, t)

    # 50 concurrent loads spread across the 5 sessions
    session_ids = list(sessions.keys())
    coros = [
        store.load_turns(session_ids[i % len(session_ids)])
        for i in range(50)
    ]
    results = await asyncio.gather(*coros)

    for turns in results:
        assert len(turns) == 3, "each session has 3 turns; all reads must agree"


# ── 3. Contradiction detector — no shared state across concurrent scans ────────

@pytest.mark.asyncio
async def test_contradiction_scan_concurrent():
    """10 concurrent scan() calls share the same detector; no cross-call contamination."""
    from graphrag.graph.contradiction_detector import ContradictionDetector

    # Fast mock that always returns empty results
    neo4j_mock = MagicMock()
    neo4j_mock.run = AsyncMock(return_value=[])

    detector = ContradictionDetector(neo4j_mock)
    coros = [detector.scan(tenant=f"tenant_{i}") for i in range(10)]
    results = await asyncio.gather(*coros)

    for i, conflicts in enumerate(results):
        assert conflicts == [], (
            f"scan for tenant_{i} returned unexpected conflicts: {conflicts}"
        )


# ── 4. Community manager — concurrent staleness checks ────────────────────────

@pytest.mark.asyncio
async def test_community_manager_concurrent_staleness():
    """20 concurrent check_staleness() calls all return the same score (no races)."""
    from graphrag.graph.community_manager import CommunityManager

    neo4j_mock = MagicMock()
    # Return consistent data for every call pair (LIMIT 1 snap + current counts)
    neo4j_mock.run = AsyncMock(side_effect=lambda *a, **kw: [
        {"entity_count": 100, "edge_count": 200, "recorded_at": "2025-01-01T00:00:00"}
    ] if "ORDER BY" in a[0] else [{"entities": 110, "edges": 210}])

    manager = CommunityManager(neo4j_mock)
    coros = [manager.check_staleness(tenant="default") for _ in range(20)]
    reports = await asyncio.gather(*coros)

    scores = {r["staleness_score"] for r in reports}
    assert len(scores) == 1, f"concurrent calls produced different scores: {scores}"


# ── 5. Inference engine — concurrent runs with distinct tenants ────────────────

@pytest.mark.asyncio
async def test_inference_engine_concurrent_runs():
    """5 concurrent engine.run() calls each return their own tenant-scoped report."""
    from graphrag.graph.inference_engine import ForwardChainingEngine

    neo4j_mock = MagicMock()
    neo4j_mock.run = AsyncMock(return_value=[])

    engine = ForwardChainingEngine(neo4j_mock)
    tenants = [f"tenant_{i}" for i in range(5)]
    coros = [engine.run(tenant=t, max_iterations=1, dry_run=True) for t in tenants]
    reports = await asyncio.gather(*coros)

    returned_tenants = {r["tenant"] for r in reports}
    assert returned_tenants == set(tenants), (
        f"each concurrent run should carry its own tenant; got {returned_tenants}"
    )
    for report in reports:
        assert "total_inferred" in report
        assert "by_rule" in report
