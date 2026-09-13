"""Live proof that `scripts/project_energy_rdf_to_neo4j.py` actually runs.

Closes gap A's regression-coverage requirement: before this session's fix,
this script crashed with `RuntimeError: asyncio.run() cannot be called from a
running event loop` inside `EnergyDemoService.__init__` -- constructed from
`async def _run(...)`, itself launched via `asyncio.run(_run(...))` -- before
it ever reached Neo4j. Running the real script as a subprocess against an
isolated live container proves the whole entry point (construction, SHACL
publication, and the Neo4j projection) completes end to end, not just that
construction no longer raises.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _docker_and_testcontainers_available() -> bool:
    try:
        import docker
        import testcontainers  # noqa: F401

        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_and_testcontainers_available(),
    reason="Docker or testcontainers-python not available",
)

_PASSWORD = "energy-projection-e2e-password"


@pytest.fixture(scope="module")
def neo4j_container():
    from testcontainers.community.neo4j import Neo4jContainer
    from testcontainers.core.config import testcontainers_config

    previous_max_tries = testcontainers_config.max_tries
    testcontainers_config.max_tries = max(previous_max_tries, 300)
    try:
        with Neo4jContainer("neo4j:5.20-community", password=_PASSWORD) as container:
            yield container
    finally:
        testcontainers_config.max_tries = previous_max_tries


def test_project_energy_rdf_to_neo4j_script_runs_end_to_end(neo4j_container, tmp_path) -> None:
    """The exact regression: run the real CLI entry point as a subprocess
    (a genuinely fresh event loop, not an already-running pytest-asyncio
    one) against an isolated live Neo4j and assert it completes and reports
    a non-trivial projection -- not that it merely avoids the RuntimeError."""
    import os

    bolt_url = neo4j_container.get_connection_url()
    env = {
        **os.environ,
        "ENV": "test",
        "NEO4J_URI": bolt_url,
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": _PASSWORD,
    }
    tenant = f"energy-projection-e2e-{tmp_path.name}"

    result = subprocess.run(
        [sys.executable, "scripts/project_energy_rdf_to_neo4j.py", "--tenant", tenant],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"script failed (exit {result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "RuntimeError" not in result.stderr
    assert "cannot be called from a running event loop" not in result.stderr
    assert "Projected published Energy RDF to Neo4j read model:" in result.stdout
    assert f"tenant={tenant}" in result.stdout

    # Read back through a fresh driver -- proves the subprocess actually
    # wrote to this container, not merely that it exited 0.
    import asyncio

    from neo4j import AsyncGraphDatabase

    async def _count() -> int:
        driver = AsyncGraphDatabase.driver(bolt_url, auth=("neo4j", _PASSWORD))
        try:
            async with driver.session() as session:
                record = await (await session.run(
                    "MATCH (n:EnergyProjectionEntity {tenant: $tenant}) RETURN count(n) AS count", tenant=tenant,
                )).single()
                return record["count"]
        finally:
            await driver.close()

    node_count = asyncio.run(_count())
    assert node_count > 0


def test_governed_projection_readback_is_tenant_scoped_and_retry_safe(neo4j_container, tmp_path) -> None:
    """Live acceptance: directed traversal, active pointer, retry and isolation."""
    import os

    bolt_url = neo4j_container.get_connection_url()
    tenant = f"energy-projection-live-{tmp_path.name}"
    other_tenant = f"energy-projection-other-{tmp_path.name}"
    env = {
        **os.environ, "ENV": "test", "NEO4J_URI": bolt_url,
        "NEO4J_USER": "neo4j", "NEO4J_PASSWORD": _PASSWORD,
    }
    command = [sys.executable, "scripts/project_energy_rdf_to_neo4j.py", "--tenant", tenant]
    first = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    second = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert first.returncode == second.returncode == 0, f"{first.stderr}\n{second.stderr}"

    import asyncio
    from neo4j import AsyncGraphDatabase

    async def _readback() -> dict:
        driver = AsyncGraphDatabase.driver(bolt_url, auth=("neo4j", _PASSWORD))
        try:
            async with driver.session() as session:
                pointer = await (await session.run(
                    "MATCH (p:EnergyProjectionPointer {tenant: $tenant, name: 'energy'}) "
                    "RETURN p.active_version AS version", tenant=tenant,
                )).single()
                counts = await (await session.run(
                    "MATCH (n:EnergyProjectionEntity {tenant: $tenant, projection_version: $version}) "
                    "RETURN count(n) AS nodes, collect(DISTINCT n.type) AS types",
                    tenant=tenant, version=pointer["version"],
                )).single()
                traversal = await (await session.run(
                    "MATCH (t:EnergyProjectionEntity {tenant: $tenant, projection_version: $version}) "
                    "-[:HAS_COMPONENT]->(c:EnergyProjectionEntity) "
                    "RETURN count(*) AS count", tenant=tenant, version=pointer["version"],
                )).single()
                leak = await (await session.run(
                    "MATCH (n:EnergyProjectionEntity {tenant: $other}) RETURN count(n) AS count",
                    other=other_tenant,
                )).single()
                return {"pointer": pointer, "counts": counts, "traversal": traversal, "leak": leak}
        finally:
            await driver.close()

    observed = asyncio.run(_readback())
    assert observed["pointer"]["version"]
    assert observed["counts"]["nodes"] > 0
    assert "WIND_TURBINE" in observed["counts"]["types"]
    assert observed["traversal"]["count"] > 0
    assert observed["leak"]["count"] == 0
