"""Docker-backed live verification of GremlinBackend against a real
TinkerPop Gremlin Server -- closes the exact gap gremlin_client.py's module
docstring names: every traversal was unit-tested against a mocked boundary,
but none had ever been run against a real engine.

Mirrors tests/e2e/test_live_graphdb.py's structure (same
testcontainers.core.container.DockerContainer generic-container pattern,
same poll-a-real-readiness-signal-not-a-fixed-sleep discipline).

Verified manually 2026-09-23 with this exact image/flow before being
committed as a permanent test (see gremlin_client.py's "Verification
status" section for the full run-through, including the Bayesian
confidence accumulation checking out numerically: 0.8, 0.5 -> 0.9).

What this does NOT verify: a real Neptune or Cosmos DB Gremlin API
endpoint. Both are TinkerPop-compliant but have their own documented
deviations from the reference server -- see gremlin_client.py's docstring.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from graphrag.core.models import ConstraintType, Entity, Relation, SourceType
from graphrag.graph.gremlin_client import GremlinBackend


def _docker_and_testcontainers_available() -> bool:
    try:
        import docker
        import testcontainers  # noqa: F401

        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _docker_and_testcontainers_available(),
        reason="Docker or testcontainers-python not available",
    ),
]

# The TinkerPop project's own reference server image -- no license, no
# vendor account, just the protocol every Gremlin-compliant store implements.
_IMAGE = "tinkerpop/gremlin-server:3.7.2"
_CONTAINER_PORT = 8182


@pytest.fixture(scope="module")
def gremlin_container():
    from testcontainers.core.container import DockerContainer

    container = DockerContainer(_IMAGE).with_exposed_ports(_CONTAINER_PORT)
    with container as c:
        _wait_for_gremlin_server(c)
        yield c


def _wait_for_gremlin_server(container, timeout: float = 90.0) -> None:
    """Poll by actually opening a GremlinBackend connection and running a
    trivial traversal -- there's no plain HTTP health endpoint the way
    GraphDB/Blazegraph have; the WebSocket protocol handshake itself is the
    real readiness signal."""
    import asyncio

    url = _ws_url(container)
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            asyncio.run(_probe(url))
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"Gremlin Server did not become ready within {timeout}s: {last_error}")


async def _probe(url: str) -> None:
    backend = GremlinBackend(url)
    try:
        await backend.entity_exists("__probe__", "__probe__")
    finally:
        await backend.close()


def _ws_url(container) -> str:
    host = container.get_container_host_ip()
    port = container.get_exposed_port(_CONTAINER_PORT)
    return f"ws://{host}:{port}/gremlin"


@pytest.fixture
async def backend(gremlin_container):
    b = GremlinBackend(_ws_url(gremlin_container))
    try:
        yield b
    finally:
        await b.close()


def _make_entity(**overrides) -> Entity:
    defaults = dict(
        id="e-faa", name="FAA", type="ORG", description="", embedding=[],
        source_type=SourceType.DOCUMENT, source_doc_id="doc1",
        extraction_model="m", prompt_version="v1",
        resolution_status="unresolved", resolution_method="",
    )
    defaults.update(overrides)
    return Entity(**defaults)


def _make_relation(**overrides) -> Relation:
    defaults = dict(
        id="r1", source_entity_id="e-faa", target_entity_id="e-boeing", relation="REGULATES",
        weight=1.0, confidence=0.8, confidence_state="ASSERTED",
        extracted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_doc_id="doc1", source_type=SourceType.DOCUMENT,
        constraint_type=ConstraintType.SOFT, valid_from=None, valid_to=None,
    )
    defaults.update(overrides)
    return Relation(**defaults)


class TestGremlinBackendLive:
    async def test_full_entity_relation_lifecycle(self, backend: GremlinBackend) -> None:
        tenant = "e2e-gremlin"

        assert await backend.entity_exists("FAA", "ORG", tenant=tenant) is False

        await backend.merge_entity(_make_entity(), tenant=tenant)
        assert await backend.entity_exists("FAA", "ORG", tenant=tenant) is True

        # ON-MATCH refresh: empty description gets filled by a second merge
        await backend.merge_entity(
            _make_entity(description="US aviation regulator", embedding=[0.1, 0.2]), tenant=tenant,
        )

        await backend.merge_entity(_make_entity(id="e-boeing", name="Boeing", description="Aircraft manufacturer"), tenant=tenant)

        # merge_mentions needs a Chunk vertex -- Chunk CRUD isn't part of
        # GraphBackend's scope, so create one directly via the low-level
        # traversal boundary, same as the ad hoc manual verification run.
        await backend._submit(
            lambda g: g.add_v("Chunk").property("id", "c1").property("tenant", tenant)
        )
        await backend.merge_mentions("c1", "FAA", "ORG", tenant=tenant)
        await backend.merge_mentions("c1", "Boeing", "ORG", tenant=tenant)

        entities = await backend.get_all_entities(tenant=tenant)
        assert {e["name"] for e in entities} == {"FAA", "Boeing"}

        await backend.merge_relation(_make_relation(), "FAA", "ORG", "Boeing", "ORG", tenant=tenant)
        relations = await backend.get_all_relations(tenant=tenant)
        assert len(relations) == 1
        assert relations[0]["relation"] == "REGULATES"

        # Repeat ingest of the same document must not duplicate the edge or
        # raise confidence -- the exact regression Neo4jClient's own Cypher
        # comments call out.
        await backend.merge_relation(_make_relation(), "FAA", "ORG", "Boeing", "ORG", tenant=tenant)
        relations = await backend.get_all_relations(tenant=tenant)
        assert len(relations) == 1

        # A second, DIFFERENT contributing document combines confidence via
        # 1 - (1-p)(1-q): 1 - (1-0.8)(1-0.5) = 0.9.
        await backend.merge_relation(
            _make_relation(id="r2", confidence=0.5, source_doc_id="doc2"),
            "FAA", "ORG", "Boeing", "ORG", tenant=tenant,
        )

        neighbors = await backend.get_entity_neighbors(["c1"], tenant=tenant)
        by_name = {n["entity"]: n["neighbors"] for n in neighbors}
        assert by_name["FAA"] == ["Boeing"]
        assert by_name["Boeing"] == ["FAA"]

        rels_for_faa = await backend.get_relations_for_entity("FAA", "ORG", tenant=tenant)
        assert len(rels_for_faa) == 1
        assert rels_for_faa[0]["direction"] == "outgoing"
        assert rels_for_faa[0]["confidence"] == pytest.approx(0.9)

        rels_for_boeing = await backend.get_relations_for_entity("Boeing", "ORG", tenant=tenant)
        assert rels_for_boeing[0]["direction"] == "incoming"

    async def test_as_of_rejects_rather_than_silently_ignoring(self, backend: GremlinBackend) -> None:
        with pytest.raises(NotImplementedError, match="as_of"):
            await backend.get_entity_neighbors(["c1"], as_of="2026-01-01T00:00:00Z")
