"""Docker-backed live GraphDB verification: R2RML materialization -> real
repository auto-provisioning -> real load -> real SPARQL 1.1 query -> proof
that data survives a container restart.

Closes the exact gap named in a follow-up platform review: *"Run the Energy
workflow against one RDF platform named in the JD, with automated load/
query/restart tests. The current adapter explicitly marks GraphDB, Stardog,
RDFox and Virtuoso loading as unverified."* GraphDB is the one of those four
with both a genuinely free/unlicensed Docker image and a documented REST API
this session could verify live -- Stardog/RDFox need a license nobody has
here, and Virtuoso has no verified image anywhere in this repo.

Mirrors tests/e2e/test_live_blazegraph.py's structure closely (same
testcontainers.core.container.DockerContainer generic-container pattern,
same poll-a-real-health-endpoint-not-a-fixed-sleep discipline, same
sys.path setup for script-shaped fixture imports).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Matches tests/unit/test_r2rml_rdf_materialization.py's exact import
# pattern for this script-shaped fixture builder.
from create_energy_demo_sqlite import create as create_energy_demo_sqlite  # noqa: E402
from graphrag.graph.triplestore import TripleStoreTarget  # noqa: E402
from graphrag.ingestion.r2rml_rdf import materialize_r2rml  # noqa: E402
from graphrag.ingestion.relational import SQLiteSourceConnector  # noqa: E402


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

# GraphDB 11.0+ requires a registered license file to start at all (Ontotext's
# own licensing docs, confirmed while scoping this test) -- only a pre-11.0
# tag runs unlicensed. 10.8.1 confirmed live (2026-09): "Product: GRAPHDB_LITE
# ... Licensee: Freeware ... Expiry date: none". compose.energy-demo.yaml
# pins 11.2.0 for the manual demo path, which cannot start without a license
# nobody has here -- deliberately not reused for that reason.
_IMAGE = "ontotext/graphdb:10.8.1"
_CONTAINER_PORT = 7200

# Same "which assets have open work orders" finding tests/e2e/test_live_blazegraph.py
# proves against Blazegraph -- proving the identical result over GraphDB too.
_MAINTENANCE_SPARQL = """
PREFIX energy: <https://example.energy.demo/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?asset ?assetLabel ?workOrder ?status WHERE {
  ?workOrder a energy:WorkOrder ;
             energy:status ?status ;
             energy:concernsAsset ?asset .
  ?asset rdfs:label ?assetLabel .
  FILTER(?status = "open")
}
ORDER BY ?asset
"""

_EXPECTED_OPEN_ASSETS = ["Wind turbine WT-01", "Wind turbine WT-02"]


@pytest.fixture(scope="module")
def graphdb_container():
    from testcontainers.core.container import DockerContainer

    container = DockerContainer(_IMAGE).with_exposed_ports(_CONTAINER_PORT)
    with container as c:
        _wait_for_graphdb(c)
        yield c


def _wait_for_graphdb(container, timeout: float = 90.0) -> None:
    """Poll /protocol (any 200 response means the workbench HTTP server is
    up) rather than a fixed sleep. /rest/monitor/health -- the endpoint
    compose.energy-demo.yaml's own healthcheck names -- was confirmed live
    (2026-09, this image) to 500 on a plain GET ("Request method 'GET' not
    supported"), so that healthcheck comment is itself slightly wrong; this
    test uses the endpoint actually confirmed to work instead of repeating
    that mistake.
    """
    host = container.get_container_host_ip()
    port = container.get_exposed_port(_CONTAINER_PORT)
    protocol_url = f"http://{host}:{port}/protocol"
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(protocol_url, timeout=5.0)
            if response.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"GraphDB did not become ready within {timeout}s: {last_error}")


def _endpoint_url(container) -> str:
    host = container.get_container_host_ip()
    port = container.get_exposed_port(_CONTAINER_PORT)
    return f"http://{host}:{port}"


class TestLiveGraphDBR2RMLPipeline:
    async def test_materialize_load_and_query_the_real_endpoint(
        self, graphdb_container, tmp_path,
    ) -> None:
        db_path = tmp_path / "energy-demo-sap.sqlite"
        create_energy_demo_sqlite(db_path)
        mapping_path = ROOT / "ontology" / "mappings" / "energy-assets.r2rml.ttl"

        graph = await materialize_r2rml(mapping_path, SQLiteSourceConnector(db_path))
        ttl_bytes = graph.serialize(format="turtle").encode("utf-8")

        target = TripleStoreTarget(
            "graphdb", _endpoint_url(graphdb_container), repository="kg_e2e",
        )
        status = await target.load(ttl_bytes)
        assert 200 <= status < 300

        rows = await target.query(_MAINTENANCE_SPARQL)
        assert sorted(row["assetLabel"] for row in rows) == _EXPECTED_OPEN_ASSETS

    async def test_repository_creation_is_idempotent(
        self, graphdb_container, tmp_path,
    ) -> None:
        """Creating the same repository twice must not fail -- GraphDB
        returns 400 with an "already exists" message for this case (not a
        clean 409 like Blazegraph), confirmed live before implementing the
        tolerance check in TripleStoreTarget._ensure_graphdb_repository()."""
        db_path = tmp_path / "energy-demo-sap.sqlite"
        create_energy_demo_sqlite(db_path)
        mapping_path = ROOT / "ontology" / "mappings" / "energy-assets.r2rml.ttl"
        graph = await materialize_r2rml(mapping_path, SQLiteSourceConnector(db_path))
        ttl_bytes = graph.serialize(format="turtle").encode("utf-8")
        endpoint = _endpoint_url(graphdb_container)

        first = await TripleStoreTarget("graphdb", endpoint, repository="kg_idempotent").load(ttl_bytes)
        second = await TripleStoreTarget("graphdb", endpoint, repository="kg_idempotent").load(ttl_bytes)

        assert 200 <= first < 300
        assert 200 <= second < 300

    async def test_data_survives_a_container_restart(
        self, graphdb_container, tmp_path,
    ) -> None:
        """The "restart" half of "automated load/query/restart tests" --
        proves persistence survives a process restart, not just an
        in-memory session within one running container."""
        db_path = tmp_path / "energy-demo-sap.sqlite"
        create_energy_demo_sqlite(db_path)
        mapping_path = ROOT / "ontology" / "mappings" / "energy-assets.r2rml.ttl"
        graph = await materialize_r2rml(mapping_path, SQLiteSourceConnector(db_path))
        ttl_bytes = graph.serialize(format="turtle").encode("utf-8")

        target = TripleStoreTarget(
            "graphdb", _endpoint_url(graphdb_container), repository="kg_restart",
        )
        await target.load(ttl_bytes)
        before = await target.query(_MAINTENANCE_SPARQL)
        assert sorted(row["assetLabel"] for row in before) == _EXPECTED_OPEN_ASSETS

        graphdb_container.get_wrapped_container().restart()
        _wait_for_graphdb(graphdb_container)

        # Re-resolve the endpoint after restart rather than reusing the
        # pre-restart TripleStoreTarget -- the exposed host port is stable
        # across a `docker restart` of the same container, but re-querying
        # it is cheap insurance against relying on that undocumented detail.
        restarted_target = TripleStoreTarget(
            "graphdb", _endpoint_url(graphdb_container), repository="kg_restart",
        )
        after = await restarted_target.query(_MAINTENANCE_SPARQL)
        assert sorted(row["assetLabel"] for row in after) == _EXPECTED_OPEN_ASSETS
