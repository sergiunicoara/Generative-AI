"""Docker-backed live Blazegraph verification: R2RML materialization -> real
triplestore load -> real SPARQL 1.1 query, proving the "local triplestore
deployment" wishlist item for real rather than only documenting the manual
steps.

This is the standing test the R2RML item's plan deliberately deferred: *"If
a standing live-Blazegraph e2e test is wanted later ... flagged as an
explicit follow-up, out of scope here."* graphrag/ingestion/r2rml_rdf.py's
own unit test (tests/unit/test_r2rml_rdf_materialization.py) already proves
correct R2RML execution semantics against a real SPARQL engine (rdflib's,
in-process) -- what it explicitly does NOT prove is the HTTP load/query path
against a real running triplestore. This file closes that gap.

Isolated per test run via testcontainers' generic DockerContainer (same
lyrasis/blazegraph:2.1.5 image pinned in docker-compose.yml), matching this
directory's existing house pattern (test_relational_postgres_neo4j.py uses
testcontainers.community.postgres/neo4j the same way) rather than reusing
the shared docker-compose stack -- no data pollution across runs, no port
collision with a developer's own `docker compose up` stack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Matches tests/unit/test_r2rml_rdf_materialization.py's exact import
# pattern for this script-shaped fixture builder.
from create_energy_demo_sqlite import create as create_energy_demo_sqlite  # noqa: E402
from load_blazegraph import load as load_blazegraph  # noqa: E402
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

# Matches docker-compose.yml's blazegraph service exactly: same image, same
# context path (lyrasis/blazegraph:2.1.5 deploys bigdata.war under /bigdata,
# NOT /blazegraph -- see scripts/load_blazegraph.py's docstring).
_IMAGE = "lyrasis/blazegraph:2.1.5"
_CONTAINER_PORT = 8080

# The exact "which assets currently have open work orders" maintenance
# query from tests/unit/test_r2rml_rdf_materialization.py -- proving the
# same finding a real operator would ask for, now over real HTTP/SPARQL
# rather than an in-process rdflib Graph.
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


@pytest.fixture(scope="module")
def blazegraph_container():
    from testcontainers.core.container import DockerContainer

    container = DockerContainer(_IMAGE).with_exposed_ports(_CONTAINER_PORT)
    with container as c:
        _wait_for_blazegraph(c)
        yield c


def _wait_for_blazegraph(container, timeout: float = 90.0) -> None:
    """Poll /bigdata/status the same way docker-compose.yml's own healthcheck
    does, rather than a fixed sleep -- the image's startup time isn't
    constant across machines/CI runners."""
    import time

    host = container.get_container_host_ip()
    port = container.get_exposed_port(_CONTAINER_PORT)
    status_url = f"http://{host}:{port}/bigdata/status"
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(status_url, timeout=5.0)
            if response.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"Blazegraph did not become ready within {timeout}s: {last_error}")


def _endpoint_url(container) -> str:
    host = container.get_container_host_ip()
    port = container.get_exposed_port(_CONTAINER_PORT)
    return f"http://{host}:{port}"


class TestLiveBlazegraphR2RMLPipeline:
    """SQLite row -> R2RML-materialized triple -> real Blazegraph load ->
    real SPARQL 1.1 query -> the correct maintenance finding. Same fixture
    and expected result as the in-process unit test, over a real HTTP
    endpoint this time."""

    async def test_materialize_load_and_query_the_real_endpoint(
        self, blazegraph_container, tmp_path,
    ) -> None:
        db_path = tmp_path / "energy-demo-sap.sqlite"
        create_energy_demo_sqlite(db_path)
        mapping_path = ROOT / "ontology" / "mappings" / "energy-assets.r2rml.ttl"

        graph = await materialize_r2rml(mapping_path, SQLiteSourceConnector(db_path))
        assert len(graph) > 0  # sanity: the earlier unit test already pins the exact count (32)
        ttl_bytes = graph.serialize(format="turtle").encode("utf-8")

        target = TripleStoreTarget("blazegraph", _endpoint_url(blazegraph_container), namespace="kb")
        status = await target.load(ttl_bytes)
        assert 200 <= status < 300

        rows = await target.query(_MAINTENANCE_SPARQL)

        # Labels confirmed live against the actual create_energy_demo_sqlite.py
        # fixture, not guessed: "Wind turbine WT-0N", lowercase 't'.
        assets_with_open_work_orders = sorted(row["assetLabel"] for row in rows)
        assert assets_with_open_work_orders == ["Wind turbine WT-01", "Wind turbine WT-02"]
        # WO-9003 (closed, WT-03) must not appear -- proves the FILTER(?status
        # = "open") and the materialized data both round-tripped correctly
        # through a real triplestore, not just rdflib's in-process engine.
        assert "Wind turbine WT-03" not in assets_with_open_work_orders

    async def test_load_blazegraph_script_function_works_against_the_real_endpoint(
        self, blazegraph_container, tmp_path,
    ) -> None:
        """Proves scripts/load_blazegraph.py's own load() -- the function the
        documented CLI (`python scripts/load_blazegraph.py --tenant ...`)
        actually calls -- not just the lower-level TripleStoreTarget it
        wraps. Deliberately a non-default namespace ("script_load_e2e", vs.
        the other test's pre-provisioned "kb"): this is the exact case that
        failed 404 before TripleStoreTarget.ensure_namespace() existed --
        the script's own docstring documents `--namespace acme_kb` as valid
        usage, and that usage was broken for any namespace but "kb" until
        this test caught it live."""
        db_path = tmp_path / "energy-demo-sap.sqlite"
        create_energy_demo_sqlite(db_path)
        mapping_path = ROOT / "ontology" / "mappings" / "energy-assets.r2rml.ttl"

        graph = await materialize_r2rml(mapping_path, SQLiteSourceConnector(db_path))
        ttl_path = tmp_path / "graph.ttl"
        graph.serialize(destination=str(ttl_path), format="turtle")

        status = await load_blazegraph(
            ttl_path, endpoint=_endpoint_url(blazegraph_container), namespace="script_load_e2e",
        )
        assert 200 <= status < 300

        # Loading the same file into the same (now-existing) namespace a
        # second time must not fail -- ensure_namespace() must treat the
        # namespace already existing as success (409), not an error.
        status_again = await load_blazegraph(
            ttl_path, endpoint=_endpoint_url(blazegraph_container), namespace="script_load_e2e",
        )
        assert 200 <= status_again < 300

        target = TripleStoreTarget(
            "blazegraph", _endpoint_url(blazegraph_container), namespace="script_load_e2e",
        )
        rows = await target.query(_MAINTENANCE_SPARQL)
        assert sorted(row["assetLabel"] for row in rows) == ["Wind turbine WT-01", "Wind turbine WT-02"]
