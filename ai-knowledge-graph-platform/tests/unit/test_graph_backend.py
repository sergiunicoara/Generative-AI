"""Structural conformance: both Neo4jClient and GremlinBackend satisfy the
GraphBackend Protocol -- see graphrag/graph/graph_backend.py's module
docstring for what's deliberately excluded from that Protocol.
"""

from __future__ import annotations

from graphrag.graph.graph_backend import GraphBackend
from graphrag.graph.gremlin_client import GremlinBackend
from graphrag.graph.neo4j_client import Neo4jClient


class TestGraphBackendConformance:
    def test_neo4j_client_satisfies_graph_backend(self) -> None:
        client = Neo4jClient.__new__(Neo4jClient)  # bypass __init__, no real driver
        assert isinstance(client, GraphBackend)

    def test_gremlin_backend_satisfies_graph_backend(self) -> None:
        backend = GremlinBackend.__new__(GremlinBackend)  # bypass __init__, no real connection
        assert isinstance(backend, GraphBackend)
