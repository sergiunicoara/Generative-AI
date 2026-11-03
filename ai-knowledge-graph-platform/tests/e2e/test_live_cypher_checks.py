"""Parity of the compiled Cypher checks with the generated SHACL on live Neo4j.

The same fixtures are loaded as a property graph into a throwaway Neo4j and as
RDF into pySHACL; for each fixture both must agree on whether it is valid.
Skipped without Docker/testcontainers (it has not been run in the
environment that wrote it).
"""

from __future__ import annotations


import pytest
from rdflib import RDF, RDFS, Graph, Literal, Namespace

from graphrag.graph.shacl_validator import SHACLValidator
from graphrag.semantic_model import compile_model, load_model
from graphrag.semantic_model.cypher_checks import compile_cypher_checks


def _docker_and_testcontainers_available() -> bool:
    try:
        import docker
        import testcontainers  # noqa: F401

        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _docker_and_testcontainers_available(),
                                reason="Docker or testcontainers-python not available")

_PASSWORD = "cypher-checks-e2e-password"
NS = Namespace("https://example.ext/#")

# (name, Cypher that builds the property graph, triples for the RDF side, expected valid)
FIXTURES = [
    ("valid", "CREATE (:PRODUCT {tenant:'t', sku:'S1'})-[:CONTAINS]->(:SAFETY_COMPONENT {partNo:'P1'})",
     [("p", "Product", {"tenant": "t", "sku": "S1"}), ("c", "SafetyComponent", {"partNo": "P1"}),
      ("p", "contains", "c")], True),
    ("missing_required", "CREATE (:PRODUCT {tenant:'t'})-[:CONTAINS]->(:SAFETY_COMPONENT {partNo:'P1'})",
     [("p", "Product", {"tenant": "t"}), ("c", "SafetyComponent", {"partNo": "P1"}), ("p", "contains", "c")], False),
    ("wrong_datatype", "CREATE (:PRODUCT {tenant:'t', sku:'S1'})-[:CONTAINS]->(:SAFETY_COMPONENT {partNo:7})",
     [("p", "Product", {"tenant": "t", "sku": "S1"}), ("c", "SafetyComponent", {"partNo": 7}),
      ("p", "contains", "c")], False),
    ("relation_minimum", "CREATE (:PRODUCT {tenant:'t', sku:'S1'})",
     [("p", "Product", {"tenant": "t", "sku": "S1"})], False),
    ("qualified_minimum", "CREATE (:PRODUCT {tenant:'t', sku:'S1'})-[:CONTAINS]->(:PLAIN_COMPONENT {partNo:'P1'})",
     [("p", "Product", {"tenant": "t", "sku": "S1"}), ("c", "PlainComponent", {"partNo": "P1"}),
      ("p", "contains", "c")], False),
]


def _rdf(spec) -> Graph:
    graph = Graph()
    graph.add((NS.SafetyComponent, RDFS.subClassOf, NS.Component))
    graph.add((NS.PlainComponent, RDFS.subClassOf, NS.Component))
    for subject, predicate, value in spec:
        if isinstance(value, dict):
            graph.add((NS[subject], RDF.type, NS[predicate]))
            for key, literal in value.items():
                graph.add((NS[subject], NS[key], Literal(literal)))
        else:
            graph.add((NS[subject], NS[predicate], NS[value]))
    return graph


@pytest.fixture(scope="module")
def neo4j_driver():
    from neo4j import GraphDatabase
    from testcontainers.community.neo4j import Neo4jContainer

    with Neo4jContainer("neo4j:5.20-community", password=_PASSWORD) as container:
        driver = GraphDatabase.driver(container.get_connection_url(), auth=("neo4j", _PASSWORD))
        yield driver
        driver.close()


@pytest.mark.parametrize("name,cypher,triples,valid", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_cypher_checks_agree_with_shacl(neo4j_driver, tmp_path, name, cypher, triples, valid):
    from tests.unit.test_semantic_model_extensions import EXT_YAML

    path = tmp_path / "ontology" / "models" / "ext.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(EXT_YAML, encoding="utf-8")
    model = load_model(path)
    shapes = tmp_path / "shapes.ttl"
    shapes.write_text(compile_model(model).shacl, encoding="utf-8")
    shacl_valid = SHACLValidator(_rdf(triples), shapes_path=shapes).validate_report(target="parity").conforms

    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run(cypher)
        violations = [row for check in compile_cypher_checks(model) for row in session.run(check.query)]
    cypher_valid = not violations
    assert shacl_valid == valid, f"{name}: SHACL"
    assert cypher_valid == shacl_valid, f"{name}: Cypher checks disagree with SHACL ({violations})"
