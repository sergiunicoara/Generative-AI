"""Typed outcomes, capability matrix, qualified cardinalities, relationship
attributes, composite keys and the GQL graph-type rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import OWL, RDF, RDFS, XSD, Graph, Literal, Namespace
from rdflib.collection import Collection

from graphrag.graph.shacl_validator import SHACLValidator
from graphrag.semantic_model import (
    SemanticModelError, SemanticMutationValidator, build_capability_matrix, compile_model, compile_to_disk,
    load_model, render_gql,
)
from graphrag.semantic_model.compiler import Diagnostic, OUTCOMES, Target, target_capabilities
from graphrag.semantic_model.matrix import CODE_RULE

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("energy-asset-intelligence", "automotive-iatf-quality", "contextual-knowledge")
NS = Namespace("https://example.ext/#")

EXT_YAML = """
id: ext
version: 1.0.0
namespace: 'https://example.ext/#'
label: Ext
prefixes: {}
unknown_property_policy: reject
mixins: {}
types:
  Product:
    properties:
      tenant: {datatype: string, required: true}
      sku: {datatype: string, required: true}
    composite_keys: [[tenant, sku]]
  Component:
    abstract: true
    properties:
      partNo: {datatype: string, required: true}
  SafetyComponent:
    extends: Component
  PlainComponent:
    extends: Component
relations:
  contains:
    source: Product
    target: Component
    cardinality: {min: 1}
    qualified:
      - on_class: SafetyComponent
        cardinality: {min: 1, max: 2}
    properties:
      quantity: {datatype: integer, required: true}
      note: {datatype: string}
artifacts:
  owl: out/a.ttl
  shacl: out/b.ttl
  neo4j: out/c.cypher
  diagnostics: out/d.json
  gql: out/e.gql
"""


@pytest.fixture
def ext(tmp_path):
    path = tmp_path / "ontology" / "models" / "ext.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(EXT_YAML, encoding="utf-8")
    return load_model(path)


def _invalid(tmp_path, fragment: str):
    path = tmp_path / "bad.yaml"
    path.write_text(EXT_YAML.replace("    composite_keys: [[tenant, sku]]\n", fragment), encoding="utf-8")
    with pytest.raises(SemanticModelError) as caught:
        load_model(path)
    return str(caught.value)


# -- typed outcomes -----------------------------------------------------------


def _diag(fidelity, control=None):
    return Diagnostic("warning", "X", Target.NEO4J, "m.yaml", 1, "e", "m", fidelity, control)


@pytest.mark.parametrize("fidelity,control,expected", [
    ("preserved", "use shacl", "warning"), ("preserved", None, "warning"),
    ("approximated", None, "translated"), ("approximated", "keep the hierarchy in owl", "runtime-mitigated"),
    ("unenforceable", "validate at runtime", "runtime-mitigated"), ("unenforceable", None, "unsupported"),
])
def test_outcome_is_derived_from_fidelity_and_compensating_control(fidelity, control, expected):
    assert _diag(fidelity, control).outcome == expected


@pytest.mark.parametrize("name", MODELS)
def test_every_diagnostic_has_a_loss_outcome_and_a_known_rule(name):
    diagnostics = compile_model(load_model(ROOT / "ontology" / "models" / f"{name}.yaml")).diagnostics
    assert diagnostics
    for item in diagnostics:
        assert item.outcome in OUTCOMES and item.outcome != "enforced", "a loss report never says enforced"
        assert item.code in CODE_RULE, f"{item.code} is not mapped to a rule kind"


def test_extension_diagnostics_are_mapped_and_cover_every_new_construct(ext):
    codes = {item.code for item in compile_model(ext).diagnostics}
    assert {"SHACL_COMPOSITE_KEY_RUNTIME", "NEO4J_QUALIFIED_CARDINALITY_RUNTIME", "OWL_OPEN_WORLD_MINIMUM_QUALIFIED",
            "OWL_RELATION_PROPERTY_REIFIED", "SHACL_RELATION_PROPERTY_REIFIED", "NEO4J_RELATION_PROPERTY_RUNTIME",
            "GQL_COMPOSITE_KEY_UNSUPPORTED", "GQL_CARDINALITY_UNSUPPORTED",
            "GQL_INHERITANCE_FLATTENED"} <= codes
    assert codes <= set(CODE_RULE)


# -- capability matrix ---------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_matrix_covers_every_rule_and_target_and_never_exceeds_the_diagnostics(name):
    model = load_model(ROOT / "ontology" / "models" / f"{name}.yaml")
    compiled = compile_model(model)
    targets = tuple(target_capabilities(model))
    matrix = build_capability_matrix(model, compiled.diagnostics, targets)
    assert matrix == build_capability_matrix(model, compiled.diagnostics, targets), "deterministic"
    assert matrix["targets"] == [t.value for t in targets]
    assert matrix["rules"]
    for row in matrix["rules"]:
        assert set(row["outcomes"]) == set(matrix["targets"])
        assert set(row["outcomes"].values()) <= set(OUTCOMES)
    for target, counts in matrix["summary"].items():
        assert sum(counts.values()) == len(matrix["rules"]), target
    by_cell = {(row["element"], row["rule"]): row["outcomes"] for row in matrix["rules"]}
    for item in compiled.diagnostics:
        cell = by_cell.get((item.element, CODE_RULE[item.code]))
        assert cell is not None, f"{item.code} on {item.element} has no matrix row"
        assert cell[item.target.value] == item.outcome


def test_matrix_gives_a_positive_record_where_the_loss_report_is_silent(ext):
    model = ext
    matrix = build_capability_matrix(model, compile_model(model).diagnostics, tuple(target_capabilities(model)))
    enforced = [row for row in matrix["rules"] if "enforced" in row["outcomes"].values()]
    assert enforced and not any(
        item.outcome == "enforced" for item in compile_model(model).diagnostics)


@pytest.mark.parametrize("name", MODELS)
def test_committed_artifacts_including_matrix_and_gql_are_not_stale(name):
    paths = compile_to_disk(ROOT / "ontology" / "models" / f"{name}.yaml", check=True, fail_on_error=True)
    assert "capability_matrix" in paths and "gql" in paths
    assert json.loads(paths["capability_matrix"].read_text(encoding="utf-8"))["model"]


def test_models_without_a_gql_artifact_compile_exactly_as_before(tmp_path):
    path = tmp_path / "ontology" / "models" / "ext.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(EXT_YAML.replace("  gql: out/e.gql\n", ""), encoding="utf-8")
    model = load_model(path)
    compiled = compile_model(model)
    assert compiled.gql == "" and Target.GQL not in target_capabilities(model)
    assert not any(item.target == Target.GQL for item in compiled.diagnostics)


# -- composite keys --------------------------------------------------------------


def test_composite_key_is_emitted_for_owl_and_neo4j_and_reported_for_shacl(ext):
    compiled = compile_model(ext)
    graph = Graph().parse(data=compiled.owl, format="turtle")
    key_lists = [Collection(graph, head) for head in graph.objects(NS.Product, OWL.hasKey)]
    assert [list(keys) for keys in key_lists] == [[NS.tenant, NS.sku]]
    assert "REQUIRE (n.tenant, n.sku) IS UNIQUE" in compiled.neo4j
    assert any(item.code == "SHACL_COMPOSITE_KEY_RUNTIME" and item.target == Target.SHACL
               for item in compiled.diagnostics)


@pytest.mark.parametrize("fragment,message", [
    ("    composite_keys: [[tenant]]\n", "at least two distinct"),
    ("    composite_keys: [[tenant, tenant]]\n", "at least two distinct"),
    ("    composite_keys: [[tenant, nope]]\n", "unknown property nope"),
])
def test_invalid_composite_keys_fail_closed(tmp_path, fragment, message):
    assert message in _invalid(tmp_path, fragment)


def test_composite_key_members_must_be_required(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(EXT_YAML.replace("sku: {datatype: string, required: true}", "sku: {datatype: string}"),
                    encoding="utf-8")
    with pytest.raises(SemanticModelError, match="member sku must be required"):
        load_model(path)


# -- qualified cardinalities -------------------------------------------------------


def _data():
    g = Graph()
    g.add((NS.SafetyComponent, RDFS.subClassOf, NS.Component))
    g.add((NS.PlainComponent, RDFS.subClassOf, NS.Component))
    g.add((NS.p1, RDF.type, NS.Product))
    g.add((NS.p1, NS.tenant, Literal("t")))
    g.add((NS.p1, NS.sku, Literal("S1")))
    return g


def _component(g, node, cls):
    g.add((node, RDF.type, cls))
    g.add((node, NS.partNo, Literal(str(node))))


def _conforms(tmp_path, ext, graph):
    shapes = tmp_path / "shapes.ttl"
    shapes.write_text(compile_model(ext).shacl, encoding="utf-8")
    return SHACLValidator(graph, shapes_path=shapes).validate_report(target="test").conforms


def test_qualified_cardinality_is_emitted_for_owl(ext):
    owl = compile_model(ext).owl
    assert "owl:onClass energy:SafetyComponent ; owl:minQualifiedCardinality" in owl
    assert "owl:onClass energy:SafetyComponent ; owl:maxQualifiedCardinality" in owl
    Graph().parse(data=owl, format="turtle")


def test_generated_shacl_enforces_qualified_minimum_and_maximum(tmp_path, ext):
    ok = _data()
    _component(ok, NS.c1, NS.SafetyComponent)
    ok.add((NS.p1, NS.contains, NS.c1))
    assert _conforms(tmp_path, ext, ok)

    only_plain = _data()
    _component(only_plain, NS.c1, NS.PlainComponent)
    only_plain.add((NS.p1, NS.contains, NS.c1))
    assert not _conforms(tmp_path, ext, only_plain), "no SafetyComponent: qualified minimum violated"

    too_many = _data()
    for node in (NS.c1, NS.c2, NS.c3):
        _component(too_many, node, NS.SafetyComponent)
        too_many.add((NS.p1, NS.contains, node))
    assert not _conforms(tmp_path, ext, too_many), "three SafetyComponents: qualified maximum violated"


def test_runtime_validator_enforces_the_qualified_maximum(ext):
    validator = SemanticMutationValidator(ext)
    ok = validator.validate_relation("contains", "Product", "SafetyComponent", tenant="t",
                                     existing_by_target_type={"SafetyComponent": 1, "PlainComponent": 5})
    assert ok == []
    full = validator.validate_relation("contains", "Product", "SafetyComponent", tenant="t",
                                       existing_by_target_type={"SafetyComponent": 2})
    assert [v.code for v in full] == ["QUALIFIED_MAX_CARDINALITY"]
    other_class = validator.validate_relation("contains", "Product", "PlainComponent", tenant="t",
                                              existing_by_target_type={"SafetyComponent": 2})
    assert other_class == [], "the qualifier only counts its own class"


def test_qualified_cardinality_must_name_a_subtype_of_the_target(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(EXT_YAML.replace("on_class: SafetyComponent", "on_class: Product"), encoding="utf-8")
    with pytest.raises(SemanticModelError, match="not a subtype of target"):
        load_model(path)
    path.write_text(EXT_YAML.replace("on_class: SafetyComponent", "on_class: Nope"), encoding="utf-8")
    with pytest.raises(SemanticModelError, match="unknown qualified class"):
        load_model(path)
    path.write_text(EXT_YAML.replace("cardinality: {min: 1, max: 2}", "cardinality: {}"), encoding="utf-8")
    with pytest.raises(SemanticModelError, match="sets no bound"):
        load_model(path)


# -- relationship attributes ---------------------------------------------------------


def test_relation_attributes_are_reified_in_owl_and_shacl(ext):
    compiled = compile_model(ext)
    owl = Graph().parse(data=compiled.owl, format="turtle")
    assert (NS.ContainsStatement, RDFS.subClassOf, RDF.Statement) in owl
    assert (NS.quantity, RDFS.domain, NS.ContainsStatement) in owl
    assert (NS.quantity, RDFS.range, XSD.integer) in owl


def test_generated_shacl_validates_a_reified_relationship_statement(tmp_path, ext):
    def statement(quantity):
        g = _data()
        _component(g, NS.c1, NS.SafetyComponent)
        g.add((NS.p1, NS.contains, NS.c1))
        g.add((NS.s1, RDF.type, NS.ContainsStatement))
        g.add((NS.s1, RDF.subject, NS.p1))
        g.add((NS.s1, RDF.object, NS.c1))
        if quantity is not None:
            g.add((NS.s1, NS.quantity, quantity))
        return g

    assert _conforms(tmp_path, ext, statement(Literal(2)))
    assert not _conforms(tmp_path, ext, statement(None)), "required attribute missing"
    assert not _conforms(tmp_path, ext, statement(Literal("two"))), "wrong datatype"


def test_runtime_validator_checks_relationship_attributes(ext):
    validator = SemanticMutationValidator(ext)
    assert validator.validate_relation_properties("contains", {"quantity": 2}, tenant="t") == []

    def codes(props):
        return sorted(v.code for v in validator.validate_relation_properties("contains", props, tenant="t"))

    assert codes({}) == ["REQUIRED_RELATION_PROPERTY"]
    assert codes({"quantity": "2"}) == ["INVALID_DATATYPE"]
    assert codes({"quantity": True}) == ["INVALID_DATATYPE"]
    assert codes({"quantity": 2, "colour": "red"}) == ["UNKNOWN_RELATION_PROPERTY"]
    assert [v.code for v in validator.validate_relation_properties("nope", {}, tenant="t")] == ["UNKNOWN_RELATION"]
    assert "TENANT_REQUIRED" in codes_for(validator, tenant="")


def codes_for(validator, *, tenant):
    return [v.code for v in validator.validate_relation_properties("contains", {"quantity": 1}, tenant=tenant)]


def test_relation_attribute_names_must_not_reuse_other_names(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(EXT_YAML.replace("quantity: {datatype: integer, required: true}",
                                     "sku: {datatype: integer}"), encoding="utf-8")
    with pytest.raises(SemanticModelError, match="reuses a property or relation name"):
        load_model(path)


# -- GQL graph type ----------------------------------------------------------------------


def test_gql_rendering_is_deterministic_and_expands_abstract_endpoints(ext):
    gql = render_gql(ext)
    assert gql == render_gql(ext) == compile_model(ext).gql
    assert gql.count("{") == gql.count("}")
    assert "CREATE GRAPH TYPE ext_type AS {" in gql
    assert "(Product :PRODUCT {" in gql and "sku STRING NOT NULL" in gql
    assert "(Component :" not in gql, "abstract types are not instantiable"
    assert "(SafetyComponent :SAFETY_COMPONENT {" in gql and "partNo STRING NOT NULL" in gql
    assert "(:PRODUCT)-[contains_Product_SafetyComponent :CONTAINS {note STRING, quantity INT64 NOT NULL}]->(:SAFETY_COMPONENT)" in gql
    assert "contains_Product_PlainComponent" in gql


@pytest.mark.parametrize("name", MODELS)
def test_committed_gql_is_balanced_and_names_every_concrete_type(name):
    model = load_model(ROOT / "ontology" / "models" / f"{name}.yaml")
    gql = render_gql(model)
    assert gql.count("{") == gql.count("}") and gql.count("(") == gql.count(")")
    for type_name, spec in model.types.items():
        assert (f"({type_name} :" in gql) == (not spec.abstract)
