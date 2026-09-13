"""Canonical semantic model, target compilation, drift, and runtime controls."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SH, XSD

from graphrag.graph.shacl_validator import SHACLValidator
from graphrag.core.models import Chunk, Entity, Relation
from graphrag.ingestion.graph_writer import GraphWriter
from graphrag.semantic_model import (
    TARGET_CAPABILITIES,
    Capability,
    MutationValidationError,
    SemanticModelError,
    SemanticMutationValidator,
    compile_model,
    compile_to_disk,
    load_model,
)
from graphrag.semantic_model.models import SemanticModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "ontology" / "models" / "energy-asset-intelligence.yaml"
ENERGY = Namespace("https://example.energy.demo/ontology#")


def _small_model(**overrides) -> SemanticModel:
    payload = {
        "id": "test", "version": "1.0.0", "namespace": "https://example.test/#",
        "label": "Test", "prefixes": {}, "unknown_property_policy": "reject",
        "mixins": {},
        "types": {
            "Parent": {"abstract": True, "properties": {"id": {"datatype": "string", "required": True, "key": True}}},
            "Child": {"extends": "Parent", "properties": {}},
        },
        "relations": {
            "hasParent": {"source": "Child", "target": "Parent", "cardinality": {"exact": 2}},
        },
        "artifacts": {"owl": "owl.ttl", "shacl": "shacl.ttl", "neo4j": "neo4j.cypher", "diagnostics": "diagnostics.json"},
    }
    payload.update(overrides)
    model = SemanticModel.model_validate(payload)
    return model


def test_energy_model_loads_inheritance_mixins_and_inherited_relations():
    model = load_model(MODEL_PATH)

    assert model.ancestors("WindTurbine") == ["Asset"]
    assert {"assetId", "createdAt", "updatedAt"} <= set(model.effective_properties("WindTurbine"))
    assert "hasComponent" in model.relations_for("WindTurbine")
    assert model.lpg_label("WindTurbine", model.types["WindTurbine"]) == "WIND_TURBINE"


@pytest.mark.parametrize("fragment", [
    "types:\n  A:\n    extends: Missing\n",
    "types:\n  A:\n    mixins: [Missing]\n",
    "types:\n  A:\n    extends: B\n  B:\n    extends: A\n",
])
def test_invalid_references_and_inheritance_cycles_fail(fragment, tmp_path):
    model = tmp_path / "bad.yaml"
    model.write_text(
        "id: bad\nversion: 1.0.0\nnamespace: 'https://bad/#'\nlabel: Bad\n"
        "prefixes: {}\nmixins: {}\n" + fragment +
        "relations: {}\nartifacts:\n  owl: a\n  shacl: b\n  neo4j: c\n  diagnostics: d\n",
        encoding="utf-8",
    )
    with pytest.raises(SemanticModelError):
        load_model(model)


def test_invalid_cardinality_fails_closed(tmp_path):
    model = tmp_path / "bad-cardinality.yaml"
    model.write_text(
        "id: bad\nversion: 1.0.0\nnamespace: 'https://bad/#'\nlabel: Bad\n"
        "prefixes: {}\nmixins: {}\ntypes:\n  A:\n    properties:\n"
        "      p:\n        datatype: string\n        cardinality: {min: 2, max: 1}\n"
        "relations: {}\nartifacts:\n  owl: a\n  shacl: b\n  neo4j: c\n  diagnostics: d\n",
        encoding="utf-8",
    )
    with pytest.raises(SemanticModelError, match="min cannot exceed max"):
        load_model(model)


def test_compilation_is_deterministic_parseable_and_generates_exact_cardinality():
    model = _small_model()
    first = compile_model(model)
    second = compile_model(model)

    assert first == second
    owl_graph = Graph().parse(data=first.owl, format="turtle")
    shacl_graph = Graph().parse(data=first.shacl, format="turtle")
    assert (ENERGY.Parent, RDF.type, OWL.Class) not in owl_graph  # model uses its own namespace
    assert any(predicate == OWL.cardinality and value == Literal(2, datatype=XSD.nonNegativeInteger)
               for _, predicate, value in owl_graph)
    assert any(predicate == SH.maxCount and int(value) == 2 for _, predicate, value in shacl_graph)
    assert "CREATE CONSTRAINT" in first.neo4j
    assert any(item.code == "NEO4J_RELATION_CARDINALITY_RUNTIME" for item in first.diagnostics)
    assert Capability.CARDINALITY in TARGET_CAPABILITIES[next(
        target for target in TARGET_CAPABILITIES if target.value == "shacl"
    )]


def test_generated_energy_owl_preserves_existing_core_vocabulary():
    compiled = compile_model(load_model(MODEL_PATH))
    graph = Graph().parse(data=compiled.owl, format="turtle")

    assert (ENERGY.Asset, RDF.type, OWL.Class) in graph
    assert (ENERGY.DocumentRevision, RDFS.subClassOf, ENERGY.TechnicalDocument) in graph
    assert (ENERGY.value, RDFS.range, XSD.decimal) in graph
    assert (ENERGY.observedAsset, RDF.type, OWL.ObjectProperty) in graph


def test_generated_shacl_is_executed_by_the_real_validator(tmp_path):
    compiled = compile_model(load_model(MODEL_PATH))
    shapes = tmp_path / "energy.shapes.ttl"
    shapes.write_text(compiled.shacl, encoding="utf-8")
    valid = Graph()
    valid.add((URIRef("urn:asset:WT-01"), RDF.type, ENERGY.Asset))
    valid.add((URIRef("urn:asset:WT-01"), ENERGY.assetId, Literal("WT-01")))
    invalid = Graph()
    invalid.add((URIRef("urn:asset:WT-02"), RDF.type, ENERGY.Asset))

    assert SHACLValidator(valid, shapes_path=shapes).validate_report(target="test").conforms is True
    assert SHACLValidator(invalid, shapes_path=shapes).validate_report(target="test").conforms is False


def test_diagnostics_are_machine_readable_located_and_explicit_about_loss():
    model = load_model(MODEL_PATH)
    diagnostics = compile_model(model).diagnostics

    abstract = next(item for item in diagnostics if item.code == "NEO4J_ABSTRACT_RUNTIME")
    assert abstract.line is not None
    assert abstract.model_path == MODEL_PATH.name
    assert abstract.fidelity == "unenforceable"
    assert abstract.runtime_control
    assert any(item.code == "OWL_OPEN_WORLD_MINIMUM" for item in diagnostics)


def test_compile_check_detects_drift_and_invalid_model_leaves_output_untouched(tmp_path):
    output = tmp_path / "generated"
    compile_to_disk(MODEL_PATH, output_dir=output)
    compile_to_disk(MODEL_PATH, output_dir=output, check=True)
    (output / "ontology.ttl").write_text("stale", encoding="utf-8")
    with pytest.raises(SemanticModelError, match="stale"):
        compile_to_disk(MODEL_PATH, output_dir=output, check=True)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("not: [a valid semantic model]", encoding="utf-8")
    sentinel = output / "sentinel"
    sentinel.write_text("unchanged", encoding="utf-8")
    with pytest.raises(SemanticModelError):
        compile_to_disk(invalid, output_dir=output)
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_fail_on_unenforceable_does_not_write_outputs(tmp_path):
    output = tmp_path / "forbidden"
    with pytest.raises(SemanticModelError, match="forbidden unenforceable semantics"):
        compile_to_disk(MODEL_PATH, output_dir=output, fail_on_unenforceable=True)
    assert not output.exists()


def test_runtime_validator_rejects_abstract_missing_unknown_and_bad_datatypes():
    validator = SemanticMutationValidator(load_model(MODEL_PATH))

    validator.require_node("ASSET", {"assetId": "WT-01"}, tenant="energy-demo")
    violations = validator.validate_node("ASSET", {"assetId": 1, "unexpected": "x"}, tenant="")
    assert {item.code for item in violations} == {"TENANT_REQUIRED", "INVALID_DATATYPE", "UNKNOWN_PROPERTY"}
    with pytest.raises(MutationValidationError) as exc:
        validator.require_node("TECHNICAL_DOCUMENT", {}, tenant="energy-demo")
    assert exc.value.as_dict()["violations"][0]["code"] == "ABSTRACT_TYPE"

    temporal = validator.validate_node(
        "OBSERVATION",
        # recordedAt supplied with a valid value: this case is isolating
        # datatype violations (value, observedAt), not the separate
        # required-property check recordedAt would otherwise also trigger
        # now that it's a required bitemporal property (see gap C's
        # recorded-time axis on Observation).
        {"value": True, "unit": "C", "observedAt": "not-a-date", "recordedAt": "2026-08-28T08:00:00Z"},
        tenant="energy-demo",
    )
    assert [item.code for item in temporal] == ["INVALID_DATATYPE", "INVALID_DATATYPE"]


def test_runtime_relation_validation_honours_inheritance_and_endpoint_types():
    validator = SemanticMutationValidator(load_model(MODEL_PATH))

    assert validator.validate_relation("HAS_COMPONENT", "WIND_TURBINE", "COMPONENT", tenant="energy-demo") == []
    violations = validator.validate_relation("HAS_COMPONENT", "SITE", "WORK_ORDER", tenant="energy-demo")
    assert {item.code for item in violations} == {"INVALID_SOURCE_TYPE", "INVALID_TARGET_TYPE"}


def test_runtime_cardinality_reports_non_atomic_limit():
    validator = SemanticMutationValidator(_small_model())
    violations = validator.validate_relation(
        "hasParent", "Child", "Parent", tenant="tenant-a", existing_count=2,
    )
    assert [item.code for item in violations] == ["MAX_CARDINALITY"]


@pytest.mark.asyncio
async def test_graph_writer_rejects_invalid_node_before_any_neo4j_mutation():
    writer = GraphWriter.__new__(GraphWriter)
    writer._semantic_validator = SemanticMutationValidator(load_model(MODEL_PATH))
    writer._ensure_registry = AsyncMock()
    writer._get_registry = MagicMock()
    writer._neo4j = AsyncMock()
    chunk = Chunk(document_id="doc-1", text="fixture", chunk_index=0, tenant="energy-demo")
    invalid = Entity(name="WT-01", type="ASSET", semantic_properties={})

    with pytest.raises(MutationValidationError, match="assetId"):
        await writer.write_entities([invalid], chunk)

    writer._neo4j.merge_entities_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_writer_rejects_invalid_relation_before_any_neo4j_mutation():
    writer = GraphWriter.__new__(GraphWriter)
    writer._semantic_validator = SemanticMutationValidator(load_model(MODEL_PATH))
    writer._ensure_registry = AsyncMock()
    writer._get_registry = MagicMock(return_value=MagicMock())
    writer._neo4j = AsyncMock()
    source = Entity(name="site", type="SITE")
    target = Entity(name="work order", type="WORK_ORDER")
    relation = Relation(
        source_entity_id=source.id,
        target_entity_id=target.id,
        relation="HAS_COMPONENT",
    )

    with pytest.raises(MutationValidationError, match="requires source type"):
        await writer.write_relations(
            [relation], {source.id: source, target.id: target}, tenant="energy-demo",
        )

    writer._neo4j.merge_relations_batch.assert_not_awaited()
