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
    render_erd_mermaid,
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


def test_compiler_is_domain_general_not_energy_specific():
    """Roadmap "P1 -- reusable domain onboarding and governance pack", bullet 4:
    "Demonstrate a second domain reusing the compiler... without modifying
    Energy code." `ontology/models/automotive-iatf-quality.yaml` re-expresses
    the vocabulary already declared in `config/ontologies/automotive_iatf.yml`
    (IATF 16949 quality management: suppliers, audits, nonconformities,
    quality documents) as an Energy-schema semantic model, compiled through
    the exact same `compile_model()`/`load_model()` used for Energy -- no
    Energy-specific branch or flag involved.
    """
    automotive_path = ROOT / "ontology" / "models" / "automotive-iatf-quality.yaml"
    model = load_model(automotive_path)
    compiled = compile_model(model)

    owl_graph = Graph().parse(data=compiled.owl, format="turtle")
    automotive_ns = Namespace("https://example.automotive.demo/ontology#")
    assert (automotive_ns.Supplier, RDF.type, OWL.Class) in owl_graph
    assert (automotive_ns.Nonconformity, RDF.type, OWL.Class) in owl_graph
    assert "CREATE CONSTRAINT" in compiled.neo4j
    assert not any("energy" in d.message.lower() for d in compiled.diagnostics), (
        "a compiler diagnostic mentioned Energy while compiling a non-Energy model -- "
        "a hardcoded-wording regression"
    )


def test_generated_energy_owl_preserves_existing_core_vocabulary():
    compiled = compile_model(load_model(MODEL_PATH))
    graph = Graph().parse(data=compiled.owl, format="turtle")

    assert (ENERGY.Asset, RDF.type, OWL.Class) in graph
    assert (ENERGY.DocumentRevision, RDFS.subClassOf, ENERGY.TechnicalDocument) in graph
    assert (ENERGY.value, RDFS.range, XSD.decimal) in graph
    assert (ENERGY.observedAsset, RDF.type, OWL.ObjectProperty) in graph


def test_generated_owl_round_trips_through_rdflib_without_losing_identifiers_or_annotations(tmp_path):
    """Roadmap "P1 -- visual and tool-friendly semantic modelling", bullet 2:
    "Export OWL in a Protégé-compatible form and validate that the generated
    ontology opens and round-trips without losing identifiers or
    annotations."

    Evidence-level caveat, stated plainly: this environment has no Protégé
    (a desktop Java application) to actually open the file in -- this
    validates the closest available proxy, an rdflib parse -> serialize ->
    reparse round trip, which exercises the same RDF/OWL/Turtle parsing
    rules Protégé's own OWL API is built on, but is not proof Protégé
    itself opens the file cleanly. That remains a manual verification step
    for whoever has Protégé installed.
    """
    from rdflib.compare import isomorphic

    compiled = compile_model(load_model(MODEL_PATH))
    first = Graph().parse(data=compiled.owl, format="turtle")

    reserialized = first.serialize(format="turtle")
    second = Graph().parse(data=reserialized, format="turtle")

    assert len(first) == len(second)
    # Graph *isomorphism*, not raw triple-set equality: blank-node identifiers
    # (owl:Restriction cardinality nodes, owl:hasKey's RDF list) are not
    # stable across a serialize -> reparse round trip -- that's normal RDF
    # semantics, not data loss. `isomorphic` compares graph structure
    # up to blank-node relabeling, which is the actually-meaningful check.
    assert isomorphic(first, second)

    classes = set(first.subjects(RDF.type, OWL.Class))
    assert classes, "no owl:Class declared -- round trip would be vacuous"
    for cls in classes:
        assert (cls, RDF.type, OWL.Class) in second, f"lost class identifier: {cls}"
    labels = set(first.subjects(RDFS.label, None))
    assert labels, "no rdfs:label declared -- round trip would be vacuous"
    for subject in labels:
        assert set(first.objects(subject, RDFS.label)) == set(second.objects(subject, RDFS.label)), \
            f"lost or changed rdfs:label annotation on {subject}"


class TestErdGeneration:
    """Roadmap "P1 -- visual and tool-friendly semantic modelling", bullet 1:
    "Generate a reviewable ERD/frame view from the canonical YAML, including
    entities, slots/properties, inheritance, mixins, typed relations and
    cardinality." """

    def test_every_type_and_mixin_becomes_a_class_block(self):
        model = _small_model()
        erd = render_erd_mermaid(model)
        assert erd.startswith("classDiagram\n")
        assert "class Parent" in erd
        assert "class Child" in erd

    def test_abstract_types_are_stamped(self):
        erd = render_erd_mermaid(_small_model())
        assert "<<abstract>>" in erd

    def test_local_inheritance_is_a_uml_arrow(self):
        erd = render_erd_mermaid(_small_model())
        assert "Parent <|-- Child" in erd

    def test_external_namespaced_parents_become_a_comment_not_a_broken_arrow(self):
        """A colon in a Mermaid class name is invalid syntax; an `extends`
        value with a colon (e.g. `prov:Entity`) is a foreign/external type
        this model doesn't declare, not a local class to box."""
        model = _small_model(types={
            "Parent": {"abstract": True, "properties": {"id": {"datatype": "string", "required": True, "key": True}}},
            "Child": {"extends": "prov:ExternalThing", "properties": {}},
        })
        erd = render_erd_mermaid(model)
        assert "prov:ExternalThing <|--" not in erd
        assert "%% Child extends external type prov:ExternalThing" in erd

    def test_mixin_usage_is_a_realization_arrow(self):
        model = _small_model(
            mixins={"Audited": {"properties": {"auditedAt": {"datatype": "dateTime"}}}},
            types={
                "Parent": {"abstract": True, "properties": {"id": {"datatype": "string", "required": True, "key": True}}},
                "Child": {"extends": "Parent", "mixins": ["Audited"], "properties": {}},
            },
        )
        erd = render_erd_mermaid(model)
        assert "class Audited" in erd
        assert "<<mixin>>" in erd
        assert "Child ..|> Audited : mixin" in erd

    def test_relations_carry_name_and_cardinality(self):
        erd = render_erd_mermaid(_small_model())  # hasParent: exact 2
        assert 'Child "1" --> "2" Parent : hasParent' in erd

    def test_open_ended_cardinality_renders_as_min_dot_dot_star(self):
        model = _small_model(relations={
            "hasParent": {"source": "Child", "target": "Parent", "cardinality": {"min": 1}},
        })
        erd = render_erd_mermaid(model)
        assert '"1..*"' in erd

    def test_key_and_required_properties_are_marked(self):
        erd = render_erd_mermaid(_small_model())
        assert "+string id*" in erd  # key marker

    def test_erd_is_generated_for_the_real_energy_model_and_is_syntactically_plausible(self):
        erd = render_erd_mermaid(load_model(MODEL_PATH))
        assert erd.startswith("classDiagram\n")
        for line in erd.splitlines():
            stripped = line.strip()
            if stripped.startswith("class "):
                class_name = stripped[len("class "):].split(" ", 1)[0].rstrip("{").strip()
                assert ":" not in class_name, f"invalid Mermaid class name: {class_name!r}"
        assert erd.count("{") == erd.count("}")


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


def _capability_matrix_model() -> SemanticModel:
    """A small synthetic model deliberately exercising every rule category
    bullet 2 of the roadmap's "P0 — target capability matrix and loss
    diagnostics" item names: abstract types, mixins, plain inheritance,
    required/key/datatype/cardinality (min and max) properties, relation
    endpoints and cardinality, and the closed-world (`unknown_property_policy`)
    setting -- so `test_no_rule_disappears_silently_across_targets` below can
    assert none of them compiles to silence."""
    return SemanticModel.model_validate({
        "id": "matrix", "version": "1.0.0", "namespace": "https://example.matrix/#",
        "label": "Matrix", "prefixes": {}, "unknown_property_policy": "reject",
        "mixins": {"Audited": {"properties": {"auditedAt": {"datatype": "dateTime"}}}},
        "types": {
            "Base": {"abstract": True, "properties": {"id": {"datatype": "string", "required": True, "key": True}}},
            "Leaf": {
                "extends": "Base", "mixins": ["Audited"],
                "properties": {
                    "name": {"datatype": "string", "cardinality": {"max": 3}},
                    "status": {"datatype": "string", "required": True},
                },
            },
        },
        "relations": {
            "linksTo": {"source": "Leaf", "target": "Leaf", "cardinality": {"min": 1, "max": 5}},
        },
        "artifacts": {"owl": "owl.ttl", "shacl": "shacl.ttl", "neo4j": "neo4j.cypher", "diagnostics": "diagnostics.json"},
    })


def test_no_rule_disappears_silently_across_targets():
    """Cross-target completeness: every rule category the compiler tracks --
    abstractness, mixins, plain inheritance, key/required/datatype/cardinality
    (both a `min` and a `max`) properties, relation endpoints and cardinality,
    and the closed-world policy -- must produce a diagnostic for every target
    that can't enforce it, for both entries this fixture exercises (its own
    declared rule and one it inherits). A future compiler change that
    silently drops one of these code paths, or narrows an existing one so it
    stops firing for this fixture, fails this test rather than silently
    shipping a schema whose actual enforcement is weaker than what the
    generated diagnostics claim."""
    diagnostics = compile_model(_capability_matrix_model()).diagnostics
    codes = {item.code for item in diagnostics}

    assert codes == {
        # Abstractness: no target accepts a direct instance of `Base`.
        "OWL_ABSTRACT_RUNTIME", "SHACL_ABSTRACT_RUNTIME", "NEO4J_ABSTRACT_RUNTIME",
        # Mixin identity and plain single inheritance are both structurally
        # flattened outside OWL (which needs no diagnostic for either --
        # rdfs:subClassOf and property expansion both preserve it natively).
        "NEO4J_MIXIN_EXPANDED", "SHACL_INHERITANCE_FLATTENED", "NEO4J_INHERITANCE_FLATTENED",
        # Property rules: open-world minimum (OWL), datatype (Neo4j), graph-
        # wide key uniqueness (SHACL), a value-count maximum (Neo4j), and a
        # required-but-not-key property (Neo4j).
        "OWL_OPEN_WORLD_MINIMUM", "NEO4J_DATATYPE_RUNTIME", "SHACL_KEY_RUNTIME",
        "NEO4J_PROPERTY_CARDINALITY_RUNTIME", "NEO4J_REQUIRED_PROPERTY_RUNTIME",
        # Relation rules: endpoint typing and cardinality (Neo4j), and the
        # same open-world minimum caveat OWL has for properties.
        "NEO4J_RELATION_ENDPOINT_RUNTIME", "NEO4J_RELATION_CARDINALITY_RUNTIME",
        "OWL_OPEN_WORLD_MINIMUM_RELATION",
        # Closed-world validation: SHACL and Neo4j both stay schema-open.
        "SHACL_UNKNOWN_PROPERTY_RUNTIME", "NEO4J_UNKNOWN_PROPERTY_RUNTIME",
    }
    # Every one of those codes must also carry a human-readable explanation --
    # a silent diagnostic is as unhelpful as no diagnostic at all. (Location
    # attribution itself is already covered against the real, file-backed
    # model by test_diagnostics_are_machine_readable_located_and_explicit_about_loss;
    # this fixture has no source file to attribute to.)
    for item in diagnostics:
        assert item.message


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
