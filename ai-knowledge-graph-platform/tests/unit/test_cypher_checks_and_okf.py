"""SHACL-to-Cypher subset and the Open Knowledge Format export prototype
(evaluation items, 2026-11-03)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from rdflib import RDF, Graph, Namespace

from graphrag.context_graph.knowledge_release import load_pack
from graphrag.context_graph.okf_export import conformance_issues, export_okf, typed_edges
from graphrag.enterprise.models import AccessContext
from graphrag.semantic_model import compile_model, load_model
from graphrag.semantic_model.cypher_checks import compile_cypher_checks

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("energy-asset-intelligence", "automotive-iatf-quality", "contextual-knowledge")
SH = Namespace("http://www.w3.org/ns/shacl#")
PACK = ROOT / "config" / "knowledge_packs" / "packaging_labeling.yml"


@pytest.mark.parametrize("name", MODELS)
def test_checks_cover_every_generated_property_shape_of_a_concrete_type(name):
    model = load_model(ROOT / "ontology" / "models" / f"{name}.yaml")
    compiled = compile_model(model)
    checks = compile_cypher_checks(model)
    assert checks == compile_cypher_checks(model), "deterministic"
    shapes = Graph().parse(data=compiled.shacl, format="turtle")
    ns = Namespace(model.namespace)
    covered = {(c.rule, c.element) for c in checks}
    expected = set()
    for shape in shapes.subjects(RDF.type, SH.NodeShape):
        target = shapes.value(shape, SH.targetClass)
        type_name = str(target).replace(str(ns), "")
        if type_name not in model.types or model.types[type_name].abstract:
            continue
        for prop_shape in shapes.objects(shape, SH.property):
            path = str(shapes.value(prop_shape, SH.path)).replace(str(ns), "")
            if shapes.value(prop_shape, SH.datatype) is not None:
                expected.add(("datatype", f"{type_name}.{path}"))
                if shapes.value(prop_shape, SH.minCount) is not None:
                    expected.add(("required", f"{type_name}.{path}"))
            elif shapes.value(prop_shape, SH["class"]) is not None:
                expected.add(("relation_endpoint", f"{type_name}.{path}"))
                if shapes.value(prop_shape, SH.minCount) is not None:
                    expected.add(("relation_min", f"{type_name}.{path}"))
    assert expected and expected <= covered, sorted(expected - covered)
    digest = hashlib.sha256(compiled.shacl.encode("utf-8")).hexdigest()
    assert {c.shapes_sha256 for c in checks} == {digest}


@pytest.mark.parametrize("name", MODELS)
def test_every_check_is_read_only_and_parameter_free(name):
    for check in compile_cypher_checks(load_model(ROOT / "ontology" / "models" / f"{name}.yaml")):
        assert check.query.startswith("MATCH ") and "$" not in check.query
        assert " AS element" in check.query and " AS rule" in check.query


def test_keys_qualified_counts_and_composite_keys_are_checked(tmp_path):
    from tests.unit.test_semantic_model_extensions import EXT_YAML

    path = tmp_path / "ontology" / "models" / "ext.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(EXT_YAML, encoding="utf-8")
    rules = {(c.rule, c.element): c.query for c in compile_cypher_checks(load_model(path))}
    assert ("composite_key", "Product(tenant, sku)") in rules
    assert "COUNT { (n)-[:CONTAINS]->(m) WHERE m:SAFETY_COMPONENT } < 1" in rules[
        ("qualified_min", "Product.contains[SafetyComponent]")]
    assert ("qualified_max", "Product.contains[SafetyComponent]") in rules
    endpoint = rules[("relation_endpoint", "Product.contains")]
    assert "m:PLAIN_COMPONENT OR m:SAFETY_COMPONENT" in endpoint, "an abstract target expands to its subtypes"


# -- OKF export ---------------------------------------------------------------------------


def _bundle(claims=None):
    pack = load_pack(PACK)
    return pack, export_okf(pack.items, tenant=pack.tenant, access=AccessContext.from_claims(claims or {"sub": "u"}))


def test_the_bundle_conforms_to_the_okf_rules_it_relies_on():
    _, files = _bundle()
    assert "index.md" in files and conformance_issues(files) == []
    concepts = [path for path in files if path != "index.md"]
    assert concepts and all(path.startswith("concepts/") and path.endswith(".md") for path in concepts)
    assert all(f"({path})" in files["index.md"] for path in concepts)


def test_typed_edges_survive_the_round_trip():
    _, files = _bundle()
    assert ("concepts/PKG-011.md", "supersedes", "concepts/PKG-002.md") in typed_edges(files)
    assert "- supersedes: [PKG-002](PKG-002.md)" in files["concepts/PKG-011.md"], "plain link for OKF readers"


def test_only_visible_knowledge_is_exported_and_trust_fields_are_not_invented():
    pack, plain = _bundle()
    assert "concepts/PKG-015.md" not in plain, "restricted knowledge needs the reader's access"
    _, privileged = _bundle({"sub": "u", "groups": ["regulatory-affairs"]})
    assert "concepts/PKG-015.md" in privileged
    import yaml

    for item in pack.published_items():
        path = f"concepts/{item.id}.md"
        if path not in plain:
            continue
        meta = yaml.safe_load(plain[path].split("---\n")[1]) or {}
        human = bool(item.reviewed_by and item.review_ref and not item.reviewed_by.startswith("system:"))
        assert ("verified" in meta) == human, item.id
        assert ("stale_after" in meta) == (item.review_due is not None)
    assert not any(path.endswith(("PKG-013.md",)) and "verified" in content for path, content in plain.items())


def test_conformance_flags_missing_type_and_reserved_names():
    files = {"concepts/a.md": "---\ntitle: x\n---\n", "concepts/index.md": "---\ntype: t\n---\n",
             "concepts/b.md": "---\ntype: t\nstatus: final\n---\n"}
    issues = conformance_issues(files)
    assert any("must carry type" in i for i in issues)
    assert any("reserved name" in i for i in issues)
    assert any("status must be" in i for i in issues)
