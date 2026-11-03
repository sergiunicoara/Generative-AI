"""RDF projection of published knowledge and its generated-SHACL gate."""

from __future__ import annotations

from pathlib import Path

from rdflib import RDF, Literal

from graphrag.context_graph.knowledge_rdf import (
    MODEL_PATH, export_graph, items_to_graph, knowledge_shapes_ttl, projection_failures, validate_projection,
)
from graphrag.context_graph.knowledge_release import check_pack, load_cases, load_pack
from graphrag.enterprise.models import AccessContext
from graphrag.semantic_model.compiler import compile_model
from graphrag.semantic_model.models import load_model

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "config" / "knowledge_packs" / "packaging_labeling.yml"
CASES = ROOT / "evals" / "knowledge_eval" / "packaging_labeling_cases.json"


def _pack():
    return load_pack(PACK)


def _graph():
    pack = _pack()
    return items_to_graph(pack.published_items(), pack.published_exceptions())


def test_shapes_are_generated_from_the_canonical_model_and_match_the_committed_file():
    committed = ROOT / "ontology" / "shapes" / "contextual-knowledge.shapes.ttl"
    assert committed.read_text(encoding="utf-8") == compile_model(load_model(MODEL_PATH)).shacl
    assert knowledge_shapes_ttl() == committed.read_text(encoding="utf-8")


def test_the_published_pack_projects_to_a_conforming_graph():
    assert validate_projection(_graph()) == []
    assert projection_failures(_pack().published_items(), _pack().published_exceptions()) == []


def test_projection_is_deterministic_and_tenant_scoped():
    first, second = _graph(), _graph()
    assert set(first) == set(second)
    items = {s for s in first.subjects(RDF.type, None) if "/KnowledgeItem/" in str(s)}
    assert items and all(str(s).startswith("https://graphrag.example.com/entity/packaging_demo/") for s in items)


def test_only_published_entries_are_projected():
    keys = {str(o) for o in _graph().objects(None, None) if isinstance(o, Literal)}
    assert "PKG-001@v1" in keys
    assert "PKG-013@v1" not in keys, "a draft is not projected"
    assert "PKG-014@v2" not in keys, "a withdrawn entry is not projected"


def test_a_published_item_without_a_cited_source_is_a_violation():
    graph = _graph()
    node = next(s for s in graph.subjects(RDF.type, None) if str(s).endswith("/KnowledgeItem/PKG-001%40v1"))
    cites = next(p for p in graph.predicates(node, None) if str(p).endswith("citesSource"))
    for obj in list(graph.objects(node, cites)):
        graph.remove((node, cites, obj))
    problems = validate_projection(graph)
    assert problems and any("PKG-001" in problem for problem in problems)


def test_a_missing_tenant_or_review_reference_is_a_violation():
    graph = _graph()
    node = next(s for s in graph.subjects(RDF.type, None) if str(s).endswith("/KnowledgeItem/PKG-001%40v1"))
    for predicate in list(graph.predicates(node, None)):
        if str(predicate).endswith(("#tenant", "#reviewRef")):
            for obj in list(graph.objects(node, predicate)):
                graph.remove((node, predicate, obj))
    assert len(validate_projection(graph)) >= 2


def test_two_versions_stated_for_one_entry_are_a_violation():
    graph = _graph()
    node = next(s for s in graph.subjects(RDF.type, None) if str(s).endswith("/KnowledgeItem/PKG-001%40v1"))
    version = next(p for p in graph.predicates(node, None) if str(p).endswith("#version"))
    graph.add((node, version, Literal(2)))
    assert validate_projection(graph)


def test_the_publication_gate_runs_the_shacl_projection(monkeypatch):
    import graphrag.context_graph.knowledge_release as release

    monkeypatch.setattr(release, "projection_failures", lambda items, exceptions: ["SHACL: injected violation"])
    assert "SHACL: injected violation" in check_pack(_pack(), load_cases(CASES))


def test_a_release_that_conforms_still_passes_the_gate():
    assert check_pack(_pack(), load_cases(CASES)) == []


def test_export_applies_the_evaluators_visibility_rules():
    pack = _pack()
    plain = AccessContext.from_claims({"sub": "u"})
    privileged = AccessContext.from_claims({"sub": "u", "groups": ["regulatory-affairs"]})
    restricted = "/KnowledgeItem/PKG-015%40v1"

    def subjects(access, tenant="packaging_demo"):
        graph = export_graph(pack.items, pack.exceptions, tenant=tenant, access=access)
        return {str(s) for s in graph.subjects(RDF.type, None)}

    assert not any(restricted in s for s in subjects(plain))
    assert any(restricted in s for s in subjects(privileged))
    assert subjects(plain, tenant="other_tenant") == set()


def test_edges_to_entries_outside_the_projection_are_omitted_not_invented():
    pack = _pack()
    only = [i for i in pack.published_items() if i.id == "PKG-011"]
    graph = items_to_graph(only, pack.published_exceptions())
    assert not any(str(o).endswith("/KnowledgeItem/PKG-001%40v1") for o in graph.objects(None, None))
