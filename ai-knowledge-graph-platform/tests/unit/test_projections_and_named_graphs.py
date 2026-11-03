"""Release stamps on projections, named-graph layout with per-tenant deletion,
JSON-LD round trip, and the model/shape registry as RDF (2026-11-03)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.compare import isomorphic

from graphrag.context_graph.knowledge_evaluation import KnowledgeReleaseRef
from graphrag.context_graph.knowledge_rdf import export_graph, items_to_graph, release_stamp, validate_projection
from graphrag.context_graph.knowledge_release import load_cases, load_pack, publish_pack
from graphrag.context_graph.knowledge_service import KnowledgeEvaluationService
from graphrag.domains.energy.governance_store import GovernanceStore
from graphrag.enterprise.models import AccessContext
from graphrag.graph.named_graphs import (
    drop_graphs_update, drop_tenant, from_jsonld, graph_iri, load_graph, tenant_graphs_query, to_jsonld,
)
from graphrag.semantic_model import compile_model, load_model
from graphrag.semantic_model.registry_rdf import registry_graph, shapes_drift, shapes_for_graph

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "config" / "knowledge_packs" / "packaging_labeling.yml"
CASES = ROOT / "evals" / "knowledge_eval" / "packaging_labeling_cases.json"
MODELS = sorted((ROOT / "ontology" / "models").glob("*.yaml"))
TENANT, DOMAIN = "packaging_demo", "packaging_labeling"
EX = Namespace("https://example.test/")


# -- release stamps ---------------------------------------------------------------


def test_a_stamped_export_names_its_release_and_still_conforms():
    pack = load_pack(PACK)
    ref = KnowledgeReleaseRef(release_id="r1", domain=DOMAIN, version=1, content_hash="h1")
    graph = items_to_graph(pack.published_items(), pack.published_exceptions(), release=ref)
    assert release_stamp(graph) == ("r1", "h1")
    assert validate_projection(graph) == []
    assert release_stamp(items_to_graph(pack.published_items())) is None, "unstamped exports say so"


async def test_stale_projection_detection_across_publish_and_rollback(tmp_path):
    store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}", blob_root=tmp_path / "b")
    await store.open()
    try:
        cases, pack = load_cases(CASES), load_pack(PACK)
        first = await publish_pack(store, pack, published_by="user:p", cases=cases)
        service = KnowledgeEvaluationService(store)
        catalog = await service.catalog(TENANT, DOMAIN)
        export = export_graph(catalog.items, catalog.exceptions, tenant=TENANT,
                              access=AccessContext.from_claims({"sub": "u"}), release=catalog.ref)
        stamps = {"rdf_export": release_stamp(export)[1], "decision": catalog.ref.content_hash}
        assert await service.stale_projections(TENANT, DOMAIN, stamps) == []

        await publish_pack(store, pack.model_copy(update={"disclaimer": "revised"}), published_by="user:p",
                           cases=cases)
        assert await service.stale_projections(TENANT, DOMAIN, stamps) == ["decision", "rdf_export"]

        await store.rollback_knowledge_release(TENANT, DOMAIN, first["release_id"], rolled_back_by="user:p")
        assert await service.stale_projections(TENANT, DOMAIN, stamps) == [], \
            "rolling back to identical content makes the projections current again"
    finally:
        await store.close()


# -- named graphs ---------------------------------------------------------------------


def test_graph_iris_are_validated_and_scoped_by_tenant_source_and_publication():
    iri = graph_iri("tenant-a", "sap", "v3")
    assert str(iri) == "https://graphrag.example.com/graph/tenant-a/sap/v3"
    for bad in ("", "../x", "a/b", "a b"):
        with pytest.raises(ValueError):
            graph_iri(bad, "sap", "v1")


def _dataset() -> Dataset:
    dataset = Dataset()
    for tenant in ("tenant-a", "tenant-b"):
        for publication in ("v1", "v2"):
            graph = Graph()
            for n in range(3):
                graph.add((EX[f"{tenant}/{publication}/{n}"], EX.value, Literal(n)))
            load_graph(dataset, graph_iri(tenant, "sap", publication), graph)
    return dataset


def _triples(dataset: Dataset, tenant: str) -> int:
    return sum(len(dataset.graph(graph_iri(tenant, "sap", p))) for p in ("v1", "v2"))


def test_dropping_a_tenant_removes_only_that_tenants_graphs():
    dataset = _dataset()
    assert _triples(dataset, "tenant-a") == _triples(dataset, "tenant-b") == 6
    dropped = drop_tenant(dataset, "tenant-a")
    assert [str(g) for g in dropped] == [str(graph_iri("tenant-a", "sap", p)) for p in ("v1", "v2")]
    assert _triples(dataset, "tenant-a") == 0
    assert _triples(dataset, "tenant-b") == 6
    assert list(dataset.query(tenant_graphs_query("tenant-a"))) == []
    assert drop_tenant(dataset, "tenant-a") == [], "dropping again is a no-op"


def test_the_drop_statement_names_each_graph_explicitly():
    update = drop_graphs_update([graph_iri("t", "s", "v2"), graph_iri("t", "s", "v1")])
    assert update.splitlines() == ["DROP SILENT GRAPH <https://graphrag.example.com/graph/t/s/v1> ;",
                                   "DROP SILENT GRAPH <https://graphrag.example.com/graph/t/s/v2>"]


def test_a_tenant_prefix_does_not_match_a_longer_tenant_name():
    dataset = _dataset()
    load_graph(dataset, graph_iri("tenant-ab", "sap", "v1"), Graph().parse(data="<urn:x> <urn:y> 1 .", format="turtle"))
    drop_tenant(dataset, "tenant-a")
    assert len(dataset.graph(graph_iri("tenant-ab", "sap", "v1"))) == 1


def test_jsonld_round_trip_preserves_iris_and_datatypes():
    pack = load_pack(PACK)
    ref = KnowledgeReleaseRef(release_id="r1", domain=DOMAIN, version=1, content_hash="h1")
    graph = items_to_graph(pack.published_items(), pack.published_exceptions(), release=ref)
    restored = from_jsonld(to_jsonld(graph))
    assert isomorphic(graph, restored)
    datatypes = {o.datatype for o in restored.objects() if isinstance(o, Literal) and o.datatype}
    assert {URIRef("http://www.w3.org/2001/XMLSchema#integer"),
            URIRef("http://www.w3.org/2001/XMLSchema#dateTime")} <= datatypes


# -- registry ---------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def test_registry_answers_which_shapes_apply_to_a_graph():
    data = str(graph_iri(TENANT, "knowledge", "v1"))
    registry = registry_graph(MODELS, {data: "contextual-knowledge"})
    rows = shapes_for_graph(registry, data)
    knowledge = load_model(ROOT / "ontology" / "models" / "contextual-knowledge.yaml")
    assert rows == [{"model": f"https://graphrag.example.com/ontology/model/contextual-knowledge/{knowledge.version}",
                     "shapes": knowledge.artifacts.shacl, "sha256": _sha(compile_model(knowledge).shacl)}]
    assert shapes_for_graph(registry, "https://graphrag.example.com/graph/other/x/v1") == []


def test_registry_hash_matches_the_committed_shapes_and_detects_drift(tmp_path):
    registry = registry_graph(MODELS)
    assert shapes_drift(registry, ROOT) == []
    for model in MODELS:
        spec = load_model(model)
        target = tmp_path / spec.artifacts.shacl
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / spec.artifacts.shacl).read_text(encoding="utf-8"), encoding="utf-8")
    edited = tmp_path / load_model(MODELS[0]).artifacts.shacl
    edited.write_text(edited.read_text(encoding="utf-8") + "\n# hand edit\n", encoding="utf-8")
    assert len(shapes_drift(registry, tmp_path)) == 1


def test_inactive_models_and_unknown_bindings():
    data = "https://graphrag.example.com/graph/t/k/v1"
    registry = registry_graph(MODELS, {data: "contextual-knowledge"}, inactive=["contextual-knowledge"])
    assert shapes_for_graph(registry, data) == []
    with pytest.raises(ValueError, match="unknown model"):
        registry_graph(MODELS, {data: "nope"})
    assert from_jsonld(to_jsonld(registry)), "the registry is plain RDF and serialises like any graph"
