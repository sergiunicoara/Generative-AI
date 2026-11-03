"""The semantic-model and shape registry as plain RDF.

Which generated model and shapes apply to which data graph is stored as
triples, not as store configuration, so the mapping travels with the data to
any SPARQL 1.1 store.  Each model is described with its id, version, the
SHA-256 of its canonical YAML and of its generated SHACL, the shapes artifact
path and an active flag; a data graph points at the model it conforms to with
``dcterms:conformsTo``.  The recorded shapes hash is computed from the same
compilation ``compile --check`` verifies, so a drifted shapes file shows up
as a hash mismatch.

    SELECT ?shapes WHERE {
      <graph> dcterms:conformsTo ?model . ?model reg:shapesPath ?shapes ; reg:active true }
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path

from rdflib import RDF, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, XSD

from graphrag.semantic_model.compiler import compile_model
from graphrag.semantic_model.models import SemanticModel, load_model

REG = Namespace("https://graphrag.example.com/ontology/registry#")
MODEL_BASE = "https://graphrag.example.com/ontology/model/"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def model_iri(model: SemanticModel) -> URIRef:
    return URIRef(f"{MODEL_BASE}{model.id}/{model.version}")


def registry_graph(
    model_paths: Iterable[Path], bindings: Mapping[str, str] | None = None,
    inactive: Iterable[str] = (),
) -> Graph:
    """Describe the models, and bind data-graph IRIs to model ids.

    ``bindings`` maps a data-graph IRI to the id of the model it conforms to;
    ``inactive`` lists model ids registered but not in force.
    """
    graph = Graph()
    graph.bind("reg", REG)
    graph.bind("dcterms", DCTERMS)
    by_id: dict[str, URIRef] = {}
    inactive = set(inactive)
    for path in sorted(Path(p) for p in model_paths):
        model = load_model(path)
        node = model_iri(model)
        by_id[model.id] = node
        graph.add((node, RDF.type, REG.SemanticModel))
        graph.add((node, REG.modelId, Literal(model.id)))
        graph.add((node, REG.version, Literal(model.version)))
        graph.add((node, REG.namespace, URIRef(model.namespace)))
        graph.add((node, REG.modelSha256, Literal(_sha256(path.read_text(encoding="utf-8")))))
        graph.add((node, REG.shapesSha256, Literal(_sha256(compile_model(model).shacl))))
        graph.add((node, REG.shapesPath, Literal(model.artifacts.shacl)))
        graph.add((node, REG.active, Literal(model.id not in inactive, datatype=XSD.boolean)))
    for data_graph, model_id in sorted((bindings or {}).items()):
        if model_id not in by_id:
            raise ValueError(f"binding names unknown model {model_id!r}")
        graph.add((URIRef(data_graph), DCTERMS.conformsTo, by_id[model_id]))
    return graph


SHAPES_FOR_GRAPH = """
PREFIX reg: <https://graphrag.example.com/ontology/registry#>
PREFIX dcterms: <http://purl.org/dc/terms/>
SELECT ?model ?shapes ?hash WHERE {
  ?graph dcterms:conformsTo ?model .
  ?model reg:shapesPath ?shapes ; reg:shapesSha256 ?hash ; reg:active true .
}"""


def shapes_for_graph(registry: Graph, data_graph: str) -> list[dict[str, str]]:
    rows = registry.query(SHAPES_FOR_GRAPH, initBindings={"graph": URIRef(data_graph)})
    return [{"model": str(r.model), "shapes": str(r.shapes), "sha256": str(r.hash)} for r in rows]


def shapes_drift(registry: Graph, root: Path) -> list[str]:
    """Registered shapes whose committed file no longer matches the recorded hash."""
    drift = []
    for model, path, recorded in registry.query(
            "SELECT ?m ?p ?h WHERE { ?m <%s> ?p ; <%s> ?h }" % (REG.shapesPath, REG.shapesSha256)):
        file = root / str(path)
        if not file.exists() or _sha256(file.read_text(encoding="utf-8")) != str(recorded):
            drift.append(str(model))
    return sorted(drift)


__all__ = ["MODEL_BASE", "REG", "SHAPES_FOR_GRAPH", "model_iri", "registry_graph", "shapes_drift", "shapes_for_graph"]
