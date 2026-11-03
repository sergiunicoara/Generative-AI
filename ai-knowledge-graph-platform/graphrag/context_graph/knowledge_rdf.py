"""RDF projection of published knowledge, validated by generated SHACL.

The authoritative model for this projection is the canonical semantic model
``ontology/models/contextual-knowledge.yaml``; its SHACL shapes are *generated*
by the semantic-model compiler (never hand-written here) and executed by the
existing :class:`SHACLValidator`.  The Pydantic contracts in ``knowledge.py``
stay the authority for the content the shapes cannot express: enumerated
values (modality, status, authority), applicability scopes, constraints and
access policies.  The projection carries identity, tenant, lifecycle, review
and provenance, which is what a publication gate and an RDF consumer need.

Conventions follow ``scripts/export_rdf.py``: instance IRIs are
tenant-scoped, ``https://graphrag.example.com/entity/<tenant>/<Type>/<key>``,
and every node also carries its tenant as a property.  Edges to entries that
are not part of the projected set (an unpublished or inaccessible target) are
omitted rather than invented.

Two entry points differ on purpose:

* :func:`items_to_graph` projects exactly what it is given.  It is for the
  publication gate, which sees the whole pack.
* :func:`export_graph` applies the evaluator's own visibility rules (tenant,
  published status, latest version, caller access) first, so nothing a caller
  could not see in a decision appears in an export.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import tempfile
from urllib.parse import quote

from rdflib import RDF, Graph, Literal, Namespace, URIRef
from rdflib.namespace import XSD

from graphrag.context_graph.knowledge import KnowledgeException, KnowledgeItem
from graphrag.context_graph.knowledge_evaluation import KnowledgeReleaseRef, visible_exceptions, visible_items
from graphrag.enterprise.models import AccessContext
from graphrag.semantic_model.compiler import compile_model
from graphrag.semantic_model.models import load_model

MODEL_PATH = Path(__file__).resolve().parents[2] / "ontology" / "models" / "contextual-knowledge.yaml"
_ENTITY_BASE = "https://graphrag.example.com/entity"
_MAX_REPORTED = 20


@lru_cache(maxsize=1)
def _model():
    return load_model(MODEL_PATH)


@lru_cache(maxsize=1)
def knowledge_shapes_ttl() -> str:
    """SHACL shapes generated from the canonical model (not hand-maintained)."""
    return compile_model(_model()).shacl


def _ns() -> Namespace:
    return Namespace(_model().namespace)


def _iri(tenant: str, kind: str, key: str) -> URIRef:
    return URIRef(f"{_ENTITY_BASE}/{quote(tenant, safe='')}/{kind}/{quote(key, safe='')}")


def _text(value) -> str:
    return str(getattr(value, "value", value))


def _when(value: datetime) -> Literal:
    return Literal(value.isoformat(), datatype=XSD.dateTime)


def items_to_graph(
    items: Iterable[KnowledgeItem], exceptions: Iterable[KnowledgeException] = (), *,
    release: KnowledgeReleaseRef | None = None,
) -> Graph:
    """Project items and exceptions to RDF, exactly as given.

    With ``release`` every projected entry points at a release node carrying
    the release id and content hash (the release stamp), so a holder of the
    export can check it against the active release.
    """
    k = _ns()
    graph = Graph()
    graph.bind("kn", k)
    items, exceptions = list(items), list(exceptions)
    release_node = None
    if release is not None:
        tenants = sorted({entry.tenant for entry in (*items, *exceptions)}) or ["_"]
        release_node = _iri(tenants[0], "KnowledgeRelease", release.release_id)
        graph.add((release_node, RDF.type, k.KnowledgeRelease))
        graph.add((release_node, k.releaseId, Literal(release.release_id)))
        graph.add((release_node, k.contentHash, Literal(release.content_hash)))
    node_of: dict[str, URIRef] = {}
    for item in items:
        node_of[item.id] = _iri(item.tenant, "KnowledgeItem", item.key)

    def governed(node: URIRef, entry, type_name: str) -> None:
        graph.add((node, RDF.type, k[type_name]))
        graph.add((node, k.knowledgeKey, Literal(entry.key)))
        graph.add((node, k.knowledgeId, Literal(entry.id)))
        graph.add((node, k.version, Literal(entry.version, datatype=XSD.integer)))
        graph.add((node, k.tenant, Literal(entry.tenant)))
        graph.add((node, k.status, Literal(_text(entry.status))))
        graph.add((node, k.recordedAt, _when(entry.recorded_at)))

    def cite(node: URIRef, tenant: str, relation, sources) -> None:
        for source in sources:
            source_node = _iri(tenant, "KnowledgeSource", source.source_id)
            graph.add((source_node, RDF.type, k.KnowledgeSource))
            graph.add((source_node, k.sourceId, Literal(source.source_id)))
            graph.add((node, relation, source_node))

    for item in items:
        node = node_of[item.id]
        governed(node, item, "KnowledgeItem")
        graph.add((node, k.title, Literal(item.title)))
        graph.add((node, k.statement, Literal(item.statement)))
        graph.add((node, k.subjectKey, Literal(item.subject_key)))
        graph.add((node, k.modality, Literal(_text(item.modality))))
        graph.add((node, k.authority, Literal(_text(item.authority))))
        if item.reviewed_by:
            graph.add((node, k.reviewedBy, Literal(item.reviewed_by)))
        if item.review_ref:
            graph.add((node, k.reviewRef, Literal(item.review_ref)))
        if item.applicability.valid_from:
            graph.add((node, k.validFrom, _when(item.applicability.valid_from)))
        if item.applicability.valid_to:
            graph.add((node, k.validTo, _when(item.applicability.valid_to)))
        cite(node, item.tenant, k.citesSource, item.sources)
        if release_node is not None:
            graph.add((node, k.inRelease, release_node))
        for target in item.supersedes:
            if target in node_of:
                graph.add((node, k.supersedes, node_of[target]))
        for target in item.overrides:
            if target in node_of:
                graph.add((node, k.overrides, node_of[target]))

    for exc in exceptions:
        node = _iri(exc.tenant, "KnowledgeException", exc.key)
        governed(node, exc, "KnowledgeException")
        graph.add((node, k.kind, Literal(_text(exc.kind))))
        graph.add((node, k.grantedBy, Literal(exc.granted_by)))
        graph.add((node, k.approvalRef, Literal(exc.approval_ref)))
        cite(node, exc.tenant, k.exceptionCitesSource, exc.sources)
        if release_node is not None:
            graph.add((node, k.exceptionInRelease, release_node))
        for target in exc.target_item_ids:
            if target in node_of:
                graph.add((node, k.exceptionTarget, node_of[target]))
        if exc.winner_item_id and exc.winner_item_id in node_of:
            graph.add((node, k.exceptionWinner, node_of[exc.winner_item_id]))
    return graph


def export_graph(
    items: Iterable[KnowledgeItem], exceptions: Iterable[KnowledgeException], *,
    tenant: str, access: AccessContext, release: KnowledgeReleaseRef | None = None,
) -> Graph:
    """Project only what ``access`` may see in ``tenant``, as the evaluator does."""
    return items_to_graph(
        visible_items(items, tenant=tenant, access=access),
        visible_exceptions(exceptions, tenant=tenant, access=access),
        release=release,
    )


def release_stamp(graph: Graph) -> tuple[str, str] | None:
    """The (release id, content hash) an export was made from, if stamped."""
    k = _ns()
    stamps = {(str(graph.value(node, k.releaseId)), str(graph.value(node, k.contentHash)))
              for node in graph.subjects(RDF.type, k.KnowledgeRelease)}
    return stamps.pop() if len(stamps) == 1 else None


def validate_projection(graph: Graph) -> list[str]:
    """SHACL violations of a projection against the generated shapes (empty = conforms).

    Fails closed: if SHACL cannot run, that is reported as a failure rather
    than treated as a pass.
    """
    from graphrag.graph.shacl_validator import SHACLValidator

    with tempfile.TemporaryDirectory() as tmp:
        shapes = Path(tmp) / "contextual-knowledge.shapes.ttl"
        shapes.write_text(knowledge_shapes_ttl(), encoding="utf-8")
        try:
            report = SHACLValidator(graph, shapes_path=shapes).validate_report(target="knowledge")
        except ImportError as exc:
            return [f"SHACL validation unavailable ({exc}); a release cannot be gated without it"]
    if report.conforms:
        return []
    problems = [
        f"{result.focus_node.rsplit('/', 1)[-1]}: {result.message}"
        + (f" ({result.result_path.rsplit('#', 1)[-1]})" if result.result_path else "")
        for result in report.results if result.severity == "Violation"
    ]
    problems = problems or ["SHACL validation did not conform"]
    if len(problems) > _MAX_REPORTED:
        problems = [*problems[:_MAX_REPORTED], f"... and {len(problems) - _MAX_REPORTED} more"]
    return problems


def projection_failures(items: Iterable[KnowledgeItem], exceptions: Iterable[KnowledgeException]) -> list[str]:
    """Publication-gate view: SHACL violations of the pack's published content."""
    return [f"SHACL: {problem}" for problem in validate_projection(items_to_graph(items, exceptions))]


__all__ = [
    "MODEL_PATH", "export_graph", "items_to_graph", "knowledge_shapes_ttl", "projection_failures",
    "release_stamp", "validate_projection",
]
