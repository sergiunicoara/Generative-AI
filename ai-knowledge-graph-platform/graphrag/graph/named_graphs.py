"""Named-graph layout for RDF stores, per-tenant deletion and JSON-LD.

Layout (see ``docs/adr/0013-named-graph-layout.md``)::

    https://graphrag.example.com/graph/{tenant}/{source}/{publication}

One named graph per tenant, source and publication version.  Tenant
boundaries, rollback and deletion then follow graph boundaries in the store:
dropping a tenant is ``DROP GRAPH`` on that tenant's graphs only, rolling back
is pointing consumers at an earlier publication graph, and provenance queries
use ``GRAPH ?g``.  Everything here is plain SPARQL 1.1 and rdflib, so the same
statements run against any SPARQL 1.1 store; locally they are exercised on an
in-memory ``rdflib.Dataset``.
"""

from __future__ import annotations

import json
import re
from urllib.parse import quote

from rdflib import Dataset, Graph, URIRef

GRAPH_BASE = "https://graphrag.example.com/graph/"
_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


def _segment(value: str, what: str) -> str:
    if not _SEGMENT.match(value or ""):
        raise ValueError(f"invalid {what} for a graph IRI: {value!r}")
    return quote(value, safe="")


def graph_iri(tenant: str, source: str, publication: str) -> URIRef:
    return URIRef(GRAPH_BASE + "/".join((_segment(tenant, "tenant"), _segment(source, "source"),
                                         _segment(publication, "publication"))))


def tenant_prefix(tenant: str) -> str:
    return GRAPH_BASE + _segment(tenant, "tenant") + "/"


def load_graph(dataset: Dataset, iri: URIRef, graph: Graph) -> None:
    target = dataset.graph(iri)
    for triple in graph:
        target.add(triple)


def tenant_graphs_query(tenant: str) -> str:
    """SPARQL 1.1 SELECT listing one tenant's named graphs."""
    return ("SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } "
            f"FILTER(STRSTARTS(STR(?g), {json.dumps(tenant_prefix(tenant))})) }}")


def drop_graphs_update(graphs: list[URIRef]) -> str:
    """SPARQL 1.1 Update dropping exactly these graphs (SPARQL cannot drop by
    prefix, so the graphs are listed first and dropped by name)."""
    return " ;\n".join(f"DROP SILENT GRAPH <{g}>" for g in sorted(graphs, key=str))


def drop_tenant(dataset: Dataset, tenant: str) -> list[URIRef]:
    """Remove every named graph of ``tenant``; returns what was dropped."""
    graphs = [row[0] for row in dataset.query(tenant_graphs_query(tenant))]
    if graphs:
        dataset.update(drop_graphs_update(graphs))
    return sorted(graphs, key=str)


def erase_resource(dataset: Dataset, tenant: str, iri: URIRef) -> int:
    """Delete every triple naming ``iri`` (as subject or object) in the
    tenant's graphs; other tenants' graphs are not touched.  Returns the
    number of triples removed."""
    removed = 0
    for row in dataset.query(tenant_graphs_query(tenant)):
        graph = dataset.graph(row[0])
        triples = [*graph.triples((iri, None, None)), *graph.triples((None, None, iri))]
        for triple in triples:
            graph.remove(triple)
        removed += len(triples)
    return removed


def to_jsonld(graph: Graph, *, prefixes: dict[str, str] | None = None) -> str:
    """JSON-LD for API consumers that cannot use Turtle (IRIs and datatypes kept)."""
    context = dict(prefixes or {name: str(ns) for name, ns in graph.namespaces() if name})
    return graph.serialize(format="json-ld", context=context, indent=2)


def from_jsonld(text: str) -> Graph:
    return Graph().parse(data=text, format="json-ld")


__all__ = [
    "GRAPH_BASE", "drop_graphs_update", "drop_tenant", "erase_resource", "from_jsonld", "graph_iri", "load_graph",
    "tenant_graphs_query", "tenant_prefix", "to_jsonld",
]
