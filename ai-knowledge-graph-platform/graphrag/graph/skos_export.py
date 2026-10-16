"""Standalone SKOS concept scheme from a domain ontology YAML (``config/ontologies/*.yml``).

The type hierarchy becomes ``skos:broader`` / ``skos:narrower`` between concepts; roots become top concepts. A
``vocabulary`` section, when the ontology has one, adds ``skos:definition`` and ``skos:altLabel`` (synonyms). The
output needs no graph and no tenant, so a taxonomy can be exchanged with tools that know nothing about this platform.
Equivalence mappings to other schemes (``skos:exactMatch`` / ``closeMatch``) are not invented here: pass them in
``mappings`` explicitly.

    python -m graphrag.graph.skos_export config/ontologies/automotive_iatf.yml out.ttl
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, SKOS

from graphrag.graph.domain_ontology import get_type_hierarchy_pairs, get_vocabulary, load_domain_ontology

TAXONOMY_BASE = "https://graphrag.example.com/taxonomy/"
_MATCH_PREDICATES = {"exactMatch": SKOS.exactMatch, "closeMatch": SKOS.closeMatch,
                     "broadMatch": SKOS.broadMatch, "narrowMatch": SKOS.narrowMatch, "relatedMatch": SKOS.relatedMatch}


def _label(term: str) -> str:
    return term.replace("_", " ").capitalize()


def ontology_to_skos(ontology: dict, *, mappings: dict[str, list[tuple[str, str]]] | None = None) -> Graph:
    """Build the SKOS graph. ``mappings`` is ``{"exactMatch": [(TYPE, external_uri), ...], ...}``."""
    ontology_id = str((ontology.get("ontology") or {}).get("id") or "ontology")
    ns = Namespace(f"{TAXONOMY_BASE}{quote(ontology_id, safe='')}#")
    scheme = URIRef(f"{TAXONOMY_BASE}{quote(ontology_id, safe='')}")
    g = Graph()
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)
    g.bind("tax", ns)

    g.add((scheme, RDF.type, SKOS.ConceptScheme))
    g.add((scheme, SKOS.prefLabel, Literal(ontology_id)))
    version = (ontology.get("ontology") or {}).get("version")
    if version:
        g.add((scheme, OWL.versionInfo, Literal(str(version))))

    pairs = get_type_hierarchy_pairs(ontology)
    vocabulary = get_vocabulary(ontology)
    terms = {t for pair in pairs for t in pair} | set(vocabulary)
    has_parent = {child for child, _ in pairs}

    def concept(term: str) -> URIRef:
        return ns[quote(term, safe="")]

    for term in sorted(terms):
        c = concept(term)
        g.add((c, RDF.type, SKOS.Concept))
        g.add((c, SKOS.inScheme, scheme))
        g.add((c, SKOS.prefLabel, Literal(_label(term), lang="en")))
        entry = vocabulary.get(term, {})
        if entry.get("definition"):
            g.add((c, SKOS.definition, Literal(str(entry["definition"]), lang="en")))
        for synonym in entry.get("synonyms") or []:
            g.add((c, SKOS.altLabel, Literal(str(synonym), lang="en")))
        if term not in has_parent:
            g.add((scheme, SKOS.hasTopConcept, c))
            g.add((c, SKOS.topConceptOf, scheme))

    for child, parent in pairs:
        g.add((concept(child), SKOS.broader, concept(parent)))
        g.add((concept(parent), SKOS.narrower, concept(child)))

    for kind, entries in (mappings or {}).items():
        predicate = _MATCH_PREDICATES.get(kind)
        if predicate is None:
            raise ValueError(f"unknown SKOS mapping kind {kind!r}; expected one of {sorted(_MATCH_PREDICATES)}")
        for term, external in entries:
            if term.upper() not in terms:
                raise ValueError(f"mapping names {term!r}, which is not a type in this ontology")
            g.add((concept(term.upper()), predicate, URIRef(external)))
    return g


def export_skos(yaml_path: str | Path, out_path: str | Path | None = None, *, rdf_format: str = "turtle") -> Graph:
    ontology = load_domain_ontology(yaml_path)
    if not ontology:
        raise FileNotFoundError(f"ontology not found or unreadable: {yaml_path}")
    g = ontology_to_skos(ontology)
    if out_path is not None:
        Path(out_path).write_text(g.serialize(format=rdf_format), encoding="utf-8")
    return g


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) < 2:
        sys.exit("usage: python -m graphrag.graph.skos_export <ontology.yml> [out.ttl]")
    graph = export_skos(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"{len(graph)} triples")
