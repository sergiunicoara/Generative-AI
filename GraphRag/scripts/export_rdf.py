"""Export the knowledge graph to Turtle (RDF) using rdflib.

Produces a standards-compliant Turtle file that can be loaded into:
  - Protégé (ontology editor / OWL reasoner)
  - SPARQL endpoints (Apache Jena Fuseki, Oxigraph)
  - Reasoners: HermiT, Pellet, FaCT++
  - Any RDF-aware linked-data tool

Mapping
-------
  Entity nodes          → owl:NamedIndividual + rdf:type
  EntityType nodes      → owl:Class + rdfs:subClassOf hierarchy
  RELATES_TO edges      → owl:ObjectProperty assertions with reified confidence
  NEGATIVE_RELATES_TO   → annotated negative assertions
  SUBCLASS_OF edges     → rdfs:subClassOf
  confidence            → :confidence annotation (xsd:float)
  valid_from / valid_to → :validFrom / :validTo annotations

Uses rdflib for guaranteed valid Turtle output (handles unicode, quotes,
special characters that hand-rolled string concatenation cannot).

Usage
-----
  python scripts/export_rdf.py
  python scripts/export_rdf.py --tenant acme --output exports/graph.ttl
  python scripts/export_rdf.py --tenant default --limit 10000
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import structlog
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

log = structlog.get_logger(__name__)

# ── Namespaces ─────────────────────────────────────────────────────────────────

BASE  = Namespace("https://graphrag.example.com/ontology#")
INST  = Namespace("https://graphrag.example.com/entity/")
ANNOT = Namespace("https://graphrag.example.com/annotation#")


def _entity_uri(name: str, etype: str, tenant: str) -> URIRef:
    """Stable entity URI — tenant-scoped to prevent cross-tenant collisions."""
    def _safe(s: str) -> str:
        import urllib.parse
        return urllib.parse.quote(s, safe="")
    return INST[f"{_safe(tenant)}/{_safe(etype)}/{_safe(name)}"]


def _type_uri(etype: str) -> URIRef:
    return BASE[etype.upper()]


def _rel_uri(relation: str) -> URIRef:
    return BASE[relation.upper()]


def _axiom_uri(s_name: str, rel: str, t_name: str) -> URIRef:
    import hashlib
    key = f"{s_name}|{rel}|{t_name}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return INST[f"axiom/{h}"]


# ── Graph builder ──────────────────────────────────────────────────────────────

def _init_graph() -> Graph:
    g = Graph()
    g.bind("base",  BASE)
    g.bind("inst",  INST)
    g.bind("annot", ANNOT)
    g.bind("owl",   OWL)
    g.bind("rdf",   RDF)
    g.bind("rdfs",  RDFS)
    g.bind("xsd",   XSD)

    # Ontology declaration
    ont = URIRef("https://graphrag.example.com/ontology")
    g.add((ont, RDF.type, OWL.Ontology))
    g.add((ont, RDFS.label, Literal("GraphRAG Knowledge Graph Ontology")))
    g.add((ont, RDFS.comment, Literal("Exported from the GraphRAG platform.")))

    # Annotation properties
    for prop_name, range_type in [
        ("confidence", XSD.float),
        ("validFrom",  XSD.string),
        ("validTo",    XSD.string),
        ("sourceDoc",  XSD.string),
        ("tenant",     XSD.string),
    ]:
        prop = ANNOT[prop_name]
        g.add((prop, RDF.type, OWL.AnnotationProperty))
        g.add((prop, RDFS.range, range_type))

    return g


async def export(tenant: str, output: Path, limit: int, infer: bool = False) -> None:
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j = get_neo4j()
    g = _init_graph()

    g.add((URIRef("https://graphrag.example.com/ontology"),
           RDFS.comment,
           Literal(f"Generated: {datetime.now(timezone.utc).isoformat()}  Tenant: {tenant}")))

    # ── Entity type hierarchy ──────────────────────────────────────────────────
    type_rows = await neo4j.run(
        "MATCH (c:EntityType)-[:SUBCLASS_OF]->(p:EntityType) RETURN c.name AS child, p.name AS parent"
    )
    declared_types: set[str] = set()
    for row in type_rows:
        child, parent = row["child"], row["parent"]
        for t in (child, parent):
            if t not in declared_types:
                t_uri = _type_uri(t)
                g.add((t_uri, RDF.type, OWL.Class))
                g.add((t_uri, RDFS.label, Literal(t)))
                declared_types.add(t)
        g.add((_type_uri(child), RDFS.subClassOf, _type_uri(parent)))

    # ── Object properties ──────────────────────────────────────────────────────
    rel_rows = await neo4j.run(
        """
        MATCH ()-[r:RELATES_TO]->()
        WHERE ($tenant = 'default' OR r.tenant = $tenant)
        RETURN DISTINCT r.relation AS rel LIMIT $limit
        """,
        tenant=tenant, limit=limit,
    )
    declared_rels: set[str] = set()
    for row in rel_rows:
        rel = (row.get("rel") or "RELATED_TO").upper()
        if rel not in declared_rels:
            r_uri = _rel_uri(rel)
            g.add((r_uri, RDF.type, OWL.ObjectProperty))
            g.add((r_uri, RDFS.label, Literal(rel)))
            declared_rels.add(rel)

    # ── Entities ───────────────────────────────────────────────────────────────
    ent_rows = await neo4j.run(
        """
        MATCH (e:Entity)
        WHERE ($tenant = 'default' OR e.tenant = $tenant)
        RETURN e.name AS name, e.type AS type, e.description AS desc,
               e.valid_from AS vf, e.valid_to AS vt, e.tenant AS tenant
        LIMIT $limit
        """,
        tenant=tenant, limit=limit,
    )
    for row in ent_rows:
        name  = row["name"] or ""
        etype = row["type"] or "CONCEPT"
        t     = row["tenant"] or "default"
        uri   = _entity_uri(name, etype, t)

        g.add((uri, RDF.type, OWL.NamedIndividual))
        g.add((uri, RDF.type, _type_uri(etype)))
        g.add((uri, RDFS.label, Literal(name)))
        g.add((uri, ANNOT.tenant, Literal(t)))
        if row.get("desc"):
            g.add((uri, RDFS.comment, Literal(str(row["desc"])[:500])))
        if row.get("vf"):
            g.add((uri, ANNOT.validFrom, Literal(str(row["vf"]))))
        if row.get("vt"):
            g.add((uri, ANNOT.validTo, Literal(str(row["vt"]))))

    # ── Relations with reified confidence ─────────────────────────────────────
    edge_rows = await neo4j.run(
        """
        MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
        WHERE ($tenant = 'default' OR r.tenant = $tenant)
        RETURN s.name AS sname, s.type AS stype,
               t.name AS tname, t.type AS ttype,
               r.relation AS rel,
               r.confidence AS conf,
               r.source_doc_id AS src_doc,
               r.tenant AS tenant
        LIMIT $limit
        """,
        tenant=tenant, limit=limit,
    )
    for row in edge_rows:
        t      = row["tenant"] or "default"
        s_uri  = _entity_uri(row["sname"], row["stype"], t)
        o_uri  = _entity_uri(row["tname"], row["ttype"], t)
        p_uri  = _rel_uri(row["rel"] or "RELATED_TO")
        conf   = row.get("conf")
        sdoc   = row.get("src_doc") or ""

        # Main triple
        g.add((s_uri, p_uri, o_uri))

        # Reify with owl:Axiom to carry confidence + provenance annotations
        if conf is not None or sdoc:
            ax = _axiom_uri(row["sname"], row["rel"], row["tname"])
            g.add((ax, RDF.type, OWL.Axiom))
            g.add((ax, OWL.annotatedSource,   s_uri))
            g.add((ax, OWL.annotatedProperty, p_uri))
            g.add((ax, OWL.annotatedTarget,   o_uri))
            if conf is not None:
                g.add((ax, ANNOT.confidence, Literal(round(float(conf), 4), datatype=XSD.float)))
            if sdoc:
                g.add((ax, ANNOT.sourceDoc, Literal(sdoc)))

    # ── Negative relations ─────────────────────────────────────────────────────
    neg_rows = await neo4j.run(
        """
        MATCH (s:Entity)-[r:NEGATIVE_RELATES_TO]->(t:Entity)
        WHERE ($tenant = 'default' OR r.tenant = $tenant)
        RETURN s.name AS sname, s.type AS stype,
               t.name AS tname, t.type AS ttype,
               r.relation AS rel, r.confidence AS conf,
               r.tenant AS tenant
        LIMIT $limit
        """,
        tenant=tenant, limit=limit,
    )
    for row in neg_rows:
        t     = row["tenant"] or "default"
        s_uri = _entity_uri(row["sname"], row["stype"], t)
        o_uri = _entity_uri(row["tname"], row["ttype"], t)
        rel   = (row["rel"] or "RELATED_TO").upper()
        # Use a negated property URI (annotation-based, OWL-DL friendly approach)
        neg_uri = BASE[f"{rel}_NEGATED"]
        g.add((neg_uri, RDF.type, OWL.ObjectProperty))
        g.add((neg_uri, RDFS.label, Literal(f"NOT {rel}")))
        g.add((s_uri, neg_uri, o_uri))
        if row.get("conf") is not None:
            ax = _axiom_uri(f"NEG_{row['sname']}", rel, row["tname"])
            g.add((ax, RDF.type, OWL.Axiom))
            g.add((ax, OWL.annotatedSource,   s_uri))
            g.add((ax, OWL.annotatedProperty, neg_uri))
            g.add((ax, OWL.annotatedTarget,   o_uri))
            g.add((ax, ANNOT.confidence,
                   Literal(round(float(row["conf"]), 4), datatype=XSD.float)))

    # ── Optional OWL-RL closure ────────────────────────────────────────────────
    if infer:
        from graphrag.graph.owl_reasoner import OWLRLReasoner
        reasoner   = OWLRLReasoner(g)
        n_inferred = reasoner.apply_closure()
        consistent = reasoner.is_consistent()
        log.info("export_rdf.owl_rl",
                 new_triples=n_inferred, consistent=consistent)
        print(f"  OWL-RL closure: {n_inferred} triples inferred  "
              f"(consistent={consistent})")
        g = reasoner._g   # use the expanded graph for serialisation

    await neo4j.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(output), format="turtle")

    entity_count = len(ent_rows)
    edge_count   = len(edge_rows)
    triple_count = len(g)
    log.info(
        "export_rdf.complete",
        output=str(output),
        entities=entity_count,
        edges=edge_count,
        triples=triple_count,
        tenant=tenant,
    )
    print(f"✅  Exported {entity_count} entities, {edge_count} edges, "
          f"{triple_count} RDF triples → {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Export knowledge graph to Turtle (RDF) using rdflib"
    )
    parser.add_argument("--tenant",  default="default",
                        help="Tenant to export (default: default)")
    parser.add_argument("--output",  default="exports/graph_export.ttl",
                        help="Output Turtle file path")
    parser.add_argument("--limit",   type=int, default=50_000,
                        help="Max entities and edges per query (default: 50000)")
    parser.add_argument("--infer", action="store_true",
                        help="Apply OWL-RL closure after export (materialises "
                             "subClass propagation, symmetric/inverse properties)")
    args = parser.parse_args()

    asyncio.run(export(
        tenant=args.tenant,
        output=Path(args.output),
        limit=args.limit,
        infer=args.infer,
    ))


if __name__ == "__main__":
    main()
