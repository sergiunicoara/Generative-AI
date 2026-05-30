"""Export the knowledge graph to Turtle (RDF) format.

Produces a standards-compliant Turtle file that can be loaded into:
  - Protégé (ontology editor / OWL reasoner)
  - SPARQL endpoints (Apache Jena Fuseki, Oxigraph)
  - Reasoners: HermiT, Pellet, FaCT++
  - Any RDF-aware linked-data tool

Mapping
-------
  Entity nodes          → owl:NamedIndividual + rdf:type
  EntityType nodes      → owl:Class + rdfs:subClassOf hierarchy
  RELATES_TO edges      → owl:ObjectProperty assertions
  NEGATIVE_RELATES_TO   → annotated negative assertions (owl:complementOf pattern)
  SUBCLASS_OF edges     → rdfs:subClassOf
  confidence            → custom annotation property :confidence (xsd:float)
  valid_from / valid_to → :validFrom / :validTo (xsd:dateTime)

Usage
-----
  python scripts/export_rdf.py
  python scripts/export_rdf.py --tenant acme --output exports/graph.ttl
  python scripts/export_rdf.py --tenant default --limit 10000
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

# ── RDF namespace constants ────────────────────────────────────────────────────

BASE_URI   = "https://graphrag.example.com/ontology#"
INST_URI   = "https://graphrag.example.com/entity/"
OWL_URI    = "http://www.w3.org/2002/07/owl#"
RDF_URI    = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_URI   = "http://www.w3.org/2000/01/rdf-schema#"
XSD_URI    = "http://www.w3.org/2001/XMLSchema#"

TURTLE_PREFIXES = f"""@prefix :        <{BASE_URI}> .
@prefix inst:    <{INST_URI}> .
@prefix owl:     <{OWL_URI}> .
@prefix rdf:     <{RDF_URI}> .
@prefix rdfs:    <{RDFS_URI}> .
@prefix xsd:     <{XSD_URI}> .

"""


def _uri_safe(s: str) -> str:
    """Percent-encode characters that are not URI-safe."""
    return s.replace(" ", "_").replace(",", "%2C").replace("(", "%28").replace(")", "%29").replace("/", "%2F")


def _entity_uri(name: str, etype: str, tenant: str) -> str:
    return f"inst:{_uri_safe(tenant)}/{_uri_safe(etype)}/{_uri_safe(name)}"


def _type_uri(etype: str) -> str:
    return f":{_uri_safe(etype)}"


def _rel_uri(relation: str) -> str:
    return f":{_uri_safe(relation.upper())}"


def _literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


# ── Ontology header block ──────────────────────────────────────────────────────

ONTOLOGY_HEADER = f"""# ─────────────────────────────────────────────────────────────────
# GraphRAG Knowledge Graph — Turtle export
# Generated: {{timestamp}}
# Tenant:    {{tenant}}
# ─────────────────────────────────────────────────────────────────

<{BASE_URI.rstrip("#")}>
    a owl:Ontology ;
    rdfs:label "GraphRAG Knowledge Graph Ontology" ;
    rdfs:comment "Exported from the GraphRAG platform." .

# ── Annotation properties ─────────────────────────────────────────
:confidence  a owl:AnnotationProperty ; rdfs:range xsd:float .
:validFrom   a owl:AnnotationProperty ; rdfs:range xsd:dateTime .
:validTo     a owl:AnnotationProperty ; rdfs:range xsd:dateTime .
:sourceDoc   a owl:AnnotationProperty ; rdfs:range xsd:string .
:tenant      a owl:AnnotationProperty ; rdfs:range xsd:string .

"""


async def export(tenant: str, output: Path, limit: int) -> None:
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j = get_neo4j()
    lines: list[str] = []

    lines.append(TURTLE_PREFIXES)
    lines.append(ONTOLOGY_HEADER.format(
        timestamp=datetime.utcnow().isoformat(),
        tenant=tenant,
    ))

    # ── Entity type hierarchy ──────────────────────────────────────────────────
    lines.append("# ── Entity type hierarchy (owl:Class + rdfs:subClassOf) ───\n")
    type_rows = await neo4j.run(
        "MATCH (c:EntityType)-[:SUBCLASS_OF]->(p:EntityType) RETURN c.name AS child, p.name AS parent"
    )
    declared_types: set[str] = set()
    for row in type_rows:
        child, parent = row["child"], row["parent"]
        for t in (child, parent):
            if t not in declared_types:
                lines.append(f"{_type_uri(t)} a owl:Class ; rdfs:label {_literal(t)} .\n")
                declared_types.add(t)
        lines.append(f"{_type_uri(child)} rdfs:subClassOf {_type_uri(parent)} .\n")
    lines.append("\n")

    # ── Relation properties ────────────────────────────────────────────────────
    lines.append("# ── Object properties (owl:ObjectProperty) ──────────────────\n")
    rel_rows = await neo4j.run(
        """
        MATCH ()-[r:RELATES_TO]->()
        WHERE ($tenant = 'default' OR r.tenant = $tenant)
        RETURN DISTINCT r.relation AS rel
        LIMIT $limit
        """,
        tenant=tenant,
        limit=limit,
    )
    declared_rels: set[str] = set()
    for row in rel_rows:
        rel = row.get("rel") or "RELATED_TO"
        if rel not in declared_rels:
            lines.append(f"{_rel_uri(rel)} a owl:ObjectProperty ; rdfs:label {_literal(rel)} .\n")
            declared_rels.add(rel)
    lines.append("\n")

    # ── Entities (owl:NamedIndividual) ─────────────────────────────────────────
    lines.append("# ── Entities (owl:NamedIndividual) ──────────────────────────\n")
    ent_rows = await neo4j.run(
        """
        MATCH (e:Entity)
        WHERE ($tenant = 'default' OR e.tenant = $tenant)
        RETURN e.name AS name, e.type AS type, e.description AS desc,
               e.valid_from AS vf, e.valid_to AS vt, e.tenant AS tenant
        LIMIT $limit
        """,
        tenant=tenant,
        limit=limit,
    )
    for row in ent_rows:
        name   = row["name"] or ""
        etype  = row["type"] or "CONCEPT"
        desc   = row.get("desc") or ""
        t      = row["tenant"] or "default"
        uri    = _entity_uri(name, etype, t)
        tclass = _type_uri(etype)

        triples = [
            f"{uri} a owl:NamedIndividual, {tclass} ;",
            f"    rdfs:label {_literal(name)} ;",
            f"    :tenant {_literal(t)} ;",
        ]
        if desc:
            triples.append(f"    rdfs:comment {_literal(desc[:500])} ;")
        if row.get("vf"):
            triples.append(f'    :validFrom "{row["vf"]}"^^xsd:string ;')
        if row.get("vt"):
            triples.append(f'    :validTo "{row["vt"]}"^^xsd:string ;')
        # Close with period on last triple
        triples[-1] = triples[-1].rstrip(" ;") + " ."
        lines.append("\n".join(triples) + "\n\n")

    # ── Relations (owl:ObjectProperty assertions) ──────────────────────────────
    lines.append("# ── Relations (ObjectProperty assertions) ───────────────────\n")
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
        tenant=tenant,
        limit=limit,
    )
    for row in edge_rows:
        t      = row["tenant"] or "default"
        s_uri  = _entity_uri(row["sname"], row["stype"], t)
        o_uri  = _entity_uri(row["tname"], row["ttype"], t)
        p_uri  = _rel_uri(row["rel"] or "RELATED_TO")
        conf   = row.get("conf")
        sdoc   = row.get("src_doc") or ""

        # Main assertion
        triple = f"{s_uri} {p_uri} {o_uri}"
        if conf is not None or sdoc:
            # Use reification (owl:Axiom) to annotate with confidence + provenance
            axiom_id = f"inst:axiom/{_uri_safe(row['sname'])}_{_uri_safe(row['rel'])}_{_uri_safe(row['tname'])}"
            lines.append(f"{triple} .\n")
            lines.append(f"{axiom_id} a owl:Axiom ;\n")
            lines.append(f"    owl:annotatedSource {s_uri} ;\n")
            lines.append(f"    owl:annotatedProperty {p_uri} ;\n")
            lines.append(f"    owl:annotatedTarget {o_uri} ;\n")
            if conf is not None:
                lines.append(f"    :confidence \"{float(conf):.4f}\"^^xsd:float ;\n")
            if sdoc:
                lines.append(f"    :sourceDoc {_literal(sdoc)} ;\n")
            lines[-1] = lines[-1].rstrip(" ;\n") + " .\n"
        else:
            lines.append(f"{triple} .\n")

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
        tenant=tenant,
        limit=limit,
    )
    if neg_rows:
        lines.append("\n# ── Negative assertions ───────────────────────────────────\n")
        for row in neg_rows:
            t     = row["tenant"] or "default"
            s_uri = _entity_uri(row["sname"], row["stype"], t)
            o_uri = _entity_uri(row["tname"], row["ttype"], t)
            p_uri = _rel_uri(row["rel"] or "RELATED_TO")
            conf  = row.get("conf")
            # Model as: s :rel_complement o (annotation-based, not OWL DL)
            neg_uri = f":{_uri_safe(row['rel'])}_NEGATED"
            lines.append(f"# NEGATED: {row['sname']} -[{row['rel']}]-> {row['tname']}\n")
            lines.append(f"{s_uri} {neg_uri} {o_uri}")
            if conf is not None:
                lines.append(f" ; :confidence \"{float(conf):.4f}\"^^xsd:float")
            lines.append(" .\n")

    await neo4j.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")

    entity_count = len(ent_rows)
    edge_count   = len(edge_rows)
    log.info(
        "export_rdf.complete",
        output=str(output),
        entities=entity_count,
        edges=edge_count,
        tenant=tenant,
    )
    print(f"✅  Exported {entity_count} entities, {edge_count} edges → {output}")


def main():
    parser = argparse.ArgumentParser(description="Export knowledge graph to Turtle (RDF)")
    parser.add_argument("--tenant", default="default", help="Tenant to export (default: all)")
    parser.add_argument("--output", default="exports/graph_export.ttl",
                        help="Output Turtle file path")
    parser.add_argument("--limit", type=int, default=50_000,
                        help="Max entities and edges to export (default: 50000)")
    args = parser.parse_args()

    asyncio.run(export(
        tenant=args.tenant,
        output=Path(args.output),
        limit=args.limit,
    ))


if __name__ == "__main__":
    main()
