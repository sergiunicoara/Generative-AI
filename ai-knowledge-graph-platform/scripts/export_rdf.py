"""Export the knowledge graph to Turtle or JSON-LD using rdflib.

Produces a standards-compliant Turtle file that can be loaded into:
  - Protégé (ontology editor / OWL reasoner)
  - SPARQL endpoints (Apache Jena Fuseki, Oxigraph)
  - Reasoners: HermiT, Pellet, FaCT++
  - Any RDF-aware linked-data tool

Mapping
-------
  Entity nodes          → owl:NamedIndividual + rdf:type
  EntityType nodes      → owl:Class + rdfs:subClassOf hierarchy
  Browseable concepts   → skos:Concept in a tenant ConceptScheme
  RELATES_TO edges      → owl:ObjectProperty assertions with reified confidence
  NEGATIVE_RELATES_TO   → annotated negative assertions
  SUBCLASS_OF edges     → rdfs:subClassOf
  confidence            → :confidence annotation (xsd:float)
  valid_from / valid_to → :validFrom / :validTo annotations

Uses rdflib for guaranteed valid RDF serialization (handles unicode, quotes,
special characters that hand-rolled string concatenation cannot).

Usage
-----
  python scripts/export_rdf.py --tenant acme
      # writes exports/acme/graph_export.ttl -- this is what POST /kg/sparql
      # reads for a caller authenticated as tenant "acme"
  python scripts/export_rdf.py --tenant acme --output custom/path.ttl
      # explicit --output bypasses the per-tenant default entirely
  python scripts/export_rdf.py --tenant acme --format json-ld
      # writes exports/acme/graph_export.jsonld for linked-data interchange
  python scripts/export_rdf.py --tenant default --limit 10000
"""

from __future__ import annotations

import argparse
import os
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog

# Windows console defaults to cp1252, which can't encode the emoji used in
# status output below — reconfigure stdout to UTF-8.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD
from graphrag.provenance.prov_o import (
    PROV,
    activity_uri,
    add_activity,
    add_agent,
    add_association,
    add_derived,
    add_entity,
    add_generated,
    add_used,
    agent_uri,
    answer_uri,
    chunk_uri,
    document_uri,
    entity_uri as prov_entity_uri,
    query_uri,
)

log = structlog.get_logger(__name__)

# ── Namespaces ─────────────────────────────────────────────────────────────────

BASE  = Namespace("https://graphrag.example.com/ontology#")
INST  = Namespace("https://graphrag.example.com/entity/")
ANNOT = Namespace("https://graphrag.example.com/annotation#")
SKOS  = Namespace("http://www.w3.org/2004/02/skos/core#")


def _entity_uri(name: str, etype: str, tenant: str) -> URIRef:
    """Stable entity URI — tenant-scoped to prevent cross-tenant collisions."""
    def _safe(s: str) -> str:
        import urllib.parse
        return urllib.parse.quote(s, safe="")
    return INST[f"{_safe(tenant)}/{_safe(etype)}/{_safe(name)}"]


def _type_uri(etype: str) -> URIRef:
    return BASE[etype.upper()]


def _scheme_uri(tenant: str) -> URIRef:
    """Stable SKOS concept scheme URI, isolated per exported tenant."""
    import urllib.parse

    return INST[f"scheme/{urllib.parse.quote(tenant, safe='')}"]


def _rel_uri(relation: str) -> URIRef:
    return BASE[relation.upper()]


def _axiom_uri(s_name: str, rel: str, t_name: str, tenant: str = "default") -> URIRef:
    import hashlib
    key = f"{tenant}|{s_name}|{rel}|{t_name}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return INST[f"axiom/{h}"]


def _document_uri(source_doc_id: str, tenant: str) -> URIRef:
    """Stable PROV-O source-artifact URI scoped to the owning tenant."""
    return document_uri(source_doc_id, tenant)


def _add_provenance_document(g: Graph, source_doc_id: str, tenant: str) -> URIRef:
    uri = _document_uri(source_doc_id, tenant)
    add_entity(g, uri, label=source_doc_id, tenant=tenant)
    return uri


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
    g.bind("prov",  PROV)
    g.bind("skos",  SKOS)

    # Ontology declaration
    ont = URIRef("https://graphrag.example.com/ontology")
    g.add((ont, RDF.type, OWL.Ontology))
    g.add((ont, RDFS.label, Literal("AI Knowledge Graph & Ontology Platform")))
    g.add((ont, RDFS.comment, Literal("Exported from the AI Knowledge Graph & Ontology Platform.")))

    # Annotation properties
    for prop_name, range_type in [
        ("confidence", XSD.float),
        ("validFrom",  XSD.string),
        ("validTo",    XSD.string),
        ("sourceDoc",  XSD.string),
        ("tenant",     XSD.string),
        ("status",     XSD.string),
        ("modelProvider", XSD.string),
        ("modelVersion", XSD.string),
        ("promptVersion", XSD.string),
        ("contentDigest", XSD.string),
    ]:
        prop = ANNOT[prop_name]
        g.add((prop, RDF.type, OWL.AnnotationProperty))
        g.add((prop, RDFS.range, range_type))

    return g


def _list_value(value: object) -> list[str]:
    """Normalise Neo4j lists and JSON-encoded legacy properties."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return [str(value)]


def _emit_provenance_rows(graph: Graph, rows: list[dict], tenant: str) -> dict[tuple[str, str], URIRef]:
    """Project durable ingestion/context records into PROV-O.

    The operational records remain authoritative in Neo4j.  The exporter only
    emits tenant-matching rows and never copies answer text or other sensitive
    payloads into the interchange graph; identifiers and content digests are
    sufficient for lineage without turning the RDF export into a data dump.
    """
    ingestion_by_document: dict[tuple[str, str], URIRef] = {}

    for row in rows:
        row_tenant = str(row.get("row_tenant") or row.get("tenant") or tenant)
        if row_tenant != tenant and tenant != "default":
            continue
        kind = str(row.get("kind") or "")
        identifier = str(row.get("id") or "")
        if not identifier:
            continue

        provider = str(row.get("model_provider") or row.get("provider") or "platform")
        version = str(row.get("model_version") or row.get("version") or "unknown")
        agent_key = str(row.get("agent_id") or f"{provider}:{version}")
        agent = agent_uri("software", agent_key, row_tenant)
        add_agent(graph, agent, label=f"{provider} ({version})", tenant=row_tenant)

        if kind == "ingestion":
            activity = activity_uri("ingestion", identifier, row_tenant)
            add_activity(
                graph, activity,
                label=f"Ingestion run {identifier}",
                tenant=row_tenant,
                started_at=row.get("started_at"),
                ended_at=row.get("ended_at") or row.get("completed_at"),
                status=str(row.get("status") or ""),
            )
            add_association(graph, activity, agent)
            document_id = str(row.get("document_id") or "")
            if document_id:
                document = _add_provenance_document(graph, document_id, row_tenant)
                add_used(graph, activity, document)
                ingestion_by_document[(row_tenant, document_id)] = activity
                for chunk_id in _list_value(row.get("chunk_ids")):
                    chunk = chunk_uri(chunk_id, row_tenant)
                    add_entity(graph, chunk, label=f"Chunk {chunk_id}", tenant=row_tenant)
                    add_derived(graph, chunk, document)
                    add_generated(graph, chunk, activity)
            continue

        if kind == "retrieval":
            activity = activity_uri("retrieval", identifier, row_tenant)
            add_activity(
                graph, activity,
                label=f"Retrieval run {identifier}",
                tenant=row_tenant,
                started_at=row.get("started_at"),
                ended_at=row.get("ended_at"),
                status=str(row.get("status") or ""),
            )
            add_association(graph, activity, agent)
            query = query_uri(identifier, row_tenant)
            add_entity(graph, query, label=f"Query {identifier}", tenant=row_tenant)
            add_used(graph, activity, query)
            manifest_id = str(row.get("manifest_id") or "")
            if manifest_id:
                manifest = prov_entity_uri(f"context-manifest/{manifest_id}", row_tenant)
                add_entity(graph, manifest, label=f"Context manifest {manifest_id}", tenant=row_tenant)
                add_used(graph, activity, manifest)
            for document_id in _list_value(row.get("document_ids")):
                add_used(graph, activity, _add_provenance_document(graph, document_id, row_tenant))
            for chunk_id in _list_value(row.get("chunk_ids")):
                chunk = chunk_uri(chunk_id, row_tenant)
                add_entity(graph, chunk, label=f"Chunk {chunk_id}", tenant=row_tenant)
                add_used(graph, activity, chunk)
            episodes = row.get("episodes") or []
            if isinstance(episodes, str):
                try:
                    episodes = json.loads(episodes)
                except (TypeError, ValueError):
                    episodes = []
            answer_digest = str(row.get("answer_digest") or "")
            if not answer_digest:
                for episode in episodes if isinstance(episodes, list) else []:
                    if isinstance(episode, dict) and str(episode.get("episode_type")) == "answer":
                        answer_digest = str(episode.get("content_digest") or "")
                        break
            answer = answer_uri(identifier, row_tenant)
            add_entity(graph, answer, label=f"Answer {answer_digest or identifier}", tenant=row_tenant)
            if answer_digest:
                graph.add((answer, ANNOT.contentDigest, Literal(answer_digest)))
            add_generated(graph, answer, activity)
            continue

        if kind == "artifact":
            activity = activity_uri("artifact-extraction", identifier, row_tenant)
            add_activity(
                graph, activity,
                label=f"Artifact extraction {identifier}",
                tenant=row_tenant,
                started_at=row.get("started_at"),
                ended_at=row.get("ended_at"),
                status=str(row.get("status") or "completed"),
            )
            add_association(graph, activity, agent)
            artifact = prov_entity_uri(f"intelligence-artifact/{identifier}", row_tenant)
            add_entity(graph, artifact, label=f"Intelligence artifact {identifier}", tenant=row_tenant)
            source_chunk_id = str(row.get("source_chunk_id") or "")
            if source_chunk_id:
                source_chunk = chunk_uri(source_chunk_id, row_tenant)
                add_entity(graph, source_chunk, label=f"Chunk {source_chunk_id}", tenant=row_tenant)
                add_used(graph, activity, source_chunk)
                add_derived(graph, artifact, source_chunk)
            add_generated(graph, artifact, activity)

    return ingestion_by_document


def _activity_for_document(
    graph: Graph,
    document_id: str,
    tenant: str,
    *,
    extraction_model: str = "",
    prompt_version: str = "",
    activity_cache: dict[tuple[str, str, str, str], URIRef] | None = None,
) -> URIRef:
    """Return a deterministic fallback activity when no manifest is present."""
    cache = activity_cache if activity_cache is not None else {}
    key = (tenant, document_id, extraction_model, prompt_version)
    if key in cache:
        return cache[key]
    identifier = f"{document_id}:{extraction_model or 'unknown'}:{prompt_version or 'unknown'}"
    activity = activity_uri("extraction", identifier, tenant)
    add_activity(graph, activity, label=f"Extraction for {document_id}", tenant=tenant)
    agent_key = f"{extraction_model or 'unknown'}:{prompt_version or 'unknown'}"
    agent = agent_uri("software", agent_key, tenant)
    add_agent(graph, agent, label=agent_key, tenant=tenant)
    add_association(graph, activity, agent)
    add_used(graph, activity, _add_provenance_document(graph, document_id, tenant))
    cache[key] = activity
    return activity


async def export(
    tenant: str,
    output: Path,
    limit: int,
    infer: bool = False,
    validate: bool = False,
    rdf_format: str = "turtle",
) -> None:
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j = get_neo4j()
    g = _init_graph()
    scheme = _scheme_uri(tenant)
    g.add((scheme, RDF.type, SKOS.ConceptScheme))
    g.add((scheme, SKOS.prefLabel, Literal(f"Knowledge graph concepts ({tenant})")))
    g.add((scheme, ANNOT.tenant, Literal(tenant)))

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
                g.add((t_uri, RDF.type, SKOS.Concept))
                g.add((t_uri, SKOS.prefLabel, Literal(t)))
                g.add((t_uri, SKOS.inScheme, scheme))
                declared_types.add(t)
        g.add((_type_uri(child), RDFS.subClassOf, _type_uri(parent)))
        g.add((_type_uri(child), SKOS.broader, _type_uri(parent)))

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

    # ── PROV-O activities, agents, and trace evidence ─────────────────────────
    # One bounded query keeps export round-trips small while covering the
    # durable ingestion and Context Graph records already present in Neo4j.
    prov_rows = await neo4j.run(
        """
        CALL {
          MATCH (m:IngestionRunManifest)
          WHERE ($tenant = 'default' OR m.tenant = $tenant)
          OPTIONAL MATCH (c:Chunk {tenant: m.tenant, document_id: m.document_id})
          WITH m, collect(DISTINCT c.id) AS chunk_ids
          RETURN 'ingestion' AS kind, m.id AS id, m.tenant AS row_tenant,
                 m.document_id AS document_id, m.model_provider AS model_provider,
                 m.model_version AS model_version, m.started_at AS started_at,
                 m.completed_at AS completed_at, m.status AS status,
                 chunk_ids AS chunk_ids, [] AS document_ids, [] AS source_chunk_ids,
                 null AS manifest_id, [] AS episodes, null AS answer_digest
          UNION ALL
          MATCH (r:CGAgentRun)
          WHERE ($tenant = 'default' OR r.tenant = $tenant)
          OPTIONAL MATCH (r)-[:USED_CONTEXT]->(m:CGContextManifest)
          OPTIONAL MATCH (r)-[:RECORDED_EPISODE]->(ep:CGEpisode)
          WITH r, m, collect(DISTINCT ep { .episode_type, .content_digest }) AS episodes
          RETURN 'retrieval' AS kind, r.id AS id, r.tenant AS row_tenant,
                 null AS document_id, r.model_provider AS model_provider,
                 r.model_version AS model_version, r.created_at AS started_at,
                 r.completed_at AS completed_at, r.status AS status,
                 coalesce(m.chunk_ids, []) AS chunk_ids,
                 coalesce(m.document_ids, []) AS document_ids,
                 [] AS source_chunk_ids, m.id AS manifest_id,
                 episodes AS episodes, null AS answer_digest
          UNION ALL
          MATCH (a:IntelligenceArtifact)
          WHERE ($tenant = 'default' OR a.tenant = $tenant)
          RETURN 'artifact' AS kind, a.id AS id, a.tenant AS row_tenant,
                 a.source_doc_id AS document_id, a.extraction_model AS model_provider,
                 a.extraction_model AS model_version, null AS started_at,
                 a.event_end AS completed_at, 'completed' AS status,
                 [] AS chunk_ids, [] AS document_ids,
                 [a.source_chunk_id] AS source_chunk_ids, null AS manifest_id,
                 [] AS episodes, null AS answer_digest
        }
        RETURN kind, id, row_tenant, document_id, model_provider, model_version,
               started_at, completed_at, status, chunk_ids, document_ids,
               source_chunk_ids, manifest_id, episodes, answer_digest
        LIMIT $limit
        """,
        tenant=tenant, limit=limit,
    )
    ingestion_by_document = _emit_provenance_rows(g, prov_rows, tenant)

    # ── Entities ───────────────────────────────────────────────────────────────
    ent_rows = await neo4j.run(
        """
        MATCH (e:Entity)
        WHERE ($tenant = 'default' OR e.tenant = $tenant)
        RETURN e.name AS name, e.type AS type, e.description AS desc,
               e.valid_from AS vf, e.valid_to AS vt, e.tenant AS tenant
               , e.source_doc_id AS src_doc, e.source_chunk_ids AS source_chunks,
               e.extraction_model AS extraction_model, e.prompt_version AS prompt_version
        LIMIT $limit
        """,
        tenant=tenant, limit=limit,
    )
    activity_cache: dict[tuple[str, str, str, str], URIRef] = {}
    for row in ent_rows:
        name  = row["name"] or ""
        etype = row["type"] or "CONCEPT"
        t     = row["tenant"] or "default"
        uri   = _entity_uri(name, etype, t)
        type_uri = _type_uri(etype)

        # Types that have no explicit SUBCLASS_OF edge still need a SKOS
        # concept so every exported entity has a navigable broader concept.
        if etype not in declared_types:
            g.add((type_uri, RDF.type, OWL.Class))
            g.add((type_uri, RDFS.label, Literal(etype)))
            g.add((type_uri, RDF.type, SKOS.Concept))
            g.add((type_uri, SKOS.prefLabel, Literal(etype)))
            g.add((type_uri, SKOS.inScheme, scheme))
            declared_types.add(etype)

        g.add((uri, RDF.type, OWL.NamedIndividual))
        g.add((uri, RDF.type, type_uri))
        g.add((uri, RDFS.label, Literal(name)))
        g.add((uri, RDF.type, SKOS.Concept))
        g.add((uri, SKOS.prefLabel, Literal(name)))
        g.add((uri, SKOS.inScheme, scheme))
        g.add((uri, SKOS.broader, type_uri))
        g.add((uri, ANNOT.tenant, Literal(t)))
        if row.get("desc"):
            g.add((uri, RDFS.comment, Literal(str(row["desc"])[:500])))
        if row.get("vf"):
            g.add((uri, ANNOT.validFrom, Literal(str(row["vf"]))))
        if row.get("vt"):
            g.add((uri, ANNOT.validTo, Literal(str(row["vt"]))))
        if row.get("src_doc"):
            document_id = str(row["src_doc"])
            document = _add_provenance_document(g, document_id, t)
            add_derived(g, uri, document)
            activity = _activity_for_document(
                g, document_id, t,
                extraction_model=str(row.get("extraction_model") or ""),
                prompt_version=str(row.get("prompt_version") or ""),
                activity_cache=activity_cache,
            )
            add_generated(g, uri, activity)
            for chunk_id in _list_value(row.get("source_chunks")):
                chunk = chunk_uri(chunk_id, t)
                add_entity(g, chunk, label=f"Chunk {chunk_id}", tenant=t)
                add_derived(g, uri, chunk)

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
               r.source_chunk_id AS source_chunk_id,
               r.extracted_at AS extracted_at,
               r.tenant AS tenant,
               r.extraction_model AS extraction_model,
               r.prompt_version AS prompt_version
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
            ax = _axiom_uri(row["sname"], row["rel"], row["tname"], t)
            g.add((ax, RDF.type, OWL.Axiom))
            g.add((ax, RDF.type, PROV.Entity))
            g.add((ax, RDFS.label, Literal(f"{row['sname']} {row['rel']} {row['tname']}")))
            g.add((ax, ANNOT.tenant, Literal(t)))
            g.add((ax, OWL.annotatedSource,   s_uri))
            g.add((ax, OWL.annotatedProperty, p_uri))
            g.add((ax, OWL.annotatedTarget,   o_uri))
            if conf is not None:
                g.add((ax, ANNOT.confidence, Literal(round(float(conf), 4), datatype=XSD.float)))
            if sdoc:
                g.add((ax, ANNOT.sourceDoc, Literal(sdoc)))
                document = _add_provenance_document(g, sdoc, t)
                add_derived(g, ax, document)
                activity = ingestion_by_document.get((t, sdoc)) if 'ingestion_by_document' in locals() else None
                if activity is None:
                    activity = _activity_for_document(
                        g, sdoc, t,
                        extraction_model=str(row.get("extraction_model") or ""),
                        prompt_version=str(row.get("prompt_version") or ""),
                        activity_cache=activity_cache,
                    )
                add_generated(g, ax, activity)
            if row.get("source_chunk_id"):
                chunk = chunk_uri(str(row["source_chunk_id"]), t)
                add_entity(g, chunk, label=f"Chunk {row['source_chunk_id']}", tenant=t)
                add_derived(g, ax, chunk)
            if row.get("extracted_at"):
                g.add((ax, PROV.generatedAtTime, Literal(str(row["extracted_at"]), datatype=XSD.dateTime)))

    # ── Negative relations ─────────────────────────────────────────────────────
    neg_rows = await neo4j.run(
        """
        MATCH (s:Entity)-[r:NEGATIVE_RELATES_TO]->(t:Entity)
        WHERE ($tenant = 'default' OR r.tenant = $tenant)
        RETURN s.name AS sname, s.type AS stype,
               t.name AS tname, t.type AS ttype,
               r.relation AS rel, r.confidence AS conf,
               r.source_doc_id AS src_doc,
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
            ax = _axiom_uri(f"NEG_{row['sname']}", rel, row["tname"], t)
            g.add((ax, RDF.type, OWL.Axiom))
            g.add((ax, RDF.type, PROV.Entity))
            g.add((ax, RDFS.label, Literal(f"NOT {row['sname']} {rel} {row['tname']}")))
            g.add((ax, ANNOT.tenant, Literal(t)))
            g.add((ax, OWL.annotatedSource,   s_uri))
            g.add((ax, OWL.annotatedProperty, neg_uri))
            g.add((ax, OWL.annotatedTarget,   o_uri))
            g.add((ax, ANNOT.confidence,
                   Literal(round(float(row["conf"]), 4), datatype=XSD.float)))
            if row.get("src_doc"):
                document = _add_provenance_document(g, str(row["src_doc"]), t)
                add_derived(g, ax, document)
                activity = _activity_for_document(g, str(row["src_doc"]), t, activity_cache=activity_cache)
                add_generated(g, ax, activity)

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

    if rdf_format not in {"turtle", "json-ld"}:
        raise ValueError("rdf_format must be 'turtle' or 'json-ld'")
    output.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(output), format=rdf_format)

    # ── Optional SHACL shape validation ────────────────────────────────────────
    if validate:
        from graphrag.graph.shacl_validator import SHACLValidator
        report = SHACLValidator(g).validate_report()
        log.info("export_rdf.shacl", conforms=report.conforms, **report.counts)
        print(f"\n{'✅' if report.conforms else '❌'}  SHACL validation: "
              f"{'conforms' if report.conforms else 'violations found'} "
              f"({report.counts['violations']} violation(s), "
              f"{report.counts['warnings']} warning(s))")
        if report.failures_by_shape:
            print("  By shape:", report.failures_by_shape)
        if not report.conforms:
            print(report.text)

    entity_count = len(ent_rows)
    edge_count   = len(edge_rows)
    triple_count = len(g)
    log.info(
        "export_rdf.complete",
        output=str(output),
        rdf_format=rdf_format,
        entities=entity_count,
        edges=edge_count,
        triples=triple_count,
        tenant=tenant,
    )
    print(f"✅  Exported {entity_count} entities, {edge_count} edges, "
          f"{triple_count} RDF triples → {output}")


def main():
    parser = argparse.ArgumentParser(description="Export knowledge graph RDF using rdflib")
    parser.add_argument("--tenant",  default="default",
                        help="Tenant to export (default: default)")
    # No static default -- POST /kg/sparql reads exports/<tenant>/graph_export.ttl
    # (GRAPHRAG_RDF_EXPORT_DIR overrides the "exports" root), so the default
    # output path must be derived from --tenant, not shared across every
    # tenant's export. A single shared file meant whichever tenant exported
    # last silently became the data every tenant's SPARQL queries saw.
    parser.add_argument("--output",  default=None,
                        help="Output RDF file path (default is derived from --format)")
    parser.add_argument("--format", choices=("turtle", "json-ld"), default="turtle",
                        help="RDF serialization format (default: turtle)")
    parser.add_argument("--limit",   type=int, default=50_000,
                        help="Max entities and edges per query (default: 50000)")
    parser.add_argument("--infer", action="store_true",
                        help="Apply OWL-RL closure after export (materialises "
                             "subClass propagation, symmetric/inverse properties)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate the exported graph against SHACL shapes "
                             "after export (entity labels/types, axiom completeness, "
                             "confidence range)")
    args = parser.parse_args()

    if args.output:
        output = Path(args.output)
    else:
        export_dir = Path(os.getenv("GRAPHRAG_RDF_EXPORT_DIR", "exports"))
        suffix = ".ttl" if args.format == "turtle" else ".jsonld"
        output = export_dir / args.tenant / f"graph_export{suffix}"

    asyncio.run(export(
        tenant=args.tenant,
        output=output,
        limit=args.limit,
        infer=args.infer,
        validate=args.validate,
        rdf_format=args.format,
    ))


if __name__ == "__main__":
    main()
