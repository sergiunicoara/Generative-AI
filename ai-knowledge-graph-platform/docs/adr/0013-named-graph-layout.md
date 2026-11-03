# ADR-0013: Named-graph layout for RDF stores

**Status:** Accepted (2026-11-03)

## Context

RDF leaves the platform today as per-tenant Turtle files
(`exports/<tenant>/graph_export.ttl`), Energy publications held in process by
`DatasetPublisher`, and the knowledge-release projection. A store-neutral SPARQL
adapter now reads Stardog, GraphDB and Fuseki. None of these use named graphs,
so a store holding several tenants, sources or publication versions has no
boundary to delete, roll back or query provenance by, other than filtering
triples.

## Decision

1. **One named graph per tenant, source and publication version:**
   `https://graphrag.example.com/graph/{tenant}/{source}/{publication}`.
   Segments are validated (`[A-Za-z0-9][A-Za-z0-9_.-]*`, at most 128
   characters) and percent-encoded (`graphrag/graph/named_graphs.py`).
2. **Tenant deletion is graph deletion.** List the tenant's graphs with a
   SPARQL SELECT on the tenant prefix, then `DROP GRAPH` each one by name
   (SPARQL 1.1 cannot drop by prefix). No other tenant's graph is touched.
3. **Rollback is a pointer, not a rewrite.** A new publication writes a new
   graph; consumers are pointed at the active publication graph, and older
   graphs are kept for replay until a retention rule removes them.
4. **Provenance by graph.** `GRAPH ?g { ... }` answers which publication and
   source a triple came from; PROV-O statements stay inside the graph they
   describe.
5. **Registry graph.** Which generated shapes apply to which data graph is
   stored as RDF (`graphrag/semantic_model/registry_rdf.py`), with
   `dcterms:conformsTo` from the data graph to the model, so the mapping
   travels with the data.
6. **JSON-LD** is an optional serialisation for API consumers that cannot use
   Turtle. It must round-trip IRIs and datatypes (`to_jsonld` / `from_jsonld`).

## Consequences

* Per-tenant deletion and rollback follow store boundaries instead of triple
  filters, which is the property the erasure audit relies on.
* A triple duplicated across publications is stored once per publication
  graph; storage grows with retained versions until retention removes them.
* Default-graph union semantics differ between stores (some expose named
  graphs in the default graph, some do not). Queries that need one
  publication must name its graph explicitly.
* Not done: the existing Turtle exports and `DatasetPublisher` still write a
  single graph; moving them onto this layout is follow-up work, and none of it
  has been run against a live triple store here. The behaviour is tested on an
  in-memory `rdflib.Dataset` with plain SPARQL 1.1.
