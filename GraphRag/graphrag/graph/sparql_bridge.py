"""SPARQL query bridge over exported Turtle knowledge graphs.

Wraps rdflib's built-in SPARQL 1.1 engine so the rest of the codebase can
issue standards-compliant SPARQL queries against any Turtle export produced
by ``scripts/export_rdf.py`` — without a live SPARQL endpoint or triplestore.

Typical usage::

    from graphrag.graph.sparql_bridge import SPARQLBridge

    bridge = SPARQLBridge.from_turtle("exports/graph_export.ttl")

    rows = bridge.query('''
        PREFIX base: <https://graphrag.example.com/ontology#>
        SELECT ?entity ?relation ?target WHERE {
            ?entity base:SUPERSEDES ?target .
        }
    ''')

    # Pre-built convenience queries
    rows  = bridge.entity_relations("https://graphrag.example.com/entity/...")
    hier  = bridge.subclass_hierarchy("REGULATION")
    confs = bridge.confidence_summary(min_conf=0.8)
"""

from __future__ import annotations

from pathlib import Path

import structlog
from rdflib import Graph, Namespace
from rdflib.namespace import OWL, RDF, RDFS, XSD

log = structlog.get_logger(__name__)

# Mirror the namespaces defined in scripts/export_rdf.py
BASE  = Namespace("https://graphrag.example.com/ontology#")
INST  = Namespace("https://graphrag.example.com/entity/")
ANNOT = Namespace("https://graphrag.example.com/annotation#")


class SPARQLBridge:
    """Execute SPARQL 1.1 queries against an rdflib Graph.

    The graph is typically an export produced by ``scripts/export_rdf.py``.
    All methods are synchronous — rdflib's SPARQL engine is in-process.

    Parameters
    ----------
    graph :
        A populated rdflib Graph.  Usually obtained via ``from_turtle()``.
    """

    _DEFAULT_NS: dict[str, str] = {
        "base":  str(BASE),
        "inst":  str(INST),
        "annot": str(ANNOT),
        "owl":   str(OWL),
        "rdf":   str(RDF),
        "rdfs":  str(RDFS),
        "xsd":   str(XSD),
    }

    def __init__(self, graph: Graph) -> None:
        self._g = graph

    # ── Constructors ───────────────────────────────────────────────────────────

    @classmethod
    def from_turtle(cls, ttl_path: Path | str) -> "SPARQLBridge":
        """Parse a Turtle file into an rdflib Graph and return a bridge.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(ttl_path)
        if not path.exists():
            raise FileNotFoundError(f"Turtle file not found: {path}")
        g = Graph()
        g.parse(str(path), format="turtle")
        log.info("sparql_bridge.loaded", path=str(path), triples=len(g))
        return cls(g)

    # ── Core query API ─────────────────────────────────────────────────────────

    def query(
        self,
        sparql: str,
        init_ns: dict[str, str] | None = None,
    ) -> list[dict]:
        """Execute a SPARQL 1.1 SELECT query.

        Parameters
        ----------
        sparql :
            A complete SPARQL 1.1 SELECT statement.
        init_ns :
            Extra namespace prefix bindings, merged with the default set
            (base, inst, annot, owl, rdf, rdfs, xsd).

        Returns
        -------
        list[dict]
            One dict per result row; keys are variable names, values are
            strings.  Returns ``[]`` for zero-match queries.

        Raises
        ------
        ValueError
            If the SPARQL is syntactically or semantically invalid.
        """
        ns = {**self._DEFAULT_NS, **(init_ns or {})}
        init_ns_rdflib = {
            k: Namespace(v) if isinstance(v, str) else v
            for k, v in ns.items()
        }

        try:
            result = self._g.query(sparql, initNs=init_ns_rdflib)
        except Exception as exc:
            raise ValueError(f"SPARQL parse/execution error: {exc}") from exc

        rows: list[dict] = []
        if hasattr(result, "vars") and result.vars:
            for row in result:
                rows.append(
                    {
                        str(var): (str(val) if val is not None else "")
                        for var, val in zip(result.vars, row)
                    }
                )
        return rows

    def describe(self, entity_uri: str) -> str:
        """Return a Turtle-serialised neighbourhood for a single entity.

        Collects all triples where the entity appears as subject *or* object
        (equivalent to a SPARQL DESCRIBE under the symmetric CBD strategy).

        Parameters
        ----------
        entity_uri :
            Full URI string, e.g.
            ``"https://graphrag.example.com/entity/default/PERSON/Alice"``.

        Returns
        -------
        str
            Turtle-encoded triples.  Returns an empty Turtle doc if the URI
            is not present in the graph.
        """
        from rdflib import URIRef

        uri = URIRef(entity_uri)
        sub = Graph()
        for triple in self._g.triples((uri, None, None)):
            sub.add(triple)
        for triple in self._g.triples((None, None, uri)):
            sub.add(triple)
        return sub.serialize(format="turtle")

    # ── Pre-built convenience queries ──────────────────────────────────────────

    def entity_relations(self, entity_uri: str) -> list[dict]:
        """All outgoing (relation, target) pairs for an entity URI.

        Excludes rdf:type and rdfs:label metadata triples.
        """
        return self.query(
            f"""
            SELECT ?relation ?target WHERE {{
                <{entity_uri}> ?relation ?target .
                FILTER(?relation != rdf:type)
                FILTER(?relation != rdfs:label)
                FILTER(?relation != rdfs:comment)
            }}
            ORDER BY ?relation
            """
        )

    def subclass_hierarchy(self, root_type: str | None = None) -> list[dict]:
        """All rdfs:subClassOf pairs in the graph.

        Parameters
        ----------
        root_type :
            Unqualified entity type name (e.g. ``"REGULATION"``).  When
            provided, restricts results to pairs whose parent is exactly this
            type URI (direct children only).  Pass ``None`` for all pairs.

        Returns
        -------
        list[dict]
            ``[{"child": uri, "parent": uri}]``
        """
        if root_type:
            root_uri = str(BASE[root_type.upper()])
            sparql = f"""
                SELECT ?child ?parent WHERE {{
                    ?child rdfs:subClassOf ?parent .
                    FILTER(str(?parent) = "{root_uri}")
                }}
                ORDER BY ?child
            """
        else:
            sparql = """
                SELECT ?child ?parent WHERE {
                    ?child rdfs:subClassOf ?parent .
                }
                ORDER BY ?parent ?child
            """
        return self.query(sparql)

    def confidence_summary(self, min_conf: float = 0.0) -> list[dict]:
        """Reified triples whose confidence annotation is >= min_conf.

        Reads ``owl:Axiom`` reification nodes produced by ``export_rdf.py``.

        Returns
        -------
        list[dict]
            ``[{"source": uri, "property": uri, "target": uri,
               "confidence": "0.95"}]``
            sorted by confidence descending.
        """
        return self.query(
            f"""
            SELECT ?source ?property ?target ?confidence WHERE {{
                ?axiom a owl:Axiom ;
                       owl:annotatedSource   ?source ;
                       owl:annotatedProperty ?property ;
                       owl:annotatedTarget   ?target ;
                       annot:confidence      ?confidence .
                FILTER(xsd:float(?confidence) >= {min_conf})
            }}
            ORDER BY DESC(xsd:float(?confidence))
            """
        )
