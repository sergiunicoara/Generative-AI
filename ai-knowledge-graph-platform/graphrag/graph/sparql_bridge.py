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

import re

import structlog
from rdflib import Graph, Namespace
from rdflib.namespace import OWL, RDF, RDFS, XSD

log = structlog.get_logger(__name__)

# Read-only forms. Anything else (INSERT/DELETE/LOAD/CLEAR/DROP/CREATE/ADD/
# MOVE/COPY) mutates the in-memory graph and has no business being reachable
# from a read-scoped HTTP endpoint.
_ALLOWED_FORMS = ("SELECT", "ASK", "CONSTRUCT", "DESCRIBE")

# SERVICE is SPARQL 1.1 federation: rdflib will issue an outbound HTTP request
# to whatever endpoint the query names. Left open, /kg/sparql is a server-side
# request-forgery primitive against anything the API container can reach.
_FORBIDDEN = ("SERVICE", "LOAD", "INSERT", "DELETE", "CLEAR", "DROP",
              "CREATE", "ADD", "MOVE", "COPY")

# Update forms. Unlike the read path, these are expected here -- update()
# only ever mutates the in-memory rdflib Graph the bridge was constructed
# with; it never reaches Neo4j, and the caller decides separately (via
# save()) whether the result is persisted anywhere.
_ALLOWED_UPDATE_FORMS = ("INSERT", "DELETE", "CLEAR", "DROP",
                          "CREATE", "ADD", "MOVE", "COPY")

# SERVICE and LOAD stay forbidden even for updates: both make rdflib issue an
# outbound HTTP request to an attacker-controlled URL (SSRF), which has
# nothing to do with mutating the local graph.
_FORBIDDEN_UPDATE = ("SERVICE", "LOAD")

_COMMENT_RE = re.compile(r"#[^\n]*")
_STRING_RE  = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')


def _reject_unsafe_sparql(sparql: str) -> None:
    """Raise ValueError unless ``sparql`` is a read-only, non-federated query.

    Comments and string literals are stripped first so a keyword appearing
    inside them (e.g. a label containing the word "delete") does not trip the
    check, and conversely so a forbidden keyword cannot be smuggled past it.
    """
    stripped = _STRING_RE.sub('""', _COMMENT_RE.sub("", sparql))
    upper = stripped.upper()

    if not any(re.search(rf"\b{form}\b", upper) for form in _ALLOWED_FORMS):
        raise ValueError(
            f"Only {', '.join(_ALLOWED_FORMS)} SPARQL queries are permitted"
        )
    for kw in _FORBIDDEN:
        if re.search(rf"\b{kw}\b", upper):
            raise ValueError(f"'{kw}' is not permitted in a read-only SPARQL query")


def _reject_unsafe_update(sparql: str) -> None:
    """Raise ValueError unless ``sparql`` is a non-federated SPARQL Update.

    Same comment/string-stripping approach as ``_reject_unsafe_sparql`` so a
    keyword inside a literal or comment can neither trip nor evade the check.
    """
    stripped = _STRING_RE.sub('""', _COMMENT_RE.sub("", sparql))
    upper = stripped.upper()

    if not any(re.search(rf"\b{form}\b", upper) for form in _ALLOWED_UPDATE_FORMS):
        raise ValueError(
            f"Only {', '.join(_ALLOWED_UPDATE_FORMS)} SPARQL update forms are permitted"
        )
    for kw in _FORBIDDEN_UPDATE:
        if re.search(rf"\b{kw}\b", upper):
            raise ValueError(f"'{kw}' is not permitted in a SPARQL update")

# Mirror the namespaces defined in scripts/export_rdf.py
BASE  = Namespace("https://graphrag.example.com/ontology#")
INST  = Namespace("https://graphrag.example.com/entity/")
ANNOT = Namespace("https://graphrag.example.com/annotation#")
SKOS  = Namespace("http://www.w3.org/2004/02/skos/core#")


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
        "skos":  str(SKOS),
        "xsd":   str(XSD),
    }

    def __init__(self, graph: Graph) -> None:
        self._g = graph

    # ── Query safety ───────────────────────────────────────────────────────────

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
            (base, inst, annot, owl, rdf, rdfs, skos, xsd).

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
        _reject_unsafe_sparql(sparql)

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

    def update(
        self,
        sparql: str,
        init_ns: dict[str, str] | None = None,
    ) -> int:
        """Execute a SPARQL 1.1 Update against the in-memory graph.

        Mutates ``self._g`` in place (INSERT/DELETE/CLEAR/DROP/CREATE/ADD/
        MOVE/COPY are all permitted -- they only ever touch this in-memory
        rdflib Graph, never Neo4j). Call ``save()`` afterwards to persist
        the result; otherwise the mutation is scoped to this bridge instance.

        Parameters
        ----------
        sparql :
            A complete SPARQL 1.1 Update statement (or ``;``-separated
            sequence of them).
        init_ns :
            Extra namespace prefix bindings, merged with the default set.

        Returns
        -------
        int
            The graph's triple count after the update.

        Raises
        ------
        ValueError
            If the SPARQL contains SERVICE/LOAD, uses no recognised update
            form, or fails to parse/execute.
        """
        _reject_unsafe_update(sparql)

        ns = {**self._DEFAULT_NS, **(init_ns or {})}
        init_ns_rdflib = {
            k: Namespace(v) if isinstance(v, str) else v
            for k, v in ns.items()
        }

        try:
            self._g.update(sparql, initNs=init_ns_rdflib)
        except Exception as exc:
            raise ValueError(f"SPARQL update parse/execution error: {exc}") from exc

        return len(self._g)

    def save(self, path: Path | str, rdf_format: str = "turtle") -> None:
        """Serialize the current in-memory graph to disk.

        Use after ``update()`` to persist a mutation -- e.g. overwriting the
        tenant export file that ``query()``/``from_turtle()`` read.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._g.serialize(destination=str(out), format=rdf_format)
        log.info("sparql_bridge.saved", path=str(out), triples=len(self._g))

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
