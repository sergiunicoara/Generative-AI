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

import asyncio
import functools
import os
import re
import tempfile
from pathlib import Path

import structlog
from rdflib import Graph, Namespace, URIRef
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

# SPARQL 1.1 section 19.2: backslash-u (4 hex digits) and backslash-U (8 hex
# digits) code-point escapes are expanded over the WHOLE query text before
# tokenizing, so an escaped spelling of SERVICE (backslash-u0053 then ERVICE) IS
# the keyword SERVICE. rdflib does exactly that
# (rdflib.plugins.sparql.parser.expandUnicodeEscapes),
# so the guard must scan the expanded text or an escaped keyword is invisible to
# it. Same pattern and semantics as rdflib's: 4 hex digits, optionally 8, any
# case, and no awareness of a preceding backslash. The backslash is built at
# runtime so no editor or tool can pre-expand the sequences in this comment.
_BACKSLASH = chr(92)
_UNICODE_ESCAPE_RE = re.compile(
    re.escape(_BACKSLASH) + r"u([0-9a-f]{4}(?:[0-9a-f]{4})?)", re.IGNORECASE
)

# One left-to-right lexer for every state in which text is NOT SPARQL keywords:
# long strings, short strings, IRIs, comments and variable names. A single
# alternation (instead of one regex per state applied in turn) is the point: the
# token that STARTS first wins, so a ``#`` inside a string or an IRI is part of
# that token and can never begin a comment, whatever order the states appear in.
# Applying the states one after another is exactly what produced two bypasses
# (comments before strings; then IRIs not known to the comment stripper).
_SPARQL_TOKEN_RE = re.compile(
    # triple-quoted "long strings": may span lines and hold lone quotes or '#'
    r'(?P<long>"""(?:[^"\\]|\\.|"{1,2}(?!"))*"""|\'\'\'(?:[^\'\\]|\\.|\'{1,2}(?!\'))*\'\'\')'
    # short strings: the grammar forbids raw CR/LF inside them
    r'|(?P<string>"(?:[^"\\\n\r]|\\.)*"|\'(?:[^\'\\\n\r]|\\.)*\')'
    # IRIREF, exactly the characters the grammar allows (no whitespace, no
    # < > " { } | ^ ` or backslash), so the less-than operator and a ``<`` that
    # is followed by whitespace or a brace are never swallowed as an IRI
    r'|(?P<iri><[^<>"{}|^`\\\x00-\x20]*>)'
    # a comment runs to the end of the line
    r'|(?P<comment>#[^\n\r]*)'
    # a variable may legitimately be named ?ADD
    r'|(?P<var>[?$][A-Za-z_][A-Za-z0-9_]*)'
)


def _expand_unicode_escapes(sparql: str) -> str:
    """Expand code-point escapes the way the SPARQL grammar (and rdflib) does.

    An escape naming an invalid code point makes the engine refuse the query, so
    the guard refuses it too.
    """

    def expand(match: re.Match) -> str:
        try:
            return chr(int(match.group(1), 16))
        except (ValueError, OverflowError) as exc:
            raise ValueError("Invalid unicode escape in SPARQL query") from exc

    return _UNICODE_ESCAPE_RE.sub(expand, sparql)


def _blank_strings_and_comments(
    sparql: str, *, blank_iris: bool = False, blank_vars: bool = False
) -> str:
    """Return the text a keyword scan should look at.

    Escapes are expanded first (see ``_UNICODE_ESCAPE_RE``), then strings are
    reduced to ``""`` and comments removed in a single lexer pass
    (``_SPARQL_TOKEN_RE``). IRIs always *protect* a ``#`` inside them from being
    read as a comment; they are additionally reduced to ``<>`` only when
    ``blank_iris`` is set (the update guard), and variables to ``?v`` only when
    ``blank_vars`` is set. The query guard leaves IRI text visible, so a keyword
    inside an IRI is still refused there: a false positive, never a false
    negative, if the lexer and the engine ever disagree about where an IRI ends.

    History, because the order of states is what keeps going wrong. (1) Comments
    were once stripped before strings, so a ``#`` inside ``"Room #101"`` hid a
    real ``SERVICE``/``LOAD`` later on the line from the scan while rdflib still
    executed it. Fixed by blanking strings first. (2) That still stripped
    comments without knowing about IRIs, so ``<http://ex/a#b> ?p ?o SERVICE
    <http://169.254.169.254/> {?a ?b ?c}`` hid its SERVICE the same way; and
    (3) an escaped spelling such as ``\\u0053ERVICE`` was never seen at all. See
    tests/unit/test_sparql_injection_adversarial.py.
    """

    def blank(match: re.Match) -> str:
        kind = match.lastgroup
        if kind in ("long", "string"):
            return '""'
        if kind == "comment":
            return ""
        if kind == "iri":
            return "<>" if blank_iris else match.group(0)
        return "?v" if blank_vars else match.group(0)

    return _SPARQL_TOKEN_RE.sub(blank, _expand_unicode_escapes(sparql))


def _reject_unsafe_sparql(sparql: str) -> None:
    """Raise ValueError unless ``sparql`` is a read-only, non-federated query.

    Comments and string literals are stripped first so a keyword appearing
    inside them (e.g. a label containing the word "delete") does not trip the
    check, and conversely so a forbidden keyword cannot be smuggled past it.
    """
    stripped = _blank_strings_and_comments(sparql)
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

    Same comment/string-stripping approach as ``_reject_unsafe_sparql``, plus
    IRIs and variables, so a keyword inside a literal, comment, IRI or variable
    name can neither trip nor evade the check. Blanking IRIs is what keeps
    ``INSERT DATA { <a> <b> <http://ex/SERVICE> }`` from being misread as
    federation, while a bare ``SERVICE`` clause is still caught.
    """
    stripped = _blank_strings_and_comments(sparql, blank_iris=True, blank_vars=True)
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

    def _init_ns_rdflib(self, init_ns: dict[str, str] | None) -> dict[str, Namespace]:
        ns = {**self._DEFAULT_NS, **(init_ns or {})}
        return {k: Namespace(v) if isinstance(v, str) else v for k, v in ns.items()}

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

    def query_typed(self, sparql: str, init_ns: dict[str, str] | None = None):
        """Execute a SPARQL 1.1 SELECT and return rdflib's own Result object.

        ``query()`` coerces every term to a plain string, which is right for
        that method's callers but loses the URI/literal/bnode distinction and
        datatype/language tag a standards-compliant SPARQL results
        serialization (application/sparql-results+json etc., see
        graphrag/graph/sparql_results.py) requires. This is a sibling, not a
        replacement -- query()'s string-coerced contract is pinned by tests
        and used by callers with nothing to gain from typed terms.

        Same safety guard as query(); raises ValueError on an unsafe or
        invalid query.
        """
        _reject_unsafe_sparql(sparql)
        try:
            return self._g.query(sparql, initNs=self._init_ns_rdflib(init_ns))
        except Exception as exc:
            raise ValueError(f"SPARQL parse/execution error: {exc}") from exc

    async def aquery(
        self,
        sparql: str,
        init_ns: dict[str, str] | None = None,
    ) -> list[dict]:
        """Async wrapper over ``query()``, off the event loop.

        rdflib's SPARQL engine is synchronous and, for a graph of any real
        size, not fast -- ``POST /kg/sparql`` called ``query()`` directly
        inside an async route handler, so every query blocked the event loop
        (and every other in-flight request on that worker) for its full
        duration. Offloading via ``run_in_executor`` is the same fix
        api/limiter.py's docstring describes for why that module moved off
        slowapi's synchronous Redis calls.

        Exists alongside the unchanged sync ``query()`` rather than replacing
        it: query()'s synchronous, string-coerced contract is pinned by
        tests and used by callers with no event loop to offload onto (the
        convenience methods below call ``query()`` directly, for instance).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(self.query, sparql, init_ns=init_ns)
        )

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

    def stamp_sparql_update(self, tenant: str = "") -> int:
        """Record in the graph itself that it was mutated by SPARQL Update.

        The export is a projection of Neo4j (ADR-0001). Once an update is
        persisted the two have diverged, and without a marker that divergence
        is invisible -- the file still looks like a clean export. Stamping the
        ontology node means any later reader, including a plain SELECT against
        this same file, can tell it is no longer purely export-derived.

        Returns the triple count after stamping.
        """
        from datetime import datetime, timezone

        from rdflib import Literal

        ont = URIRef("https://graphrag.example.com/ontology")
        self._g.add((
            ont,
            ANNOT.sparqlUpdatedAt,
            Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime),
        ))
        if tenant:
            self._g.add((ont, ANNOT.sparqlUpdatedBy, Literal(tenant)))
        return len(self._g)

    def save(self, path: Path | str, rdf_format: str = "turtle") -> None:
        """Atomically serialize the current in-memory graph to disk.

        Use after ``update()`` to persist a mutation -- e.g. overwriting the
        tenant export file that ``query()``/``from_turtle()`` read.

        The write goes to a temporary file in the destination directory and
        is then renamed over the target. Serialising straight onto the live
        export would leave it truncated if the process died mid-write, and
        that file is what every subsequent ``POST /kg/sparql`` reads.
        ``os.replace`` is atomic on both POSIX and Windows.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(out.parent), prefix=f".{out.name}.", suffix=".tmp"
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            self._g.serialize(destination=str(tmp), format=rdf_format)
            os.replace(tmp, out)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
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
