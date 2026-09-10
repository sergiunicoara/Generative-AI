"""SPARQL 1.1 Protocol content negotiation for POST /kg/sparql.

The route's original shape was a custom JSON API: a POST body of
``{"query": ..., "namespaces": ...}`` in, ``{"rows": [...], "count": N}``
out. No off-the-shelf SPARQL client (YASGUI, rdflib's own ``SPARQLStore``,
Jena, Comunica) can talk to that. This module adds the standard forms on top
without changing the legacy ones -- ``application/json`` (or an absent
Content-Type) and a default/``application/json`` Accept still produce exactly
what they always did, so every existing caller is unaffected.

Request side
------------
``Content-Type: application/sparql-query`` -- the raw query is the entire
body, per SPARQL 1.1 Protocol section 2.1.2.

``Content-Type: application/x-www-form-urlencoded`` -- the query is the
``query`` form field, per section 2.1.3.

Response side
-------------
``Accept: application/sparql-results+json`` / ``+xml`` / ``text/csv``
select the corresponding W3C SPARQL Results serialization instead of the
legacy shape. Negotiation here is deliberately simple -- exact substring
match against a small fixed set, not a full RFC 7231 q-value parser -- which
is enough for the handful of SPARQL tools this exists to interoperate with.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs

from fastapi import HTTPException, Request

LEGACY_CONTENT_TYPE = "application/json"
SPARQL_RESULTS_JSON = "application/sparql-results+json"
SPARQL_RESULTS_XML = "application/sparql-results+xml"
SPARQL_RESULTS_CSV = "text/csv"

# rdflib's Result.serialize() format name for each standard content type.
_RDFLIB_FORMAT_BY_CONTENT_TYPE = {
    SPARQL_RESULTS_JSON: "json",
    SPARQL_RESULTS_XML: "xml",
    SPARQL_RESULTS_CSV: "csv",
}

_STANDARD_CONTENT_TYPES = (SPARQL_RESULTS_JSON, SPARQL_RESULTS_XML, SPARQL_RESULTS_CSV)


async def parse_sparql_query_request(request: Request) -> tuple[str, dict[str, str]]:
    """Return (query, namespaces) from a POST /kg/sparql request body.

    ``namespaces`` is a platform-specific convenience representable only in
    the legacy JSON shape -- there is no standard SPARQL 1.1 Protocol carrier
    for it. The standard content types below return an empty dict; the
    bridge's own default prefixes (base/inst/annot/owl/rdf/rdfs/skos/xsd)
    still apply regardless, and a query needing anything beyond those can
    PREFIX it inline.
    """
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    body = await request.body()

    if content_type in ("", LEGACY_CONTENT_TYPE):
        try:
            payload = json.loads(body) if body else {}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        query = payload.get("query")
        if not isinstance(query, str) or not query:
            raise HTTPException(status_code=400, detail="'query' is required")
        namespaces = payload.get("namespaces") or {}
        if not isinstance(namespaces, dict):
            raise HTTPException(status_code=400, detail="'namespaces' must be an object")
        return query, namespaces

    if content_type == "application/sparql-query":
        query = body.decode("utf-8", errors="replace")
        if not query.strip():
            raise HTTPException(status_code=400, detail="request body is empty")
        return query, {}

    if content_type == "application/x-www-form-urlencoded":
        parsed = parse_qs(body.decode("utf-8", errors="replace"))
        values = parsed.get("query") or []
        if not values or not values[0]:
            raise HTTPException(status_code=400, detail="'query' form field is required")
        return values[0], {}

    raise HTTPException(status_code=415, detail=f"unsupported Content-Type: {content_type!r}")


def negotiate_accept(accept_header: str) -> str:
    """Pick the response content-type this route will serve.

    Returns one of the three standard content types, or LEGACY_CONTENT_TYPE
    as the default -- covering an absent header, "*/*", "application/json",
    and anything else unrecognised.
    """
    for candidate in _STANDARD_CONTENT_TYPES:
        if candidate in accept_header:
            return candidate
    return LEGACY_CONTENT_TYPE


def serialize_typed_result(result, content_type: str) -> bytes:
    """Serialize an rdflib SPARQL query Result for one of the standard types.

    Callers must have already restricted ``content_type`` to a value from
    _STANDARD_CONTENT_TYPES (e.g. via negotiate_accept()) -- this raises
    ValueError for anything else rather than guessing a fallback format.
    """
    fmt = _RDFLIB_FORMAT_BY_CONTENT_TYPE.get(content_type)
    if fmt is None:
        raise ValueError(f"no SPARQL results serializer for {content_type!r}")
    return result.serialize(format=fmt)


__all__ = [
    "LEGACY_CONTENT_TYPE",
    "SPARQL_RESULTS_CSV",
    "SPARQL_RESULTS_JSON",
    "SPARQL_RESULTS_XML",
    "negotiate_accept",
    "parse_sparql_query_request",
    "serialize_typed_result",
]
