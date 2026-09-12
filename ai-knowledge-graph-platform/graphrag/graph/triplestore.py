"""Remote SPARQL 1.1 query + vendor-neutral Graph Store load.

``SPARQLBridge`` (sparql_bridge.py) only ever wraps an in-process rdflib
Graph parsed from a Turtle export -- a real triplestore (Stardog, GraphDB,
Neptune, RDFox, Virtuoso, Blazegraph) could be *loaded into* via
scripts/load_blazegraph.py, but never *queried*. This module is the missing
half: ``RemoteSPARQLEndpoint`` queries a live SPARQL 1.1 Protocol endpoint,
and ``TripleStoreTarget`` generalises the Blazegraph-specific load path
(scripts/load_blazegraph.py) to other vendors' Graph Store HTTP Protocol
surfaces.

Security -- non-negotiable
---------------------------
SPARQL 1.1 query is standardised, so *querying* needs no vendor branching --
only the URL to reach the endpoint differs. ``_reject_unsafe_sparql`` from
sparql_bridge.py is applied to every outbound query here too, before it ever
leaves the process. ADR-0001's addendum records that the platform's own
Blazegraph mirror "has no authentication and its endpoint accepts SPARQL
Update plus LOAD/SERVICE" -- proxying unfiltered client SPARQL text to any
remote store would reopen exactly the SSRF/write hole those guards exist to
close. This module never sends a query that guard would reject.

Verification status of the vendor Graph Store URL builders
------------------------------------------------------------
Blazegraph (see scripts/load_blazegraph.py's own docstring: tested against
lyrasis/blazegraph:2.1.5) and GraphDB (tested against
ontotext/graphdb:10.8.1, the Free/unlicensed "GRAPHDB_LITE" edition -- see
tests/e2e/test_live_graphdb.py) are verified end to end, including
``ensure_namespace()``'s auto-provisioning path for both. Stardog, RDFox and
Virtuoso follow each vendor's published SPARQL 1.1 Graph Store HTTP Protocol
documentation but are not exercised against a running instance anywhere in
this repo -- Stardog and RDFox have no obtainable license here, and Virtuoso
has no verified image. Treat them as a documented starting point, not a
verified claim. Note GraphDB 11.0+ requires a registered license to start at
all (Ontotext's own licensing docs); the 10.x tag above runs unlicensed.
Amazon Neptune does not expose a standard direct Graph Store Protocol load
path (AWS recommends its bulk loader from S3), so ``load()`` raises
``NotImplementedError`` for it rather than emitting a request that would
silently fail or behave unexpectedly.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

import httpx
import structlog

from graphrag.graph.sparql_bridge import _reject_unsafe_sparql

log = structlog.get_logger(__name__)

_DEFAULT_TIMEOUT = 30.0


@runtime_checkable
class SPARQLSource(Protocol):
    """What a caller needs from any SPARQL query backend, local or remote.

    ``SPARQLBridge.query()`` (sync) and ``SPARQLBridge.aquery()`` (async,
    added alongside this module) satisfy this structurally, as does
    ``RemoteSPARQLEndpoint.query()`` below -- callers that only need to run a
    read query can depend on this Protocol instead of a concrete class.
    """

    async def query(
        self, sparql: str, init_ns: dict[str, str] | None = None,
    ) -> list[dict]: ...


def _auth_from_env(value: str) -> tuple[str, str] | None:
    """Parse GRAPHRAG_SPARQL_AUTH="user:pass" into an httpx basic-auth tuple.

    Returns None for an unset/malformed value rather than raising -- an
    auth misconfiguration should surface as the remote store rejecting the
    request (a clear 401/403), not as a startup crash.
    """
    if ":" not in value:
        return None
    user, _, password = value.partition(":")
    return (user, password) if user else None


def _bindings_to_rows(payload: dict[str, Any]) -> list[dict]:
    """SPARQL 1.1 ``application/sparql-results+json`` -> SPARQLBridge's shape.

    SPARQLBridge.query() coerces every term to a plain string (see its
    docstring); mirrored here so RemoteSPARQLEndpoint is a drop-in swap for
    it under the shared SPARQLSource Protocol, not a differently-shaped
    result a caller has to branch on.
    """
    bindings = payload.get("results", {}).get("bindings", [])
    return [
        {var: str(term.get("value", "")) for var, term in row.items()}
        for row in bindings
    ]


class RemoteSPARQLEndpoint:
    """Query a live SPARQL 1.1 Protocol endpoint over HTTP.

    Read-only by design: this class has no ``update()``. ``/kg/sparql/update``
    keeps targeting the local Turtle snapshot even when a remote endpoint is
    configured for reads -- writing through to the mirror would silently
    diverge it from the export that regenerates it (see
    api/routes/kg/knowledge.py).
    """

    def __init__(
        self,
        query_url: str,
        *,
        auth: tuple[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._query_url = query_url
        self._auth = auth
        self._client = client
        self._timeout = timeout

    async def _post_query(self, sparql: str) -> dict[str, Any]:
        _reject_unsafe_sparql(sparql)

        headers = {
            "Content-Type": "application/sparql-query",
            "Accept": "application/sparql-results+json",
        }
        try:
            if self._client is not None:
                response = await self._client.post(
                    self._query_url, content=sparql, headers=headers, auth=self._auth,
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        self._query_url, content=sparql, headers=headers, auth=self._auth,
                    )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ValueError(f"remote SPARQL query failed: {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise ValueError(
                f"remote endpoint returned a non-JSON response: {exc}"
            ) from exc

    async def query(
        self, sparql: str, init_ns: dict[str, str] | None = None,
    ) -> list[dict]:
        """Execute a read-only SPARQL 1.1 query against the remote endpoint.

        ``init_ns`` is accepted for interface parity with
        ``SPARQLBridge.query()`` but unused here: a remote store resolves
        prefixes from the query text itself (standard SPARQL PREFIX clauses),
        it has no notion of the bridge's local default-namespace convenience.
        """
        return _bindings_to_rows(await self._post_query(sparql))

    async def query_json_raw(self, sparql: str) -> dict[str, Any]:
        """The remote store's own ``application/sparql-results+json`` body,
        unflattened.

        A real triplestore already speaks the standard results format
        natively -- for a caller that wants to *serve* a standards-compliant
        response (POST /kg/sparql with Accept: application/sparql-results+json,
        see graphrag/graph/sparql_results.py), passing this through verbatim
        is both simpler and more correct than round-tripping through
        query()'s flattened, string-coerced rows and reconstructing the
        typed shape.
        """
        return await self._post_query(sparql)


# ── Vendor Graph Store URL builders ─────────────────────────────────────────
#
# Each entry returns (query_url, load_url | None). load_url is None where the
# vendor has no standard direct-POST Graph Store Protocol path (see module
# docstring re: Neptune).

def _blazegraph_urls(base_url: str, **kw: str) -> tuple[str, str | None]:
    # Matches scripts/load_blazegraph.py exactly -- the one verified vendor.
    context_path = kw.get("context_path", "bigdata")
    namespace = kw.get("namespace", "kb")
    url = f"{base_url.rstrip('/')}/{context_path.strip('/')}/namespace/{namespace}/sparql"
    return url, url


def _graphdb_urls(base_url: str, **kw: str) -> tuple[str, str | None]:
    repository = kw.get("repository", "")
    if not repository:
        raise ValueError("GraphDB requires a 'repository' name")
    base = f"{base_url.rstrip('/')}/repositories/{repository}"
    return base, f"{base}/statements"


def _stardog_urls(base_url: str, **kw: str) -> tuple[str, str | None]:
    database = kw.get("database", "")
    if not database:
        raise ValueError("Stardog requires a 'database' name")
    base = f"{base_url.rstrip('/')}/{database}"
    return f"{base}/query", f"{base}?default"


def _rdfox_urls(base_url: str, **kw: str) -> tuple[str, str | None]:
    datastore = kw.get("datastore", "")
    if not datastore:
        raise ValueError("RDFox requires a 'datastore' name")
    base = f"{base_url.rstrip('/')}/datastores/{datastore}"
    return f"{base}/sparql", f"{base}/content?graph=default"


def _virtuoso_urls(base_url: str, **kw: str) -> tuple[str, str | None]:
    root = base_url.rstrip('/')
    return f"{root}/sparql", f"{root}/sparql-graph-crud?graph=default"


def _neptune_urls(base_url: str, **kw: str) -> tuple[str, str | None]:
    # Neptune exposes a standard SPARQL query endpoint but not a supported
    # direct-POST Graph Store Protocol load path -- AWS's documented bulk
    # load mechanism is from S3. See module docstring.
    return f"{base_url.rstrip('/')}/sparql", None


_VENDOR_URL_BUILDERS = {
    "blazegraph": _blazegraph_urls,
    "graphdb": _graphdb_urls,
    "stardog": _stardog_urls,
    "rdfox": _rdfox_urls,
    "virtuoso": _virtuoso_urls,
    "neptune": _neptune_urls,
}


class TripleStoreTarget:
    """Query and load a named vendor's triplestore over its HTTP surface.

    Generalises scripts/load_blazegraph.py's Blazegraph-specific URL
    construction (~3 of its ~35 functional lines) to the other vendors named
    in ``_VENDOR_URL_BUILDERS``. The write path is still exactly what
    load_blazegraph.py already does -- a raw Turtle document POSTed with
    ``Content-Type: text/turtle`` -- just against a URL resolved for the
    configured vendor instead of hardcoded to Blazegraph's shape.
    """

    def __init__(
        self,
        vendor: str,
        base_url: str,
        *,
        auth: tuple[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
        **vendor_kwargs: str,
    ) -> None:
        builder = _VENDOR_URL_BUILDERS.get(vendor)
        if builder is None:
            raise ValueError(
                f"unknown triplestore vendor {vendor!r}; expected one of "
                f"{', '.join(sorted(_VENDOR_URL_BUILDERS))}"
            )
        self.vendor = vendor
        self.query_url, self.load_url = builder(base_url, **vendor_kwargs)
        self._auth = auth
        self._client = client
        self._timeout = timeout
        # Kept for ensure_namespace(): Blazegraph's namespace-management
        # endpoint (POST <base>/<context_path>/namespace) is a different URL
        # from query_url/load_url (<base>/<context_path>/namespace/<ns>/sparql),
        # and building it needs base_url/context_path/namespace back out --
        # cheaper to keep the inputs than to parse them back out of query_url.
        self._base_url = base_url
        self._vendor_kwargs = vendor_kwargs

    def _endpoint(self) -> RemoteSPARQLEndpoint:
        return RemoteSPARQLEndpoint(
            self.query_url, auth=self._auth, client=self._client, timeout=self._timeout,
        )

    async def query(
        self, sparql: str, init_ns: dict[str, str] | None = None,
    ) -> list[dict]:
        return await self._endpoint().query(sparql, init_ns=init_ns)

    async def _post(self, url: str, **kwargs) -> httpx.Response:
        """Shared raw-POST helper for the management calls below -- both
        branches of ensure_namespace() need "use the injected client if
        present, else a scratch one," and duplicating that once per vendor
        would drift the two copies over time."""
        if self._client is not None:
            return await self._client.post(url, auth=self._auth, **kwargs)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.post(url, auth=self._auth, **kwargs)

    async def ensure_namespace(self) -> None:
        """Create this target's namespace/repository if it doesn't already
        exist. Idempotent -- an "already exists" response is treated as
        success, not an error. A no-op for every vendor without a verified
        per-request provisioning call (Stardog databases and Virtuoso have
        none confirmed here).
        """
        if self.vendor == "blazegraph":
            await self._ensure_blazegraph_namespace()
        elif self.vendor == "graphdb":
            await self._ensure_graphdb_repository()

    async def _ensure_blazegraph_namespace(self) -> None:
        """Idempotent -- a 409 Conflict (namespace already exists) is
        treated as success, not an error.

        Confirmed live (2026-09, lyrasis/blazegraph:2.1.5): POSTing a Turtle
        document to a namespace that was never created returns 404, not an
        implicit auto-create -- found by actually running this module's own
        documented `load_blazegraph.py --namespace acme_kb` usage example
        against a real container, which failed 404 before this existed.
        Blazegraph's default "kb" namespace ships pre-created, which is why
        that specific case was never caught before.
        """
        context_path = self._vendor_kwargs.get("context_path", "bigdata")
        namespace = self._vendor_kwargs.get("namespace", "kb")
        management_url = f"{self._base_url.rstrip('/')}/{context_path.strip('/')}/namespace"
        # Minimal required property is com.bigdata.rdf.sail.namespace; the
        # rest are the same defaults Blazegraph's own "kb" namespace ships
        # with, kept explicit rather than relying on server-side defaults
        # that could differ across image versions.
        body = (
            f"com.bigdata.rdf.sail.namespace={namespace}\n"
            "com.bigdata.rdf.sail.truthMaintenance=false\n"
            "com.bigdata.rdf.store.AbstractTripleStore.quads=false\n"
            "com.bigdata.rdf.store.AbstractTripleStore.geoSpatial=false\n"
            "com.bigdata.rdf.store.AbstractTripleStore.statementIdentifiers=false\n"
            "com.bigdata.rdf.store.AbstractTripleStore.textIndex=false\n"
            "com.bigdata.rdf.store.AbstractTripleStore.axiomsClass=com.bigdata.rdf.axioms.NoAxioms\n"
        )
        try:
            response = await self._post(
                management_url, content=body, headers={"Content-Type": "text/plain"},
            )
            if response.status_code == 409:
                return  # already exists -- exactly what we want
            response.raise_for_status()
            log.info("triplestore.namespace_created", vendor=self.vendor, namespace=namespace)
        except httpx.HTTPError as exc:
            raise ValueError(f"triplestore namespace creation failed: {exc}") from exc

    async def _ensure_graphdb_repository(self) -> None:
        """Idempotent -- "repository already exists" is treated as success.

        Confirmed live (2026-09, ontotext/graphdb:10.8.1, Free/unlicensed
        edition): docs/demos/energy_asset_intelligence.md previously
        documented creating a GraphDB repository *manually through the
        GraphDB UI* -- there was no automated path at all, unlike
        Blazegraph's namespace (which at least 404s cleanly). Unlike
        Blazegraph's 409, an existing repository here returns **400** with a
        JSON body `{"message": "Repository <id> already exists."}` --
        confirmed by actually creating the same repository twice against a
        running container, not assumed from the vendor's docs (which don't
        document the conflict status at all).
        """
        repository = self._vendor_kwargs.get("repository", "")
        management_url = f"{self._base_url.rstrip('/')}/rest/repositories"
        # Minimal RDF4J/GraphDB SAIL repository config -- graphdb:Sail is the
        # vendor's own default rule-set-backed store, the same kind the
        # GraphDB Workbench UI creates when an operator does this by hand.
        config_ttl = (
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
            "@prefix rep: <http://www.openrdf.org/config/repository#> .\n"
            "@prefix sr: <http://www.openrdf.org/config/repository/sail#> .\n"
            "@prefix sail: <http://www.openrdf.org/config/sail#> .\n"
            "@prefix graphdb: <http://www.ontotext.com/config/graphdb#> .\n\n"
            "[] a rep:Repository ;\n"
            f'    rep:repositoryID "{repository}" ;\n'
            '    rdfs:label "" ;\n'
            "    rep:repositoryImpl [\n"
            '        rep:repositoryType "graphdb:SailRepository" ;\n'
            "        sr:sailImpl [\n"
            '            sail:sailType "graphdb:Sail"\n'
            "        ]\n"
            "    ] .\n"
        )
        try:
            response = await self._post(
                management_url,
                files={"config": ("repo-config.ttl", config_ttl.encode(), "text/turtle")},
            )
            if response.status_code == 400:
                message = ""
                try:
                    message = str(response.json().get("message", ""))
                except ValueError:
                    pass
                if "already exists" in message.lower():
                    return  # already exists -- exactly what we want
            response.raise_for_status()
            log.info("triplestore.repository_created", vendor=self.vendor, repository=repository)
        except httpx.HTTPError as exc:
            raise ValueError(f"triplestore repository creation failed: {exc}") from exc

    async def load(self, ttl_bytes: bytes) -> int:
        """POST a Turtle document to this vendor's Graph Store endpoint.

        Creates the target namespace first when the vendor supports it and
        it doesn't already exist (see ensure_namespace()) -- Blazegraph
        returns 404 rather than auto-creating on load, so skipping this for
        any namespace other than the pre-provisioned default would silently
        break every non-default namespace.

        Returns the HTTP status code on success; raises for network errors,
        a non-2xx response, or a vendor with no supported load path.
        """
        if self.load_url is None:
            raise NotImplementedError(
                f"{self.vendor} has no supported direct-POST Graph Store "
                f"Protocol load path -- see this module's docstring"
            )
        await self.ensure_namespace()
        headers = {"Content-Type": "text/turtle"}
        try:
            if self._client is not None:
                response = await self._client.post(
                    self.load_url, content=ttl_bytes, headers=headers, auth=self._auth,
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        self.load_url, content=ttl_bytes, headers=headers, auth=self._auth,
                    )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ValueError(f"triplestore load failed: {exc}") from exc

        log.info(
            "triplestore.loaded",
            vendor=self.vendor,
            endpoint=self.load_url,
            bytes=len(ttl_bytes),
            status=response.status_code,
        )
        return response.status_code


def remote_sparql_source_from_env() -> RemoteSPARQLEndpoint | None:
    """Build a RemoteSPARQLEndpoint from GRAPHRAG_SPARQL_* env vars, or None.

    Mirrors api/limiter.py's ``_storage_uri()`` pattern: read from the process
    environment only, return None (not raise) when unset, so a deployment
    that never configures a remote endpoint is completely unaffected --
    /kg/sparql keeps reading the local Turtle snapshot exactly as before.

    GRAPHRAG_SPARQL_ENDPOINT is the fully-resolved SPARQL query URL (e.g.
    "http://host:9999/bigdata/namespace/kb/sparql"), not a bare host --
    resolving a base host into a vendor-specific query URL needs a vendor
    plus namespace/repository/database, which belongs to TripleStoreTarget
    (construct one directly when that resolution is needed, e.g. for the
    load path). GRAPHRAG_SPARQL_VENDOR is accepted only for observability
    (logged, not used to build the URL) so a misconfigured deployment can
    still be diagnosed from its logs.
    """
    endpoint = os.getenv("GRAPHRAG_SPARQL_ENDPOINT", "").strip()
    if not endpoint:
        return None
    auth_raw = os.getenv("GRAPHRAG_SPARQL_AUTH", "").strip()
    auth = _auth_from_env(auth_raw) if auth_raw else None
    log.info(
        "triplestore.remote_source_configured",
        endpoint=endpoint,
        vendor=os.getenv("GRAPHRAG_SPARQL_VENDOR", "").strip() or "unspecified",
    )
    return RemoteSPARQLEndpoint(endpoint, auth=auth)


__all__ = [
    "RemoteSPARQLEndpoint",
    "SPARQLSource",
    "TripleStoreTarget",
    "remote_sparql_source_from_env",
]
