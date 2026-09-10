"""Tests for graphrag.graph.triplestore — the remote-query half of the
SPARQL story that scripts/load_blazegraph.py alone could never provide
(it could only load a store, never query one back)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.graph.triplestore import (
    RemoteSPARQLEndpoint,
    TripleStoreTarget,
    _auth_from_env,
    _bindings_to_rows,
    remote_sparql_source_from_env,
)


def _mock_client(json_payload: dict | None = None, status_ok: bool = True) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock() if status_ok else MagicMock(
        side_effect=Exception("HTTP error")
    )
    response.json = MagicMock(return_value=json_payload or {})
    response.status_code = 200
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    return client


_SPARQL_JSON_RESULTS = {
    "head": {"vars": ["s", "label"]},
    "results": {
        "bindings": [
            {
                "s": {"type": "uri", "value": "https://example.org/entity/Alice"},
                "label": {"type": "literal", "value": "Alice"},
            }
        ]
    },
}


class TestBindingsToRows:
    def test_matches_sparqlbridge_querys_string_coerced_shape(self):
        # RemoteSPARQLEndpoint must be a drop-in swap for SPARQLBridge under
        # the shared SPARQLSource Protocol -- a caller must not have to
        # branch on which backend answered.
        rows = _bindings_to_rows(_SPARQL_JSON_RESULTS)
        assert rows == [{"s": "https://example.org/entity/Alice", "label": "Alice"}]

    def test_empty_bindings_returns_empty_list(self):
        assert _bindings_to_rows({"results": {"bindings": []}}) == []


class TestRemoteSPARQLEndpointQuery:
    @pytest.mark.asyncio
    async def test_returns_rows_from_a_successful_response(self):
        client = _mock_client(_SPARQL_JSON_RESULTS)
        endpoint = RemoteSPARQLEndpoint("http://store.example/sparql", client=client)

        rows = await endpoint.query("SELECT ?s ?label WHERE { ?s rdfs:label ?label }")

        assert rows == [{"s": "https://example.org/entity/Alice", "label": "Alice"}]
        client.post.assert_awaited_once()
        _, kwargs = client.post.call_args
        assert kwargs["headers"]["Content-Type"] == "application/sparql-query"
        assert kwargs["headers"]["Accept"] == "application/sparql-results+json"

    @pytest.mark.asyncio
    async def test_the_ssrf_guard_still_applies_to_remote_queries(self):
        # The whole reason _reject_unsafe_sparql exists is SERVICE-based SSRF
        # from the local rdflib engine; proxying an unfiltered client query to
        # a remote store must not reopen the same hole through a new path.
        client = _mock_client()
        endpoint = RemoteSPARQLEndpoint("http://store.example/sparql", client=client)

        with pytest.raises(ValueError, match="SERVICE"):
            await endpoint.query(
                "SELECT * WHERE { SERVICE <http://internal.example/admin> { ?s ?p ?o } }"
            )
        client.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_non_select_write_form_is_rejected_before_any_request(self):
        client = _mock_client()
        endpoint = RemoteSPARQLEndpoint("http://store.example/sparql", client=client)

        with pytest.raises(ValueError):
            await endpoint.query("INSERT DATA { <a> <b> <c> }")
        client.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auth_tuple_is_forwarded(self):
        client = _mock_client(_SPARQL_JSON_RESULTS)
        endpoint = RemoteSPARQLEndpoint(
            "http://store.example/sparql", client=client, auth=("user", "pass"),
        )

        await endpoint.query("SELECT ?s WHERE { ?s ?p ?o }")

        _, kwargs = client.post.call_args
        assert kwargs["auth"] == ("user", "pass")

    @pytest.mark.asyncio
    async def test_a_non_json_response_raises_value_error(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = MagicMock(side_effect=ValueError("not json"))
        client = MagicMock()
        client.post = AsyncMock(return_value=response)
        endpoint = RemoteSPARQLEndpoint("http://store.example/sparql", client=client)

        with pytest.raises(ValueError, match="non-JSON"):
            await endpoint.query("SELECT ?s WHERE { ?s ?p ?o }")

    @pytest.mark.asyncio
    async def test_has_no_update_method(self):
        # /kg/sparql/update must keep targeting the local Turtle snapshot
        # even when a remote endpoint is configured for reads -- this class
        # deliberately offers no write path to make that impossible to
        # accidentally wire up.
        assert not hasattr(RemoteSPARQLEndpoint, "update")


class TestVendorURLBuilders:
    def test_blazegraph_matches_load_blazegraph_pys_own_url_shape(self):
        target = TripleStoreTarget(
            "blazegraph", "http://localhost:9999", namespace="kb", context_path="bigdata",
        )
        assert target.query_url == "http://localhost:9999/bigdata/namespace/kb/sparql"
        assert target.load_url == target.query_url

    def test_blazegraph_defaults_match_load_blazegraph_pys_defaults(self):
        target = TripleStoreTarget("blazegraph", "http://localhost:9999")
        assert "/bigdata/namespace/kb/sparql" in target.query_url

    def test_graphdb_requires_a_repository_name(self):
        with pytest.raises(ValueError, match="repository"):
            TripleStoreTarget("graphdb", "http://localhost:7200")

    def test_graphdb_urls(self):
        target = TripleStoreTarget("graphdb", "http://localhost:7200", repository="kg")
        assert target.query_url == "http://localhost:7200/repositories/kg"
        assert target.load_url == "http://localhost:7200/repositories/kg/statements"

    def test_stardog_requires_a_database_name(self):
        with pytest.raises(ValueError, match="database"):
            TripleStoreTarget("stardog", "http://localhost:5820")

    def test_neptune_query_url_but_no_load_url(self):
        target = TripleStoreTarget("neptune", "https://neptune.example:8182")
        assert target.query_url.endswith("/sparql")
        assert target.load_url is None

    def test_unknown_vendor_is_rejected(self):
        with pytest.raises(ValueError, match="unknown triplestore vendor"):
            TripleStoreTarget("not-a-real-vendor", "http://localhost:1234")


class TestTripleStoreTargetLoad:
    @pytest.mark.asyncio
    async def test_posts_turtle_bytes_with_the_correct_content_type(self):
        client = _mock_client()
        target = TripleStoreTarget(
            "blazegraph", "http://localhost:9999", client=client, namespace="kb",
        )

        status = await target.load(b"<a> <b> <c> .")

        assert status == 200
        _, kwargs = client.post.call_args
        assert kwargs["headers"]["Content-Type"] == "text/turtle"
        assert kwargs["content"] == b"<a> <b> <c> ."

    @pytest.mark.asyncio
    async def test_neptune_load_raises_not_implemented_rather_than_guessing(self):
        target = TripleStoreTarget("neptune", "https://neptune.example:8182")
        with pytest.raises(NotImplementedError, match="neptune"):
            await target.load(b"<a> <b> <c> .")


class TestAuthFromEnv:
    def test_parses_user_password(self):
        assert _auth_from_env("alice:secret") == ("alice", "secret")

    def test_missing_colon_returns_none(self):
        assert _auth_from_env("just-a-token") is None

    def test_empty_user_returns_none(self):
        assert _auth_from_env(":secret") is None


class TestRemoteSourceFromEnv:
    def test_unset_endpoint_returns_none(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GRAPHRAG_SPARQL_ENDPOINT", None)
            assert remote_sparql_source_from_env() is None

    def test_configured_endpoint_builds_a_remote_endpoint(self):
        with patch.dict(os.environ, {"GRAPHRAG_SPARQL_ENDPOINT": "http://store.example/sparql"}):
            source = remote_sparql_source_from_env()
        assert isinstance(source, RemoteSPARQLEndpoint)
        assert source._query_url == "http://store.example/sparql"

    def test_auth_env_var_is_applied(self):
        with patch.dict(os.environ, {
            "GRAPHRAG_SPARQL_ENDPOINT": "http://store.example/sparql",
            "GRAPHRAG_SPARQL_AUTH": "user:pass",
        }):
            source = remote_sparql_source_from_env()
        assert source._auth == ("user", "pass")
