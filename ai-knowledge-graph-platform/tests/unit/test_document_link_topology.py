"""Regression guards for explicit document topology and contextual entities."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.core.models import Entity
from graphrag.enterprise.models import ACLState, AccessContext, DocumentAccessPolicy, DocumentLink
from graphrag.graph.neo4j_client import Neo4jClient
from graphrag.ingestion.document_loader import extract_document_links, load_document_content
from graphrag.retrieval.trajectory import graph_edge_ids


def _client(result=None) -> Neo4jClient:
    client = Neo4jClient.__new__(Neo4jClient)
    client.run = AsyncMock(return_value=result if result is not None else [])
    return client


def test_html_link_extraction_is_explicit_normalised_and_does_not_follow_urls() -> None:
    html = b"""
      <html><body>
        <a href="/sites/legal/target.html#section-2">Target policy</a>
        <a href="mailto:legal@example.com">Mail legal</a>
        <script>window.location = 'https://not-a-document.example'</script>
      </body></html>
    """

    assert "Target policy" in load_document_content("source.html", html)
    links = extract_document_links(
        "source.html", html, base_url="https://sharepoint.example/sites/legal/source.html",
        source_system="sharepoint", source_version="v7",
    )

    assert len(links) == 1
    assert links[0].target_url == "https://sharepoint.example/sites/legal/target.html"
    assert links[0].anchor_text == "Target policy"
    assert links[0].source_system == "sharepoint"
    assert links[0].source_version == "v7"


@pytest.mark.asyncio
async def test_link_persistence_keeps_provenance_tenant_and_acl_snapshot() -> None:
    client = _client([[], [], [{"references": 1}]])
    policy = DocumentAccessPolicy(
        mode="restricted", state=ACLState.KNOWN,
        allow_principals=["group:legal"], requires_group_resolution=True,
    )

    result = await client.merge_document_links(
        "doc-a", [DocumentLink(target_url="https://sharepoint.example/doc-b", anchor_text="B")],
        tenant="legal", access_policy={
            "access_mode": policy.mode,
            "acl_state": policy.state.value,
            "allow_principals": policy.allow_principals,
            "deny_principals": policy.deny_principals,
            "requires_group_resolution": policy.requires_group_resolution,
        },
    )

    cypher = client.run.await_args.args[0]
    params = client.run.await_args.kwargs
    assert result == 1
    assert "DocumentLinkReference" in cypher
    assert "LINKS_TO" in cypher
    assert "link.observed_at" in cypher
    assert "link.allow_principals" in cypher
    assert params["tenant"] == "legal"
    assert params["allow_principals"] == ["group:legal"]


@pytest.mark.asyncio
async def test_reingest_removes_links_that_are_no_longer_explicit() -> None:
    client = _client([[], []])

    await client.merge_document_links("doc-a", [], tenant="legal")

    calls = [call.args[0] for call in client.run.await_args_list]
    assert "DELETE link" in calls[0]
    assert "DETACH DELETE ref" in calls[1]


@pytest.mark.asyncio
async def test_late_target_reconciliation_materialises_edge_from_durable_reference() -> None:
    client = _client([{"resolved": 1}])

    resolved = await client.reconcile_document_links("doc-b", tenant="legal")

    cypher = client.run.await_args.args[0]
    assert resolved == 1
    assert "DECLARES_LINK" in cypher
    assert "ref.target_url = target.source_url" in cypher
    assert "MERGE (source)-[link:LINKS_TO" in cypher
    assert "link.allow_principals = ref.allow_principals" in cypher


@pytest.mark.asyncio
async def test_link_retrieval_enforces_source_edge_and_target_acl_before_text() -> None:
    client = _client([])

    result = await client.get_linked_document_chunks(
        ["seed-a"], tenant="legal", query_embedding=[0.1, 0.2],
        access_context=AccessContext(principals=["group:legal"], groups_resolved=True),
    )

    cypher = client.run.await_args.args[0]
    assert result == []
    assert "-[link:LINKS_TO" in cypher
    assert "document_access_predicate" not in cypher
    assert cypher.count("acl_state") >= 3
    assert "chunk.text AS text" in cypher
    assert client.run.await_args.kwargs["acl_principals"] == ["group:legal"]


@pytest.mark.asyncio
async def test_contextual_representations_are_system_scoped_not_name_scoped() -> None:
    client = _client([{"assertions": 1}])
    customer = Entity(name="Customer", type="CONCEPT")
    customer.redirect_to("Customer", "CONCEPT")

    await client.merge_contextual_entity_representations(
        "chunk-crm", [customer], source_system="crm", source_doc_id="crm-doc", tenant="tenant-a",
    )
    crm_params = client.run.await_args.kwargs
    crm_cypher = client.run.await_args.args[0]
    await client.merge_contextual_entity_representations(
        "chunk-erp", [customer], source_system="erp", source_doc_id="erp-doc", tenant="tenant-a",
    )
    erp_params = client.run.await_args.kwargs

    assert "SystemRepresentation" in crm_cypher
    assert "HAS_SYSTEM_REPRESENTATION" in crm_cypher
    assert "ContextualAssertion" in crm_cypher
    assert crm_params["source_system"] == "crm"
    assert erp_params["source_system"] == "erp"
    assert crm_params["rows"] == erp_params["rows"]


def test_trajectory_includes_document_link_edges_for_golden_expectations() -> None:
    assert graph_edge_ids({
        "document_link_edges": [{"src": "A.html", "relation": "LINKS_TO", "tgt": "B.html"}],
    }) == ["A.html|LINKS_TO|B.html"]
