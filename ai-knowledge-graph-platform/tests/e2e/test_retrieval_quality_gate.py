"""Deterministic, LLM-free retrieval quality gate (docs/IMPLEMENTATION_AUDIT.md item #2).

The existing golden/aerospace/automotive eval scripts require a live LLM and
a pre-populated corpus -- deliberately excluded from CI (see ci.yml's own
comment). This is the opposite: a tiny, fixed, seeded fixture
(evals/retrieval_gate_fixture.json) run against a disposable Neo4j container
with vector search, the reranker, and the GNN all disabled, so the whole
pipeline is BM25-over-Neo4j only -- no API key, no model download, fully
deterministic. It is a regression tripwire, not a quality benchmark: it
would have caught the A174 cross-tenant/filter leak (tasks/lessons.md)
immediately, since that bug made BM25 return every tenant's/document's
chunks regardless of relevance filtering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from graphrag.evaluation.ir_metrics import recall_at_k, reciprocal_rank  # noqa: E402

_FIXTURE_PATH = ROOT / "evals" / "retrieval_gate_fixture.json"


def _docker_and_testcontainers_available() -> bool:
    try:
        import docker
        import testcontainers  # noqa: F401

        docker.from_env().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_and_testcontainers_available(),
    reason="Docker or testcontainers-python not available",
)

_PASSWORD = "retrieval-gate-e2e-password"
_TENANT = "retrieval-gate-fixture"


@pytest.fixture(scope="module")
def neo4j_container():
    from testcontainers.community.neo4j import Neo4jContainer
    from testcontainers.core.config import testcontainers_config

    previous_max_tries = testcontainers_config.max_tries
    testcontainers_config.max_tries = max(previous_max_tries, 300)
    try:
        with Neo4jContainer("neo4j:5.20-community", password=_PASSWORD) as container:
            yield container
    finally:
        testcontainers_config.max_tries = previous_max_tries


def _client_for(driver):
    """Use Neo4jClient.run() without reading the developer's settings file."""
    from graphrag.graph.neo4j_client import Neo4jClient

    client = Neo4jClient.__new__(Neo4jClient)
    client._driver = driver
    client._filtered_vector_search = False
    client._filtered_vector_indexes = set()
    client._in_flight = 0
    return client


async def _seed_fixture(client, fixture: dict) -> None:
    for doc in fixture["documents"]:
        await client.run(
            "CREATE (d:Document {id: $id, tenant: $tenant, filename: $filename})",
            id=doc["id"], tenant=_TENANT, filename=doc["filename"],
        )
    for chunk in fixture["chunks"]:
        await client.run(
            """
            MATCH (d:Document {id: $document_id, tenant: $tenant})
            CREATE (c:Chunk {id: $id, tenant: $tenant, text: $text})
            CREATE (c)-[:PART_OF]->(d)
            """,
            id=chunk["id"], tenant=_TENANT, text=chunk["text"],
            document_id=chunk["document_id"],
        )


class TestDeterministicRetrievalGate:
    async def test_bm25_retrieval_meets_recall_and_mrr_thresholds(self, neo4j_container) -> None:
        from unittest.mock import patch

        from neo4j import AsyncGraphDatabase

        fixture = json.loads(_FIXTURE_PATH.read_text())
        thresholds = fixture["thresholds"]

        driver = AsyncGraphDatabase.driver(
            neo4j_container.get_connection_url(), auth=("neo4j", _PASSWORD),
        )
        client = _client_for(driver)
        try:
            await client.init_schema()
            await _seed_fixture(client, fixture)
            await client.run("CALL db.awaitIndexes()")

            with (
                patch("graphrag.retrieval.local_search.get_neo4j", return_value=client),
                patch("graphrag.retrieval.bm25_search.get_neo4j", return_value=client),
            ):
                from graphrag.retrieval.local_search import LocalSearch

                local_search = LocalSearch()

            recalls: list[float] = []
            reciprocal_ranks: list[float] = []
            for query in fixture["queries"]:
                result = await local_search.search(
                    query["question"],
                    tenant=_TENANT,
                    config_overrides={
                        "vector_search_enabled": False,
                        "reranker_enabled": False,
                        "gnn_enabled": False,
                    },
                )
                ranked = [c["chunk_id"] for c in result["chunks"]]
                relevant = set(query["expected_relevant"])
                recalls.append(recall_at_k(ranked, relevant, k=5))
                reciprocal_ranks.append(reciprocal_rank(ranked, relevant))

            mean_recall = sum(recalls) / len(recalls)
            mean_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

            assert mean_recall >= thresholds["recall_at_5_min"], (
                f"recall@5 regressed: {mean_recall:.3f} < {thresholds['recall_at_5_min']} "
                f"(per-query: {recalls})"
            )
            assert mean_mrr >= thresholds["mrr_min"], (
                f"MRR regressed: {mean_mrr:.3f} < {thresholds['mrr_min']} "
                f"(per-query: {reciprocal_ranks})"
            )
        finally:
            await client.run("MATCH (n {tenant: $tenant}) DETACH DELETE n", tenant=_TENANT)
            await driver.close()
