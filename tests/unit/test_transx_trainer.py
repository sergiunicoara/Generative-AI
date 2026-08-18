"""Unit tests for TransXTrainer — tenant-scoped persistence of trained
relation embeddings (F13).

TransXTrainer shares its `rel_emb` dict by reference with
EdgeEmbeddingService's own cache, which is now keyed (tenant, relation). The
trainer's internal reads/writes to that shared dict, and its final persist
loop, must use the same key shape — a plain-string key here would be
invisible to get_relation_embedding()'s lookups after training, and would
break the Cypher parameter binding in the persist loop (a tuple bound as a
string relation name).

See docs/context_graph_gap_plan.md F13. No test previously existed for this
class at all.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from graphrag.graph.transx_trainer import TransXTrainer


def _entity_row(emb):
    return {"embedding": emb}


def _triple_row(head, rel, tail):
    return {"head_emb": head, "relation": rel, "tail_emb": tail}


def _make_trainer(triples, entities):
    """A trainer whose neo4j_client.run() answers the trainer's three
    queries in the order it issues them: triples, then entities."""
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(side_effect=[triples, entities])
    rel_emb: dict[tuple[str, str], list[float]] = {}
    trainer = TransXTrainer(neo4j_client=neo4j, rel_emb=rel_emb, embed_dim=4)
    return trainer, neo4j, rel_emb


class TestSharedCacheKeyShape:
    async def test_trained_vectors_land_under_tenant_relation_tuple(self):
        triples = [_triple_row([0.1] * 4, "CEO_OF", [0.2] * 4)]
        entities = [_entity_row([0.1] * 4), _entity_row([0.2] * 4), _entity_row([0.3] * 4)]
        trainer, neo4j, rel_emb = _make_trainer(triples, entities)

        # A trailing empty result for the persist loop's MERGE calls.
        neo4j.run.side_effect = list(neo4j.run.side_effect) + [[]] * 5

        await trainer.train(tenant="acme", epochs=1, neg_samples=1, batch_size=10)

        assert ("acme", "CEO_OF") in rel_emb
        assert "CEO_OF" not in rel_emb  # no legacy bare-string key leaked in

    async def test_persist_writes_tenant_matching_the_cache_key(self):
        triples = [_triple_row([0.1] * 4, "CEO_OF", [0.2] * 4)]
        entities = [_entity_row([0.1] * 4), _entity_row([0.2] * 4), _entity_row([0.3] * 4)]
        trainer, neo4j, rel_emb = _make_trainer(triples, entities)
        neo4j.run.side_effect = list(neo4j.run.side_effect) + [[]] * 5

        await trainer.train(tenant="acme", epochs=1, neg_samples=1, batch_size=10)

        merge_calls = [
            c for c in neo4j.run.await_args_list
            if c.args and "MERGE (re:RelationEmbedding" in c.args[0]
        ]
        assert merge_calls, "expected at least one persist write"
        for call in merge_calls:
            assert call.kwargs["tenant"] == "acme"
            assert call.kwargs["rel"] == "CEO_OF"

    async def test_persist_query_includes_tenant_in_merge_key(self):
        triples = [_triple_row([0.1] * 4, "CEO_OF", [0.2] * 4)]
        entities = [_entity_row([0.1] * 4), _entity_row([0.2] * 4), _entity_row([0.3] * 4)]
        trainer, neo4j, rel_emb = _make_trainer(triples, entities)
        neo4j.run.side_effect = list(neo4j.run.side_effect) + [[]] * 5

        await trainer.train(tenant="acme", epochs=1, neg_samples=1, batch_size=10)

        merge_call = next(
            c for c in neo4j.run.await_args_list
            if c.args and "MERGE (re:RelationEmbedding" in c.args[0]
        )
        assert "MERGE (re:RelationEmbedding {relation: $rel, tenant: $tenant})" in merge_call.args[0]


class TestNoTriplesOrEntities:
    async def test_no_triples_returns_error_without_touching_cache(self):
        trainer, neo4j, rel_emb = _make_trainer([], [])
        result = await trainer.train(tenant="acme")
        assert result["error"] == "no_triples_found"
        assert rel_emb == {}
