"""No API key, no network call (model is cached locally after first
download) — proves the local embedding provider actually produces real,
normalized, differentiating vectors."""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.asyncio


async def test_embed_returns_one_vector_per_text_at_the_declared_dimension(embedding_provider):
    vectors = await embedding_provider.embed(["Volkswagen Group", "Acme Corp"])
    assert len(vectors) == 2
    assert len(vectors[0]) == embedding_provider.dimension == 384


async def test_vectors_are_normalized(embedding_provider):
    vectors = await embedding_provider.embed(["Volkswagen Group"])
    norm = math.sqrt(sum(x * x for x in vectors[0]))
    assert abs(norm - 1.0) < 1e-4


async def test_identical_text_embeds_identically(embedding_provider):
    a, b = await embedding_provider.embed(["Volkswagen Group", "Volkswagen Group"])
    assert a == b


async def test_similar_names_score_higher_than_unrelated_ones(embedding_provider):
    from src.resolution.scoring import cosine_similarity

    mention_vec, vw_vec, unrelated_vec = await embedding_provider.embed(
        ["Volks Wagen", "Volkswagen Group", "Totally Unrelated Company"]
    )
    similar_score = cosine_similarity(mention_vec, vw_vec)
    unrelated_score = cosine_similarity(mention_vec, unrelated_vec)
    assert similar_score > unrelated_score
