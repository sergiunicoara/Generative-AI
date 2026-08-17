"""Unit tests for AliasRegistry — normalization, resolution, cache update."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.graph.alias_registry import (
    AliasRegistry,
    _normalize,
    _normalize_regulatory,
    _normalize_ro,
    _stem_ro_token,
)


# ── _normalize ─────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_lowercase(self):
        assert _normalize("SpaceX") == "spacex"

    def test_strips_punctuation(self):
        assert _normalize("Apple, Inc.") == "apple inc"

    def test_collapses_whitespace(self):
        assert _normalize("foo  bar") == "foo bar"
        assert _normalize("foo   bar") == "foo bar"

    def test_strips_leading_trailing(self):
        assert _normalize("  hello  ") == "hello"

    def test_empty_string(self):
        assert _normalize("") == ""

    def test_normalization_is_idempotent(self):
        key = _normalize("Elon  Musk!")
        assert _normalize(key) == key

    def test_consistent_with_registry_load(self):
        """Ensure write-key (graph_writer) equals read-key (registry.resolve)."""
        raw = "Foo  Bar!"
        assert _normalize(raw) == _normalize(_normalize(raw))


# ── _normalize_regulatory ────────────────────────────────────────────────────────

class TestNormalizeRegulatory:
    """Regulatory-agency prefix stripping so 'EASA AD 2022-0201' / 'AD 2022-0201'
    (and other regulator+identifier variants) converge to one canonical key,
    enabling forward-chaining transitivity across a supersession chain. See
    INF-01 in evals/golden_set.json — the space-only prefix pattern silently
    never matched the hyphenated form real corpora actually use
    ('FAA-AD-2022-03-07'), leaving it permanently un-deduped against its
    space-separated twin ('AD 2022-03-07')."""

    def test_space_separated_prefix_stripped(self):
        assert _normalize_regulatory("EASA AD 2022-0201") == _normalize_regulatory("AD 2022-0201")

    def test_hyphenated_prefix_stripped(self):
        assert _normalize_regulatory("FAA-AD-2022-03-07") == _normalize_regulatory("AD 2022-03-07")

    def test_hyphenated_and_space_separated_converge(self):
        assert (
            _normalize_regulatory("FAA-AD-2020-05-11")
            == _normalize_regulatory("FAA AD 2020-05-11")
            == _normalize_regulatory("AD 2020-05-11")
        )

    def test_unprefixed_name_unchanged_by_stripping(self):
        assert _normalize_regulatory("AD 2022-03-07") == _normalize("AD 2022-03-07")

    def test_non_regulatory_name_unaffected(self):
        assert _normalize_regulatory("Boeing 737 MAX") == _normalize("Boeing 737 MAX")


# ── _normalize_ro / _stem_ro_token ──────────────────────────────────────────────

class TestNormalizeRo:
    def test_furnizor_variants_converge(self):
        assert _normalize_ro("furnizor") == "furnizor"
        assert _normalize_ro("furnizori") == "furnizor"
        assert _normalize_ro("furnizorul") == "furnizor"
        assert _normalize_ro("Furnizorii") == "furnizor"
        assert _normalize_ro("furnizorilor") == "furnizor"

    def test_phrase_variants_converge(self):
        assert _normalize_ro("furnizori activi") == _normalize_ro("furnizorii activi")

    def test_short_words_not_over_stemmed(self):
        # "audit" ends in "it", not a stripped suffix -- unchanged either way
        assert _stem_ro_token("audit") == "audit"
        # "casa" minus "a" -> "cas" (3 chars, meets _RO_MIN_STEM_LEN) -- allowed
        assert _stem_ro_token("casa") == "cas"
        # A 3-letter word ending in a stemmable suffix must not be hollowed out
        # below _RO_MIN_STEM_LEN (e.g. "lei" minus "i" -> "le", only 2 chars).
        assert len(_stem_ro_token("lei")) >= 3 or _stem_ro_token("lei") == "lei"

    def test_idempotent(self):
        key = _normalize_ro("Furnizorii activi!")
        assert _normalize_ro(key) == key


# ── AliasRegistry.resolve ─────────────────────────────────────────────────────

class TestResolve:
    def _registry(self, entries: dict) -> AliasRegistry:
        neo4j = AsyncMock()
        reg = AliasRegistry(neo4j, tenant="test")
        reg._exact = entries
        reg._loaded = True
        return reg

    def test_exact_match(self):
        reg = self._registry({_normalize("SpaceX"): ("SpaceX", "ORG")})
        assert reg.resolve("SpaceX") == ("SpaceX", "ORG")

    def test_case_insensitive_match(self):
        reg = self._registry({_normalize("SpaceX"): ("SpaceX", "ORG")})
        assert reg.resolve("spacex") == ("SpaceX", "ORG")
        assert reg.resolve("SPACEX") == ("SpaceX", "ORG")

    def test_punctuation_ignored(self):
        reg = self._registry({_normalize("Apple Inc"): ("Apple Inc", "ORG")})
        assert reg.resolve("Apple, Inc.") == ("Apple Inc", "ORG")

    def test_whitespace_collapse(self):
        reg = self._registry({_normalize("Elon Musk"): ("Elon Musk", "PERSON")})
        assert reg.resolve("Elon  Musk") == ("Elon Musk", "PERSON")

    def test_unknown_returns_none(self):
        reg = self._registry({})
        assert reg.resolve("Unknown Corp") is None

    def test_alias_resolves_to_canonical(self):
        entries = {
            _normalize("SpaceX"): ("SpaceX", "ORG"),
            _normalize("Space Exploration Technologies"): ("SpaceX", "ORG"),
        }
        reg = self._registry(entries)
        assert reg.resolve("Space Exploration Technologies") == ("SpaceX", "ORG")

    def test_romanian_stem_fallback_match(self):
        """No exact entry for 'furnizorii', but the stem table maps
        'furnizor' -> ('furnizor', 'ORG') -- resolve() should fall back to it."""
        reg = self._registry({_normalize("furnizor"): ("furnizor", "ORG")})
        reg._stemmed = {_normalize_ro("furnizor"): ("furnizor", "ORG")}
        assert reg.resolve("furnizorii") == ("furnizor", "ORG")
        assert reg.resolve("Furnizorilor") == ("furnizor", "ORG")


# ── AliasRegistry.register_alias ──────────────────────────────────────────────

class TestRegisterAlias:
    async def test_updates_in_memory_cache(self):
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        reg = AliasRegistry(neo4j, tenant="test")
        reg._loaded = True

        await reg.register_alias(
            raw_value="SX",
            canonical_name="SpaceX",
            canonical_type="ORG",
        )

        assert reg.resolve("SX") == ("SpaceX", "ORG")

    async def test_neo4j_run_called_once(self):
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        reg = AliasRegistry(neo4j, tenant="test")
        reg._loaded = True

        await reg.register_alias("SX", "SpaceX", "ORG")
        assert neo4j.run.call_count == 1

    async def test_normalized_key_used_for_cache(self):
        """register_alias must normalize the key so resolve() finds it."""
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        reg = AliasRegistry(neo4j, tenant="test")
        reg._loaded = True

        await reg.register_alias("Space X!!!", "SpaceX", "ORG")
        # resolve normalizes too — both paths must produce the same key
        assert reg.resolve("Space X!!!") == ("SpaceX", "ORG")


# ── AliasRegistry.load ────────────────────────────────────────────────────────

class TestLoad:
    async def test_load_populates_exact(self):
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[
            {"canonical_name": "SpaceX", "canonical_type": "ORG",
             "aliases": ["Space X", "SXC"]},
        ])
        reg = AliasRegistry(neo4j, tenant="default")
        await reg.load()

        assert reg._loaded is True
        # Canonical + 2 aliases = 3 entries
        assert len(reg._exact) == 3
        assert reg.resolve("SpaceX") == ("SpaceX", "ORG")
        assert reg.resolve("Space X") == ("SpaceX", "ORG")
        assert reg.resolve("SXC") == ("SpaceX", "ORG")

    async def test_load_clears_stale_entries(self):
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        reg = AliasRegistry(neo4j, tenant="default")
        reg._exact = {_normalize("OldEntity"): ("OldEntity", "ORG")}

        await reg.load()
        assert len(reg._exact) == 0   # cleared on reload


# ── Embedding-search ANN candidate pool (A148) ─────────────────────────────────
# entity_embeddings is a shared index across all tenants — a small,
# hardcoded k (previously 5/10) could starve a tenant's own true duplicate
# out of the candidate pool if other tenants scored higher for the same
# query vector, causing a silent false negative on dedup.

class TestFindDuplicateByEmbeddingFetchK:
    async def test_uses_fetch_k_not_hardcoded_five(self):
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        reg = AliasRegistry(neo4j, tenant="aerospace")

        await reg.find_duplicate_by_embedding([0.1, 0.2], "ORG")

        query = neo4j.run.call_args.args[0]
        kwargs = neo4j.run.call_args.kwargs
        assert "$fetch_k" in query
        assert ", 5, " not in query  # the old hardcoded literal is gone
        assert kwargs["fetch_k"] == 100


class TestFindCandidateByEmbeddingFetchK:
    async def test_uses_fetch_k_not_hardcoded_ten(self):
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        reg = AliasRegistry(neo4j, tenant="aerospace")

        await reg.find_candidate_by_embedding([0.1, 0.2], "ORG")

        query = neo4j.run.call_args.args[0]
        kwargs = neo4j.run.call_args.kwargs
        assert "$fetch_k" in query
        assert ", 10, " not in query  # the old hardcoded literal is gone
        assert kwargs["fetch_k"] == 100
