"""Unit tests for AliasRegistry — normalization, resolution, cache update."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.graph.alias_registry import AliasRegistry, _normalize


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
