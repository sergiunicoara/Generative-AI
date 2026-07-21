"""Unit tests for cross-ontology entity linking — CrossOntologyLinker."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from graphrag.graph.alias_registry import AmbiguousMatch
from graphrag.graph.cross_ontology_linker import CrossOntologyLinker, _entity_uri

EX = Namespace("https://customer.example.com/onto#")


def _write_external_ttl(tmp_path, entries: list[tuple[str, str, str | None]]) -> str:
    """entries: [(local_name, label, rdf_type_or_None)]"""
    g = Graph()
    for local_name, label, rdf_type in entries:
        subj = EX[local_name]
        g.add((subj, RDFS.label, Literal(label)))
        g.add((subj, RDF.type, OWL.NamedIndividual))
        if rdf_type:
            g.add((subj, RDF.type, EX[rdf_type]))
    path = tmp_path / "external.ttl"
    g.serialize(str(path), format="turtle")
    return str(path)


def _make_linker(mock_registry, embedder=None) -> CrossOntologyLinker:
    neo4j = AsyncMock()
    with patch("graphrag.graph.cross_ontology_linker.get_alias_registry", return_value=mock_registry):
        linker = CrossOntologyLinker(neo4j, tenant="telecom", embedder=embedder)
    return linker


# ── _entity_uri ──────────────────────────────────────────────────────────────

class TestEntityUri:
    def test_matches_export_rdf_scheme(self):
        uri = _entity_uri("BRN-2201", "CIRCUIT", "telecom")
        assert str(uri) == "https://graphrag.example.com/entity/telecom/CIRCUIT/BRN-2201"

    def test_url_safe_escaping(self):
        uri = _entity_uri("Aurora Fintech SRL", "ORG", "telecom")
        assert " " not in str(uri)


# ── Candidate extraction ─────────────────────────────────────────────────────

class TestExtractCandidates:
    def test_extracts_label_and_type_hint(self, tmp_path):
        path = _write_external_ttl(tmp_path, [("ne1", "Router BRN-2201-A", "NetworkElement")])
        mock_registry = MagicMock()
        linker = _make_linker(mock_registry)

        candidates = linker._extract_candidates(path)
        assert len(candidates) == 1
        assert candidates[0].label == "Router BRN-2201-A"
        assert candidates[0].type_hint == "NETWORKELEMENT"
        assert candidates[0].external_uri.endswith("ne1")

    def test_no_type_defaults_to_concept(self, tmp_path):
        path = _write_external_ttl(tmp_path, [("x1", "Some Entity", None)])
        mock_registry = MagicMock()
        linker = _make_linker(mock_registry)

        candidates = linker._extract_candidates(path)
        assert candidates[0].type_hint == "CONCEPT"

    def test_dedupes_by_subject_uri(self, tmp_path):
        g = Graph()
        subj = EX["dup1"]
        g.add((subj, RDFS.label, Literal("First Label")))
        g.add((subj, RDFS.label, Literal("Second Label")))
        path = tmp_path / "dup.ttl"
        g.serialize(str(path), format="turtle")

        mock_registry = MagicMock()
        linker = _make_linker(mock_registry)
        candidates = linker._extract_candidates(str(path))
        assert len(candidates) == 1


# ── link() — exact/fuzzy path ────────────────────────────────────────────────

class TestLinkExactFuzzy:
    @pytest.mark.asyncio
    async def test_exact_match_auto_links(self, tmp_path):
        path = _write_external_ttl(tmp_path, [("ne1", "BRN-2201", "Circuit")])
        mock_registry = MagicMock()
        mock_registry.load = AsyncMock()
        mock_registry.resolve = MagicMock(return_value=("BRN-2201", "CIRCUIT"))
        linker = _make_linker(mock_registry)

        with patch.object(linker, "_review_queue_enabled", return_value=False):
            result = await linker.link(path)

        assert len(result.auto_linked) == 1
        assert result.auto_linked[0]["internal_name"] == "BRN-2201"
        assert len(result.same_as_graph) == 1
        triple = next(iter(result.same_as_graph))
        assert triple[1] == OWL.sameAs

    @pytest.mark.asyncio
    async def test_ambiguous_match_enqueues_for_review(self, tmp_path):
        path = _write_external_ttl(tmp_path, [("ne1", "Circuit BRN 2201", "Circuit")])
        mock_registry = MagicMock()
        mock_registry.load = AsyncMock()
        mock_registry.resolve = MagicMock(return_value=AmbiguousMatch(
            candidate=("BRN-2201", "CIRCUIT"), score=76.0, match_type="fuzzy",
        ))
        linker = _make_linker(mock_registry)

        mock_review = AsyncMock()
        mock_review.enqueue = AsyncMock(return_value="item-abc")
        with patch("graphrag.graph.review_queue.ReviewQueueService", return_value=mock_review):
            result = await linker.link(path)

        assert len(result.queued_for_review) == 1
        assert result.queued_for_review[0]["item_id"] == "item-abc"
        mock_review.enqueue.assert_called_once()
        kwargs = mock_review.enqueue.call_args[1]
        assert kwargs["match_type"] == "cross_ontology_fuzzy"
        assert kwargs["tenant"] == "telecom"

    @pytest.mark.asyncio
    async def test_no_match_and_no_embedder_is_unmatched(self, tmp_path):
        path = _write_external_ttl(tmp_path, [("ne1", "Totally Unknown Thing", "Circuit")])
        mock_registry = MagicMock()
        mock_registry.load = AsyncMock()
        mock_registry.resolve = MagicMock(return_value=None)
        linker = _make_linker(mock_registry, embedder=None)

        with patch.object(linker, "_review_queue_enabled", return_value=False):
            result = await linker.link(path)

        assert len(result.unmatched) == 1
        assert result.unmatched[0]["reason"] == "no_match"

    @pytest.mark.asyncio
    async def test_review_queue_disabled_falls_to_unmatched(self, tmp_path):
        path = _write_external_ttl(tmp_path, [("ne1", "Circuit BRN 2201", "Circuit")])
        mock_registry = MagicMock()
        mock_registry.load = AsyncMock()
        mock_registry.resolve = MagicMock(return_value=AmbiguousMatch(
            candidate=("BRN-2201", "CIRCUIT"), score=76.0, match_type="fuzzy",
        ))
        linker = _make_linker(mock_registry)

        with patch.object(linker, "_review_queue_enabled", return_value=False):
            result = await linker.link(path)

        assert len(result.queued_for_review) == 0
        assert result.unmatched[0]["reason"] == "review_queue_disabled"


# ── link() — embedding fallback ──────────────────────────────────────────────

class TestLinkEmbeddingFallback:
    @pytest.mark.asyncio
    async def test_embedding_auto_match(self, tmp_path):
        path = _write_external_ttl(tmp_path, [("ne1", "Unlabeled Router X", "NetworkElement")])
        mock_registry = MagicMock()
        mock_registry.load = AsyncMock()
        mock_registry.resolve = MagicMock(return_value=None)
        mock_registry.find_duplicate_by_embedding = AsyncMock(
            return_value=("NE-BUC-CORE-03", "NETWORKELEMENT", 0.97)
        )

        mock_embedder = AsyncMock()
        mock_embedder.embed_texts = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        linker = _make_linker(mock_registry, embedder=mock_embedder)
        with patch.object(linker, "_review_queue_enabled", return_value=False):
            result = await linker.link(path)

        assert len(result.auto_linked) == 1
        assert result.auto_linked[0]["match_type"] == "embedding"
        mock_registry.find_duplicate_by_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedding_ambiguous_enqueues(self, tmp_path):
        path = _write_external_ttl(tmp_path, [("ne1", "Some Router Y", "NetworkElement")])
        mock_registry = MagicMock()
        mock_registry.load = AsyncMock()
        mock_registry.resolve = MagicMock(return_value=None)
        mock_registry.find_duplicate_by_embedding = AsyncMock(return_value=None)
        mock_registry.find_candidate_by_embedding = AsyncMock(
            return_value=("NE-PLO-EDGE-07", "NETWORKELEMENT", 0.88)
        )

        mock_embedder = AsyncMock()
        mock_embedder.embed_texts = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        linker = _make_linker(mock_registry, embedder=mock_embedder)
        mock_review = AsyncMock()
        mock_review.enqueue = AsyncMock(return_value="item-xyz")
        with patch("graphrag.graph.review_queue.ReviewQueueService", return_value=mock_review):
            result = await linker.link(path)

        assert len(result.queued_for_review) == 1
        kwargs = mock_review.enqueue.call_args[1]
        assert kwargs["match_type"] == "cross_ontology_embedding"

    @pytest.mark.asyncio
    async def test_embed_failure_falls_to_unmatched(self, tmp_path):
        path = _write_external_ttl(tmp_path, [("ne1", "Broken Embed Thing", "NetworkElement")])
        mock_registry = MagicMock()
        mock_registry.load = AsyncMock()
        mock_registry.resolve = MagicMock(return_value=None)

        mock_embedder = AsyncMock()
        mock_embedder.embed_texts = AsyncMock(side_effect=RuntimeError("embed service down"))

        linker = _make_linker(mock_registry, embedder=mock_embedder)
        with patch.object(linker, "_review_queue_enabled", return_value=False):
            result = await linker.link(path)

        assert len(result.unmatched) == 1
        assert result.unmatched[0]["reason"] == "no_match"


# ── merge_graphs ─────────────────────────────────────────────────────────────

class TestMergeGraphs:
    def test_merges_two_files(self, tmp_path):
        g1 = Graph()
        g1.add((EX["a"], RDFS.label, Literal("A")))
        p1 = tmp_path / "g1.ttl"
        g1.serialize(str(p1), format="turtle")

        g2 = Graph()
        g2.add((EX["b"], RDFS.label, Literal("B")))
        p2 = tmp_path / "g2.ttl"
        g2.serialize(str(p2), format="turtle")

        merged = CrossOntologyLinker.merge_graphs(str(p1), str(p2))
        assert len(merged) == 2

    def test_merges_with_extra_same_as_graph(self, tmp_path):
        g1 = Graph()
        g1.add((EX["a"], RDFS.label, Literal("A")))
        p1 = tmp_path / "g1.ttl"
        g1.serialize(str(p1), format="turtle")

        extra = Graph()
        extra.add((EX["a"], OWL.sameAs, URIRef("https://graphrag.example.com/entity/telecom/CIRCUIT/BRN-2201")))

        merged = CrossOntologyLinker.merge_graphs(str(p1), extra=extra)
        assert len(merged) == 2
        assert (EX["a"], OWL.sameAs, None) in merged


# ── LinkResult.summary ───────────────────────────────────────────────────────

class TestLinkResultSummary:
    def test_summary_counts(self):
        from graphrag.graph.cross_ontology_linker import LinkResult
        result = LinkResult(
            auto_linked=[{}, {}],
            queued_for_review=[{}],
            unmatched=[{}, {}, {}],
        )
        s = result.summary()
        assert "2 auto-linked" in s
        assert "1 queued for review" in s
        assert "3 unmatched" in s
        assert "6 candidates total" in s
