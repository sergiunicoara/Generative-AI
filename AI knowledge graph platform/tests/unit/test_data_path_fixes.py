"""Unit tests for round-3 audit data-path fixes.

Covers:
- Confidence clamping in extractor (P1)
- Embedder count mismatch detection (P1)
- Alias normalization key consistency in graph_writer (P1)
- Real p50/p95 percentile computation in kpi_tracker (P2)
- Aware datetime usage (no naive utcnow) (P2)
- Extractor handles None/empty LLM response (P2)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Confidence clamping ────────────────────────────────────────────────────────

class TestConfidenceClamping:
    """Extractor must clamp LLM confidence to [0, 1] before creating Relations."""

    def _make_chunk(self):
        from graphrag.core.models import Chunk
        return Chunk(document_id="doc1", text="A uses B.", chunk_index=0)

    def _payload(self, confidence):
        """Return the raw JSON string Groq's generate() would produce."""
        return json.dumps({
            "entities": [
                {"name": "A", "type": "PRODUCT", "description": ""},
                {"name": "B", "type": "PRODUCT", "description": ""},
            ],
            "relations": [
                {"source": "A", "target": "B", "relation": "USES", "confidence": confidence}
            ],
        })

    @staticmethod
    def _llm_stub(payload: str):
        """A get_llm() replacement whose generate() returns `payload`."""
        stub = MagicMock()
        stub.generate = AsyncMock(return_value=payload)
        return stub

    def _build_extractor(self, response):
        """Patch the API call so extract() returns our synthetic response."""
        from graphrag.ingestion.extractor import Extractor
        with patch("graphrag.ingestion.extractor.genai") as mock_genai:
            e = Extractor.__new__(Extractor)
            e._client = MagicMock()
            e._model_name = "test-model"
            e._gen_config = MagicMock()
            e._entity_types = ["PRODUCT"]
        return e

    async def test_confidence_above_one_is_clamped(self):
        from graphrag.ingestion.extractor import Extractor
        extractor = Extractor.__new__(Extractor)
        extractor._client = MagicMock()
        extractor._model_name = "m"
        extractor._gen_config = MagicMock()
        extractor._entity_types = ["PRODUCT"]

        chunk = self._make_chunk()
        stub = self._llm_stub(self._payload(confidence=1.5))

        with patch("graphrag.ingestion.extractor.get_llm", return_value=stub):
            _, relations = await extractor.extract(chunk)

        assert len(relations) == 1
        assert relations[0].confidence <= 1.0, "Confidence must not exceed 1.0"

    async def test_confidence_below_zero_is_clamped(self):
        from graphrag.ingestion.extractor import Extractor
        extractor = Extractor.__new__(Extractor)
        extractor._client = MagicMock()
        extractor._model_name = "m"
        extractor._gen_config = MagicMock()
        extractor._entity_types = ["PRODUCT"]

        chunk = self._make_chunk()
        stub = self._llm_stub(self._payload(confidence=-0.3))

        with patch("graphrag.ingestion.extractor.get_llm", return_value=stub):
            _, relations = await extractor.extract(chunk)

        assert len(relations) == 1
        assert relations[0].confidence >= 0.0, "Confidence must not be negative"

    async def test_valid_confidence_preserved(self):
        from graphrag.ingestion.extractor import Extractor
        extractor = Extractor.__new__(Extractor)
        extractor._client = MagicMock()
        extractor._model_name = "m"
        extractor._gen_config = MagicMock()
        extractor._entity_types = ["PRODUCT"]

        chunk = self._make_chunk()
        stub = self._llm_stub(self._payload(confidence=0.85))

        with patch("graphrag.ingestion.extractor.get_llm", return_value=stub):
            _, relations = await extractor.extract(chunk)

        assert len(relations) == 1
        assert relations[0].confidence == pytest.approx(0.85)


class TestExtractorNoneResponse:
    """Extractor must handle None or empty Groq generate() output without crashing."""

    def _make_chunk(self):
        from graphrag.core.models import Chunk
        return Chunk(document_id="doc1", text="Some text.", chunk_index=0)

    @staticmethod
    def _llm_stub(raw: str):
        stub = MagicMock()
        stub.generate = AsyncMock(return_value=raw)
        return stub

    async def test_none_response_returns_empty(self):
        """Groq returning None (API error / blocked) → empty result, no crash."""
        from graphrag.ingestion.extractor import Extractor
        extractor = Extractor.__new__(Extractor)
        extractor._model_name = "m"
        extractor._entity_types = ["PRODUCT"]

        with patch("graphrag.ingestion.extractor.get_llm", return_value=self._llm_stub(None)):
            entities, relations = await extractor.extract(self._make_chunk())

        assert entities == []
        assert relations == []

    async def test_empty_string_response_returns_empty(self):
        """Groq returning empty string → empty result, no crash."""
        from graphrag.ingestion.extractor import Extractor
        extractor = Extractor.__new__(Extractor)
        extractor._model_name = "m"
        extractor._entity_types = ["PRODUCT"]

        with patch("graphrag.ingestion.extractor.get_llm", return_value=self._llm_stub("")):
            entities, relations = await extractor.extract(self._make_chunk())

        assert entities == []
        assert relations == []


# ── Embedder count mismatch ────────────────────────────────────────────────────

class TestEmbedderCountMismatch:
    """Embedder must raise ValueError when embedding count != chunk count."""

    def _make_chunks(self, n):
        from graphrag.core.models import Chunk
        return [Chunk(document_id="d", text=f"text {i}", chunk_index=i) for i in range(n)]

    def _mock_embedder(self):
        from graphrag.ingestion.embedder import Embedder
        e = Embedder.__new__(Embedder)
        e._client = MagicMock()
        e._model = "embed-model"
        e._batch_size = 100
        return e

    async def test_count_mismatch_raises_value_error(self):
        embedder = self._mock_embedder()
        chunks = self._make_chunks(3)

        # API returns only 2 embeddings for 3 chunks
        mock_result = MagicMock()
        mock_result.embeddings = [MagicMock(values=[0.1, 0.2]) for _ in range(2)]

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_result)
            with pytest.raises(ValueError, match="count mismatch"):
                await embedder.embed_chunks(chunks)

    async def test_correct_count_succeeds(self):
        embedder = self._mock_embedder()
        chunks = self._make_chunks(2)

        mock_result = MagicMock()
        mock_result.embeddings = [
            MagicMock(values=[0.1, 0.2]),
            MagicMock(values=[0.3, 0.4]),
        ]

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_result)
            result = await embedder.embed_chunks(chunks)

        assert result[0].embedding == [0.1, 0.2]
        assert result[1].embedding == [0.3, 0.4]


# ── Alias normalization key consistency ────────────────────────────────────────

class TestAliasNormalizationKey:
    """graph_writer must use registry._normalize so write and read keys match."""

    def test_graph_writer_uses_alias_normalize(self):
        """Verify the import exists and the function is the same object."""
        from graphrag.ingestion.graph_writer import _alias_normalize
        from graphrag.graph.alias_registry import _normalize
        assert _alias_normalize is _normalize

    def test_normalize_collapses_whitespace(self):
        """Ensures the key written by graph_writer matches what resolve() queries."""
        from graphrag.graph.alias_registry import _normalize
        # Double-space variant
        assert _normalize("Foo  Bar") == _normalize("Foo Bar")
        # The old inline code would NOT have collapsed these — it only had one sub()
        assert _normalize("Foo  Bar") == "foo bar"

    def test_normalize_strips_punctuation(self):
        from graphrag.graph.alias_registry import _normalize
        assert _normalize("SpaceX!") == "spacex"
        assert _normalize("Apple, Inc.") == "apple inc"


# ── Real percentile computation ────────────────────────────────────────────────

class TestPercentileComputation:
    """kpi_tracker._percentile must compute the correct p-th percentile."""

    def _p(self, values, p):
        from graphrag.business_matrix.kpi_tracker import _percentile
        return _percentile(values, p)

    def test_empty_list_returns_zero(self):
        assert self._p([], 0.95) == pytest.approx(0.0)

    def test_single_element(self):
        assert self._p([42.0], 0.5) == pytest.approx(42.0)
        assert self._p([42.0], 0.95) == pytest.approx(42.0)

    def test_median_of_odd_list(self):
        # [1, 2, 3, 4, 5] → p50 = 3.0
        assert self._p([1, 2, 3, 4, 5], 0.5) == pytest.approx(3.0)

    def test_p95_of_hundred_values(self):
        # [1..100], p95 should be around 95-96
        values = list(range(1, 101))
        result = self._p(values, 0.95)
        assert 94.0 <= result <= 96.0

    def test_p95_is_not_max(self):
        """Confirms p95 ≠ max (the old buggy behaviour)."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]  # 100 is an outlier
        p95 = self._p(values, 0.95)
        assert p95 < 100.0, "p95 should not be the max value"

    def test_p50_is_not_min(self):
        """Confirms p50 ≠ min (the other buggy label)."""
        values = [10.0, 20.0, 30.0]
        p50 = self._p(values, 0.5)
        assert p50 > 10.0, "p50 should not be the min value"

    def test_unsorted_input_still_correct(self):
        """_percentile must sort internally."""
        assert self._p([5, 1, 3, 2, 4], 0.5) == pytest.approx(3.0)


# ── Aware datetimes ────────────────────────────────────────────────────────────

class TestAwareDatetimes:
    """Model default factories must produce timezone-aware datetimes."""

    def test_document_ingested_at_is_aware(self):
        from graphrag.core.models import Document
        doc = Document(filename="f", source_path="p", raw_text="t")
        assert doc.ingested_at.tzinfo is not None, "ingested_at must be timezone-aware"

    def test_relation_extracted_at_is_aware(self):
        from graphrag.core.models import Relation
        rel = Relation(source_entity_id="a", target_entity_id="b", relation="USES")
        assert rel.extracted_at.tzinfo is not None

    def test_session_turn_timestamp_is_aware(self):
        from graphrag.core.models import SessionTurn
        turn = SessionTurn(question="q", answer="a")
        assert turn.timestamp.tzinfo is not None

    def test_kpi_event_recorded_at_is_aware(self):
        from graphrag.core.models import KPIEvent
        kpi = KPIEvent(query_id="q1", latency_ms=100.0)
        assert kpi.recorded_at.tzinfo is not None

    def test_eval_result_scored_at_is_aware(self):
        from graphrag.core.models import EvalResult
        r = EvalResult(job_id="j", query_id="q")
        assert r.scored_at.tzinfo is not None
