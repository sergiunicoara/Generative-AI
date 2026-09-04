"""Tests for the graph-backed multi-modal wiring: OCR and perceptual hashing.

These cover ``MultiModalEntityService``'s Neo4j-facing methods rather than the
pure helpers in test_ocr.py / test_perceptual_hash.py. That split matters: the
first version of this feature had two runtime-breaking bugs (a dict written to
a Neo4j property, and every transformation MERGEing onto one SourceArtifact
node) that unit tests of the pure functions structurally could not catch.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.graph.multimodal import MediaTransformation, MultiModalEntityService


def _svc(run_return=None):
    neo4j = MagicMock()
    neo4j.run = AsyncMock(return_value=run_return if run_return is not None else [])
    return MultiModalEntityService(neo4j), neo4j


def _calls(neo4j):
    """All (cypher, kwargs) pairs passed to neo4j.run, in order."""
    return [(c.args[0], c.kwargs) for c in neo4j.run.await_args_list]


# ── record_transformation ─────────────────────────────────────────────────────

class TestRecordTransformation:
    async def test_metadata_is_json_encoded_not_a_map(self) -> None:
        """Neo4j rejects map property values, so metadata must arrive as a string."""
        svc, neo4j = _svc()
        await svc.record_transformation(MediaTransformation(
            tenant="acme",
            input_attachment_id="media-1",
            output_artifact_id="artifact-1",
            transform_type="ocr",
            metadata={"confidence": 0.9, "chars": 12},
        ))
        _, kwargs = _calls(neo4j)[0]
        assert isinstance(kwargs["metadata"], str)
        assert json.loads(kwargs["metadata"]) == {"confidence": 0.9, "chars": 12}

    async def test_empty_metadata_is_still_a_string(self) -> None:
        svc, neo4j = _svc()
        await svc.record_transformation(MediaTransformation(
            tenant="acme",
            input_attachment_id="media-1",
            output_artifact_id="artifact-1",
            transform_type="perceptual_hash",
        ))
        _, kwargs = _calls(neo4j)[0]
        assert kwargs["metadata"] == "{}"


# ── Perceptual hashing ────────────────────────────────────────────────────────

class TestSetPerceptualHash:
    async def test_stores_hash_and_records_provenance(self) -> None:
        svc, neo4j = _svc()
        await svc.set_perceptual_hash("acme", "media-1", "abcd1234")

        set_call, prov_call = _calls(neo4j)
        assert set_call[1]["phash"] == "abcd1234"
        assert set_call[1]["tenant"] == "acme"
        assert prov_call[1]["transform_type"] == "perceptual_hash"
        assert prov_call[1]["output_digest"] == "abcd1234"

    async def test_output_artifact_id_is_distinct_from_attachment(self) -> None:
        """Reusing the attachment id collapses every transform onto one node."""
        svc, neo4j = _svc()
        await svc.set_perceptual_hash("acme", "media-1", "abcd1234")

        _, prov_kwargs = _calls(neo4j)[1]
        assert prov_kwargs["input_id"] == "media-1"
        assert prov_kwargs["output_id"] != "media-1"

    async def test_requires_tenant(self) -> None:
        svc, _ = _svc()
        with pytest.raises(ValueError):
            await svc.set_perceptual_hash("", "media-1", "abcd1234")


class TestFindSimilarImages:
    def _rows(self):
        return [
            {"id": "b", "entity_name": "B", "entity_type": "PRODUCT",
             "media_url": "", "caption": "", "phash": "f" * 16},
            {"id": "c", "entity_name": "C", "entity_type": "PRODUCT",
             "media_url": "", "caption": "", "phash": "0" * 16},
        ]

    async def test_returns_empty_when_target_has_no_phash(self) -> None:
        svc, neo4j = _svc(run_return=[])
        assert await svc.find_similar_images("acme", "missing") == []

    async def test_filters_by_distance_and_excludes_target(self) -> None:
        neo4j = MagicMock()
        neo4j.run = AsyncMock(side_effect=[
            [{"phash": "f" * 16}],   # target lookup
            self._rows(),            # scan
        ])
        svc = MultiModalEntityService(neo4j)

        out = await svc.find_similar_images("acme", "a", max_distance=0)

        assert [m["id"] for m in out] == ["b"]      # identical hash only
        assert "phash" not in out[0]
        assert out[0]["distance"] == 0

    async def test_sorts_closest_first_and_applies_limit(self) -> None:
        neo4j = MagicMock()
        neo4j.run = AsyncMock(side_effect=[
            [{"phash": "f" * 16}],
            self._rows(),
        ])
        svc = MultiModalEntityService(neo4j)

        out = await svc.find_similar_images("acme", "a", max_distance=64, limit=1)

        assert len(out) == 1
        assert out[0]["id"] == "b"                  # distance 0 beats distance 64

    async def test_scan_is_bounded_and_excludes_target_in_cypher(self) -> None:
        neo4j = MagicMock()
        neo4j.run = AsyncMock(side_effect=[[{"phash": "f" * 16}], []])
        svc = MultiModalEntityService(neo4j)

        await svc.find_similar_images("acme", "a", scan_limit=500)

        scan_cypher, scan_kwargs = _calls(neo4j)[1]
        assert "LIMIT $scan_limit" in scan_cypher
        assert "m.id <> $id" in scan_cypher
        assert scan_kwargs["scan_limit"] == 500


# ── OCR ───────────────────────────────────────────────────────────────────────

class TestRunOCR:
    async def test_digest_is_a_hash_not_the_text(self) -> None:
        svc, neo4j = _svc()
        with patch("graphrag.graph.ocr.extract_text", return_value=("hello world", 0.9)):
            result, text = await svc.run_ocr("acme", "media-1", b"bytes")

        assert result.output_digest.startswith("sha256:")
        assert "hello world" not in result.output_digest
        # The caller still gets the real text back, not the digest.
        assert text == "hello world"

    async def test_metadata_carries_confidence_and_bounded_excerpt(self) -> None:
        svc, neo4j = _svc()
        long_text = "x" * 2000
        with patch("graphrag.graph.ocr.extract_text", return_value=(long_text, 0.5)):
            result, _ = await svc.run_ocr("acme", "media-1", b"bytes")

        assert result.metadata["confidence"] == 0.5
        assert result.metadata["chars"] == 2000
        assert len(result.metadata["text_excerpt"]) == 500

    async def test_output_artifact_id_is_distinct_from_attachment(self) -> None:
        svc, neo4j = _svc()
        with patch("graphrag.graph.ocr.extract_text", return_value=("t", 0.9)):
            result, _ = await svc.run_ocr("acme", "media-1", b"bytes")

        assert result.output_artifact_id != "media-1"

    async def test_ocr_and_phash_do_not_share_an_artifact_node(self) -> None:
        """Regression: both transforms MERGEd onto one SourceArtifact and the
        second silently overwrote the first's provenance."""
        svc, neo4j = _svc()
        await svc.set_perceptual_hash("acme", "media-1", "abcd1234")
        with patch("graphrag.graph.ocr.extract_text", return_value=("t", 0.9)):
            await svc.run_ocr("acme", "media-1", b"bytes")

        output_ids = [
            kwargs["output_id"] for _, kwargs in _calls(neo4j) if "output_id" in kwargs
        ]
        assert len(output_ids) == 2
        assert output_ids[0] != output_ids[1]

    async def test_caption_backfill_is_guarded_to_empty_captions(self) -> None:
        svc, neo4j = _svc()
        with patch("graphrag.graph.ocr.extract_text", return_value=("found text", 0.9)):
            await svc.run_ocr("acme", "media-1", b"bytes")

        caption_cypher, caption_kwargs = _calls(neo4j)[-1]
        assert "SET m.caption" in caption_cypher
        # The guard is what stops OCR clobbering a human-written caption.
        assert "m.caption IS NULL OR m.caption = ''" in caption_cypher
        assert caption_kwargs["text"] == "found text"

    async def test_requires_tenant(self) -> None:
        svc, _ = _svc()
        with pytest.raises(ValueError):
            await svc.run_ocr("", "media-1", b"bytes")

    async def test_runs_extraction_off_the_event_loop(self) -> None:
        """Blocking CPU work must not run inline in the async endpoint."""
        svc, _ = _svc()
        with patch("asyncio.to_thread", new=AsyncMock(return_value=("t", 0.1))) as to_thread:
            await svc.run_ocr("acme", "media-1", b"bytes")

        to_thread.assert_awaited_once()


# ── attach_image ──────────────────────────────────────────────────────────────

class TestAttachImage:
    async def test_computes_phash_when_bytes_supplied(self) -> None:
        svc, neo4j = _svc()
        with patch("graphrag.graph.perceptual_hash.compute_phash", return_value="deadbeef"):
            await svc.attach_image("E", "PRODUCT", tenant="acme", image_bytes=b"img")

        phash_calls = [kw for _, kw in _calls(neo4j) if "phash" in kw]
        assert phash_calls and phash_calls[0]["phash"] == "deadbeef"

    async def test_skips_phash_when_no_bytes(self) -> None:
        svc, neo4j = _svc()
        await svc.attach_image("E", "PRODUCT", tenant="acme")

        assert not [kw for _, kw in _calls(neo4j) if "phash" in kw]
