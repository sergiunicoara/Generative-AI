"""Tests for graphrag.graph.ocr.

easyocr.Reader downloads real detection/recognition models on first use, so
unit tests mock it out entirely (consistent with how this project reserves
live-service/model dependencies for its testcontainers-backed e2e suite).
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import graphrag.graph.ocr as ocr_module
from graphrag.graph.ocr import extract_text


def _image_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (32, 32), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _reset_reader_singleton():
    """Each test gets a clean lazy-singleton slate."""
    ocr_module._reader = None
    yield
    ocr_module._reader = None


class TestExtractText:
    def test_joins_fragments_and_averages_confidence(self) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "Hello", 0.9),
            ([[0, 20], [10, 20], [10, 30], [0, 30]], "World", 0.7),
        ]
        with patch("easyocr.Reader", return_value=mock_reader) as mock_ctor:
            text, confidence = extract_text(_image_bytes())

        mock_ctor.assert_called_once()
        assert text == "Hello World"
        assert confidence == pytest.approx(0.8)

    def test_no_detections_returns_empty(self) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        with patch("easyocr.Reader", return_value=mock_reader):
            text, confidence = extract_text(_image_bytes())

        assert text == ""
        assert confidence == 0.0

    def test_blank_fragments_are_skipped(self) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "  ", 0.5),
            ([[0, 20], [10, 20], [10, 30], [0, 30]], "Real text", 0.95),
        ]
        with patch("easyocr.Reader", return_value=mock_reader):
            text, _ = extract_text(_image_bytes())

        assert text == "Real text"

    def test_reader_is_a_lazy_singleton(self) -> None:
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        with patch("easyocr.Reader", return_value=mock_reader) as mock_ctor:
            extract_text(_image_bytes())
            extract_text(_image_bytes())

        mock_ctor.assert_called_once()

    def test_invalid_bytes_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="decode"):
            extract_text(b"not an image")
