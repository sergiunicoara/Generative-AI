"""Tests for the media endpoints' base64 payload validation.

Covers `_decode_image_b64` in api/routes/kg/knowledge.py: malformed input must
be a client error rather than an unhandled 500, and the payload must be bounded
before it reaches PIL/EasyOCR.
"""

from __future__ import annotations

import base64

import pytest
from fastapi import HTTPException

from api.routes.kg.knowledge import _MAX_IMAGE_B64_CHARS, _decode_image_b64


class TestDecodeImageB64:
    def test_decodes_valid_payload(self) -> None:
        raw = b"\x89PNG\r\n\x1a\n some bytes"
        assert _decode_image_b64(base64.b64encode(raw).decode()) == raw

    def test_malformed_base64_is_400_not_500(self) -> None:
        with pytest.raises(HTTPException) as exc:
            _decode_image_b64("!!!not base64!!!")
        assert exc.value.status_code == 400
        assert "Invalid base64" in exc.value.detail

    def test_oversized_payload_is_413(self) -> None:
        with pytest.raises(HTTPException) as exc:
            _decode_image_b64("A" * (_MAX_IMAGE_B64_CHARS + 1))
        assert exc.value.status_code == 413

    def test_size_check_precedes_decode(self) -> None:
        """An oversized payload must be rejected without decoding it first."""
        with pytest.raises(HTTPException) as exc:
            _decode_image_b64("!" * (_MAX_IMAGE_B64_CHARS + 1))
        assert exc.value.status_code == 413  # not 400 from the malformed content

    def test_non_base64_characters_rejected_by_validation(self) -> None:
        # validate=True is what makes stray characters an error rather than
        # silently-discarded input producing surprising bytes.
        with pytest.raises(HTTPException) as exc:
            _decode_image_b64("YWJj*ZGVm")
        assert exc.value.status_code == 400
