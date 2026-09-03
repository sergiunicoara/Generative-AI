"""OCR extraction for ingested images.

Wires the ``transform_type: ocr`` hook that ``graphrag/graph/multimodal.py``
declared but left unimplemented in Phase 1 -- see
``MultiModalEntityService.run_ocr``.

Model loading note
-------------------
``easyocr.Reader`` downloads its detection/recognition models to
``~/.EasyOCR`` on first use. The Dockerfile's non-root ``app`` user has no
real home directory (same issue ``docker-compose.yml`` documents for
``HF_HOME``/``TRANSFORMERS_CACHE`` on ``query_worker``), so any worker that
calls into this module needs ``EASYOCR_MODULE_STORAGE_DIRECTORY`` (or
``MODULE_PATH`` passed to ``easyocr.Reader(...)``) pointed at a writable
path, e.g. ``/tmp/easyocr``.
"""

from __future__ import annotations

from io import BytesIO
from threading import Lock

import numpy as np
import structlog
from PIL import Image

log = structlog.get_logger(__name__)

_reader = None
_reader_lock = Lock()


def _get_reader():
    """Lazily construct a singleton ``easyocr.Reader`` (model load is slow)."""
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr
                _reader = easyocr.Reader(["en"], gpu=False)
                log.info("ocr.reader_initialised")
    return _reader


def extract_text(image_bytes: bytes) -> tuple[str, float]:
    """Run OCR over an image and return (joined_text, mean_confidence).

    Text fragments are joined in the order EasyOCR returns them (top-to-
    bottom, left-to-right for a typical document), separated by single
    spaces. Returns ``("", 0.0)`` if no text is detected.

    Raises
    ------
    ValueError
        If ``image_bytes`` cannot be decoded as an image.
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img.load()
    except Exception as exc:
        raise ValueError(f"Could not decode image bytes: {exc}") from exc

    reader = _get_reader()
    results = reader.readtext(np.array(img))  # [(bbox, text, confidence), ...]

    if not results:
        return "", 0.0

    texts = [str(text).strip() for _, text, _ in results if str(text).strip()]
    confidences = [float(conf) for _, _, conf in results]

    joined = " ".join(texts)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return joined, mean_conf
