"""Perceptual hashing for near-duplicate / similar-image detection.

Complements the embedding-based cross-modal search in
``graphrag/graph/multimodal.py`` (CLIP-style vectors via ANN) with a cheap,
deterministic signal that doesn't require a model call: a perceptual hash
(pHash) is stable under recompression, minor cropping, and small edits, and
two images' similarity is just the Hamming distance between their hashes.

Usage
-----
    from graphrag.graph.perceptual_hash import compute_phash, hamming_distance

    h1 = compute_phash(open("a.jpg", "rb").read())
    h2 = compute_phash(open("b.jpg", "rb").read())
    hamming_distance(h1, h2)   # 0 == identical, <= ~10 == likely near-duplicate

See ``MultiModalEntityService.set_perceptual_hash`` / ``find_similar_images``
in ``graphrag/graph/multimodal.py`` for the graph-backed wiring.
"""

from __future__ import annotations

from io import BytesIO

import imagehash
from PIL import Image


def compute_phash(image_bytes: bytes) -> str:
    """Compute a perceptual hash (pHash) for an image, returned as hex.

    Raises
    ------
    ValueError
        If ``image_bytes`` cannot be decoded as an image.
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()
    except Exception as exc:
        raise ValueError(f"Could not decode image bytes: {exc}") from exc
    return str(imagehash.phash(img))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Hamming distance between two hex-encoded perceptual hashes.

    0 means identical (or a hash collision); the pHash algorithm's default
    8x8 DCT hash yields a 64-bit hash, so distances range 0-64. Values below
    roughly 10 are typically near-duplicates; this is a heuristic, not a
    calibrated threshold -- callers pick ``max_distance`` for their use case.

    Raises
    ------
    ValueError
        If either hash string is not a valid hex perceptual hash.
    """
    try:
        a = imagehash.hex_to_hash(hash_a)
        b = imagehash.hex_to_hash(hash_b)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid perceptual hash: {exc}") from exc
    return a - b
