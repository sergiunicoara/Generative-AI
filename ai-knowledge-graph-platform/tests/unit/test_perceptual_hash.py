"""Tests for graphrag.graph.perceptual_hash."""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from graphrag.graph.perceptual_hash import compute_phash, hamming_distance


def _image_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _solid_image(color: tuple[int, int, int], size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), color=color)


def _shape_image(size: int = 64) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 40, 40], fill=(0, 0, 0))
    draw.ellipse([20, 20, 55, 55], fill=(128, 128, 128))
    return img


class TestComputePhash:
    def test_returns_hex_string(self) -> None:
        h = compute_phash(_image_bytes(_solid_image((200, 50, 50))))
        assert isinstance(h, str)
        assert len(h) > 0
        int(h, 16)  # should parse as hex without raising

    def test_stable_for_same_image(self) -> None:
        img_bytes = _image_bytes(_shape_image())
        h1 = compute_phash(img_bytes)
        h2 = compute_phash(img_bytes)
        assert h1 == h2

    def test_invalid_bytes_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="decode"):
            compute_phash(b"not an image")


class TestHammingDistance:
    def test_identical_images_zero_distance(self) -> None:
        img_bytes = _image_bytes(_shape_image())
        h = compute_phash(img_bytes)
        assert hamming_distance(h, h) == 0

    def test_near_duplicate_small_distance(self) -> None:
        base = _shape_image()
        h1 = compute_phash(_image_bytes(base))

        # Slightly perturb: re-encode at lower quality / minor edit.
        perturbed = base.copy()
        draw = ImageDraw.Draw(perturbed)
        draw.point((5, 5), fill=(1, 1, 1))
        h2 = compute_phash(_image_bytes(perturbed, fmt="JPEG"))

        assert hamming_distance(h1, h2) <= 10

    def test_distinct_images_large_distance(self) -> None:
        h1 = compute_phash(_image_bytes(_solid_image((255, 0, 0))))
        h2 = compute_phash(_image_bytes(_shape_image()))
        assert hamming_distance(h1, h2) > 10

    def test_invalid_hash_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid perceptual hash"):
            hamming_distance("not-a-hash!!", "also-not-a-hash!!")
