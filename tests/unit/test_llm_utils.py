"""Unit tests for graphrag.core.llm_utils — safe_response_text, normalize_dashes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from graphrag.core.llm_utils import normalize_dashes, safe_response_text


class TestSafeResponseText:
    """safe_response_text must handle all Gemini blocked/None response variants."""

    def test_normal_text_returned_stripped(self):
        resp = SimpleNamespace(text="  Hello world  ")
        assert safe_response_text(resp) == "Hello world"

    def test_none_text_returns_default(self):
        resp = SimpleNamespace(text=None)
        assert safe_response_text(resp) == ""

    def test_none_text_returns_custom_default(self):
        resp = SimpleNamespace(text=None)
        assert safe_response_text(resp, default="fallback") == "fallback"

    def test_empty_string_returns_default(self):
        resp = SimpleNamespace(text="")
        assert safe_response_text(resp) == ""

    def test_whitespace_only_returns_default(self):
        # strip() → "" which is falsy → returns default
        resp = SimpleNamespace(text="   ")
        assert safe_response_text(resp) == ""

    def test_none_response_object_returns_default(self):
        assert safe_response_text(None) == ""

    def test_object_without_text_attr_returns_default(self):
        resp = SimpleNamespace()   # no .text attribute
        assert safe_response_text(resp) == ""

    def test_object_with_text_raises_on_strip_returns_default(self):
        """If .text exists but strip() raises AttributeError, return default."""
        class WeirdResp:
            @property
            def text(self):
                return None
        assert safe_response_text(WeirdResp()) == ""

    def test_multiline_text_stripped(self):
        resp = SimpleNamespace(text="\n\nAnswer here\n")
        assert safe_response_text(resp) == "Answer here"

    def test_normal_response_custom_default_not_used(self):
        resp = SimpleNamespace(text="real answer")
        assert safe_response_text(resp, default="fallback") == "real answer"

    def test_blocked_response_triggers_low_confidence_signal(self):
        """Verify the default for hybrid_retriever triggers agentic fallback."""
        from graphrag.retrieval.agentic_retriever import _is_low_confidence

        resp = SimpleNamespace(text=None)
        answer = safe_response_text(
            resp, default="Insufficient context to answer this question."
        )
        # This must be detected as low-confidence → triggers agentic fallback
        assert _is_low_confidence(answer, citations=[])


class TestNormalizeDashes:
    """Regression test for the bug found diagnosing PRE-01/PRE-02/AUT-01/
    AUT-03/CON-01/AGT-02 golden-eval failures (2026-08-17, see
    docs/audit-2026-08-13.md): Groq's gpt-oss models write hyphenated
    document IDs/dates using U+2011 NON-BREAKING HYPHEN instead of ASCII
    "-", confirmed via a live UnicodeEncodeError on the raw response text,
    not a display artifact. Uses \\uXXXX escapes throughout (not literal
    characters) so a visually-identical-but-wrong dash can't be pasted into
    the test itself by mistake."""

    def test_non_breaking_hyphen_normalized(self):
        # The exact confirmed culprit: U+2011 in "FAA-AD-2022-03-07".
        nbh = "‑"
        raw = f"FAA{nbh}AD{nbh}2022{nbh}03{nbh}07"
        assert normalize_dashes(raw) == "FAA-AD-2022-03-07"

    def test_all_dash_variants_normalized(self):
        variants = "‐‑‒–—−"
        assert normalize_dashes(variants) == "-" * len(variants)

    def test_ascii_hyphen_passes_through_unchanged(self):
        assert normalize_dashes("FAA-AD-2022-03-07") == "FAA-AD-2022-03-07"

    def test_plain_text_without_dashes_unchanged(self):
        assert normalize_dashes("The aircraft is airworthy.") == "The aircraft is airworthy."

    def test_mixed_ascii_and_unicode_dashes(self):
        nbh = "‑"
        raw = f"AD 2022-03-07 supersedes AD{nbh}2020{nbh}05{nbh}11"
        assert normalize_dashes(raw) == "AD 2022-03-07 supersedes AD-2020-05-11"
