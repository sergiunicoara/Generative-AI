"""Unit tests for graphrag.core.llm_utils — safe_response_text."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from graphrag.core.llm_utils import safe_response_text


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
