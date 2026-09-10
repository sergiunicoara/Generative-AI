"""Token estimation used for context-size accounting.

The estimator feeds a composition metric and an opt-in budget, never a hard
provider limit, so the properties that matter are that it is *consistent* and
that it cannot fail — not that it matches any particular tokenizer exactly.
"""

from __future__ import annotations

import graphrag.core.tokens as tokens_module
from graphrag.core.tokens import estimate_tokens


class TestEstimation:
    def test_empty_text_costs_nothing(self):
        assert estimate_tokens("") == 0

    def test_any_non_empty_text_costs_at_least_one_token(self):
        # A section that exists must never measure as free, or a metric can
        # report a context assembled from sections totalling zero.
        assert estimate_tokens("a") >= 1

    def test_longer_text_never_measures_smaller(self):
        short = estimate_tokens("alpha beta")
        long = estimate_tokens("alpha beta " * 100)
        assert long > short

    def test_estimation_is_stable_across_calls(self):
        text = "the same passage measured twice"
        assert estimate_tokens(text) == estimate_tokens(text)


class TestNeverRaises:
    """A counting failure must not break a prompt that would have answered."""

    def test_a_broken_encoder_falls_back_instead_of_propagating(self, monkeypatch):
        class _Exploding:
            def encode(self, text: str):
                raise RuntimeError("encoder is broken")

        monkeypatch.setattr(tokens_module, "_get_encoding", lambda: _Exploding())
        assert estimate_tokens("alpha beta gamma") >= 1

    def test_an_unavailable_tokenizer_still_produces_a_usable_number(self, monkeypatch):
        monkeypatch.setattr(tokens_module, "_get_encoding", lambda: None)
        assert estimate_tokens("alpha beta gamma delta") >= 1
