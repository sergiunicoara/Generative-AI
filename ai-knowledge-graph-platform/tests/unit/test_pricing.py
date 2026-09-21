"""Real per-call cost computation (IMPLEMENTATION_AUDIT.md).

Before this fix, `cost_usd` was hardcoded to 0.0 at every call site that
emitted a `CostEvent`, even for genuinely paid Groq/DeepSeek/Cerebras calls.
`pricing.py` turns real (provider, model, tokens) into a real dollar figure,
and `genai_telemetry._finish()` emits it for every LLM call.
"""

from __future__ import annotations

from graphrag.observability.pricing import estimated_cost_usd


class TestEstimatedCostUsd:
    def test_known_model_computes_a_real_nonzero_cost(self):
        cost = estimated_cost_usd("groq", "openai/gpt-oss-120b", 1_000_000, 1_000_000)
        assert cost == 0.15 + 0.60  # input_per_1m + output_per_1m at 1M tokens each

    def test_scales_linearly_with_token_count(self):
        cost = estimated_cost_usd("groq", "openai/gpt-oss-120b", 500_000, 0)
        assert abs(cost - 0.075) < 1e-9

    def test_unknown_model_returns_none_not_zero(self):
        # A fabricated 0.0 would be indistinguishable from a genuinely free
        # call and corrupt cost dashboards — must be None (unknown).
        assert estimated_cost_usd("groq", "some-brand-new-model", 100, 100) is None

    def test_unknown_provider_returns_none(self):
        assert estimated_cost_usd("some-new-provider", "openai/gpt-oss-120b", 100, 100) is None

    def test_missing_token_counts_return_none(self):
        assert estimated_cost_usd("groq", "openai/gpt-oss-120b", None, 100) is None
        assert estimated_cost_usd("groq", "openai/gpt-oss-120b", 100, None) is None

    def test_openrouter_free_tier_is_a_real_confirmed_zero(self):
        # Distinct from "unknown" -- OpenRouter's :free models genuinely
        # cost nothing, so this is a real 0.0, not a missing-price None.
        cost = estimated_cost_usd("openrouter", "nvidia/nemotron-3-super-120b-a12b:free", 1000, 1000)
        assert cost == 0.0

    def test_legacy_deepseek_model_name_still_priced(self):
        # DeepSeek retired "deepseek-v4-flash" and now serves it as
        # "deepseek-flash" at the same price — both keys must resolve.
        cost_new = estimated_cost_usd("deepseek", "deepseek-flash", 1_000_000, 0)
        cost_legacy = estimated_cost_usd("deepseek", "deepseek-v4-flash", 1_000_000, 0)
        assert cost_new == cost_legacy == 0.15


class TestFinishEmitsRealCostEvent:
    async def test_llm_call_span_emits_a_nonzero_cost_event_for_a_known_model(self, monkeypatch):
        from graphrag.observability import genai_telemetry
        from graphrag.core.llm_client import GroqLLM
        from unittest.mock import MagicMock
        from types import SimpleNamespace

        recorded = []
        import graphrag.observability.cost_attribution as cost_attribution
        monkeypatch.setattr(cost_attribution, "record_cost_event", lambda event: recorded.append(event))

        llm = GroqLLM(api_key="test-key", default_model="llama-3.3-70b-versatile")
        llm._client.chat.completions.create = MagicMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
                model="openai/gpt-oss-120b",
                usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000),
            ),
        )

        with genai_telemetry.llm_call_span(provider="groq", model="requested"):
            await llm.generate("prompt")

        assert len(recorded) == 1
        assert recorded[0].cost_usd == 0.15 + 0.60
        assert recorded[0].provider == "groq"
        assert recorded[0].model == "openai/gpt-oss-120b"

    async def test_no_cost_event_emitted_for_an_unpriced_model(self, monkeypatch):
        from graphrag.observability import genai_telemetry
        from graphrag.core.llm_client import GroqLLM
        from unittest.mock import MagicMock
        from types import SimpleNamespace

        recorded = []
        import graphrag.observability.cost_attribution as cost_attribution
        monkeypatch.setattr(cost_attribution, "record_cost_event", lambda event: recorded.append(event))

        llm = GroqLLM(api_key="test-key", default_model="llama-3.3-70b-versatile")
        llm._client.chat.completions.create = MagicMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
                model="some-brand-new-unpriced-model",
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=100),
            ),
        )

        with genai_telemetry.llm_call_span(provider="groq", model="requested"):
            await llm.generate("prompt")

        assert recorded == []
