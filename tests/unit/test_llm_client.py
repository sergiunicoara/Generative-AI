"""Unit tests for graphrag.core.llm_client — fail-fast retry behavior and
the FallbackLLM primary/secondary generalization.

Regression coverage for the 2026-07-24 incident: DeepSeek deprecated its
"deepseek-chat" model id, and get_llm()'s default path (bare DeepSeekLLM, no
fallback) broke answer synthesis entirely for ~40 minutes. These tests guard
against both halves of the fix: (1) a broken provider drops to 1 fail-fast
retry attempt instead of burning the full retry budget, (2) get_llm()'s
default path is a redundant FallbackLLM, not a single point of failure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIStatusError, APITimeoutError

from graphrag.core import provider_health as ph
from graphrag.core.llm_client import CerebrasLLM, DeepSeekLLM, FallbackLLM, GroqLLM, OpenRouterLLM, get_llm


@pytest.fixture(autouse=True)
def _reset_provider_health():
    ph.reset()
    yield
    ph.reset()


def _api_status_error() -> APIStatusError:
    """Build a real APIStatusError the way the openai SDK does — needed
    because its constructor requires a response/body, not just a message."""
    response = MagicMock()
    response.status_code = 400
    return APIStatusError("bad request", response=response, body=None)


class TestDeepSeekFailFast:
    async def test_healthy_provider_uses_full_retry_budget(self):
        llm = DeepSeekLLM(api_key="test-key")
        llm._client.chat.completions.create = MagicMock(side_effect=_api_status_error())

        with patch("graphrag.core.llm_client.asyncio.sleep", return_value=None):
            with pytest.raises(APIStatusError):
                await llm.generate("prompt")

        assert llm._client.chat.completions.create.call_count == llm._max_retries

    async def test_unhealthy_provider_drops_to_one_attempt(self):
        ph.record_result("deepseek", False)
        ph.record_result("deepseek", False)
        ph.record_result("deepseek", False)  # trips the breaker

        llm = DeepSeekLLM(api_key="test-key")
        llm._client.chat.completions.create = MagicMock(side_effect=_api_status_error())

        with patch("graphrag.core.llm_client.asyncio.sleep", return_value=None):
            with pytest.raises(APIStatusError):
                await llm.generate("prompt")

        assert llm._client.chat.completions.create.call_count == 1

    async def test_success_records_health(self):
        llm = DeepSeekLLM(api_key="test-key")
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="answer"))]
        llm._client.chat.completions.create = MagicMock(return_value=response)

        result = await llm.generate("prompt")

        assert result == "answer"
        assert ph.is_healthy("deepseek") is True


class TestCerebrasFailFast:
    """Mirrors TestDeepSeekFailFast — CerebrasLLM shares the same
    OpenAI-compatible request/retry shape as DeepSeekLLM."""

    async def test_healthy_provider_uses_full_retry_budget(self):
        llm = CerebrasLLM(api_key="test-key")
        llm._client.chat.completions.create = MagicMock(side_effect=_api_status_error())

        with patch("graphrag.core.llm_client.asyncio.sleep", return_value=None):
            with pytest.raises(APIStatusError):
                await llm.generate("prompt")

        assert llm._client.chat.completions.create.call_count == llm._max_retries

    async def test_unhealthy_provider_drops_to_one_attempt(self):
        ph.record_result("cerebras", False)
        ph.record_result("cerebras", False)
        ph.record_result("cerebras", False)  # trips the breaker

        llm = CerebrasLLM(api_key="test-key")
        llm._client.chat.completions.create = MagicMock(side_effect=_api_status_error())

        with patch("graphrag.core.llm_client.asyncio.sleep", return_value=None):
            with pytest.raises(APIStatusError):
                await llm.generate("prompt")

        assert llm._client.chat.completions.create.call_count == 1

    async def test_success_records_health(self):
        llm = CerebrasLLM(api_key="test-key")
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="answer"))]
        llm._client.chat.completions.create = MagicMock(return_value=response)

        result = await llm.generate("prompt")

        assert result == "answer"
        assert ph.is_healthy("cerebras") is True

    async def test_passes_max_tokens_when_given(self):
        llm = CerebrasLLM(api_key="test-key")
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="answer"))]
        llm._client.chat.completions.create = MagicMock(return_value=response)

        await llm.generate("prompt", max_tokens=300)

        _, kwargs = llm._client.chat.completions.create.call_args
        assert kwargs["max_tokens"] == 300


class TestGroqFailFast:
    async def test_unhealthy_provider_drops_to_one_attempt(self):
        ph.record_result("groq", False)
        ph.record_result("groq", False)
        ph.record_result("groq", False)

        llm = GroqLLM(api_key="test-key", default_model="test-model")
        timeout_exc = APITimeoutError(request=MagicMock())
        llm._client.chat.completions.create = MagicMock(side_effect=timeout_exc)

        with patch("graphrag.core.llm_client.asyncio.sleep", return_value=None):
            with pytest.raises(APITimeoutError):
                await llm.generate("prompt")

        assert llm._client.chat.completions.create.call_count == 1


class TestMaxTokens:
    """max_tokens (added for global_search.py's reduce step — A144): must be
    an opt-in per-call kwarg that never leaks onto calls that don't ask for
    it, and must survive FallbackLLM's primary->secondary failover."""

    async def test_deepseek_omits_max_tokens_key_when_not_given(self):
        llm = DeepSeekLLM(api_key="test-key")
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="answer"))]
        llm._client.chat.completions.create = MagicMock(return_value=response)

        await llm.generate("prompt")

        _, kwargs = llm._client.chat.completions.create.call_args
        assert "max_tokens" not in kwargs

    async def test_deepseek_passes_max_tokens_when_given(self):
        llm = DeepSeekLLM(api_key="test-key")
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="answer"))]
        llm._client.chat.completions.create = MagicMock(return_value=response)

        await llm.generate("prompt", max_tokens=300)

        _, kwargs = llm._client.chat.completions.create.call_args
        assert kwargs["max_tokens"] == 300

    async def test_groq_omits_max_tokens_key_when_not_given(self):
        llm = GroqLLM(api_key="test-key", default_model="test-model")
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="answer"))]
        llm._client.chat.completions.create = MagicMock(return_value=response)

        await llm.generate("prompt")

        _, kwargs = llm._client.chat.completions.create.call_args
        assert "max_tokens" not in kwargs

    async def test_groq_passes_max_tokens_when_given(self):
        llm = GroqLLM(api_key="test-key", default_model="test-model")
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="answer"))]
        llm._client.chat.completions.create = MagicMock(return_value=response)

        await llm.generate("prompt", max_tokens=300)

        _, kwargs = llm._client.chat.completions.create.call_args
        assert kwargs["max_tokens"] == 300

    async def test_fallback_forwards_max_tokens_to_secondary_on_failover(self):
        """The path most likely to be missed: max_tokens must survive
        primary->secondary failover, unlike `model` which is deliberately
        NOT forwarded (a model name valid on the primary isn't valid on a
        different provider)."""
        cfg = MagicMock(deepseek_api_key="ds-key", groq_api_key="groq-key", groq_model="groq-model")
        fb = FallbackLLM.deepseek_primary(cfg)

        fb._primary.generate = AsyncMock(side_effect=_api_status_error())
        fb._secondary.generate = AsyncMock(return_value="answer from groq")

        await fb.generate("prompt", max_tokens=300)

        _, kwargs = fb._secondary.generate.call_args
        assert kwargs.get("max_tokens") == 300


class TestFallbackLLMClassmethods:
    def test_deepseek_primary_uses_deepseek_first(self):
        cfg = MagicMock(deepseek_api_key="ds-key", groq_api_key="groq-key", groq_model="groq-model")
        fb = FallbackLLM.deepseek_primary(cfg)
        assert fb._primary_name == "deepseek"
        assert isinstance(fb._primary, DeepSeekLLM)
        assert isinstance(fb._secondary, GroqLLM)

    def test_groq_primary_uses_groq_first(self):
        # openrouter_api_key="" (unset) so the secondary stays a bare
        # DeepSeekLLM — see test_groq_primary_adds_openrouter_hop_when_keyed
        # below for the keyed case.
        cfg = MagicMock(
            deepseek_api_key="ds-key", groq_api_key="groq-key", groq_model="groq-model",
            openrouter_api_key="",
        )
        fb = FallbackLLM.groq_primary(cfg)
        assert fb._primary_name == "groq"
        assert isinstance(fb._primary, GroqLLM)
        assert isinstance(fb._secondary, DeepSeekLLM)

    def test_groq_primary_adds_openrouter_hop_when_keyed(self):
        cfg = MagicMock(
            deepseek_api_key="ds-key", groq_api_key="groq-key", groq_model="groq-model",
            openrouter_api_key="or-key", openrouter_model="nvidia/nemotron-3-super-120b-a12b:free",
        )
        fb = FallbackLLM.groq_primary(cfg)
        assert fb._primary_name == "groq"
        assert isinstance(fb._secondary, FallbackLLM)
        assert fb._secondary._primary_name == "deepseek"
        assert isinstance(fb._secondary._secondary, OpenRouterLLM)

    def test_groq_primary_model_override_for_fast_llm(self):
        cfg = MagicMock(deepseek_api_key="ds-key", groq_api_key="groq-key", groq_model="big-model")
        fb = FallbackLLM.groq_primary(cfg, model="fast-model")
        assert fb._primary._default_model == "fast-model"

    async def test_deepseek_primary_falls_over_to_groq_on_api_status_error(self):
        """This is the exact incident scenario: DeepSeek returns a 400
        (bad/deprecated model id) — the answer must still come back, via Groq."""
        cfg = MagicMock(deepseek_api_key="ds-key", groq_api_key="groq-key", groq_model="groq-model")
        fb = FallbackLLM.deepseek_primary(cfg)

        fb._primary.generate = AsyncMock(side_effect=_api_status_error())
        fb._secondary.generate = AsyncMock(return_value="answer from groq")

        result = await fb.generate("prompt")

        assert result == "answer from groq"

    def test_cerebras_primary_uses_cerebras_first_with_deepseek_groq_chain(self):
        """cerebras_primary() must be Cerebras -> (DeepSeek -> Groq), built by
        wrapping deepseek_primary()'s own FallbackLLM as the secondary — not
        a flat Cerebras -> Groq pair that silently drops DeepSeek from the
        chain."""
        cfg = MagicMock(
            cerebras_api_key="cb-key", cerebras_model="cb-model",
            deepseek_api_key="ds-key", groq_api_key="groq-key", groq_model="groq-model",
        )
        fb = FallbackLLM.cerebras_primary(cfg)

        assert fb._primary_name == "cerebras"
        assert isinstance(fb._primary, CerebrasLLM)
        assert isinstance(fb._secondary, FallbackLLM)
        assert fb._secondary._primary_name == "deepseek"
        assert isinstance(fb._secondary._primary, DeepSeekLLM)
        assert isinstance(fb._secondary._secondary, GroqLLM)

    async def test_cerebras_primary_falls_all_the_way_to_groq(self):
        """Both Cerebras and DeepSeek down -> Groq still answers."""
        cfg = MagicMock(
            cerebras_api_key="cb-key", cerebras_model="cb-model",
            deepseek_api_key="ds-key", groq_api_key="groq-key", groq_model="groq-model",
        )
        fb = FallbackLLM.cerebras_primary(cfg)

        fb._primary.generate = AsyncMock(side_effect=_api_status_error())
        fb._secondary._primary.generate = AsyncMock(side_effect=_api_status_error())
        fb._secondary._secondary.generate = AsyncMock(return_value="answer from groq")

        result = await fb.generate("prompt")

        assert result == "answer from groq"


class TestGetLlmDefaultHasFallback:
    """Regression test for the actual incident: get_llm()'s default path
    must be a redundant FallbackLLM, not a bare single-provider client with
    no failover. Default primary changed 2026-08-17: DeepSeek -> Cerebras
    (free tier) -> Groq (free tier, current) — Cerebras was found unfunded
    on this key the same day, see llm_client.py module docstring."""

    def test_default_provider_is_groq_with_deepseek_fallback(self):
        import graphrag.core.llm_client as llm_client_module
        llm_client_module._llm = None  # clear the singleton so get_llm() rebuilds

        settings = MagicMock(
            llm_ingest_provider="",
            cerebras_api_key="cb-key",
            cerebras_model="cb-model",
            deepseek_api_key="ds-key",
            groq_api_key="groq-key",
            groq_model="groq-model",
            openrouter_api_key="",  # unset — see test_groq_primary_adds_openrouter_hop_when_keyed
        )
        with patch("graphrag.core.config.get_settings", return_value=settings):
            llm = get_llm()

        # Default as of 2026-08-17: Groq primary (free tier), DeepSeek
        # fallback — Cerebras is skipped by default because the account on
        # this key is unfunded (see llm_ingest_provider's docstring in
        # graphrag/core/config.py). groq_primary() is flat (no nested
        # FallbackLLM), unlike the old cerebras_primary() chain.
        assert isinstance(llm, FallbackLLM)
        assert llm._primary_name == "groq"
        assert not isinstance(llm._secondary, FallbackLLM)

        llm_client_module._llm = None  # don't leak the mocked singleton to other tests

    def test_deepseek_override_skips_cerebras(self):
        import graphrag.core.llm_client as llm_client_module
        llm_client_module._llm = None

        settings = MagicMock(
            llm_ingest_provider="deepseek",
            deepseek_api_key="ds-key",
            groq_api_key="groq-key",
            groq_model="groq-model",
        )
        with patch("graphrag.core.config.get_settings", return_value=settings):
            llm = get_llm()

        assert isinstance(llm, FallbackLLM)
        assert llm._primary_name == "deepseek"
        assert isinstance(llm._secondary, GroqLLM)  # flat chain, no Cerebras hop

        llm_client_module._llm = None

    def test_cerebras_override_restores_old_default_chain(self):
        import graphrag.core.llm_client as llm_client_module
        llm_client_module._llm = None

        settings = MagicMock(
            llm_ingest_provider="cerebras",
            cerebras_api_key="cb-key",
            cerebras_model="cb-model",
            deepseek_api_key="ds-key",
            groq_api_key="groq-key",
            groq_model="groq-model",
        )
        with patch("graphrag.core.config.get_settings", return_value=settings):
            llm = get_llm()

        assert isinstance(llm, FallbackLLM)
        assert llm._primary_name == "cerebras"
        assert isinstance(llm._secondary, FallbackLLM)
        assert llm._secondary._primary_name == "deepseek"

        llm_client_module._llm = None
