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
from graphrag.core.llm_client import DeepSeekLLM, FallbackLLM, GroqLLM, get_llm


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


class TestFallbackLLMClassmethods:
    def test_deepseek_primary_uses_deepseek_first(self):
        cfg = MagicMock(deepseek_api_key="ds-key", groq_api_key="groq-key", groq_model="groq-model")
        fb = FallbackLLM.deepseek_primary(cfg)
        assert fb._primary_name == "deepseek"
        assert isinstance(fb._primary, DeepSeekLLM)
        assert isinstance(fb._secondary, GroqLLM)

    def test_groq_primary_uses_groq_first(self):
        cfg = MagicMock(deepseek_api_key="ds-key", groq_api_key="groq-key", groq_model="groq-model")
        fb = FallbackLLM.groq_primary(cfg)
        assert fb._primary_name == "groq"
        assert isinstance(fb._primary, GroqLLM)
        assert isinstance(fb._secondary, DeepSeekLLM)

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


class TestGetLlmDefaultHasFallback:
    """Regression test for the actual incident: get_llm()'s default path
    must be a redundant FallbackLLM, not a bare single-provider client with
    no failover."""

    def test_default_provider_is_fallback_llm_not_bare_deepseek(self):
        import graphrag.core.llm_client as llm_client_module
        llm_client_module._llm = None  # clear the singleton so get_llm() rebuilds

        settings = MagicMock(
            llm_ingest_provider="",
            deepseek_api_key="ds-key",
            groq_api_key="groq-key",
            groq_model="groq-model",
        )
        with patch("graphrag.core.config.get_settings", return_value=settings):
            llm = get_llm()

        assert isinstance(llm, FallbackLLM)
        assert llm._primary_name == "deepseek"

        llm_client_module._llm = None  # don't leak the mocked singleton to other tests
