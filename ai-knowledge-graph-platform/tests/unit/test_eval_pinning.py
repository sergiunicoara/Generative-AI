"""Evaluation mode (EVAL_PINNED_MODELS): one generator, one router, one judge, no fallback."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import graphrag.core.llm_client as llm_client_module
from graphrag.core.exceptions import EvaluationError
from graphrag.core.llm_client import FallbackLLM, NoFallbackLLM, get_fast_llm, get_generation_route, get_llm


def _cfg(**overrides):
    base = dict(
        eval_pinned_models=True, eval_judge_provider="deepseek", llm_ingest_provider="groq",
        groq_api_key="groq-key", groq_model="openai/gpt-oss-120b", groq_fast_model="openai/gpt-oss-20b",
        deepseek_api_key="ds-key", openrouter_api_key="or-key", cerebras_model="c", google_api_key="",
        gemini_ingest_model="g",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _reset_singletons():
    llm_client_module._llm = None
    llm_client_module._fast_llm = None
    yield
    llm_client_module._llm = None
    llm_client_module._fast_llm = None


def test_pinned_generator_and_router_have_no_secondary_even_when_other_keys_exist():
    with patch("graphrag.core.config.get_settings", return_value=_cfg()):
        generator, router = get_llm(), get_fast_llm()
    for llm, model in ((generator, "openai/gpt-oss-120b"), (router, "openai/gpt-oss-20b")):
        assert isinstance(llm._secondary, NoFallbackLLM)
        assert llm._fallback_exceptions == ()
        assert llm._primary._default_model == model


def test_pinned_provider_error_surfaces_instead_of_switching_provider():
    with patch("graphrag.core.config.get_settings", return_value=_cfg()):
        llm = get_llm()
    llm._primary.generate = AsyncMock(side_effect=ConnectionError("groq down"))
    with pytest.raises(ConnectionError):
        asyncio.run(llm.generate("q"))


def test_pinned_without_a_groq_key_fails_loudly():
    with patch("graphrag.core.config.get_settings", return_value=_cfg(groq_api_key="")):
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            get_llm()


def test_pinned_conflicts_with_a_non_groq_provider_override():
    with patch("graphrag.core.config.get_settings", return_value=_cfg(llm_ingest_provider="deepseek")):
        with pytest.raises(RuntimeError, match="LLM_INGEST_PROVIDER"):
            get_llm()


def test_route_reports_no_fallback_when_pinned_and_the_chain_otherwise():
    with patch("graphrag.core.config.get_settings", return_value=_cfg()):
        route = get_generation_route()
    assert route == {"primary": "groq:openai/gpt-oss-120b", "fallback": "none (EVAL_PINNED_MODELS)",
                     "agentic_fast": "groq:openai/gpt-oss-20b"}
    with patch("graphrag.core.config.get_settings", return_value=_cfg(eval_pinned_models=False)):
        assert get_generation_route()["fallback"].startswith("deepseek:")


def test_not_pinned_keeps_the_production_fallback_chain():
    with patch("graphrag.core.config.get_settings", return_value=_cfg(eval_pinned_models=False)):
        llm = get_llm()
    assert isinstance(llm, FallbackLLM) and llm._fallback_exceptions != ()


def test_pinned_setting_defaults_to_off():
    from graphrag.core.config import Settings

    assert Settings.model_fields["eval_pinned_models"].default is False


# ── RAGAS judge ────────────────────────────────────────────────────────────────

def _judge(cfg):
    from graphrag.evaluation.ragas_evaluator import RagasEvaluator

    with patch("graphrag.evaluation.ragas_evaluator.get_settings", return_value=cfg):
        return RagasEvaluator()


def _eval_cfg(**overrides):
    cfg = _cfg(**overrides)
    cfg.evaluation = {"ragas_metrics": ["faithfulness"]}
    return cfg


def test_pinned_judge_is_exactly_the_configured_provider_at_temperature_zero():
    judge = _judge(_eval_cfg())
    assert judge.judge_info == {"provider": "deepseek", "model": "deepseek-v4-flash"}
    assert judge._llm.temperature == 0.0


def test_pinned_judge_never_falls_through_to_another_provider():
    with pytest.raises(EvaluationError, match="DEEPSEEK_API_KEY"):
        _judge(_eval_cfg(deepseek_api_key=""))
    with pytest.raises(EvaluationError, match="unknown EVAL_JUDGE_PROVIDER"):
        _judge(_eval_cfg(eval_judge_provider="bogus"))


def test_unpinned_judge_keeps_the_fallback_order():
    evaluator = _judge(_eval_cfg(eval_pinned_models=False))
    assert evaluator.judge_info["provider"] == "deepseek"  # DeepSeek first, as before


# ── DeepSeek generator (EVAL_GENERATOR_PROVIDER=deepseek) ───────────────────────

def test_pinned_deepseek_generator_and_router_are_one_model_with_no_secondary():
    with patch("graphrag.core.config.get_settings", return_value=_cfg(eval_generator_provider="deepseek")):
        generator, router = get_llm(), get_fast_llm()
        route = get_generation_route()
    for llm in (generator, router):
        assert isinstance(llm._secondary, NoFallbackLLM) and llm._fallback_exceptions == ()
        assert llm._primary._default_model == "deepseek-v4-flash"
    assert route == {"primary": "deepseek:deepseek-v4-flash", "fallback": "none (EVAL_PINNED_MODELS)",
                     "agentic_fast": "deepseek:deepseek-v4-flash"}


def test_pinned_deepseek_without_a_key_fails_loudly_and_ignores_a_stale_ingest_provider():
    with patch("graphrag.core.config.get_settings",
               return_value=_cfg(eval_generator_provider="deepseek", deepseek_api_key="")):
        with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
            get_llm()
    llm_client_module._llm = None
    with patch("graphrag.core.config.get_settings",
               return_value=_cfg(eval_generator_provider="deepseek", llm_ingest_provider="groq")):
        assert get_llm()._primary._default_model == "deepseek-v4-flash"


def test_generator_provider_defaults_to_groq():
    from graphrag.core.config import Settings

    assert Settings.model_fields["eval_generator_provider"].default == "groq"
