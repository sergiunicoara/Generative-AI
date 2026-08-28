import pytest

from limina_benchmark.cli import _run
from limina_benchmark.config import Settings
from limina_benchmark.evaluators import case_to_limina_payload, evaluate_with_limina, historical_judge_result
from limina_benchmark.schemas import EvaluationCase, TraceNode

pytestmark = pytest.mark.offline_eval


def _case() -> EvaluationCase:
    return EvaluationCase(
        case_id="case",
        category="healthy",
        source="historical",
        source_reference="fixture",
        trajectory=[TraceNode(node_id="u", kind="user", text="hello")],
        expected_failure=False,
    )


def test_limina_is_disabled_without_a_network_call():
    settings = Settings(None, False, "standard", True, None, None, 60)
    result = evaluate_with_limina(_case(), settings)
    assert result.status == "skipped"
    assert result.detected_failure is None


def test_historical_judge_requires_attached_result():
    result = historical_judge_result(_case())
    assert result.status == "skipped"


def test_cli_rejects_enabled_limina_without_api_key():
    class Args:
        repeats = 1
        limina = True
        dataset = None

    with pytest.raises(ValueError, match="LIMINA_API_KEY is missing"):
        _run(Args(), Settings(None, True, "standard", True, None, None, 60))


def test_settings_accepts_recruiter_native_internal_key_name(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_KEY", "test-recruiter-key")
    monkeypatch.delenv("RECRUITER_INTERNAL_API_KEY", raising=False)
    assert Settings.from_environment().recruiter_internal_api_key == "test-recruiter-key"


def test_adapter_preserves_order_and_tool_latency():
    case = _case().model_copy(
        update={
            "trajectory": [
                TraceNode(node_id="u", kind="user", text="hello"),
                TraceNode(node_id="t", kind="tool", name="lookup", text="{}", latency_ms=12.5),
                TraceNode(node_id="a", kind="agent", text="answer"),
            ]
        }
    )
    payload = case_to_limina_payload(case)
    assert payload["edges"] == [{"from": "u", "to": "t"}, {"from": "t", "to": "a"}]
    assert payload["nodes"][1]["execution_time_ms"] == 12.5
