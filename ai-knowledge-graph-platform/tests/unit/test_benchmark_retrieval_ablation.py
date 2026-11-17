"""Result schema of scripts/benchmark_retrieval_ablation.py (offline: no model, no database)."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from graphrag.core.models import QueryResult, RetrievalStep, RetrievalTrajectory  # noqa: E402
from graphrag.evaluation import run_metadata  # noqa: E402
from scripts import benchmark_retrieval_ablation as ablation  # noqa: E402

@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch):
    """main() sets EVAL_PINNED_MODELS and rebuilds the settings cache: undo both afterwards."""
    from graphrag.core.config import get_settings

    monkeypatch.setenv("EVAL_PINNED_MODELS", "1")
    yield
    monkeypatch.undo()
    get_settings.cache_clear()


QUESTION = {"id": "Q1", "type": "single_hop", "question": "Which AD supersedes AD-2022?",
            "expected_citations": ["FAA-AD-2024"], "required_answer_terms": ["supersede"]}


def _result(*, fallback=None, completed_by="synthesis", retrieval_ms=120.0, answer="AD-2024 supersedes it.",
            contexts=("ctx",)):
    trajectory = RetrievalTrajectory(
        planned_mode="agentic" if completed_by != "synthesis" else "local", completed_by=completed_by,
        steps=[RetrievalStep(step=1, action="search", query="q", latency_ms=retrieval_ms)])
    return QueryResult(
        question="q", answer=answer, contexts=list(contexts), citations=["FAA-AD-2024-01-02"],
        retrieval_mode="local", routing_reason="r", retrieval_trajectory=trajectory,
        explanation={"fallback": fallback or {"triggered": False, "reason": None}})


def test_row_records_golden_latency_and_fallback_signals():
    row = ablation.build_row(
        QUESTION, _result(fallback={"triggered": True, "reason": "low_confidence"}, completed_by="final_synthesis"),
        1500.0, {"status": "scored", "faithfulness": 0.8, "attempts": 1, "notes": []})
    assert row["golden_pass"] is True and row["golden_failures"] == []
    assert row["status"] == "scored" and row["faithfulness"] == 0.8
    assert row["latency_e2e_ms"] == 1500.0 and row["latency_retrieval_ms"] == 120.0
    assert row["agentic_fallback"] == {"fired": True, "reason": "low_confidence", "completed_by": "final_synthesis"}
    assert row["citation_recall"] == 1.0 and row["n_contexts"] == 1


def test_row_without_judge_is_marked_not_judged_and_failures_are_listed():
    row = ablation.build_row(QUESTION, _result(answer="no idea"), 10.0, None)
    assert row["status"] == "not_judged" and row["faithfulness"] is None
    assert row["golden_pass"] is False and row["golden_failures"]


def test_agentic_fallback_falls_back_to_the_trajectory_without_an_explanation():
    result = _result(completed_by="reasoning_answer")
    result.explanation = None
    assert run_metadata.agentic_fallback(result)["fired"] is True
    assert run_metadata.agentic_fallback(_result())["fired"] is False


def test_retrieval_latency_is_none_when_the_trajectory_was_not_captured():
    result = _result()
    result.retrieval_trajectory = None
    assert run_metadata.retrieval_latency_ms(result) is None


def test_summary_counts_every_status_and_ignores_errors_in_rates():
    rows = [
        ablation.build_row(QUESTION, _result(), 1000.0, {"status": "scored", "faithfulness": 1.0, "attempts": 1, "notes": []}),
        ablation.build_row(QUESTION, _result(fallback={"triggered": True, "reason": "x"}), 3000.0,
                           {"status": "scored", "faithfulness": 0.5, "attempts": 1, "notes": []}),
        ablation.build_row(QUESTION, _result(), 500.0, {"status": "refused", "faithfulness": None, "attempts": 0, "notes": []}),
        ablation.build_row(QUESTION, _result(), 500.0, {"status": "unscorable", "faithfulness": None, "attempts": 3, "notes": ["n"]}),
        ablation.error_row(QUESTION, RuntimeError("boom"), 5.0),
    ]
    summary = ablation.summarise(rows)
    assert (summary["n_questions"], summary["n_errors"], summary["n_scored"]) == (5, 1, 2)
    assert (summary["n_refused"], summary["n_unscorable"], summary["n_agentic_fallback"]) == (1, 1, 1)
    assert summary["faithfulness_mean"] == 0.75 and summary["golden_pass_rate"] == 1.0
    assert summary["latency_e2e_ms"]["n"] == 4 and summary["latency_e2e_ms"]["max"] == 3000.0
    assert rows[-1]["error"] == "RuntimeError: boom"


def test_judge_distinguishes_refused_scored_and_unscorable():
    ok = SimpleNamespace(evaluate_single=AsyncMock(return_value=SimpleNamespace(faithfulness=0.91234)))
    nan = SimpleNamespace(evaluate_single=AsyncMock(return_value=SimpleNamespace(faithfulness=float("nan"))))
    scored = asyncio.run(ablation._judge(ok, QUESTION, "AD-2024 supersedes it.", ["ctx"]))
    assert scored["status"] == "scored" and scored["faithfulness"] == 0.9123
    refused = asyncio.run(ablation._judge(ok, QUESTION, "The context does not contain that.", ["ctx"]))
    assert refused["status"] == "refused" and refused["faithfulness"] is None
    assert asyncio.run(ablation._judge(ok, QUESTION, "answer", []))["status"] == "refused"  # no contexts
    unscorable = asyncio.run(ablation._judge(nan, QUESTION, "answer", ["ctx"]))
    assert unscorable["status"] == "unscorable" and unscorable["faithfulness"] is None


def test_metadata_names_everything_needed_to_compare_runs():
    meta = run_metadata.collect_run_metadata(
        tenant="aerospace", profile="vector_only", profile_overrides={"bm25_enabled": False}, mode="local",
        config_overrides={}, pinned=True,
        models={"generator": "groq:g", "router": "groq:r", "judge": {"provider": "deepseek", "model": "m"}},
        golden={"version": "2.3", "n_questions": 34})
    for key in ("timestamp", "git", "retrieval_profile", "models", "ragas_version", "golden_set", "tenant",
                "temperature", "pinned_models"):
        assert key in meta
    assert set(meta["git"]) == {"sha", "branch", "dirty"} and meta["temperature"] == 0.0
    assert meta["ragas_version"] is None or isinstance(meta["ragas_version"], str)


def test_output_file_is_one_per_profile_run():
    report = {"run": {"tenant": "aerospace", "retrieval_profile": "text_hybrid",
                      "timestamp": "2026-11-17T10:20:30+00:00"}}
    assert ablation.output_path(Path("out"), report).name == "aerospace__text_hybrid__20261117T102030Z.json"


def test_default_golden_set_is_the_v23_file_for_aerospace_and_tenant_file_otherwise():
    assert ablation.default_golden_set("aerospace") == ROOT / "evals" / "golden_set.json"
    payload = json.loads((ROOT / "evals" / "golden_set.json").read_text(encoding="utf-8"))
    assert payload["version"] == "2.3" and len(payload["questions"]) == 34
    assert ablation.default_golden_set("automotive").name == "queries_automotive.json"


def test_dry_run_makes_no_model_call_and_prints_the_plan(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--tenant", "aerospace", "--profiles", "vector_only", "--dry-run"])
    ablation.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["pinned_models"] is True and plan["golden_set"]["n_questions"] == 34
    assert plan["models"]["generator"].startswith("groq:") and plan["models"]["judge"]["provider"]
    assert list(plan["profiles"]) == ["vector_only"]


def test_unknown_profile_is_rejected_before_anything_runs(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--tenant", "aerospace", "--profiles", "nope", "--dry-run"])
    with pytest.raises(Exception):
        ablation.main()


def _run_with_flags(monkeypatch, tmp_path, extra: list[str]) -> list[dict]:
    """Drive the real _run_profile with a fake retriever; return the kwargs it received."""
    calls: list[dict] = []

    class FakeRetriever:
        async def retrieve_and_answer(self, question, **kwargs):
            calls.append(kwargs)
            return _result()

    monkeypatch.setattr(ablation, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(sys, "argv", ["x", "--tenant", "aerospace", "--profiles", "vector_only"] + extra)
    args = SimpleNamespace(tenant="aerospace", mode="local")
    overrides = {"agentic_fallback": False} if "--no-agentic-fallback" in extra else {}
    report = asyncio.run(ablation._run_profile(
        [QUESTION], args, {"version": "2.3"}, None, {}, "vector_only", overrides, True))
    assert report["run"]["config_overrides"] == overrides
    return calls


def test_no_agentic_fallback_flag_reaches_the_retriever_and_the_result_file(monkeypatch, tmp_path):
    (call,) = _run_with_flags(monkeypatch, tmp_path, ["--no-agentic-fallback"])
    assert call["config_overrides"] == {"agentic_fallback": False}
    assert call["retrieval_profile"] == "vector_only"
    (call,) = _run_with_flags(monkeypatch, tmp_path, [])
    assert call["config_overrides"] is None


def test_main_turns_the_flag_into_the_override(monkeypatch):
    """main() itself must build {"agentic_fallback": False} from --no-agentic-fallback."""
    seen = {}

    async def fake_run_all(questions, args, golden, profiles, config_overrides, pinned):
        seen["overrides"] = config_overrides

    monkeypatch.setattr(ablation, "_run_all", fake_run_all)
    for flags, expected in ((["--no-agentic-fallback"], {"agentic_fallback": False}), ([], {})):
        monkeypatch.setattr(sys, "argv", ["x", "--tenant", "aerospace", "--profiles", "full"] + flags)
        ablation.main()
        assert seen["overrides"] == expected


def _utc(text: str):
    from datetime import datetime, timezone
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


@pytest.mark.parametrize("when, peak", [
    ("2026-11-17T01:00:00", True),    # Tuesday, first peak hour
    ("2026-11-17T03:59:00", True),
    ("2026-11-17T04:00:00", False),   # window is [start, end)
    ("2026-11-17T06:00:00", True),
    ("2026-11-17T09:59:00", True),
    ("2026-11-17T10:00:00", False),
    ("2026-11-17T13:43:00", False),   # the evaluation runs started off-peak
    ("2026-11-21T07:00:00", False),   # Saturday: no peak pricing
    ("2026-11-22T02:00:00", False),   # Sunday
])
def test_deepseek_peak_windows_are_utc_weekday_only(when, peak):
    assert run_metadata.deepseek_peak(_utc(when)) is peak


def test_peak_windows_are_reported_in_bucharest_time_with_dst():
    assert run_metadata.deepseek_peak_windows_bucharest(_utc("2026-11-17T12:00:00")) == ["03:00-06:00", "08:00-12:00"]
    assert run_metadata.deepseek_peak_windows_bucharest(_utc("2026-07-15T12:00:00")) == ["04:00-07:00", "09:00-13:00"]
    window = run_metadata.deepseek_pricing_window(_utc("2026-11-17T13:43:00"))
    assert window["now_bucharest"] == "2026-11-17 15:43 EET" and window["peak_now"] is False
    assert window["peak_multiplier"] == 2.0 and window["peak_days"] == "Mon-Fri"


def test_run_metadata_records_the_start_time_in_bucharest_and_the_peak_flag():
    meta = run_metadata.collect_run_metadata(
        tenant="aerospace", profile="full", profile_overrides={}, mode="local", config_overrides={},
        pinned=True, models={}, golden={})
    assert meta["timestamp_bucharest"].endswith(("+02:00", "+03:00"))
    assert isinstance(meta["deepseek_peak_at_start"], bool)


def test_dry_run_reports_the_pricing_window(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--tenant", "aerospace", "--profiles", "full", "--dry-run"])
    ablation.main()
    window = json.loads(capsys.readouterr().out)["deepseek_pricing_window"]
    assert len(window["peak_windows_bucharest"]) == 2 and isinstance(window["peak_now"], bool)
