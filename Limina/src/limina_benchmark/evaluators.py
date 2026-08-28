"""Adapters for Limina and the Recruiter Agent's existing judge."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests

from .config import Settings
from .schemas import EvaluationCase, EvaluatorResult


@contextmanager
def _evaluation_directory(path: Path | None):
    """Keep SDK-generated `report.html` with the corresponding result artifact."""
    if path is None:
        yield
        return
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _message(case: EvaluationCase, kind: str) -> str:
    for node in case.trajectory:
        if node.kind == kind:
            return node.text
    return ""


def historical_judge_result(case: EvaluationCase) -> EvaluatorResult:
    """Replay the saved Recruiter LLM-judge verdict without pretending it is new."""
    raw = case.source_payload.get("historical_judge_result")
    if not isinstance(raw, dict):
        return EvaluatorResult(
            evaluator="recruiter_llm_judge_historical",
            case_id=case.case_id,
            status="skipped",
            raw_result={},
            error="No historical judge result is attached to this case.",
        )
    score = raw.get("score")
    passed = raw.get("passed")
    return EvaluatorResult(
        evaluator="recruiter_llm_judge_historical",
        case_id=case.case_id,
        status="ok",
        detected_failure=(not bool(passed)) if isinstance(passed, bool) else None,
        failure_types=list(raw.get("state_issues") or []) + list(raw.get("issues") or []),
        score=float(score) if isinstance(score, (int, float)) else None,
        # Historical file has no evaluator timing/cost; preserve unknown rather than estimate.
        latency_ms=None,
        estimated_cost_usd=None,
        raw_result=raw,
    )


def call_recruiter_judge(case: EvaluationCase, settings: Settings) -> EvaluatorResult:
    """Call the existing `/mcp/call` judge endpoint on an already captured trajectory."""
    if not settings.recruiter_base_url:
        return EvaluatorResult(
            evaluator="recruiter_llm_judge",
            case_id=case.case_id,
            status="skipped",
            error="RECRUITER_BASE_URL is not configured.",
        )
    headers = (
        {"X-Internal-Api-Key": settings.recruiter_internal_api_key}
        if settings.recruiter_internal_api_key
        else {}
    )
    payload = {
        "tool": "judge_recruiter_turn",
        "arguments": {
            "role": case.role,
            "criteria": case.criteria,
            "user_message": _message(case, "user"),
            "agent_reply": _message(case, "agent"),
        },
    }
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{settings.recruiter_base_url.rstrip('/')}/mcp/call",
            json=payload,
            headers=headers,
            timeout=settings.request_timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        judge = body.get("result", body)
        if not isinstance(judge, dict):
            raise ValueError("Judge endpoint returned a non-object result")
        score = float(judge.get("score")) if judge.get("score") is not None else None
        return EvaluatorResult(
            evaluator="recruiter_llm_judge",
            case_id=case.case_id,
            status="ok",
            detected_failure=score < 3.5 if score is not None else None,
            failure_types=list(judge.get("issues") or []),
            score=score,
            latency_ms=(time.perf_counter() - started) * 1000,
            estimated_cost_usd=None,
            raw_result=judge,
        )
    except Exception as exc:
        return EvaluatorResult(
            evaluator="recruiter_llm_judge",
            case_id=case.case_id,
            status="error",
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )


def _limina_failure_fields(report: dict[str, Any]) -> tuple[bool | None, list[str], float | None]:
    summary = report.get("executive_summary")
    if not isinstance(summary, dict):
        return None, [], None
    errors = summary.get("errors_detected")
    detected = bool(errors and float(errors) > 0) if isinstance(errors, (int, float)) else None
    vulnerable = summary.get("most_vulnerable_component")
    failure_types = (
        [part.strip() for part in str(vulnerable).split(",") if part.strip()]
        if vulnerable
        else []
    )
    success = summary.get("success_rate_percentage")
    score = float(success) / 100 if isinstance(success, (int, float)) else None
    return detected, failure_types, score


def case_to_limina_payload(case: EvaluationCase) -> dict[str, Any]:
    """Convert one normalized trajectory into the SDK's documented DAG envelope."""
    return {
        "session_id": case.case_id,
        "description": case.notes or case.category,
        "nodes": [
            {
                "id": node.node_id,
                "type": "agent" if node.kind == "state" else node.kind,
                "label": node.name or node.kind.upper(),
                "text": node.text,
                **({"execution_time_ms": node.latency_ms} if node.latency_ms is not None else {}),
            }
            for node in case.trajectory
        ],
        "edges": [
            {"from": case.trajectory[index].node_id, "to": case.trajectory[index + 1].node_id}
            for index in range(max(0, len(case.trajectory) - 1))
        ],
    }


def evaluate_with_limina(
    case: EvaluationCase, settings: Settings, output_dir: Path | None = None
) -> EvaluatorResult:
    """Evaluate one case at a time because Limina returns an aggregate batch report.

    ``limina-ai==1.0.4`` exposes ``run_stress_test`` on its public evaluation
    methods. Normal benchmark cases remain isolated single-trajectory calls so
    the prior 1.0.3 baseline remains directly comparable.
    """
    if not settings.limina_enabled:
        return EvaluatorResult(
            evaluator="limina",
            case_id=case.case_id,
            status="skipped",
            error="LIMINA_ENABLED is false.",
        )
    if not settings.limina_api_key:
        return EvaluatorResult(
            evaluator="limina",
            case_id=case.case_id,
            status="skipped",
            error="LIMINA_API_KEY is not configured.",
        )
    started = time.perf_counter()
    try:
        from limina import LiminaMonitor  # installed only for explicit Limina runs

        monitor = LiminaMonitor(
            api_key=settings.limina_api_key,
            profile=settings.limina_profile,
            export_html=settings.limina_export_html,
        )
        with _evaluation_directory(output_dir):
            report = monitor.evaluate([case_to_limina_payload(case)])
        if not isinstance(report, dict):
            raise ValueError("Limina returned a non-object report")
        if report.get("status") == "ERROR" or report.get("error"):
            return EvaluatorResult(
                evaluator="limina",
                case_id=case.case_id,
                status="error",
                latency_ms=(time.perf_counter() - started) * 1000,
                raw_result=report,
                error=str(report.get("error") or "Limina returned an error status"),
            )
        detected, failure_types, score = _limina_failure_fields(report)
        return EvaluatorResult(
            evaluator="limina",
            case_id=case.case_id,
            status="ok",
            detected_failure=detected,
            failure_types=failure_types,
            score=score,
            latency_ms=(time.perf_counter() - started) * 1000,
            estimated_cost_usd=None,
            raw_result=report,
        )
    except Exception as exc:
        return EvaluatorResult(
            evaluator="limina",
            case_id=case.case_id,
            status="error",
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )
