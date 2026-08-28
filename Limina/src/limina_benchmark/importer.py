"""Import the Recruiter Agent's historical golden-run output without changing it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .sanitize import sanitize_value
from .schemas import EvaluationCase, TraceNode


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def import_recruiter_history(recruiter_root: Path) -> list[EvaluationCase]:
    """Convert ``eval_results.json`` + ``ops/eval_data.json`` to stable case records.

    The source result only represents successful golden checks. Its source fields are
    retained under ``source_payload`` after sanitization; nothing is silently lost.
    """
    results_path = recruiter_root / "eval_results.json"
    cases_path = recruiter_root / "ops" / "eval_data.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing historical judge output: {results_path}")
    if not cases_path.exists():
        raise FileNotFoundError(f"Missing golden dataset: {cases_path}")

    raw_results = _load_json(results_path)
    raw_cases = _load_json(cases_path)
    if not isinstance(raw_results, dict) or not isinstance(raw_results.get("results"), list):
        raise ValueError("eval_results.json must have a top-level results list")
    case_by_id = {item.get("id"): item for item in raw_cases if isinstance(item, dict)}

    imported: list[EvaluationCase] = []
    for result in raw_results["results"]:
        if not isinstance(result, dict) or not isinstance(result.get("case_id"), str):
            continue
        case_id = result["case_id"]
        golden = case_by_id.get(case_id, {})
        raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
        chat = raw.get("chat") if isinstance(raw.get("chat"), dict) else {}
        reply = str(chat.get("reply") or "")
        message = str(golden.get("user_message") or "")
        state = chat.get("state") if isinstance(chat.get("state"), dict) else {}
        safe_result, redacted_result = sanitize_value(result)
        safe_golden, redacted_golden = sanitize_value(golden)
        safe_message, redacted_message = sanitize_value(message)
        safe_reply, redacted_reply = sanitize_value(reply)
        safe_state, redacted_state = sanitize_value(state)

        imported.append(
            EvaluationCase(
                case_id=case_id,
                category="healthy",
                source="historical",
                synthetic=False,
                source_reference=str(results_path),
                expected_failure=False,
                expected_failure_types=[],
                role=safe_golden.get("expected_role"),
                criteria=safe_golden.get("expected_criteria") or [],
                notes=safe_golden.get("description"),
                redaction_applied=any(
                    [redacted_result, redacted_golden, redacted_message, redacted_reply, redacted_state]
                ),
                trajectory=[
                    TraceNode(node_id="user-1", kind="user", text=safe_message),
                    TraceNode(
                        node_id="agent-1",
                        kind="agent",
                        text=safe_reply,
                        attributes={"state": safe_state},
                    ),
                ],
                source_payload={
                    "golden_case": safe_golden,
                    "historical_judge_result": safe_result,
                    "preservation_note": "Source fields retained after explicit sanitization.",
                },
            )
        )
    if not imported:
        raise ValueError("No valid case_id records found in historical output")
    return imported


def write_dataset(cases: list[EvaluationCase], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([case.model_dump(mode="json") for case in cases], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_dataset(path: Path) -> list[EvaluationCase]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list")
    return [EvaluationCase.model_validate(item) for item in data]
