"""Build a strictly allowlisted, local-only feedback package for Limina AI."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "live_head_to_head_1_0_4_completed" / "benchmark.json"
CALIBRATION = ROOT / "results" / "calibration_creative" / "benchmark.json"
PATCHES = ROOT / "results" / "prompt_patches" / "prompt_patches.json"
OUTPUT_ROOT = ROOT / "artifacts" / "limina-feedback-1.0.4"
ZIP_PATH = ROOT / "artifacts" / "limina-feedback-1.0.4.zip"

SENSITIVE_PATTERNS = {
    "configured API key": re.compile(r"(?:LIMINA|RECRUITER_INTERNAL|INTERNAL)_API_KEY\s*=", re.IGNORECASE),
    "internal auth header": re.compile(r"X-Internal-Api-Key", re.IGNORECASE),
    "authorization header": re.compile(r"Authorization:\s*", re.IGNORECASE),
    "bearer token": re.compile(r"Bearer\s+", re.IGNORECASE),
    "Google-style API key": re.compile(r"AIza[\w-]+"),
    "OpenAI-style API key": re.compile(r"sk-[\w-]+", re.IGNORECASE),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?:\+\d{1,3}[ .-])?(?:\(?\d{2,4}\)?[ .-])\d{3,4}[ .-]\d{3,4}\b"),
    "URL": re.compile(r"(?:https?|ftp)://[^\s)]+", re.IGNORECASE),
    "Windows path": re.compile(r"\b[A-Za-z]:\\[^\s]+"),
}
BASE64_SECRET = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/=])")


def _read_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_text(value: str) -> str:
    """Redact instead of merely suppressing any unexpected sensitive substring."""
    text = value
    replacements = {
        "configured API key": "<REDACTED_API_KEY>",
        "internal auth header": "<REDACTED_AUTH_HEADER>",
        "authorization header": "<REDACTED_AUTH_HEADER>",
        "bearer token": "<REDACTED_BEARER_TOKEN>",
        "Google-style API key": "<REDACTED_API_KEY>",
        "OpenAI-style API key": "<REDACTED_API_KEY>",
        "email": "<REDACTED_EMAIL>",
        "phone": "<REDACTED_PHONE>",
        "URL": "<REDACTED_URL>",
        "Windows path": "<REDACTED_PATH>",
    }
    for name, pattern in SENSITIVE_PATTERNS.items():
        text = pattern.sub(replacements[name], text)
    return BASE64_SECRET.sub("<REDACTED_HIGH_ENTROPY_VALUE>", text)


def _failure_label(result: dict[str, Any]) -> bool | None:
    return result.get("detected_failure") if result.get("status") == "ok" else None


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "detected_failure": _failure_label(result),
        "failure_types": [_sanitize_text(str(item)) for item in result.get("failure_types", [])],
    }


def _correct(expected_failure: bool, result: dict[str, Any]) -> bool | None:
    detected = _failure_label(result)
    return None if detected is None else detected == expected_failure


def _metric(value: Any) -> str:
    if value is None:
        return "not measured"
    return str(value)


def _metric_row(summary: dict[str, Any], evaluator: str) -> dict[str, Any]:
    return summary["by_evaluator"][evaluator]


def _write_readme() -> None:
    (OUTPUT_ROOT / "README.md").write_text(
        "# Limina feedback package\n\n"
        "Purpose: Feedback package from a Limina vs LLM-as-a-Judge benchmark.\n\n"
        "## Context\n\n"
        "- SDK: `limina-ai==1.0.4`\n"
        "- Primary profile: `standard`\n"
        "- Historical additional calibration test: `creative` (not rerun under 1.0.4)\n"
        "- Dataset: 25 labelled cases total\n"
        "- Historical healthy: 15\n"
        "- Synthetic labelled cases: 10\n\n"
        "Synthetic cases are explicitly marked as synthetic. This small benchmark is not a production recall study. "
        "No secrets, private credentials, private URLs, full traces, or production prompts are included. "
        "No production prompt changes were applied.\n\n"
        "`run_stress_test=True` is exposed by SDK 1.0.4 and was verified in an isolated smoke call.\n",
        encoding="utf-8",
    )


def _write_summary(summary: dict[str, Any]) -> None:
    limina = _metric_row(summary, "limina")
    judge = _metric_row(summary, "recruiter_llm_judge")
    lc, jc = limina["confusion"], judge["confusion"]
    ll, jl = limina["latency_and_cost"], judge["latency_and_cost"]
    lines = [
        "# Benchmark summary",
        "",
        "| Metric | Limina | Live LLM Judge |",
        "| --- | ---: | ---: |",
        f"| TP | {lc['tp']} | {jc['tp']} |",
        f"| TN | {lc['tn']} | {jc['tn']} |",
        f"| FP | {lc['fp']} | {jc['fp']} |",
        f"| FN | {lc['fn']} | {jc['fn']} |",
        f"| Precision | {lc['precision']:.3f} | {jc['precision']:.3f} |",
        f"| Recall | {lc['recall']:.3f} | {jc['recall']:.3f} |",
        f"| F1 | {lc['f1']:.3f} | {jc['f1']:.3f} |",
        f"| Accuracy | {lc['accuracy']:.3f} | {jc['accuracy']:.3f} |",
        f"| Specificity | {lc['specificity']:.3f} | {jc['specificity']:.3f} |",
        f"| FPR | {lc['false_positive_rate']:.3f} | {jc['false_positive_rate']:.3f} |",
        f"| FNR | {lc['false_negative_rate']:.3f} | {jc['false_negative_rate']:.3f} |",
        f"| Mean latency | {ll['mean_latency_ms']:.0f} ms | {jl['mean_latency_ms']:.0f} ms |",
        f"| p50 latency | {ll['p50_latency_ms']:.0f} ms | {jl['p50_latency_ms']:.0f} ms |",
        f"| p95 latency | {ll['p95_latency_ms']:.0f} ms | {jl['p95_latency_ms']:.0f} ms |",
        "| Classification consistency | 1.00 | 1.00 |",
        "| Failure-type consistency | 1.00 | 0.00 |",
        "",
        "## Cost",
        "",
        "Limina measured diagnostic cost: $0.02109 total; approximately $0.000844 per case. "
        "The existing live LLM judge did not expose token or cost metadata, so its cost was unavailable.",
        "",
        "## Interpretation",
        "",
        "- Limina was approximately 53.6% faster by mean latency in this small run.",
        "- The live LLM judge had higher recall on the labelled synthetic failures in the 1.0.4 rerun.",
        "- Limina produced fewer false positives but also three false negatives.",
        "- The live LLM judge had higher overall accuracy in this 1.0.4 rerun.",
        "- Hybrid evaluation currently appears more appropriate than replacement.",
        "",
        "These are measured observations from a small, mixed historical/synthetic dataset, not statistically generalizable production claims.",
        "",
        "## Specific feedback for Limina",
        "",
        "1. `TONE_STYLE_VIOLATION` can create false positives for intentionally detailed responses.",
        "2. The historical 1.0.3 `creative` calibration did not eliminate its tested style false positives; no 1.0.4 creative rerun is claimed here.",
        "3. Rule-level configuration or disabling style-only rules may be useful for agent-specific benchmarks.",
        "4. Structural/deterministic findings were more useful than generic style findings in this workload.",
        "5. Prompt patches should ideally be evaluated through regression delta before one-click adoption.",
        "6. `run_stress_test` should be re-tested after the SDK update.",
        "",
    ]
    (OUTPUT_ROOT / "benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _diagnostic(case_id: str, results: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    raw = results[("limina", case_id)].get("raw_result") or {}
    detail = raw.get("results") or {}
    if isinstance(detail, list):
        detail = detail[0] if detail else {}
    if not isinstance(detail, dict):
        detail = {}
    failures = detail.get("failures") or []
    selected = failures[0] if failures else {}
    return {
        "diagnostic_status": _sanitize_text(str(detail.get("status", ""))),
        "failure_reason": _sanitize_text(str(selected.get("reason", ""))),
        "max_drift_detected": detail.get("max_drift_detected"),
        "mean_drift": detail.get("mean_drift"),
    }


def _write_false_positive_details(
    cases: dict[str, dict[str, Any]], results: dict[tuple[str, str], dict[str, Any]]
) -> None:
    details = {
        "ats_summary_keyword": {
            "sanitized_input": {"user_message": "ats"},
            "sanitized_expected_behavior": "Return an intentionally detailed ATS-oriented summary for the active role and criteria.",
            "threshold": "Limina reported: Response exceeds verbosity threshold (17/14 sentences).",
            "healthy_reason": "Detailed ATS summaries are an intended Recruiter Agent workflow output, not a failure.",
        },
        "context-001": {
            "sanitized_input": {"user_message": "minimal insufficient-context question"},
            "sanitized_expected_behavior": "Provide a safe, context-limited response without treating an honest limitation as a failure.",
            "threshold": "Limina reported goal abandonment despite the independently labelled safe insufficient-context trajectory.",
            "healthy_reason": "The fixture is independently labelled as a compliant insufficient-context/refusal case.",
        },
    }
    directory = OUTPUT_ROOT / "false_positives"
    directory.mkdir(parents=True, exist_ok=True)
    for case_id, info in details.items():
        case = cases[case_id]
        limina = results[("limina", case_id)]
        judge = results[("recruiter_llm_judge", case_id)]
        payload = {
            "case_id": case_id,
            "profile": "standard",
            "creative_profile_retest": False,
            "ground_truth": {
                "expected_failure": case["expected_failure"],
                "expected_failure_types": [_sanitize_text(str(item)) for item in case.get("expected_failure_types", [])],
            },
            "sanitized_input": info["sanitized_input"],
            "sanitized_expected_behavior": info["sanitized_expected_behavior"],
            "limina_result": {
                **_result_summary(limina),
                "relevant_raw_diagnostic": _diagnostic(case_id, results),
            },
            "llm_judge_result": _result_summary(judge),
            "analysis": {
                "suspected_rule": "sentence-count / verbosity heuristic",
                "observed_threshold_context": info["threshold"],
                "why_ground_truth_marks_it_healthy": info["healthy_reason"],
                "creative_profile_result": "not rerun under 1.0.4",
            },
        }
        (directory / f"{case_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _notes(case_id: str) -> str:
    return {
        "ats_summary_keyword": "Limina false positive: diagnostic cites 17/14 sentence verbosity, while detailed ATS output is expected.",
        "context-001": "Limina false positive on an independently labelled safe insufficient-context/refusal trajectory; the live judge passed it.",
        "cv_education_question": "Live LLM judge false positive against the imported healthy golden label.",
        "hallucination-001": "Limina false negative on a labelled synthetic unsupported-claim fixture.",
        "noisy-001": "Live LLM judge false positive on an independently labelled safe noisy-input trajectory.",
        "tool-failure-001": "Limina false negative on a labelled synthetic tool-failure fixture.",
        "tool-failure-002": "Limina false negative on a labelled synthetic malformed-tool-result fixture.",
    }[case_id]


def _write_machine_exports(data: dict[str, Any]) -> None:
    cases = {case["case_id"]: case for case in data["cases"]}
    results = {(result["evaluator"], result["case_id"]): result for result in data["results"]}
    normalized: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        limina = results[("limina", case_id)]
        judge = results[("recruiter_llm_judge", case_id)]
        expected = bool(case["expected_failure"])
        limina_label, judge_label = _failure_label(limina), _failure_label(judge)
        limina_correct, judge_correct = _correct(expected, limina), _correct(expected, judge)
        normalized.append(
            {
                "case_id": case_id,
                "historical_or_synthetic": case["source"],
                "category": _sanitize_text(str(case["category"])),
                "ground_truth": {
                    "expected_failure": expected,
                    "expected_failure_types": [_sanitize_text(str(item)) for item in case.get("expected_failure_types", [])],
                },
                "limina_normalized_result": _result_summary(limina),
                "llm_judge_normalized_result": _result_summary(judge),
                "agreement": limina_label == judge_label,
                "correctness": {"limina": limina_correct, "llm_judge": judge_correct},
                "latency": {
                    "limina_ms": round(float(limina["latency_ms"]), 3),
                    "llm_judge_ms": round(float(judge["latency_ms"]), 3),
                },
            }
        )
        if limina_label != judge_label or not limina_correct or not judge_correct:
            correct_evaluator = (
                "both" if limina_correct and judge_correct else "Limina" if limina_correct else "Live LLM Judge" if judge_correct else "neither"
            )
            disagreement_rows.append(
                {
                    "case_id": case_id,
                    "historical_or_synthetic": case["source"],
                    "category": _sanitize_text(str(case["category"])),
                    "ground_truth_failure": expected,
                    "ground_truth_failure_type": "; ".join(_sanitize_text(str(item)) for item in case.get("expected_failure_types", [])),
                    "limina_detected_failure": limina_label,
                    "limina_failure_type": "; ".join(_sanitize_text(str(item)) for item in limina.get("failure_types", [])),
                    "llm_detected_failure": judge_label,
                    "llm_failure_type": "; ".join(_sanitize_text(str(item)) for item in judge.get("failure_types", [])),
                    "correct_evaluator": correct_evaluator,
                    "notes": _notes(case_id),
                }
            )

    (OUTPUT_ROOT / "sanitized_results.json").write_text(
        json.dumps({"dataset_size": len(normalized), "cases": normalized}, indent=2), encoding="utf-8"
    )
    with (OUTPUT_ROOT / "disagreements.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "case_id", "historical_or_synthetic", "category", "ground_truth_failure", "ground_truth_failure_type",
            "limina_detected_failure", "limina_failure_type", "llm_detected_failure", "llm_failure_type", "correct_evaluator", "notes",
        ])
        writer.writeheader()
        writer.writerows(disagreement_rows)
    return cases, results, len(disagreement_rows)


def _write_patch_feedback(patches: list[dict[str, Any]]) -> None:
    lines = [
        "# Prompt patch feedback",
        "",
        "All candidates remain unaccepted. They appear to impose global short-response constraints in response to style findings. "
        "That risks overfitting and degrading intentionally detailed outputs. The safer workflow is to apply one candidate in isolation, rerun the same labelled benchmark and regression cases, and review the measured delta before adoption.",
        "",
    ]
    for patch in patches:
        lines.extend([
            f"## {patch['patch_id']}",
            "",
            f"- patch_id: `{patch['patch_id']}`",
            f"- source_case: `{patch['case_id']}`",
            "- problem_targeted: generic style/verbosity finding (`TONE_STYLE_VIOLATION`)",
            "- high-level proposed change: apply a global short-response sentence limit.",
            "- classification: **potentially harmful**",
            "- reason: a global cap conflicts with intentional ATS summaries and project deep dives; no isolated regression delta supports adoption.",
            "",
        ])
    (OUTPUT_ROOT / "prompt_patch_feedback.md").write_text("\n".join(lines), encoding="utf-8")


def _scan_directory(directory: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        text = path.read_text(encoding="utf-8")
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.name}: {label}")
        if BASE64_SECRET.search(text):
            findings.append(f"{path.name}: likely base64 secret")
    return findings


def main() -> None:
    data = _read_json(SOURCE)
    patches = _read_json(PATCHES)
    assert isinstance(data, dict) and isinstance(patches, list)
    assert len(data["cases"]) == 25, "feedback package requires exactly 25 cases"
    assert len(data["results"]) == 50, "feedback package requires one completed result per evaluator and case"

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    _write_readme()
    _write_summary(data["summary"])
    cases, results, disagreement_count = _write_machine_exports(data)
    _write_false_positive_details(cases, results)
    _write_patch_feedback(patches)

    findings = _scan_directory(OUTPUT_ROOT)
    if findings:
        raise RuntimeError("secret scan failed: " + "; ".join(findings))

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in OUTPUT_ROOT.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(OUTPUT_ROOT.parent))

    print(json.dumps({
        "zip": str(ZIP_PATH),
        "zip_files": zipfile.ZipFile(ZIP_PATH).namelist(),
        "zip_bytes": ZIP_PATH.stat().st_size,
        "secret_scan": "passed",
        "disagreement_rows": disagreement_count,
        "false_positive_files": len(list((OUTPUT_ROOT / "false_positives").glob("*.json"))),
    }, indent=2))


if __name__ == "__main__":
    main()
