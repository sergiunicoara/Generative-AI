"""Reproducible CLI; ordinary commands make no paid/external evaluator calls."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

from .config import Settings
from .evaluators import call_recruiter_judge, evaluate_with_limina, historical_judge_result
from .importer import import_recruiter_history, load_dataset, write_dataset
from .reporting import write_artifacts, write_case_accounting, write_head_to_head_comparison
from .sanitize import sanitize_value
from .schemas import EvaluationCase, EvaluatorResult, TraceNode
from .synthetic import build_synthetic_cases


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recruiter Agent / Limina isolated benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    import_command = commands.add_parser("import-historical", help="Convert existing Recruiter eval artifacts into a sanitized dataset")
    import_command.add_argument("--recruiter-root", type=Path, required=True)
    import_command.add_argument("--output", type=Path, default=Path("datasets/recruiter_historical.json"))

    synthetic = commands.add_parser("build-failure-dataset", help="Append independently labelled synthetic failure fixtures")
    synthetic.add_argument("--historical", type=Path, required=True)
    synthetic.add_argument("--output", type=Path, default=Path("datasets/failure_benchmark.json"))

    capture = commands.add_parser("capture", help="Capture one real /chat response without changing Recruiter code")
    capture.add_argument("--case-id", required=True)
    capture.add_argument("--category", required=True)
    capture.add_argument("--message", required=True)
    capture.add_argument("--expected-failure", action="store_true")
    capture.add_argument("--failure-type", action="append", default=[])
    capture.add_argument("--expected-refusal", action="store_true")
    capture.add_argument("--role")
    capture.add_argument("--criterion", action="append", default=[])
    capture.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run", help="Evaluate one dataset with the selected evaluators")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=Path("results/latest"))
    run.add_argument("--judge", choices=("historical", "http", "none"), default="historical")
    run.add_argument("--limina", action="store_true", help="Use Limina only when LIMINA_ENABLED=true and LIMINA_API_KEY is set")
    run.add_argument("--repeats", type=int, default=1)

    report = commands.add_parser("report", help="Regenerate artifacts from an existing benchmark JSON")
    report.add_argument("--dataset", type=Path, required=True)
    report.add_argument("--results-json", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)

    merge = commands.add_parser("merge-historical", help="Pair completed Limina results with saved same-case judge results")
    merge.add_argument("--dataset", type=Path, required=True)
    merge.add_argument("--limina-results", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)

    merge_results = commands.add_parser(
        "merge-results",
        help="Merge benchmark artifacts, preferring a successful retry for the same evaluator and case",
    )
    merge_results.add_argument("--dataset", type=Path, required=True)
    merge_results.add_argument("--results-json", type=Path, action="append", required=True)
    merge_results.add_argument("--output-dir", type=Path, required=True)

    patches = commands.add_parser("extract-patches", help="Save Limina patch suggestions as unaccepted candidates")
    patches.add_argument("--results-json", type=Path, required=True)
    patches.add_argument("--output-dir", type=Path, required=True)

    accounting = commands.add_parser("accounting", help="Write auditable per-case TP/TN/FP/FN rows")
    accounting.add_argument("--dataset", type=Path, required=True)
    accounting.add_argument("--results-json", type=Path, required=True)
    accounting.add_argument("--evaluator", required=True)
    accounting.add_argument("--output-dir", type=Path, required=True)

    compare = commands.add_parser("compare", help="Write a same-case side-by-side evaluator ledger")
    compare.add_argument("--dataset", type=Path, required=True)
    compare.add_argument("--results-json", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)

    subset = commands.add_parser("subset", help="Write a named subset of an existing dataset")
    subset.add_argument("--dataset", type=Path, required=True)
    subset.add_argument("--case-id", action="append", required=True)
    subset.add_argument("--output", type=Path, required=True)
    return parser


def _capture(args: argparse.Namespace, settings: Settings) -> int:
    if not settings.recruiter_base_url:
        raise ValueError("RECRUITER_BASE_URL is required for capture")
    started = time.perf_counter()
    response = requests.post(
        f"{settings.recruiter_base_url.rstrip('/')}/chat",
        json={"session_id": f"limina-{args.case_id}", "message": args.message, "source": "limina_benchmark"},
        timeout=settings.request_timeout_s,
    )
    response.raise_for_status()
    body = response.json()
    safe_message, message_redacted = sanitize_value(args.message)
    safe_reply, reply_redacted = sanitize_value(str(body.get("reply") or ""))
    safe_body, body_redacted = sanitize_value(body)
    case = EvaluationCase(
        case_id=args.case_id,
        category=args.category,
        source="synthetic",
        source_reference=f"{settings.recruiter_base_url.rstrip('/')}/chat",
        trajectory=[
            TraceNode(node_id="user-1", kind="user", text=safe_message),
            TraceNode(node_id="agent-1", kind="agent", text=safe_reply, latency_ms=(time.perf_counter() - started) * 1000, attributes={"state": safe_body.get("state", {})}),
        ],
        expected_failure=args.expected_failure,
        expected_failure_types=args.failure_type,
        expected_refusal=True if args.expected_refusal else None,
        role=args.role,
        criteria=args.criterion,
        notes="Captured by isolated Limina benchmark harness; label independently before comparison.",
        redaction_applied=message_redacted or reply_redacted or body_redacted,
        source_payload={"chat_response": safe_body, "sanitization": "Common direct identifiers and credential-like strings are redacted before persistence."},
    )
    write_dataset([case], args.output)
    print(f"Captured sanitized-case envelope: {args.output}")
    return 0


def _run(args: argparse.Namespace, settings: Settings) -> int:
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.limina and settings.limina_enabled and not settings.limina_api_key:
        raise ValueError(
            "Limina evaluation was requested with LIMINA_ENABLED=true, but "
            "LIMINA_API_KEY is missing. Set it in the environment or local .env."
        )
    cases = load_dataset(args.dataset)
    results = []
    for _ in range(args.repeats):
        for case in cases:
            if args.judge == "historical":
                results.append(historical_judge_result(case))
            elif args.judge == "http":
                results.append(call_recruiter_judge(case, settings))
            if args.limina:
                results.append(evaluate_with_limina(case, settings, args.output_dir))
    paths = write_artifacts(cases, results, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.from_environment()
    try:
        if args.command == "import-historical":
            cases = import_recruiter_history(args.recruiter_root)
            write_dataset(cases, args.output)
            print(f"Imported {len(cases)} sanitized historical cases into {args.output}")
            return 0
        if args.command == "capture":
            return _capture(args, settings)
        if args.command == "build-failure-dataset":
            cases = load_dataset(args.historical) + build_synthetic_cases()
            write_dataset(cases, args.output)
            print(f"Built {len(cases)} cases ({len(cases) - len(load_dataset(args.historical))} synthetic) into {args.output}")
            return 0
        if args.command == "report":
            cases = load_dataset(args.dataset)
            payload = json.loads(args.results_json.read_text(encoding="utf-8"))
            results = [EvaluatorResult.model_validate(item) for item in payload.get("results", [])]
            paths = write_artifacts(cases, results, args.output_dir)
            print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
            return 0
        if args.command == "merge-historical":
            cases = load_dataset(args.dataset)
            payload = json.loads(args.limina_results.read_text(encoding="utf-8"))
            limina_results = [EvaluatorResult.model_validate(item) for item in payload.get("results", []) if item.get("evaluator") == "limina"]
            judge_results = []
            by_id = {case.case_id: case for case in cases}
            for limina_result in limina_results:
                case = by_id.get(limina_result.case_id)
                if case is not None and case.source == "historical":
                    judge_results.append(historical_judge_result(case))
            paths = write_artifacts(cases, limina_results + judge_results, args.output_dir)
            print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
            return 0
        if args.command == "merge-results":
            cases = load_dataset(args.dataset)
            merged: dict[tuple[str, str], EvaluatorResult] = {}
            for result_json in args.results_json:
                payload = json.loads(result_json.read_text(encoding="utf-8"))
                for item in payload.get("results", []):
                    result = EvaluatorResult.model_validate(item)
                    key = (result.evaluator, result.case_id)
                    previous = merged.get(key)
                    if previous is None or (previous.status != "ok" and result.status == "ok"):
                        merged[key] = result
            paths = write_artifacts(cases, list(merged.values()), args.output_dir)
            print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
            return 0
        if args.command == "extract-patches":
            payload = json.loads(args.results_json.read_text(encoding="utf-8"))
            args.output_dir.mkdir(parents=True, exist_ok=True)
            candidates = []
            patch_counts: dict[str, int] = {}
            for result in payload.get("results", []):
                narrative = (result.get("raw_result") or {}).get("narrative_report", "")
                if not narrative:
                    continue
                blocks = re.findall(r"```diff\s*\n(.*?)\n```", str(narrative), flags=re.DOTALL)
                for index, block in enumerate(blocks, start=1):
                    base_id = f"{result.get('case_id', 'unknown')}-{index}"
                    patch_counts[base_id] = patch_counts.get(base_id, 0) + 1
                    candidates.append({
                        "patch_id": f"{base_id}-run{patch_counts[base_id]}",
                        "case_id": result.get("case_id"),
                        "proposed_change": block,
                        "target_result_before": result,
                        "target_result_after": None,
                        "regressions": None,
                        "accepted": False,
                        "classification": "potentially_harmful",
                        "status": "candidate_only",
                        "reason": "The proposed global sentence cap conflicts with detailed ATS/deep-dive outputs; not applied or regression-tested.",
                    })
            (args.output_dir / "prompt_patches.json").write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
            (args.output_dir / "summary.md").write_text(
                f"# Prompt patch candidates\n\nExtracted {len(candidates)} candidate patch(es). All remain rejected/candidate-only because no isolated before/after regression run was performed. Production prompts were not modified.\n",
                encoding="utf-8",
            )
            print(f"Extracted {len(candidates)} unaccepted candidates into {args.output_dir}")
            return 0
        if args.command == "accounting":
            cases = load_dataset(args.dataset)
            payload = json.loads(args.results_json.read_text(encoding="utf-8"))
            results = [EvaluatorResult.model_validate(item) for item in payload.get("results", [])]
            paths = write_case_accounting(cases, results, args.evaluator, args.output_dir)
            print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
            return 0
        if args.command == "compare":
            cases = load_dataset(args.dataset)
            payload = json.loads(args.results_json.read_text(encoding="utf-8"))
            results = [EvaluatorResult.model_validate(item) for item in payload.get("results", [])]
            path = write_head_to_head_comparison(cases, results, args.output_dir)
            print(json.dumps({"comparison": str(path)}, indent=2))
            return 0
        if args.command == "subset":
            requested = set(args.case_id)
            cases = [case for case in load_dataset(args.dataset) if case.case_id in requested]
            missing = requested - {case.case_id for case in cases}
            if missing:
                raise ValueError(f"Unknown case ids: {', '.join(sorted(missing))}")
            write_dataset(cases, args.output)
            print(f"Wrote {len(cases)} cases to {args.output}")
            return 0
        if args.command == "run":
            return _run(args, settings)
    except Exception as exc:
        print(f"benchmark error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
