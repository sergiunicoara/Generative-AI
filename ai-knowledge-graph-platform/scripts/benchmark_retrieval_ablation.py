#!/usr/bin/env python
"""Run reproducible retrieval-profile ablations against a golden set.

Each profile run writes ONE self-describing JSON file:
  * ``run``      commit, timestamp, profile, models (generator / router / judge),
                 RAGAS version, golden-set version and hash, tenant, temperature;
  * ``summary``  pass rate, citation recall, faithfulness mean, refusal / unscorable /
                 error counts, agentic-fallback count, latency percentiles;
  * ``results``  one row per question: golden pass/fail, faithfulness (or refused /
                 unscorable), end-to-end and retrieval latency, whether the agentic
                 fallback fired.

Evaluation mode is ON by default (``EVAL_PINNED_MODELS=1``): one generator (Groq
``groq_model``), one router (Groq ``groq_fast_model``) and one RAGAS judge
(``EVAL_JUDGE_PROVIDER``), temperature 0, no fallback chain. A missing key or a
provider error fails the run instead of silently switching provider. The retriever
is called without a query ID, which disables the governed answer cache.

``latency_e2e_ms`` is the wall time of ``retrieve_and_answer`` (retrieval + query
rewrite + generation, plus the agentic fallback when it fires). It EXCLUDES RAGAS
judging. ``latency_retrieval_ms`` is the search stage only.

Examples:
    python scripts/benchmark_retrieval_ablation.py --tenant aerospace --dry-run
    python scripts/benchmark_retrieval_ablation.py --tenant aerospace --profiles vector_only
    python scripts/benchmark_retrieval_ablation.py --tenant aerospace --profiles full --no-agentic-fallback
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.evaluation.run_metadata import (  # noqa: E402
    agentic_fallback,
    collect_run_metadata,
    deepseek_pricing_window,
    golden_set_info,
    percentile,
    retrieval_latency_ms,
)
from graphrag.evaluation import citations, full_context  # noqa: E402
from graphrag.evaluation.citations import strict_citations  # noqa: E402
from graphrag.retrieval.answer_policy import answer_prompt  # noqa: E402
from graphrag.retrieval.hybrid_retriever import (  # noqa: E402
    HybridRetriever,
    retrieval_profile_overrides as _retriever_profile_overrides,
)
from run_golden_eval import _check  # noqa: E402

def retrieval_profile_overrides(profile: str) -> dict:
    """Retriever profiles plus the evaluation-only ``full_context`` (no retrieval, so no overrides)."""
    return {} if profile == full_context.PROFILE else _retriever_profile_overrides(profile)


DEFAULT_OUTPUT_DIR = ROOT / "evals" / "ablation"
AEROSPACE_GOLDEN = ROOT / "evals" / "golden_set.json"


def default_golden_set(tenant: str) -> Path:
    """evals/golden_set.json (v2.3) when it targets this tenant, else the tenant's own file."""
    try:
        if json.loads(AEROSPACE_GOLDEN.read_text(encoding="utf-8")).get("tenant") == tenant:
            return AEROSPACE_GOLDEN
    except (OSError, ValueError):
        pass
    return ROOT / "data" / "eval_golden" / f"queries_{tenant}.json"


async def _judge(evaluator, question: dict, answer: str, contexts: list[str]) -> dict:
    """Faithfulness for one answer: refused (judge skipped), scored or unscorable."""
    from scripts.run_faithfulness_eval import _evaluate_with_retries, _is_refusal

    if _is_refusal(answer, contexts):
        return {"status": "refused", "faithfulness": None, "attempts": 0, "notes": []}
    result, attempts, notes = await _evaluate_with_retries(
        evaluator, question["id"], question["question"], answer, contexts)
    value = getattr(result, "faithfulness", None)
    if isinstance(value, (int, float)) and value == value and abs(value) != float("inf"):
        return {"status": "scored", "faithfulness": round(float(value), 4),
                "attempts": attempts, "notes": notes}
    return {"status": "unscorable", "faithfulness": None, "attempts": attempts, "notes": notes}


def split_answer(result):
    """``(result with the [DOC: id] tags stripped from its answer, raw tag ids)``: the same for every profile."""
    cleaned, tag_ids = citations.parse_doc_tags(result.answer)
    return result.model_copy(update={"answer": cleaned}), tag_ids


def build_row(question: dict, result, elapsed_ms: float, judged: dict | None, *,
              answer_ids: list[str] | None = None, basis: str = "context", raw_answer: str | None = None) -> dict:
    """One per-question record from a QueryResult whose answer is already tag-free (pure: no I/O).

    Two strictly normalised citation sets are recorded: ``context`` (the retriever's provenance list; None
    for full_context, which has no retrieval) and ``answer`` (ids parsed from the answer's [DOC: id] tags).
    The golden verdict scores the ``basis`` set: context for the retrieval profiles (comparable with the
    earlier runs), answer for full_context."""
    lenient_passed, lenient_failures = _check(question, result.model_dump(mode="json"))
    context_set = None if basis == "answer" else strict_citations(list(result.citations))
    answer_set = strict_citations(list(answer_ids or []))
    scored = answer_set if basis == "answer" else context_set
    citation_failure = citations.citation_failure(question, scored)
    failures = [f for f in lenient_failures if not f.startswith(citations.CITATION_FAILURE)]
    failures += [citation_failure] if citation_failure else []
    if judged is not None:
        refused = judged["status"] == "refused"
    else:
        from scripts.run_faithfulness_eval import _is_refusal   # the judge's own refusal rule

        refused = _is_refusal(result.answer, list(result.contexts))
    tag_ids = list(answer_ids or [])
    resolved_tags = [tag for tag in tag_ids if strict_citations([tag])]
    judged = judged or {"status": "not_judged", "faithfulness": None, "attempts": 0, "notes": []}
    context_m, answer_m = citations.set_metrics(question, context_set), citations.set_metrics(question, answer_set)
    basis_m = answer_m if basis == "answer" else context_m
    return {
        "id": question["id"],
        "type": question.get("type"),
        "golden_pass": not failures,                # citations scored strictly on the `citation_basis` set
        "golden_failures": failures,
        "golden_pass_lenient": lenient_passed,      # the earlier rule: raw substring / document-key match
        "citation_basis": basis,
        "citation_recall": basis_m["recall"],       # the basis set; both sets are reported below
        "citation_precision": basis_m["precision"],
        "citation_f1": basis_m["f1"],
        "n_distinct_cited": basis_m["n_distinct"],
        "citations_context": context_set,
        "citations_answer": answer_set,
        "citation_metrics": {"context": context_m, "answer": answer_m},
        # tag compliance of the answer: how many [DOC: id] ids it wrote and how many resolve to a canonical id
        "answer_tags": {"n_tags": len(tag_ids), "n_resolved": len(resolved_tags),
                        "valid": bool(resolved_tags), "refused": refused},
        "status": judged["status"],
        "faithfulness": judged["faithfulness"],
        "faithfulness_attempts": judged["attempts"],
        "judge_notes": judged["notes"],
        "latency_e2e_ms": round(elapsed_ms, 2),
        "latency_retrieval_ms": retrieval_latency_ms(result),
        "agentic_fallback": agentic_fallback(result),
        "retrieval_mode": result.retrieval_mode,
        "routing_reason": result.routing_reason,
        "n_contexts": len(result.contexts),
        "answer": result.answer,
        "answer_raw": raw_answer if raw_answer is not None else result.answer,
        "citations": result.citations,
        "error": None,
    }


def error_row(question: dict, exc: Exception, elapsed_ms: float) -> dict:
    return {
        "id": question["id"], "type": question.get("type"), "golden_pass": None,
        "golden_failures": [], "citation_recall": None, "status": "error",
        "faithfulness": None, "faithfulness_attempts": 0, "judge_notes": [],
        "latency_e2e_ms": round(elapsed_ms, 2), "latency_retrieval_ms": None,
        "agentic_fallback": {"fired": None, "reason": None, "completed_by": None},
        "retrieval_mode": None, "routing_reason": None, "n_contexts": 0,
        "answer": "", "citations": [], "error": f"{type(exc).__name__}: {str(exc)[:300]}",
    }


def tag_compliance(ok_rows: list[dict]) -> dict:
    """Answers with at least one valid [DOC: id] tag among the non-refused answers, and the share of tags that resolve."""
    tagged = [row["answer_tags"] for row in ok_rows if row.get("answer_tags")]
    answered = [tags for tags in tagged if not tags["refused"]]
    n_tags = sum(tags["n_tags"] for tags in tagged)
    return {
        "compliance": round(sum(tags["valid"] for tags in answered) / len(answered), 4) if answered else None,
        "n_nonrefused": len(answered),
        "n_tags": n_tags,
        "tag_resolution_rate": round(sum(tags["n_resolved"] for tags in tagged) / n_tags, 4) if n_tags else None,
    }


def summarise(rows: list[dict]) -> dict:
    ok = [row for row in rows if row["status"] != "error"]
    def pooled(which: str, key: str) -> float | None:
        values = [row["citation_metrics"][which][key] for row in ok
                  if row.get("citation_metrics") and row["citation_metrics"][which][key] is not None]
        return round(statistics.mean(values), 4) if values else None

    sets = {which: {"recall": pooled(which, "recall"), "precision": pooled(which, "precision"),
                    "f1_macro": pooled(which, "f1"),    # per question, then averaged
                    "mean_distinct_docs": pooled(which, "n_distinct")}
            for which in ("context", "answer")}
    basis_rows = [row["citation_metrics"]["answer" if row["citation_basis"] == "answer" else "context"]
                  for row in ok if row.get("citation_metrics")]

    def basis_mean(key: str) -> float | None:
        values = [m[key] for m in basis_rows if m[key] is not None]
        return round(statistics.mean(values), 4) if values else None

    faith = [row["faithfulness"] for row in rows if row["status"] == "scored"]
    e2e = [row["latency_e2e_ms"] for row in ok]
    retrieval = [row["latency_retrieval_ms"] for row in ok if row["latency_retrieval_ms"] is not None]

    def stats(values: list[float]) -> dict:
        return {
            "n": len(values),
            "mean": round(statistics.mean(values), 2) if values else None,
            "p50": percentile(values, 50), "p95": percentile(values, 95),
            "max": round(max(values), 2) if values else None,
        }

    return {
        "n_questions": len(rows),
        "n_errors": sum(row["status"] == "error" for row in rows),
        "golden_pass_rate": round(sum(bool(row["golden_pass"]) for row in ok) / len(ok), 4) if ok else None,
        "citation_recall": basis_mean("recall"),   # the golden basis set (context; answer for full_context)
        "citation_precision": basis_mean("precision"),
        "citation_f1_macro": basis_mean("f1"),
        "mean_distinct_cited": basis_mean("n_distinct"),
        "citations": {**sets, "answer": {**sets["answer"], **tag_compliance(ok)}},   # headline = ["answer"]
        "golden_pass_rate_lenient": (round(sum(bool(row["golden_pass_lenient"]) for row in ok) / len(ok), 4)
                                     if ok else None),
        "faithfulness_mean": round(statistics.mean(faith), 4) if faith else None,
        "n_scored": len(faith),
        "n_refused": sum(row["status"] == "refused" for row in rows),
        "n_unscorable": sum(row["status"] == "unscorable" for row in rows),
        "n_agentic_fallback": sum(bool(row["agentic_fallback"]["fired"]) for row in ok),
        "latency_e2e_ms": stats(e2e),
        "latency_retrieval_ms": stats(retrieval),
    }


def _models(evaluator) -> dict:
    from graphrag.core.llm_client import get_generation_route

    route = get_generation_route()
    judge = evaluator.judge_info if evaluator is not None else None
    return {
        "generator": route["primary"],
        "generator_fallback": route["fallback"],
        "router": route["agentic_fast"],
        "judge": judge,
        # A judge from the generator's own provider is a known bias risk: flag it in the file.
        "judge_same_provider_as_generator": bool(
            judge and judge.get("provider") and route["primary"].split(":")[0] == judge["provider"]),
    }


def _planned_models() -> dict:
    """Models a run would use, from settings only (no client is built)."""
    from graphrag.core.config import get_settings
    from graphrag.core.llm_client import get_generation_route

    cfg, route = get_settings(), get_generation_route()
    judge_models = {"deepseek": "deepseek-v4-flash", "groq": cfg.groq_model,
                    "gemini": cfg.gemini_ingest_model}
    provider = cfg.eval_judge_provider.lower()
    return {
        "generator": route["primary"], "generator_fallback": route["fallback"],
        "router": route["agentic_fast"],
        "judge": {"provider": provider, "model": judge_models.get(provider)},
    }


async def _run_profile(questions: list[dict], args, golden: dict, evaluator, models: dict,
                       profile: str, config_overrides: dict, pinned: bool) -> dict:
    from graphrag.core.config import get_settings, resolve_tenant_config

    rows: list[dict] = []
    extra = None
    is_full_context = profile == full_context.PROFILE
    corpus = await full_context.load_corpus(args.tenant)       # also gives the id list for the shared rule
    instruction = citations.citation_instruction(corpus.names)
    tenant_cfg = {**resolve_tenant_config(get_settings().retrieval, args.tenant),
                  "eval_citation_instruction": instruction}
    if is_full_context:
        extra = full_context.corpus_report(corpus, tenant_cfg, window=args.context_window)
        full_context.require_fit(extra)
        answer = lambda text: full_context.answer_full_context(  # noqa: E731
            text, corpus, tenant=args.tenant, citation_instruction=instruction)
    else:
        config_overrides = {**config_overrides, "eval_citation_instruction": instruction}
        retriever = HybridRetriever()
        answer = lambda text: retriever.retrieve_and_answer(  # noqa: E731
            text, mode=args.mode, tenant=args.tenant,
            retrieval_profile=profile, config_overrides=config_overrides)
    for question in questions:
        started = time.perf_counter()
        try:
            result = await answer(question["question"])
        except Exception as exc:  # recorded and counted; the run still exits non-zero
            rows.append(error_row(question, exc, (time.perf_counter() - started) * 1000))
            print(f"  [{profile}] {question['id']}: ERROR {type(exc).__name__}", file=sys.stderr)
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000
        raw_answer = result.answer
        result, answer_ids = split_answer(result)
        judged = await _judge(evaluator, question, result.answer, result.contexts) if evaluator else None
        rows.append(build_row(question, result, elapsed_ms, judged, answer_ids=answer_ids,
                              basis="answer" if is_full_context else "context", raw_answer=raw_answer))
    run = collect_run_metadata(
        tenant=args.tenant, profile=profile, profile_overrides=retrieval_profile_overrides(profile),
        mode=args.mode, config_overrides=config_overrides, models=models, golden=golden,
        pinned=pinned, extra=extra)
    run["citation"] = {
        "normaliser": citations.NORMALISER, "alias_map": "evals/citation_aliases.json",
        "alias_map_sha256": citations.alias_map_sha256(),
        "answer_prompt": "shared base prompt + the same [DOC: id] citation rule for every profile",
        "citation_instruction": instruction,
        "citation_instruction_sha256": citations.sha256_text(instruction),
        "answer_prompt_sha256": citations.sha256_text(answer_prompt(tenant_cfg)),
        "golden_citation_basis": "answer_citations" if is_full_context else "context_citations",
        "citation_sets": "context_citations (retriever provenance; none for full_context) and answer_citations "
                         "(ids parsed from the answer's [DOC: id] tags); headline comparison: answer_citations",
    }
    return {
        "run": run,
        "summary": summarise(rows),
        "results": rows,
    }


def output_path(output_dir: Path, report: dict) -> Path:
    run = report["run"]
    stamp = run["timestamp"].replace("-", "").replace(":", "").replace("+0000", "Z").split("+")[0]
    return output_dir / f"{run['tenant']}__{run['retrieval_profile']}__{stamp}.json"


async def _run_all(questions, args, golden, profiles, config_overrides, pinned) -> list[Path]:
    from graphrag.core.llm_client import get_fast_llm, get_llm

    # Fail loudly BEFORE any paid retrieval call: build every model client up front.
    get_llm(), get_fast_llm()
    evaluator = None
    if not args.skip_faithfulness:
        from graphrag.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator()
    models = _models(evaluator)
    written, failed = [], False
    # Serial on purpose: concurrent profiles would contend for provider rate limits
    # and bias latency.
    for profile in profiles:
        report = await _run_profile(questions, args, golden, evaluator, models, profile,
                                    config_overrides, pinned)
        path = output_path(args.output_dir, report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"profile": profile, "file": str(path), **report["summary"]}, indent=2))
        written.append(path)
        failed = failed or report["summary"]["n_errors"] > 0
    if failed:
        raise SystemExit("one or more questions errored: see the 'error' field in the result file(s)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--golden-set", type=Path, default=None,
                        help="default: evals/golden_set.json when it targets the tenant (aerospace, v2.3)")
    parser.add_argument("--profiles", nargs="+", default=["vector_only", "text_hybrid", "full"])
    parser.add_argument("--mode", default="local", choices=["local", "hybrid", "global"])
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-agentic-fallback", action="store_true",
                        help="set agentic_fallback=false, so the profile is measured without the IRCoT fallback")
    parser.add_argument("--skip-faithfulness", action="store_true", help="skip the RAGAS judge (cheaper)")
    parser.add_argument("--unpinned", action="store_true",
                        help="DEV ONLY: keep the production fallback chain; results are not comparable")
    parser.add_argument("--context-window", type=int, default=full_context.DEFAULT_CONTEXT_WINDOW,
                        help="generator context window in tokens, for the full_context fit check")
    parser.add_argument("--corpus-report", action="store_true",
                        help="read the tenant corpus (database read only, no model call) and print its size/fit")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no model or database call")
    args = parser.parse_args()

    # Stall diagnostics: every EVAL_STALL_DUMP_SECONDS (default 600) the stacks of all threads are
    # appended to a file, so a run that goes silent shows where it is waiting. Stdlib only.
    stall_s = int(os.environ.get("EVAL_STALL_DUMP_SECONDS", "600"))
    if stall_s > 0 and not args.dry_run:
        stall_dir = args.output_dir / "stalls"
        stall_dir.mkdir(parents=True, exist_ok=True)
        stall_file = open(stall_dir / f"stacks_{'-'.join(args.profiles)}_{time.strftime('%Y%m%dT%H%M%S')}.txt", "a")
        faulthandler.dump_traceback_later(stall_s, repeat=True, file=stall_file)

    # Evaluation mode must be set before the settings object is first built.
    os.environ["EVAL_PINNED_MODELS"] = "0" if args.unpinned else os.environ.get("EVAL_PINNED_MODELS", "1")
    from graphrag.core.config import get_settings
    get_settings.cache_clear()
    pinned = get_settings().eval_pinned_models is True
    if not pinned:
        print("WARNING: models are not pinned; results are not comparable across runs", file=sys.stderr)

    golden_path = args.golden_set or default_golden_set(args.tenant)
    payload = json.loads(golden_path.read_text(encoding="utf-8"))
    if args.golden_set is None and payload.get("tenant") not in (None, args.tenant):
        sys.exit(f"{golden_path} targets tenant {payload.get('tenant')!r}, not {args.tenant!r}")
    questions = payload["questions"]
    if args.ids:
        questions = [question for question in questions if question["id"] in set(args.ids)]
    for profile in args.profiles:
        retrieval_profile_overrides(profile)
    if args.corpus_report:
        async def _report() -> dict:
            from graphrag.core.config import get_settings, resolve_tenant_config
            corpus = await full_context.load_corpus(args.tenant)
            return full_context.corpus_report(
                corpus, resolve_tenant_config(get_settings().retrieval, args.tenant), window=args.context_window)
        print(json.dumps(asyncio.run(_report()), indent=2))
        return
    golden = golden_set_info(golden_path, payload, len(questions))
    pricing_window = deepseek_pricing_window()
    if pricing_window["peak_now"] and not args.skip_faithfulness and not args.dry_run:
        print(
            f"WARNING: DeepSeek peak hours ({pricing_window['peak_multiplier']:g}x price). Now "
            f"{pricing_window['now_bucharest']}; today's peak windows in Bucharest time: "
            f"{', '.join(pricing_window['peak_windows_bucharest'])} (Mon-Fri). "
            "Cheaper outside them, or use --skip-faithfulness.", file=sys.stderr)
    config_overrides = {"agentic_fallback": False} if args.no_agentic_fallback else {}

    if args.dry_run:
        print(json.dumps({
            "pinned_models": pinned, "models": _planned_models(), "golden_set": golden,
            "config_overrides": config_overrides, "skip_faithfulness": args.skip_faithfulness,
            "deepseek_pricing_window": pricing_window,
            "profiles": {profile: retrieval_profile_overrides(profile) for profile in args.profiles},
            "output_dir": str(args.output_dir), "cache": "disabled (no query ID)",
        }, indent=2))
        return

    asyncio.run(_run_all(questions, args, golden, args.profiles, config_overrides, pinned))


if __name__ == "__main__":
    main()
