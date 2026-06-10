"""
Full 40-question faithfulness eval — Groq as primary RAGAS judge.

Run this after Groq daily quota resets (midnight UTC).
Results written to evals/faithfulness_eval_results.json.
"""
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

# Fail fast on a wrong interpreter (e.g. another project's venv on PATH).
# Without this, retrieval works but every RAGAS judge call errors out and a
# 15-minute run produces a results file full of errors instead of scores.
try:
    import ragas  # noqa: F401
    import langchain_community  # noqa: F401
except ImportError as _exc:
    sys.exit(
        f"Missing eval dependency ({_exc.name}) — wrong Python interpreter?\n"
        f"  running : {sys.executable}\n"
        f"  expected: the project's Python 3.11 with requirements installed"
    )

_REFUSAL = (
    "does not contain",
    "no information",
    "not specify",
    "cannot find",
    "not mentioned",
    "sufficient information",
    "not available",
    "no details",
)


async def main():
    from graphrag.retrieval.hybrid_retriever import HybridRetriever
    from graphrag.evaluation.ragas_evaluator import RagasEvaluator
    from graphrag.graph.confidence_calibration import CalibrationService
    from graphrag.graph.neo4j_client import get_neo4j

    golden = json.loads((Path(__file__).parents[1] / "evals" / "golden_set.json").read_text())
    questions = golden["questions"]

    retriever = HybridRetriever()
    evaluator = RagasEvaluator()
    neo4j = get_neo4j()
    cal_svc = CalibrationService(neo4j)
    tenant = "aerospace"

    scores, refusals, errors = [], 0, 0
    results = []
    cal_samples = 0

    print(f"\n{'='*70}")
    print(f"  GraphRAG Faithfulness Eval — {len(questions)} questions")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*70}\n")

    for q in questions:
        t0 = time.monotonic()
        try:
            res = await retriever.retrieve_and_answer(question=q["question"], tenant="aerospace")
            elapsed = time.monotonic() - t0
            ans = res.answer.lower()

            if not res.contexts or any(s in ans for s in _REFUSAL):
                refusals += 1
                print(f"  [{q['id']:8s}] REFUSAL  ({elapsed:.1f}s)  {res.answer[:60]!r}")
                results.append({"id": q["id"], "type": q["type"], "status": "refusal",
                                 "answer": res.answer, "latency": round(elapsed, 1)})
                continue

            er = await evaluator.evaluate_single(
                q["id"], q["question"], res.answer, res.contexts, ""
            )
            f = er.faithfulness
            scores.append(f)
            # Append result before print — print errors (e.g. Windows encoding) must not discard score
            results.append({"id": q["id"], "type": q["type"], "status": "scored",
                             "faithfulness": round(f, 4), "answer": res.answer,
                             "latency": round(elapsed, 1)})
            flag = "  LOW" if (not isinstance(f, float) or f < 0.8) else ""
            print(f"  [{q['id']:8s}] faith={f:.3f}  ({elapsed:.1f}s)  {res.answer[:60]!r}{flag}")

            # ── Calibration sample: predicted = context_precision, actual = faithfulness
            try:
                await cal_svc.add_sample(
                    predicted_confidence=er.context_precision,
                    actual_outcome=f,
                    relation=q.get("type", ""),
                    source_doc_id=q["id"],
                    prompt_version="run_faithfulness_eval",
                    tenant=tenant,
                    verified_by="ragas",
                )
                cal_samples += 1
            except Exception as _cal_exc:
                pass  # calibration failure must not abort the eval

        except Exception as e:
            errors += 1
            print(f"  [{q['id']:8s}] ERROR: {e}")
            results.append({"id": q["id"], "type": q["type"], "status": "error", "error": str(e)})

    # Summary
    avg = sum(scores) / len(scores) if scores else 0.0
    by_type: dict = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    print(f"\n{'='*70}")
    print(f"  Faithfulness ({len(scores)} answerable): {avg:.3f}")
    print(f"  Baseline: 0.840   Delta: {avg - 0.840:+.3f}")
    print(f"  Refusals (correct, excluded): {refusals}/{len(questions)}")
    if errors:
        print(f"  Errors: {errors}")
    print(f"\n  By question type:")
    for qtype, rows in sorted(by_type.items()):
        type_scores = [r["faithfulness"] for r in rows if r.get("status") == "scored"]
        type_refusals = sum(1 for r in rows if r.get("status") == "refusal")
        if type_scores:
            print(f"    {qtype:20s}  faith={sum(type_scores)/len(type_scores):.3f}  "
                  f"scored={len(type_scores)}  refusals={type_refusals}")
    print(f"{'='*70}\n")

    # Write results
    out = Path(__file__).parents[1] / "evals" / "faithfulness_eval_results.json"
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "faithfulness_answerable": round(avg, 4),
        "baseline": 0.840,
        "delta": round(avg - 0.840, 4),
        "n_scored": len(scores),
        "n_refusals": refusals,
        "n_errors": errors,
        "n_total": len(questions),
        "by_type": {
            qtype: {
                "faithfulness": round(
                    sum(r["faithfulness"] for r in rows if r.get("status") == "scored") /
                    max(1, sum(1 for r in rows if r.get("status") == "scored")), 4
                ),
                "scored": sum(1 for r in rows if r.get("status") == "scored"),
                "refusals": sum(1 for r in rows if r.get("status") == "refusal"),
            }
            for qtype, rows in by_type.items()
        },
        "questions": results,
    }, indent=2))
    print(f"  Results written to: {out}")

    # ── Calibration snapshot: persist aggregate metrics for the dashboard trend chart
    if cal_samples > 0:
        try:
            snap_id = await cal_svc.persist_snapshot(
                tenant=tenant,
                label=f"faithfulness-eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}",
            )
            print(f"  Calibration snapshot written ({cal_samples} samples): {snap_id[:8]}...")
        except Exception as _snap_exc:
            print(f"  Warning: calibration snapshot failed: {_snap_exc}")
    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(main())
