"""
Score real retrieval rankings against evals/golden_set.json's declared
thresholds (min_context_precision, min_citation_recall) using classical IR
metrics (precision@k, recall@k, MRR, MAP) -- see
graphrag/evaluation/retrieval_quality_eval.py for why this exists: those two
thresholds are declared in golden_set.json but never enforced by
scripts/run_golden_eval.py, the script the CONTRIBUTING.md eval gate runs.

Calls LocalSearch.search() in-process (no LLM synthesis needed -- this scores
ranking quality, not answer quality) with the same "text_hybrid" profile
POST /search uses, so the ranking measured here is the one a real /search
caller would see. Requires a local Neo4j with the target tenant's corpus
loaded -- the same prerequisite scripts/run_faithfulness_eval.py already has;
this script does not add any new service dependency.

Usage:
    python scripts/run_retrieval_quality_eval.py --tenant aerospace
    python scripts/run_retrieval_quality_eval.py --golden-set evals/golden_set.json --k 10

Writes evals/retrieval_quality_last_run.json. Exit code 1 if the aggregate
means fall below either declared threshold.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

# Imported after the sys.path insert above, matching run_golden_eval.py's own
# convention (see that file's comment on this exact pattern).
from graphrag.evaluation.retrieval_quality_eval import (  # noqa: E402
    RetrievalQualityReport,
    evaluate_retrieval_quality,
)
from graphrag.retrieval.hybrid_retriever import retrieval_profile_overrides  # noqa: E402
from graphrag.retrieval.local_search import LocalSearch  # noqa: E402

GOLDEN_SET = Path(__file__).parents[1] / "evals" / "golden_set.json"
OUTPUT = Path(__file__).parents[1] / "evals" / "retrieval_quality_last_run.json"

# Matches api/routes/search.py's fixed ACL-safe profile -- see that module's
# docstring for why /search cannot let a caller pick a different one.
_PROFILE = "text_hybrid"


def _rank_key(chunk: dict) -> float:
    # Identical fallback chain to api/routes/search.py's _rank_key -- kept as
    # a separate copy rather than importing a private helper across modules,
    # matching this repo's existing convention (run_golden_eval.py duplicates
    # small helpers rather than reaching into another script's private names).
    return chunk.get("final_score", chunk.get("rerank_score", chunk.get("score", 0.0)))


def _report_to_dict(report: RetrievalQualityReport) -> dict:
    return {
        "passed": report.passed,
        "mean_precision": report.mean_precision,
        "mean_recall": report.mean_recall,
        "mean_reciprocal_rank": report.mean_reciprocal_rank,
        "mean_average_precision": report.mean_average_precision,
        "min_context_precision_threshold": report.min_context_precision_threshold,
        "min_citation_recall_threshold": report.min_citation_recall_threshold,
        "failing_questions": report.failing_questions,
        "skipped_ids": report.skipped_ids,
        "questions": [
            {
                "id": s.id,
                "precision_at_k": s.precision_at_k,
                "recall_at_k": s.recall_at_k,
                "reciprocal_rank": s.reciprocal_rank,
                "average_precision": s.average_precision,
            }
            for s in report.scored
        ],
    }


async def _run(golden_set_path: Path, tenant: str | None, k: int) -> int:
    gs = json.loads(golden_set_path.read_text(encoding="utf-8"))
    questions = gs["questions"]
    thresholds = gs.get("thresholds", {})
    resolved_tenant = tenant or gs.get("tenant", "default")

    searcher = LocalSearch()
    overrides = retrieval_profile_overrides(_PROFILE)

    async def retrieve(question: str) -> list[str]:
        result = await searcher.search(
            question,
            tenant=resolved_tenant,
            config_overrides=overrides,
        )
        chunks = sorted(result.get("chunks", []), key=_rank_key, reverse=True)
        return [c.get("_doc_name") or c.get("source") or c["chunk_id"] for c in chunks]

    report = await evaluate_retrieval_quality(questions, retrieve, thresholds, k=k)

    OUTPUT.write_text(json.dumps(_report_to_dict(report), indent=2), encoding="utf-8")
    print(
        f"mean_precision={report.mean_precision:.3f} "
        f"(threshold {report.min_context_precision_threshold}), "
        f"mean_recall={report.mean_recall:.3f} "
        f"(threshold {report.min_citation_recall_threshold}), "
        f"mean_mrr={report.mean_reciprocal_rank:.3f}, "
        f"scored={len(report.scored)}, skipped={len(report.skipped_ids)}"
    )
    if report.failing_questions:
        print(f"failing: {report.failing_questions}")
    print(f"written to {OUTPUT}")

    return 0 if report.passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", default=None, help=f"Default: {GOLDEN_SET}")
    parser.add_argument("--tenant", default=None, help="Tenant override (default: from golden set)")
    parser.add_argument("--k", type=int, default=10, help="Rank depth for precision@k/recall@k (default: 10)")
    args = parser.parse_args()

    golden_set_path = Path(args.golden_set) if args.golden_set else GOLDEN_SET
    exit_code = asyncio.run(_run(golden_set_path, args.tenant, args.k))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
