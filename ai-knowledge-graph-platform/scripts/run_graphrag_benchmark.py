"""Run controlled GraphRAG-Benchmark routes against one tenant corpus.

Example:
  python scripts/run_graphrag_benchmark.py --questions data/questions.jsonl \
    --tenant aerospace --route full:full --route vector:vector_only
"""

from __future__ import annotations

import argparse
import asyncio

from graphrag.evaluation.graphrag_benchmark import (
    ControlledRoute, load_questions, run_controlled_routes, write_report,
)
from graphrag.retrieval.hybrid_retriever import HybridRetriever


def _route(value: str) -> ControlledRoute:
    name, separator, profile = value.partition(":")
    if not name or not separator or not profile:
        raise argparse.ArgumentTypeError("route must be NAME:PROFILE")
    return ControlledRoute(name=name, profile=profile)


async def _run(args) -> None:
    retriever = HybridRetriever()

    async def query(question: str, mode: str, overrides: dict) -> dict:
        return (await retriever.retrieve_and_answer(
            question, mode=mode, tenant=args.tenant, config_overrides=overrides,
        )).model_dump(mode="json")

    report = await run_controlled_routes(
        load_questions(args.questions), args.route, query, tenant=args.tenant,
    )
    write_report(report, args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--route", action="append", type=_route, required=True)
    parser.add_argument("--output", default="artifacts/graphrag-benchmark-report.json")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
