import argparse
import os
from engine_loader import get_engines_by_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", nargs="*", default=[])
    parser.add_argument("--scenarios", nargs="*", default=["concurrency", "metadata", "fragmentation", "frontier", "reranker"])
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    os.environ["RAG_STRESS_TEST_MODE"] = args.mode

    from scenarios.concurrency_burst import run_scenario as run_concurrency
    from scenarios.metadata_filter_penalty import run_scenario as run_metadata
    from scenarios.index_fragmentation import run_scenario as run_fragmentation
    from scenarios.recall_latency_frontier import run_scenario as run_frontier
    from scenarios.reranker_cost import run_scenario as run_reranker

    engines = get_engines_by_names(args.engines)
    if not engines:
        print("No engines resolved.")
        return

    if "concurrency" in args.scenarios: run_concurrency(engines)
    if "metadata" in args.scenarios: run_metadata(engines)
    if "fragmentation" in args.scenarios: run_fragmentation(engines)
    if "frontier" in args.scenarios: run_frontier(engines)
    if "reranker" in args.scenarios: run_reranker(engines)


if __name__ == "__main__":
    main()