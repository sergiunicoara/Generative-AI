"""Evaluate an agent on the ontology actions it proposes (offline by default).

    python scripts/run_action_eval.py                       # oracle agent: checks the dataset itself
    python scripts/run_action_eval.py --agent llm --allow-paid   # calls the configured LLM (costs money)

Exit code 1 when any case fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.evaluation.action_eval import LLMActionAgent, evaluate_agent, load_cases, oracle_agent  # noqa: E402
from graphrag.semantic_model import load_model  # noqa: E402

DEFAULT_CASES = ROOT / "evals" / "actions" / "automotive_supplier_cases.json"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--agent", choices=("oracle", "llm"), default="oracle")
    ap.add_argument("--allow-paid", action="store_true", help="required for --agent llm")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    model = load_model(ROOT / payload["model"])
    cases = load_cases(args.cases)
    if args.agent == "llm":
        if not args.allow_paid:
            print("--agent llm calls a paid model; pass --allow-paid to confirm", file=sys.stderr)
            return 2
        from graphrag.core.llm_client import get_llm

        agent = LLMActionAgent(get_llm())
    else:
        agent = oracle_agent(cases)
    report = asyncio.run(evaluate_agent(agent, cases, model))
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("total", "passed", "pass_rate", "by_action", "failures")}, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
