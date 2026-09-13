"""CLI for deterministic semantic-model compilation and drift checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from graphrag.semantic_model.compiler import compile_model, compile_to_disk
from graphrag.semantic_model.models import SemanticModelError, load_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m graphrag.semantic_model")
    sub = parser.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile", help="compile a canonical model")
    compile_parser.add_argument("model", type=Path)
    compile_parser.add_argument("--output-dir", type=Path)
    compile_parser.add_argument("--check", action="store_true")
    compile_parser.add_argument(
        "--fail-on-unenforceable", action="store_true",
        help="fail before writing if any target rule requires runtime enforcement",
    )
    args = parser.parse_args(argv)
    try:
        paths = compile_to_disk(
            args.model, output_dir=args.output_dir, check=args.check,
            fail_on_unenforceable=args.fail_on_unenforceable,
        )
        diagnostics = compile_model(load_model(args.model)).diagnostics
    except SemanticModelError as exc:
        print(f"semantic-model: {exc}", file=sys.stderr)
        return 1
    action = "verified" if args.check else "generated"
    print(f"semantic-model: {action} {len(paths)} artifacts")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    for item in diagnostics:
        location = f"{item.model_path}:{item.line}" if item.line else item.model_path
        print(
            f"  [{item.severity}] {item.target.value}/{item.code} "
            f"{item.element} ({location}): {item.message}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
