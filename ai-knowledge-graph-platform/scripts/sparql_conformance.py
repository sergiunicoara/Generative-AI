"""Run the SPARQL competency suite against a SPARQL target, read-only.

With no target it runs the local rdflib baseline. With ``--endpoint`` or ``--vendor`` it queries a live store that
the operator has loaded with the Turtle files listed in the output; the dataset fingerprint printed first is the
sha256 of exactly those inputs plus cases.json (see graphrag.graph.sparql_conformance.dataset_fingerprint). Only
SELECT/ASK queries are sent; this script never loads data into a target. Exit code 1 on any mismatch, 2 on bad usage.

Credentials: ``--auth-env NAME`` names an environment variable holding ``user:password``. The value is never printed
or written to the JSON report.

Examples:
    python scripts/sparql_conformance.py
    python scripts/sparql_conformance.py --vendor stardog --base-url http://localhost:5820 --db kg \
        --domain automotive --level instance --auth-env STARDOG_AUTH --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from graphrag.graph.sparql_conformance import (  # noqa: E402
    ConformanceReport,
    load_suite,
    run_conformance,
    run_local_baseline,
)
from graphrag.graph.triplestore import (  # noqa: E402
    RemoteSPARQLEndpoint,
    TripleStoreTarget,
    _auth_from_env,
)

# Vendor -> the TripleStoreTarget keyword that names its database/repository/dataset.
_VENDOR_NAME_KWARG = {"stardog": "database", "graphdb": "repository", "fuseki": "dataset"}


class UsageError(ValueError):
    pass


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--endpoint", help="full SPARQL 1.1 query URL of the target")
    parser.add_argument("--vendor", choices=sorted(_VENDOR_NAME_KWARG), help="build the query URL for this vendor")
    parser.add_argument("--base-url", help="vendor server base URL (with --vendor)")
    parser.add_argument("--db", help="database / repository / dataset name (with --vendor)")
    parser.add_argument("--auth-env", help="name of an env var holding user:password")
    parser.add_argument("--domain", choices=["automotive", "energy"])
    parser.add_argument("--level", choices=["schema", "instance"])
    parser.add_argument("--json", dest="json_out", help="write the report to this path")
    return parser.parse_args(argv)


def _auth(env_name: str | None) -> tuple[str, str] | None:
    if not env_name:
        return None
    raw = os.environ.get(env_name, "")
    auth = _auth_from_env(raw)
    if auth is None:
        raise UsageError(f"environment variable {env_name} is unset or not in user:password form")
    return auth


def _endpoint_label(url: str) -> str:
    parts = urlsplit(url)
    return f"endpoint:{parts.hostname}{parts.path}"  # never the userinfo or query string


def _build_target(args: argparse.Namespace):
    """Return (source, label), or (None, None) for the local baseline."""
    if args.endpoint and args.vendor:
        raise UsageError("use either --endpoint or --vendor, not both")
    if not args.endpoint and not args.vendor:
        return None, None
    # One remote store holds one dataset, so the operator must say which one it was loaded with.
    if not (args.domain and args.level):
        raise UsageError("a remote target needs --domain and --level naming the dataset it holds")
    auth = _auth(args.auth_env)
    if args.endpoint:
        return RemoteSPARQLEndpoint(args.endpoint, auth=auth), _endpoint_label(args.endpoint)
    if not (args.base_url and args.db):
        raise UsageError("--vendor needs --base-url and --db")
    target = TripleStoreTarget(args.vendor, args.base_url, auth=auth, **{_VENDOR_NAME_KWARG[args.vendor]: args.db})
    return target, f"{args.vendor}:{args.db}"


def _print_report(report: ConformanceReport) -> None:
    print(f"target: {report.target}")
    print(f"dataset fingerprint (sha256): {report.dataset_fingerprint}")
    print("the target must hold exactly these files:")
    for line in report.dataset_files:
        print(f"  {line}")
    for case in report.cases:
        print(f"{'PASS' if case.passed else 'FAIL'} {case.case_id}")
        if not case.passed:
            detail = case.error or f"expected {case.expected} got {case.actual}"
            print(f"  - {detail}")
    print(f"{report.passed}/{report.total} passed")


async def _run(args: argparse.Namespace) -> ConformanceReport:
    suite = load_suite()
    source, label = _build_target(args)
    if source is None:
        return await run_local_baseline(suite, domain=args.domain, level=args.level)
    return await run_conformance(source, suite=suite, domain=args.domain, level=args.level, target=label)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = asyncio.run(_run(args))
    except ValueError as exc:  # UsageError and URL-safety rejections
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_report(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    return int(report.failed > 0 or report.total == 0)


if __name__ == "__main__":
    raise SystemExit(main())
