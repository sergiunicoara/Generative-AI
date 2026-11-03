"""Read-only conformance of a SPARQL target against the competency-question suite.

The suite in evals/competency/cases.json is the contract: each case is a SELECT projecting ``?v`` with an exact
expected set. A *target* is anything satisfying ``SPARQLSource`` -- the in-process rdflib baseline below, or a
``RemoteSPARQLEndpoint`` / ``TripleStoreTarget`` pointed at Stardog, GraphDB or Fuseki. The platform's Turtle files
stay the source of truth; a target is an interoperability check that it loaded the same data and answers the same
questions the same way. It never writes: queries go through the shared read-only guard and nothing here issues an
update or loads data into a target.

Evidence level: the offline tests run the remote adapter against a MockTransport that executes the query on a local
graph. That shows the adapter and the verdict logic, not that any vendor's engine returns the same rows. A real
store has to be loaded with the files named in the report and run through scripts/sparql_conformance.py.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rdflib import Graph

from graphrag.graph.sparql_bridge import SPARQLBridge
from graphrag.graph.triplestore import SPARQLSource

ROOT = Path(__file__).resolve().parents[2]
CASES_FILE = Path("evals") / "competency" / "cases.json"
ONTOLOGY_FILES = {
    "automotive": Path("ontology") / "automotive" / "automotive-iatf-quality.ttl",
    "energy": Path("ontology") / "energy" / "energy-asset-intelligence.ttl",
}
INSTANCE_FILE = Path("evals") / "competency" / "automotive_iatf_instances.ttl"

# A remote store resolves prefixes only from the query text (RemoteSPARQLEndpoint ignores init_ns), while the local
# bridge pre-binds these. Declaring them in the text makes both paths run byte-identical queries.
_STANDARD_PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


def load_suite(root: Path = ROOT) -> dict:
    return json.loads((root / CASES_FILE).read_text(encoding="utf-8"))


def dataset_files(domain: str, level: str) -> list[Path]:
    """Repo-relative Turtle files that make up the (domain, level) dataset, in load order."""
    files = [ONTOLOGY_FILES[domain]]
    if level == "instance":
        files.append(INSTANCE_FILE)
    return files


def build_graph(domain: str, level: str, root: Path = ROOT) -> Graph:
    g = Graph()
    for rel in dataset_files(domain, level):
        g.parse(str(root / rel), format="turtle")
    return g


def _selected_cases(suite: dict, domain: str | None, level: str | None) -> list[dict]:
    return [
        c for c in suite["cases"]
        if (domain is None or c["domain"] == domain) and (level is None or c["level"] == level)
    ]


def _datasets_of(cases: list[dict]) -> list[tuple[str, str]]:
    return sorted({(c["domain"], c["level"]) for c in cases})


def _feed(digest, label: str, data: bytes) -> None:
    # Length-framed so two different (label, data) splits can never concatenate to the same byte stream.
    digest.update(f"{label}\0{len(data)}\0".encode("utf-8"))
    digest.update(data)


def _normalised(path: Path) -> bytes:
    # CRLF checkouts must fingerprint like LF ones.
    return path.read_bytes().replace(b"\r\n", b"\n")


def dataset_fingerprint(
    *, domain: str | None = None, level: str | None = None, root: Path = ROOT,
) -> str:
    """sha256 an operator can compare with the data they loaded into a target.

    Canonical input, in order: the bytes of cases.json (``cases.json``), then for every distinct (domain, level)
    among the selected cases, sorted by (domain, level), each of that dataset's Turtle files in load order (the
    ontology, then the instance file at instance level). Each part is framed as ``<repo-relative path>``,
    ``<dataset>``, length, bytes, with CRLF folded to LF. ``domain``/``level`` narrow the selection exactly as they
    narrow the cases; with neither, every dataset in the suite is covered.
    """
    suite = load_suite(root)
    digest = hashlib.sha256()
    _feed(digest, CASES_FILE.as_posix(), _normalised(root / CASES_FILE))
    for dom, lvl in _datasets_of(_selected_cases(suite, domain, level)):
        for rel in dataset_files(dom, lvl):
            _feed(digest, f"{dom}/{lvl}/{rel.as_posix()}", _normalised(root / rel))
    return digest.hexdigest()


class LocalSPARQLSource:
    """The baseline target: an rdflib graph queried in-process, shaped like a remote ``SPARQLSource``."""

    def __init__(self, graph: Graph) -> None:
        self._bridge = SPARQLBridge(graph)

    async def query(self, sparql: str, init_ns: dict[str, str] | None = None) -> list[dict]:
        return await self._bridge.aquery(sparql, init_ns=init_ns)


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    domain: str
    level: str
    expected: list[str]
    actual: list[str]
    passed: bool
    error: str | None = None


@dataclass(frozen=True)
class ConformanceReport:
    target: str
    dataset_fingerprint: str
    # Which (domain, level) datasets the target is expected to hold for these cases; one entry per dataset.
    expected_datasets: list[dict[str, str]]
    dataset_files: list[str]
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "dataset_fingerprint": self.dataset_fingerprint,
            "expected_datasets": self.expected_datasets,
            "dataset_files": self.dataset_files,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "cases": [asdict(c) for c in self.cases],
        }


def _with_prefixes(sparql: str, suite_prefixes: dict[str, str]) -> str:
    declared = {**_STANDARD_PREFIXES, **suite_prefixes}
    header = "".join(f"PREFIX {name}: <{iri}>\n" for name, iri in declared.items())
    return header + sparql


async def _run_case(source: SPARQLSource, case: dict, prefixes: dict[str, str]) -> CaseResult:
    expected = sorted(case["expected"])
    meta = (case["id"], case["domain"], case["level"], expected)
    try:
        rows = await source.query(_with_prefixes(case["sparql"], prefixes))
    except ValueError as exc:
        # The read-only guard and the remote adapter both raise ValueError; one bad case must not hide the rest.
        return CaseResult(*meta, actual=[], passed=False, error=str(exc))
    # A remote store omits an unbound ?v where the local bridge yields "", so both are dropped.
    actual = sorted(row["v"] for row in rows if row.get("v"))
    return CaseResult(*meta, actual=actual, passed=actual == expected)


def _report(
    target: str, cases: list[dict], results: list[CaseResult],
    domain: str | None, level: str | None, root: Path,
) -> ConformanceReport:
    datasets = _datasets_of(cases)
    return ConformanceReport(
        target=target,
        dataset_fingerprint=dataset_fingerprint(domain=domain, level=level, root=root),
        expected_datasets=[{"domain": d, "level": lv} for d, lv in datasets],
        dataset_files=[f"{d}/{lv}: {rel.as_posix()}" for d, lv in datasets for rel in dataset_files(d, lv)],
        cases=results,
    )


async def run_conformance(
    source: SPARQLSource,
    *,
    suite: dict,
    domain: str | None = None,
    level: str | None = None,
    target: str = "unnamed",
    root: Path = ROOT,
) -> ConformanceReport:
    """Run the selected cases against ``source`` and compare each answer with its asserted set.

    ``source`` is expected to hold the (domain, level) dataset(s) named in the report; running cases against a
    target holding a different dataset is reported as failures, not hidden.
    """
    cases = _selected_cases(suite, domain, level)
    results = [await _run_case(source, c, suite["prefixes"]) for c in cases]
    return _report(target, cases, results, domain, level, root)


async def run_local_baseline(
    suite: dict, *, domain: str | None = None, level: str | None = None, root: Path = ROOT,
) -> ConformanceReport:
    """The rdflib baseline over every selected dataset, folded into one report.

    One in-process graph holds one dataset, so each (domain, level) is run against its own graph.
    """
    cases = _selected_cases(suite, domain, level)
    results: list[CaseResult] = []
    for dom, lvl in _datasets_of(cases):
        source = LocalSPARQLSource(build_graph(dom, lvl, root))
        for case in _selected_cases(suite, dom, lvl):
            results.append(await _run_case(source, case, suite["prefixes"]))
    return _report("local-rdflib", cases, results, domain, level, root)
