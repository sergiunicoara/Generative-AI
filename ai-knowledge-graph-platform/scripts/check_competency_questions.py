"""Static regression gate for competency questions (roadmap "P0 --
competency questions and model-impact testing", bullet 4).

For every domain in graphrag.graph.competency_gate.competency_questions_by_domain()
that has a config/ontologies/*.yml (aerospace, automotive, marketing --
Energy's schema drift is separately gated by `make semantic-model-check`),
this verifies every relation/type name each domain's competency-question
Cypher literally references still exists in that domain's *current*
ontology YAML. It catches "someone renamed or removed a type/relation a
competency question still expects" the moment the YAML changes, from static
text alone -- no live Neo4j required. It does not prove the Cypher returns
real rows against a real graph; see
graphrag/graph/competency_gate.py's module docstring for that evidence-level
distinction, and run_competency_suite_async against a real Neo4jClient for
the live-execution half this script does not cover.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from graphrag.graph.competency_gate import (  # noqa: E402
    check_competency_questions_against_ontology,
    domains_with_static_ontology_checks,
)


def main() -> int:
    failed = False
    for domain_id in domains_with_static_ontology_checks():
        problems = check_competency_questions_against_ontology(domain_id)
        status = "OK" if not problems else "FAIL"
        print(f"{status} {domain_id}")
        if problems:
            failed = True
            for problem in problems:
                print(f"  - {problem}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
