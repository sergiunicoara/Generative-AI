"""Generate OWL and SHACL files from domain ontology YAMLs.

    python scripts/export_ontology_owl.py config/ontologies/legal_contracts.yml ontology/from_config
    python scripts/export_ontology_owl.py --all ontology/from_config

``--all`` skips ontologies that already have a compiled model under ``ontology/models`` (automotive, energy) and
``synthetic_large`` (a load-test fixture). Output is deterministic; rerun and commit when a YAML changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.graph.domain_ontology import load_domain_ontology  # noqa: E402
from graphrag.graph.ontology_owl_export import export_ontology  # noqa: E402

SKIP_WITH_ALL = {"automotive_iatf", "synthetic_large"}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    target, out_dir = argv[1], argv[2]
    sources = (sorted(p for p in (ROOT / "config" / "ontologies").glob("*.yml") if p.stem not in SKIP_WITH_ALL)
               if target == "--all" else [Path(target)])
    for source in sources:
        ontology = load_domain_ontology(source)
        if not ontology:
            print(f"skipped (unreadable): {source}")
            continue
        owl_path, shapes_path = export_ontology(ontology, out_dir)
        print(f"{source.name}: {owl_path.name}, {shapes_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
