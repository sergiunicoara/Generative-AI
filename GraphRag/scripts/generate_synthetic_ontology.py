"""Generate a synthetic domain ontology YAML for scale benchmarking.

Creates a realistic tree-structured YAML in the same format as
``config/ontologies/aerospace_regulatory.yml`` so ``load_domain_ontology()``
can parse it without modifications.  Used to demonstrate that the platform
handles ontologies far larger than the 28-type aerospace sample.

Usage::

    # Generate 500-type ontology and run a load benchmark
    python scripts/generate_synthetic_ontology.py --types 500 --relations 50 \\
        --output config/ontologies/synthetic_large.yml --benchmark

    # Minimal silent generation (no benchmark)
    python scripts/generate_synthetic_ontology.py --types 100 \\
        --output /tmp/test_ontology.yml

Benchmark output example::

    Benchmark results:
      Types loaded:           500
      Relations registered:   50
      YAML parse time:        3.21 ms
      Registry wire time:     1.47 ms
      Throughput:             34013 relations/sec
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import yaml


# ── Type hierarchy generation ─────────────────────────────────────────────────

def generate_type_hierarchy(
    n_types: int,
    branching: int = 5,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """BFS tree of n_types entity types rooted at CONCEPT.

    All generated types are subtypes of CONCEPT (already in the base
    taxonomy), ensuring they integrate with the existing type system.

    Names follow ``DOMAIN_TYPE_NNNN`` to avoid collisions with built-in
    types while remaining readable in log output.

    Parameters
    ----------
    n_types :
        Total number of types to generate (exclusive of root CONCEPT).
    branching :
        Maximum children per node; actual child count is random in [1, branching].
    seed :
        Random seed for reproducibility.

    Returns
    -------
    list[tuple[str, str]]
        ``[(child_type, parent_type)]`` pairs.
    """
    rng    = random.Random(seed)
    root   = "CONCEPT"
    queue  = [root]
    pairs: list[tuple[str, str]] = []
    counter = 0

    while queue and counter < n_types:
        parent = queue.pop(0)
        n_children = rng.randint(1, branching)
        for _ in range(n_children):
            if counter >= n_types:
                break
            child = f"DOMAIN_TYPE_{counter:04d}"
            pairs.append((child, parent))
            queue.append(child)
            counter += 1

    return pairs


# ── Relation rule generation ───────────────────────────────────────────────────

_RELATION_VERBS = [
    "RELATES_TO", "DEPENDS_ON", "PART_OF", "CONTAINS", "PRECEDES",
    "FOLLOWS", "DERIVED_FROM", "ASSOCIATED_WITH", "GOVERNS", "SUPERSEDES",
    "REFERENCES", "VALIDATES", "CONFLICTS_WITH", "SUPPORTS", "REQUIRES",
    "CLASSIFIES", "GENERATES", "MODIFIES", "EXTENDS", "IMPLEMENTS",
]


def generate_relation_rules(
    types: list[str],
    n_relations: int = 20,
    seed: int = 42,
) -> dict[str, dict]:
    """Generate n_relations relation rules with random domain/range constraints.

    Parameters
    ----------
    types :
        Pool of type names to sample domain and target from.
    n_relations :
        Number of distinct relations to generate.
    seed :
        Random seed.

    Returns
    -------
    dict[str, dict]
        ``{RELATION_NAME: {"domain": [...], "target": [...]}}``
    """
    rng   = random.Random(seed)
    rules: dict[str, dict] = {}

    for i in range(n_relations):
        base  = _RELATION_VERBS[i % len(_RELATION_VERBS)]
        suffix = f"_{i // len(_RELATION_VERBS)}" if i >= len(_RELATION_VERBS) else ""
        rel   = base + suffix

        n_dom = rng.randint(1, min(3, len(types)))
        n_tgt = rng.randint(1, min(3, len(types)))
        rules[rel] = {
            "domain": rng.sample(types, n_dom),
            "target": rng.sample(types, n_tgt),
        }

    return rules


# ── Inference rule generation ──────────────────────────────────────────────────

def generate_inference_rules(
    relations: list[str],
    n_rules: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Generate transitivity, inverse, and composition rules for a sample of relations.

    Parameters
    ----------
    relations :
        Pool of relation names.
    n_rules :
        Number of inference rules to generate (capped at len(relations)).
    seed :
        Random seed.

    Returns
    -------
    list[dict]
        Each dict has keys compatible with ``build_inference_rules_from_ontology()``:
        ``name``, ``rule_type``, ``relation``, ``max_depth``,
        ``confidence_decay``, and optional ``inverse_relation`` /
        ``via_relation`` / ``inferred_relation``.
    """
    rng        = random.Random(seed)
    rule_types = ["transitivity", "inverse", "composition"]
    selected   = rng.sample(relations, min(n_rules, len(relations)))
    rules: list[dict] = []

    for i, rel in enumerate(selected):
        rtype  = rule_types[i % len(rule_types)]
        others = [r for r in relations if r != rel]
        rule: dict = {
            "name":             f"synthetic_{rel.lower()}_rule",
            "rule_type":        rtype,
            "relation":         rel,
            "max_depth":        3,
            "confidence_decay": 0.9,
        }
        if rtype == "inverse":
            rule["inverse_relation"] = rng.choice(others) if others else rel
        elif rtype == "composition":
            rule["via_relation"]      = rng.choice(others) if others else rel
            rule["inferred_relation"] = rel
        rules.append(rule)

    return rules


# ── YAML output ───────────────────────────────────────────────────────────────

def write_yaml(
    pairs: list[tuple[str, str]],
    relation_rules: dict[str, dict],
    inference_rules: list[dict],
    output: Path,
    domain: str = "synthetic",
) -> None:
    """Write a domain ontology YAML loadable by ``load_domain_ontology()``.

    Parameters
    ----------
    pairs :
        ``[(child, parent)]`` type hierarchy pairs.
    relation_rules :
        ``{relation: {"domain": [...], "target": [...]}}`` dict.
    inference_rules :
        List of rule dicts.
    output :
        Destination file path.  Parent directories are created if absent.
    domain :
        Value for the ``domain:`` header field.
    """
    data = {
        "domain":          domain,
        "version":         "1.0",
        "type_hierarchy":  [{"child": c, "parent": p} for c, p in pairs],
        "relation_rules":  relation_rules,
        "inference_rules": inference_rules,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data, f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"Written: {output}  "
          f"({len(pairs)} types, {len(relation_rules)} relations, "
          f"{len(inference_rules)} inference rules)")


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(ontology_path: Path) -> dict:
    """Benchmark OntologyRegistry loading against the generated ontology.

    Measures:
    - YAML parse time (``load_domain_ontology()``)
    - Registry wiring time (``add_domain_range_rules()`` + type registration)
    - Throughput in relations/sec

    Parameters
    ----------
    ontology_path :
        Path to the generated YAML.

    Returns
    -------
    dict
        ``{types, relations, load_ms, wire_ms, throughput_rel_s}``
    """
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from graphrag.graph.domain_ontology import (
        get_relation_rules,
        get_type_hierarchy_pairs,
        load_domain_ontology,
    )
    from graphrag.graph.ontology_registry import OntologyRegistry
    from unittest.mock import AsyncMock

    # Stub Neo4j — we're benchmarking in-process logic only
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(return_value=[])
    registry = OntologyRegistry(neo4j)

    # YAML parse time
    t0 = time.perf_counter()
    ontology = load_domain_ontology(ontology_path)
    t_load = time.perf_counter() - t0

    rules = get_relation_rules(ontology)
    pairs = get_type_hierarchy_pairs(ontology)

    # Registry wiring time
    t1 = time.perf_counter()
    registry.add_domain_range_rules(rules)
    for child, _ in pairs:
        registry._allowed_types.add(child)
    t_wire = time.perf_counter() - t1

    n_rel      = len(rules)
    throughput = n_rel / (t_wire or 1e-9)

    result = {
        "types":            len(pairs),
        "relations":        n_rel,
        "load_ms":          round(t_load * 1000, 2),
        "wire_ms":          round(t_wire * 1000, 2),
        "throughput_rel_s": round(throughput, 0),
    }

    print("\n  Benchmark results:")
    print(f"    Types loaded:           {result['types']}")
    print(f"    Relations registered:   {result['relations']}")
    print(f"    YAML parse time:        {result['load_ms']} ms")
    print(f"    Registry wire time:     {result['wire_ms']} ms")
    print(f"    Throughput:             {result['throughput_rel_s']:.0f} relations/sec")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic domain ontology YAML for benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--types", type=int, default=500,
        help="Number of entity types to generate",
    )
    parser.add_argument(
        "--relations", type=int, default=50,
        help="Number of relation rules to generate",
    )
    parser.add_argument(
        "--branching", type=int, default=5,
        help="Max children per node in the type tree",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output", default="config/ontologies/synthetic_large.yml",
        help="Output YAML file path",
    )
    parser.add_argument(
        "--domain", default="synthetic",
        help="Domain name in the YAML header",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run OntologyRegistry benchmark after generation",
    )
    args = parser.parse_args()

    output = Path(args.output)
    pairs       = generate_type_hierarchy(args.types, args.branching, args.seed)
    type_pool   = ["CONCEPT"] + [c for c, _ in pairs]
    rules       = generate_relation_rules(type_pool, args.relations, args.seed)
    inf_rules   = generate_inference_rules(list(rules.keys()), seed=args.seed)

    write_yaml(pairs, rules, inf_rules, output, args.domain)

    if args.benchmark:
        benchmark(output)


if __name__ == "__main__":
    main()
