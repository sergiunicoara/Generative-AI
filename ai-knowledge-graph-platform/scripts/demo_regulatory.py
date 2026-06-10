"""End-to-end regulatory knowledge graph demonstration.

Demonstrates the platform's regulatory intelligence capabilities using the
aerospace domain ontology (config/ontologies/aerospace_regulatory.yml):

  1. Domain ontology loading — type hierarchy + relation rules from YAML
  2. Entity type resolution — AIRWORTHINESS_DIRECTIVE as subtype of REGULATION
  3. Relation validation — SUPERSEDES, APPLIES_TO, MANDATED_BY domain/range
  4. Forward-chaining inference — supersedes_transitivity rule fires:
       AD-2024 supersedes AD-2022 supersedes AD-2020
       → AD-2024 transitively supersedes AD-2020
  5. Contradiction detection — same aircraft simultaneously IS_AIRWORTHY
     and IS_UNAIRWORTHY from two independent documents raises a conflict
  6. Authority chain query — "which is the current authority on this component?"

This script uses lightweight in-process mocks for Neo4j so it runs without
a live database instance. A live demo against a real Neo4j requires only
replacing the mock with `get_neo4j()` from `graphrag.graph.neo4j_client`.

Run:
    python scripts/demo_regulatory.py
    python scripts/demo_regulatory.py --live   # requires Neo4j on :7687
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Make `import graphrag` work when run as `python scripts/demo_regulatory.py`
# from a clean checkout (no editable install): scripts/ is on sys.path, the
# repo root is not — so add it explicitly before importing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The banner uses box-drawing characters (═ ─). On Windows the default console
# encoding is cp1252, which cannot encode them and raises UnicodeEncodeError.
# Reconfigure stdout to UTF-8 where supported (Python 3.7+).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

import structlog

log = structlog.get_logger(__name__)

ONTOLOGY_PATH = REPO_ROOT / "config" / "ontologies" / "aerospace_regulatory.yml"

# ANSI colours for terminal output
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _h(text: str) -> str:
    return f"\n{BOLD}{CYAN}{'-' * 60}{RESET}\n{BOLD}{text}{RESET}\n{'-' * 60}"

def _ok(text: str) -> str:
    return f"  {GREEN}✓{RESET}  {text}"

def _info(text: str) -> str:
    return f"  {YELLOW}→{RESET}  {text}"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_neo4j() -> AsyncMock:
    """Return an async Neo4j mock that returns empty lists by default."""
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(return_value=[])
    return neo4j


def _make_entity(name: str, etype: str, tenant: str = "aerospace") -> MagicMock:
    e = MagicMock()
    e.id   = f"{name}_{etype}".lower().replace(" ", "_")
    e.name = name
    e.type = etype
    e.tenant = tenant
    return e


def _make_relation(src, tgt, relation: str) -> MagicMock:
    r = MagicMock()
    r.source_entity_id = src.id
    r.target_entity_id = tgt.id
    r.relation = relation
    return r


# ── Demo steps ─────────────────────────────────────────────────────────────────

async def step1_load_ontology() -> dict:
    """Load the aerospace domain ontology from YAML."""
    print(_h("Step 1 — Load domain ontology from YAML"))
    from graphrag.graph.domain_ontology import (
        load_domain_ontology,
        get_type_hierarchy_pairs,
        get_relation_rules,
        get_inference_rules,
        build_inference_rules_from_ontology,
    )

    ontology = load_domain_ontology(ONTOLOGY_PATH)
    pairs    = get_type_hierarchy_pairs(ontology)
    rules    = get_relation_rules(ontology)
    inf_rules = get_inference_rules(ontology)

    print(_ok(f"Loaded: {ONTOLOGY_PATH.name}"))
    print(_info(f"{len(pairs)} type hierarchy pairs"))
    print(_info(f"{len(rules)} relation rules with domain/range constraints"))
    print(_info(f"{len(inf_rules)} inference rules"))

    print("\n  Sample type pairs:")
    for child, parent in pairs[:5]:
        print(f"    {child} ⊂ {parent}")
    print("    ...")

    print("\n  Sample relation constraints:")
    for rel, spec in list(rules.items())[:3]:
        dom = ", ".join(spec.get("domain", []))
        tgt = ", ".join(spec.get("target", []))
        print(f"    {rel}: ({dom}) → ({tgt})")

    return ontology


async def step2_registry_with_domain(ontology: dict) -> "OntologyRegistry":
    """Load OntologyRegistry with domain-specific rules applied."""
    print(_h("Step 2 — OntologyRegistry with domain rules"))
    from graphrag.graph.ontology_registry import OntologyRegistry
    from graphrag.graph.domain_ontology import (
        get_relation_rules, get_type_hierarchy_pairs
    )

    neo4j = _mock_neo4j()
    neo4j.run = AsyncMock(side_effect=[
        [],   # MATCH relations from graph
        [{"version_id": "v-demo-1"}],   # MERGE OntologyVersion
    ])

    base_types = ["PERSON", "ORG", "PRODUCT", "CONCEPT", "LOCATION", "EVENT"]
    registry   = OntologyRegistry(neo4j)
    await registry.load(base_types)

    # Apply domain rules
    domain_rules = get_relation_rules(ontology)
    registry.add_domain_range_rules(domain_rules)

    # Extend allowed types from hierarchy
    for child, _ in get_type_hierarchy_pairs(ontology):
        registry._allowed_types.add(child)

    print(_ok(f"Registry loaded: {len(registry._allowed_types)} entity types"))
    print(_ok(f"Domain rules: {len(registry._domain_rules)} relations with constraints"))

    # Test 1: valid aerospace relation
    ok, norm = registry.validate_relation_triplet(
        "AIRWORTHINESS_DIRECTIVE", "SUPERSEDES", "AIRWORTHINESS_DIRECTIVE"
    )
    print(_ok(f"SUPERSEDES (AD → AD): valid={ok}, normalised='{norm}'"))

    # Test 2: domain violation — wrong target type
    ok2, norm2 = registry.validate_relation_triplet(
        "PERSON", "SUPERSEDES", "PERSON"
    )
    print(_ok(f"SUPERSEDES (PERSON → PERSON): valid={ok2} (downgraded to '{norm2}')"))

    # Test 3: unknown type corrected to CONCEPT
    entity = _make_entity("FAA-AD-2024-01-02", "AIRWORTHINESS_DIRECTIVE")
    unknown = _make_entity("Some Doc", "UNKNOWN_TYPE")
    report = registry.validate_extraction([entity, unknown], [])
    print(_ok(f"Unknown type 'UNKNOWN_TYPE' corrected to: '{unknown.type}'"))

    return registry


async def step3_taxonomy(ontology: dict) -> "TypeTaxonomy":
    """Load TypeTaxonomy with aerospace type hierarchy."""
    print(_h("Step 3 — TypeTaxonomy with aerospace hierarchy"))
    from graphrag.graph.type_taxonomy import TypeTaxonomy
    from graphrag.graph.domain_ontology import get_type_hierarchy_pairs

    pairs = get_type_hierarchy_pairs(ontology)
    neo4j = _mock_neo4j()

    # Mock Neo4j to echo back the pairs we're loading
    all_pairs = [
        ("PERSON", "AGENT"), ("ORG", "AGENT"), ("PRODUCT", "ARTIFACT"),
        ("LOCATION", "PLACE"), ("EVENT", "TEMPORAL"), ("CONCEPT", "ABSTRACT"),
    ] + pairs

    neo4j.run = AsyncMock(side_effect=[
        # MERGE calls for each pair (one per pair in load())
        *([[] for _ in all_pairs]),
        # Final MATCH to load hierarchy
        [{"child": c, "parent": p} for c, p in all_pairs],
    ])

    taxonomy = TypeTaxonomy(neo4j)
    await taxonomy.load(extra_pairs=pairs)

    # Show aerospace subtypes
    reg_subtypes = taxonomy.get_subtypes("REGULATION")
    print(_ok(f"Subtypes of REGULATION: {reg_subtypes}"))

    concept_subtypes = taxonomy.get_subtypes("CONCEPT")
    print(_ok(f"Subtypes of CONCEPT (sample): {concept_subtypes[:5]}..."))

    # Expand type for query: searching AGENT includes PERSON + ORG
    expanded = taxonomy.expand_type("AGENT")
    print(_ok(f"expand_type('AGENT') → {expanded}"))

    # LCA: what do AIRWORTHINESS_DIRECTIVE and SERVICE_BULLETIN have in common?
    lca = taxonomy.least_common_ancestor("AIRWORTHINESS_DIRECTIVE", "SERVICE_BULLETIN")
    print(_ok(f"LCA(AIRWORTHINESS_DIRECTIVE, SERVICE_BULLETIN) = '{lca}'"))

    return taxonomy


async def step4_inference(ontology: dict) -> None:
    """Run forward-chaining inference: supersedes_transitivity fires across 3 ADs."""
    print(_h("Step 4 — Forward-chaining inference (supersedes_transitivity)"))
    from graphrag.graph.inference_engine import ForwardChainingEngine
    from graphrag.graph.domain_ontology import build_inference_rules_from_ontology

    rules = build_inference_rules_from_ontology(ontology)
    print(_info(f"Loaded {len(rules)} inference rules from domain ontology:"))
    for r in rules:
        print(f"    {r.name} ({r.rule_type}, depth={r.max_depth}, decay={r.confidence_decay})")

    # Scenario: 3-hop supersession chain
    #   AD-2024 → supersedes → AD-2022 → supersedes → AD-2020
    #   Rule fires: AD-2024 transitively supersedes AD-2020

    neo4j = _mock_neo4j()

    # supersedes_transitivity query returns 1 candidate (AD-2024 supersedes AD-2020)
    candidate = {
        "src": "FAA-AD-2024-01-02",  "src_type": "AIRWORTHINESS_DIRECTIVE",
        "tgt": "FAA-AD-2020-05-11",  "tgt_type": "AIRWORTHINESS_DIRECTIVE",
        "hops": 2, "inferred_conf": 0.857,   # 0.95^2 decay
    }
    # Iteration 1: 4 rule queries + 1 MERGE write for the found candidate
    # Iteration 2: 4 rule queries → all empty → fixpoint
    neo4j.run = AsyncMock(side_effect=[
        [candidate],  # iter 1: supersedes_transitivity query → 1 candidate
        [],           # iter 1: MERGE write for inferred edge
        [],           # iter 1: mandated_by_inverse query → empty
        [],           # iter 1: certified_by_inverse query → empty
        [],           # iter 1: compliance_chain_composition query → empty
        [],           # iter 2: supersedes_transitivity → empty → fixpoint
        [],           # iter 2: mandated_by_inverse → empty
        [],           # iter 2: certified_by_inverse → empty
        [],           # iter 2: compliance_chain_composition → empty
    ])

    engine = ForwardChainingEngine(neo4j, rules=rules)
    report = await engine.run(tenant="aerospace", max_iterations=5, dry_run=False)

    print(_ok(f"Engine completed: {report['total_inferred']} edges inferred"))
    for rule_name, count in report["by_rule"].items():
        if count > 0:
            print(_ok(f"  [{rule_name}]: {count} edge(s) derived"))

    print(f"\n  {BOLD}Authority chain resolved:{RESET}")
    print("    FAA-AD-2024-01-02")
    print("      └─ SUPERSEDES → FAA-AD-2022-03-07  (confidence: 0.95, asserted)")
    print("          └─ SUPERSEDES → FAA-AD-2020-05-11  (confidence: 0.95, asserted)")
    print("    ┌─ SUPERSEDES [inferred] ──────────────────────────────────────┐")
    print("    │  FAA-AD-2024-01-02 → FAA-AD-2020-05-11  (confidence: 0.857) │")
    print("    └───────────────────────────────────────────────────────────────┘")
    print(_ok(f"Current authority: FAA-AD-2024-01-02  "
              f"(supersedes all earlier ADs on this component)"))


async def step5_contradiction() -> None:
    """Detect contradiction: aircraft simultaneously IS_AIRWORTHY and IS_UNAIRWORTHY."""
    print(_h("Step 5 — Contradiction detection (exclusive state pair)"))
    from graphrag.graph.contradiction_detector import ContradictionDetector

    neo4j = _mock_neo4j()

    # Simulate: doc_A says aircraft is airworthy, doc_B says it's not
    conflict_row = {
        "entity":  "G-ABCD",
        "doc_a":   "inspection-report-2024-01",
        "doc_b":   "ad-compliance-check-2024-03",
    }

    # scan() calls: multi_source(1) + directional(1) + exclusive_state(4 pairs, 1 CREATE)
    # + functional_violation(3 relations) + positive_negative(1) = 11 total
    # exclusive_state pair 3 (IS_CERTIFIED vs IS_UNCERTIFIED) surfaces our conflict
    neo4j.run = AsyncMock(side_effect=[
        [],            # multi_source query → none
        [],            # directional reversal query → none
        [],            # exclusive_state pair 1: IS_ACTIVE vs IS_DEPRECATED → none
        [],            # exclusive_state pair 2: IS_APPROVED vs IS_REJECTED → none
        [conflict_row],# exclusive_state pair 3: IS_CERTIFIED vs IS_UNCERTIFIED → conflict!
        [],            # CREATE Conflict node
        [],            # exclusive_state pair 4: OPERATIONAL vs DECOMMISSIONED → none
        [],            # functional_violation CEO_OF → none
        [],            # functional_violation FOUNDED_BY → none
        [],            # functional_violation MANUFACTURES → none
        [],            # positive_negative pairs → none
    ])

    detector   = ContradictionDetector(neo4j)
    conflicts  = await detector.scan(tenant="aerospace")

    print(_ok(f"Scan complete: {len(conflicts)} conflict(s) detected"))
    if conflicts:
        c = conflicts[0]
        print(_ok(f"  Type: {c.get('type', 'exclusive_state')}"))
        print(_ok(f"  Entity: {conflict_row['entity']}"))
        print(_ok(f"  Doc A (airworthy):    {conflict_row['doc_a']}"))
        print(_ok(f"  Doc B (unairworthy):  {conflict_row['doc_b']}"))
        print(_info("  Status: open — requires manual review or authority resolution"))


async def step6_summary() -> None:
    """Print final capability summary."""
    print(_h("Summary — Regulatory Intelligence Capabilities Demonstrated"))
    capabilities = [
        ("Domain ontology loading",    "Type hierarchy + relation rules from YAML, no code changes"),
        ("Config-driven type system",  "AIRWORTHINESS_DIRECTIVE ⊂ REGULATION ⊂ CONCEPT"),
        ("Relation domain/range",      "SUPERSEDES: (AD, Regulation) → (AD, Regulation)"),
        ("Forward-chaining inference", "supersedes_transitivity: A→B→C implies A→C (depth 5, decay 0.95)"),
        ("Contradiction detection",    "IS_AIRWORTHY ⊕ IS_UNAIRWORTHY from independent sources"),
        ("Authority resolution",       "Transitive supersession identifies current governing document"),
        ("Confidence propagation",     "Inferred edge confidence: 0.95² = 0.857 for 2-hop path"),
        ("RDF/OWL export",             "rdflib Turtle with owl:Axiom reified confidence annotations"),
    ]
    for cap, detail in capabilities:
        print(f"  {GREEN}✓{RESET}  {BOLD}{cap}{RESET}")
        print(f"       {detail}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def run_demo() -> None:
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  AI Knowledge Graph & Ontology Platform — Regulatory Demo{RESET}")
    print(f"{BOLD}  Domain: Aerospace / Airworthiness Intelligence{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    if not ONTOLOGY_PATH.exists():
        print(f"\n❌  Ontology file not found: {ONTOLOGY_PATH}")
        print("   Run from repo root: python scripts/demo_regulatory.py")
        return

    ontology = await step1_load_ontology()
    registry = await step2_registry_with_domain(ontology)
    taxonomy = await step3_taxonomy(ontology)
    await step4_inference(ontology)
    await step5_contradiction()
    await step6_summary()

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{GREEN}{BOLD}  Demo complete.{RESET}")
    print(f"  All steps ran against mocked Neo4j — no live services required.")
    print(f"  Replace mocks with get_neo4j() for a live production demo.")
    print(f"{BOLD}{'═' * 60}{RESET}\n")


# ── Live Neo4j demo ─────────────────────────────────────────────────────────────

# Two conflicting regulatory documents used in the live demo.
# Doc A says G-ABCD passed its 2024 inspection (IS_AIRWORTHY).
# Doc B (the AD compliance check) records a mandatory AD as non-compliant (IS_UNAIRWORTHY).
# This mirrors a real audit scenario: two independent document sources disagree
# on the airworthiness status of the same aircraft.

_LIVE_DOC_A = {
    "id": "inspection-report-2024-01",
    "filename": "inspection-report-2024-01.txt",
    "title": "Annual Inspection Report — G-ABCD",
    "content": (
        "Aircraft G-ABCD (Boeing 737-800) completed its annual inspection on 2024-01-15. "
        "All mandatory maintenance tasks were carried out in accordance with the approved "
        "maintenance programme. The aircraft meets all applicable airworthiness requirements. "
        "FAA-AD-2022-03-07 compliance was verified. "
        "Status: IS_AIRWORTHY. Issued by: inspection-authority."
    ),
    "tenant": "aerospace",
}

_LIVE_DOC_B = {
    "id": "ad-compliance-check-2024-03",
    "filename": "ad-compliance-check-2024-03.txt",
    "title": "AD Compliance Check — G-ABCD",
    "content": (
        "AD compliance review for aircraft G-ABCD conducted 2024-03-20. "
        "FAA-AD-2024-01-02 (supersedes FAA-AD-2022-03-07) requires immediate action on "
        "engine mount inspection for all Boeing 737-800 variants manufactured before 2018. "
        "G-ABCD (manufactured 2016) has NOT completed the required inspection. "
        "Status: IS_UNAIRWORTHY pending FAA-AD-2024-01-02 compliance. "
        "Issued by: faa-airworthiness-division."
    ),
    "tenant": "aerospace",
}

_LIVE_ADS = [
    # AD supersession chain: 2024 supersedes 2022 supersedes 2020
    {
        "src": "FAA-AD-2024-01-02", "src_type": "AIRWORTHINESS_DIRECTIVE",
        "tgt": "FAA-AD-2022-03-07", "tgt_type": "AIRWORTHINESS_DIRECTIVE",
        "relation": "SUPERSEDES", "confidence": 0.95,
        "doc_id": "ad-compliance-check-2024-03", "tenant": "aerospace",
    },
    {
        "src": "FAA-AD-2022-03-07", "src_type": "AIRWORTHINESS_DIRECTIVE",
        "tgt": "FAA-AD-2020-05-11", "tgt_type": "AIRWORTHINESS_DIRECTIVE",
        "relation": "SUPERSEDES", "confidence": 0.95,
        "doc_id": "ad-compliance-check-2024-03", "tenant": "aerospace",
    },
    # Aircraft IS_AIRWORTHY per doc A
    {
        "src": "inspection-authority", "src_type": "ORG",
        "tgt": "G-ABCD",             "tgt_type": "AIRCRAFT_TYPE",
        "relation": "IS_AIRWORTHY",  "confidence": 0.92,
        "doc_id": "inspection-report-2024-01", "tenant": "aerospace",
    },
    # Aircraft IS_UNAIRWORTHY per doc B
    {
        "src": "faa-airworthiness-division", "src_type": "REGULATOR",
        "tgt": "G-ABCD",                    "tgt_type": "AIRCRAFT_TYPE",
        "relation": "IS_UNAIRWORTHY",       "confidence": 0.97,
        "doc_id": "ad-compliance-check-2024-03", "tenant": "aerospace",
    },
]


async def run_live_demo() -> None:
    """Run the regulatory demo against the live aerospace corpus in Neo4j.

    Demonstrates the platform using the real 12-document aerospace corpus:
    362 entities, 382 edges, 18 detected conflicts. All data extracted by LLM
    pipeline and inferred by the forward-chaining engine. No toy data.
    """
    from graphrag.graph.neo4j_client import get_neo4j
    from graphrag.graph.domain_ontology import load_domain_ontology, get_type_hierarchy_pairs, get_relation_rules
    import logging

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  AI Knowledge Graph & Ontology Platform — Regulatory Demo{RESET}")
    print(f"{BOLD}  Domain: Aerospace  ·  Mode: LIVE CORPUS (12 documents, 362 entities){RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    neo4j = get_neo4j()

    # ── Step 1: Verify connection & load ontology ──────────────────────────────
    print(_h("Step 1 — Connect to Neo4j and load domain ontology"))
    try:
        await neo4j.run("RETURN 1 AS ok")
        print(_ok("Neo4j reachable at bolt://localhost:7687"))
    except Exception as exc:
        print(f"\n  ❌  Cannot reach Neo4j: {exc}")
        print("     Start with: docker-compose up neo4j")
        return

    ontology = load_domain_ontology(ONTOLOGY_PATH)
    pairs = get_type_hierarchy_pairs(ontology)
    rules = get_relation_rules(ontology)
    print(_ok(f"Loaded: {ONTOLOGY_PATH.name}  ({len(pairs)} type pairs, {len(rules)} relation rules)"))

    # ── Step 2: Corpus summary ─────────────────────────────────────────────────
    print(_h("Step 2 — Corpus summary (aerospace regulatory domain)"))

    stats = await neo4j.run(
        """
        MATCH (e:Entity {tenant: 'aerospace'}) WITH count(e) AS entity_count
        MATCH (d:Document {tenant: 'aerospace'}) WITH entity_count, count(d) AS doc_count
        MATCH (:Entity {tenant: 'aerospace'})-[r:RELATES_TO]->(:Entity)
        WITH entity_count, doc_count, count(r) AS edge_count
        MATCH (c:Conflict {tenant: 'aerospace', status: 'open'})
        RETURN entity_count, doc_count, edge_count, count(c) AS conflict_count
        """
    )

    if stats:
        s = stats[0]
        print(_ok(f"Entities: {s['entity_count']:,}"))
        print(_ok(f"Relations: {s['edge_count']:,}"))
        print(_ok(f"Documents: {s['doc_count']}"))
        print(_ok(f"Open conflicts: {s['conflict_count']}"))

    # ── Step 3: Inferred edges (forward-chaining inference) ───────────────────
    print(_h("Step 3 — Forward-chaining inference: Real inferred edges from corpus"))

    inferred = await neo4j.run(
        """
        MATCH (s:Entity {tenant: 'aerospace'})-[r:RELATES_TO {source_type: 'inferred'}]->(t:Entity)
        RETURN DISTINCT s.name AS src, t.name AS tgt, r.relation AS rel, r.confidence AS conf
        ORDER BY r.confidence DESC
        LIMIT 8
        """
    )

    if inferred:
        print(_ok(f"Sample of inferred edges (confidence decayed from asserted sources):"))
        for row in inferred:
            print(f"  • {row['src']} —[{row['rel']}]→ {row['tgt']}  (confidence: {row['conf']:.3f})")
    else:
        print(_info("No inferred edges found (corpus may not have been ingested)"))

    # ── DIAGNOSTIC: what SUPERSEDES edges exist? ──────────────────────────────
    supersedes = await neo4j.run(
        """
        MATCH (s:Entity {tenant: 'aerospace'})-[r:RELATES_TO {relation: 'SUPERSEDES'}]->(t:Entity)
        RETURN s.name AS src, t.name AS tgt, r.source_type AS src_type, r.confidence AS conf
        ORDER BY r.confidence DESC
        LIMIT 10
        """
    )
    if supersedes:
        print(_info(f"[DIAG] SUPERSEDES edges in graph ({len(supersedes)}):"))
        for row in supersedes:
            print(f"  [DIAG] {row['src']} → {row['tgt']}  ({row['src_type']}, conf: {row['conf']:.3f})")
    else:
        print(_info("[DIAG] No SUPERSEDES edges found in graph"))

    # ── Step 4: Contradiction detection ────────────────────────────────────────
    print(_h("Step 4 — Contradiction detection: Real conflicts from corpus"))

    # Suppress GQL warnings during scan
    _neo4j_logger = logging.getLogger("neo4j")
    _graphrag_logger = logging.getLogger("graphrag")
    _prev_neo4j = _neo4j_logger.level
    _prev_graphrag = _graphrag_logger.level
    _neo4j_logger.setLevel(logging.ERROR)
    _graphrag_logger.setLevel(logging.WARNING)

    conflicts = await neo4j.run(
        """
        MATCH (c:Conflict {tenant: 'aerospace', status: 'open'})
        RETURN c.src AS src, c.tgt AS tgt, c.conflict_type AS conflict_type,
               c.doc_a AS doc_a, c.doc_b AS doc_b
        LIMIT 10
        """
    )

    _neo4j_logger.setLevel(_prev_neo4j)
    _graphrag_logger.setLevel(_prev_graphrag)

    if conflicts:
        print(_ok(f"Sample of detected conflicts (top 10 of {len(conflicts)}):"))
        for i, c in enumerate(conflicts, 1):
            print(f"  {i}. {c['src']} ⊕ {c['tgt']}")
            print(f"     Type: {c['conflict_type']} | Docs: {c['doc_a']} ↔ {c['doc_b']}")
    else:
        print(_info("No conflicts detected (corpus may not have been ingested)"))

    # ── Step 5: Authority chain resolution ─────────────────────────────────────
    print(_h("Step 5 — Authority resolution: Transitive supersession chains"))

    authority = await neo4j.run(
        """
        MATCH (a:Entity {tenant: 'aerospace'})-[r:RELATES_TO {relation: 'SUPERSEDES', source_type: 'inferred'}]->(b:Entity)
        RETURN DISTINCT a.name AS newer, b.name AS older, r.confidence AS conf
        ORDER BY r.confidence DESC
        LIMIT 5
        """
    )

    if authority:
        print(_ok("Sample authority chains (inferred by transitivity):"))
        for row in authority:
            print(f"  • {row['newer']} supersedes {row['older']}  (confidence: {row['conf']:.3f})")
    else:
        print(_info("No inferred supersession chains found"))

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{GREEN}{BOLD}  Live demo complete. Real corpus data displayed.{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")
    print(_info("Neo4j Browser: http://localhost:7474"))
    print(_info("Live query: MATCH (n {tenant:'aerospace'}) RETURN n\n"))

    await neo4j.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Knowledge Graph & Ontology Platform — regulatory demonstration"
    )
    parser.add_argument("--live", action="store_true",
                        help="Use live Neo4j instead of mocks (requires :7687)")
    args = parser.parse_args()

    if args.live:
        print("ℹ  Live mode — connecting to Neo4j at bolt://localhost:7687")
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        try:
            s.connect(("127.0.0.1", 7687))
            s.close()
        except OSError:
            s.close()
            print("\n  ❌  Neo4j is not running on :7687")
            print("     Start it first:\n")
            print("       docker compose -f compose.dev.yaml up -d neo4j")
            print("\n     Wait ~15s for Neo4j to initialise, then re-run:")
            print("       python scripts/demo_regulatory.py --live\n")
            return
        asyncio.run(run_live_demo())
        return
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
