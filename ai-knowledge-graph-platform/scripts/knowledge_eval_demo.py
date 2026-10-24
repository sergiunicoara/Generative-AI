"""Walkthrough: one synthetic carton, several business contexts, different outcomes.

    python scripts/knowledge_eval_demo.py
    python scripts/knowledge_eval_demo.py --json        # full decisions

Publishes the synthetic packaging/labeling pack through the release gate into
a throwaway governance store, then evaluates the same PX-200 carton in
different contexts through the same service the MCP capability and the
``POST /knowledge/evaluate`` route use.  At the end, a minimal "agent" acts on
each structured decision without re-implementing any rule logic.

All rules are invented demonstration rules (see the pack's disclaimer).  No
network, no model calls; nothing is written outside a temporary directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.context_graph.knowledge import BusinessContext, Fact  # noqa: E402
from graphrag.context_graph.knowledge_evaluation import EvaluationDecision, OverallOutcome  # noqa: E402
from graphrag.context_graph.knowledge_release import load_cases, load_pack, publish_pack  # noqa: E402
from graphrag.context_graph.knowledge_service import KnowledgeEvaluationService  # noqa: E402
from graphrag.domains.energy.governance_store import GovernanceStore  # noqa: E402
from graphrag.enterprise.models import AccessContext  # noqa: E402

PACK = ROOT / "config" / "knowledge_packs" / "packaging_labeling.yml"
CASES = ROOT / "evals" / "knowledge_eval" / "packaging_labeling_cases.json"
EVIDENCE = [{"source_id": "ARTWORK-PX200-R3", "locator": "artwork spec"}]

BASE_FACTS = {
    "label.batch_code_present": True, "label.min_font_pt": 7, "label.front_claims": ["clinically_proven"],
    "label.brand_panel_color": "green", "label.allergen_statement": "Contains lactose",
    "label.languages": ["sl", "en"], "label.storage_statement": "Store below 25 C",
}
SCENARIOS = [
    ("Northland, before revision 2 (7 pt allowed)",
     {"market": "NORTHLAND", "customer": "CORNERSHOP", "as_of": "2026-03-01T00:00:00Z"}, {}),
    ("Northland, after revision 2 (8 pt required)",
     {"market": "NORTHLAND", "customer": "CORNERSHOP", "as_of": "2026-08-01T00:00:00Z"}, {}),
    ("Northland mini-pack with an authorised waiver",
     {"market": "NORTHLAND", "customer": "CORNERSHOP", "product": "PX-200-MINI", "as_of": "2026-08-01T00:00:00Z"},
     {"label.front_claims": []}),
    ("Northland for MegaMart: permission vs prohibition",
     {"market": "NORTHLAND", "customer": "MEGAMART", "as_of": "2026-08-01T00:00:00Z"},
     {"label.min_font_pt": 8, "label.storage_statement": "Store below 30 C"}),
    ("Southland for MegaMart: contradictory storage rules",
     {"market": "SOUTHLAND", "customer": "MEGAMART", "as_of": "2026-08-01T00:00:00Z"}, {"label.front_claims": []}),
    ("Market not stated: incomplete context",
     {"customer": "CORNERSHOP", "as_of": "2026-08-01T00:00:00Z"}, {"label.front_claims": []}),
]


def _context(overrides: dict) -> BusinessContext:
    base = {"brand": "OTHERBRAND", "product": "PX-200", "product_family": "otc_tablets",
            "workflow": "label_review"}
    return BusinessContext.model_validate({**base, **overrides})


def _facts(overrides: dict) -> dict[str, Fact]:
    merged = {**BASE_FACTS, **overrides}
    return {name: Fact(value=value, evidence=EVIDENCE) for name, value in merged.items()}


def agent_action(decision: EvaluationDecision) -> str:
    """What a consuming agent does with the result -- no rule logic of its own.

    Approval is an allowlist: only a complete ``compliant`` decision with no
    review flag approves.  Anything unresolved, incomplete, or with no
    applicable knowledge goes to a person; it never falls through to approval.
    """
    if decision.overall == OverallOutcome.NON_COMPLIANT:
        failed = [f"{r.item_id} ({r.title})" for r in decision.results if r.blocking and r.outcome.value == "fail"]
        action = "block release of the artwork; fix " + ", ".join(failed)
        if decision.human_review_required:
            action += "; also route to human review: " + "; ".join(decision.human_review_reasons)
        return action
    if decision.overall != OverallOutcome.COMPLIANT or decision.human_review_required:
        reasons = list(decision.human_review_reasons) or [f"decision is {decision.overall.value}"]
        return "route to human review: " + "; ".join(reasons)
    advisories = [r.item_id for r in decision.results if not r.blocking and r.outcome.value == "fail"]
    return "approve artwork" + (f" (advisory notes: {', '.join(advisories)})" if advisories else "")


async def main(as_json: bool) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        store = GovernanceStore(f"sqlite+aiosqlite:///{(Path(tmp) / 'governance.sqlite').as_posix()}",
                                blob_root=Path(tmp) / "blobs")
        await store.open()
        try:
            pack = load_pack(PACK)
            release = await publish_pack(store, pack, published_by="user:demo-publisher",
                                         cases=load_cases(CASES))
            print(f"Published release v{release['version']} {release['release_id'][:12]} "
                  f"({len(release['item_keys'])} items, {len(release['exception_keys'])} exceptions; "
                  f"gate: {release['checks']})")
            print(f"Disclaimer: {pack.disclaimer.strip()}\n")
            service = KnowledgeEvaluationService(store)
            access = AccessContext.from_claims({"sub": "artwork-agent"})
            outputs = []
            for title, context, fact_overrides in SCENARIOS:
                decision = await service.evaluate(
                    tenant=pack.tenant, domain=pack.domain, access=access,
                    context=_context(context), facts=_facts(fact_overrides))
                outputs.append(decision.model_dump(mode="json"))
                print(f"== {title}")
                print(f"   overall: {decision.overall.value}   human review: {decision.human_review_required}")
                for result in decision.results:
                    if result.applicability.value == "does_not_apply":
                        continue
                    print(f"   {result.item_id:8s} {result.modality.value:11s} "
                          f"{result.applicability.value:8s} -> {result.outcome.value:21s} {result.outcome_reason}")
                for conflict in decision.conflicts:
                    state = "resolved by " + conflict.resolution_ref if conflict.resolved else "UNRESOLVED"
                    print(f"   conflict {conflict.kind} {conflict.item_keys} {state}")
                for exc in decision.exceptions_applied:
                    print(f"   exception {exc.exception_key} ({exc.kind}) approved {exc.approval_ref}")
                print(f"   agent: {agent_action(decision)}\n")
            if as_json:
                print(json.dumps(outputs, indent=2))
        finally:
            await store.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args().json)))
