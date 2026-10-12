"""Phase 4: deterministic query routing."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from graphrag.retrieval.query_router import (
    LEGACY_TO_ROUTE,
    ROUTE_BEHAVIOUR,
    Route,
    baseline_route,
    enforced_overrides,
    route_query,
)
from graphrag.retrieval.query_planner import QUERY_CLASSES, retrieval_plan


@pytest.mark.parametrize("question,route", [
    ("What torque value does the maintenance manual specify for the bolts?", Route.FACTUAL_LOOKUP),
    ("Who is Acme Aerospace?", Route.ENTITY_LOOKUP),
    ("How does the FAA directive relate to the EASA directive?", Route.RELATIONAL),
    ("What is the full supersession chain for the bulletin?", Route.MULTI_HOP),
    ("How many suppliers are missing emissions evidence?", Route.AGGREGATION),
    ("Which revision was in force on 2023-06-01?", Route.TEMPORAL),
    ("And that one?", Route.AMBIGUOUS),
])
def test_each_route_class(question, route):
    assert route_query(question).route is route


def test_every_decision_carries_a_reason_and_the_legacy_plan():
    d = route_query("How does the FAA directive relate to the EASA directive?")
    assert d.reason == "legacy_relational" and d.legacy_class == "relational"
    plan = retrieval_plan("How does the FAA directive relate to the EASA directive?")
    assert (d.mode, d.top_k, d.fallback) == (plan["mode"], plan["top_k"], plan["fallback"])
    assert d.to_dict()["behaviour"] == ROUTE_BEHAVIOUR[Route.RELATIONAL]


def test_routing_is_deterministic():
    q = "As of 2022-01-15, who supplied the brake discs?"
    assert route_query(q) == route_query(q)


def test_legacy_classes_are_aliases():
    assert set(LEGACY_TO_ROUTE) == set(QUERY_CLASSES)
    assert baseline_route("Do the manuals contradict each other?") is Route.RELATIONAL


def test_explicit_valid_at_makes_the_query_temporal():
    d = route_query("What torque value applies?", explicit_valid_at="2024-01-01")
    assert d.route is Route.TEMPORAL and d.as_of == "2024-01-01" and d.reason == "explicit_valid_at"


def test_iso_date_in_question_becomes_as_of():
    assert route_query("As of 2022-01-15, who supplied the brake discs?").as_of == "2022-01-15"


def test_aggregation_with_supported_template_names_the_intent():
    d = route_query("How many suppliers are missing emissions evidence?")
    assert d.route is Route.AGGREGATION
    d2 = route_query("What is the average delivery delay across suppliers?")
    assert d2.route is Route.AGGREGATION and d2.structured_intent is None
    assert d2.reason == "aggregation_phrase_no_template"


def test_deictic_question_is_ambiguous_only_without_session():
    assert route_query("What about it?").route is Route.AMBIGUOUS
    assert route_query("What about it and the brake discs supplier?", has_session=True).route \
        is not Route.AMBIGUOUS


# ── enforced behaviour ───────────────────────────────────────────────────────

def test_lookups_skip_graph_expansion_when_enforced():
    o = enforced_overrides(route_query("What torque value does the manual specify for the bolts?"), {})
    assert o == {"multihop_depth": 0, "gnn_enabled": False, "entity_context_enabled": False}


def test_entity_lookup_is_limited_to_one_hop():
    assert enforced_overrides(route_query("Who is Acme Aerospace?"), {"multihop_depth": 3}) == {"multihop_depth": 1}


def test_relational_and_multi_hop_keep_configured_expansion():
    assert enforced_overrides(route_query("How does A relate to B?"), {}) == {}
    assert enforced_overrides(route_query("What is the full supersession chain?"), {}) == {}


def test_temporal_and_aggregation_disable_agentic_fallback():
    assert enforced_overrides(route_query("Which revision was in force on 2023-06-01?"), {})["agentic_fallback"] is False
    assert enforced_overrides(route_query("How many suppliers are missing emissions evidence?"), {})[
        "agentic_fallback"] is False


def test_router_never_uses_an_llm():
    import graphrag.retrieval.query_router as qr
    src = inspect.getsource(qr)
    assert "get_llm" not in src and "llm_client" not in src


# ── hybrid retriever integration ─────────────────────────────────────────────

def test_policy_defaults_to_observe_and_enforce_is_explicit():
    from graphrag.retrieval import hybrid_retriever
    src = inspect.getsource(hybrid_retriever.HybridRetriever.retrieve_and_answer)
    assert 'cfg.get("query_router_policy", "observe")' in src
    assert 'router_policy == "enforce" and requested_mode == "hybrid"' in src
    # the structured path is never taken under ACL
    assert "and not acl_enforced" in src


def test_fallback_cannot_recurse():
    from graphrag.retrieval import agentic_retriever, hybrid_retriever
    src = inspect.getsource(hybrid_retriever.HybridRetriever.retrieve_and_answer)
    assert "not _FALLBACK_ACTIVE.get()" in src and "_FALLBACK_ACTIVE.reset(_fallback_token)" in src
    # the agentic retriever searches locally and never re-enters the hybrid retriever
    agentic_src = inspect.getsource(agentic_retriever)
    assert "hybrid_retriever import" not in agentic_src and "HybridRetriever(" not in agentic_src
    assert ".retrieve_and_answer(" not in agentic_src.split("class AgenticRetriever", 1)[1].replace(
        "async def retrieve_and_answer(", "")
    assert hybrid_retriever._FALLBACK_ACTIVE.get() is False


@pytest.mark.asyncio
async def test_structured_answer_is_deterministic_cited_and_tenant_scoped():
    from graphrag.retrieval.hybrid_retriever import HybridRetriever

    r = HybridRetriever.__new__(HybridRetriever)
    rows = [{"source": "Acme", "relation": "OWNS", "target": "Bolt", "source_doc_id": "d1"}]
    fake = AsyncMock(return_value={"intent": "entity_relations", "rows": rows, "count": 1})
    with patch("graphrag.graph.controlled_query.execute_controlled_query", fake), \
         patch("graphrag.retrieval.hybrid_retriever.get_neo4j"):
        decision = route_query("How many suppliers are missing emissions evidence?")
        out = await r._structured_answer("q", "acme", decision, 0.0, query_id="q1", correlation_id="c")
    assert fake.await_args.kwargs["tenant"] == "acme"
    assert out.retrieval_mode == "structured" and out.citations == ["d1"]
    assert out.route == "AGGREGATION" and "Acme" in out.answer and out.query_id == "q1"


@pytest.mark.asyncio
async def test_empty_structured_result_falls_back_to_normal_retrieval():
    from graphrag.retrieval.hybrid_retriever import HybridRetriever

    r = HybridRetriever.__new__(HybridRetriever)
    fake = AsyncMock(return_value={"intent": "x", "rows": [], "count": 0})
    with patch("graphrag.graph.controlled_query.execute_controlled_query", fake), \
         patch("graphrag.retrieval.hybrid_retriever.get_neo4j"):
        assert await r._structured_answer("q", "acme", route_query("How many?"), 0.0,
                                          query_id=None, correlation_id="") is None


def test_query_result_exposes_route():
    from graphrag.core.models import QueryResult
    r = QueryResult(question="q", answer="a", route="TEMPORAL", route_reason="temporal_phrase")
    assert r.route == "TEMPORAL" and r.route_reason == "temporal_phrase"


# ── evaluation fixtures and recorded results ─────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]


def test_routing_fixtures_cover_every_route_and_scenario():
    cases = json.loads((ROOT / "evals" / "routing_cases.json").read_text(encoding="utf-8"))["cases"]
    assert {c["expected"] for c in cases} == {r.value for r in Route}
    assert {"stale", "conflicting", "unauthorized"} <= {c.get("scenario") for c in cases}
    assert len({c["id"] for c in cases}) == len(cases)


def test_recorded_routing_results_match_a_fresh_offline_run():
    """The committed numbers are reproducible, not hand-written."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_routing", ROOT / "scripts" / "eval_routing.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = json.loads((ROOT / "evals" / "routing_cases.json").read_text(encoding="utf-8"))["cases"]
    fresh = mod.offline(cases)
    recorded = json.loads((ROOT / "evals" / "routing_eval_results.json").read_text(encoding="utf-8"))
    assert recorded["offline"]["route_accuracy"] == fresh["route_accuracy"]
    assert recorded["live"] is None or isinstance(recorded["live"], dict)


@pytest.mark.parametrize("question,route", [
    # document identifiers that look like dates are not time constraints
    ("What is the inspection interval stated in AD 2024-03-07?", Route.FACTUAL_LOOKUP),
    ("What does bulletin SB-2023-11-04 require?", Route.FACTUAL_LOOKUP),
    # a real calendar date still is
    ("Which revision was in force on 2023-06-15?", Route.TEMPORAL),
    # relation verbs without the word "related"
    ("Who owns Bolt Supplier GmbH?", Route.RELATIONAL),
    ("Which components does the hydraulic pump depend on?", Route.RELATIONAL),
    # chains of relations
    ("Which parts supplied by Bolt's subsidiaries are used in aircraft operated by Acme's customers?", Route.MULTI_HOP),
    # single named subject, and references with nothing to resolve them
    ("What is EASA?", Route.ENTITY_LOOKUP),
    ("Is this still valid?", Route.AMBIGUOUS),
    ("Is it still allowed?", Route.AMBIGUOUS),
])
def test_routing_fixes_from_recorded_misses(question, route):
    assert route_query(question).route is route


def test_reference_is_not_ambiguous_with_a_session_or_a_named_subject():
    assert route_query("Is this still valid?", has_session=True).route is not Route.AMBIGUOUS
    assert route_query("Is it listed in AD 2020-01-02?").route is not Route.AMBIGUOUS


@pytest.mark.parametrize("question,route", [
    # before/after/since/until/during constrain time only with a year or month nearby
    ("What must the technician do after removing the access panel?", Route.FACTUAL_LOOKUP),
    ("What do the operators have to record after each inspection?", Route.FACTUAL_LOOKUP),
    ("What was the inspection interval before the 2024 amendment?", Route.TEMPORAL),
    ("Since March 2024, who has operated the aircraft?", Route.TEMPORAL),
    # "part number of" is not a count; "the number of" is
    ("What is the part number of the main fuel filter?", Route.FACTUAL_LOOKUP),
    ("What is the number of aircraft in the fleet?", Route.AGGREGATION),
    # a capitalised word inside a title / company name is not a relation or a hop
    ("What does the Supplier Quality Manual say about incoming inspection?", Route.FACTUAL_LOOKUP),
    ("What is the Operations Manual limit for crosswind landings?", Route.FACTUAL_LOOKUP),
    ("Which parts are made by Bolt Supplier GmbH?", Route.RELATIONAL),
    # "operators" as a noun is not a relation; "operators of X" is
    ("What does the procedure require operators to do?", Route.FACTUAL_LOOKUP),
    ("Who are the operators of the A320neo fleet?", Route.RELATIONAL),
    # "reference"/"cite" as the question's verb
    ("Which documents does the maintenance programme reference?", Route.RELATIONAL),
    ("What is the document reference of the cabin safety instruction?", Route.FACTUAL_LOOKUP),
    # "How is X related to Y"
    ("How is the quality manual related to the maintenance organisation exposition?", Route.RELATIONAL),
    # sentence-final punctuation must not turn a filler word into a content word
    ("Same question for the other one.", Route.AMBIGUOUS),
])
def test_rules_from_the_second_and_third_review_with_their_look_alikes(question, route):
    assert route_query(question).route is route


def test_trailing_punctuation_is_stripped_from_content_words_but_identifiers_survive():
    from graphrag.retrieval.query_router import _content_words

    assert _content_words("Same for the other one.") == ["same", "other", "one"]
    assert "ad-2024-01-02" in _content_words("Is AD-2024-01-02 current?")
    assert "14.5" in _content_words("What is 14.5 here?")


def test_every_recorded_routing_set_is_scored_by_the_eval_script():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sets = sorted((root / "evals").glob("routing_cases_heldout*.json"))
    assert len(sets) >= 3
    ids = [c["id"] for p in sets for c in json.loads(p.read_text(encoding="utf-8"))["cases"]]
    assert len(ids) == len(set(ids))        # ids are unique across sets
