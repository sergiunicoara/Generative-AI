"""Answer-attributed citations: the opt-in plan, the parser and the result field (offline, no model, no database)."""
from __future__ import annotations

from graphrag.core.models import QueryResult
from graphrag.evaluation import citations as evaluation_citations
from graphrag.retrieval import answer_citations as ac
from graphrag.retrieval.answer_policy import answer_prompt

DOCS = ["FAA-AD-2024-01-02.txt", "SWA_fleet_registry_2024.txt", "14CFR_Part39_excerpt.txt"]
ON = {"answer_citation_tags_enabled": True}


def test_the_option_is_off_by_default_and_then_changes_nothing():
    assert ac.tag_plan({}, DOCS) is None
    assert ac.tag_plan({"answer_citation_tags_enabled": False}, DOCS) is None


def test_the_plan_lists_every_document_id_once_sorted_without_the_txt_suffix():
    instruction, ids = ac.tag_plan(ON, DOCS + ["FAA-AD-2024-01-02.txt", ""])
    assert ids == ["14CFR_Part39_excerpt", "FAA-AD-2024-01-02", "SWA_fleet_registry_2024"]
    assert "[DOC: <id>]" in instruction and "  - FAA-AD-2024-01-02\n" in instruction
    assert instruction == ac.citation_instruction(ids)


def test_the_plan_fails_closed_without_documents_or_above_the_cap():
    assert ac.tag_plan(ON, []) is None
    assert ac.tag_plan({**ON, "answer_citation_tags_max_documents": 2}, DOCS) is None
    assert ac.tag_plan({**ON, "answer_citation_tags_max_documents": 3}, DOCS) is not None


def test_attribute_strips_the_tags_and_returns_canonical_ids_in_first_seen_order():
    ids = ["14CFR_Part39_excerpt", "FAA-AD-2024-01-02", "SWA_fleet_registry_2024"]
    answer = ("It supersedes the 2022 directive [DOC: FAA-AD-2024-01-02]; see the registry "
              "[swa_fleet_registry_2024] and [DOC: Aircraft, FAA-AD-2024-01-02]. [Note] stays.")
    text, cited, parsed = ac.attribute(answer, ids)
    assert cited == ["FAA-AD-2024-01-02", "SWA_fleet_registry_2024"]       # "Aircraft" is no document: dropped
    assert "[DOC:" not in text and "[swa_fleet" not in text and "[Note] stays." in text
    assert (parsed["n_prefixed"], parsed["n_bare"]) == (3, 1)


def test_attribute_never_fails_a_query_on_ids_that_collide_after_normalisation():
    text, cited, _ = ac.attribute("A [DOC: FAA-AD-2024-01-02].", ["FAA-AD-2024-01-02", "FAA AD 2024 01 02"])
    assert cited == [] and text == "A."


def test_the_evaluation_module_shares_the_production_parser_and_rule():
    assert evaluation_citations.citation_instruction is ac.citation_instruction
    assert evaluation_citations.build_alias_index is ac.build_alias_index
    parsed = evaluation_citations.parse_answer_tags("x [DOC: FAA-AD-2024-01-02]")      # default: committed alias map
    assert parsed == ac.parse_answer_tags("x [DOC: FAA-AD-2024-01-02]", {"FAA-AD-2024-01-02"})


def test_the_answer_prompt_is_byte_identical_when_the_option_is_off():
    cfg = {"answer_language_rule_enabled": False}
    plan = ac.tag_plan(cfg, DOCS)
    prompt_cfg = {**cfg, "eval_citation_instruction": plan[0]} if plan else cfg
    assert answer_prompt(prompt_cfg) == answer_prompt(cfg)
    instruction, _ = ac.tag_plan(ON, DOCS)
    assert instruction in answer_prompt({**ON, "eval_citation_instruction": instruction})


def test_query_result_has_an_empty_cited_list_by_default_and_serialises_it():
    result = QueryResult(question="q", answer="a", citations=["DocA"])
    assert result.cited == [] and result.model_dump(mode="json")["cited"] == []
    assert QueryResult(question="q", answer="a", cited=["DocA"]).model_dump()["cited"] == ["DocA"]
