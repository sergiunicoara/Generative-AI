"""Strict citation normaliser, precision/recall and the [DOC: id] parser (offline, no model, no database)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from graphrag.evaluation import citations as cit  # noqa: E402

AD = "FAA-AD-2024-01-02"
OLD_AD = "FAA-AD-2022-03-07"
SWA = "SWA_fleet_registry_2024"


@pytest.mark.parametrize("raw, canonical", [
    ("FAA-AD-2024-01-02", AD),                  # canonical id itself
    ("FAA AD 2024-01-02", AD),                  # surface form written in the corpus text
    ("AD 2024-01-02", AD),                      # bare AD number
    ("faa‑ad‑2024‑01‑02", AD),                  # case and U+2011 hyphens
    ("SWA_fleet_registry_2024.txt", SWA),       # file name
    ("swa fleet registry 2024", SWA),
    ("D6-87700-100", "Boeing_MCAS_SWChangeRecord"),   # document number
])
def test_aliases_map_to_one_canonical_id(raw, canonical):
    assert cit.strict_citations([raw]) == [canonical]


@pytest.mark.parametrize("junk", [
    "Aircraft", "Appliances", "Propellers", "Boeing 737 MAX", "737 NG Legacy Fleet",
    "G-ABCD",                    # tail number: mentioned by two documents, so it is not an alias
    "Boeing", "AD 2024-01",      # a prefix of an alias is not the alias (no substring matching)
    "the FAA-AD-2024-01-02 directive",   # nor is a phrase that merely contains one
    "",
])
def test_strings_that_do_not_name_a_document_are_dropped(junk):
    assert cit.strict_citations([junk]) == []


def test_ids_are_deduplicated_keeping_first_seen_order():
    raw = ["AD 2024-01-02", "Aircraft", OLD_AD, "FAA AD 2024-01-02", "FAA-AD-2024-01-02", "Propellers"]
    assert cit.strict_citations(raw) == [AD, OLD_AD]


def test_the_committed_alias_map_is_unambiguous_and_covers_every_golden_expected_citation():
    golden = json.loads((ROOT / "evals" / "golden_set.json").read_text(encoding="utf-8"))
    index = cit.build_alias_index(json.loads(cit.ALIAS_MAP_PATH.read_text(encoding="utf-8"))["documents"])
    expected = {c for q in golden["questions"] for c in q.get("expected_citations", [])}
    assert expected <= set(index.values())
    assert len(cit.alias_map_sha256()) == 64


def test_an_alias_denoting_two_documents_is_a_data_error():
    with pytest.raises(ValueError, match="denotes both"):
        cit.build_alias_index({"A": ["shared name"], "B": ["Shared  Name"]})


QUESTION = {"expected_citations": [AD, OLD_AD]}


def test_recall_and_precision_use_canonical_ids_only_so_over_citing_is_penalised():
    cited = cit.strict_citations(["AD 2024-01-02", "FAA-AD-2020-05-11", SWA])
    assert cit.recall(QUESTION, cited) == 0.5
    assert cit.precision(QUESTION, cited) == pytest.approx(1 / 3)
    assert cit.precision(QUESTION, [AD, OLD_AD]) == 1.0


def test_precision_and_recall_are_undefined_without_expected_citations_and_precision_without_citations():
    assert cit.recall({}, [AD]) is None and cit.precision({}, [AD]) is None
    assert cit.precision(QUESTION, []) is None and cit.recall(QUESTION, []) == 0.0


def test_citation_failure_applies_the_golden_threshold_to_canonical_ids():
    assert cit.citation_failure(QUESTION, [AD, OLD_AD]) is None
    message = cit.citation_failure(QUESTION, [AD])
    assert message and message.startswith(cit.CITATION_FAILURE) and OLD_AD in message
    assert cit.citation_failure({"expected_citations": [AD, OLD_AD], "min_citation_recall": 0.5}, [AD]) is None
    assert cit.citation_failure({}, []) is None


def test_f1_is_per_question_and_macro_averaging_differs_from_f1_of_the_means():
    two = {"expected_citations": [AD, OLD_AD]}
    one = {"expected_citations": [AD]}
    assert cit.f1(two, [AD, OLD_AD]) == 1.0
    assert cit.f1(two, [AD, SWA]) == pytest.approx(0.5)        # P=0.5, R=0.5
    assert cit.f1(one, [AD, OLD_AD, SWA]) == pytest.approx(0.5)  # P=1/3, R=1
    assert cit.f1(one, []) == 0.0 and cit.f1(one, [SWA]) == 0.0 and cit.f1({}, [AD]) is None
    macro = (cit.f1(two, [AD, SWA]) + cit.f1(one, [AD, OLD_AD, SWA])) / 2
    mean_p = (cit.precision(two, [AD, SWA]) + cit.precision(one, [AD, OLD_AD, SWA])) / 2
    mean_r = (cit.recall(two, [AD, SWA]) + cit.recall(one, [AD, OLD_AD, SWA])) / 2
    assert macro == pytest.approx(0.5) and 2 * mean_p * mean_r / (mean_p + mean_r) != pytest.approx(macro)


@pytest.mark.parametrize("answer, cleaned, ids", [
    ("It supersedes AD 2022 [DOC: FAA-AD-2024-01-02].", "It supersedes AD 2022.", [AD]),
    ("A [DOC: FAA-AD-2024-01-02, FAA-AD-2022-03-07] and B [doc: SWA_fleet_registry_2024]",
     "A and B", [AD, OLD_AD, SWA]),
    ("No tags here.", "No tags here.", []),
    ("Empty tag [DOC: ] stays harmless.", "Empty tag stays harmless.", []),
])
def test_doc_tag_parser_extracts_ids_and_strips_the_tags(answer, cleaned, ids):
    assert cit.parse_doc_tags(answer) == (cleaned, ids)


def test_the_instruction_lists_the_ids_and_is_identical_for_the_same_ids():
    text = cit.citation_instruction([AD, SWA])
    assert "[DOC: <id>]" in text and f"  - {AD}\n  - {SWA}\n" in text
    assert text == cit.citation_instruction([AD, SWA]) and text != cit.citation_instruction([AD])
    assert cit.sha256_text(text) == cit.sha256_text(cit.citation_instruction([AD, SWA]))


def test_the_parser_reads_ids_the_prompt_asks_for_and_the_normaliser_maps_or_drops_them():
    answer = "It supersedes it [DOC: FAA-AD-2024-01-02]; also [DOC: Aircraft] and [DOC: AD 2022-03-07, SWA_fleet_registry_2024]."
    _, ids = cit.parse_doc_tags(answer)
    assert ids == ["FAA-AD-2024-01-02", "Aircraft", "AD 2022-03-07", "SWA_fleet_registry_2024"]
    assert cit.strict_citations(ids) == [AD, OLD_AD, SWA]


def test_set_metrics_of_none_means_the_set_does_not_exist_and_of_empty_means_nothing_cited():
    question = {"expected_citations": [AD]}
    assert cit.set_metrics(question, None) == {
        "recall_raw": None, "recall_citable": None, "precision": None, "f1": None, "n_distinct": None,
        "n_expected_uncitable": 0}
    assert cit.set_metrics(question, [], [AD]) == {
        "recall_raw": 0.0, "recall_citable": 0.0, "precision": None, "f1": 0.0, "n_distinct": 0,
        "n_expected_uncitable": 0}
    both = cit.set_metrics(question, [AD, SWA])
    assert both["recall_raw"] == both["recall_citable"] == 1.0 and both["precision"] == 0.5
    assert both["f1"] == pytest.approx(2 / 3) and both["n_distinct"] == 2


def test_recall_citable_ignores_expected_ids_the_prompt_could_not_offer_and_raw_recall_does_not():
    question = {"expected_citations": [AD, OLD_AD]}
    citable = [AD, SWA]                                  # OLD_AD (superseded) is not in the instruction
    metrics = cit.set_metrics(question, [AD], citable, "citable")
    assert metrics["recall_raw"] == 0.5 and metrics["recall_citable"] == 1.0
    assert metrics["n_expected_uncitable"] == 1 and metrics["precision"] == 1.0 and metrics["f1"] == 1.0
    assert cit.set_metrics(question, [AD], citable, "raw")["f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    only_uncitable = cit.set_metrics({"expected_citations": [OLD_AD]}, [AD], citable, "citable")
    assert only_uncitable["recall_citable"] is None and only_uncitable["f1"] is None
    assert only_uncitable["n_expected_uncitable"] == 1


def test_bare_exact_canonical_ids_are_read_and_nothing_fuzzy_is():
    parsed = cit.parse_answer_tags(
        "A [FAA-ad-2024-01-02] B [DOC: SWA_fleet_registry_2024]. C [AD 2024-01-02] D [Note] "
        "E [FAA-AD-2024-01-0] F [14CFR_Part39_excerpt].")
    assert parsed["ids"] == ["FAA-ad-2024-01-02", "SWA_fleet_registry_2024", "14CFR_Part39_excerpt"]
    assert (parsed["n_prefixed"], parsed["n_bare"]) == (1, 2)
    # an alias, a different word and a truncated id are not canonical ids: left in the text, not read
    assert parsed["answer"] == "A B. C [AD 2024-01-02] D [Note] E [FAA-AD-2024-01-0] F."
    assert cit.strict_citations(parsed["ids"]) == [AD, SWA, "14CFR_Part39_excerpt"]
    assert cit.parse_doc_tags("x [FAA-AD-2024-01-02]") == ("x", ["FAA-AD-2024-01-02"])


def test_the_accepted_forms_are_declared_for_the_run_metadata():
    assert len(cit.ACCEPTED_TAG_FORMS) == 2 and "bare" in cit.ACCEPTED_TAG_FORMS[1]


def test_citation_failure_on_the_citable_basis_ignores_expected_ids_the_prompt_never_offered():
    question = {"expected_citations": [AD, OLD_AD]}
    citable = [AD, SWA]
    assert cit.citation_failure(question, [AD]) is not None                 # raw: OLD_AD missing
    assert cit.citation_failure(question, [AD], citable) is None            # citable: only AD is required
    assert cit.citation_failure(question, [SWA], citable) is not None       # the citable one is still required
    assert cit.citation_failure({"expected_citations": [OLD_AD]}, [], citable) is None   # nothing citable expected
