from graphrag.retrieval.answer_policy import answer_prompt, apply_answer_policy


def test_generic_answer_policy_contains_no_aerospace_specific_language():
    prompt = answer_prompt({"answer_policy": "generic"})

    assert "FAA" not in prompt
    assert "airworthy" not in prompt
    assert "revision labels" not in prompt


def test_aerospace_policy_is_opt_in_and_keeps_existing_grounding_repair():
    answer, citations = apply_answer_policy(
        "The aircraft was unairworthy.",
        "The aircraft remains airworthy. FAA-AD-2024-01-02 applies.",
        "Which AD applies?", [], ["FAA-AD-2024-01-02.txt"],
        {"answer_policy": "aerospace_regulatory"},
    )

    assert "airworthy" in answer
    assert "FAA-AD-2024-01-02" in citations


def test_generic_policy_does_not_apply_aerospace_post_processing():
    answer, citations = apply_answer_policy(
        "The aircraft was unairworthy.", "The aircraft remains airworthy.",
        "Which AD applies?", ["source"], [], {"answer_policy": "generic"},
    )

    assert answer == "The aircraft was unairworthy."
    assert citations == ["source"]
