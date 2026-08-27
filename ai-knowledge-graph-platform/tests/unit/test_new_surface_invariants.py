"""Generated-input guards for newly added integration boundaries."""

from hypothesis import given, settings
from hypothesis import strategies as st

from graphrag.retrieval.answer_policy import answer_prompt
from graphrag.ingestion.relational import EntityTableMapping

FAST = settings(max_examples=60, deadline=None)


class TestAnswerPolicySelection:
    @FAST
    @given(policy=st.text(max_size=100))
    def test_only_the_explicit_aerospace_policy_can_enable_domain_rules(self, policy):
        prompt = answer_prompt({"answer_policy": policy})
        if policy == "aerospace_regulatory":
            assert "revision labels" in prompt
        else:
            assert "revision labels" not in prompt


class TestRelationalMappingFuzz:
    @FAST
    @given(
        prefix=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", max_size=12),
        unsafe=st.sampled_from([" ", "-", ";", "\"", "'", "/", "\n"]),
        suffix=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", max_size=12),
    )
    def test_non_identifier_columns_are_rejected_before_connector_access(self, prefix, unsafe, suffix):
        name = prefix + unsafe + suffix
        try:
            EntityTableMapping(table="safe", entity_type="ITEM", id_column=name, name_column="name")
        except ValueError:
            return
        raise AssertionError(f"unsafe identifier accepted: {name!r}")
