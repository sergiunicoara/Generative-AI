"""Red-team coverage for the SPARQL keyword guard in graphrag/graph/sparql_bridge.py.

``_reject_unsafe_sparql``/``_reject_unsafe_update`` are regex-based text
scanning, not a real SPARQL parser -- tests/unit/test_sparql_bridge.py's
``TestSPARQLBridgeUpdateGuardPrecision`` already proves a few IRI/variable
disguise cases for the *update* guard specifically. This file goes looking
for bypasses across both guards deliberately, the way an attacker would:
can a real SERVICE/write clause be smuggled past the scanned copy while
staying fully intact in the string actually handed to rdflib?

That search found a genuine, confirmed bypass (comments were stripped
*before* strings, so a ``#`` legitimately inside a string literal --
``"Room #101"`` -- hid everything after it on the same line from the scan,
including a real ``SERVICE`` clause placed right after the string closed).
It is fixed in sparql_bridge.py's ``_blank_strings_and_comments`` (strings
now blanked first); the fix is exercised here, not just asserted by
reading the diff.
"""

from __future__ import annotations

import pytest

from graphrag.graph.sparql_bridge import _reject_unsafe_sparql, _reject_unsafe_update


class TestHashInStringNoLongerHidesAForbiddenClause:
    """The actual bypass found while writing this file."""

    def test_query_guard_still_catches_service_after_a_hashed_string(self):
        payload = (
            'SELECT * WHERE { ?s ?p "Room #101" . '
            "SERVICE <http://169.254.169.254/latest/meta-data/> {?a ?b ?c} }"
        )
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql(payload)

    def test_update_guard_still_catches_service_after_a_hashed_string(self):
        payload = (
            'INSERT DATA { <a> <b> "Room #101" } ; '
            "SERVICE <http://169.254.169.254/latest/meta-data/> {?a ?b ?c}"
        )
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_update(payload)

    def test_update_guard_still_catches_load_after_a_hashed_string(self):
        # DROP itself is a permitted update form (_ALLOWED_UPDATE_FORMS) --
        # only SERVICE/LOAD are forbidden for updates, since they're the
        # outbound-HTTP-request primitives, not local-graph mutation.
        payload = 'INSERT DATA { <a> <b> "Item #7" } ; LOAD <http://evil/>'
        with pytest.raises(ValueError, match="LOAD"):
            _reject_unsafe_update(payload)

    def test_a_hash_inside_a_string_is_not_treated_as_a_comment_at_all(self):
        """The narrowest possible regression pin: a lone hashed string with
        nothing dangerous after it must still be accepted -- proves the fix
        didn't overcorrect into rejecting ordinary data."""
        _reject_unsafe_sparql('SELECT * WHERE { ?s ?p "Room #101" }')

    def test_multiple_hashes_in_one_string_before_a_forbidden_clause(self):
        payload = 'SELECT * WHERE { ?s ?p "C# and F# both use #" . SERVICE <http://evil/> {?a ?b ?c} }'
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql(payload)


class TestLongStringsDoNotHideOrLeakAForbiddenClause:
    def test_triple_quoted_string_containing_a_hash_is_still_safe_alone(self):
        payload = 'SELECT * WHERE { ?s rdfs:label """multi\nline # not a comment\ntext""" }'
        _reject_unsafe_sparql(payload)  # must not raise

    def test_forbidden_clause_after_a_triple_quoted_string_is_still_caught(self):
        payload = (
            'SELECT * WHERE { ?s rdfs:label """multi\nline # not a comment\ntext""" . '
            "SERVICE <http://evil/> {?a ?b ?c} }"
        )
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql(payload)

    def test_triple_quoted_string_containing_the_literal_word_service(self):
        """A legitimate label/description containing the word "service" as
        ordinary text must not be rejected -- this is the false-positive
        direction, not a security concern, but worth pinning down since the
        long-string pattern is new."""
        payload = 'SELECT * WHERE { ?s rdfs:comment """Customer Service Center""" }'
        _reject_unsafe_sparql(payload)  # must not raise


class TestOrdinaryLiteralsContainingForbiddenWordsAreNotFalsePositives:
    def test_short_string_containing_the_word_service(self):
        _reject_unsafe_sparql('SELECT * WHERE { ?s ?p "has a bare SERVICE word" }')

    def test_short_string_containing_the_word_drop(self):
        _reject_unsafe_sparql('SELECT * WHERE { ?s ?p "please drop by the office" }')

    def test_trailing_real_comment_is_still_stripped(self):
        _reject_unsafe_sparql("SELECT * WHERE { ?s ?p ?o } # SERVICE mentioned only in a comment")


class TestKeywordCannotBeSplitAcrossACommentOrStringToEvadeDetectionOrParsing:
    """These aren't bypasses -- a SPARQL keyword can't be split by a comment
    or string and still tokenize as that keyword to rdflib's own parser
    either -- but they pin down that the guard's behavior (reject, for lack
    of a recognized form) matches "this input cannot do anything", not "the
    guard was fooled into passing something dangerous"."""

    def test_select_split_by_a_newline_mid_keyword_is_rejected(self):
        with pytest.raises(ValueError, match="Only"):
            _reject_unsafe_sparql("SEL\nECT * WHERE { ?s ?p ?o }")

    def test_select_split_by_a_comment_mid_keyword_is_rejected(self):
        with pytest.raises(ValueError, match="Only"):
            _reject_unsafe_sparql("SEL#not a real split\nECT * WHERE { ?s ?p ?o }")


class TestCaseAndNestingDoNotEvadeTheForbiddenCheck:
    def test_lowercase_service_is_still_caught(self):
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql("SELECT * WHERE { service <http://evil/> {?a ?b ?c} }")

    def test_mixed_case_service_is_still_caught(self):
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql("SELECT * WHERE { SeRvIcE <http://evil/> {?a ?b ?c} }")

    def test_service_nested_inside_a_subquery_is_still_caught(self):
        payload = (
            "SELECT * WHERE { { SELECT * WHERE { "
            "SERVICE <http://169.254.169.254/> {?a ?b ?c} } } }"
        )
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql(payload)

    def test_load_is_still_caught_case_insensitively(self):
        with pytest.raises(ValueError, match="LOAD"):
            _reject_unsafe_sparql("SELECT * WHERE { ?s ?p ?o } ; load <http://evil/>")
