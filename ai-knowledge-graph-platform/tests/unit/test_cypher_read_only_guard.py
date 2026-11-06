"""Adversarial tests for ``assert_read_only`` (graph validation read-only guard).

The guard removes comments and string literals before looking for write
clauses. Doing that with two independent regex passes lets one pass change what
the other sees: a ``//`` inside a string is not a comment, a ``'`` inside a
backtick identifier or after a backslash does not start or end a string. Every
payload below is valid Cypher by the documented grammar (strings with backslash
escapes, ``/* */`` and ``//`` comments, backtick-quoted identifiers) and
performs a write, so the guard must reject it.
"""
from __future__ import annotations

import pytest

from graphrag.graph.validation.graph_checks import WriteQueryRejected, assert_read_only

BS = chr(92)  # a backslash, built at runtime so no editor or tool rewrites escapes


@pytest.mark.parametrize("cypher", [
    # a '//' inside a string is not a comment: the old comment pass ate the rest of the line
    "MATCH (n {u:'http://x/y'}) CREATE (m:X)",
    'MATCH (n {u:"http://x/y"}) DETACH DELETE n',
    # a '//' inside a block comment hid whatever followed on that line
    "MATCH (n) /* a // b */ CREATE (m:X)",
    # a comment is whitespace to the engine: it must not glue two words together
    "LOAD/**/CSV FROM 'file:///x.csv' AS r RETURN r",
    "CALL/**/db.createLabel('X')",
    # a backslash-escaped quote does not end the string, but the old pass thought it did
    "MATCH (n) WHERE n.a = '" + BS + "'' CREATE (m:X) RETURN '" + BS + "''",
    # a quote inside a backtick identifier does not start a string
    "MATCH (`a'b`) CREATE (m:X) RETURN `a'b`",
    # plain writes keep being rejected
    "MATCH (n) CREATE (m:X)",
    "MATCH (n) SET n.x = 1",
    "MATCH (n) REMOVE n.x",
    "UNWIND [1] AS i FOREACH (x IN [1] | SET x = 1)",
])
def test_write_payloads_are_rejected(cypher):
    with pytest.raises(WriteQueryRejected):
        assert_read_only(cypher)


@pytest.mark.parametrize("cypher", [
    "MATCH (n) RETURN n",
    "MATCH (n {u:'http://x/y'}) RETURN n",
    "MATCH (n) WHERE n.note = 'please CREATE a ticket' RETURN n",
    "MATCH (n) WHERE n.note = 'it" + BS + "'s a MERGE of two things' RETURN n",
    "MATCH (n) /* CREATE nothing here */ RETURN n",
    "MATCH (n) RETURN n // SET is only mentioned in this comment",
    "MATCH (n) RETURN n.created_at, n.set_by, n.deleted",
    "MATCH (n) RETURN n.`create` AS c",
    "MATCH (n) RETURN n.`set` AS s",
])
def test_read_queries_are_accepted(cypher):
    assert_read_only(cypher)


def test_unterminated_string_or_comment_does_not_hide_a_write():
    """An unterminated literal is a syntax error for the engine; the guard must
    still not treat the rest of the text as safe."""
    for cypher in ["MATCH (n) RETURN 'abc CREATE (m:X)", "MATCH (n) /* open CREATE (m:X)"]:
        with pytest.raises(WriteQueryRejected):
            assert_read_only(cypher)
