"""Unit tests for the agentic-fallback gate (_is_low_confidence).

The strict gate (hedge AND no citations) keeps the fallback trigger rate low,
but has a blind spot on multi-hop questions: retrieval returns related chunks
(non-empty citations) without the bridging document, the answer hedges, and the
IRCoT agent never fires. `require_no_citations=False` opts a tenant into the
looser hedge-only gate.
"""

from __future__ import annotations

from graphrag.retrieval.agentic_retriever import _is_low_confidence


class TestStrictGateIsDefault:
    """Default behavior must be unchanged: hedge AND no citations."""

    def test_hedge_with_no_citations_triggers(self):
        assert _is_low_confidence("The context does not specify.", citations=[])

    def test_hedge_with_citations_does_not_trigger(self):
        # The blind spot: retrieval found something, just not the answer.
        assert not _is_low_confidence(
            "The context does not specify which airlines are affected.",
            citations=["Boeing_MCAS_SWChangeRecord"],
        )

    def test_confident_answer_with_no_citations_does_not_trigger(self):
        # Confident answers on freshly-ingested corpora often lack citation IDs;
        # the AND gate exists so these don't trigger the agent.
        assert not _is_low_confidence("Southwest Airlines is affected.", citations=[])

    def test_confident_answer_with_citations_does_not_trigger(self):
        assert not _is_low_confidence(
            "Southwest Airlines is affected.", citations=["SWA_fleet_registry_2024"]
        )


class TestHedgeOnlyGate:
    """require_no_citations=False: the hedge signal alone is enough."""

    def test_hedge_with_citations_now_triggers(self):
        # The case the strict gate misses — this is the whole point of the knob.
        assert _is_low_confidence(
            "The context does not specify which airlines are affected.",
            citations=["Boeing_MCAS_SWChangeRecord", "FAA-AD-2020-05-11"],
            require_no_citations=False,
        )

    def test_hedge_with_no_citations_still_triggers(self):
        assert _is_low_confidence(
            "The context does not specify.", citations=[], require_no_citations=False
        )

    def test_confident_answer_still_does_not_trigger(self):
        # Loosening the citation condition must NOT make confident answers fire.
        assert not _is_low_confidence(
            "Southwest Airlines operates the Boeing 737 MAX 8.",
            citations=["SWA_fleet_registry_2024"],
            require_no_citations=False,
        )

    def test_confident_answer_no_citations_still_does_not_trigger(self):
        assert not _is_low_confidence(
            "Southwest Airlines operates the Boeing 737 MAX 8.",
            citations=[],
            require_no_citations=False,
        )


class TestHedgeSignals:
    """Representative hedge phrasings the aerospace corpus actually produces."""

    def test_recognized_hedges(self):
        for answer in (
            "The context does not specify which airlines are affected.",
            "There is not enough information to answer.",
            "I don't know.",
            "No relevant documents were found.",
            "Insufficient context to answer this question.",
        ):
            assert _is_low_confidence(answer, citations=["doc"], require_no_citations=False), answer
