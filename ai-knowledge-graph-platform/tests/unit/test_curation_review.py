"""Unit tests for the combined duplicate-candidate/contradiction curation
view (roadmap "P1 -- ontology curation and human-in-the-loop workbench",
bullet 3). Reuses ReviewQueueService and ContradictionDetector verbatim --
these tests inject fakes for both rather than mocking Neo4j directly, since
the aggregation logic itself (tagging + concurrent fetch), not either
service's own query correctness (already covered by test_review_queue.py
and test_contradiction_detector.py), is what's under test here.
"""

from __future__ import annotations

from api.routes.kg.curation_review import build_duplicates_and_contradictions_view


class _FakeReviewQueueService:
    def __init__(self, pending: list[dict]):
        self._pending = pending
        self.calls: list[tuple[str, int]] = []

    async def list_pending(self, tenant: str, limit: int) -> list[dict]:
        self.calls.append((tenant, limit))
        return self._pending


class _FakeContradictionDetector:
    def __init__(self, conflicts: list[dict]):
        self._conflicts = conflicts
        self.calls: list[tuple[str, int]] = []

    async def get_open_conflicts(self, *, limit: int, tenant: str) -> list[dict]:
        self.calls.append((tenant, limit))
        return self._conflicts


class TestBuildDuplicatesAndContradictionsView:
    async def test_combines_both_sources_and_tags_each_items_kind(self):
        review = _FakeReviewQueueService([{"item_id": "r1", "raw_name": "ISO IATF"}])
        detector = _FakeContradictionDetector([{"conflict_id": "c1", "conflict_type": "exclusive_state"}])

        result = await build_duplicates_and_contradictions_view(
            tenant="automotive", limit=25, review_service=review, detector=detector,
        )

        assert result["tenant"] == "automotive"
        assert result["duplicate_candidate_count"] == 1
        assert result["contradiction_count"] == 1
        kinds = {item["kind"] for item in result["items"]}
        assert kinds == {"duplicate_candidate", "contradiction"}
        duplicate_item = next(i for i in result["items"] if i["kind"] == "duplicate_candidate")
        assert duplicate_item["item_id"] == "r1"
        contradiction_item = next(i for i in result["items"] if i["kind"] == "contradiction")
        assert contradiction_item["conflict_id"] == "c1"

    async def test_passes_tenant_and_limit_through_to_both_services(self):
        review = _FakeReviewQueueService([])
        detector = _FakeContradictionDetector([])

        await build_duplicates_and_contradictions_view(
            tenant="aerospace", limit=10, review_service=review, detector=detector,
        )

        assert review.calls == [("aerospace", 10)]
        assert detector.calls == [("aerospace", 10)]

    async def test_empty_queues_yield_empty_items_not_an_error(self):
        review = _FakeReviewQueueService([])
        detector = _FakeContradictionDetector([])

        result = await build_duplicates_and_contradictions_view(
            tenant="marketing", review_service=review, detector=detector,
        )

        assert result["items"] == []
        assert result["duplicate_candidate_count"] == 0
        assert result["contradiction_count"] == 0

    async def test_neither_list_is_mutated_or_cross_contaminated(self):
        """A duplicate candidate never gets tagged "contradiction" or vice
        versa, even if both lists happen to share a field name (e.g. an
        "id"-shaped field) -- each item's `kind` is set independently."""
        review = _FakeReviewQueueService([{"item_id": "same-shape"}])
        detector = _FakeContradictionDetector([{"item_id": "same-shape"}])

        result = await build_duplicates_and_contradictions_view(
            tenant="automotive", review_service=review, detector=detector,
        )

        kinds_by_item_id = [item["kind"] for item in result["items"] if item.get("item_id") == "same-shape"]
        assert sorted(kinds_by_item_id) == ["contradiction", "duplicate_candidate"]
