"""graphrag.dashboard.tabs.review_queue -- no existing dashboard-tab test file
to mirror (conflicts.py, the closest analog, has none either), so this tests
the tab's pure render()/_decide() logic directly against a monkeypatched
httpx layer (_get/_post) rather than a live dashboard or browser session.
"""
from __future__ import annotations

from dash import html

from graphrag.dashboard.tabs import review_queue

ITEMS = [
    {"item_id": "r-1", "raw_name": "ISO IATF", "raw_type": "CONCEPT",
     "candidate_name": "IATF 16949:2016", "candidate_type": "CONCEPT",
     "score": 0.79, "match_type": "fuzzy", "source_doc": "doc.pdf"},
]


class TestRender:
    def test_pending_items_render_a_table_and_action_panel(self, monkeypatch):
        monkeypatch.setattr(review_queue, "_get", lambda path, params=None: {"items": ITEMS})
        result = review_queue.render("aerospace")
        assert isinstance(result, html.Div)
        # section_title + table + action_panel = 3 top-level children.
        assert len(result.children) == 3

    def test_empty_queue_renders_the_all_clear_message(self, monkeypatch):
        monkeypatch.setattr(review_queue, "_get", lambda path, params=None: {"items": []})
        result = review_queue.render("aerospace")
        text = str(result)
        assert "No pending items" in text

    def test_http_error_without_demo_mode_shows_error(self, monkeypatch):
        monkeypatch.setattr(review_queue, "_get", lambda path, params=None: {"_http_error": "HTTP 503"})
        monkeypatch.setattr(review_queue, "DEMO_MODE", False)
        result = review_queue.render("aerospace")
        assert "unavailable" in str(result)

    def test_http_error_with_demo_mode_falls_back_to_demo_data(self, monkeypatch):
        monkeypatch.setattr(review_queue, "_get", lambda path, params=None: {"_http_error": "HTTP 503"})
        monkeypatch.setattr(review_queue, "DEMO_MODE", True)
        result = review_queue.render("aerospace")
        # Demo fixture is non-empty, so this should render the table, not the error.
        assert "unavailable" not in str(result)


class TestDecide:
    def test_no_selection_prompts_to_select_a_row(self):
        assert review_queue._decide(1, [], [], "approve") == "Select a row first."

    def test_no_click_yet_prompts_to_select_a_row(self):
        assert review_queue._decide(None, [0], ITEMS, "approve") == "Select a row first."

    def test_approve_posts_to_the_correct_endpoint_and_reports_success(self, monkeypatch):
        calls = []

        def fake_post(path, json=None):
            calls.append(path)
            return {"item_id": "r-1", "status": "approved"}

        monkeypatch.setattr(review_queue, "_post", fake_post)
        result = review_queue._decide(1, [0], ITEMS, "approve")
        assert calls == ["/kg/review-queue/r-1/approve?reviewed_by=admin_ui"]
        assert "approved" in result

    def test_reject_posts_to_the_correct_endpoint_and_reports_success(self, monkeypatch):
        monkeypatch.setattr(review_queue, "_post", lambda path, json=None: {"item_id": "r-1", "status": "rejected"})
        result = review_queue._decide(1, [0], ITEMS, "reject")
        assert "rejected" in result

    def test_backend_error_in_response_body_is_surfaced(self, monkeypatch):
        monkeypatch.setattr(review_queue, "_post", lambda path, json=None: {"error": "Item not found"})
        result = review_queue._decide(1, [0], ITEMS, "approve")
        assert "Item not found" in result

    def test_http_error_is_surfaced(self, monkeypatch):
        monkeypatch.setattr(review_queue, "_post", lambda path, json=None: {"_http_error": "HTTP 500"})
        result = review_queue._decide(1, [0], ITEMS, "approve")
        assert "failed" in result.lower()
