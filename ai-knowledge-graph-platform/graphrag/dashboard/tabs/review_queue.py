"""Tab 6: Review Queue — pending ambiguous alias matches + approve/reject action.

/kg/review-queue (graphrag/graph/review_queue.py, api/routes/kg/review_queue.py)
has been a working read+write API since the ambiguous-match review flow was
built (see docs/entity-resolution.md's Stage 2/3 review-band description) but
had no dashboard tab, unlike conflicts/health/communities/gdpr/calibration
which all do -- an operator could only act on it via curl/Postman. This tab
closes that gap, mirroring tabs/conflicts.py's table + action-panel pattern
exactly (httpx to the REST API, a selectable table, buttons that POST and
report the result inline).
"""

from __future__ import annotations

from dash import Input, Output, State, callback, html

from graphrag.dashboard import demo_data
from graphrag.dashboard.utils import (
    DEMO_MODE, GOOD, NAV, TEAL, _get, _post, card_panel, err, http_error,
    section_title, themed_table,
)


def render(tenant: str) -> html.Div:
    data = _get("/kg/review-queue", {"tenant": tenant, "limit": 100})
    if e := http_error(data):
        if not DEMO_MODE:
            return err(f"Review queue unavailable — {e}")
        data = demo_data.REVIEW_QUEUE
    items = data if isinstance(data, list) else (data or {}).get("items", [])

    if not items:
        return html.Div([
            section_title("Review Queue", "Ambiguous alias matches awaiting a human decision"),
            html.Div("✓ No pending items — every recent match was resolved automatically.",
                     style={"color": GOOD, "fontWeight": "600", "fontSize": "14px",
                            "padding": "14px 16px", "background": "#E8F6EC",
                            "borderRadius": "10px"}),
        ])

    cols = ["item_id", "raw_name", "candidate_name", "score", "match_type", "source_doc"]
    table = themed_table(
        data=[{k: str(i.get(k, "")) for k in cols} for i in items],
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols],
        id="review-queue-table",
        row_selectable="single",
        selected_rows=[],
        page_size=20,
    )

    action_panel = card_panel(html.Div([
        html.Div("Decide selected match",
                 style={"fontSize": "15px", "fontWeight": "700", "color": NAV,
                        "marginBottom": "12px"}),
        html.Div([
            html.Button("Approve — same entity", id="review-approve-btn",
                        style={"padding": "9px 22px",
                               "background": f"linear-gradient(135deg,{TEAL},{NAV})",
                               "color": "white", "border": "none", "borderRadius": "8px",
                               "cursor": "pointer", "fontWeight": "700"}),
            html.Button("Reject — keep separate", id="review-reject-btn",
                        style={"marginLeft": "12px", "padding": "9px 22px",
                               "background": "white", "color": NAV,
                               "border": f"1px solid {NAV}", "borderRadius": "8px",
                               "cursor": "pointer", "fontWeight": "700"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(id="review-queue-result", style={"marginTop": "12px", "color": GOOD,
                                                    "fontWeight": "600"}),
    ]))

    return html.Div([
        section_title(f"Review Queue ({len(items)})",
                      "Ambiguous alias matches awaiting a human decision"),
        table,
        action_panel,
    ])


def _decide(n_clicks, selected_rows, rows, action: str):
    if not n_clicks or not selected_rows:
        return "Select a row first."
    selected = selected_rows[0]
    # AG Grid returns selected row records. Keep integer support for direct
    # helper tests and legacy callback-replay payloads.
    selected_row = selected if isinstance(selected, dict) else rows[selected]
    item_id = selected_row.get("item_id", "")
    # tenant is resolved server-side from the auth token (get_tenant), never
    # client-supplied here -- see api/routes/kg/review_queue.py. reviewed_by
    # is a query param on that route, not a body field, so it's appended to
    # the path rather than sent as JSON (there is no body model to receive
    # it).
    result = _post(f"/kg/review-queue/{item_id}/{action}?reviewed_by=admin_ui")
    if e := http_error(result):
        return f"⚠ {action.title()} failed — {e}"
    if result and result.get("error"):
        return f"⚠ {result['error']}"
    verb = "approved" if action == "approve" else "rejected"
    return f"✅ Item {item_id[:8]}… {verb}."


@callback(
    Output("review-queue-result", "children"),
    Input("review-approve-btn", "n_clicks"),
    Input("review-reject-btn", "n_clicks"),
    State("review-queue-table", "selectedRows"),
    State("review-queue-table", "rowData"),
    prevent_initial_call=True,
)
def decide_review_item(approve_clicks, reject_clicks, selected_rows, rows):
    from dash import ctx

    triggered = ctx.triggered_id
    if triggered == "review-approve-btn":
        return _decide(approve_clicks, selected_rows, rows, "approve")
    if triggered == "review-reject-btn":
        return _decide(reject_clicks, selected_rows, rows, "reject")
    return ""
