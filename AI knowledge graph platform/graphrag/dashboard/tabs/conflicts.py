"""Tab 2: Conflicts — open Conflict nodes table + resolve action."""

from __future__ import annotations

from dash import Input, Output, State, callback, dcc, html

from graphrag.dashboard.utils import (
    BAD, FONT, GOOD, NAV, TEAL, _get, _post, card_panel, err, http_error,
    section_title, themed_table,
)


def render(tenant: str) -> html.Div:
    data = _get("/corrections/list-conflicts", {"tenant": tenant, "limit": 100})
    if e := http_error(data):
        return err(f"Conflicts unavailable — {e}")
    conflicts = data if isinstance(data, list) else (data or {}).get("conflicts", [])

    if not conflicts:
        return html.Div([
            section_title("Open Conflicts", "Contradictory facts awaiting resolution"),
            html.Div("✓ No open conflicts — the graph is internally consistent.",
                     style={"color": GOOD, "fontWeight": "600", "fontSize": "14px",
                            "padding": "14px 16px", "background": "#E8F6EC",
                            "borderRadius": "10px"}),
        ])

    cols = ["conflict_id", "type", "src", "tgt", "relation", "tenant"]
    table = themed_table(
        data=[{k: str(c.get(k, "")) for k in cols} for c in conflicts],
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols],
        id="conflicts-table",
        row_selectable="single",
        selected_rows=[],
        page_size=20,
    )

    resolve_panel = card_panel(html.Div([
        html.Div("Resolve selected conflict",
                 style={"fontSize": "15px", "fontWeight": "700", "color": NAV,
                        "marginBottom": "12px"}),
        html.Div([
            dcc.Dropdown(
                id="resolve-resolution",
                options=[
                    {"label": "Resolved — manual",    "value": "resolved_manual"},
                    {"label": "Resolved — authority", "value": "resolved_authority"},
                    {"label": "False positive",        "value": "false_positive"},
                ],
                placeholder="Resolution type...",
                style={"width": "300px", "fontFamily": FONT},
            ),
            dcc.Input(id="resolve-winner-doc", type="text",
                      placeholder="Winner doc ID (optional)",
                      style={"marginLeft": "12px", "padding": "8px 12px", "width": "240px",
                             "borderRadius": "8px", "border": "1px solid #DCE5F3",
                             "fontFamily": FONT}),
            html.Button("Resolve", id="resolve-btn",
                        style={"marginLeft": "12px", "padding": "9px 22px",
                               "background": f"linear-gradient(135deg,{TEAL},{NAV})",
                               "color": "white", "border": "none", "borderRadius": "8px",
                               "cursor": "pointer", "fontWeight": "700", "fontFamily": FONT}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(id="resolve-result", style={"marginTop": "12px", "color": GOOD,
                                              "fontWeight": "600"}),
    ]))

    return html.Div([
        section_title(f"Open Conflicts ({len(conflicts)})",
                      "Contradictory facts detected across source documents"),
        table,
        resolve_panel,
    ])


@callback(
    Output("resolve-result", "children"),
    Input("resolve-btn", "n_clicks"),
    State("conflicts-table", "selected_rows"),
    State("conflicts-table", "data"),
    State("resolve-resolution", "value"),
    State("resolve-winner-doc", "value"),
    prevent_initial_call=True,
)
def resolve_conflict(n_clicks, selected_rows, rows, resolution, winner_doc):
    if not selected_rows or not resolution:
        return "Select a row and choose a resolution type."
    conflict_id = rows[selected_rows[0]].get("conflict_id", "")
    result = _post("/corrections/resolve-conflict", {
        "conflict_id":   conflict_id,
        "resolution":    resolution,
        "winner_doc_id": winner_doc or "",
        "resolved_by":   "admin_ui",
    })
    if e := http_error(result):
        return f"⚠ Resolve failed — {e}"
    if result:
        return f"✅ Conflict {conflict_id[:8]}… resolved as '{resolution}'."
    return "⚠ Failed to resolve conflict."
