"""Tab 2: Conflicts — open Conflict nodes table + resolve action."""

from __future__ import annotations

from dash import Input, Output, State, callback, dash_table, dcc, html

from graphrag.dashboard.utils import H2, _get, _post, err, http_error


def render(tenant: str) -> html.Div:
    data = _get("/corrections/list-conflicts", {"tenant": tenant, "limit": 100})
    if e := http_error(data):
        return err(f"Conflicts unavailable — {e}")
    conflicts = data if isinstance(data, list) else (data or {}).get("conflicts", [])

    if not conflicts:
        return html.Div([
            html.H2("Open Conflicts", style=H2),
            html.P("✅ No open conflicts.", style={"color": "#2a9d4f"}),
        ])

    cols = ["conflict_id", "type", "src", "tgt", "relation", "tenant"]
    table = dash_table.DataTable(
        id="conflicts-table",
        data=[{k: str(c.get(k, "")) for k in cols} for c in conflicts],
        columns=[{"name": c, "id": c} for c in cols],
        row_selectable="single",
        selected_rows=[],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
        style_header={"fontWeight": "bold"},
        page_size=20,
    )

    resolve_panel = html.Div([
        html.H3("Resolve selected conflict", style={"marginTop": "20px"}),
        html.Div([
            dcc.Dropdown(
                id="resolve-resolution",
                options=[
                    {"label": "Resolved — manual",    "value": "resolved_manual"},
                    {"label": "Resolved — authority", "value": "resolved_authority"},
                    {"label": "False positive",        "value": "false_positive"},
                ],
                placeholder="Resolution type...",
                style={"width": "300px"},
            ),
            dcc.Input(id="resolve-winner-doc", type="text",
                      placeholder="Winner doc ID (optional)",
                      style={"marginLeft": "12px", "padding": "6px", "width": "240px"}),
            html.Button("Resolve", id="resolve-btn",
                        style={"marginLeft": "12px", "padding": "6px 16px",
                               "backgroundColor": "#1a1a2e", "color": "white",
                               "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(id="resolve-result", style={"marginTop": "10px", "color": "#2a9d4f"}),
    ])

    return html.Div([
        html.H2(f"Open Conflicts ({len(conflicts)})", style=H2),
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
