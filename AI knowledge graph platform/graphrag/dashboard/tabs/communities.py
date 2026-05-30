"""Tab 3: Communities — staleness badge, rebuild button, version history."""

from __future__ import annotations

from dash import Input, Output, State, callback, dash_table, html

from graphrag.dashboard.utils import CARD_STYLE, H2, _get, _post, err, http_error


def render(tenant: str) -> html.Div:
    stale_data   = _get("/kg/incremental-community/summary", {"tenant": tenant})
    history_data = _get("/community-history", {"tenant": tenant, "limit": 20})

    if e := http_error(stale_data):
        return err(f"Community data unavailable — {e}")

    summary      = stale_data or {}
    score        = summary.get("change_fraction", None)
    badge_color  = "#c00" if score is not None and score > 0.20 else "#2a9d4f"
    badge_text   = (
        f"{round(float(score or 0) * 100, 1)} % changed"
        if score is not None else "—"
    )

    history      = (history_data or {}).get("history", []) if isinstance(history_data, dict) else []
    hist_table   = dash_table.DataTable(
        data=[{k: str(v) for k, v in h.items()} for h in history],
        columns=[{"name": c, "id": c} for c in
                 ["snapshot_id", "entity_count", "edge_count", "recorded_at", "is_rebuild"]],
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px"},
        page_size=10,
    ) if history else html.P("No version history yet.", style={"color": "#888"})

    return html.Div([
        html.H2("Community Health", style=H2),
        html.Div([
            html.Div([
                html.Div(badge_text,
                         style={"fontSize": "32px", "fontWeight": "bold", "color": badge_color}),
                html.Div("Entity change fraction since last rebuild",
                         style={"fontSize": "12px", "color": "#666"}),
            ], style=CARD_STYLE),
            html.Div([
                html.Div(str(summary.get("changed_entities", "—")),
                         style={"fontSize": "32px", "fontWeight": "bold"}),
                html.Div("Changed entities", style={"fontSize": "12px", "color": "#666"}),
            ], style=CARD_STYLE),
        ], style={"display": "flex"}),
        html.Button(
            "🔄 Rebuild Affected Communities",
            id="rebuild-btn",
            style={"margin": "16px 0", "padding": "8px 20px",
                   "backgroundColor": "#1a1a2e", "color": "white",
                   "border": "none", "borderRadius": "4px", "cursor": "pointer"},
        ),
        html.Div(id="rebuild-result"),
        html.H3("Version History", style=H2),
        hist_table,
    ])


@callback(
    Output("rebuild-result", "children"),
    Input("rebuild-btn", "n_clicks"),
    State("tenant-store", "data"),
    prevent_initial_call=True,
)
def trigger_rebuild(n_clicks, tenant):
    result = _post("/kg/incremental-community/rebuild-affected",
                   {"tenant": tenant or "default", "dry_run": False})
    if e := http_error(result):
        return html.P(f"⚠ Rebuild failed — {e}", style={"color": "#c00"})
    if result:
        rebuilt = result.get("communities_rebuilt", "?")
        return html.P(f"✅ Rebuilt {rebuilt} communities.", style={"color": "#2a9d4f"})
    return html.P("⚠ Rebuild failed or timed out.", style={"color": "#c00"})
