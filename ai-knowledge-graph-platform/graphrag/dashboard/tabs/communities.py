"""Tab 3: Communities — staleness badge, rebuild button, version history."""

from __future__ import annotations

from dash import Input, Output, State, callback, html

from graphrag.dashboard import demo_data
from graphrag.dashboard.utils import (
    BAD, DEMO_MODE, FONT, GOOD, NAV, TEAL, _get, _post, err, http_error,
    kpi_card, section_title, themed_table,
)


def render(tenant: str) -> html.Div:
    stale_data   = _get("/kg/incremental-community/summary", {"tenant": tenant})
    history_data = _get("/kg/community-history", {"tenant": tenant, "limit": 20})

    if e := http_error(stale_data):
        if not DEMO_MODE:
            return err(f"Community data unavailable — {e}")
        stale_data, history_data = demo_data.COMMUNITY_SUMMARY, demo_data.COMMUNITY_HISTORY

    summary      = stale_data or {}
    score        = summary.get("change_fraction", None)
    badge_color  = BAD if score is not None and score > 0.20 else GOOD
    badge_text   = (
        f"{round(float(score or 0) * 100, 1)}%"
        if score is not None else "—"
    )

    history      = (history_data or {}).get("history", []) if isinstance(history_data, dict) else []
    hist_table   = themed_table(
        data=[{k: str(v) for k, v in h.items()} for h in history],
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in
                 ["snapshot_id", "entity_count", "edge_count",
                  "community_coherence", "recorded_at"]],
        page_size=10,
    ) if history else html.Div("No version history yet.",
                               style={"color": "#8A99B5", "padding": "10px"})

    return html.Div([
        section_title("Community Health",
                      "Drift tracking & incremental Leiden rebuild status"),
        html.Div([
            kpi_card("Change fraction", badge_text, color=badge_color, accent=badge_color,
                     hint="entities changed since rebuild"),
            kpi_card("Changed entities", str(summary.get("changed_entities", "—")),
                     accent=TEAL, hint="pending re-clustering"),
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "2px"}),
        html.Button(
            "↻  Rebuild Affected Communities",
            id="rebuild-btn",
            style={"margin": "18px 0", "padding": "11px 24px",
                   "background": f"linear-gradient(135deg,{TEAL},{NAV})", "color": "white",
                   "border": "none", "borderRadius": "9px", "cursor": "pointer",
                   "fontWeight": "700", "fontFamily": FONT, "fontSize": "13.5px",
                   "boxShadow": "0 4px 14px rgba(0,150,180,0.28)"},
        ),
        html.Div(id="rebuild-result"),
        html.Div("Version History", style={"fontSize": "15px", "fontWeight": "700",
                                           "color": NAV, "margin": "22px 0 10px"}),
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
