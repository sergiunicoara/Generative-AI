"""Tab 1: Graph Health — KPI cards, health gauges, contradiction trend, alerts."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

from graphrag.dashboard import demo_data
from graphrag.dashboard.utils import (
    BAD, DEMO_MODE, GOOD, NAV, TEAL, TEAL2, WARN, _get, card_panel, err, gauge,
    http_error, kpi_card, section_title, style_fig, themed_table,
)


def render(tenant: str) -> html.Div:
    data        = _get("/kg/graph-snapshots/list", {"tenant": tenant})
    alerts_data = _get("/kg/health/alerts", {"limit": 10})

    if http_error(data):
        if not DEMO_MODE:
            return err(f"Graph snapshots unavailable — {http_error(data)}")
        data, alerts_data = demo_data.HEALTH_SNAPSHOTS, demo_data.HEALTH_ALERTS

    snaps  = data.get("snapshots", []) if isinstance(data, dict) else []
    latest = snaps[0] if snaps else {}

    def _f(key):  return float(latest.get(key) or 0)
    def _pct(v):  return f"{round(float(v or 0) * 100, 1)}%"

    orphan = _f("orphan_rate")

    # ── KPI strip ───────────────────────────────────────────────────────────
    kpis = html.Div([
        kpi_card("Entities",   f'{int(_f("entity_count")):,}', accent=TEAL),
        kpi_card("Edges",      f'{int(_f("edge_count")):,}',   accent=TEAL2),
        kpi_card("Alias coverage",    _pct(latest.get("alias_coverage")),
                 accent=GOOD, hint="resolved → canonical"),
        kpi_card("High-conf rate",    _pct(latest.get("high_conf_rate")),
                 accent=GOOD, hint="edges ≥ 0.75"),
        kpi_card("Contradiction /1k", f'{_f("contradiction_rate"):.2f}',
                 color=BAD if _f("contradiction_rate") > 5.0
                       else WARN if _f("contradiction_rate") > 3.0 else NAV,
                 accent=BAD if _f("contradiction_rate") > 5.0
                        else WARN if _f("contradiction_rate") > 3.0 else GOOD,
                 hint="< 2.0 healthy · > 3.0 warning · > 5.0 critical"),
        kpi_card("Orphan rate",       _pct(orphan),
                 color=BAD if orphan > 0.10 else NAV,
                 accent=BAD if orphan > 0.10 else GOOD),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "2px"})

    # ── Health gauges ───────────────────────────────────────────────────────
    gauges = html.Div([
        html.Div(dcc.Graph(figure=gauge(_f("alias_coverage"), "Entity Resolution",
                                        good_high=True), config={"displayModeBar": False}),
                 style={"flex": "1", "minWidth": "220px"}),
        html.Div(dcc.Graph(figure=gauge(_f("high_conf_rate"), "Relation Confidence",
                                        good_high=True), config={"displayModeBar": False}),
                 style={"flex": "1", "minWidth": "220px"}),
        html.Div(dcc.Graph(figure=gauge(_f("community_coherence"), "Community Coherence",
                                        good_high=True), config={"displayModeBar": False}),
                 style={"flex": "1", "minWidth": "220px"}),
        html.Div(dcc.Graph(figure=gauge(orphan, "Orphan Rate", good_high=False),
                                config={"displayModeBar": False}),
                 style={"flex": "1", "minWidth": "220px"}),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px"})

    # ── Contradiction trend ─────────────────────────────────────────────────
    fig = go.Figure()
    if snaps:
        xs = [s.get("recorded_at", "") for s in reversed(snaps)]
        ys = [float(s.get("contradiction_rate") or 0) for s in reversed(snaps)]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name="Contradiction rate",
            line={"color": TEAL, "width": 3, "shape": "spline"},
            marker={"size": 7, "color": NAV, "line": {"color": "white", "width": 1.5}},
            fill="tozeroy", fillcolor="rgba(0,150,180,0.10)",
        ))
        style_fig(fig, "Contradiction Rate Trend", height=300)
        fig.update_layout(xaxis_title="Recorded at", yaxis_title="Conflicts / 1k edges")

    # ── Alerts ──────────────────────────────────────────────────────────────
    alerts = (alerts_data or {}).get("alerts", []) if isinstance(alerts_data, dict) else []
    if alerts:
        alert_block = themed_table(
            data=alerts,
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in
                     ["metric", "value", "threshold", "direction", "tenant", "fired_at"]],
            page_size=10,
            style_data_conditional=[{
                "if": {"filter_query": '{direction} = "above"'},
                "backgroundColor": "#FCF3E0",
            }],
        )
    else:
        alert_block = html.Div("✓ No active alerts — all metrics within thresholds.",
                               style={"color": GOOD, "fontWeight": "600",
                                      "padding": "12px 4px", "fontSize": "13.5px"})

    return html.Div([
        section_title("Graph Health",
                      "Live structural & quality metrics for the knowledge graph"),
        kpis,
        card_panel(gauges) if snaps else err(
            "No snapshots found. Run /kg/graph-snapshots/create."),
        card_panel(dcc.Graph(figure=fig, config={"displayModeBar": False})) if snaps else None,
        html.Div("Recent Alerts", style={"fontSize": "15px", "fontWeight": "700",
                                          "color": NAV, "margin": "22px 0 10px"}),
        alert_block,
    ])
