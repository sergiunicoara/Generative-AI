"""Tab 1: Graph Health — KPI cards, contradiction trend, recent alerts."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dash_table, dcc, html

from graphrag.dashboard.utils import H2, _get, err, http_error, kpi_card


def render(tenant: str) -> html.Div:
    data        = _get("/kg/graph-snapshots/list", {"tenant": tenant})
    alerts_data = _get("/kg/health/alerts", {"limit": 10})

    if e := http_error(data):
        return err(f"Graph snapshots unavailable — {e}")

    snaps  = data.get("snapshots", []) if isinstance(data, dict) else []
    latest = snaps[0] if snaps else {}

    def _pct(v):  return f"{round(float(v or 0) * 100, 1)} %"
    def _rate(v): return f"{round(float(v or 0), 4)}"

    kpis = html.Div([
        kpi_card("Entities",           str(latest.get("entity_count", "—"))),
        kpi_card("Edges",              str(latest.get("edge_count", "—"))),
        kpi_card("Alias coverage",     _pct(latest.get("alias_coverage"))),
        kpi_card("High-conf rate",     _pct(latest.get("high_conf_rate"))),
        kpi_card("Contradiction /1k",  _rate(latest.get("contradiction_rate"))),
        kpi_card("Orphan rate",        _pct(latest.get("orphan_rate")),
                 color="#c00" if float(latest.get("orphan_rate") or 0) > 0.10 else "#1a1a2e"),
        kpi_card("Community coherence", _pct(latest.get("community_coherence"))),
    ], style={"display": "flex", "flexWrap": "wrap"})

    fig = go.Figure()
    if snaps:
        xs = [s.get("recorded_at", "") for s in reversed(snaps)]
        ys = [float(s.get("contradiction_rate") or 0) for s in reversed(snaps)]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                                 name="Contradiction rate"))
        fig.update_layout(
            title="Contradiction Rate Trend",
            xaxis_title="Recorded at",
            yaxis_title="Conflicts / 1k edges",
            margin={"t": 40, "b": 40},
            height=280,
        )

    alerts = (alerts_data or {}).get("alerts", []) if isinstance(alerts_data, dict) else []
    alert_table = dash_table.DataTable(
        data=alerts,
        columns=[{"name": c, "id": c} for c in
                 ["metric", "value", "threshold", "direction", "tenant", "fired_at"]],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
        style_header={"fontWeight": "bold"},
        style_data_conditional=[{
            "if": {"filter_query": '{direction} = "above"'},
            "backgroundColor": "#fff3cd",
        }],
        page_size=10,
    ) if alerts else html.P("No recent alerts.", style={"color": "#888"})

    return html.Div([
        html.H2("Graph Health", style=H2),
        kpis,
        dcc.Graph(figure=fig) if snaps else err(
            "No snapshots found. Run /kg/graph-snapshots/create."
        ),
        html.H3("Recent Alerts", style=H2),
        alert_table,
    ])
