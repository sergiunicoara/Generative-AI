"""FastAPI + Plotly Dash dashboard for live KPI monitoring.

Premium operator UI matching the GraphRAG pitch deck: gradient navy header,
status-coloured KPI tiles, and a branded time-series chart with alert
thresholds. Visual layer only — all data comes from KPITracker unchanged.
"""

from __future__ import annotations

import asyncio

import dash
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from graphrag.business_matrix.kpi_tracker import KPITracker
from graphrag.core.config import get_settings
from graphrag.dashboard.utils import (
    BAD, CANVAS, FONT, GOOD, MUTED, NAV, NAV2, TEAL, TEAL2, WARN,
    BRAND_TEMPLATE, card_panel, kpi_card, style_fig,
)

# ── FastAPI app ────────────────────────────────────────────────────────────────
api = FastAPI(title="GraphRAG Business Matrix")


@api.get("/kpis/summary")
async def kpi_summary(window_days: int = 7):
    tracker = KPITracker()
    return await tracker.get_summary(window_days=window_days)


@api.get("/kpis/timeseries")
async def kpi_timeseries(metric: str = "latency_ms", window_days: int = 7):
    tracker = KPITracker()
    return await tracker.get_timeseries(metric=metric, window_days=window_days)


@api.get("/health")
async def health():
    return {"status": "ok"}


# ── Plotly Dash app ────────────────────────────────────────────────────────────
dash_app = dash.Dash(
    __name__,
    requests_pathname_prefix="/dashboard/",
    suppress_callback_exceptions=True,
)

# Inject Inter webfont + canvas styling.
dash_app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>GraphRAG Business Matrix</title>{%favicon%}{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  html,body{margin:0;background:""" + CANVAS + """;font-family:""" + FONT + """;}
  *{box-sizing:border-box;}
  ::-webkit-scrollbar{height:10px;width:10px;}
  ::-webkit-scrollbar-thumb{background:#C3D1E8;border-radius:6px;}
  .Select-control,.Select-menu-outer{font-family:""" + FONT + """;}
</style></head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""

# metric value → (display label, y-axis title, "higher is better"?)
METRIC_META = {
    "latency_ms":        ("Latency",          "milliseconds",  False),
    "faithfulness":      ("Faithfulness",     "score (0–1)",   True),
    "answer_relevancy":  ("Answer Relevancy", "score (0–1)",   True),
    "context_precision": ("Context Precision","score (0–1)",   True),
    "context_recall":    ("Context Recall",   "score (0–1)",   True),
}
METRICS = list(METRIC_META)


def _brand_header() -> html.Div:
    nodes = [html.Div(style={
        "position": "absolute", "width": f"{d}px", "height": f"{d}px",
        "borderRadius": "50%", "background": c, "top": f"{t}px", "left": f"{l}px",
    }) for d, c, t, l in [(11, TEAL2, 4, 2), (8, "#FFF", 22, 0), (9, TEAL, 24, 22), (7, TEAL2, 6, 24)]]
    return html.Div([
        html.Div([
            html.Div(nodes, style={"position": "relative", "width": "36px",
                                   "height": "36px", "marginRight": "12px"}),
            html.Div([
                html.Div("GraphRAG Business Matrix", style={"fontSize": "19px",
                         "fontWeight": "800", "color": "white", "lineHeight": "1.05"}),
                html.Div("LIVE KPIs · COST · QUALITY · LATENCY", style={"fontSize": "9px",
                         "fontWeight": "700", "color": TEAL2, "letterSpacing": "0.16em"}),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Span("● LIVE  ·  refresh 30s", style={"color": "#5EEAD4", "fontSize": "10px",
                  "fontWeight": "700", "letterSpacing": "0.08em"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "16px 28px",
        "background": f"linear-gradient(110deg,{NAV} 0%,{NAV2} 60%,#234079 100%)",
        "boxShadow": "0 6px 24px rgba(15,31,71,0.22)", "borderBottom": f"3px solid {TEAL}",
    })


dash_app.layout = html.Div([
    _brand_header(),
    html.Div([
        html.Div(id="summary-table"),  # KPI tile row
        card_panel(html.Div([
            html.Div([
                html.Label("METRIC", style={"fontSize": "10px", "fontWeight": "700",
                           "color": MUTED, "letterSpacing": "0.1em", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="metric-selector",
                    options=[{"label": METRIC_META[m][0], "value": m} for m in METRICS],
                    value="latency_ms", clearable=False,
                    style={"width": "260px", "fontFamily": FONT},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            dcc.Graph(id="timeseries-chart", config={"displayModeBar": False}),
        ])),
        dcc.Interval(id="refresh-interval", interval=30_000, n_intervals=0),
    ], style={"padding": "22px 28px", "maxWidth": "1180px", "margin": "0 auto"}),
], style={"fontFamily": FONT, "background": CANVAS, "minHeight": "100vh"})


def _summary_cards(summary: dict) -> html.Div:
    q   = summary.get("total_queries", 0)
    win = summary.get("window_days", 7)

    # Graceful empty state — neutral tiles, never alarming red, when no queries
    # fall inside the window. Colours activate only once real data flows in.
    if not q:
        dash = "—"
        cards = [
            kpi_card("Total queries", "0", accent=MUTED, hint=f"last {win} days"),
            kpi_card("Avg latency", dash, color=MUTED, accent=MUTED, hint="awaiting queries"),
            kpi_card("p95 latency", dash, color=MUTED, accent=MUTED, hint="alert > 3000 ms"),
            kpi_card("Faithfulness", dash, color=MUTED, accent=MUTED, hint="RAGAS · target ≥ 0.70"),
            kpi_card("Context recall", dash, color=MUTED, accent=MUTED, hint="RAGAS · target ≥ 0.80"),
        ]
        return html.Div(cards, style={"display": "flex", "flexWrap": "wrap", "gap": "2px"})

    avg = summary.get("avg_latency_ms", 0)
    p95 = summary.get("p95_latency_ms", 0)
    fth = summary.get("avg_faithfulness", 0)
    rcl = summary.get("avg_context_recall", 0)

    def lat_color(v): return GOOD if v < 3000 else WARN if v < 5000 else BAD
    def sc_color(v):  return GOOD if v >= 0.75 else WARN if v >= 0.5 else BAD

    cards = [
        kpi_card("Total queries", f"{q:,}", accent=TEAL, hint=f"last {win} days"),
        kpi_card("Avg latency", f"{avg:,.0f} ms", color=lat_color(avg),
                 accent=lat_color(avg), hint=f'p50 {summary.get("p50_latency_ms",0):,.0f} ms'),
        kpi_card("p95 latency", f"{p95:,.0f} ms", color=lat_color(p95),
                 accent=lat_color(p95), hint="alert > 3000 ms"),
        kpi_card("Faithfulness", f"{fth:.3f}", color=sc_color(fth),
                 accent=sc_color(fth), hint="RAGAS · target ≥ 0.70"),
        kpi_card("Context recall", f"{rcl:.3f}", color=sc_color(rcl),
                 accent=sc_color(rcl), hint="RAGAS · target ≥ 0.80"),
    ]
    return html.Div(cards, style={"display": "flex", "flexWrap": "wrap", "gap": "2px"})


@dash_app.callback(
    Output("timeseries-chart", "figure"),
    Output("summary-table", "children"),
    Input("metric-selector", "value"),
    Input("refresh-interval", "n_intervals"),
)
def update_dashboard(metric: str, _):
    tracker = KPITracker()
    loop = asyncio.new_event_loop()
    ts_data = loop.run_until_complete(tracker.get_timeseries(metric=metric))
    summary = loop.run_until_complete(tracker.get_summary())
    loop.close()

    label, ytitle, _higher = METRIC_META.get(metric, (metric, metric, True))
    xs = [d["recorded_at"] for d in ts_data]
    ys = [d.get(metric, 0) for d in ts_data]

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="lines+markers", name=label,
        line={"color": TEAL, "width": 3, "shape": "spline"},
        marker={"size": 7, "color": NAV, "line": {"color": "white", "width": 1.5}},
        fill="tozeroy", fillcolor="rgba(0,150,180,0.10)",
    ))
    style_fig(fig, f"{label} — Trend", height=360)
    fig.update_layout(xaxis_title="Time", yaxis_title=ytitle)

    if not ys:
        fig.add_annotation(
            text="No queries in the selected window — run a query to populate",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font={"color": MUTED, "size": 13, "family": FONT},
        )

    cfg = get_settings().business_matrix
    thresholds = (cfg or {}).get("alert_thresholds", {}) if isinstance(cfg, dict) else {}
    if metric in thresholds:
        fig.add_hline(
            y=thresholds[metric], line_dash="dash", line_color=BAD, line_width=1.5,
            annotation_text=f"alert threshold · {thresholds[metric]}",
            annotation_position="top left",
            annotation_font={"color": BAD, "size": 10, "family": FONT},
        )

    return fig, _summary_cards(summary)


# Mount Dash under /dashboard
api.mount("/dashboard", WSGIMiddleware(dash_app.server))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8050)
