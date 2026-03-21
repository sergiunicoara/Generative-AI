"""FastAPI + Plotly Dash dashboard for live KPI monitoring."""

from __future__ import annotations

import asyncio

import dash
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from graphrag.business_matrix.kpi_tracker import KPITracker
from graphrag.core.config import get_settings

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

METRICS = [
    "latency_ms",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

dash_app.layout = html.Div(
    [
        html.H1("GraphRAG Business Matrix", style={"fontFamily": "monospace"}),
        dcc.Dropdown(
            id="metric-selector",
            options=[{"label": m, "value": m} for m in METRICS],
            value="latency_ms",
            clearable=False,
            style={"width": "300px"},
        ),
        dcc.Interval(id="refresh-interval", interval=30_000, n_intervals=0),
        dcc.Graph(id="timeseries-chart"),
        html.Div(id="summary-table"),
    ],
    style={"padding": "20px"},
)


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

    xs = [d["recorded_at"] for d in ts_data]
    ys = [d.get(metric, 0) for d in ts_data]

    fig = go.Figure(
        go.Scatter(x=xs, y=ys, mode="lines+markers", name=metric)
    )
    fig.update_layout(
        title=f"{metric} over time",
        xaxis_title="Time",
        yaxis_title=metric,
        template="plotly_dark",
    )

    cfg = get_settings().business_matrix
    thresholds = cfg.get("alert_thresholds", {})
    if metric in thresholds:
        fig.add_hline(
            y=thresholds[metric],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Alert threshold: {thresholds[metric]}",
        )

    table = html.Table(
        [html.Tr([html.Th(k), html.Td(str(v))]) for k, v in summary.items()],
        style={"marginTop": "20px", "fontFamily": "monospace"},
    )

    return fig, table


# Mount Dash under /dashboard
api.mount("/dashboard", WSGIMiddleware(dash_app.server))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8050)
