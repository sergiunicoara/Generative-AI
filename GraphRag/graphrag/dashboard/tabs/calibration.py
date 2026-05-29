"""Tab 5: Calibration — Brier score trend + isotonic calibration curve."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

from graphrag.dashboard.utils import H2, _get, err, http_error


def render(tenant: str) -> html.Div:
    cal_data = _get("/kg/calibration/snapshots", {"tenant": tenant, "limit": 20})
    if e := http_error(cal_data):
        return err(f"Calibration data unavailable — {e}")
    snaps = cal_data if isinstance(cal_data, list) else (cal_data or {}).get("snapshots", [])

    fig_brier = go.Figure()
    if snaps:
        xs = [s.get("recorded_at", "") for s in reversed(snaps)]
        ys = [float(s.get("brier_score") or 0) for s in reversed(snaps)]
        fig_brier.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                                       name="Brier score", line={"color": "#e76f51"}))
        fig_brier.update_layout(
            title="Brier Score Trend (lower is better)",
            yaxis_title="Brier score",
            margin={"t": 40, "b": 40},
            height=280,
        )

    latest_snap = snaps[0] if snaps else {}
    bins_raw    = latest_snap.get("calibration_bins", [])
    fig_bins    = go.Figure()
    if bins_raw:
        predicted = [b.get("predicted", 0) for b in bins_raw]
        actual    = [b.get("actual",    0) for b in bins_raw]
        fig_bins.add_trace(go.Bar(x=predicted, y=actual, name="Actual frequency"))
        fig_bins.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      name="Perfect calibration",
                                      line={"dash": "dash", "color": "#aaa"}))
        fig_bins.update_layout(
            title="Calibration Curve (latest snapshot)",
            xaxis_title="Predicted confidence",
            yaxis_title="Actual frequency",
            margin={"t": 40, "b": 40},
            height=280,
        )

    return html.Div([
        html.H2("Confidence Calibration", style=H2),
        dcc.Graph(figure=fig_brier) if snaps else err(
            "No calibration snapshots. Run /kg/calibration/snapshot."
        ),
        dcc.Graph(figure=fig_bins) if bins_raw else html.P(
            "No calibration bins in latest snapshot.", style={"color": "#888"}
        ),
    ])
