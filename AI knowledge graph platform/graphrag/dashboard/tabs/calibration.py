"""Tab 5: Calibration — Brier score trend + isotonic calibration curve."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

from graphrag.dashboard import demo_data
from graphrag.dashboard.utils import (
    BAD, DEMO_MODE, GOOD, MUTED, NAV, TEAL, _get, card_panel, err, http_error,
    kpi_card, section_title, style_fig,
)


def render(tenant: str) -> html.Div:
    cal_data = _get("/kg/calibration/snapshots", {"tenant": tenant, "limit": 20})

    if http_error(cal_data):
        if not DEMO_MODE:
            return err(f"Calibration data unavailable — {http_error(cal_data)}")
        cal_data = demo_data.CALIBRATION_SNAPSHOTS

    snaps = cal_data if isinstance(cal_data, list) else (cal_data or {}).get("snapshots", [])

    latest_snap = snaps[0] if snaps else {}
    brier = float(latest_snap.get("brier_score") or 0)
    rating = ("Excellent" if brier <= 0.15 else "Good" if brier <= 0.25
              else "Degraded" if brier <= 0.40 else "Poor")
    rating_color = (GOOD if brier <= 0.25 else "#E8A317" if brier <= 0.40 else BAD)

    # ── Brier trend ─────────────────────────────────────────────────────────
    fig_brier = go.Figure()
    if snaps:
        xs = [s.get("recorded_at", "") for s in reversed(snaps)]
        ys = [float(s.get("brier_score") or 0) for s in reversed(snaps)]
        fig_brier.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name="Brier score",
            line={"color": TEAL, "width": 3, "shape": "spline"},
            marker={"size": 7, "color": NAV, "line": {"color": "white", "width": 1.5}},
            fill="tozeroy", fillcolor="rgba(0,150,180,0.10)",
        ))
        style_fig(fig_brier, "Brier Score Trend  (lower is better)", height=300)
        fig_brier.update_layout(yaxis_title="Brier score")

    # ── Calibration curve ───────────────────────────────────────────────────
    bins_raw = latest_snap.get("calibration_bins", [])
    fig_bins = go.Figure()
    if bins_raw:
        predicted = [b.get("predicted", 0) for b in bins_raw]
        actual    = [b.get("actual",    0) for b in bins_raw]
        fig_bins.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
            line={"dash": "dash", "color": MUTED, "width": 1.5}))
        fig_bins.add_trace(go.Bar(
            x=predicted, y=actual, name="Actual frequency",
            marker={"color": TEAL, "line": {"color": NAV, "width": 0.5}}, opacity=0.85))
        style_fig(fig_bins, "Calibration Curve  (latest snapshot)", height=300)
        fig_bins.update_layout(xaxis_title="Predicted confidence",
                               yaxis_title="Actual frequency")

    summary = html.Div([
        kpi_card("Latest Brier", f"{brier:.3f}", color=rating_color, accent=rating_color,
                 hint=rating),
        kpi_card("Snapshots", str(len(snaps)), hint="calibration history"),
        kpi_card("Model", str(latest_snap.get("model_version", "—")),
                 hint="under evaluation"),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "2px"})

    return html.Div([
        section_title("Confidence Calibration",
                      "How well predicted confidence matches observed correctness"),
        summary,
        card_panel(dcc.Graph(figure=fig_brier, config={"displayModeBar": False}))
        if snaps else err("No calibration snapshots. Run /kg/calibration/snapshot."),
        card_panel(dcc.Graph(figure=fig_bins, config={"displayModeBar": False}))
        if bins_raw else html.Div("No calibration bins in latest snapshot.",
                                  style={"color": MUTED, "padding": "10px"}),
    ])
