"""Shared helpers + design system for the GraphRAG admin dashboard.

Provides the API client functions (_get, _post), error sentinels, the brand
design tokens, a branded Plotly template, and the UI primitives (KPI cards,
gauges, section headers, themed tables) used across all tab modules.

Design language matches the GraphRAG pitch deck: deep-navy canvas accents,
teal highlights, Inter typography, soft elevation. Built to look credible on
a projector in front of a technical audience.
"""

from __future__ import annotations

import os

import httpx
import plotly.graph_objects as go
import structlog
from dash import dash_table, html

log = structlog.get_logger(__name__)

# ── API client config ──────────────────────────────────────────────────────────

API_BASE  = os.getenv("GRAPHRAG_API_URL",   "http://localhost:8000")
API_TOKEN = os.getenv("GRAPHRAG_API_TOKEN", "")
_HEADERS  = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

# Demo mode: when set, tabs fall back to representative sample data if the live
# API is unreachable — purely for screenshots / walkthroughs without a backend.
# Unset in production, so real data or real error panels are shown.
DEMO_MODE = os.getenv("GRAPHRAG_DASHBOARD_DEMO", "").lower() not in ("", "0", "false", "no")

# ══════════════════════════════════════════════════════════════════════════════
# Brand design tokens  (mirror graphrag_pitch.js palette)
# ══════════════════════════════════════════════════════════════════════════════

NAV    = "#0F1F47"   # deep navy — headers, primary text
NAV2   = "#1A2C5B"   # lifted navy — gradient stop
TEAL   = "#0096B4"   # primary accent
TEAL2  = "#00B4D8"   # bright accent — highlights, lines
INK    = "#1A2438"   # body text
INK2   = "#5A6B85"   # secondary text
MUTED  = "#8A99B5"   # tertiary / captions
CANVAS = "#EEF3FB"   # page background
CARDBG = "#FFFFFF"   # card surface
BORDER = "#DCE5F3"   # hairline borders
GOOD   = "#16A34A"   # healthy / pass
WARN   = "#E8A317"   # caution
BAD    = "#DC2626"   # alert / fail
GRID   = "#E4EBF6"   # chart gridlines

FONT = ('Inter, "Segoe UI", system-ui, -apple-system, sans-serif')

# ── Shared UI style constants ──────────────────────────────────────────────────

H2 = {
    "marginTop": "26px", "marginBottom": "14px",
    "color": NAV, "fontFamily": FONT,
    "fontSize": "19px", "fontWeight": "700",
    "letterSpacing": "-0.01em",
}

H3 = {
    "marginTop": "18px", "marginBottom": "8px",
    "color": NAV, "fontFamily": FONT,
    "fontSize": "15px", "fontWeight": "600",
}

TAB_STYLE = {
    "fontFamily": FONT, "padding": "12px 22px",
    "fontWeight": "600", "fontSize": "14px",
    "color": INK2, "border": "none",
    "backgroundColor": "transparent",
}

TAB_SELECTED_STYLE = {
    "fontFamily": FONT, "padding": "12px 22px",
    "fontWeight": "700", "fontSize": "14px",
    "color": NAV, "border": "none",
    "borderBottom": f"3px solid {TEAL}",
    "backgroundColor": "transparent",
}

CARD_STYLE = {
    "background": CARDBG,
    "border": f"1px solid {BORDER}",
    "borderRadius": "14px",
    "padding": "18px 20px",
    "margin": "8px",
    "minWidth": "168px",
    "flex": "1",
    "boxShadow": "0 4px 18px rgba(15,31,71,0.06)",
    "fontFamily": FONT,
}


# ══════════════════════════════════════════════════════════════════════════════
# Branded Plotly template
# ══════════════════════════════════════════════════════════════════════════════

def _build_template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font={"family": FONT, "color": INK, "size": 13},
        title={"font": {"family": FONT, "color": NAV, "size": 16}, "x": 0.01, "xanchor": "left"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[TEAL, TEAL2, NAV, "#7C9CBF", GOOD, WARN, BAD],
        xaxis={"gridcolor": GRID, "linecolor": BORDER, "zerolinecolor": GRID,
               "tickfont": {"color": INK2, "size": 11}},
        yaxis={"gridcolor": GRID, "linecolor": BORDER, "zerolinecolor": GRID,
               "tickfont": {"color": INK2, "size": 11}},
        legend={"font": {"color": INK2, "size": 11}, "bgcolor": "rgba(0,0,0,0)"},
        margin={"t": 48, "b": 44, "l": 56, "r": 24},
        hoverlabel={"font": {"family": FONT}},
    )
    return t


BRAND_TEMPLATE = _build_template()


def style_fig(fig: go.Figure, title: str | None = None, height: int = 300) -> go.Figure:
    """Apply the brand template + common layout to a figure."""
    fig.update_layout(template=BRAND_TEMPLATE, height=height)
    if title is not None:
        fig.update_layout(title=title)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# API client
# ══════════════════════════════════════════════════════════════════════════════

def _get(path: str, params: dict | None = None) -> dict | list | None:
    """Synchronous HTTP GET against the REST API.

    Returns the parsed JSON on success, or a dict with key ``_http_error``
    containing a human-readable error string on any failure.
    """
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, headers=_HEADERS, timeout=4)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        detail = f"HTTP {exc.response.status_code} {exc.response.reason_phrase} — {path}"
        log.warning("dashboard.http_error", path=path,
                    status=exc.response.status_code, detail=detail)
        return {"_http_error": detail}
    except Exception as exc:  # broad: any network/parse failure falls back to error sentinel
        detail = f"{type(exc).__name__}: {exc} — {path}"
        log.warning("dashboard.request_error", path=path, error=detail)
        return {"_http_error": detail}


def _post(path: str, json: dict | None = None) -> dict | None:
    """Synchronous HTTP POST.  Returns parsed JSON or ``{"_http_error": …}``."""
    try:
        r = httpx.post(f"{API_BASE}{path}", json=json, headers=_HEADERS, timeout=4)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        detail = f"HTTP {exc.response.status_code} {exc.response.reason_phrase} — {path}"
        log.warning("dashboard.http_error", path=path,
                    status=exc.response.status_code, detail=detail)
        return {"_http_error": detail}
    except Exception as exc:  # broad: any network/parse failure falls back to error sentinel
        detail = f"{type(exc).__name__}: {exc} — {path}"
        log.warning("dashboard.request_error", path=path, error=detail)
        return {"_http_error": detail}


def http_error(data: dict | list | None) -> str | None:
    """Return the HTTP error string if *data* is an error sentinel, else None."""
    if isinstance(data, dict):
        return data.get("_http_error")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# UI primitives
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(label: str, value: str, color: str = NAV,
             *, hint: str | None = None, accent: str = TEAL) -> html.Div:
    """Elevated KPI tile with a coloured accent rail and optional sub-hint.

    Backward compatible with the original ``kpi_card(label, value, color)``
    signature; ``hint`` and ``accent`` are optional enhancements.
    """
    body = [
        html.Div(value, style={
            "fontSize": "30px", "fontWeight": "800", "color": color,
            "fontFamily": FONT, "letterSpacing": "-0.02em", "lineHeight": "1.1",
        }),
        html.Div(label.upper(), style={
            "fontSize": "10.5px", "color": INK2, "marginTop": "6px",
            "fontFamily": FONT, "fontWeight": "600", "letterSpacing": "0.04em",
        }),
    ]
    if hint:
        body.append(html.Div(hint, style={
            "fontSize": "11px", "color": MUTED, "marginTop": "4px", "fontFamily": FONT,
        }))

    return html.Div(
        html.Div(body, style={"paddingLeft": "12px"}),
        style={
            **CARD_STYLE,
            "borderLeft": f"4px solid {accent}",
            "padding": "16px 18px",
        },
    )


def gauge(value: float, title: str, *, good_high: bool = True,
          suffix: str = "", height: int = 220) -> go.Figure:
    """A clean radial gauge for a 0–1 (or 0–100) score.

    ``good_high`` controls the colour ramp direction: True → higher is better
    (green at top), False → lower is better (green at bottom).
    """
    pct = value * 100 if value <= 1 else value
    pct = max(0.0, min(100.0, pct))

    if good_high:
        bar = GOOD if pct >= 75 else WARN if pct >= 45 else BAD
        steps = [
            {"range": [0, 45],  "color": "#FBE9E9"},
            {"range": [45, 75], "color": "#FCF3E0"},
            {"range": [75, 100], "color": "#E8F6EC"},
        ]
    else:
        bar = GOOD if pct <= 25 else WARN if pct <= 55 else BAD
        steps = [
            {"range": [0, 25],  "color": "#E8F6EC"},
            {"range": [25, 55], "color": "#FCF3E0"},
            {"range": [55, 100], "color": "#FBE9E9"},
        ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": suffix or " %", "font": {"size": 30, "color": NAV, "family": FONT}},
        title={"text": title, "font": {"size": 13, "color": INK2, "family": FONT}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": BORDER,
                     "tickfont": {"size": 9, "color": MUTED}},
            "bar": {"color": bar, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": steps,
        },
    ))
    fig.update_layout(
        template=BRAND_TEMPLATE, height=height,
        margin={"t": 36, "b": 8, "l": 24, "r": 24},
    )
    return fig


def themed_table(data: list[dict], columns: list[dict], **kwargs) -> dash_table.DataTable:
    """A DataTable pre-styled to match the brand."""
    style = {
        "style_table": {"overflowX": "auto", "borderRadius": "12px",
                        "border": f"1px solid {BORDER}", "overflow": "hidden"},
        "style_cell": {
            "textAlign": "left", "padding": "11px 14px",
            "fontFamily": FONT, "fontSize": "13px", "color": INK,
            "border": "none", "borderBottom": f"1px solid {BORDER}",
        },
        "style_header": {
            "fontWeight": "700", "fontSize": "11px", "color": "white",
            "backgroundColor": NAV, "border": "none",
            "textTransform": "uppercase", "letterSpacing": "0.04em",
            "padding": "12px 14px",
        },
        "style_data_conditional": [
            {"if": {"row_index": "odd"}, "backgroundColor": "#F6F9FE"},
        ],
    }
    # caller overrides win
    for k, v in kwargs.items():
        if k in style and isinstance(v, list) and k == "style_data_conditional":
            style[k] = style[k] + v
        else:
            style[k] = v
    return dash_table.DataTable(data=data, columns=columns, **style)


def section_title(text: str, subtitle: str | None = None) -> html.Div:
    """A title block with a teal accent rail, matching the deck's section headers."""
    children = [html.Div(text, style={
        "fontSize": "20px", "fontWeight": "800", "color": NAV,
        "fontFamily": FONT, "letterSpacing": "-0.01em",
    })]
    if subtitle:
        children.append(html.Div(subtitle, style={
            "fontSize": "12.5px", "color": INK2, "marginTop": "3px", "fontFamily": FONT,
        }))
    return html.Div([
        html.Div(style={"width": "4px", "borderRadius": "3px",
                        "background": f"linear-gradient(180deg,{TEAL},{TEAL2})",
                        "marginRight": "12px"}),
        html.Div(children),
    ], style={"display": "flex", "alignItems": "stretch", "margin": "8px 0 18px"})


def card_panel(children, *, pad: str = "20px") -> html.Div:
    """A rounded white surface for grouping charts/content."""
    return html.Div(children, style={
        "background": CARDBG, "border": f"1px solid {BORDER}",
        "borderRadius": "16px", "padding": pad, "margin": "10px 0",
        "boxShadow": "0 4px 18px rgba(15,31,71,0.06)",
    })


def err(msg: str) -> html.Div:
    return html.Div([
        html.Span("⚠", style={"marginRight": "8px", "fontSize": "15px"}),
        html.Span(msg),
    ], style={
        "color": BAD, "padding": "14px 16px", "fontFamily": FONT,
        "background": "#FCEBEB", "border": f"1px solid #F4C7C7",
        "borderRadius": "10px", "margin": "10px 0", "fontSize": "13px",
    })
