"""Admin dashboard — Dash-based operator UI for GraphRAG.

Five tabs covering all operational concerns:

  1. Graph Health   — KPI cards + staleness trend line       (tabs/health.py)
  2. Conflicts      — DataTable + resolve action             (tabs/conflicts.py)
  3. Communities    — staleness badge, rebuild, history      (tabs/communities.py)
  4. GDPR & PII     — audit log, forget-entity form          (tabs/gdpr.py)
  5. Calibration    — Brier score trend, isotonic bins       (tabs/calibration.py)

Architecture
------------
All data is fetched from the existing REST API via httpx (no direct Neo4j here).
Shared helpers live in graphrag.dashboard.utils.
Tab layouts and callbacks live in graphrag.dashboard.tabs.*.

The Dash server is mounted under FastAPI at /admin via a2wsgi in api/main.py.

Standalone mode (dev)::

    python -m graphrag.dashboard.app
    # → http://localhost:8050

FastAPI mounted mode::

    uvicorn api.main:app
    # → http://localhost:8000/admin/
"""

from __future__ import annotations

import os
import secrets

import dash
import flask
import structlog
from dash import Input, Output, State, callback, dcc, html

# Tab modules imported here so their @callback decorators are registered.
import graphrag.dashboard.tabs  # noqa: F401 — side-effect import
from graphrag.dashboard.tabs import calibration, communities, conflicts, gdpr, health
from graphrag.dashboard.utils import (
    CANVAS, FONT, NAV, NAV2, TAB_SELECTED_STYLE, TAB_STYLE, TEAL, TEAL2,
)

log = structlog.get_logger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

ADMIN_TOKEN = os.getenv("GRAPHRAG_ADMIN_TOKEN", "")

# ── Dash app ───────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="GraphRAG Admin",
    requests_pathname_prefix="/admin/",
    suppress_callback_exceptions=True,
    server=flask.Flask(__name__),
)
app.server.secret_key = os.getenv("GRAPHRAG_ADMIN_SECRET", secrets.token_hex(32))

# Inject Inter webfont + global polish (scrollbars, focus rings, canvas bg).
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  html, body { margin:0; background:""" + CANVAS + """; font-family:""" + FONT + """; }
  * { box-sizing:border-box; }
  ::-webkit-scrollbar { height:10px; width:10px; }
  ::-webkit-scrollbar-thumb { background:#C3D1E8; border-radius:6px; }
  ::-webkit-scrollbar-thumb:hover { background:#A9BCDC; }
  .tab-parent .tab { transition: color .15s ease, border-color .15s ease; }
  .dash-spreadsheet-container .dash-spreadsheet-inner table { border-collapse:separate !important; }
  input:focus, button:focus { outline:2px solid """ + TEAL2 + """; outline-offset:1px; }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

# ── Auth guard ─────────────────────────────────────────────────────────────────

_LOGIN_PATH  = "/admin/login"
_LOGIN_POST  = "/admin/_login"
_STATIC_PFXS = ("/admin/_dash", "/admin/assets")


@app.server.before_request
def _require_auth():
    """Redirect unauthenticated browsers; fail closed in production."""
    if not ADMIN_TOKEN:
        # No token configured: open in dev/test, but hard-deny in production.
        try:
            from graphrag.core.config import get_settings
            if get_settings().env == "production":
                flask.abort(403)  # fail closed — never open-access in prod
        except Exception:  # noqa: BLE001 — config error must not expose dashboard
            flask.abort(403)
        return  # dev mode

    path = flask.request.path
    if path in (_LOGIN_PATH, _LOGIN_POST) or any(path.startswith(p) for p in _STATIC_PFXS):
        return
    if flask.request.headers.get("X-Admin-Token") == ADMIN_TOKEN:
        return
    if flask.session.get("admin_authenticated"):
        return
    log.warning("dashboard.unauthenticated", path=path, ip=flask.request.remote_addr)
    return flask.redirect(_LOGIN_PATH)


@app.server.route(_LOGIN_PATH, methods=["GET"])
def _login_page():
    error = "Invalid token. Try again." if flask.request.args.get("error") else ""
    return f"""<!DOCTYPE html>
<html><head><title>GraphRAG Admin Login</title>
<style>body{{font-family:system-ui;display:flex;justify-content:center;
padding-top:120px;background:#f4f4f5}}
form{{background:#fff;padding:36px;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.15);
min-width:320px}}h2{{margin:0 0 24px;color:#1a1a2e}}
input{{width:100%;padding:8px;margin:4px 0 16px;box-sizing:border-box;
border:1px solid #ccc;border-radius:4px}}
button{{width:100%;padding:10px;background:#1a1a2e;color:#fff;border:none;
border-radius:4px;cursor:pointer;font-size:15px}}
.err{{color:#c00;margin-bottom:12px}}</style></head>
<body><form method="POST" action="{_LOGIN_POST}">
<h2>🔐 GraphRAG Admin</h2>
<p class="err">{error}</p>
<label>Admin token</label>
<input type="password" name="token" placeholder="Enter admin token" autofocus>
<button type="submit">Sign in</button>
</form></body></html>"""


@app.server.route(_LOGIN_POST, methods=["POST"])
def _login_submit():
    token = flask.request.form.get("token", "")
    if secrets.compare_digest(token, ADMIN_TOKEN):
        flask.session["admin_authenticated"] = True
        return flask.redirect("/admin/")
    return flask.redirect(f"{_LOGIN_PATH}?error=1")


# ── Layout ─────────────────────────────────────────────────────────────────────

def _brand_mark() -> html.Div:
    """Small graph-node glyph + wordmark, echoing the deck's network motif."""
    return html.Div([
        html.Div([
            html.Div(style={"position": "absolute", "width": "26px", "height": "2px",
                            "background": TEAL2, "top": "14px", "left": "5px",
                            "transform": "rotate(28deg)", "opacity": "0.75"}),
            html.Div(style={"position": "absolute", "width": "22px", "height": "2px",
                            "background": TEAL2, "top": "20px", "left": "8px",
                            "transform": "rotate(-34deg)", "opacity": "0.6"}),
            *[html.Div(style={
                "position": "absolute", "width": f"{d}px", "height": f"{d}px",
                "borderRadius": "50%", "background": c,
                "top": f"{t}px", "left": f"{l}px",
            }) for d, c, t, l in [
                (11, TEAL2, 4, 2), (8, "#FFFFFF", 22, 0),
                (9, TEAL, 24, 22), (7, TEAL2, 6, 24),
            ]],
        ], style={"position": "relative", "width": "36px", "height": "36px",
                  "marginRight": "12px"}),
        html.Div([
            html.Div("GraphRAG", style={"fontSize": "19px", "fontWeight": "800",
                                        "color": "white", "lineHeight": "1.05",
                                        "letterSpacing": "-0.01em"}),
            html.Div("ADMIN · OBSERVABILITY", style={"fontSize": "9px",
                     "fontWeight": "700", "color": TEAL2, "letterSpacing": "0.18em"}),
        ]),
    ], style={"display": "flex", "alignItems": "center"})


app.layout = html.Div([
    dcc.Interval(id="auto-refresh", interval=30_000, n_intervals=0),
    dcc.Store(id="tenant-store", data="default"),

    # ── Gradient header bar ─────────────────────────────────────────────────
    html.Div([
        _brand_mark(),
        html.Div([
            html.Span("● LIVE", style={
                "color": "#5EEAD4", "fontSize": "10px", "fontWeight": "700",
                "letterSpacing": "0.1em", "marginRight": "18px",
            }),
            html.Label("TENANT", style={"marginRight": "8px", "color": "#AFC4E4",
                                        "fontSize": "10px", "fontWeight": "700",
                                        "letterSpacing": "0.1em"}),
            dcc.Input(
                id="tenant-input", type="text", value="default", debounce=True,
                style={
                    "width": "170px", "padding": "7px 12px",
                    "borderRadius": "8px", "border": "1px solid rgba(255,255,255,0.18)",
                    "background": "rgba(255,255,255,0.08)", "color": "white",
                    "fontFamily": FONT, "fontSize": "13px", "fontWeight": "600",
                },
            ),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "16px 28px",
        "background": f"linear-gradient(110deg, {NAV} 0%, {NAV2} 60%, #234079 100%)",
        "boxShadow": "0 6px 24px rgba(15,31,71,0.22)",
        "borderBottom": f"3px solid {TEAL}",
    }),

    # ── Tabs ────────────────────────────────────────────────────────────────
    html.Div(
        dcc.Tabs(id="tabs", value="health", className="tab-parent", children=[
            dcc.Tab(label="Graph Health",  value="health",      style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Conflicts",     value="conflicts",   style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Communities",   value="communities", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="GDPR & PII",    value="gdpr",        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Calibration",   value="calibration", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        ]),
        style={"background": "white", "padding": "0 16px",
               "borderBottom": f"1px solid {TEAL}", "boxShadow": "0 2px 8px rgba(15,31,71,0.04)"},
    ),

    html.Div(id="tab-content", style={"padding": "22px 28px", "maxWidth": "1180px",
                                      "margin": "0 auto"}),
], style={"fontFamily": FONT, "background": CANVAS, "minHeight": "100vh"})


# ── Global callbacks ───────────────────────────────────────────────────────────

@callback(
    Output("tenant-store", "data"),
    Input("tenant-input", "value"),
)
def sync_tenant(value: str):
    return value or "default"


@callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("auto-refresh", "n_intervals"),
    State("tenant-store", "data"),
)
def render_tab(tab: str, _n: int, tenant: str):
    tenant = tenant or "default"
    if tab == "health":      return health.render(tenant)
    if tab == "conflicts":   return conflicts.render(tenant)
    if tab == "communities": return communities.render(tenant)
    if tab == "gdpr":        return gdpr.render(tenant)
    if tab == "calibration": return calibration.render(tenant)
    return html.P("Unknown tab")


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # requests_pathname_prefix is set at construction and read-only; keep /admin/
    app.run(debug=True, port=8050)
