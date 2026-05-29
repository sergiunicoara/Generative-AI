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
from graphrag.dashboard.utils import TAB_STYLE

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

app.layout = html.Div([
    dcc.Interval(id="auto-refresh", interval=30_000, n_intervals=0),
    dcc.Store(id="tenant-store", data="default"),

    html.Div([
        html.H1("GraphRAG Admin", style={"margin": "0 0 0 8px", "color": "#1a1a2e"}),
        html.Div([
            html.Label("Tenant:", style={"marginRight": "6px"}),
            dcc.Input(
                id="tenant-input", type="text", value="default",
                debounce=True,
                style={"width": "180px", "padding": "4px"},
            ),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "justifyContent": "space-between",
        "alignItems": "center", "padding": "12px 20px",
        "backgroundColor": "#1a1a2e", "color": "white",
    }),

    dcc.Tabs(id="tabs", value="health", children=[
        dcc.Tab(label="📊 Graph Health",   value="health",      style=TAB_STYLE),
        dcc.Tab(label="⚡ Conflicts",       value="conflicts",   style=TAB_STYLE),
        dcc.Tab(label="🏘️ Communities",    value="communities", style=TAB_STYLE),
        dcc.Tab(label="🔒 GDPR & PII",     value="gdpr",        style=TAB_STYLE),
        dcc.Tab(label="📐 Calibration",    value="calibration", style=TAB_STYLE),
    ], style={"marginTop": "4px"}),

    html.Div(id="tab-content", style={"padding": "16px"}),
], style={"fontFamily": "system-ui, sans-serif"})


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
    app.config.requests_pathname_prefix = "/"
    app.run(debug=True, port=8050)
