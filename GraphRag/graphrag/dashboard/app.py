"""Admin dashboard — Dash-based operator UI for GraphRAG.

Features
--------
Five tabs covering all operational concerns:

  1. Graph Health   — KPI cards + staleness trend line
  2. Conflicts      — DataTable of open Conflict nodes + resolve action
  3. Communities    — staleness badge, rebuild button, version history
  4. GDPR & PII     — audit log table, forget-entity form
  5. Calibration    — Brier score trend, isotonic bins bar chart

Architecture
------------
All data is fetched from the existing REST API (api/routes/kg_features.py +
api/routes/corrections.py) via httpx.  No new Neo4j queries here — the
dashboard is a pure consumer of the REST layer.

The Dash server is mounted under FastAPI at /admin via WSGIMiddleware in
api/main.py.

Standalone mode (dev)::

    python -m graphrag.dashboard.app
    # → http://localhost:8050

FastAPI mounted mode::

    uvicorn api.main:app
    # → http://localhost:8000/admin/

Configuration
-------------
API_BASE_URL: default http://localhost:8000  (set env var GRAPHRAG_API_URL to override)
API_TOKEN:    default ""  (set env var GRAPHRAG_API_TOKEN for Bearer auth)
"""

from __future__ import annotations

import os
import secrets

import dash
import flask
import httpx
import plotly.graph_objects as go
import structlog
from dash import Input, Output, State, callback, dcc, html
from dash import dash_table

log = structlog.get_logger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE    = os.getenv("GRAPHRAG_API_URL",     "http://localhost:8000")
API_TOKEN   = os.getenv("GRAPHRAG_API_TOKEN",   "")
# Set GRAPHRAG_ADMIN_TOKEN to enable dashboard auth.  When empty, the dashboard
# is open to all (dev mode only — never leave unset in production).
ADMIN_TOKEN = os.getenv("GRAPHRAG_ADMIN_TOKEN", "")

_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

# ── Dash app ───────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="GraphRAG Admin",
    requests_pathname_prefix="/admin/",   # prefix when mounted under FastAPI
    suppress_callback_exceptions=True,
    server=flask.Flask(__name__),         # explicit Flask server for before_request
)
# Use a random secret key so sessions survive restarts only when the env var is set
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
    # Always allow the login page and Dash static assets
    if path in (_LOGIN_PATH, _LOGIN_POST) or any(path.startswith(p) for p in _STATIC_PFXS):
        return
    # Accept X-Admin-Token header for scripted / API access
    if flask.request.headers.get("X-Admin-Token") == ADMIN_TOKEN:
        return
    # Accept valid session cookie
    if flask.session.get("admin_authenticated"):
        return
    log.warning("dashboard.unauthenticated", path=path,
                ip=flask.request.remote_addr)
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

_TAB_STYLE = {"fontFamily": "system-ui, sans-serif", "padding": "12px 20px"}
_CARD_STYLE = {
    "border": "1px solid #e0e0e0",
    "borderRadius": "8px",
    "padding": "16px",
    "margin": "8px",
    "minWidth": "150px",
    "textAlign": "center",
    "backgroundColor": "#fafafa",
}
_H2 = {"marginTop": "20px", "marginBottom": "10px", "color": "#333"}

app.layout = html.Div([
    dcc.Interval(id="auto-refresh", interval=30_000, n_intervals=0),   # 30-s refresh
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
        dcc.Tab(label="📊 Graph Health",   value="health",      style=_TAB_STYLE),
        dcc.Tab(label="⚡ Conflicts",       value="conflicts",   style=_TAB_STYLE),
        dcc.Tab(label="🏘️ Communities",    value="communities", style=_TAB_STYLE),
        dcc.Tab(label="🔒 GDPR & PII",     value="gdpr",        style=_TAB_STYLE),
        dcc.Tab(label="📐 Calibration",    value="calibration", style=_TAB_STYLE),
    ], style={"marginTop": "4px"}),

    html.Div(id="tab-content", style={"padding": "16px"}),
], style={"fontFamily": "system-ui, sans-serif"})


# ── Tenant store sync ──────────────────────────────────────────────────────────

@callback(
    Output("tenant-store", "data"),
    Input("tenant-input", "value"),
)
def sync_tenant(value: str):
    return value or "default"


# ── Tab routing ────────────────────────────────────────────────────────────────

@callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("auto-refresh", "n_intervals"),
    State("tenant-store", "data"),
)
def render_tab(tab: str, _n: int, tenant: str):
    tenant = tenant or "default"
    if tab == "health":
        return _render_health(tenant)
    if tab == "conflicts":
        return _render_conflicts(tenant)
    if tab == "communities":
        return _render_communities(tenant)
    if tab == "gdpr":
        return _render_gdpr(tenant)
    if tab == "calibration":
        return _render_calibration(tenant)
    return html.P("Unknown tab")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(path: str, params: dict | None = None) -> dict | list | None:
    """Synchronous HTTP GET against the REST API.

    Returns the parsed JSON on success, or a dict with key ``_http_error``
    containing a human-readable error string on any failure.
    """
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        detail = f"HTTP {exc.response.status_code} {exc.response.reason_phrase} — {path}"
        log.warning("dashboard.http_error", path=path,
                    status=exc.response.status_code, detail=detail)
        return {"_http_error": detail}
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc} — {path}"
        log.warning("dashboard.request_error", path=path, error=detail)
        return {"_http_error": detail}


def _post(path: str, json: dict | None = None) -> dict | None:
    """Synchronous HTTP POST.  Returns parsed JSON or ``{"_http_error": …}``."""
    try:
        r = httpx.post(f"{API_BASE}{path}", json=json, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        detail = f"HTTP {exc.response.status_code} {exc.response.reason_phrase} — {path}"
        log.warning("dashboard.http_error", path=path,
                    status=exc.response.status_code, detail=detail)
        return {"_http_error": detail}
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc} — {path}"
        log.warning("dashboard.request_error", path=path, error=detail)
        return {"_http_error": detail}


def _http_error(data: dict | list | None) -> str | None:
    """Return the HTTP error string if *data* is an error sentinel, else None."""
    if isinstance(data, dict):
        return data.get("_http_error")
    return None


def _kpi_card(label: str, value: str, color: str = "#1a1a2e") -> html.Div:
    return html.Div([
        html.Div(value, style={"fontSize": "28px", "fontWeight": "bold", "color": color}),
        html.Div(label, style={"fontSize": "12px", "color": "#666", "marginTop": "4px"}),
    ], style=_CARD_STYLE)


def _err(msg: str) -> html.P:
    return html.P(f"⚠ {msg}", style={"color": "#c00", "padding": "8px"})


# ── Tab 1: Graph Health ────────────────────────────────────────────────────────

def _render_health(tenant: str):
    data = _get("/kg/graph-snapshots/list", {"tenant": tenant})
    alerts_data = _get("/kg/health/alerts", {"limit": 10})

    if err := _http_error(data):
        return _err(f"Graph snapshots unavailable — {err}")

    # Latest snapshot KPIs
    snaps = data.get("snapshots", []) if isinstance(data, dict) else []
    latest = snaps[0] if snaps else {}

    def _pct(v): return f"{round(float(v or 0) * 100, 1)} %"
    def _rate(v): return f"{round(float(v or 0), 4)}"

    kpis = html.Div([
        _kpi_card("Entities",           str(latest.get("entity_count", "—"))),
        _kpi_card("Edges",              str(latest.get("edge_count", "—"))),
        _kpi_card("Alias coverage",     _pct(latest.get("alias_coverage"))),
        _kpi_card("High-conf rate",     _pct(latest.get("high_conf_rate"))),
        _kpi_card("Contradiction /1k",  _rate(latest.get("contradiction_rate"))),
        _kpi_card("Orphan rate",        _pct(latest.get("orphan_rate")),
                  color="#c00" if float(latest.get("orphan_rate") or 0) > 0.10 else "#1a1a2e"),
        _kpi_card("Community coherence", _pct(latest.get("community_coherence"))),
    ], style={"display": "flex", "flexWrap": "wrap"})

    # Trend line (contradiction_rate over time)
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

    # Recent alerts table
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
        html.H2("Graph Health", style=_H2),
        kpis,
        dcc.Graph(figure=fig) if snaps else _err("No snapshots found. Run /kg/graph-snapshots/create."),
        html.H3("Recent Alerts", style=_H2),
        alert_table,
    ])


# ── Tab 2: Conflicts ───────────────────────────────────────────────────────────

def _render_conflicts(tenant: str):
    data = _get("/corrections/list-conflicts", {"tenant": tenant, "limit": 100})
    if err := _http_error(data):
        return _err(f"Conflicts unavailable — {err}")
    conflicts = data if isinstance(data, list) else (data or {}).get("conflicts", [])

    if not conflicts:
        return html.Div([
            html.H2("Open Conflicts", style=_H2),
            html.P("✅ No open conflicts.", style={"color": "#2a9d4f"}),
        ])

    cols = ["conflict_id", "type", "src", "tgt", "relation", "tenant"]
    table = dash_table.DataTable(
        id="conflicts-table",
        data=[{k: str(c.get(k, "")) for k in cols} for c in conflicts],
        columns=[{"name": c, "id": c} for c in cols],
        row_selectable="single",
        selected_rows=[],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
        style_header={"fontWeight": "bold"},
        page_size=20,
    )

    resolve_panel = html.Div([
        html.H3("Resolve selected conflict", style={"marginTop": "20px"}),
        html.Div([
            dcc.Dropdown(
                id="resolve-resolution",
                options=[
                    {"label": "Resolved — manual", "value": "resolved_manual"},
                    {"label": "Resolved — authority", "value": "resolved_authority"},
                    {"label": "False positive", "value": "false_positive"},
                ],
                placeholder="Resolution type...",
                style={"width": "300px"},
            ),
            dcc.Input(id="resolve-winner-doc", type="text",
                      placeholder="Winner doc ID (optional)",
                      style={"marginLeft": "12px", "padding": "6px", "width": "240px"}),
            html.Button("Resolve", id="resolve-btn",
                        style={"marginLeft": "12px", "padding": "6px 16px",
                               "backgroundColor": "#1a1a2e", "color": "white",
                               "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(id="resolve-result", style={"marginTop": "10px", "color": "#2a9d4f"}),
    ])

    return html.Div([
        html.H2(f"Open Conflicts ({len(conflicts)})", style=_H2),
        table,
        resolve_panel,
    ])


@callback(
    Output("resolve-result", "children"),
    Input("resolve-btn", "n_clicks"),
    State("conflicts-table", "selected_rows"),
    State("conflicts-table", "data"),
    State("resolve-resolution", "value"),
    State("resolve-winner-doc", "value"),
    prevent_initial_call=True,
)
def resolve_conflict(n_clicks, selected_rows, rows, resolution, winner_doc):
    if not selected_rows or not resolution:
        return "Select a row and choose a resolution type."
    conflict_id = rows[selected_rows[0]].get("conflict_id", "")
    result = _post("/corrections/resolve-conflict", {
        "conflict_id": conflict_id,
        "resolution":  resolution,
        "winner_doc_id": winner_doc or "",
        "resolved_by": "admin_ui",
    })
    if err := _http_error(result):
        return f"⚠ Resolve failed — {err}"
    if result:
        return f"✅ Conflict {conflict_id[:8]}… resolved as '{resolution}'."
    return "⚠ Failed to resolve conflict."


# ── Tab 3: Communities ─────────────────────────────────────────────────────────

def _render_communities(tenant: str):
    stale_data = _get("/kg/incremental-community/summary", {"tenant": tenant})
    history_data = _get("/community-history", {"tenant": tenant, "limit": 20})

    if err := _http_error(stale_data):
        return _err(f"Community data unavailable — {err}")

    summary = stale_data or {}
    score = summary.get("change_fraction", None)
    badge_color = "#c00" if score is not None and score > 0.20 else "#2a9d4f"
    badge_text = f"{round(float(score or 0) * 100, 1)} % changed" if score is not None else "—"

    history = (history_data or {}).get("history", []) if isinstance(history_data, dict) else []
    hist_table = dash_table.DataTable(
        data=[{k: str(v) for k, v in h.items()} for h in history],
        columns=[{"name": c, "id": c} for c in
                 ["snapshot_id", "entity_count", "edge_count", "recorded_at", "is_rebuild"]],
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px"},
        page_size=10,
    ) if history else html.P("No version history yet.", style={"color": "#888"})

    return html.Div([
        html.H2("Community Health", style=_H2),
        html.Div([
            html.Div([
                html.Div(badge_text,
                         style={"fontSize": "32px", "fontWeight": "bold", "color": badge_color}),
                html.Div("Entity change fraction since last rebuild",
                         style={"fontSize": "12px", "color": "#666"}),
            ], style=_CARD_STYLE),
            html.Div([
                html.Div(str(summary.get("changed_entities", "—")),
                         style={"fontSize": "32px", "fontWeight": "bold"}),
                html.Div("Changed entities", style={"fontSize": "12px", "color": "#666"}),
            ], style=_CARD_STYLE),
        ], style={"display": "flex"}),
        html.Button(
            "🔄 Rebuild Affected Communities",
            id="rebuild-btn",
            style={"margin": "16px 0", "padding": "8px 20px",
                   "backgroundColor": "#1a1a2e", "color": "white",
                   "border": "none", "borderRadius": "4px", "cursor": "pointer"},
        ),
        html.Div(id="rebuild-result"),
        html.H3("Version History", style=_H2),
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
    if err := _http_error(result):
        return html.P(f"⚠ Rebuild failed — {err}", style={"color": "#c00"})
    if result:
        rebuilt = result.get("communities_rebuilt", "?")
        return html.P(f"✅ Rebuilt {rebuilt} communities.", style={"color": "#2a9d4f"})
    return html.P("⚠ Rebuild failed or timed out.", style={"color": "#c00"})


# ── Tab 4: GDPR & PII ─────────────────────────────────────────────────────────

def _render_gdpr(tenant: str):
    audit_data = _get("/kg/gdpr/audit-log", {"tenant": tenant, "limit": 50})
    if err := _http_error(audit_data):
        return _err(f"GDPR audit log unavailable — {err}")
    audit = audit_data if isinstance(audit_data, list) else (audit_data or {}).get("log", [])

    audit_table = dash_table.DataTable(
        data=[{k: str(v) for k, v in a.items()} for a in audit],
        columns=[{"name": c, "id": c} for c in
                 ["audit_id", "action", "entity_name", "entity_type",
                  "tenant", "requested_by", "executed_at"]],
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px"},
        page_size=20,
    ) if audit else html.P("No audit entries yet.", style={"color": "#888"})

    forget_form = html.Div([
        html.H3("Forget Entity", style=_H2),
        html.Div([
            dcc.Input(id="forget-name", type="text", placeholder="Entity name",
                      style={"padding": "6px", "width": "200px"}),
            dcc.Input(id="forget-type", type="text", placeholder="Entity type (e.g. PERSON)",
                      style={"padding": "6px", "marginLeft": "8px", "width": "180px"}),
            dcc.Input(id="forget-requested-by", type="text", placeholder="Requested by",
                      style={"padding": "6px", "marginLeft": "8px", "width": "180px"}),
            html.Button("Forget", id="forget-btn",
                        style={"marginLeft": "12px", "padding": "6px 16px",
                               "backgroundColor": "#c00", "color": "white",
                               "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(id="forget-result", style={"marginTop": "10px"}),
    ])

    return html.Div([
        html.H2("GDPR Audit Log", style=_H2),
        audit_table,
        forget_form,
    ])


@callback(
    Output("forget-result", "children"),
    Input("forget-btn", "n_clicks"),
    State("forget-name", "value"),
    State("forget-type", "value"),
    State("forget-requested-by", "value"),
    State("tenant-store", "data"),
    prevent_initial_call=True,
)
def forget_entity(n_clicks, name, etype, requested_by, tenant):
    if not name or not etype:
        return html.P("Entity name and type are required.", style={"color": "#c00"})
    result = _post("/kg/gdpr/forget-entity", {
        "entity_name":  name,
        "entity_type":  etype,
        "tenant":       tenant or "default",
        "requested_by": requested_by or "admin_ui",
    })
    if err := _http_error(result):
        return html.P(f"⚠ Erasure failed — {err}", style={"color": "#c00"})
    if result:
        return html.P(f"✅ Entity '{name}' erasure complete.", style={"color": "#2a9d4f"})
    return html.P("⚠ Erasure failed.", style={"color": "#c00"})


# ── Tab 5: Calibration ─────────────────────────────────────────────────────────

def _render_calibration(tenant: str):
    cal_data = _get("/kg/calibration/snapshots", {"tenant": tenant, "limit": 20})
    if err := _http_error(cal_data):
        return _err(f"Calibration data unavailable — {err}")
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

    # Latest isotonic bins
    latest_snap = snaps[0] if snaps else {}
    bins_raw = latest_snap.get("calibration_bins", [])
    fig_bins = go.Figure()
    if bins_raw:
        predicted = [b.get("predicted", 0) for b in bins_raw]
        actual    = [b.get("actual", 0) for b in bins_raw]
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
        html.H2("Confidence Calibration", style=_H2),
        dcc.Graph(figure=fig_brier) if snaps else _err(
            "No calibration snapshots. Run /kg/calibration/snapshot."
        ),
        dcc.Graph(figure=fig_bins) if bins_raw else html.P(
            "No calibration bins in latest snapshot.", style={"color": "#888"}
        ),
    ])


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # Override prefix for standalone mode (no FastAPI mount)
    app.config.requests_pathname_prefix = "/"
    app.run(debug=True, port=8050)
