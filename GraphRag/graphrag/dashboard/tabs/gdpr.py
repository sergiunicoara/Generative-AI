"""Tab 4: GDPR & PII — audit log table + forget-entity form."""

from __future__ import annotations

from dash import Input, Output, State, callback, dash_table, dcc, html

from graphrag.dashboard.utils import H2, _get, _post, err, http_error


def render(tenant: str) -> html.Div:
    audit_data = _get("/kg/gdpr/audit-log", {"tenant": tenant, "limit": 50})
    if e := http_error(audit_data):
        return err(f"GDPR audit log unavailable — {e}")
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
        html.H3("Forget Entity", style=H2),
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
        html.H2("GDPR Audit Log", style=H2),
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
    if e := http_error(result):
        return html.P(f"⚠ Erasure failed — {e}", style={"color": "#c00"})
    if result:
        return html.P(f"✅ Entity '{name}' erasure complete.", style={"color": "#2a9d4f"})
    return html.P("⚠ Erasure failed.", style={"color": "#c00"})
