"""Tab 4: GDPR & PII — audit log table + forget-entity form."""

from __future__ import annotations

from dash import Input, Output, State, callback, dcc, html

from graphrag.dashboard import demo_data
from graphrag.dashboard.utils import (
    BAD, DEMO_MODE, FONT, MUTED, NAV, _get, _post, card_panel, err, http_error,
    section_title, themed_table,
)


def render(tenant: str) -> html.Div:
    audit_data = _get("/kg/gdpr/audit-log", {"tenant": tenant, "limit": 50})
    if e := http_error(audit_data):
        if not DEMO_MODE:
            return err(f"GDPR audit log unavailable — {e}")
        audit_data = demo_data.GDPR_AUDIT
    audit = audit_data if isinstance(audit_data, list) else (audit_data or {}).get("log", [])

    audit_table = themed_table(
        data=[{k: str(v) for k, v in a.items()} for a in audit],
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in
                 ["audit_id", "action", "entity_name", "entity_type",
                  "tenant", "requested_by", "executed_at"]],
        page_size=20,
    ) if audit else html.Div("No audit entries yet.",
                             style={"color": MUTED, "padding": "10px"})

    _inp = {"padding": "8px 12px", "borderRadius": "8px",
            "border": "1px solid #DCE5F3", "fontFamily": FONT}
    forget_form = card_panel(html.Div([
        html.Div("Forget Entity  ·  GDPR Article 17",
                 style={"fontSize": "15px", "fontWeight": "700", "color": NAV,
                        "marginBottom": "12px"}),
        html.Div([
            dcc.Input(id="forget-name", type="text", placeholder="Entity name",
                      style={**_inp, "width": "200px"}),
            dcc.Input(id="forget-type", type="text", placeholder="Entity type (e.g. PERSON)",
                      style={**_inp, "marginLeft": "8px", "width": "180px"}),
            dcc.Input(id="forget-requested-by", type="text", placeholder="Requested by",
                      style={**_inp, "marginLeft": "8px", "width": "180px"}),
            html.Button("Forget", id="forget-btn",
                        style={"marginLeft": "12px", "padding": "9px 22px",
                               "background": BAD, "color": "white", "border": "none",
                               "borderRadius": "8px", "cursor": "pointer",
                               "fontWeight": "700", "fontFamily": FONT}),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                  "gap": "6px"}),
        html.Div(id="forget-result", style={"marginTop": "12px"}),
    ]))

    return html.Div([
        section_title("GDPR & PII", "Erasure audit trail and right-to-be-forgotten controls"),
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
