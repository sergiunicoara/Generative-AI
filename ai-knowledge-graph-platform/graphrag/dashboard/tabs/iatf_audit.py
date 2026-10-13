"""Tab: IATF Audit — clause evidence status, chain validity, and auditor sign-off.

Reads /audit/iatf/clauses and /audit/iatf/signoffs (api/routes/iatf_audit.py) and posts sign-offs to
/audit/iatf/clauses/{id}/signoff. The reviewer is taken server-side from the dashboard's token subject, never from
this UI, and the evidence digest is computed server-side, so the tab cannot sign off on evidence it did not show.
"""

from __future__ import annotations

from dash import Input, Output, State, callback, html

from graphrag.dashboard.utils import (
    GOOD, NAV, TEAL, _get, _post, card_panel, err, http_error, section_title, themed_table,
)

_DECISIONS = {"iatf-accept-btn": "accepted", "iatf-reject-btn": "rejected",
              "iatf-more-btn": "needs_more_evidence"}
_STATUS_LABEL = {"evidenced": "✓ evidenced", "stale": "⚠ stale", "only_superseded": "⚠ only superseded",
                 "missing": "✗ missing"}


def clause_rows(view: dict, latest_signoffs: dict) -> list[dict]:
    rows = []
    for c in view.get("clauses", []):
        signoff = latest_signoffs.get(c["clause"]) or {}
        rows.append({
            "clause": c["clause"],
            "heading": c["heading"],
            "status": _STATUS_LABEL.get(c["status"], c["status"]),
            "evidence": ", ".join(e["doc_id"] for e in c.get("evidence", [])) or "—",
            "conflicts": " ".join(x["tag"] for x in c.get("conflicts", [])) or "—",
            "mapping": (c.get("mapping") or {}).get("status", ""),
            "signoff": (f"{signoff.get('decision')} by {signoff.get('reviewer')}"
                        + ("" if signoff.get("evidence_digest") == c.get("evidence_digest")
                           else " (evidence changed since)")) if signoff else "—",
        })
    return rows


def _latest_evidence_signoffs(entries: list[dict]) -> dict:
    latest = {}
    for e in sorted(entries, key=lambda x: int(x.get("seq") or 0)):
        if e.get("decision") in ("accepted", "rejected", "needs_more_evidence"):
            latest[e["clause"]] = e
    return latest


def render(tenant: str) -> html.Div:
    view = _get("/audit/iatf/clauses")
    if e := http_error(view):
        return err(f"IATF audit view unavailable — {e}")
    signoffs = _get("/audit/iatf/signoffs")
    chain_ok, latest = None, {}
    if not http_error(signoffs):
        chain_ok = (signoffs.get("chain") or {}).get("valid")
        latest = _latest_evidence_signoffs(signoffs.get("entries") or [])

    counts = view.get("counts", {})
    summary = html.Div([
        html.Span(f"Coverage {view.get('coverage')}  ·  " + "  ·  ".join(f"{k}: {v}" for k, v in counts.items())),
        html.Span(
            "  ·  sign-off chain: " + ("✓ verified" if chain_ok else "✗ BROKEN" if chain_ok is False else "unknown"),
            style={"color": GOOD if chain_ok else "#B42318", "fontWeight": "700"}),
    ], style={"marginBottom": "10px", "color": NAV})

    cols = ["clause", "heading", "status", "evidence", "conflicts", "mapping", "signoff"]
    table = themed_table(
        data=clause_rows(view, latest),
        columns=[{"name": c.title(), "id": c} for c in cols],
        id="iatf-clause-table", row_selectable="single", selected_rows=[], page_size=20,
    )
    button = {"padding": "9px 18px", "borderRadius": "8px", "cursor": "pointer", "fontWeight": "700",
              "marginRight": "10px", "border": f"1px solid {NAV}", "background": "white", "color": NAV}
    panel = card_panel(html.Div([
        html.Div("Sign off the selected clause's evidence", style={"fontSize": "15px", "fontWeight": "700",
                                                                    "color": NAV, "marginBottom": "12px"}),
        html.Button("Accept", id="iatf-accept-btn",
                    style={**button, "background": f"linear-gradient(135deg,{TEAL},{NAV})", "color": "white",
                           "border": "none"}),
        html.Button("Reject", id="iatf-reject-btn", style=button),
        html.Button("Needs more evidence", id="iatf-more-btn", style=button),
        html.Div(id="iatf-signoff-result", style={"marginTop": "12px", "color": GOOD, "fontWeight": "600"}),
    ]))
    return html.Div([
        section_title("IATF 16949 Audit", view.get("disclaimer", "")),
        summary, table, panel,
    ])


def sign_off(trigger_id: str, selected_rows, rows) -> str:
    decision = _DECISIONS.get(trigger_id)
    if not decision:
        return ""
    if not selected_rows:
        return "Select a clause first."
    selected = selected_rows[0]
    row = selected if isinstance(selected, dict) else rows[selected]
    result = _post(f"/audit/iatf/clauses/{row['clause']}/signoff", {"decision": decision})
    if e := http_error(result):
        return f"⚠ Sign-off failed — {e}"
    return f"✅ Clause {row['clause']}: {decision} recorded (#{result.get('seq')})."


@callback(
    Output("iatf-signoff-result", "children"),
    Input("iatf-accept-btn", "n_clicks"),
    Input("iatf-reject-btn", "n_clicks"),
    Input("iatf-more-btn", "n_clicks"),
    State("iatf-clause-table", "selectedRows"),
    State("iatf-clause-table", "rowData"),
    prevent_initial_call=True,
)
def handle_signoff(_a, _r, _m, selected_rows, rows):
    from dash import ctx

    return sign_off(ctx.triggered_id, selected_rows, rows)
