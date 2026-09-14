"""Read-only, tenant-scoped Energy Asset & Maintenance Intelligence POC API."""

from dataclasses import asdict
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from api.auth.dependencies import get_current_user, get_tenant, require_scope
from graphrag.domains.energy.demo import EnergyDemoService
from graphrag.domains.energy.evidence_requests import (
    EvidenceRequestConflict, EvidenceRequestError, EvidenceRequestService, UnknownAssetError,
)
from graphrag.domains.energy.publication import PublicationRollbackError
from graphrag.domains.energy.workflow import MaintenanceWorkflow, WorkflowTransitionError

router = APIRouter()
DIAGNOSTICS_PATH = Path(__file__).resolve().parents[2] / "ontology" / "generated" / "energy" / "diagnostics.json"


def _capability_diagnostics() -> dict:
    metadata = {"artifact": "ontology/generated/energy/diagnostics.json", "model": "energy-asset-intelligence.yaml"}
    try:
        payload = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unavailable", "message": "Capability diagnostics are currently unavailable.", **metadata, "diagnostics": []}
    if not isinstance(payload, dict) or not isinstance(payload.get("diagnostics"), list) or not payload["diagnostics"]:
        return {"status": "unavailable", "message": "Capability diagnostics are currently unavailable.", **metadata, "diagnostics": []}
    diagnostics = []
    for item in payload["diagnostics"]:
        if not isinstance(item, dict):
            continue
        fidelity = item.get("fidelity")
        status = "Enforced" if fidelity == "preserved" and item.get("severity") == "info" else "Unsupported" if fidelity == "unenforceable" else "Partially enforced"
        diagnostics.append({
            "canonical_rule": str(item.get("element", "Unknown rule")), "target": str(item.get("target", "Unknown target")),
            "status": status, "message": str(item.get("message", "No further detail available.")),
            "impact": "The target cannot guarantee this rule natively." if status != "Enforced" else "The target preserves this semantic rule.",
            "mitigation": str(item.get("runtime_control") or "Use controlled loading and application-level validation."),
            "source": metadata["artifact"], "version": str(payload.get("version", "generated")),
        })
    if not diagnostics:
        return {"status": "unavailable", "message": "Capability diagnostics are currently unavailable.", **metadata, "diagnostics": []}
    return {"status": "ok", "message": "Generated capability diagnostics loaded.", **metadata, "version": str(payload.get("version", "generated")), "diagnostics": diagnostics}


def get_energy_service(request: Request) -> EnergyDemoService:
    service = getattr(request.app.state, "energy_demo_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Energy demo is starting")
    return service


def get_energy_workflow(request: Request) -> MaintenanceWorkflow:
    workflow = getattr(request.app.state, "energy_demo_workflow", None)
    if workflow is None:
        raise HTTPException(status_code=503, detail="Energy workflow is starting")
    return workflow


def get_energy_evidence_requests(request: Request) -> EvidenceRequestService:
    service = getattr(request.app.state, "energy_demo_evidence_requests", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Energy evidence requests are starting")
    return service


class WorkflowTransitionRequest(BaseModel):
    to_state: str = Field(pattern="^(approved|rejected|completed|cancelled)$")
    reason: str = Field(min_length=1, max_length=500)
    expected_version: int = Field(ge=0)
    command_id: str = Field(min_length=1, max_length=128)


class EvidenceRequestCreate(BaseModel):
    asset_id: str = Field(min_length=1, max_length=64)
    missing_field: str = Field(pattern="^(temperature_c|work_order_status)$")
    source_system: str = Field(pattern="^(Snowflake|SAP)$")
    owner: str = Field(min_length=1, max_length=200)
    priority: str = Field(pattern="^(low|medium|high)$")
    reason: str = Field(min_length=1, max_length=500)
    command_id: str = Field(min_length=1, max_length=128)


class EvidenceRequestTransitionRequest(BaseModel):
    to_state: str = Field(pattern="^(in_progress|fulfilled|cancelled)$")
    reason: str = Field(min_length=1, max_length=500)
    expected_version: int = Field(ge=0)
    command_id: str = Field(min_length=1, max_length=128)


def _evidence_remediation_html() -> str:
    return (
        '<section id="evidence-remediation" class="workflow" hidden>'
        '<div class="label">Evidence remediation</div>'
        '<p class="disclaimer">This creates a governed follow-up task. '
        "It does not infer or create the missing operational evidence.</p>"
        '<div id="remediation-assets"></div>'
        '<div id="remediation-status" role="status"></div>'
        '<div class="label history-label">Existing requests</div>'
        '<div id="remediation-requests"><small>No evidence requests yet.</small></div>'
        "</section>"
    )


def _evidence_remediation_script() -> str:
    # A plain (non-f) string: every brace here is literal JS/CSS syntax, with
    # no Python interpolation, so it is safe to build separately and splice
    # into `_dashboard_html`'s f-string as a single `{remediation_script}`
    # substitution -- avoiding hand-doubling braces throughout this block.
    return (
        "function remediationLabel(field){return field==='temperature_c'"
        "?'Request telemetry from Snowflake':'Request work-order status from SAP'}\n"
        "function remediationSystem(field){return field==='temperature_c'?'Snowflake':'SAP'}\n"
        "function renderRemediationAssets(){const rows=data.incomplete.query_rows||[];"
        "const box=document.getElementById('remediation-assets');"
        "if(!rows.length){box.innerHTML='<small>No assets are currently blocked on missing evidence.</small>';return}"
        "box.innerHTML=rows.map(r=>{const assetId=String(r.asset).split('/').pop();const fields=[];"
        "if(String(r.hasTemperature).toLowerCase()!=='true')fields.push('temperature_c');"
        "if(String(r.hasWorkOrder).toLowerCase()!=='true')fields.push('work_order_status');"
        "return '<div class=\"remediation-asset\"><b>'+esc(assetId)+'</b>'+fields.map(f=>"
        "'<button type=\"button\" onclick=\"openRemediationForm(this)\">'+remediationLabel(f)+'</button>'+"
        "'<form class=\"remediation-form\" hidden data-asset=\"'+esc(assetId)+'\" data-field=\"'+f+'\" "
        "onsubmit=\"return submitRemediation(event)\">'+"
        "'<label>Owner<input type=\"text\" name=\"owner\" required maxlength=\"200\"></label>'+"
        "'<label>Priority<select name=\"priority\"><option value=\"low\">Low</option>"
        "<option value=\"medium\" selected>Medium</option><option value=\"high\">High</option></select></label>'+"
        "'<label>Reason<textarea name=\"reason\" required maxlength=\"500\"></textarea></label>'+"
        "'<button type=\"submit\">Submit request</button></form>'"
        ").join('')+'</div>'}).join('')}\n"
        "function openRemediationForm(button){const form=button.nextElementSibling;form.hidden=!form.hidden}\n"
        "function csrfHeader(){const match=document.cookie.match(/(?:^|; )csrf_token=([^;]+)/);"
        "return match?{'x-csrf-token':decodeURIComponent(match[1])}:{}}\n"
        "async function submitRemediation(event){event.preventDefault();const form=event.target;"
        "const assetId=form.dataset.asset,field=form.dataset.field;"
        "const owner=form.owner.value.trim(),priority=form.priority.value,reason=form.reason.value.trim();"
        "const status=document.getElementById('remediation-status');"
        "const response=await fetch('/energy-demo/evidence-requests',{method:'POST',"
        "headers:Object.assign({'Content-Type':'application/json'},csrfHeader()),body:JSON.stringify({asset_id:assetId,"
        "missing_field:field,source_system:remediationSystem(field),owner,priority,reason,"
        "command_id:'ui-'+Date.now()+'-'+Math.random().toString(36).slice(2)})});"
        "if(!response.ok){let message='Request failed';try{message=(await response.json()).detail||message}"
        "catch(_){}status.textContent=message;status.className='workflow-error';return false}"
        "form.reset();form.hidden=true;status.textContent='Evidence request created.';"
        "status.className='workflow-success';loadRemediationRequests();return false}\n"
        "async function loadRemediationRequests(){const box=document.getElementById('remediation-requests');"
        "try{const response=await fetch('/energy-demo/evidence-requests');"
        "if(!response.ok)throw new Error('Unable to load evidence requests');"
        "const items=await response.json();"
        "if(!items.length){box.innerHTML='<small>No evidence requests yet.</small>';return}"
        "const details=await Promise.all(items.map(r=>fetch('/energy-demo/evidence-requests/'"
        "+encodeURIComponent(r.request_id)).then(x=>x.json())));"
        "box.innerHTML=details.map(d=>{const rec=d.request;"
        "const history=(d.transitions||[]).map(t=>'<li><small>'+esc(t.from_state)+' → '+esc(t.to_state)+"
        "' · '+esc(t.reason)+' · '+esc(t.changed_by)+'</small></li>').join('');"
        "return '<div class=\"remediation-request\"><b>'+esc(rec.asset_id)+' · '+esc(rec.missing_field)+"
        "'</b><span class=\"pill unknown\">'+esc(rec.state)+'</span><br><small>'+esc(rec.target_source_system)+"
        "' · owner '+esc(rec.owner)+' · priority '+esc(rec.priority)+'</small>'+(history?'<ol>'+history+'</ol>':'')"
        "+'</div>'}).join('')}catch(error){box.innerHTML='<small>'+esc(error.message)+'</small>'}}"
    )


def _publication_audit_html() -> str:
    return (
        '<section id="publication-audit" class="workflow">'
        '<div class="label">Publication and quarantine audit</div>'
        '<p class="disclaimer">Operators can inspect why a record was excluded. '
        "They cannot silently release invalid RDF from the dashboard.</p>"
        '<div id="publication-summary"><small>Loading publication status…</small></div>'
        '<div class="label history-label">Quarantined records</div>'
        '<div id="quarantine-list"><small>Loading…</small></div>'
        '<div class="label history-label">Publication history</div>'
        '<div id="publication-history"><small>Loading publication history…</small></div>'
        "</section>"
    )


def _publication_audit_script() -> str:
    # A plain (non-f) string, spliced into `_dashboard_html`'s f-string as a
    # single `{publication_audit_script}` substitution -- see
    # `_evidence_remediation_script()`'s docstring for why this avoids
    # hand-doubling braces in a large new JS block.
    #
    # Fetches client-side from the existing GET /publication and
    # GET /quarantine routes on page load -- never re-derives publication
    # state server-side into the initial `data` blob, so this panel cannot
    # silently drift from what those routes actually return. Read-only: no
    # control here can mutate published RDF, quarantine, or the active
    # version pointer.
    return (
        "async function loadPublicationAudit(){"
        "const summaryBox=document.getElementById('publication-summary');"
        "try{"
        "const [publicationResponse,quarantineResponse]=await Promise.all(["
        "fetch('/energy-demo/publication'),fetch('/energy-demo/quarantine')"
        "]);"
        "if(!publicationResponse.ok||!quarantineResponse.ok)throw new Error('Unable to load publication status');"
        "const publication=await publicationResponse.json();"
        "const quarantine=await quarantineResponse.json();"
        "renderPublicationAudit(publication,quarantine)"
        "}catch(error){"
        "summaryBox.textContent=error.message;"
        "summaryBox.className='workflow-error'"
        "}}\n"
        "function renderPublicationAudit(publication,quarantine){"
        "const records=quarantine.quarantined_records||[];"
        "document.getElementById('publication-summary').innerHTML="
        "'<div><b>Version</b> '+esc(publication.version_id)+'</div>'+"
        "'<div><b>Published at</b> '+esc(publication.published_at)+'</div>'+"
        "'<div><b>Published triples</b> '+esc(publication.published_triple_count)+'</div>'+"
        "'<div><b>Candidate records</b> '+esc(publication.candidate_record_count)+'</div>'+"
        "'<div><b>Quarantined records</b> '+esc(records.length)+'</div>'+"
        "'<div><b>Conforms</b> '+(publication.conforms?'Yes':'No')+'</div>';"
        "const listBox=document.getElementById('quarantine-list');"
        "if(!records.length){"
        "listBox.innerHTML='<small>No records are currently quarantined in the active publication.</small>';"
        "return}"
        "listBox.innerHTML=records.map(r=>"
        "'<div class=\"quarantine-record\"><b>'+esc(String(r.subject).split('/').pop())+'</b> "
        "— excluded from publication"
        "<details><summary>Technical detail</summary>'+"
        "'<div><b>Resource / focus node</b> '+esc(r.subject)+'</div>'+"
        "'<div><b>Reasons</b><ul>'+(r.reasons||[]).map(reason=>'<li>'+esc(reason)+'</li>').join('')+'</ul></div>'+"
        "'<div><b>Source</b> '+(r.source_type?esc(String(r.source_type).replaceAll('_',' ')):'Not available')+'</div>'+"
        "'<div><b>Provenance</b> '+(r.provenance?esc(r.provenance):'Not available')+'</div>'+"
        "'<div><b>Publication version</b> '+esc(quarantine.version_id)+'</div>'+"
        "'</details></div>'"
        ").join('')"
        "}\n"
        "async function loadPublicationHistory(){"
        "const box=document.getElementById('publication-history');"
        "try{"
        "const response=await fetch('/energy-demo/publication/history');"
        "if(!response.ok)throw new Error('Unable to load publication history');"
        "renderPublicationHistory(await response.json())"
        "}catch(error){"
        "box.textContent=error.message;"
        "box.className='workflow-error'"
        "}}\n"
        "function renderPublicationHistory(history){"
        "const box=document.getElementById('publication-history');"
        "const items=history.publications||[];"
        "if(!items.length){box.innerHTML='<small>No publication history is available.</small>';return}"
        "box.innerHTML='<ol>'+items.slice().reverse().map(p=>{"
        "const active=p.version_id===history.active_version_id;"
        "const kind=p.rolled_back_from?'Rollback':'Publication';"
        "return '<li><b>'+esc(kind)+'</b> '+(active?'<span class=\"pill unknown\">Active</span>':'')+"
        "'<br><small>'+esc(p.published_at)+' · version '+esc(p.version_id)+"
        "(p.rolled_back_from?' · restored from '+esc(p.rolled_back_from):'')+"
        "' · '+esc(p.quarantined_count)+' quarantined · '+(p.conforms?'conforms':'non-conforming')+"
        "'</small></li>'"
        "}).join('')+'</ol>'"
        "}"
    )


def _dashboard_html(service: EnergyDemoService) -> str:
    current = service.answer("maintenance_review", tenant="energy-demo")
    incomplete = service.answer("insufficient_evidence", tenant="energy-demo")
    # Summary tiles and the guidance-history panel are derived from the same
    # queries the answers are, rather than the hard-coded "1 / 2 / 3 of 10 /
    # Checked" figures and "Before 1 Jun: R1 90°C" panel they used to show
    # regardless of what the data said.
    summary = service.summary(tenant="energy-demo")
    validation = service.validate_candidate()
    data = json.dumps({
        "current": current,
        "incomplete": incomplete,
        "validation": validation,
        "summary": summary,
        "bulletins": service.bulletin_history(),
    }).replace("</", "<\\/")
    remediation_section = _evidence_remediation_html()
    remediation_script = _evidence_remediation_script()
    publication_audit_section = _publication_audit_html()
    publication_audit_script = _publication_audit_script()
    return f"""<!doctype html><html><head><title>Energy Asset Intelligence</title><style>
body{{margin:0;background:#f4f7fa;color:#102235;font:15px Segoe UI,system-ui,sans-serif}}header{{background:linear-gradient(115deg,#0b2035,#155ba4);color:#fff;padding:28px 7%;display:flex;justify-content:space-between}}h1{{margin:0;font-size:28px}}main{{max-width:1180px;margin:26px auto;padding:0 22px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.layout{{display:grid;grid-template-columns:1fr 2fr;gap:18px;margin-top:18px}}.card,.panel{{background:#fff;border:1px solid #d5e1ec;border-radius:12px;padding:18px;box-shadow:0 3px 12px #1c34470d}}.label{{font-size:12px;color:#5e7285;text-transform:uppercase}}.metric{{font-size:28px;font-weight:700;margin:8px 0}}.red{{color:#c7352c}}.green{{color:#137a53}}button{{width:100%;text-align:left;margin:8px 0;padding:13px;border:1px solid #d5e1ec;border-radius:9px;background:#fff;font:inherit;cursor:pointer}}button.active,button:hover{{border-color:#4a91dc;background:#edf6ff}}.pill{{float:right;padding:4px 8px;border-radius:12px;font-size:11px;font-weight:700}}.critical{{background:#fce9e7;color:#a5211d}}.unknown{{background:#edf1f5;color:#607385}}.answer{{margin-top:14px;padding:16px;border-left:4px solid #1069c7;background:#f1f7fc;line-height:1.5}}.evidence{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}}.evidence div{{border:1px solid #d5e1ec;border-radius:8px;padding:11px}}.evidence b{{display:block;font-size:12px;color:#476175}}.workflow{{margin-top:18px;padding:18px;border:1px solid #b7d4e7;border-radius:12px;background:#f8fcff}}.workflow-head{{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}}.state-badge{{padding:6px 10px;border-radius:14px;background:#fff1d6;color:#8b5b00;font-weight:700;font-size:12px;white-space:nowrap}}.state-approved{{background:#e4f7ed;color:#137a53}}.state-rejected{{background:#fce9e7;color:#a5211d}}.workflow-actions{{display:flex;gap:10px;margin-top:12px}}.workflow-actions button{{width:auto;flex:0 0 auto;margin:0;text-align:center}}.action.approve{{background:#137a53;color:#fff;border-color:#137a53}}.action.reject{{background:#fff;color:#a5211d;border-color:#d98b87}}.workflow-success{{margin-top:10px;color:#137a53;font-weight:700}}.workflow-error{{margin-top:10px;color:#a5211d;font-weight:700}}.history-label{{margin-top:16px;font-weight:700}}#workflow-history{{margin:8px 0 0;padding-left:24px}}#workflow-history li{{margin:8px 0}}details{{margin-top:18px}}pre{{white-space:pre-wrap;word-break:break-word;background:#102235;color:#e7f1fb;padding:14px;border-radius:8px;font-size:12px}}small{{color:#5e7285}}.disclaimer{{margin:6px 0 14px;color:#476175;font-size:13px}}.remediation-asset{{padding:10px 0;border-bottom:1px solid #e3ecf3}}.remediation-asset button{{width:auto;display:inline-block;margin:4px 8px 4px 0;padding:8px 12px}}.remediation-form{{margin-top:8px;padding:12px;border:1px solid #d5e1ec;border-radius:8px;background:#f8fcff}}.remediation-form label{{display:block;margin:8px 0;font-size:13px;color:#345}}.remediation-form input,.remediation-form select,.remediation-form textarea{{width:100%;margin-top:4px;padding:8px;border:1px solid #cfe0ee;border-radius:6px;font:inherit;box-sizing:border-box}}.remediation-request{{padding:10px 0;border-bottom:1px solid #e3ecf3}}#publication-summary div{{padding:2px 0}}.quarantine-record{{padding:10px 0;border-bottom:1px solid #e3ecf3}}.quarantine-record details{{margin-top:6px}}.quarantine-record ul{{margin:4px 0;padding-left:20px}}#publication-history ol{{margin:8px 0 0;padding-left:20px}}#publication-history li{{margin:8px 0}}@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,1fr)}}.layout,.evidence{{grid-template-columns:1fr}}.workflow-actions{{flex-direction:column}}}}
</style><header><div><h1>Energy Asset Intelligence</h1><div>Maintenance review workspace · North Sea Demonstration Wind Farm</div></div><div>Synthetic advisory demo</div></header><main>
<section class="grid" id="tiles"></section><details id="capability-gaps"><summary>Schema capability gaps</summary><p class="disclaimer">The canonical semantic model remains authoritative. This report identifies where a target system cannot enforce a rule directly and where validation or governance controls must compensate.</p><div id="capability-gaps-status"><small>Loading generated diagnostics…</small></div><div id="capability-gaps-list"></div></details>
<section class="layout"><aside class="panel"><div class="label">Asset overview</div><h2>Maintenance priorities</h2><div id="priorities"></div><hr><b>Guidance history</b><div id="guidance"></div></aside>
<section class="panel"><div class="label">Operations question</div><h2 id="question">Which assets need maintenance review?</h2><div id="answer" class="answer"></div><div id="evidence" class="evidence"></div><section id="workflow" class="workflow" hidden><div class="workflow-head"><div><div class="label">Governed review workflow</div><b id="workflow-title">WO-9001 · review required</b><div id="workflow-meta"><small>Loading current state…</small></div></div><div id="workflow-state" class="state-badge">REVIEW REQUIRED</div></div><div class="workflow-actions"><button id="approve" class="action approve" onclick="transitionState('approved')">Approve review</button><button id="reject" class="action reject" onclick="transitionState('rejected')">Reject review</button></div><div id="workflow-status" role="status"></div><div class="label history-label">Transition history</div><ol id="workflow-history"><li><small>No transitions recorded yet.</small></li></ol></section><details><summary>Why am I seeing this?</summary><p>The recommendation is built from mapped work orders, RDF evidence, and a version-controlled SPARQL query. This is the technical trail behind the operational answer.</p><pre id="technical"></pre></details>{remediation_section}</section></section>{publication_audit_section}<p><small>All records are synthetic. This workspace is advisory and does not control equipment.</small></p></main>
<script>const data={data};
const s=data.summary,esc=t=>String(t).replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
const reviewed=(data.current.query_rows||[]).map(r=>String(r.asset).split('/').pop());
const workflowOrder='WO-9001';
document.getElementById('tiles').innerHTML=[
  ['Assets under review',s.assets_under_review,reviewed.length?esc(reviewed.join(', '))+' needs attention':'None above threshold',s.assets_under_review?'red':'green'],
  ['Open work orders',s.open_work_orders,'Covered by '+esc(s.authoritative_bulletin),''],
  ['Evidence coverage',s.assets_with_telemetry+' / '+s.assets_total,'Assets with telemetry',''],
  ['Data quality',data.validation.conforms?'Unchecked':'Checked','Invalid records rejected','green']
].map(([l,m,sub,cls])=>'<div class="card"><div class="label">'+l+'</div><div class="metric '+cls+'">'+esc(m)+'</div>'+sub+'</div>').join('');
document.getElementById('priorities').innerHTML=
  '<button class="active" onclick="show(\\'current\\',this)"><b>'+(reviewed.length?esc(reviewed.join(', ')):'No assets')+'</b><span class="pill '+(reviewed.length?'critical':'unknown')+'">'+(reviewed.length?'Review now':'Within threshold')+'</span><br><small>'+esc(data.current.answer.slice(0,90))+'</small></button>'+
  '<button onclick="show(\\'incomplete\\',this)"><b>'+s.assets_blocked_on_evidence+' assets</b><span class="pill unknown">Evidence missing</span><br><small>No maintenance conclusion</small></button>';
document.getElementById('guidance').innerHTML='<p>'+(data.bulletins||[]).map(b=>esc(b.bulletinId)+': '+esc(b.componentType)+' threshold '+esc(b.threshold)+'<br><small>from '+esc(b.validFrom)+(b.validTo?' until '+esc(b.validTo):' (current)')+'</small>').join('<br>')+'</p>';
function renderWorkflow(lifecycle){{const box=document.getElementById('workflow'),state=document.getElementById('workflow-state'),meta=document.getElementById('workflow-meta'),history=document.getElementById('workflow-history'),approve=document.getElementById('approve'),reject=document.getElementById('reject');box.hidden=false;state.textContent=lifecycle.current_state.replaceAll('_',' ').toUpperCase();state.className='state-badge state-'+lifecycle.current_state;meta.innerHTML='<small>Object version '+esc(lifecycle.object_version)+' · Work order '+esc(lifecycle.work_order_id)+'</small>';history.innerHTML=(lifecycle.transitions||[]).length?lifecycle.transitions.map(t=>'<li><b>'+esc(t.from_state.replaceAll('_',' '))+' → '+esc(t.to_state.replaceAll('_',' '))+'</b><br><small>'+esc(t.reason)+' · '+esc(t.changed_by)+' · '+esc(t.changed_at)+'</small></li>').join(''):'<li><small>No transitions recorded yet.</small></li>';const review=lifecycle.current_state==='review_required';approve.hidden=!review;reject.hidden=!review}}
async function loadWorkflow(){{try{{const response=await fetch('/energy-demo/work-orders/'+encodeURIComponent(workflowOrder)+'/lifecycle');if(!response.ok)throw new Error('Unable to load workflow state');renderWorkflow(await response.json())}}catch(error){{document.getElementById('workflow').hidden=false;document.getElementById('workflow-status').textContent=error.message;document.getElementById('workflow-status').className='workflow-error'}}}}
async function transitionState(toState){{const response0=await fetch('/energy-demo/work-orders/'+encodeURIComponent(workflowOrder)+'/lifecycle');if(!response0.ok){{document.getElementById('workflow-status').textContent='Could not load the latest workflow version.';return}}const lifecycle=await response0.json();const reason=window.prompt('Reason for '+toState+' this review:');if(!reason||!reason.trim())return;const response=await fetch('/energy-demo/work-orders/'+encodeURIComponent(workflowOrder)+'/transition',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{to_state:toState,reason:reason.trim(),expected_version:lifecycle.object_version,command_id:'ui-'+Date.now()+'-'+Math.random().toString(36).slice(2)}})}});if(!response.ok){{let message='Transition failed';try{{message=(await response.json()).detail||message}}catch(_){{}}document.getElementById('workflow-status').textContent=message;document.getElementById('workflow-status').className='workflow-error';return}}renderWorkflow(await response.json());document.getElementById('workflow-status').textContent='Transition recorded successfully.';document.getElementById('workflow-status').className='workflow-success'}}
function show(key,button){{let d=key==='current'?data.current:data.incomplete;document.querySelectorAll('#priorities button').forEach(x=>x.classList.remove('active'));button.classList.add('active');document.getElementById('question').textContent=key==='current'?'Which assets need maintenance review?':'Where is further evidence required?';document.getElementById('answer').textContent=d.answer;document.getElementById('evidence').innerHTML=(d.evidence||[]).map(e=>'<div><b>'+esc(e.source_type.replaceAll('_',' '))+'</b>'+esc(e.value)+'<br><small>'+esc(e.field_or_span)+'</small></div>').join('');document.getElementById('technical').textContent=JSON.stringify({{bulletin:d.authoritative_bulletin,query_rows:d.query_rows||'Not required',answer_source:d.answer_source||'Evidence contract',validation:data.validation}},null,2);document.getElementById('workflow').hidden=key!=='current';if(key==='current')loadWorkflow();document.getElementById('evidence-remediation').hidden=key!=='incomplete';if(key==='incomplete'){{renderRemediationAssets();loadRemediationRequests()}}}}
async function loadCapabilityGaps(){{const status=document.getElementById('capability-gaps-status'),list=document.getElementById('capability-gaps-list');try{{const response=await fetch('/energy-demo/capability-diagnostics');const payload=await response.json();if(!response.ok||payload.status!=='ok')throw new Error(payload.message||'Capability diagnostics unavailable');status.textContent=payload.diagnostics.length+' generated diagnostics';list.innerHTML=payload.diagnostics.map(d=>'<article class="card"><b>'+esc(d.canonical_rule)+'</b><span class="pill '+(d.status==='Unsupported'?'critical':d.status==='Partially enforced'?'unknown':'state-approved')+'">'+esc(d.status)+'</span><p><b>Target:</b> '+esc(d.target)+'</p><p>'+esc(d.message)+'</p><p><b>Impact:</b> '+esc(d.impact)+'</p><p><b>Mitigation:</b> '+esc(d.mitigation)+'</p><small>'+esc(d.source)+' · version '+esc(d.version)+'</small></article>').join('')}}catch(error){{status.textContent=error.message;status.className='workflow-error'}}}}
loadCapabilityGaps();
{remediation_script}
{publication_audit_script}
show('current',document.querySelector('#priorities button'));
loadPublicationAudit();
loadPublicationHistory();</script></body></html>"""

@router.get("", response_class=HTMLResponse)
async def demo_page(tenant: str = Depends(get_tenant), service: EnergyDemoService = Depends(get_energy_service)):
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    return _dashboard_html(service)


@router.get("/questions")
async def questions(tenant: str = Depends(get_tenant), service: EnergyDemoService = Depends(get_energy_service)):
    if tenant != "energy-demo":
        return []
    return [{"id": key, "question": value} for key, value in service.questions.items()]


@router.get("/answer/{question_id}")
async def answer(
    question_id: str,
    as_of: str | None = Query(default=None, description="Valid time: the instant the question is about"),
    known_as: str | None = Query(default=None, description="Recorded time: only use facts recorded by this instant. Defaults to as_of, i.e. 'as we knew it then'."),
    tenant: str = Depends(get_tenant),
    service: EnergyDemoService = Depends(get_energy_service),
):
    try:
        return service.answer(question_id, tenant=tenant, as_of=as_of, known_as=known_as)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Unknown energy demonstration question") from exc


@router.get("/rdf")
async def rdf_export(tenant: str = Depends(get_tenant), service: EnergyDemoService = Depends(get_energy_service)):
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    return {"format": "text/turtle", "data": service.export_turtle()}


@router.get("/validation")
async def validation(tenant: str = Depends(get_tenant), service: EnergyDemoService = Depends(get_energy_service)):
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    return service.validate_candidate()


@router.get("/capability-diagnostics", dependencies=[Depends(require_scope("read"))])
async def capability_diagnostics(tenant: str = Depends(get_tenant)):
    """Read the compiler-generated report; never regenerates or mutates it."""
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    return _capability_diagnostics()


@router.get("/publication")
async def publication(tenant: str = Depends(get_tenant), service: EnergyDemoService = Depends(get_energy_service)):
    """The currently published version: id, publish timestamp,
    published/candidate record counts, and quarantined records -- the real
    state of the SHACL publication gate, not a synthetic probe (see
    GET /validation for that)."""
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    return asdict(service.publication_report())


@router.get("/quarantine")
async def quarantine(tenant: str = Depends(get_tenant), service: EnergyDemoService = Depends(get_energy_service)):
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    report = service.publication_report()
    return {"version_id": report.version_id, "quarantined_records": [asdict(r) for r in report.quarantined_records]}


@router.get("/publication/history")
async def publication_history(tenant: str = Depends(get_tenant), service: EnergyDemoService = Depends(get_energy_service)):
    """The durable publication/rollback timeline: every version ever
    published for this tenant, oldest first -- for auditing *which* version
    is active and *whether* it was reached through a rollback, not for
    inspecting quarantined-record detail (see GET /quarantine for that).
    Read-only: never touches the active-version pointer. Deliberately
    exposes only version id, timestamp, conformance, counts, and
    `rolled_back_from` -- no blob hash, no filesystem path, no secrets.
    """
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    history = await service.publication_history_durable()
    return {
        "active_version_id": service.publication_report().version_id,
        "publications": [
            {
                "version_id": report.version_id,
                "published_at": report.published_at,
                "conforms": report.conforms,
                "published_triple_count": report.published_triple_count,
                "quarantined_count": report.quarantined_count,
                "rolled_back_from": report.rolled_back_from,
            }
            for report in history
        ],
    }


@router.get("/work-orders/{work_order_id}/lifecycle")
async def work_order_lifecycle(
    work_order_id: str, tenant: str = Depends(get_tenant), workflow: MaintenanceWorkflow = Depends(get_energy_workflow),
):
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    try:
        return {
            "work_order_id": work_order_id,
            "current_state": await workflow.current_state(work_order_id),
            "object_version": (await workflow.current(work_order_id))[1],
            "transitions": [item.as_dict() for item in await workflow.history(work_order_id)],
            "rdf": (await workflow.rdf_projection(work_order_id)).serialize(format="turtle"),
        }
    except WorkflowTransitionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/work-orders/{work_order_id}/transition", dependencies=[Depends(require_scope("write"))])
async def transition_work_order(
    work_order_id: str,
    request: WorkflowTransitionRequest,
    tenant: str = Depends(get_tenant),
    user: dict = Depends(get_current_user),
    workflow: MaintenanceWorkflow = Depends(get_energy_workflow),
):
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    try:
        transition, response_json, _replayed = await workflow.transition(
            work_order_id,
            to_state=request.to_state,
            changed_by=str(user.get("sub", "unknown")),
            reason=request.reason,
            expected_version=request.expected_version,
            command_id=request.command_id,
        )
    except WorkflowTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(content=response_json, media_type="application/json")


@router.post("/rollback", dependencies=[Depends(require_scope("write"))])
async def rollback(
    version_id: str | None = Query(default=None), tenant: str = Depends(get_tenant),
    service: EnergyDemoService = Depends(get_energy_service),
):
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    try:
        report = await service.rollback_durable(version_id)
    except PublicationRollbackError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(report)


@router.get("/evidence-requests")
async def evidence_requests(
    tenant: str = Depends(get_tenant),
    evidence: EvidenceRequestService = Depends(get_energy_evidence_requests),
):
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    return [item.as_dict() for item in await evidence.list()]


@router.post("/evidence-requests", dependencies=[Depends(require_scope("write"))])
async def create_evidence_request(
    request: EvidenceRequestCreate,
    tenant: str = Depends(get_tenant),
    user: dict = Depends(get_current_user),
    service: EnergyDemoService = Depends(get_energy_service),
    evidence: EvidenceRequestService = Depends(get_energy_evidence_requests),
):
    """Create a governed evidence-remediation request.

    Re-validates the asset and the missing field against a fresh
    `insufficient_evidence` answer rather than trusting the request body --
    this is the boundary that keeps a remediation request from ever being
    fabricated evidence. Creating the request never touches published RDF,
    so the `insufficient_evidence` answer is unchanged by this call.
    """
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    query_rows = service.answer("insufficient_evidence", tenant=tenant).get("query_rows", [])
    try:
        _record, response_json, _replayed = await evidence.create(
            asset_id=request.asset_id, missing_field=request.missing_field,
            source_system=request.source_system, owner=request.owner, priority=request.priority,
            reason=request.reason, created_by=str(user.get("sub", "unknown")),
            command_id=request.command_id, graph=service.graph, query_rows=query_rows,
        )
    except UnknownAssetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EvidenceRequestConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EvidenceRequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(content=response_json, media_type="application/json", status_code=201)


@router.get("/evidence-requests/{request_id}")
async def evidence_request_detail(
    request_id: str, tenant: str = Depends(get_tenant),
    evidence: EvidenceRequestService = Depends(get_energy_evidence_requests),
):
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    try:
        record, history = await evidence.get(request_id)
        rdf = (await evidence.rdf_projection(request_id)).serialize(format="turtle")
    except EvidenceRequestError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "request": record.as_dict(),
        "transitions": [item.as_dict() for item in history],
        "rdf": rdf,
    }


@router.post("/evidence-requests/{request_id}/transition", dependencies=[Depends(require_scope("write"))])
async def transition_evidence_request(
    request_id: str,
    request: EvidenceRequestTransitionRequest,
    tenant: str = Depends(get_tenant),
    user: dict = Depends(get_current_user),
    evidence: EvidenceRequestService = Depends(get_energy_evidence_requests),
):
    """Advance a request's governance state.

    Never writes to published RDF, so fulfilling or cancelling a request
    cannot change the `insufficient_evidence` answer that motivated it.
    Mirrors `transition_work_order`: any rejection here (unknown request,
    stale version, illegal transition, reused command id) maps to 409,
    matching the existing workflow-transition convention.
    """
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    try:
        _record, response_json, _replayed = await evidence.transition(
            request_id, to_state=request.to_state,
            changed_by=str(user.get("sub", "unknown")), reason=request.reason,
            expected_version=request.expected_version, command_id=request.command_id,
        )
    except EvidenceRequestError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(content=response_json, media_type="application/json")
