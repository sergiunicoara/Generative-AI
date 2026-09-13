"""Read-only, tenant-scoped Energy Asset & Maintenance Intelligence POC API."""

from dataclasses import asdict
import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from api.auth.dependencies import get_current_user, get_tenant, require_scope
from graphrag.domains.energy.demo import EnergyDemoService
from graphrag.domains.energy.publication import PublicationRollbackError
from graphrag.domains.energy.workflow import MaintenanceWorkflow, WorkflowTransitionError

router = APIRouter()


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


class WorkflowTransitionRequest(BaseModel):
    to_state: str = Field(pattern="^(approved|rejected|completed|cancelled)$")
    reason: str = Field(min_length=1, max_length=500)
    expected_version: int = Field(ge=0)
    command_id: str = Field(min_length=1, max_length=128)

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
    return f"""<!doctype html><html><head><title>Energy Asset Intelligence</title><style>
body{{margin:0;background:#f4f7fa;color:#102235;font:15px Segoe UI,system-ui,sans-serif}}header{{background:linear-gradient(115deg,#0b2035,#155ba4);color:#fff;padding:28px 7%;display:flex;justify-content:space-between}}h1{{margin:0;font-size:28px}}main{{max-width:1180px;margin:26px auto;padding:0 22px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.layout{{display:grid;grid-template-columns:1fr 2fr;gap:18px;margin-top:18px}}.card,.panel{{background:#fff;border:1px solid #d5e1ec;border-radius:12px;padding:18px;box-shadow:0 3px 12px #1c34470d}}.label{{font-size:12px;color:#5e7285;text-transform:uppercase}}.metric{{font-size:28px;font-weight:700;margin:8px 0}}.red{{color:#c7352c}}.green{{color:#137a53}}button{{width:100%;text-align:left;margin:8px 0;padding:13px;border:1px solid #d5e1ec;border-radius:9px;background:#fff;font:inherit;cursor:pointer}}button.active,button:hover{{border-color:#4a91dc;background:#edf6ff}}.pill{{float:right;padding:4px 8px;border-radius:12px;font-size:11px;font-weight:700}}.critical{{background:#fce9e7;color:#a5211d}}.unknown{{background:#edf1f5;color:#607385}}.answer{{margin-top:14px;padding:16px;border-left:4px solid #1069c7;background:#f1f7fc;line-height:1.5}}.evidence{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}}.evidence div{{border:1px solid #d5e1ec;border-radius:8px;padding:11px}}.evidence b{{display:block;font-size:12px;color:#476175}}details{{margin-top:18px}}pre{{white-space:pre-wrap;word-break:break-word;background:#102235;color:#e7f1fb;padding:14px;border-radius:8px;font-size:12px}}small{{color:#5e7285}}@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,1fr)}}.layout,.evidence{{grid-template-columns:1fr}}}}</style></head><body>
<header><div><h1>Energy Asset Intelligence</h1><div>Maintenance review workspace · North Sea Demonstration Wind Farm</div></div><div>Synthetic advisory demo</div></header><main>
<section class="grid" id="tiles"></section>
<section class="layout"><aside class="panel"><div class="label">Asset overview</div><h2>Maintenance priorities</h2><div id="priorities"></div><hr><b>Guidance history</b><div id="guidance"></div></aside>
<section class="panel"><div class="label">Operations question</div><h2 id="question">Which assets need maintenance review?</h2><div id="answer" class="answer"></div><div id="evidence" class="evidence"></div><details><summary>Why am I seeing this?</summary><p>The recommendation is built from mapped work orders, RDF evidence, and a version-controlled SPARQL query. This is the technical trail behind the operational answer.</p><pre id="technical"></pre></details></section></section><p><small>All records are synthetic. This workspace is advisory and does not control equipment.</small></p></main>
<script>const data={data};
const s=data.summary,esc=t=>String(t).replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
const reviewed=(data.current.query_rows||[]).map(r=>String(r.asset).split('/').pop());
document.getElementById('tiles').innerHTML=[
  ['Assets under review',s.assets_under_review,reviewed.length?esc(reviewed.join(', '))+' need attention':'None above threshold',s.assets_under_review?'red':'green'],
  ['Open work orders',s.open_work_orders,'Covered by '+esc(s.authoritative_bulletin),''],
  ['Evidence coverage',s.assets_with_telemetry+' / '+s.assets_total,'Assets with telemetry',''],
  ['Data quality',data.validation.conforms?'Unchecked':'Checked','Invalid records rejected','green']
].map(([l,m,sub,cls])=>'<div class="card"><div class="label">'+l+'</div><div class="metric '+cls+'">'+esc(m)+'</div>'+sub+'</div>').join('');
document.getElementById('priorities').innerHTML=
  '<button class="active" onclick="show(\\'current\\',this)"><b>'+(reviewed.length?esc(reviewed.join(', ')):'No assets')+'</b><span class="pill '+(reviewed.length?'critical':'unknown')+'">'+(reviewed.length?'Review now':'Within threshold')+'</span><br><small>'+esc(data.current.answer.slice(0,90))+'</small></button>'+
  '<button onclick="show(\\'incomplete\\',this)"><b>'+s.assets_blocked_on_evidence+' assets</b><span class="pill unknown">Evidence missing</span><br><small>No maintenance conclusion</small></button>';
document.getElementById('guidance').innerHTML='<p>'+(data.bulletins||[]).map(b=>esc(b.bulletinId)+': '+esc(b.componentType)+' threshold '+esc(b.threshold)+'<br><small>from '+esc(b.validFrom)+(b.validTo?' until '+esc(b.validTo):' (current)')+'</small>').join('<br>')+'</p>';
function show(key,button){{let d=key==='current'?data.current:data.incomplete;document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));button.classList.add('active');document.getElementById('question').textContent=key==='current'?'Which assets need maintenance review?':'Where is further evidence required?';document.getElementById('answer').textContent=d.answer;document.getElementById('evidence').innerHTML=(d.evidence||[]).map(e=>'<div><b>'+esc(e.source_type.replaceAll('_',' '))+'</b>'+esc(e.value)+'<br><small>'+esc(e.field_or_span)+'</small></div>').join('');document.getElementById('technical').textContent=JSON.stringify({{bulletin:d.authoritative_bulletin,query_rows:d.query_rows||'Not required',answer_source:d.answer_source||'Evidence contract',validation:data.validation}},null,2)}}
show('current',document.querySelector('#priorities button'));</script></body></html>"""

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
