"""Read-only, tenant-scoped Energy Asset & Maintenance Intelligence POC API."""

from pathlib import Path
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse

from api.auth.dependencies import get_tenant
from graphrag.domains.energy.demo import EnergyDemoService

router = APIRouter()
_source_db = Path(__file__).resolve().parents[2] / "artifacts/energy-demo-sap.sqlite"
_service = EnergyDemoService(source_db=_source_db if _source_db.exists() else None)

def _dashboard_html() -> str:
    current = _service.answer("maintenance_review", tenant="energy-demo")
    incomplete = _service.answer("insufficient_evidence", tenant="energy-demo")
    data = json.dumps({"current": current, "incomplete": incomplete, "validation": _service.validate_candidate()}).replace("</", "<\\/")
    return f"""<!doctype html><html><head><title>Energy Asset Intelligence</title><style>
body{{margin:0;background:#f4f7fa;color:#102235;font:15px Segoe UI,system-ui,sans-serif}}header{{background:linear-gradient(115deg,#0b2035,#155ba4);color:#fff;padding:28px 7%;display:flex;justify-content:space-between}}h1{{margin:0;font-size:28px}}main{{max-width:1180px;margin:26px auto;padding:0 22px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.layout{{display:grid;grid-template-columns:1fr 2fr;gap:18px;margin-top:18px}}.card,.panel{{background:#fff;border:1px solid #d5e1ec;border-radius:12px;padding:18px;box-shadow:0 3px 12px #1c34470d}}.label{{font-size:12px;color:#5e7285;text-transform:uppercase}}.metric{{font-size:28px;font-weight:700;margin:8px 0}}.red{{color:#c7352c}}.green{{color:#137a53}}button{{width:100%;text-align:left;margin:8px 0;padding:13px;border:1px solid #d5e1ec;border-radius:9px;background:#fff;font:inherit;cursor:pointer}}button.active,button:hover{{border-color:#4a91dc;background:#edf6ff}}.pill{{float:right;padding:4px 8px;border-radius:12px;font-size:11px;font-weight:700}}.critical{{background:#fce9e7;color:#a5211d}}.unknown{{background:#edf1f5;color:#607385}}.answer{{margin-top:14px;padding:16px;border-left:4px solid #1069c7;background:#f1f7fc;line-height:1.5}}.evidence{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}}.evidence div{{border:1px solid #d5e1ec;border-radius:8px;padding:11px}}.evidence b{{display:block;font-size:12px;color:#476175}}details{{margin-top:18px}}pre{{white-space:pre-wrap;word-break:break-word;background:#102235;color:#e7f1fb;padding:14px;border-radius:8px;font-size:12px}}small{{color:#5e7285}}@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,1fr)}}.layout,.evidence{{grid-template-columns:1fr}}}}</style></head><body>
<header><div><h1>Energy Asset Intelligence</h1><div>Maintenance review workspace · North Sea Demonstration Wind Farm</div></div><div>Synthetic advisory demo</div></header><main>
<section class="grid"><div class="card"><div class="label">Assets under review</div><div class="metric red">1</div>WT-01 needs attention</div><div class="card"><div class="label">Open work orders</div><div class="metric">2</div>Gearbox-related work</div><div class="card"><div class="label">Evidence coverage</div><div class="metric">3 / 10</div>Assets with telemetry</div><div class="card"><div class="label">Data quality</div><div class="metric green">Checked</div>Invalid records rejected</div></section>
<section class="layout"><aside class="panel"><div class="label">Asset overview</div><h2>Maintenance priorities</h2><button class="active" onclick="show('current',this)"><b>WT-01</b><span class="pill critical">Review now</span><br><small>Gearbox temperature: 96°C</small></button><button onclick="show('incomplete',this)"><b>WT-04 — WT-10</b><span class="pill unknown">Evidence missing</span><br><small>No maintenance conclusion</small></button><hr><b>Guidance history</b><p>Before 1 Jun: R1 threshold 90°C<br>Current: R2 threshold 85°C</p></aside>
<section class="panel"><div class="label">Operations question</div><h2 id="question">Which assets need maintenance review?</h2><div id="answer" class="answer"></div><div id="evidence" class="evidence"></div><details><summary>Why am I seeing this?</summary><p>The recommendation is built from mapped work orders, RDF evidence, and a version-controlled SPARQL query. This is the technical trail behind the operational answer.</p><pre id="technical"></pre></details></section></section><p><small>All records are synthetic. This workspace is advisory and does not control equipment.</small></p></main>
<script>const data={data};function show(key,button){{let d=key==='current'?data.current:data.incomplete;document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));button.classList.add('active');document.getElementById('question').textContent=key==='current'?'Which assets need maintenance review?':'Where is further evidence required?';document.getElementById('answer').textContent=d.answer;document.getElementById('evidence').innerHTML=(d.evidence||[]).map(e=>'<div><b>'+e.source_type.replaceAll('_',' ')+'</b>'+e.value+'<br><small>'+e.field_or_span+'</small></div>').join('');document.getElementById('technical').textContent=JSON.stringify({{bulletin:d.authoritative_bulletin,query_rows:d.query_rows||'Not required',answer_source:d.answer_source||'Evidence contract',validation:data.validation}},null,2)}}show('current',document.querySelector('button'));</script></body></html>"""

@router.get("", response_class=HTMLResponse)
async def demo_page(tenant: str = Depends(get_tenant)):
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    return _dashboard_html()


@router.get("/questions")
async def questions(tenant: str = Depends(get_tenant)):
    if tenant != "energy-demo":
        return []
    return [{"id": key, "question": value} for key, value in _service.questions.items()]


@router.get("/answer/{question_id}")
async def answer(question_id: str, as_of: str | None = Query(default=None), tenant: str = Depends(get_tenant)):
    try:
        return _service.answer(question_id, tenant=tenant, as_of=as_of)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Unknown energy demonstration question") from exc


@router.get("/rdf")
async def rdf_export(tenant: str = Depends(get_tenant)):
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    return {"format": "text/turtle", "data": _service.export_turtle()}


@router.get("/validation")
async def validation(tenant: str = Depends(get_tenant)):
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    return _service.validate_candidate()
