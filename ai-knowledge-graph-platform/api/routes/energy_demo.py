"""Read-only, tenant-scoped Energy Asset & Maintenance Intelligence POC API."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse

from api.auth.dependencies import get_tenant
from graphrag.domains.energy.demo import EnergyDemoService

router = APIRouter()
_service = EnergyDemoService()


@router.get("", response_class=HTMLResponse)
async def demo_page(tenant: str = Depends(get_tenant)):
    if tenant != "energy-demo":
        raise HTTPException(status_code=404, detail="Energy demonstration not found")
    return """<!doctype html><html><head><title>Energy Asset Intelligence</title>
    <style>body{font:16px system-ui;margin:3rem;max-width:900px;color:#172033}button{margin:.25rem;padding:.6rem;background:#075985;color:white;border:0;border-radius:4px}pre{padding:1rem;background:#f1f5f9;white-space:pre-wrap}small{color:#475569}</style></head>
    <body><h1>Energy Asset &amp; Maintenance Intelligence</h1><p>Synthetic, advisory demonstration with source evidence.</p>
    <div id='questions'></div><h2>Assessment</h2><pre id='result'>Choose a question.</pre>
    <details><summary>Technical view</summary><p>Each response contains the authoritative bulletin, current-as-of instant, mapping version, and typed evidence fields.</p></details>
    <script>const base=location.pathname.replace(/\\/$/, '');const out=document.getElementById('result');fetch(base+'/questions').then(r=>r.json()).then(qs=>{const root=document.getElementById('questions');qs.forEach(q=>{const b=document.createElement('button');b.textContent=q.question;b.onclick=()=>fetch(base+'/answer/'+q.id).then(r=>r.json()).then(x=>out.textContent=JSON.stringify(x,null,2));root.append(b)})})</script>
    <small>All sources are synthetic exports. No equipment actions are executed.</small></body></html>"""


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
