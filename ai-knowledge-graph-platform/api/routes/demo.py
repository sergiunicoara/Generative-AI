"""GET /demo — self-contained chat UI for live demos (dev mode only)."""

from __future__ import annotations
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from graphrag.core.config import get_settings

router = APIRouter()

_GRAPH_HTML_PATHS = {
    "automotive": "graphify-out/graph.html",
    "marketing":  "data/wpp_demo/graphify-out/graph.html",
}

_TENANT_META = {
    "automotive": {
        "title": "GraphRAG — IATF 16949",
        "badge": "tenant: automotive · mode: hybrid",
        "subtitle_ro": "Pune o întrebare despre corpusul de 30 de documente automotive",
        "subtitle_en": "Ask a question about the 30-document automotive corpus",
        "placeholder_ro": "Scrie o întrebare despre documentele IATF 16949...",
        "placeholder_en": "Ask a question about the IATF 16949 documents...",
        "suggestions_ro": [
            "Care este ținta procentuală pentru rata de livrare la timp a furnizorilor?",
            "Cu ce frecvență trebuie efectuată reevaluarea furnizorilor activi, conform MC-01 și PQ-07?",
            "Ce consecință apare dacă rata de neconformitate a unui furnizor PlastiAuto depășește 1%, și care este ținta procentuală pentru livrarea la timp?",
            "Câte oferte competitive sunt necesare pentru aprobarea unui furnizor nou?",
        ],
        "suggestions_en": [
            "What is the on-time delivery percentage target for suppliers?",
            "How often must active suppliers be re-evaluated, per Quality Manual MC-01 and procedure PQ-07?",
            "What happens if a PlastiAuto supplier's non-conformity rate exceeds 1%, and what is the on-time delivery target?",
            "How many competitive offers are required to approve a new supplier?",
        ],
    },
    "marketing": {
        "title": "GraphRAG — AdTech Compliance",
        "badge": "tenant: marketing · mode: hybrid",
        "subtitle_ro": "Pune o întrebare despre campania Nova Beverages Q3 2024",
        "subtitle_en": "Ask a question about the Nova Beverages Q3 2024 campaign corpus",
        "placeholder_ro": "Scrie o întrebare despre documentele campaniei...",
        "placeholder_en": "Ask a question about the campaign documents...",
        "suggestions_ro": [
            "Ce categorii de reclame sunt excluse conform SOW-ului Nova Beverages?",
            "Putem rula plasamente pentru pariuri sportive în Germania pentru campania EU Q3 SummerRush?",
            "Care document are cea mai înaltă autoritate asupra deciziilor de campanie?",
            "Ce conflicte deschise există în tenant-ul marketing?",
        ],
        "suggestions_en": [
            "What ad categories are excluded under the Nova Beverages SOW?",
            "Can we run sports-betting placements in Germany for the EU Q3 SummerRush campaign?",
            "Which document has the highest authority over campaign decisions?",
        ],
    },
}

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ro">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f1117; color: #e2e8f0; min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
    padding: 40px 20px;
  }
  header { text-align: center; margin-bottom: 28px; }
  header h1 { font-size: 1.5rem; font-weight: 600; color: #f8fafc; }
  header p  { font-size: 0.85rem; color: #64748b; margin-top: 6px; }
  .badge {
    display: inline-block; background: #1e293b; border: 1px solid #334155;
    border-radius: 999px; padding: 3px 12px; font-size: 0.75rem; color: #94a3b8;
    margin-top: 10px;
  }
  .lang-toggle {
    display: flex; gap: 4px; margin-top: 14px; justify-content: center;
  }
  .lang-btn {
    padding: 4px 14px; border-radius: 6px; border: 1px solid #334155;
    background: #1e293b; color: #64748b; font-size: 0.78rem; cursor: pointer;
  }
  .lang-btn.active { background: #2563eb; border-color: #2563eb; color: #fff; }
  #chat {
    width: 100%; max-width: 780px;
    display: flex; flex-direction: column; gap: 16px;
    flex: 1; min-height: 200px;
  }
  .bubble {
    padding: 14px 18px; border-radius: 12px;
    font-size: 0.9rem; line-height: 1.6; max-width: 90%;
  }
  .bubble.user {
    background: #1e40af; color: #eff6ff;
    align-self: flex-end; border-bottom-right-radius: 4px;
  }
  .bubble.assistant {
    background: #1e293b; color: #e2e8f0;
    align-self: flex-start; border-bottom-left-radius: 4px;
    border: 1px solid #334155;
  }
  .bubble.thinking {
    background: #1e293b; color: #64748b;
    align-self: flex-start; border-bottom-left-radius: 4px;
    border: 1px dashed #334155; font-style: italic;
  }
  .citations {
    margin-top: 10px; padding-top: 10px;
    border-top: 1px solid #334155;
    font-size: 0.78rem; color: #64748b;
  }
  .citations span {
    display: inline-block; background: #0f172a; border: 1px solid #334155;
    border-radius: 4px; padding: 2px 7px; margin: 2px 3px 2px 0;
    font-family: monospace;
  }
  .latency { font-size: 0.75rem; color: #475569; margin-top: 6px; }
  .trace-step {
    font-size: 0.82rem; color: #94a3b8; padding: 2px 0;
    animation: fadein 0.3s ease;
  }
  @keyframes fadein { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; } }
  #input-row {
    width: 100%; max-width: 780px; margin-top: 24px;
    display: flex; gap: 10px;
  }
  #question {
    flex: 1; padding: 12px 16px;
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; color: #f1f5f9; font-size: 0.9rem; outline: none;
  }
  #question:focus { border-color: #3b82f6; }
  #question::placeholder { color: #475569; }
  #send {
    padding: 12px 22px; background: #2563eb; color: #fff;
    border: none; border-radius: 10px; font-size: 0.9rem;
    cursor: pointer; font-weight: 500; white-space: nowrap;
  }
  #send:disabled { background: #1e3a5f; color: #475569; cursor: not-allowed; }
  #send:hover:not(:disabled) { background: #1d4ed8; }
  .suggestions {
    width: 100%; max-width: 780px; margin-bottom: 16px;
    display: flex; flex-wrap: wrap; gap: 8px;
  }
  .suggestion {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 7px 12px; font-size: 0.78rem;
    color: #94a3b8; cursor: pointer; transition: border-color 0.15s;
  }
  .suggestion:hover { border-color: #3b82f6; color: #e2e8f0; }
  .tabs {
    display: flex; gap: 4px; margin-bottom: 20px;
    width: 100%; max-width: 780px;
  }
  .tab-btn {
    padding: 7px 20px; border-radius: 8px; border: 1px solid #334155;
    background: #1e293b; color: #64748b; font-size: 0.82rem; cursor: pointer;
    font-weight: 500; transition: all 0.15s;
  }
  .tab-btn.active { background: #2563eb; border-color: #2563eb; color: #fff; }
  .tab-btn:hover:not(.active) { border-color: #3b82f6; color: #e2e8f0; }
  #graph-panel {
    width: 100%; max-width: 780px; display: none;
    border: 1px solid #334155; border-radius: 12px; overflow: hidden;
  }
  #graph-panel iframe {
    width: 100%; height: 620px; border: none; display: block;
  }
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>{subtitle_en}</p>
  <span class="badge">{badge}</span>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('chat')">💬 Chat</button>
  <button class="tab-btn" onclick="showTab('graph')">🕸️ Knowledge Graph</button>
</div>

<div id="chat-panel">
  <div class="suggestions" id="suggestions"></div>
  <div id="chat"></div>
  <div id="input-row">
    <input id="question" type="text" placeholder="{placeholder_en}" onkeydown="if(event.key==='Enter')sendQuery()">
    <button id="send" onclick="sendQuery()">Send</button>
  </div>
</div>

<div id="graph-panel">
  <iframe src="/demo/graph?tenant={tenant}" loading="lazy"></iframe>
</div>

<script>
const SUGGESTIONS = {suggestions_en_js};
let token = null;

function renderSuggestions() {
  document.getElementById('suggestions').innerHTML = SUGGESTIONS
    .map(s => `<div class="suggestion" onclick="ask(this.innerText)">${s}</div>`)
    .join('');
}

async function getToken() {
  if (token) return token;
  const r = await fetch('/auth/dev-token', {method: 'POST'});
  const d = await r.json();
  token = d.access_token;
  return token;
}

function addBubble(text, cls, id) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'bubble ' + cls;
  if (id) div.id = id;
  div.innerHTML = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function sendQuery() {
  const input = document.getElementById('question');
  const q = input.value.trim();
  if (!q) return;
  input.value = '';
  await ask(q);
}

async function ask(question) {
  const send = document.getElementById('send');
  send.disabled = true;
  document.getElementById('question').value = '';

  addBubble(question, 'user');

  // Thinking bubble (static label)
  const thinkId = 'think-' + Date.now();
  addBubble('Searching the graph…', 'thinking', thinkId);

  // Separate trace container — appended below the thinking bubble
  const traceId = 'trace-' + Date.now();
  const traceEl = document.createElement('div');
  traceEl.id = traceId;
  traceEl.style.cssText = 'padding: 0 18px 4px; max-width: 90%; align-self: flex-start;';
  document.getElementById('chat').appendChild(traceEl);

  let renderedCount = 0;
  function renderSteps(steps) {
    console.log('[demo] renderSteps', steps);
    for (let i = renderedCount; i < steps.length; i++) {
      const d = document.createElement('div');
      d.className = 'trace-step';
      d.textContent = steps[i];
      traceEl.appendChild(d);
    }
    renderedCount = steps.length;
    document.getElementById('chat').scrollTop = 99999;
  }

  const t0 = Date.now();
  try {
    const tok = await getToken();
    console.log('[demo] token', tok ? tok.slice(0,20)+'...' : 'NULL');
    const headers = {'Authorization': 'Bearer ' + tok, 'Content-Type': 'application/json'};

    const r = await fetch('/query', {
      method: 'POST', headers,
      body: JSON.stringify({question, mode: 'hybrid', tenant: '{tenant}'})
    });
    const postData = await r.json();
    const query_id = postData.query_id;
    console.log('[demo] query_id', query_id, 'status', postData.status);

    let result = null;
    for (let i = 0; i < 120; i++) {
      await new Promise(res => setTimeout(res, i < 4 ? 500 : 1000));
      const pr = await fetch('/query/' + query_id, {headers: {'Authorization': 'Bearer ' + tok}});
      result = await pr.json();
      console.log('[demo] poll', i, 'status', result.status, 'steps', result.steps?.length ?? 0);
      if (result.steps?.length) renderSteps(result.steps);
      if (result.status === 'completed' || result.status === 'error') break;
    }

    const latency = ((Date.now() - t0) / 1000).toFixed(1);
    await new Promise(res => setTimeout(res, 300));
    document.getElementById(thinkId)?.remove();

    if (!result || result.status !== 'completed') {
      addBubble('Error: response did not arrive in time.', 'assistant');
    } else {
      let html = result.answer || '(no answer)';
      if (result.citations?.length) {
        const unique = [...new Set(result.citations)];
        const chips = unique.map(c => `<span>${c}</span>`).join('');
        html += `<div class="citations">Sources: ${chips}</div>`;
      }
      html += `<div class="latency">⏱ ${latency}s</div>`;
      addBubble(html, 'assistant');
    }
  } catch(e) {
    console.error('[demo] error', e);
    document.getElementById(thinkId)?.remove();
    addBubble('Error: ' + e.message, 'assistant');
  }
  send.disabled = false;
}

function showTab(name) {
  const chatPanel  = document.getElementById('chat-panel');
  const graphPanel = document.getElementById('graph-panel');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  if (name === 'graph') {
    chatPanel.style.display  = 'none';
    graphPanel.style.display = 'block';
    document.querySelectorAll('.tab-btn')[1].classList.add('active');
  } else {
    chatPanel.style.display  = '';
    graphPanel.style.display = 'none';
    document.querySelectorAll('.tab-btn')[0].classList.add('active');
  }
}

// Init
renderSuggestions();
</script>
</body>
</html>
"""


@router.get("/demo/graph", response_class=HTMLResponse, include_in_schema=False)
async def demo_graph(tenant: str = Query(default=None)):
    """Serves the knowledge graph HTML visualization for the given tenant."""
    if get_settings().env != "development":
        raise HTTPException(status_code=404, detail="Not found")
    if tenant is None:
        tenant = os.environ.get("GRAPHRAG_DEFAULT_TENANT", "automotive")
    path = _GRAPH_HTML_PATHS.get(tenant, _GRAPH_HTML_PATHS["automotive"])
    if not os.path.exists(path):
        return HTMLResponse("<p style='color:#94a3b8;font-family:sans-serif;padding:40px'>Graph not generated yet. Run: graphify cluster-only .</p>")
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@router.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def demo_ui(tenant: str = Query(default=None)):
    """Chat UI for live demos. Only available in development mode."""
    if get_settings().env != "development":
        raise HTTPException(status_code=404, detail="Not found")
    if tenant is None:
        tenant = os.environ.get("GRAPHRAG_DEFAULT_TENANT", "automotive")
    meta = _TENANT_META.get(tenant, _TENANT_META["automotive"])
    import json
    html = (
        _HTML_TEMPLATE
        .replace("{tenant}", tenant)
        .replace("{title}", meta["title"])
        .replace("{badge}", meta["badge"])
        .replace("{subtitle_en}", meta["subtitle_en"])
        .replace("{placeholder_en}", meta["placeholder_en"])
        .replace("{suggestions_en_js}", json.dumps(meta["suggestions_en"], ensure_ascii=False))
    )
    return HTMLResponse(html)
