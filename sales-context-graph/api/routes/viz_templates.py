"""Static page/style/script templates and brand tokens for api/routes/viz.py.

Split out of that file (Phase-9-era god-file cleanup) purely to separate
~1,500 lines of static HTML/CSS/JS content and the color-token machinery
that builds it from the actual route handlers. No behavior changes: every
name below is imported back into api.routes.viz and used exactly as it was
when defined there -- BRAND_PALETTE, TYPOGRAPHY, _root_css_vars(),
_js_color_constants(), and _legend_swatches_html() are the single source of
truth for every color/font token the templates below reference, and
_PAGE/_PANEL_PAGE/_BUYER_PORTAL_PAGE are plain (non-f) strings substituted
via .replace() at request time in viz.py, not interpolated here.
"""

from __future__ import annotations

# finding) ────────────────────────────────────────────────────────────────
# Before this, the same semantic color existed as up to 4 independent hex
# literals across this file: CSS strings (_SHARED_STYLES), inline style=
# attributes, string-concatenated innerHTML, and JS constants
# (polarityColor/entityColor/literalColor) driving the SVG graph -- no
# shared token, so the legend and the JS could (and did) drift apart. This
# one flat dict is now the *only* place a color is spelled out as a
# literal: every CSS rule below reads var(--color-*), the legend swatches
# read the same var(), and _js_color_constants() generates the JS object
# from these exact same values -- not hand-typed a second time. Palette
# values are Showpad's public brand tokens (brandcenter.showpad.com):
# primary Navy with 3 secondary tones, Brick/Plum accents, warm Sand/Cream/
# White neutrals.
BRAND_PALETTE = {
    # Raw Showpad brand tokens, verbatim.
    "navy": "#0d5189",
    "navy-dark": "#15254e",
    "navy-mid": "#0b4472",
    "navy-light": "#539dc4",
    "brick": "#dd7159",
    "plum": "#8c3fcc",
    "sand": "#e8ded4",
    "cream": "#f0ece8",
    "white": "#eeeeee",
    # Context Graph tab semantics (edge polarity + node kind) -- the exact
    # values the legend swatches and the JS polarityColor/entityColor/
    # literalColor constants below are generated from.
    "entity": "#0d5189",        # was Tailwind blue #2563eb -> Showpad Navy
    "literal": "#6b6258",       # was cool grey #6b7280 -> warmed neutral
    "affirmed": "#1f7a4d",      # was Tailwind green #16a34a -> warmed green
    "negated": "#dd7159",       # was Tailwind red #dc2626 -> Showpad Brick
    "hypothetical": "#c98a2c",  # was Tailwind amber #ca8a04 -> warmed amber
    # Interactive controls (buttons, active tab)
    "accent": "#8c3fcc",        # Plum
    "accent-hover": "#7530ad",  # darkened Plum for :hover
    "on-accent": "#ffffff",     # text/icon color drawn on an accent background
    # Layout / base text
    "border": "#ddd0c4",        # warmed border grey, was #ddd
    "surface": "#f0ece8",       # Cream, was #f3f4f6
    "surface-alt": "#e8ded4",   # Sand, was #e5e7eb
    "text": "#15254e",          # navy-dark, was implicit black/#555/#374151
    "text-muted": "#6b6258",
    # Status badges / ambiguity callouts
    "success-bg": "#dbeee1",
    "danger-bg": "#f7e3dd",
    "danger-text": "#b3492f",   # was #b91c1c / #991b1b (two slightly
                                 # different reds for the same "error" role)
    "warning-bg": "#f5e7c6",
    "warning-border": "#c98a2c",
    "legend-bg": "rgba(240,236,232,0.92)",  # Cream, translucent
}

# Showpad's type system (docs/evaluation.md): Nib Pro SemiBold for
# headlines (fallback Lora), Söhne for body (fallback Mona Sans), Söhne
# Mono for tabular/technical text (fallback Noto Sans Mono). No font files
# are bundled (no StaticFiles mount in this repo, docs/evaluation.md notes
# branding here is text-only) -- these resolve to the fallback unless the
# brand fonts happen to be installed locally, same caveat
# docs/architecture.html already carries for its own font stack.
TYPOGRAPHY = {
    "headline": "'Nib Pro SemiBold', Lora, serif",
    "body": "Söhne, 'Mona Sans', system-ui, sans-serif",
    "mono": "'Söhne Mono', 'Noto Sans Mono', monospace",
}


def _root_css_vars() -> str:
    """The single generation point for every --color-*/--font-* custom
    property this file's CSS reads. See BRAND_PALETTE's module comment."""
    color_lines = "\n".join(f"  --color-{key}: {value};" for key, value in BRAND_PALETTE.items())
    font_lines = "\n".join(f"  --font-{key}: {value};" for key, value in TYPOGRAPHY.items())
    return ":root {\n" + color_lines + "\n" + font_lines + "\n}"


def _js_color_constants() -> str:
    """Generates polarityColor/entityColor/literalColor straight from
    BRAND_PALETTE -- the JS the SVG graph renderer reads can no longer
    drift from the CSS custom properties or the legend swatches below; all
    three now trace back to the same dict keys instead of 3 independent
    hand-typed copies."""
    polarity_entries = ", ".join(
        f'{polarity}: "{BRAND_PALETTE[role]}"'
        for polarity, role in (("AFFIRMED", "affirmed"), ("NEGATED", "negated"), ("HYPOTHETICAL", "hypothetical"))
    )
    return (
        f"const polarityColor = {{ {polarity_entries} }};\n"
        f'const entityColor = "{BRAND_PALETTE["entity"]}";\n'
        f'const literalColor = "{BRAND_PALETTE["literal"]}";'
    )


def _legend_swatches_html() -> str:
    """Same source as _js_color_constants() above -- each swatch reads the
    CSS custom property (itself generated from BRAND_PALETTE), so a legend
    swatch and its corresponding JS graph color can never show a different
    hex for the same semantic role."""
    rows = (
        ("affirmed", "AFFIRMED"),
        ("negated", "NEGATED"),
        ("hypothetical", "HYPOTHETICAL"),
        ("entity", "entity node"),
        ("literal", "literal value node"),
    )
    return "\n".join(
        f'        <div><span class="swatch" style="background:var(--color-{key})"></span>{label}</div>'
        for key, label in rows
    )


# Shared between /viz and /viz/panel so both pages render JSON identically
# without duplicating the renderer.
_RENDER_JSON_JS = """
// Generic, recursive JSON -> readable-HTML renderer: scalars print inline,
// arrays of objects become tables (one row per item, columns from the union
// of keys), arrays of scalars become a comma list, objects become a
// definition-list-like block of "key: value" rows with nested rendering.
function renderJson(value) {
  const wrap = document.createElement("div");
  if (value === null || value === undefined) {
    wrap.className = "result-scalar"; wrap.textContent = "(none)"; return wrap;
  }
  if (typeof value !== "object") {
    wrap.className = "result-scalar";
    if (typeof value === "boolean") {
      const badge = document.createElement("span");
      badge.className = "badge " + value;
      badge.textContent = String(value);
      wrap.appendChild(badge);
    } else {
      wrap.textContent = String(value);
    }
    return wrap;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) { wrap.className = "result-scalar"; wrap.textContent = "(empty)"; return wrap; }
    const allScalar = value.every(v => v === null || typeof v !== "object");
    if (allScalar) {
      wrap.className = "result-scalar";
      wrap.textContent = value.join(", ");
      return wrap;
    }
    const cols = Array.from(value.reduce((set, row) => {
      if (row && typeof row === "object" && !Array.isArray(row)) Object.keys(row).forEach(k => set.add(k));
      return set;
    }, new Set()));
    const table = document.createElement("table");
    table.className = "result";
    const thead = document.createElement("tr");
    for (const c of cols) { const th = document.createElement("th"); th.textContent = c; thead.appendChild(th); }
    table.appendChild(thead);
    for (const row of value) {
      const tr = document.createElement("tr");
      if (row && typeof row === "object" && !Array.isArray(row)) {
        for (const c of cols) { const td = document.createElement("td"); td.appendChild(renderJson(row[c])); tr.appendChild(td); }
      } else {
        const td = document.createElement("td"); td.colSpan = cols.length || 1; td.appendChild(renderJson(row)); tr.appendChild(td);
      }
      table.appendChild(tr);
    }
    wrap.appendChild(table);
    return wrap;
  }
  for (const [k, v] of Object.entries(value)) {
    const keyEl = document.createElement("div");
    keyEl.className = "result-key";
    keyEl.textContent = k;
    wrap.appendChild(keyEl);
    wrap.appendChild(renderJson(v));
  }
  return wrap;
}
"""

_SHARED_STYLES = """
""" + _root_css_vars() + """
  body { font-family: var(--font-body); color: var(--color-text); }
  h3, h4 { font-family: var(--font-headline); }
  table.result { border-collapse: collapse; margin: 6px 0 14px 0; font-size: 12px; font-family: var(--font-mono); }
  table.result td, table.result th { border: 1px solid var(--color-border); padding: 4px 8px; text-align: left; vertical-align: top; }
  table.result th { background: var(--color-surface); }
  .result-key { font-weight: 600; color: var(--color-text); margin-top: 10px; font-family: var(--font-body); }
  .result-scalar { font-size: 13px; margin: 4px 0; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; background: var(--color-surface-alt); }
  .badge.true { background: var(--color-success-bg); color: var(--color-affirmed); }
  .badge.false { background: var(--color-danger-bg); color: var(--color-danger-text); }
"""

_PAGE = """<!doctype html>
<html lang="__LOCALE__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="manifest" href="/viz/manifest.webmanifest">
<title>Sales Context Graph Viz</title>
<style>
""" + _SHARED_STYLES + """
  body { margin: 0; display: flex; height: 100vh; }
  #panel { width: 380px; padding: 16px; box-sizing: border-box; border-right: 1px solid var(--color-border); overflow-y: auto; }
  #panel label { display: block; margin-top: 10px; font-size: 12px; color: var(--color-text-muted); }
  #panel input, #panel select, #panel textarea { width: 100%; padding: 6px; box-sizing: border-box; margin-top: 2px; font-family: inherit; }
  #panel button { margin-top: 14px; width: 100%; padding: 8px; background: var(--color-accent); color: var(--color-on-accent); border: none; border-radius: 4px; cursor: pointer; font-family: var(--font-body); }
  #panel button:hover { background: var(--color-accent-hover); }
  .quick-questions { display: grid; grid-template-columns: minmax(0, 1fr); gap: 6px; margin: 8px 0 12px; }
  .quick-questions > span { display: block; }
  .quick-question { width: 100% !important; min-height: 30px; margin: 0 !important; padding: 5px 9px !important; border-radius: 5px !important; text-align: left; }
  .audio-player-wrap { display: flex; flex-direction: column; align-items: flex-start; gap: 6px; width: auto; margin: 6px 0; }
  .audio-controls { display: flex; align-items: center; gap: 8px; width: 360px; max-width: 100%; }
  .audio-controls button { width: 34px !important; height: 30px; margin: 0 !important; padding: 0 !important; }
  .audio-progress { flex: 1; min-width: 120px; }
  .audio-time { min-width: 72px; font-size: 12px; font-variant-numeric: tabular-nums; }
  .audio-speed { display: none !important; }
  #panel input[type=checkbox] { width: auto; }
  #status, #qaStatus, #askStatus, #alertsStatus { margin-top: 10px; font-size: 12px; color: var(--color-danger-text); white-space: pre-wrap; }
  #reviewStatus { margin-top: 10px; font-size: 12px; color: var(--color-danger-text); white-space: pre-wrap; }
  #meta { margin-top: 14px; font-size: 12px; color: var(--color-text-muted); }
  #detail { margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--color-border); font-size: 12px; }
  #detail h4 { margin: 0 0 6px 0; }
  #main { flex: 1; position: relative; overflow: auto; }
  #graph { position: absolute; inset: 0; }
  svg { width: 100%; height: 100%; user-select: none; }
  .node circle { stroke: var(--color-on-accent); stroke-width: 1.5px; cursor: pointer; }
  .node text { font-size: 10px; pointer-events: none; font-family: var(--font-mono); }
  .edge-label { font-size: 9px; fill: var(--color-text-muted); pointer-events: none; font-family: var(--font-mono); }
  .legend { position: absolute; bottom: 10px; left: 10px; font-size: 11px; background: var(--color-legend-bg); padding: 8px; border-radius: 4px; }
  .legend div { display: flex; align-items: center; margin-bottom: 3px; }
  .legend span.swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }
  .tabs { display: flex; border-bottom: 1px solid var(--color-border); flex-wrap: wrap; }
  .tab { flex: 1; padding: 10px 6px; text-align: center; cursor: pointer; font-size: 12px; color: var(--color-text-muted); border: 0; background: transparent; border-bottom: 2px solid transparent; font-family: var(--font-body); }
  .tab.active { color: var(--color-accent); border-bottom-color: var(--color-accent); font-weight: 600; }
  .tab:focus-visible, #panel input:focus-visible, #panel select:focus-visible, #panel textarea:focus-visible, #panel button:focus-visible { outline: 3px solid var(--color-accent); outline-offset: 2px; }
  .tabpage { display: none; padding: 20px; }
  .tabpage.active { display: block; }
  .citation { font-size: 11px; color: var(--color-text-muted); margin: 2px 0; }
  .uncited { font-size: 11px; color: var(--color-danger-text); margin: 2px 0; }
  .ambiguity { background: var(--color-warning-bg); border: 1px solid var(--color-warning-border); padding: 8px; border-radius: 4px; margin: 6px 0; font-size: 12px; }
  .product-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }
  .product-card { border:1px solid var(--color-border); border-radius:6px; padding:12px; background:var(--color-surface); }
  .sr-only { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
  @media (max-width: 760px) { body { display:block; height:auto; min-height:100vh; } #panel { width:100%; border-right:0; border-bottom:1px solid var(--color-border); } #main { min-height:58vh; } #graph-page { min-height:58vh; } .tab { flex-basis:31%; } .tabpage { padding:12px; } }
</style>
</head>
<body>
<div id="panel">
  <label for="localeSelect" class="sr-only">Language</label>
  <select id="localeSelect" aria-label="Language"><option value="en">English</option><option value="ro">Română</option></select>
  <div class="tabs" role="tablist" aria-label="Sales Context Graph sections">
    <button class="tab active" id="graph-tab" role="tab" aria-selected="true" aria-controls="graph-page" data-tab="graph" data-i18n="graph">Context Graph</button>
    <button class="tab" id="qa-tab" role="tab" aria-selected="false" aria-controls="qa-page" data-tab="qa" data-i18n="browse">Browse Intents</button>
    <button class="tab" id="ask-tab" role="tab" aria-selected="false" aria-controls="ask-page" data-tab="ask" data-i18n="ask">Ask</button>
    <button class="tab" id="alerts-tab" role="tab" aria-selected="false" aria-controls="alerts-page" data-tab="alerts" data-i18n="alerts">Alerts</button>
    <button class="tab" id="review-tab" role="tab" aria-selected="false" aria-controls="review-page" data-tab="review" data-i18n="review">Review Console</button>
    <button class="tab" id="workflows-tab" role="tab" aria-selected="false" aria-controls="workflows-page" data-tab="workflows" data-i18n="workflows">Workflows</button>
  </div>

  <div id="graph-controls">
    <h3>Context Graph</h3>
    <label>Workspace ID
      <input id="workspaceId" value="ws-demo">
    </label>
    <label>API Key
      <input id="apiKey" type="password" placeholder="X-Api-Key">
    </label>
    <label>Subject ID (contact/account/etc.)
      <input id="subjectId" placeholder="optional">
    </label>
    <label>Conversation ID
      <input id="conversationId" placeholder="optional">
    </label>
    <label>Max nodes
      <input id="maxNodes" placeholder="default">
    </label>
    <button id="buildBtn">Build</button>
    <div id="status"></div>
    <div id="meta"></div>
    <div id="detail"></div>
  </div>

  <div id="qa-controls" style="display:none">
    <h3>Browse Intents</h3>
    <label>Workspace ID
      <input id="qaWorkspaceId" value="ws-demo">
    </label>
    <label>API Key
      <input id="qaApiKey" type="password" placeholder="X-Api-Key">
    </label>
    <label>Question
      <select id="qaSelect"></select>
    </label>
    <div id="qaFields"></div>
    <button id="qaRunBtn">Run</button>
    <div id="qaStatus"></div>
  </div>

  <div id="ask-controls" style="display:none">
    <h3>Ask</h3>
    <p style="font-size:12px;color:var(--color-text-muted)">Free-form question -&gt; natural-language intent layer (Increment 15). Requires LLM_PROVIDER configured server-side.</p>
    <label>Workspace ID
      <input id="askWorkspaceId" value="ws-demo">
    </label>
    <label>API Key
      <input id="askApiKey" type="password" placeholder="X-Api-Key">
    </label>
    <label>Question
      <textarea id="askQuestion" rows="3" placeholder="e.g. what objections has Volkswagen raised?"></textarea>
    </label>
    <div class="quick-questions" aria-label="Quick questions">
      <span style="font-size:12px;color:var(--color-text-muted)">Quick:</span>
      <button type="button" class="quick-question" data-question="What objections are currently open?"><b>1</b> What objections are currently open?</button>
      <button type="button" class="quick-question" data-question="Who have we not engaged in this opportunity?"><b>2</b> Who have we not engaged?</button>
      <button type="button" class="quick-question" data-question="What content should I send next?"><b>3</b> What content should I send?</button>
      <button type="button" class="quick-question" data-question="What changed since June 1, 2026?"><b>4</b> What changed since June 1, 2026?</button>
    </div>
    <label><input id="askNarrative" type="checkbox"> Include narrative summary (next Ask)</label>
    <label id="askVoiceControl"><input id="askVoice" type="checkbox"> Read answer aloud (optional TTS)</label>
    <details style="margin-top:10px">
      <summary style="font-size:12px;cursor:pointer">Optional context (ids the UI would already know)</summary>
      <label>Opportunity ID <input id="askOpportunityId"></label>
      <label>Seller ID <input id="askSellerId"></label>
      <label>Conversation ID <input id="askConversationId"></label>
      <label>Subject ID <input id="askSubjectId"></label>
      <label>Buyer Contact ID <input id="askBuyerContactId"></label>
    </details>
    <button id="askRunBtn">Ask</button>
    <div id="askStatus"></div>
  </div>

  <div id="alerts-controls" style="display:none">
    <h3>Alerts</h3>
    <p style="font-size:12px;color:var(--color-text-muted)">Proactive signals (Increment 17): single-threaded deals, unanswered objections, unopened content, unresolved conflicts, stalled deals.</p>
    <label>Workspace ID
      <input id="alertsWorkspaceId" value="ws-demo">
    </label>
    <label>API Key
      <input id="alertsApiKey" type="password" placeholder="X-Api-Key">
    </label>
    <label>Seller ID (optional, narrows to one rep's pipeline)
      <input id="alertsSellerId" placeholder="optional">
    </label>
    <button id="alertsRunBtn">Get digest</button>
    <div id="alertsStatus"></div>
  </div>

  <div id="review-controls" style="display:none">
    <h3>Review Console</h3>
    <p style="font-size:12px;color:var(--color-text-muted)">Resolve ambiguous mentions, contradictory claims, or inspect seller-wide objection patterns.</p>
    <label>Workspace ID
      <input id="reviewWorkspaceId" value="ws-demo">
    </label>
    <label>API Key
      <input id="reviewApiKey" type="password" placeholder="X-Api-Key">
    </label>
    <label>Reviewer ID
      <input id="reviewerId" placeholder="required for mention decisions">
    </label>
    <label>Review SLA (hours)
      <input id="reviewSlaHours" type="number" min="1" max="720" value="24">
    </label>
    <button id="reviewMentionsBtn">Load pending mentions</button>
    <button id="reviewAssignmentsBtn">Load my assignments</button>
    <label>Opportunity ID (conflict review)
      <input id="reviewOpportunityId" placeholder="opportunity id">
    </label>
    <button id="reviewConflictsBtn">Load open conflicts</button>
    <label>Seller ID (cross-deal aggregation)
      <input id="reviewSellerId" placeholder="seller id">
    </label>
    <button id="reviewObjectionsBtn">Load top objections</button>
    <div id="reviewStatus"></div>
  </div>

  <div id="workflows-controls" style="display:none">
    <h3>Sales workflows</h3>
    <p style="font-size:12px;color:var(--color-text-muted)">Readiness, Buyer Spaces, revenue outcomes and meeting preparation. Draft inputs are retained locally for offline reconnect.</p>
    <label>Workspace ID <input id="workflowWorkspaceId" value="ws-demo"></label>
    <label>API Key <input id="workflowApiKey" type="password" placeholder="X-Api-Key"></label>
    <label>Opportunity ID <input id="workflowOpportunityId" placeholder="required for Buyer Space and meeting brief"></label>
    <label>Seller ID <input id="workflowSellerId" placeholder="required for readiness"></label>
    <label>New Buyer Space title <input id="workflowSpaceTitle" placeholder="Mutual action plan"></label>
    <button id="workflowLoadBtn" type="button">Load workflow dashboard</button>
    <button id="workflowCreateSpaceBtn" type="button">Create Buyer Space</button>
    <div id="workflowStatus" role="status" aria-live="polite"></div>
  </div>
</div>

<div id="main">
  <div id="graph-page" class="tabpage active" role="tabpanel" aria-labelledby="graph-tab" style="position:absolute;inset:0;padding:0">
    <div id="graph">
      <svg id="svg"></svg>
      <div class="legend">
""" + _legend_swatches_html() + """
      </div>
    </div>
  </div>
  <div id="qa-page" class="tabpage" role="tabpanel" aria-labelledby="qa-tab">
    <div id="qaResult"></div>
  </div>
  <div id="ask-page" class="tabpage" role="tabpanel" aria-labelledby="ask-tab">
    <div id="askResult"></div>
  </div>
  <div id="alerts-page" class="tabpage" role="tabpanel" aria-labelledby="alerts-tab">
    <div id="alertsResult"></div>
  </div>
  <div id="review-page" class="tabpage" role="tabpanel" aria-labelledby="review-tab">
    <div id="reviewResult"></div>
  </div>
  <div id="workflows-page" class="tabpage" role="tabpanel" aria-labelledby="workflows-tab">
    <div id="workflowResult" class="product-grid"></div>
  </div>
</div>

<script>
if ('serviceWorker' in navigator) navigator.serviceWorker.register('/viz/service-worker.js').catch(() => {});
const localeSelect = document.getElementById('localeSelect');
const I18N = {
  en: {graph:'Context Graph', browse:'Browse Intents', ask:'Ask', alerts:'Alerts', review:'Review Console', workflows:'Workflows'},
  ro: {graph:'Graf de context', browse:'Intentii', ask:'Intreaba', alerts:'Alerte', review:'Consola de revizuire', workflows:'Fluxuri'}
};
function applyLocale(locale) {
  const labels = I18N[locale] || I18N.en;
  document.documentElement.lang = locale;
  document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = labels[el.dataset.i18n]; });
  localStorage.setItem('scgLocale', locale);
}
localeSelect.value = localStorage.getItem('scgLocale') || document.documentElement.lang || 'en';
applyLocale(localeSelect.value);
localeSelect.addEventListener('change', () => applyLocale(localeSelect.value));
/* ── Tabs ─────────────────────────────────────────────────────────────── */
const demoOpportunityId = __DEMO_OPPORTUNITY_ID_JSON__;
const TAB_NAMES = ["graph", "qa", "ask", "alerts", "review", "workflows"];
const DEMO_BROWSER_TTS_ENABLED = __DEMO_BROWSER_TTS_ENABLED__;
for (const tab of document.querySelectorAll(".tab")) {
  function activate() {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab").forEach(t => t.setAttribute("aria-selected", "false"));
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    const target = tab.dataset.tab;
    for (const name of TAB_NAMES) {
      document.getElementById(name + "-controls").style.display = name === target ? "" : "none";
      document.getElementById(name + "-page").classList.toggle("active", name === target);
    }
    if (target === "qa" && qaSelect.options.length === 0) loadIntents();
  }
  tab.addEventListener("click", activate);
  tab.addEventListener("keydown", event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const tabs = Array.from(document.querySelectorAll('.tab'));
    const current = tabs.indexOf(tab);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    tabs[next].focus(); tabs[next].click();
  });
}

/* ── Context Graph (unchanged from the original single-tab page) ────────── */
""" + _js_color_constants() + """

let nodes = [];
let edges = [];
let svgEl = document.getElementById("svg");

// Drag-to-reposition: pointer capture lives on svgEl itself (not the
// per-node <g>), because render() below does svgEl.innerHTML = "" on every
// frame — a capture held by a <g> would be silently dropped the instant a
// drag causes a re-render, killing the drag mid-motion.
let draggingNode = null;
let dragMoved = false;
let currentWorkspaceId = null;

svgEl.addEventListener("pointermove", (ev) => {
  if (!draggingNode) return;
  dragMoved = true;
  const rect = svgEl.getBoundingClientRect();
  const W = svgEl.clientWidth || 800, H = svgEl.clientHeight || 600;
  draggingNode.x = Math.max(30, Math.min(W - 30, ev.clientX - rect.left));
  draggingNode.y = Math.max(30, Math.min(H - 30, ev.clientY - rect.top));
  draggingNode.vx = 0;
  draggingNode.vy = 0;
  render(currentWorkspaceId);
});
svgEl.addEventListener("pointerup", () => { draggingNode = null; });
svgEl.addEventListener("pointercancel", () => { draggingNode = null; });

document.getElementById("buildBtn").addEventListener("click", build);

async function build() {
  const statusEl = document.getElementById("status");
  const metaEl = document.getElementById("meta");
  const detailEl = document.getElementById("detail");
  statusEl.textContent = "";
  metaEl.textContent = "";
  detailEl.innerHTML = "";

  const workspaceId = document.getElementById("workspaceId").value.trim();
  const apiKey = document.getElementById("apiKey").value.trim();
  const subjectId = document.getElementById("subjectId").value.trim();
  const conversationId = document.getElementById("conversationId").value.trim();
  const maxNodesRaw = document.getElementById("maxNodes").value.trim();

  if (!workspaceId) { statusEl.textContent = "Workspace ID is required."; return; }
  if (!apiKey) { statusEl.textContent = "API Key is required."; return; }

  const body = {};
  if (subjectId) body.subject_id = subjectId;
  if (conversationId) body.conversation_id = conversationId;
  if (maxNodesRaw) body.max_nodes = parseInt(maxNodesRaw, 10);

  let resp;
  try {
    resp = await fetch("/api/v1/context/build", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Workspace-Id": workspaceId, "X-Api-Key": apiKey },
      body: JSON.stringify(body),
    });
  } catch (e) {
    statusEl.textContent = "Request failed: " + e;
    return;
  }
  if (!resp.ok) {
    statusEl.textContent = "HTTP " + resp.status + ": " + (await resp.text());
    return;
  }
  const result = await resp.json();

  metaEl.innerHTML =
    "nodes_used: " + result.nodes_used + " / " + result.budget_max_nodes + "<br>" +
    "tokens_used: " + result.tokens_used + " / " + result.budget_max_tokens + "<br>" +
    "truncated: " + result.truncated + "<br>" +
    "claims: " + result.claims.length + "<br>" +
    "unresolved_mentions: " + result.unresolved_mention_ids.length + "<br>" +
    "conflicts: " + result.conflicts.length;

  buildGraph(result, workspaceId, apiKey);
}

function buildGraph(result, workspaceId, apiKey) {
  const nodeById = new Map();
  edges = [];

  function ensureNode(id, label, kind) {
    if (!nodeById.has(id)) {
      nodeById.set(id, { id, label, kind, x: Math.random() * 600 + 50, y: Math.random() * 400 + 50, vx: 0, vy: 0 });
    }
    return nodeById.get(id);
  }

  for (const claim of result.claims) {
    const subj = ensureNode(claim.subject_id, shorten(claim.subject_id), "entity");
    const objId = claim.object_id || ("lit:" + claim.claim_id);
    const objLabel = claim.object_value || shorten(claim.object_id);
    const obj = ensureNode(objId, objLabel, claim.object_id ? "entity" : "literal");
    edges.push({
      source: subj.id, target: obj.id,
      predicate: claim.predicate, polarity: claim.polarity,
      claimId: claim.claim_id, workspaceId, apiKey,
    });
  }

  nodes = Array.from(nodeById.values());
  currentWorkspaceId = workspaceId;
  runLayout();
  render(workspaceId);
}

function shorten(s) {
  if (!s) return "?";
  return s.length > 14 ? s.slice(0, 6) + "…" + s.slice(-4) : s;
}

function runLayout() {
  const W = svgEl.clientWidth || 800, H = svgEl.clientHeight || 600;
  const cx = W / 2, cy = H / 2;
  for (let iter = 0; iter < 300; iter++) {
    for (const a of nodes) {
      let fx = (cx - a.x) * 0.002, fy = (cy - a.y) * 0.002;
      for (const b of nodes) {
        if (a === b) continue;
        const dx = a.x - b.x, dy = a.y - b.y;
        const distSq = Math.max(dx * dx + dy * dy, 1);
        const force = 2500 / distSq;
        fx += (dx / Math.sqrt(distSq)) * force;
        fy += (dy / Math.sqrt(distSq)) * force;
      }
      a.vx = (a.vx + fx) * 0.8;
      a.vy = (a.vy + fy) * 0.8;
    }
    for (const e of edges) {
      const a = nodes.find(n => n.id === e.source), b = nodes.find(n => n.id === e.target);
      const dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const diff = (dist - 140) * 0.02;
      const ux = dx / dist, uy = dy / dist;
      a.vx += ux * diff; a.vy += uy * diff;
      b.vx -= ux * diff; b.vy -= uy * diff;
    }
    for (const n of nodes) {
      n.x += n.vx; n.y += n.vy;
      n.x = Math.max(30, Math.min(W - 30, n.x));
      n.y = Math.max(30, Math.min(H - 30, n.y));
    }
  }
}

function render(workspaceId) {
  svgEl.innerHTML = "";
  const ns = "http://www.w3.org/2000/svg";

  for (const e of edges) {
    const a = nodes.find(n => n.id === e.source), b = nodes.find(n => n.id === e.target);
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", a.x); line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
    line.setAttribute("stroke", polarityColor[e.polarity] || literalColor);
    line.setAttribute("stroke-width", "1.5");
    line.style.cursor = "pointer";
    line.addEventListener("click", () => showEvidence(e));
    svgEl.appendChild(line);

    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", (a.x + b.x) / 2);
    label.setAttribute("y", (a.y + b.y) / 2);
    label.setAttribute("class", "edge-label");
    label.textContent = e.predicate;
    svgEl.appendChild(label);
  }

  for (const n of nodes) {
    const g = document.createElementNS(ns, "g");
    g.setAttribute("class", "node");
    g.setAttribute("transform", "translate(" + n.x + "," + n.y + ")");

    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("r", n.kind === "entity" ? 8 : 5);
    circle.setAttribute("fill", n.kind === "entity" ? entityColor : literalColor);
    g.appendChild(circle);

    const text = document.createElementNS(ns, "text");
    text.setAttribute("x", 10);
    text.setAttribute("y", 4);
    text.textContent = n.label;
    g.appendChild(text);

    g.style.cursor = "grab";
    g.addEventListener("pointerdown", (ev) => {
      draggingNode = n;
      dragMoved = false;
      svgEl.setPointerCapture(ev.pointerId);
      ev.stopPropagation();
    });
    // click still fires after a drag's pointerup — dragMoved suppresses the
    // detail panel from popping open on every drag release.
    g.addEventListener("click", () => { if (!dragMoved) showNode(n); });
    svgEl.appendChild(g);
  }
}

async function showEvidence(edge) {
  const detailEl = document.getElementById("detail");
  detailEl.innerHTML = "<h4>Loading evidence…</h4>";
  try {
    const resp = await fetch("/api/v1/claims/" + encodeURIComponent(edge.claimId) + "/evidence", {
      headers: { "X-Workspace-Id": edge.workspaceId, "X-Api-Key": edge.apiKey },
    });
    const data = await resp.json();
    detailEl.innerHTML =
      "<h4>Claim: " + data.claim_id + "</h4>" +
      "<b>" + data.predicate + "</b> → " + (data.object_value || data.object_id) + "<br>" +
      "polarity: " + data.polarity + "<br>" +
      "speaker_role: " + data.speaker_role + "<br>" +
      "confidence: " + data.confidence + "<br>" +
      "adjudication: " + data.adjudication_status + "<br>" +
      "<blockquote style='margin:6px 0;padding:6px;background:var(--color-surface);'>" +
      (data.excerpt || "(no excerpt)") + "</blockquote>";
  } catch (e) {
    detailEl.textContent = "Failed to load evidence: " + e;
  }
}

function showNode(node) {
  const detailEl = document.getElementById("detail");
  detailEl.innerHTML = "<h4>Node</h4>id: " + node.id + "<br>kind: " + node.kind;
}

/* ── Browse Intents (Q&A / insights runner) ──────────────────────────────
   Increment 20: ENDPOINTS is no longer a hardcoded array — it's populated
   from GET /api/v1/qa/intents (src/nlq/catalog.py), the same catalog the
   natural-language layer and its structural tests already treat as the
   single source of truth. */
let ENDPOINTS = [];

async function loadIntents() {
  const statusEl = document.getElementById("qaStatus");
  const workspaceId = document.getElementById("qaWorkspaceId").value.trim();
  const apiKey = document.getElementById("qaApiKey").value.trim();
  if (!workspaceId || !apiKey) {
    statusEl.textContent = "Enter Workspace ID and API Key, then reopen this tab to load the intent list.";
    return;
  }
  statusEl.textContent = "Loading intents…";
  try {
    const resp = await fetch("/api/v1/qa/intents", {
      headers: { "X-Workspace-Id": workspaceId, "X-Api-Key": apiKey },
    });
    if (!resp.ok) { statusEl.textContent = "HTTP " + resp.status + ": " + (await resp.text()); return; }
    const data = await resp.json();
    ENDPOINTS = data.intents.map(i => ({
      id: i.intent_id, label: i.question, method: i.method, path: i.path,
      fields: i.params.map(p => ({
        name: p.name, required: p.required,
        label: p.name.replace(/_/g, " ") + (p.kind === "datetime" ? " (ISO datetime, e.g. 2026-06-01T00:00:00Z)" : ""),
      })),
    }));
    qaSelect.innerHTML = "";
    for (const ep of ENDPOINTS) {
      const opt = document.createElement("option");
      opt.value = ep.id;
      opt.textContent = ep.label;
      qaSelect.appendChild(opt);
    }
    statusEl.textContent = "";
    renderQaFields();
  } catch (e) {
    statusEl.textContent = "Failed to load intents: " + e;
  }
}

const qaSelect = document.getElementById("qaSelect");
qaSelect.addEventListener("change", renderQaFields);

function renderQaFields() {
  const ep = ENDPOINTS.find(e => e.id === qaSelect.value);
  const container = document.getElementById("qaFields");
  container.innerHTML = "";
  if (!ep) return;
  for (const f of ep.fields) {
    const label = document.createElement("label");
    label.textContent = f.label + (f.required ? " *" : "");
    const input = document.createElement("input");
    input.id = "qaField_" + f.name;
    if (f.name === "opportunity_id" && demoOpportunityId) input.value = demoOpportunityId;
    label.appendChild(input);
    container.appendChild(label);
  }
}

document.getElementById("qaRunBtn").addEventListener("click", runQa);

async function runQa() {
  const statusEl = document.getElementById("qaStatus");
  const resultEl = document.getElementById("qaResult");
  statusEl.textContent = "";
  const ep = ENDPOINTS.find(e => e.id === qaSelect.value);
  if (!ep) { statusEl.textContent = "Load the intent list first (re-open this tab with credentials filled in)."; return; }

  const workspaceId = document.getElementById("qaWorkspaceId").value.trim();
  const apiKey = document.getElementById("qaApiKey").value.trim();
  if (!workspaceId) { statusEl.textContent = "Workspace ID is required."; return; }
  if (!apiKey) { statusEl.textContent = "API Key is required."; return; }

  const values = {};
  for (const f of ep.fields) {
    const raw = document.getElementById("qaField_" + f.name).value.trim();
    if (f.required && !raw) { statusEl.textContent = f.label + " is required."; return; }
    if (raw) values[f.name] = raw;
  }

  let path = ep.path;
  const bodyFields = {};
  for (const [key, val] of Object.entries(values)) {
    if (path.includes("{" + key + "}")) {
      path = path.replace("{" + key + "}", encodeURIComponent(val));
    } else {
      bodyFields[key] = val;
    }
  }

  const options = {
    method: ep.method,
    headers: { "X-Workspace-Id": workspaceId, "X-Api-Key": apiKey },
  };
  if (ep.method !== "GET") {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(bodyFields);
  }

  resultEl.innerHTML = "<p>Loading…</p>";
  let resp;
  try {
    resp = await fetch(path, options);
  } catch (e) {
    statusEl.textContent = "Request failed: " + e;
    resultEl.innerHTML = "";
    return;
  }
  const text = await resp.text();
  if (!resp.ok) {
    statusEl.textContent = "HTTP " + resp.status + ": " + text;
    resultEl.innerHTML = "";
    return;
  }
  let data;
  try { data = JSON.parse(text); } catch { data = text; }
  resultEl.innerHTML = "<h3>" + ep.label + "</h3>";
  resultEl.appendChild(renderJson(data));
}

/* ── Ask (natural language, Increment 15/16) ─────────────────────────────── */
document.getElementById("askRunBtn").addEventListener("click", runAsk);
document.getElementById("askNarrative").addEventListener("change", (event) => {
  const narrative = document.getElementById("narrativeSummary");
  const status = document.getElementById("askStatus");
  if (!event.target.checked) {
    // A summary is optional presentation on top of the grounded answer. Hide
    // an already-rendered one immediately rather than making the user wonder
    // whether the control had any effect.
    narrative?.remove();
    status.textContent = "Narrative summary hidden. Your next Ask will not generate one.";
  } else {
    status.textContent = "A narrative summary will be generated with your next Ask.";
  }
});
document.querySelectorAll(".quick-question").forEach((button) => {
  button.addEventListener("click", () => {
    document.getElementById("askQuestion").value = button.dataset.question;
    document.getElementById("askQuestion").focus();
  });
});
window.addEventListener("keydown", (event) => {
  if (event.altKey || event.ctrlKey || event.metaKey || !["1", "2", "3", "4"].includes(event.key)) return;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
  const button = document.querySelector(`.quick-question:nth-of-type(${Number(event.key)})`);
  if (button) button.click();
});

async function runAsk() {
  const statusEl = document.getElementById("askStatus");
  const resultEl = document.getElementById("askResult");
  statusEl.textContent = "";
  resultEl.innerHTML = "";

  const workspaceId = document.getElementById("askWorkspaceId").value.trim();
  const apiKey = document.getElementById("askApiKey").value.trim();
  const question = document.getElementById("askQuestion").value.trim();
  if (!workspaceId) { statusEl.textContent = "Workspace ID is required."; return; }
  if (!apiKey) { statusEl.textContent = "API Key is required."; return; }
  if (!question) { statusEl.textContent = "Question is required."; return; }

  const body = { question, include_narrative: document.getElementById("askNarrative").checked };
  const contextFields = {
    opportunity_id: "askOpportunityId", seller_id: "askSellerId",
    conversation_id: "askConversationId", subject_id: "askSubjectId",
    buyer_contact_id: "askBuyerContactId",
  };
  for (const [key, elId] of Object.entries(contextFields)) {
    const v = document.getElementById(elId).value.trim();
    if (v) body[key] = v;
  }

  resultEl.innerHTML = "<p>Thinking…</p>";
  let resp;
  try {
    resp = await fetch("/api/v1/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Workspace-Id": workspaceId, "X-Api-Key": apiKey },
      body: JSON.stringify(body),
    });
  } catch (e) {
    statusEl.textContent = "Request failed: " + e;
    resultEl.innerHTML = "";
    return;
  }
  const text = await resp.text();
  if (!resp.ok) {
    statusEl.textContent = "HTTP " + resp.status + ": " + text;
    resultEl.innerHTML = "";
    return;
  }
  const data = JSON.parse(text);
  resultEl.innerHTML = "";

  const header = document.createElement("div");
  header.innerHTML =
    "<h3>" + (data.answered ? "Answer" : "Could not answer") + "</h3>" +
    "<div class='result-scalar'>intent: <b>" + (data.intent_id || "?") + "</b>" +
    (data.confidence != null ? " (confidence " + data.confidence.toFixed(2) + ")" : "") + "</div>" +
    (data.reasoning ? "<div class='result-scalar'>reasoning: " + data.reasoning + "</div>" : "");
  resultEl.appendChild(header);

  if (document.getElementById("askVoice")?.checked && data.answered) {
    // Speak only human-readable prose. Never send the structured result to
    // TTS: it contains opaque claim/contact/opportunity IDs that are useful in
    // the UI but distracting and unsafe to read aloud.
    const rawAudioText = data.narrative?.text || data.reasoning || "The answer is available in the text result.";
    const audioText = rawAudioText
      .replace(/\s*\[[a-f0-9]{32,}\]\s*/gi, " ")
      .replace(/\b[a-f0-9]{40,}\b/gi, " ")
      .replace(/\s{2,}/g, " ")
      .trim();
    const audioStatus = document.createElement("div");
    audioStatus.className = "result-scalar";
    audioStatus.textContent = "Preparing audio…";
    resultEl.appendChild(audioStatus);
    if (DEMO_BROWSER_TTS_ENABLED && "speechSynthesis" in window) {
      // Public demo fallback: use the browser's local voice, so the preview
      // remains useful without exposing a paid cloud-TTS credential.
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(audioText);
      utterance.onstart = () => { audioStatus.textContent = "Reading answer aloud in this browser..."; };
      utterance.onend = () => { audioStatus.textContent = "Audio playback complete."; };
      utterance.onerror = () => { audioStatus.textContent = "Browser audio could not start; text answer remains available."; };
      window.speechSynthesis.speak(utterance);
    } else {
      fetch("/api/v1/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Workspace-Id": workspaceId, "X-Api-Key": apiKey },
      body: JSON.stringify({ text: audioText }),
    }).then(async (audioResp) => {
      if (!audioResp.ok) throw new Error("TTS HTTP " + audioResp.status);
      const player = document.createElement("audio");
      player.controls = false;
      player.autoplay = true;
      player.style.display = "none";
      player.src = URL.createObjectURL(await audioResp.blob());
      const audioWrap = document.createElement("div");
      audioWrap.className = "audio-player-wrap";
      const controls = document.createElement("div");
      controls.className = "audio-controls";
      const play = document.createElement("button");
      play.type = "button";
      play.textContent = "▶";
      play.title = "Play audio";
      const progress = document.createElement("input");
      progress.className = "audio-progress";
      progress.type = "range";
      progress.min = "0";
      progress.max = "100";
      progress.value = "0";
      progress.setAttribute("aria-label", "Audio progress");
      const time = document.createElement("span");
      time.className = "audio-time";
      time.textContent = "0:00 / 0:00";
      const formatTime = (seconds) => {
        if (!Number.isFinite(seconds)) return "0:00";
        return Math.floor(seconds / 60) + ":" + String(Math.floor(seconds % 60)).padStart(2, "0");
      };
      play.addEventListener("click", () => {
        if (player.paused) player.play(); else player.pause();
      });
      player.addEventListener("play", () => { play.textContent = "⏸"; });
      player.addEventListener("pause", () => { play.textContent = "▶"; });
      player.addEventListener("loadedmetadata", () => { time.textContent = "0:00 / " + formatTime(player.duration); });
      player.addEventListener("timeupdate", () => {
        progress.value = player.duration ? String((player.currentTime / player.duration) * 100) : "0";
        time.textContent = formatTime(player.currentTime) + " / " + formatTime(player.duration);
      });
      progress.addEventListener("input", () => {
        if (player.duration) player.currentTime = (Number(progress.value) / 100) * player.duration;
      });
      controls.appendChild(play);
      controls.appendChild(progress);
      controls.appendChild(time);
      audioWrap.appendChild(controls);
      audioWrap.appendChild(player);
      const speedLabel = document.createElement("label");
      speedLabel.className = "audio-speed";
      speedLabel.textContent = "Speed ";
      const speed = document.createElement("select");
      for (const rate of [0.75, 1, 1.25, 1.5, 2]) {
        const option = document.createElement("option");
        option.value = String(rate);
        option.textContent = rate + "×";
        if (rate === 1) option.selected = true;
        speed.appendChild(option);
      }
      speed.addEventListener("change", () => { player.playbackRate = Number(speed.value); });
      speedLabel.appendChild(speed);
      audioWrap.appendChild(speedLabel);
      audioStatus.replaceWith(audioWrap);
      // Browsers may still require an explicit gesture after the async fetch;
      // the controls remain visible and the status explains that fallback.
      player.play().catch(() => {
        const hint = document.createElement("span");
        hint.className = "result-scalar";
        hint.textContent = "Audio ready — press Play if autoplay is blocked by the browser.";
        player.insertAdjacentElement("afterend", hint);
      });
      }).catch((error) => { audioStatus.textContent = "Audio unavailable; text answer remains available. " + error; });
    }
  }

  for (const a of data.ambiguities || []) {
    const div = document.createElement("div");
    div.className = "ambiguity";
    div.textContent = (a.param ? "[" + a.param + "] " : "") + a.reason;
    resultEl.appendChild(div);
  }

  if (data.narrative) {
    const narrativeWrap = document.createElement("section");
    narrativeWrap.id = "narrativeSummary";
    const narrDiv = document.createElement("div");
    narrDiv.className = "result-key";
    narrDiv.textContent = "Narrative";
    narrativeWrap.appendChild(narrDiv);
    const p = document.createElement("p");
    p.textContent = data.narrative.text;
    narrativeWrap.appendChild(p);
    for (const c of data.narrative.citations || []) {
      const cDiv = document.createElement("div");
      cDiv.className = "citation";
      cDiv.textContent = "[" + c.claim_id + "] " + c.excerpt;
      narrativeWrap.appendChild(cDiv);
    }
    for (const u of data.narrative.uncited_sentences || []) {
      const uDiv = document.createElement("div");
      uDiv.className = "uncited";
      uDiv.textContent = "(uncited) " + u;
      narrativeWrap.appendChild(uDiv);
    }
    resultEl.appendChild(narrativeWrap);
  }

  if (data.result) {
    const resDiv = document.createElement("div");
    resDiv.className = "result-key";
    resDiv.textContent = "Result";
    resultEl.appendChild(resDiv);
    resultEl.appendChild(renderJson(data.result));
  }
}

/* ── Alerts (proactive digest, Increment 17) ─────────────────────────────── */
document.getElementById("alertsRunBtn").addEventListener("click", runAlerts);

async function runAlerts() {
  const statusEl = document.getElementById("alertsStatus");
  const resultEl = document.getElementById("alertsResult");
  statusEl.textContent = "";

  const workspaceId = document.getElementById("alertsWorkspaceId").value.trim();
  const apiKey = document.getElementById("alertsApiKey").value.trim();
  const sellerId = document.getElementById("alertsSellerId").value.trim();
  if (!workspaceId) { statusEl.textContent = "Workspace ID is required."; return; }
  if (!apiKey) { statusEl.textContent = "API Key is required."; return; }

  let path = "/api/v1/digest";
  if (sellerId) path += "?seller_id=" + encodeURIComponent(sellerId);

  resultEl.innerHTML = "<p>Loading…</p>";
  let resp;
  try {
    resp = await fetch(path, { headers: { "X-Workspace-Id": workspaceId, "X-Api-Key": apiKey } });
  } catch (e) {
    statusEl.textContent = "Request failed: " + e;
    resultEl.innerHTML = "";
    return;
  }
  const text = await resp.text();
  if (!resp.ok) {
    statusEl.textContent = "HTTP " + resp.status + ": " + text;
    resultEl.innerHTML = "";
    return;
  }
  const data = JSON.parse(text);
  resultEl.innerHTML =
    "<h3>Digest</h3><div class='result-scalar'>" +
    data.opportunity_count + " open opportunities, " + data.signals.length + " signal(s)</div>";
  resultEl.appendChild(renderJson(data.signals));
}

/* ── Review Console: human mention review, conflicts, cross-deal objections ── */
function reviewCredentials() {
  return {
    workspaceId: document.getElementById("reviewWorkspaceId").value.trim(),
    apiKey: document.getElementById("reviewApiKey").value.trim(),
  };
}

function reviewHeaders(credentials, json = false) {
  const headers = { "X-Workspace-Id": credentials.workspaceId, "X-Api-Key": credentials.apiKey };
  if (json) headers["Content-Type"] = "application/json";
  return headers;
}

function requireReviewCredentials() {
  const credentials = reviewCredentials();
  const statusEl = document.getElementById("reviewStatus");
  statusEl.textContent = "";
  if (!credentials.workspaceId || !credentials.apiKey) {
    statusEl.textContent = "Workspace ID and API Key are required.";
    return null;
  }
  return credentials;
}

document.getElementById("reviewMentionsBtn").addEventListener("click", loadPendingMentions);
document.getElementById("reviewAssignmentsBtn").addEventListener("click", loadReviewAssignments);
document.getElementById("reviewConflictsBtn").addEventListener("click", loadOpenConflicts);
document.getElementById("reviewObjectionsBtn").addEventListener("click", loadTopObjections);

async function loadPendingMentions() {
  const credentials = requireReviewCredentials();
  if (!credentials) return;
  const resultEl = document.getElementById("reviewResult");
  const statusEl = document.getElementById("reviewStatus");
  resultEl.innerHTML = "<h3>Pending mentions</h3><p>Loading…</p>";
  try {
    const resp = await fetch("/api/v1/unresolved-mentions", { headers: reviewHeaders(credentials) });
    const data = await resp.json();
    if (!resp.ok) { statusEl.textContent = "HTTP " + resp.status + ": " + JSON.stringify(data); resultEl.innerHTML = ""; return; }
    resultEl.innerHTML = "<h3>Pending mentions (" + data.mentions.length + ")</h3>";
    if (!data.mentions.length) { resultEl.appendChild(renderJson(data)); return; }
    for (const mention of data.mentions) resultEl.appendChild(await mentionReviewCard(mention, credentials));
  } catch (error) {
    statusEl.textContent = "Request failed: " + error;
    resultEl.innerHTML = "";
  }
}

async function mentionReviewCard(mention, credentials) {
  const card = document.createElement("section");
  card.className = "ambiguity";
  const title = document.createElement("h4");
  title.textContent = mention.surface_text + " (" + mention.entity_type + ")";
  card.appendChild(title);
  const detail = document.createElement("div");
  detail.className = "result-scalar";
  detail.textContent = "mention: " + mention.mention_id;
  card.appendChild(detail);

  const candidates = document.createElement("select");
  candidates.setAttribute("aria-label", "Candidate entity for " + mention.surface_text);
  candidates.disabled = true;
  const placeholder = document.createElement("option");
  placeholder.value = ""; placeholder.textContent = "Loading candidates…";
  candidates.appendChild(placeholder);
  card.appendChild(candidates);

  const reason = document.createElement("textarea");
  reason.rows = 2; reason.placeholder = "Decision reason (recommended)";
  card.appendChild(reason);
  const accept = document.createElement("button");
  accept.type = "button"; accept.textContent = "Accept selected candidate";
  const reject = document.createElement("button");
  reject.type = "button"; reject.textContent = "Reject all candidates";
  const assign = document.createElement("button");
  assign.type = "button"; assign.textContent = "Assign to reviewer";
  const history = document.createElement("button");
  history.type = "button"; history.textContent = "View history";
  card.appendChild(accept); card.appendChild(reject); card.appendChild(assign); card.appendChild(history);
  const decisionStatus = document.createElement("div");
  decisionStatus.className = "result-scalar";
  card.appendChild(decisionStatus);

  let scoreById = {};
  let candidateIds = [];
  try {
    const resp = await fetch("/api/v1/unresolved-mentions/" + encodeURIComponent(mention.mention_id) + "/candidates", { headers: reviewHeaders(credentials) });
    const data = await resp.json();
    if (!resp.ok) throw new Error("HTTP " + resp.status + ": " + JSON.stringify(data));
    candidates.innerHTML = "";
    const choose = document.createElement("option");
    choose.value = ""; choose.textContent = "Choose a candidate"; candidates.appendChild(choose);
    for (const candidate of data.candidates) {
      const option = document.createElement("option");
      option.value = candidate.entity_id;
      option.textContent = candidate.name + " — score " + candidate.final_score.toFixed(3);
      candidates.appendChild(option);
      candidateIds.push(candidate.entity_id);
      scoreById[candidate.entity_id] = candidate;
    }
    candidates.disabled = false;
  } catch (error) {
    candidates.innerHTML = "";
    const failure = document.createElement("option");
    failure.textContent = "Candidates unavailable"; candidates.appendChild(failure);
    decisionStatus.textContent = String(error);
  }

  async function submit(rejected) {
    const reviewerId = document.getElementById("reviewerId").value.trim();
    if (!reviewerId) { decisionStatus.textContent = "Reviewer ID is required."; return; }
    const selectedEntityId = candidates.value;
    if (!rejected && !selectedEntityId) { decisionStatus.textContent = "Select a candidate or reject all."; return; }
    decisionStatus.textContent = "Saving…";
    try {
      const resp = await fetch("/api/v1/unresolved-mentions/" + encodeURIComponent(mention.mention_id) + "/resolve", {
        method: "POST", headers: reviewHeaders(credentials, true),
        body: JSON.stringify({
          reviewer_id: reviewerId,
          selected_entity_id: rejected ? null : selectedEntityId,
          rejected,
          candidates_shown: candidateIds,
          original_scores: scoreById,
          reason: reason.value.trim() || null,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error("HTTP " + resp.status + ": " + JSON.stringify(data));
      decisionStatus.textContent = "Saved. Affected claims: " + (data.affected_claim_ids || []).length;
      accept.disabled = true; reject.disabled = true; candidates.disabled = true;
    } catch (error) {
      decisionStatus.textContent = String(error);
    }
  }
  accept.addEventListener("click", () => submit(false));
  reject.addEventListener("click", () => submit(true));
  assign.addEventListener("click", async () => {
    const reviewerId = document.getElementById("reviewerId").value.trim();
    const slaHours = Number(document.getElementById("reviewSlaHours").value || 24);
    if (!reviewerId) { decisionStatus.textContent = "Reviewer ID is required."; return; }
    try {
      const response = await fetch("/api/v1/unresolved-mentions/" + encodeURIComponent(mention.mention_id) + "/assignment", {
        method: "POST", headers: reviewHeaders(credentials, true),
        body: JSON.stringify({ reviewer_id: reviewerId, sla_hours: slaHours }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || ("HTTP " + response.status));
      decisionStatus.textContent = "Assigned until " + data.due_at;
      assign.disabled = true;
    } catch (error) { decisionStatus.textContent = String(error); }
  });
  history.addEventListener("click", async () => {
    try {
      const response = await fetch("/api/v1/unresolved-mentions/" + encodeURIComponent(mention.mention_id) + "/history", { headers: reviewHeaders(credentials) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || ("HTTP " + response.status));
      const historyEl = document.createElement("div");
      historyEl.appendChild(renderJson(data));
      card.appendChild(historyEl);
      history.disabled = true;
    } catch (error) { decisionStatus.textContent = String(error); }
  });
  return card;
}

async function loadReviewAssignments() {
  const credentials = requireReviewCredentials();
  if (!credentials) return;
  const reviewerId = document.getElementById("reviewerId").value.trim();
  if (!reviewerId) { document.getElementById("reviewStatus").textContent = "Reviewer ID is required."; return; }
  const resultEl = document.getElementById("reviewResult");
  try {
    const response = await fetch("/api/v1/unresolved-mentions/assignments?reviewer_id=" + encodeURIComponent(reviewerId), { headers: reviewHeaders(credentials) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || ("HTTP " + response.status));
    resultEl.replaceChildren(Object.assign(document.createElement("h3"), {textContent: "Review assignments"}), renderJson(data));
  } catch (error) { document.getElementById("reviewStatus").textContent = String(error); }
}

async function loadOpenConflicts() {
  const credentials = requireReviewCredentials();
  if (!credentials) return;
  const opportunityId = document.getElementById("reviewOpportunityId").value.trim();
  const resultEl = document.getElementById("reviewResult");
  const statusEl = document.getElementById("reviewStatus");
  if (!opportunityId) { statusEl.textContent = "Opportunity ID is required for conflict review."; return; }
  resultEl.innerHTML = "<h3>Open conflicts</h3><p>Loading…</p>";
  try {
    const resp = await fetch("/api/v1/opportunities/" + encodeURIComponent(opportunityId) + "/conflicts", { headers: reviewHeaders(credentials) });
    const data = await resp.json();
    if (!resp.ok) { statusEl.textContent = "HTTP " + resp.status + ": " + JSON.stringify(data); resultEl.innerHTML = ""; return; }
    resultEl.innerHTML = "<h3>Open conflicts (" + data.conflicts.length + ")</h3>";
    for (const conflict of data.conflicts) resultEl.appendChild(conflictReviewCard(conflict, opportunityId, credentials));
  } catch (error) {
    statusEl.textContent = "Request failed: " + error;
    resultEl.innerHTML = "";
  }
}

function conflictReviewCard(conflict, opportunityId, credentials) {
  const card = document.createElement("section");
  card.className = "ambiguity";
  card.appendChild(renderJson(conflict));
  const winner = document.createElement("select");
  for (const id of ["", conflict.claim_id_a, conflict.claim_id_b]) {
    const option = document.createElement("option");
    option.value = id; option.textContent = id ? "Choose " + id : "Use automatic arbitration";
    winner.appendChild(option);
  }
  const resolve = document.createElement("button");
  resolve.type = "button"; resolve.textContent = "Resolve conflict";
  const result = document.createElement("div"); result.className = "result-scalar";
  resolve.addEventListener("click", async () => {
    result.textContent = "Resolving…";
    try {
      const resp = await fetch("/api/v1/opportunities/" + encodeURIComponent(opportunityId) + "/conflicts/" + encodeURIComponent(conflict.conflict_id) + "/resolve", {
        method: "POST", headers: reviewHeaders(credentials, true),
        body: JSON.stringify({ winner_claim_id: winner.value || null }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error("HTTP " + resp.status + ": " + JSON.stringify(data));
      result.textContent = data.resolved ? "Resolved: " + data.reason : "Left open: " + data.reason;
      if (data.resolved) { resolve.disabled = true; winner.disabled = true; }
    } catch (error) { result.textContent = String(error); }
  });
  card.appendChild(winner); card.appendChild(resolve); card.appendChild(result);
  return card;
}

async function loadTopObjections() {
  const credentials = requireReviewCredentials();
  if (!credentials) return;
  const sellerId = document.getElementById("reviewSellerId").value.trim();
  const resultEl = document.getElementById("reviewResult");
  const statusEl = document.getElementById("reviewStatus");
  if (!sellerId) { statusEl.textContent = "Seller ID is required for cross-deal aggregation."; return; }
  resultEl.innerHTML = "<h3>Top objections</h3><p>Loading…</p>";
  try {
    const resp = await fetch("/api/v1/sellers/" + encodeURIComponent(sellerId) + "/top-objections", { headers: reviewHeaders(credentials) });
    const data = await resp.json();
    if (!resp.ok) { statusEl.textContent = "HTTP " + resp.status + ": " + JSON.stringify(data); resultEl.innerHTML = ""; return; }
    resultEl.innerHTML = "<h3>Top objections across open deals</h3>";
    resultEl.appendChild(renderJson(data));
  } catch (error) {
    statusEl.textContent = "Request failed: " + error;
    resultEl.innerHTML = "";
  }
}

/* â”€â”€ Product workflows: responsive seller console + reconnect-safe drafts â”€â”€ */
function workflowCredentials() {
  return { workspaceId: document.getElementById('workflowWorkspaceId').value.trim(), apiKey: document.getElementById('workflowApiKey').value.trim() };
}
function workflowHeaders(json = false) {
  const credentials = workflowCredentials();
  const headers = { 'X-Workspace-Id': credentials.workspaceId, 'X-Api-Key': credentials.apiKey };
  if (json) headers['Content-Type'] = 'application/json';
  return headers;
}
function workflowDraft() {
  return { opportunityId: document.getElementById('workflowOpportunityId').value.trim(), sellerId: document.getElementById('workflowSellerId').value.trim(), title: document.getElementById('workflowSpaceTitle').value.trim() };
}
function saveWorkflowDraft() { localStorage.setItem('scg-workflow-draft', JSON.stringify(workflowDraft())); }
function restoreWorkflowDraft() {
  try { const draft = JSON.parse(localStorage.getItem('scg-workflow-draft') || '{}');
    if (draft.opportunityId) document.getElementById('workflowOpportunityId').value = draft.opportunityId;
    if (draft.sellerId) document.getElementById('workflowSellerId').value = draft.sellerId;
    if (draft.title) document.getElementById('workflowSpaceTitle').value = draft.title;
  } catch (_) { /* invalid local draft is safely ignored */ }
}
for (const id of ['workflowOpportunityId', 'workflowSellerId', 'workflowSpaceTitle']) document.getElementById(id).addEventListener('input', saveWorkflowDraft);
restoreWorkflowDraft();

async function workflowFetch(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { ...workflowHeaders(Boolean(options.body)), ...(options.headers || {}) } });
  const text = await response.text();
  if (!response.ok) throw new Error('HTTP ' + response.status + ': ' + text);
  return text ? JSON.parse(text) : null;
}
function workflowCard(title, data) {
  const card = document.createElement('section'); card.className = 'product-card';
  const heading = document.createElement('h3'); heading.textContent = title; card.appendChild(heading);
  card.appendChild(renderJson(data)); return card;
}
async function loadWorkflowDashboard() {
  const status = document.getElementById('workflowStatus'); const result = document.getElementById('workflowResult');
  const credentials = workflowCredentials(); const draft = workflowDraft();
  saveWorkflowDraft(); status.textContent = ''; result.innerHTML = '';
  if (!credentials.workspaceId || !credentials.apiKey) { status.textContent = 'Workspace ID and API Key are required.'; return; }
  if (!navigator.onLine) { status.textContent = 'Offline: draft saved locally. Reconnect to refresh server data.'; return; }
  const requests = [workflowFetch('/api/v1/revenue/summary').then(data => ['Revenue intelligence', data])];
  if (draft.sellerId) requests.push(workflowFetch('/api/v1/readiness/sellers/' + encodeURIComponent(draft.sellerId)).then(data => ['Sales readiness', data]));
  if (draft.opportunityId) {
    requests.push(workflowFetch('/api/v1/opportunities/' + encodeURIComponent(draft.opportunityId) + '/buyer-spaces').then(data => ['Buyer Spaces', data]));
    requests.push(workflowFetch('/api/v1/opportunities/' + encodeURIComponent(draft.opportunityId) + '/meeting-brief').then(data => ['Meeting brief', data]));
  }
  try { for (const [title, data] of await Promise.all(requests)) result.appendChild(workflowCard(title, data)); }
  catch (error) { status.textContent = String(error); }
}
async function createWorkflowSpace() {
  const status = document.getElementById('workflowStatus'); const draft = workflowDraft(); saveWorkflowDraft();
  if (!draft.opportunityId || !draft.title) { status.textContent = 'Opportunity ID and Buyer Space title are required.'; return; }
  if (!navigator.onLine) { status.textContent = 'Offline: draft saved locally; reconnect before creating the Buyer Space.'; return; }
  try {
    const data = await workflowFetch('/api/v1/opportunities/' + encodeURIComponent(draft.opportunityId) + '/buyer-spaces', { method: 'POST', body: JSON.stringify({ title: draft.title }) });
    status.textContent = 'Buyer Space created: ' + data.space_id; document.getElementById('workflowSpaceTitle').value = ''; saveWorkflowDraft(); await loadWorkflowDashboard();
  } catch (error) { status.textContent = String(error); }
}
document.getElementById('workflowLoadBtn').addEventListener('click', loadWorkflowDashboard);
document.getElementById('workflowCreateSpaceBtn').addEventListener('click', createWorkflowSpace);

""" + _RENDER_JSON_JS + """
</script>
</body>
</html>
"""

_PANEL_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Opportunity Panel</title>
<style>
""" + _SHARED_STYLES + """
  body { margin: 0; padding: 12px; font-size: 13px; }
  h4 { margin: 14px 0 4px 0; }
  #status { color: var(--color-danger-text); font-size: 12px; }
</style>
</head>
<body>
<div id="status"></div>
<div id="content"></div>
<script>
// Injected server-side by GET /viz/panel (api/routes/viz.py) from the
// *validated* panel token -- never re-parsed from client-editable query
// params. See src/viz/panel_tokens.py.
const workspaceId = __WORKSPACE_ID_JSON__;
const opportunityId = __OPPORTUNITY_ID_JSON__;
const panelToken = __PANEL_TOKEN_JSON__;
const statusEl = document.getElementById("status");
const contentEl = document.getElementById("content");

""" + _RENDER_JSON_JS + """

async function loadPanel() {
  // Only the scoped panel token -- the real workspace API key never
  // reaches this page. api/dependencies.py::verify_api_key_or_panel_token
  // accepts this header on the two opportunity-scoped endpoints below as an alternative to a
  // real API key.
  const headers = { "X-Panel-Token": panelToken };

  await section("Buying committee", async () => {
    const r = await fetch("/api/v1/opportunities/" + encodeURIComponent(opportunityId) + "/buying-committee", { headers });
    return r.json();
  });

  await section("Open objections", async () => {
    const r = await fetch("/api/v1/qa/account-objections", {
      method: "POST",
      headers: { ...headers, "Content-Type": "application/json" },
      body: JSON.stringify({ opportunity_id: opportunityId }),
    });
    return r.json();
  });

}

async function section(title, fetchFn) {
  const h = document.createElement("h4");
  h.textContent = title;
  contentEl.appendChild(h);
  try {
    const data = await fetchFn();
    contentEl.appendChild(renderJson(data));
  } catch (e) {
    const err = document.createElement("div");
    err.className = "result-scalar";
    err.textContent = "Failed to load: " + e;
    contentEl.appendChild(err);
  }
}

loadPanel();
</script>
</body>
</html>
"""


_BUYER_PORTAL_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Buyer Space</title>
<style>
  body{max-width:860px;margin:0 auto;padding:24px;font-family:system-ui,sans-serif;background:#f0ece8;color:#15254e}
  main{background:#fff;padding:24px;border-radius:10px;box-shadow:0 2px 12px #15254e22} label{display:block;margin:12px 0 4px}
  input,textarea,button{font:inherit;padding:8px;max-width:100%;box-sizing:border-box} input,textarea{width:100%}
  button{background:#8c3fcc;color:white;border:0;border-radius:4px;cursor:pointer;margin-top:10px} button:focus-visible,input:focus-visible,textarea:focus-visible{outline:3px solid #0d5189;outline-offset:2px}
  #status{white-space:pre-wrap;color:#b3492f}.card{border-top:1px solid #ddd0c4;margin-top:18px;padding-top:12px}.muted{color:#6b6258;font-size:.9rem}
</style>
</head>
<body><main>
<h1>Buyer Space</h1><p class="muted">Use the secure invitation link supplied by your seller. The token remains in this browser tab only.</p>
<label for="spaceId">Space ID</label><input id="spaceId" autocomplete="off" placeholder="Provided with your invitation">
<button id="open">Open Buyer Space</button><div id="status" role="status" aria-live="polite"></div><section id="content" hidden></section>
</main><script>
const status=document.getElementById('status'), content=document.getElementById('content'), space=document.getElementById('spaceId');
const hash=new URLSearchParams(location.hash.slice(1)); if(hash.get('token')) sessionStorage.setItem('scgBuyerToken',hash.get('token'));
const headers=()=>({'X-Buyer-Token':sessionStorage.getItem('scgBuyerToken')||''});
function item(label,value){const p=document.createElement('p');const b=document.createElement('b');b.textContent=label+': ';p.append(b,document.createTextNode(value||'—'));return p}
async function openSpace(){status.textContent='';content.hidden=true;const id=space.value.trim();if(!id){status.textContent='Space ID is required.';return}try{await fetch('/api/v1/buyer-portal/accept',{method:'POST',headers});const r=await fetch('/api/v1/buyer-portal/'+encodeURIComponent(id),{headers});const data=await r.json();if(!r.ok) throw new Error(data.detail||'Unable to open Buyer Space');content.replaceChildren();content.append(Object.assign(document.createElement('h2'),{textContent:data.space.title}));content.append(item('Status',data.space.status));for(const [title,entries,field] of [['Next steps',data.next_steps,'title'],['Comments',data.comments,'body'],['Uploads',data.uploads,'filename']]){const section=document.createElement('div');section.className='card';section.append(Object.assign(document.createElement('h3'),{textContent:title}));if(!entries.length)section.append(Object.assign(document.createElement('p'),{textContent:'Nothing yet.'}));for(const entry of entries)section.append(item('',entry[field]));content.append(section)}content.hidden=false}catch(error){status.textContent=error.message||String(error)}}
document.getElementById('open').addEventListener('click',openSpace);
</script></body></html>"""
