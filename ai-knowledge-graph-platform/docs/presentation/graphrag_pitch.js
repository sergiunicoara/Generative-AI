const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "GraphRAG for Enterprise Intelligence";
pres.author = "Sergiu";

// ── Palette ────────────────────────────────────────────────────────────────
const NAV   = "0F1F47";
const TEAL  = "0096B4";
const TEAL2 = "00B4D8";
const WHITE = "FFFFFF";
const LBKG  = "F4F7FB";
const TXT   = "1A2438";
const TXT2  = "4A5568";
const CBRD  = "D1DCF0";

const makeShadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.10 });

function darkBg(slide) { slide.background = { color: NAV }; }
function lightBg(slide) { slide.background = { color: LBKG }; }
function sectionLabel(slide, text, y = 0.3) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 0.05, h: 0.32, fill: { color: TEAL }, line: { color: TEAL } });
  slide.addText(text.toUpperCase(), { x: 0.65, y, w: 9, h: 0.32, fontSize: 9, bold: true, color: TEAL, charSpacing: 3, margin: 0, fontFace: "Calibri" });
}
function slideTitle(slide, text, y = 0.72) {
  slide.addText(text, { x: 0.5, y, w: 9, h: 0.75, fontSize: 30, bold: true, color: TXT, fontFace: "Calibri", margin: 0 });
}
function card(slide, x, y, w, h, color = "FFFFFF") {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color }, line: { color: CBRD, width: 0.5 }, shadow: makeShadow() });
}

// ══════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title  (fixes: no accent underline; balanced right-side geometry)
// ══════════════════════════════════════════════════════════════════════════
const s1 = pres.addSlide();
darkBg(s1);

// Left accent bar
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.06, h: 5.625, fill: { color: TEAL }, line: { color: TEAL } });

// Right-side decorative geometry: overlapping circles suggest a "graph / network" motif
const circleData = [
  { x: 7.4, y: 0.3, w: 2.6, h: 2.6, t: 88 },
  { x: 8.0, y: 1.2, w: 2.0, h: 2.0, t: 82 },
  { x: 6.8, y: 1.8, w: 3.0, h: 3.0, t: 92 },
  { x: 7.6, y: 2.5, w: 1.5, h: 1.5, t: 78 },
];
circleData.forEach(c => {
  s1.addShape(pres.shapes.OVAL, { x: c.x, y: c.y, w: c.w, h: c.h, fill: { color: TEAL, transparency: c.t }, line: { color: TEAL2, transparency: c.t - 5, width: 0.5 } });
});
// Node dots
[[8.9, 1.0], [7.8, 2.4], [9.1, 3.0], [7.2, 3.5]].forEach(([x, y]) => {
  s1.addShape(pres.shapes.OVAL, { x, y, w: 0.14, h: 0.14, fill: { color: TEAL2, transparency: 30 }, line: { color: TEAL2, transparency: 30 } });
});

// Top label
s1.addText("ENTERPRISE AI CONSULTING  ·  GRAPH RAG PLATFORM", { x: 0.4, y: 0.35, w: 6.5, h: 0.3, fontSize: 8.5, bold: true, color: TEAL2, charSpacing: 3, fontFace: "Calibri" });

// Main title (no underline/divider — removed)
s1.addText("GraphRAG for", { x: 0.4, y: 1.0, w: 6.8, h: 0.9, fontSize: 48, bold: true, color: WHITE, fontFace: "Calibri", margin: 0 });
s1.addText("Enterprise Intelligence", { x: 0.4, y: 1.88, w: 6.8, h: 0.9, fontSize: 40, bold: true, color: TEAL2, fontFace: "Calibri", margin: 0 });

// Subtitle — brighter so it shows on a projector
s1.addText("Agentic Knowledge Graph Platform  ·  Built for Client Delivery", {
  x: 0.4, y: 2.98, w: 6.4, h: 0.45, fontSize: 14.5, color: "B8D4F0", fontFace: "Calibri", italic: true
});

// Name + GitHub — brighter for projector visibility
s1.addText("Sergiu  |  Graph RAG Engineer", { x: 0.4, y: 4.88, w: 4.5, h: 0.38, fontSize: 11.5, color: "9BBFD8", fontFace: "Calibri" });
s1.addText("github.com/sergiunicoara/Generative-AI", { x: 5.1, y: 4.88, w: 4.5, h: 0.38, fontSize: 11.5, color: TEAL2, fontFace: "Calibri", align: "right" });


// ══════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Problem  (fix: tighten gap between heading and body inside cards)
// ══════════════════════════════════════════════════════════════════════════
const s2 = pres.addSlide();
lightBg(s2);
sectionLabel(s2, "The Problem");
slideTitle(s2, "Standard AI Fails Enterprise Knowledge");

const problems = [
  { title: "Hallucination at Scale", body: "LLMs invent facts when knowledge is complex, contradictory or spread across hundreds of documents" },
  { title: "No Reasoning Chain",    body: "Can't trace logic across connected entities — a contract links to a regulation links to a company" },
  { title: "No Audit Trail",        body: "Clients in regulated industries need to know exactly where an answer came from and who authorised it" },
];

problems.forEach((p, i) => {
  const x = 0.5 + i * 3.05;
  card(s2, x, 1.65, 2.85, 2.5);
  // Red top accent
  s2.addShape(pres.shapes.RECTANGLE, { x, y: 1.65, w: 2.85, h: 0.07, fill: { color: "DC2626" }, line: { color: "DC2626" } });
  // Heading — tighter to top
  s2.addText(p.title, { x: x + 0.18, y: 1.82, w: 2.5, h: 0.5, fontSize: 13, bold: true, color: TXT, fontFace: "Calibri", margin: 0 });
  // Body — immediately below heading (was y: 2.38, now y: 2.2 — reduced gap by 0.18")
  s2.addText(p.body, { x: x + 0.18, y: 2.2, w: 2.5, h: 1.72, fontSize: 11.5, color: TXT2, fontFace: "Calibri", margin: 0 });
});

// Bottom callout — padding fix (moved text x from 0.7 to 0.85 for left padding)
s2.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.35, w: 9.0, h: 0.72, fill: { color: NAV }, line: { color: NAV }, shadow: makeShadow() });
s2.addText("PwC clients in banking, regulatory, and audit need explainable, traceable, contradiction-free AI answers — not black boxes.", {
  x: 0.85, y: 4.4, w: 8.2, h: 0.62, fontSize: 12.5, color: WHITE, fontFace: "Calibri", italic: true, valign: "middle"
});


// ══════════════════════════════════════════════════════════════════════════
// SLIDE 3 — Architecture  (fix: consistent top stripe color; better bullet indent; brighter subtitles)
// ══════════════════════════════════════════════════════════════════════════
const s3 = pres.addSlide();
lightBg(s3);
sectionLabel(s3, "The Platform");
slideTitle(s3, "Three Layers. One Coherent System.");

const layers = [
  { title: "Knowledge Graph", sub: "Neo4j 5.x", bullets: ["Entity resolution (4-stage)", "Ontology enforcement", "Forward-chaining inference", "Contradiction detection", "Bitemporal modeling"], bg: NAV },
  { title: "Retrieval Pipeline", sub: "6-stage hybrid", bullets: ["Vector ANN (3072d OpenAI)", "BM25 + RRF fusion", "Cross-encoder reranking", "Multi-hop graph traversal", "GAT/GCN GNN scoring"], bg: "065A82" },
  { title: "Agent Layer", sub: "Groq + DeepSeek fallback", bullets: ["Two-model IRCoT (8B + 70B)", "AND-logic agentic trigger", "Session context (Redis)", "RAGAS evaluation", "JWT-secured REST API"], bg: "164E63" },
];

layers.forEach((l, i) => {
  const x = 0.5 + i * 3.05;
  s3.addShape(pres.shapes.RECTANGLE, { x, y: 1.55, w: 2.85, h: 3.65, fill: { color: l.bg }, line: { color: l.bg }, shadow: makeShadow() });
  // Consistent TEAL top stripe on all cards
  s3.addShape(pres.shapes.RECTANGLE, { x, y: 1.55, w: 2.85, h: 0.07, fill: { color: TEAL }, line: { color: TEAL } });
  s3.addText(l.title, { x: x + 0.18, y: 1.72, w: 2.5, h: 0.45, fontSize: 13.5, bold: true, color: WHITE, fontFace: "Calibri", margin: 0 });
  // Subtitle brighter: was TEAL2 (fine) but use WHITE with italic for better projector contrast
  s3.addText(l.sub, { x: x + 0.18, y: 2.14, w: 2.5, h: 0.28, fontSize: 10.5, color: TEAL2, fontFace: "Calibri", italic: true, bold: true, margin: 0 });

  // Bullets — reduced indent (x: x+0.18 instead of x+0.12, but margin:0 so the bullet aligns at x+0.18)
  const bulletItems = l.bullets.map((b, bi) => ({
    text: b,
    options: { bullet: true, color: "C8DCF0", fontSize: 10, fontFace: "Calibri", breakLine: bi < l.bullets.length - 1, paraSpaceAfter: 4 }
  }));
  s3.addText(bulletItems, { x: x + 0.18, y: 2.5, w: 2.6, h: 2.55, margin: 0 });

  // Larger, more visible arrows
  if (i < 2) {
    s3.addShape(pres.shapes.RECTANGLE, { x: x + 2.85, y: 3.28, w: 0.22, h: 0.05, fill: { color: TEAL2 }, line: { color: TEAL2 } });
    s3.addText("›", { x: x + 2.97, y: 3.13, w: 0.2, h: 0.32, fontSize: 14, bold: true, color: TEAL2, margin: 0, align: "center" });
  }
});


// ══════════════════════════════════════════════════════════════════════════
// SLIDE 4 — Capabilities (fix: row 1 cell width adjusted to reduce wrapping)
// ══════════════════════════════════════════════════════════════════════════
const s4 = pres.addSlide();
lightBg(s4);
sectionLabel(s4, "Capabilities");
slideTitle(s4, "Every JD Requirement — Demonstrated in Code");

const hdrY = 1.58;
s4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: hdrY, w: 9.0, h: 0.38, fill: { color: NAV }, line: { color: NAV } });
s4.addText("Role Requirement", { x: 0.65, y: hdrY + 0.05, w: 3.5, h: 0.28, fontSize: 10.5, bold: true, color: TEAL2, fontFace: "Calibri", margin: 0 });
s4.addText("What's Built & Where", { x: 4.25, y: hdrY + 0.05, w: 5.1, h: 0.28, fontSize: 10.5, bold: true, color: TEAL2, fontFace: "Calibri", margin: 0 });

const rows = [
  ["Neo4j + graph-based data modeling", "39 KG modules · 572-line neo4j_client.py · production Cypher (UNWIND, COUNT{}, vector ANN, BM25)"],
  ["Graph RAG patterns", "6-stage pipeline: vector → BM25+RRF → reranker → multi-hop traversal → GNN → LLM synthesis"],
  ["LLM integration + tool invocation", "Groq llama-3.3-70b (synthesis) · llama-3.1-8b (routing) · DeepSeek-V3 fallback · agentic IRCoT"],
  ["Design + validate MVP solutions", "364 passing tests · CI/CD (GitHub Actions) · Docker multi-stage · runbook · regulatory demo"],
  ["Agentic chatbot, autonomous tools", "QueryAgent + IngestionAgent + EvaluationAgent · session context · Redis result store"],
  ["Secure integration", "OAuth 2.0 · JWT scopes · rate limiting · per-tenant isolation · GDPR erasure"],
];

rows.forEach((row, i) => {
  const y = hdrY + 0.38 + i * 0.49;
  const bg = i % 2 === 0 ? "FFFFFF" : "EEF3FB";
  s4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 9.0, h: 0.49, fill: { color: bg }, line: { color: CBRD, width: 0.3 } });
  s4.addShape(pres.shapes.OVAL, { x: 0.6, y: y + 0.14, w: 0.21, h: 0.21, fill: { color: "16A34A" }, line: { color: "16A34A" } });
  s4.addText("✓", { x: 0.57, y: y + 0.10, w: 0.27, h: 0.28, fontSize: 9, color: WHITE, bold: true, fontFace: "Calibri", align: "center", margin: 0 });
  s4.addText(row[0], { x: 0.9, y: y + 0.08, w: 3.15, h: 0.33, fontSize: 10.5, bold: true, color: TXT, fontFace: "Calibri", margin: 0 });
  // Wider value column: 5.2 → 5.2 wide, with smaller font to prevent wrapping
  s4.addText(row[1], { x: 4.25, y: y + 0.06, w: 5.1, h: 0.4, fontSize: 9.5, color: TXT2, fontFace: "Calibri", margin: 0 });
});


// ══════════════════════════════════════════════════════════════════════════
// SLIDE 5 — Demo  (fixes: code block moved up; list not cramped; consistent color)
// ══════════════════════════════════════════════════════════════════════════
const s5 = pres.addSlide();
darkBg(s5);

// Right-side partial circle decorative (more visible: using two rings)
s5.addShape(pres.shapes.OVAL, { x: 6.5, y: -1.0, w: 5.5, h: 5.5, fill: { color: TEAL, transparency: 90 }, line: { color: TEAL2, transparency: 75, width: 1.0 } });
s5.addShape(pres.shapes.OVAL, { x: 7.2, y: -0.2, w: 3.8, h: 3.8, fill: { color: TEAL, transparency: 85 }, line: { color: TEAL2, transparency: 65, width: 0.8 } });

s5.addText("LIVE DEMO", { x: 0.6, y: 0.68, w: 4, h: 0.38, fontSize: 10, bold: true, color: TEAL2, charSpacing: 5, fontFace: "Calibri" });
s5.addText("Regulatory\nCompliance\nIntelligence", { x: 0.6, y: 1.05, w: 7.8, h: 2.2, fontSize: 44, bold: true, color: WHITE, fontFace: "Calibri", margin: 0 });

// "You will see:" — consistent TEAL2 (same as rest of deck)
s5.addText("You will see:", { x: 0.6, y: 3.35, w: 9, h: 0.32, fontSize: 11, bold: true, color: TEAL2, fontFace: "Calibri", margin: 0 });

const demoItems = [
  "Domain ontology loading from YAML — aerospace regulatory type hierarchy",
  "Forward-chaining inference — AD-2024 transitively supersedes AD-2020",
  "Contradiction detection — same aircraft IS_AIRWORTHY and IS_UNAIRWORTHY",
  "Authority chain query — which document is the current governing authority?",
  "Live REST API — ingest a document, query with session context",
];
const demoRich = demoItems.map((d, i) => ({
  text: `${i + 1}.  ${d}`,
  options: { breakLine: i < demoItems.length - 1, color: "B0C8E0", fontSize: 10.5, fontFace: "Calibri", paraSpaceAfter: 2 }
}));
// Tighter spacing, moved up so code block has room
s5.addText(demoRich, { x: 0.6, y: 3.7, w: 9.0, h: 1.1 });

// Code block — moved up (was 5.05, now 4.85 — keeps it off the edge)
s5.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 4.85, w: 5.5, h: 0.35, fill: { color: "0A1428" }, line: { color: TEAL, width: 1.0 } });
s5.addText("  python scripts/demo_regulatory.py", { x: 0.6, y: 4.85, w: 5.5, h: 0.35, fontSize: 10.5, color: TEAL2, fontFace: "Consolas", margin: 0, valign: "middle" });


// ══════════════════════════════════════════════════════════════════════════
// SLIDE 6 — Observability  ("Measured, Not Claimed" — metrics + admin dashboard)
// ══════════════════════════════════════════════════════════════════════════
const sM = pres.addSlide();
lightBg(sM);
sectionLabel(sM, "Observability");
slideTitle(sM, "Measured, Not Claimed");

// Four headline stat tiles
const mStats = [
  { num: "16",  label: "Tracked metrics" },
  { num: "4",   label: "Measurement layers" },
  { num: "5",   label: "Live dashboard tabs" },
  { num: "30s", label: "Auto-refresh cadence" },
];
mStats.forEach((st, i) => {
  const x = 0.5 + i * 2.28;
  card(sM, x, 1.5, 2.1, 1.0, "FFFFFF");
  sM.addShape(pres.shapes.RECTANGLE, { x, y: 1.5, w: 2.1, h: 0.055, fill: { color: TEAL }, line: { color: TEAL } });
  sM.addText(st.num,   { x: x + 0.12, y: 1.6,  w: 1.86, h: 0.5, fontSize: 26, bold: true, color: NAV, fontFace: "Calibri", margin: 0 });
  sM.addText(st.label, { x: x + 0.12, y: 2.12, w: 1.86, h: 0.32, fontSize: 9.5, color: TXT2, fontFace: "Calibri", margin: 0 });
});

// Two metric cards
function metricCard(slide, x, title, sub, rows) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y: 2.72, w: 4.35, h: 2.06, fill: { color: "FFFFFF" }, line: { color: CBRD, width: 0.5 }, shadow: makeShadow() });
  slide.addShape(pres.shapes.RECTANGLE, { x, y: 2.72, w: 4.35, h: 0.055, fill: { color: TEAL }, line: { color: TEAL } });
  slide.addText(title, { x: x + 0.18, y: 2.8,  w: 4.0, h: 0.3,  fontSize: 12, bold: true, color: NAV, fontFace: "Calibri", margin: 0 });
  slide.addText(sub,   { x: x + 0.18, y: 3.08, w: 4.0, h: 0.24, fontSize: 8.5, italic: true, color: TEAL, fontFace: "Calibri", margin: 0 });
  rows.forEach((r, i) => {
    const y = 3.38 + i * 0.275;
    slide.addText(r[0], { x: x + 0.18, y, w: 2.85, h: 0.26, fontSize: 9.5, color: TXT, fontFace: "Calibri", margin: 0 });
    slide.addText(r[1], { x: x + 3.0,  y, w: 1.2,  h: 0.26, fontSize: 9.5, bold: true, color: "0096B4", fontFace: "Calibri", margin: 0, align: "right" });
  });
}

metricCard(sM, 0.5, "Answer Quality — RAGAS", "104 queries · 20% sampled · llama-3.3-70b", [
  ["Faithfulness",      "0.937  (answerable) · 0.842 overall  ✓"],
  ["Answer relevancy",  "0.816  ✓"],
  ["Context precision", "0.907  ✓"],
  ["Context recall",    "0.867  ✓"],
  ["Hybrid p95",        "2.2 s  ✓  (agentic: 3.4 s)"],
]);
metricCard(sM, 5.15, "Graph Health + Calibration", "368 entities · 422 edges · aerospace corpus", [
  ["Alias dedup rate",     "45%  (665 raw → 364 resolved)"],
  ["Contradiction density","28.95 /1k edges"],
  ["Community coherence",  "92%  ✓  (55 Leiden)"],
  ["Orphan rate",          "0%  ✓"],
  ["Brier score",          "0.809  (calibrating — 48 samples)"],
]);

// Bottom callout — the live dashboard
sM.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.92, w: 9.0, h: 0.55, fill: { color: NAV }, line: { color: NAV }, shadow: makeShadow() });
sM.addText([
  { text: "Live operator dashboard  ", options: { bold: true, color: WHITE } },
  { text: "/admin", options: { bold: true, color: TEAL2, fontFace: "Consolas" } },
  { text: "   ·   Health · Conflicts · Communities · GDPR · Calibration — branded gauges, trend lines, drill-downs", options: { color: "C8D8EC" } },
], { x: 0.75, y: 4.92, w: 8.5, h: 0.55, fontSize: 10.5, fontFace: "Calibri", valign: "middle", margin: 0 });


// ══════════════════════════════════════════════════════════════════════════
// SLIDE 7 — The Dashboard, Live  (native recreation of /admin Graph Health)
// ══════════════════════════════════════════════════════════════════════════
const sD = pres.addSlide();
lightBg(sD);
sectionLabel(sD, "The Product");
slideTitle(sD, "The Operator Dashboard — Live");

// KPI tile strip
const dTiles = [
  { num: "368",       label: "Entities",              accent: TEAL     },
  { num: "422",       label: "Edges",                 accent: TEAL2    },
  { num: "99.5%",     label: "High-conf edges",       accent: "16A34A" },
  { num: "28.9/1k",   label: "Contradiction density", accent: "E8A317" },
];
dTiles.forEach((t, i) => {
  const x = 0.5 + i * 2.28;
  card(sD, x, 1.5, 2.1, 0.95, "FFFFFF");
  sD.addShape(pres.shapes.RECTANGLE, { x, y: 1.5, w: 0.05, h: 0.95, fill: { color: t.accent }, line: { color: t.accent } });
  sD.addText(t.num,   { x: x + 0.16, y: 1.58, w: 1.9, h: 0.48, fontSize: 22, bold: true, color: NAV, fontFace: "Calibri", margin: 0 });
  sD.addText(t.label.toUpperCase(), { x: x + 0.16, y: 2.06, w: 1.9, h: 0.3, fontSize: 8, bold: true, color: TXT2, charSpacing: 1, fontFace: "Calibri", margin: 0 });
});

// Four gauges (doughnut rings) — mirror the live Graph Health tab
function gauge(slide, x, value, color, label, displayValue) {
  const display = displayValue !== undefined ? displayValue : `${value}%`;
  slide.addChart(pres.charts.DOUGHNUT,
    [{ name: label, labels: ["v", "rest"], values: [value, 100 - value] }],
    {
      x, y: 2.7, w: 2.15, h: 1.85,
      chartColors: [color, "E4EBF6"],
      holeSize: 70, showLegend: false, showTitle: false,
      showValue: false, showPercent: false, dataBorder: { pt: 0, color: "FFFFFF" },
    });
  // Center value
  slide.addText(display, { x: x + 0.18, y: 3.32, w: 1.8, h: 0.5, fontSize: 19, bold: true, color: NAV, align: "center", fontFace: "Calibri", margin: 0 });
  // Caption under ring
  slide.addText(label, { x: x - 0.05, y: 4.5, w: 2.4, h: 0.3, fontSize: 9, bold: true, color: TXT2, align: "center", fontFace: "Calibri", margin: 0 });
}
gauge(sD, 0.5,  14.7, "DC2626", "Entity Resolution");        // red — 14.7% alias coverage (live)
gauge(sD, 2.85, 100,  "16A34A", "Relation Confidence", "100%"); // full green — 100% high-conf edges
gauge(sD, 5.2,  94,   "16A34A", "Community Coherence");         // green — 94% Leiden coherence
gauge(sD, 7.55, 100,  "16A34A", "Orphan Rate", "0%");           // inverted: full=best, 0% orphans

// Caption bar
sD.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.95, w: 9.0, h: 0.5, fill: { color: NAV }, line: { color: NAV }, shadow: makeShadow() });
sD.addText([
  { text: "Shown live during the demo at  ", options: { color: "C8D8EC" } },
  { text: "/admin", options: { bold: true, color: TEAL2, fontFace: "Consolas" } },
  { text: "  ·  5 tabs · radial gauges · trend lines · one-click conflict resolution & GDPR erasure", options: { color: "C8D8EC" } },
], { x: 0.75, y: 4.95, w: 8.5, h: 0.5, fontSize: 10, fontFace: "Calibri", valign: "middle", margin: 0 });


// ══════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Client Scenarios  (fixes: reduced heading-body gap; better bottom label contrast)
// ══════════════════════════════════════════════════════════════════════════
const s6 = pres.addSlide();
lightBg(s6);
sectionLabel(s6, "Business Value");
slideTitle(s6, "Three Enterprise Client Scenarios");

const scenarios = [
  {
    num: "01", title: "Regulatory Intelligence", client: "Banking / Insurance clients",
    body: "Ingest 10,000 regulatory documents. Detect when a new directive supersedes an old one. Surface contradictions before they reach an audit. Show exactly which document mandates a given control.",
    cap: "Document authority · Inference · Contradiction detection",
    bg: NAV,
  },
  {
    num: "02", title: "Audit Knowledge Base", client: "Audit & Assurance teams",
    body: "Link entities across evidence documents. Multi-hop traversal finds connections human reviewers miss. Agentic re-search iterates until an answer is grounded in source material.",
    cap: "6-stage RAG · IRCoT fallback · Session context",
    bg: "065A82",
  },
  {
    num: "03", title: "Compliance Monitoring", client: "Risk & Regulatory consulting",
    body: "Temporal knowledge modeling tracks when facts changed. Authority-weighted confidence scores flag low-quality sources. GDPR-compliant entity erasure on request.",
    cap: "Bitemporal · Confidence calibration · GDPR",
    bg: "164E63",
  },
];

scenarios.forEach((sc, i) => {
  const x = 0.5 + i * 3.05;
  s6.addShape(pres.shapes.RECTANGLE, { x, y: 1.55, w: 2.85, h: 3.72, fill: { color: sc.bg }, line: { color: sc.bg }, shadow: makeShadow() });
  s6.addShape(pres.shapes.RECTANGLE, { x, y: 1.55, w: 2.85, h: 0.07, fill: { color: TEAL }, line: { color: TEAL } });

  s6.addText(sc.num,   { x: x + 0.15, y: 1.68, w: 1, h: 0.52, fontSize: 28, bold: true, color: TEAL2, fontFace: "Calibri", margin: 0 });
  s6.addText(sc.title, { x: x + 0.15, y: 2.17, w: 2.62, h: 0.44, fontSize: 13, bold: true, color: WHITE, fontFace: "Calibri", margin: 0 });
  s6.addText(sc.client, { x: x + 0.15, y: 2.56, w: 2.62, h: 0.28, fontSize: 9.5, color: TEAL2, fontFace: "Calibri", italic: true, margin: 0 });
  // Body moved up (was 2.9, now 2.85) and font slightly larger for readability
  s6.addText(sc.body,   { x: x + 0.15, y: 2.85, w: 2.6, h: 1.65, fontSize: 10.5, color: "C8D8EC", fontFace: "Calibri", margin: 0 });

  // Bottom capability tag — WHITE text for maximum contrast
  s6.addShape(pres.shapes.RECTANGLE, { x: x + 0.1, y: 4.72, w: 2.65, h: 0.42, fill: { color: "0A1428" }, line: { color: TEAL2, width: 0.7 } });
  s6.addText(sc.cap, { x: x + 0.2, y: 4.74, w: 2.45, h: 0.38, fontSize: 8.5, color: TEAL2, fontFace: "Calibri", margin: 0, bold: true });
});


// ══════════════════════════════════════════════════════════════════════════
// SLIDE 7 — Technical proof  (fix: consistent top stripe; "3" ADR note)
// ══════════════════════════════════════════════════════════════════════════
const s7 = pres.addSlide();
lightBg(s7);
sectionLabel(s7, "Technical Foundation");
slideTitle(s7, "Production-Grade. Not a Prototype.");

const stats = [
  { num: "26,600", label: "Lines of production Python" },
  { num: "364",    label: "Passing tests (unit + integration)" },
  { num: "38",     label: "Knowledge graph modules" },
  { num: "6",      label: "ADRs: graph, inference, confidence, LLM, cache, dual-model" },
];

stats.forEach((st, i) => {
  const x = 0.5 + i * 2.28;
  card(s7, x, 1.52, 2.1, 1.32, "FFFFFF");
  // Consistent TEAL stripe on all 4 boxes
  s7.addShape(pres.shapes.RECTANGLE, { x, y: 1.52, w: 2.1, h: 0.055, fill: { color: TEAL }, line: { color: TEAL } });
  s7.addText(st.num,   { x: x + 0.12, y: 1.66, w: 1.86, h: 0.6, fontSize: 29, bold: true, color: NAV, fontFace: "Calibri", margin: 0 });
  s7.addText(st.label, { x: x + 0.12, y: 2.27, w: 1.86, h: 0.45, fontSize: 9.5, color: TXT2, fontFace: "Calibri", margin: 0 });
});

// Two capability columns
const leftCaps = [
  "Forward-chaining inference (Datalog rules)",
  "OWL-RL reasoning + SPARQL bridge (rdflib)",
  "TransE link prediction + entity embeddings",
  "Leiden community detection (graspologic)",
  "Bitemporal: valid time + transaction time",
];
const rightCaps = [
  "Two-model IRCoT: 8B routing (0.2 s) + 70B synthesis (1.5 s) · 9% trigger rate",
  "OAuth 2.0 + JWT + rate limiting + tenant isolation",
  "GDPR erasure cascade + PII detection",
  "RAGAS evaluation (faithfulness, recall, precision)",
  "Runbook, ADRs, contributing guide",
];

card(s7, 0.5, 3.02, 4.35, 2.3, "FFFFFF");
s7.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.02, w: 4.35, h: 0.055, fill: { color: TEAL }, line: { color: TEAL } });
s7.addText("Knowledge Graph Capabilities", { x: 0.65, y: 3.1, w: 4.05, h: 0.35, fontSize: 11, bold: true, color: NAV, fontFace: "Calibri", margin: 0 });
const leftItems = leftCaps.map((c, i) => ({ text: c, options: { bullet: true, color: TXT2, fontSize: 10, fontFace: "Calibri", breakLine: i < leftCaps.length - 1, paraSpaceAfter: 3 } }));
s7.addText(leftItems, { x: 0.58, y: 3.48, w: 4.15, h: 1.75 });

card(s7, 5.15, 3.02, 4.35, 2.3, "FFFFFF");
s7.addShape(pres.shapes.RECTANGLE, { x: 5.15, y: 3.02, w: 4.35, h: 0.055, fill: { color: TEAL }, line: { color: TEAL } });
s7.addText("Production & Operations", { x: 5.3, y: 3.1, w: 4.05, h: 0.35, fontSize: 11, bold: true, color: NAV, fontFace: "Calibri", margin: 0 });
const rightItems = rightCaps.map((c, i) => ({ text: c, options: { bullet: true, color: TXT2, fontSize: 10, fontFace: "Calibri", breakLine: i < rightCaps.length - 1, paraSpaceAfter: 3 } }));
s7.addText(rightItems, { x: 5.23, y: 3.48, w: 4.15, h: 1.75 });


// ══════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Close  (fixes: CTA in a visible box; brighter decorative circles; thicker accent lines)
// ══════════════════════════════════════════════════════════════════════════
const s8 = pres.addSlide();
darkBg(s8);

s8.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.06, h: 5.625, fill: { color: TEAL }, line: { color: TEAL } });

// More visible right-side circles (lighter tint)
s8.addShape(pres.shapes.OVAL, { x: 6.0, y: -1.5, w: 6.0, h: 6.0, fill: { color: TEAL, transparency: 82 }, line: { color: TEAL2, transparency: 65, width: 0.8 } });
s8.addShape(pres.shapes.OVAL, { x: 7.0, y: -0.4, w: 4.0, h: 4.0, fill: { color: TEAL, transparency: 78 }, line: { color: TEAL2, transparency: 55, width: 0.6 } });

s8.addText("READY TO DELIVER", { x: 0.4, y: 0.65, w: 6, h: 0.38, fontSize: 10, bold: true, color: TEAL2, charSpacing: 4, fontFace: "Calibri" });
s8.addText("Built for Client Work.", { x: 0.4, y: 1.02, w: 7.2, h: 1.0, fontSize: 46, bold: true, color: WHITE, fontFace: "Calibri", margin: 0 });

const props = [
  { head: "Day-one delivery", body: "Complete platform ready — no ramp-up on fundamentals" },
  { head: "Regulatory domain expertise", body: "Aerospace compliance demo runs today; maps to banking, audit, and insurance use cases" },
  { head: "Open-source proof", body: "Every claim is verifiable in code — grep for the function, run the test, see it working" },
];

props.forEach((p, i) => {
  const y = 2.2 + i * 0.82;
  // Thicker accent bar (was 0.04, now 0.07)
  s8.addShape(pres.shapes.RECTANGLE, { x: 0.4, y, w: 0.07, h: 0.55, fill: { color: TEAL }, line: { color: TEAL } });
  s8.addText(p.head, { x: 0.58, y: y + 0.02, w: 7.5, h: 0.28, fontSize: 12.5, bold: true, color: WHITE, fontFace: "Calibri", margin: 0 });
  // Body text brighter (was "8BA8CC", now "A8C4DE")
  s8.addText(p.body, { x: 0.58, y: y + 0.29, w: 8.8, h: 0.28, fontSize: 10.5, color: "A8C4DE", fontFace: "Calibri", margin: 0 });
});

// Footer bar
s8.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.1, w: 10, h: 0.525, fill: { color: "070F24" }, line: { color: "070F24" } });
s8.addText("github.com/sergiunicoara/Generative-AI", { x: 0.4, y: 5.13, w: 5.0, h: 0.35, fontSize: 11, color: TEAL2, fontFace: "Calibri" });

// "What's the next step?" — now inside a visible teal box
s8.addShape(pres.shapes.RECTANGLE, { x: 6.4, y: 5.1, w: 3.6, h: 0.525, fill: { color: TEAL }, line: { color: TEAL } });
s8.addText("What's the next step?", { x: 6.45, y: 5.13, w: 3.5, h: 0.38, fontSize: 11.5, bold: true, color: WHITE, fontFace: "Calibri", align: "center", valign: "middle" });


// ══════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Contact / Final
// ══════════════════════════════════════════════════════════════════════════
const sContact = pres.addSlide();
darkBg(sContact);

sContact.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.06, h: 5.625, fill: { color: TEAL }, line: { color: TEAL } });

// Decorative circles
sContact.addShape(pres.shapes.OVAL, { x: 6.0, y: -1.5, w: 6.0, h: 6.0, fill: { color: TEAL, transparency: 82 }, line: { color: TEAL2, transparency: 65, width: 0.8 } });
sContact.addShape(pres.shapes.OVAL, { x: 7.2, y: -0.2, w: 3.8, h: 3.8, fill: { color: TEAL, transparency: 78 }, line: { color: TEAL2, transparency: 55, width: 0.6 } });

sContact.addText("SERGIU NICOARĂ", { x: 0.4, y: 0.55, w: 6, h: 0.38, fontSize: 10, bold: true, color: TEAL2, charSpacing: 4, fontFace: "Calibri" });
sContact.addText("AI Engineer", { x: 0.4, y: 0.9, w: 7.2, h: 0.8, fontSize: 42, bold: true, color: WHITE, fontFace: "Calibri", margin: 0 });
sContact.addText("Graph RAG · Knowledge Graphs · LLM Orchestration", {
  x: 0.4, y: 1.75, w: 6.4, h: 0.4, fontSize: 13, color: "B8D4F0", fontFace: "Calibri", italic: true
});

const contactItems = [
  { icon: "📧", label: "mail4sergiu@gmail.com" },
  { icon: "💼", label: "linkedin.com/in/sergiu-nicoara-31b27013" },
  { icon: "🚀", label: "github.com/sergiunicoara/Generative-AI" },
  { icon: "🌐", label: "sergiunicoara.github.io/iatf-demo" },
];

contactItems.forEach((c, i) => {
  const y = 2.42 + i * 0.58;
  sContact.addShape(pres.shapes.RECTANGLE, { x: 0.4, y, w: 0.07, h: 0.38, fill: { color: TEAL }, line: { color: TEAL } });
  sContact.addText(c.icon + "  " + c.label, { x: 0.6, y: y + 0.04, w: 8.5, h: 0.32, fontSize: 13, color: "C8D8EC", fontFace: "Calibri", margin: 0 });
});

// Bottom CTA box
sContact.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.1, w: 10, h: 0.525, fill: { color: TEAL }, line: { color: TEAL } });
sContact.addText("Timișoara, Romania  ·  Available for remote & on-site engagements", {
  x: 0.4, y: 5.13, w: 9.2, h: 0.38, fontSize: 12, bold: true, color: WHITE, fontFace: "Calibri", align: "center", valign: "middle"
});


// ── Write ──────────────────────────────────────────────────────────────────
const os = require("os");
const path = require("path");
const fs = require("fs");

const targetFile = "C:\\Users\\Sergiu\\Desktop\\GraphRAG_PwC_Pitch.pptx";
const tempFile = path.join(os.tmpdir(), "GraphRAG_PwC_Pitch_temp.pptx");

pres.writeFile({ fileName: tempFile })
  .then(() => {
    // Try to replace the target file
    try {
      if (fs.existsSync(targetFile)) {
        fs.unlinkSync(targetFile);
      }
      fs.renameSync(tempFile, targetFile);
      console.log("Saved: GraphRAG_PwC_Pitch.pptx");
    } catch (err) {
      console.error("Error replacing file (is it still open?):", err.message);
      console.log("Temp file written to:", tempFile);
    }
  })
  .catch(e => console.error("Error:", e));
