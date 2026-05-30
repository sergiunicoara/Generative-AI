/**
 * generate-distractors.js
 *
 * Uses Claude to replace poor distractors in questions.js with
 * semantically appropriate wrong-answer choices for every 4-choice question.
 *
 * Run:  node generate-distractors.js
 * Env:  ANTHROPIC_API_KEY must be set
 */

'use strict';
const fs = require('fs');
const Anthropic = require('@anthropic-ai/sdk');

// ─── Config ───────────────────────────────────────────────────────────────────
const MODEL          = 'claude-haiku-4-5';   // cheapest; fast enough
const BATCH_SIZE     = 10;                    // questions per API call
const OUTPUT_FILE    = 'questions.js';
const PROGRESS_FILE  = '.distractor-progress.json'; // resume from crashes
// ─────────────────────────────────────────────────────────────────────────────

const client = new Anthropic();

// ─── Load questions ───────────────────────────────────────────────────────────
function loadQuestions() {
  const src = fs.readFileSync(OUTPUT_FILE, 'utf8');
  // Use vm to safely eval without polluting global scope
  const vm = require('vm');
  const ctx = {};
  vm.createContext(ctx);
  // Replace const/let/var with plain assignment so vm can expose it
  const evalsrc = src.replace(/^\s*(?:const|let|var)\s+QUESTIONS\s*=/m, 'QUESTIONS =');
  vm.runInContext(evalsrc, ctx);
  return ctx.QUESTIONS;
}

// ─── Save questions ───────────────────────────────────────────────────────────
function saveQuestions(qs) {
  const src    = fs.readFileSync(OUTPUT_FILE, 'utf8');
  // Find where const QUESTIONS = starts and replace everything from that point
  const idx    = src.indexOf('const QUESTIONS');
  if (idx === -1) throw new Error('Cannot find const QUESTIONS in ' + OUTPUT_FILE);
  const header = src.slice(0, idx);
  const body   = JSON.stringify(qs, null, 2);
  fs.writeFileSync(OUTPUT_FILE, header + 'const QUESTIONS = ' + body + ';\n', 'utf8');
}

// ─── Load / save progress ─────────────────────────────────────────────────────
function loadProgress() {
  if (fs.existsSync(PROGRESS_FILE)) {
    return new Set(JSON.parse(fs.readFileSync(PROGRESS_FILE, 'utf8')));
  }
  return new Set();
}
function saveProgress(done) {
  fs.writeFileSync(PROGRESS_FILE, JSON.stringify([...done]), 'utf8');
}

// ─── Prompt builder ───────────────────────────────────────────────────────────
function buildPrompt(batch) {
  const items = batch.map(({ globalIdx, question, correctAnswer }) =>
    `{"idx":${globalIdx},"question":${JSON.stringify(question)},"correct":${JSON.stringify(correctAnswer)}}`
  ).join('\n');

  return `You are building a Bible quiz about the Gospel of John (Ioan), written in Romanian.

For each question below, I need exactly 3 WRONG answer choices (distractors).
The distractors MUST be:
  • The same type/category as the correct answer
    – person name correct → give 3 other person names from the Gospel of John
    – place name correct  → give 3 other place names from the Gospel of John
    – short phrase/word   → give 3 similarly short phrases in the same grammatical form
    – Bible verse ref     → give 3 other verse references from Ioan (e.g. "Ioan 3:5")
  • Written in Romanian
  • Plausible-sounding (a student might be fooled)
  • Different from the correct answer and from each other
  • Do NOT repeat the correct answer as a distractor
  • Do NOT use ASCII double-quote character " inside distractor text
    (use Romanian curly quotes or single quotes if quoting is needed)

Return ONLY a valid JSON array — no markdown, no explanation.
Each element: {"idx": <number>, "distractors": ["wrong1","wrong2","wrong3"]}

Questions:
${items}`;
}

// ─── Repair JSON that may contain unescaped " inside string values ───────────
// Walks the string character-by-character; any `"` inside a string that is
// NOT followed (skipping whitespace) by , ] } or : is treated as a stray
// content quote and replaced with a single-quote.
function sanitizeJson(text) {
  // 1. Strip markdown fences
  let s = text.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '').trim();

  // 2. \" → ' (escaped-but-problematic quotes become single quotes)
  s = s.replace(/\\"/g, "'");

  // 3. Walk through, repairing bare " inside string values
  let out = '';
  let inStr = false;

  for (let i = 0; i < s.length; i++) {
    const c = s[i];

    if (!inStr) {
      if (c === '"') inStr = true;
      out += c;
      continue;
    }

    // We are inside a JSON string
    if (c === '\\') {
      // Escaped char — pass both characters through unchanged
      out += c;
      if (i + 1 < s.length) { out += s[++i]; }
      continue;
    }

    if (c === '"') {
      // Peek ahead (skip whitespace) to decide: is this the closing delimiter?
      let j = i + 1;
      while (j < s.length && ' \t\r\n'.includes(s[j])) j++;
      const nxt = s[j] || '';
      if (nxt === ',' || nxt === ']' || nxt === '}' || nxt === ':' || nxt === '') {
        // Valid closing " — end of string
        inStr = false;
        out += c;
      } else {
        // Stray quote inside string value — replace with single quote
        out += "'";
      }
      continue;
    }

    out += c;
  }

  return out;
}

// ─── Call Claude for one batch ─────────────────────────────────────────────────
async function callClaude(batch) {
  const prompt = buildPrompt(batch);
  const msg = await client.messages.create({
    model:      MODEL,
    max_tokens: 1024,
    messages: [{ role: 'user', content: prompt }],
  });

  const text = msg.content[0].text.trim();

  // Strip optional markdown fences, then sanitize
  const cleaned = sanitizeJson(text);
  let parsed;
  try {
    parsed = JSON.parse(cleaned);
  } catch (e) {
    console.error('⚠️  JSON parse error for batch', batch.map(b => b.globalIdx), '\nRaw:', text);
    throw e;
  }
  return parsed; // [{idx, distractors:[...]}]
}

// ─── Apply distractors to questions array ─────────────────────────────────────
function applyDistractors(questions, results) {
  for (const { idx, distractors } of results) {
    const q = questions[idx];
    if (!q || q.answers.length !== 4 || typeof q.correct !== 'number') continue;
    if (!Array.isArray(distractors) || distractors.length !== 3) {
      console.warn(`  ⚠ idx ${idx}: got ${distractors?.length} distractors, expected 3`);
      continue;
    }
    // Rebuild answers: keep correct in its position, fill others with distractors
    const correctText = q.answers[q.correct];
    const newAnswers  = [...distractors];
    newAnswers.splice(q.correct, 0, correctText); // re-insert correct at same index
    q.answers = newAnswers;
  }
}

// ─── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error('Error: ANTHROPIC_API_KEY is not set.');
    process.exit(1);
  }

  const questions  = loadQuestions();
  const done       = loadProgress();

  // Collect all 4-choice questions
  const targets = questions
    .map((q, i) => ({ q, i }))
    .filter(({ q }) => Array.isArray(q.answers) && q.answers.length === 4 && typeof q.correct === 'number')
    .filter(({ i }) => !done.has(i));  // skip already-processed

  console.log(`📋 ${questions.length} total questions`);
  console.log(`🎯 ${targets.length} 4-choice questions to process (${done.size} already done)`);

  // Process in batches
  let processed = 0;
  for (let start = 0; start < targets.length; start += BATCH_SIZE) {
    const slice = targets.slice(start, start + BATCH_SIZE);
    const batch = slice.map(({ q, i }) => ({
      globalIdx:     i,
      question:      q.question,
      correctAnswer: q.answers[q.correct],
    }));

    const batchNums = batch.map(b => b.globalIdx);
    process.stdout.write(`  Batch [${batchNums[0]}..${batchNums[batchNums.length - 1]}] — calling Claude... `);

    let results;
    try {
      results = await callClaude(batch);
    } catch (err) {
      console.error('\n  ❌ Error:', err.message, '— skipping batch');
      continue;
    }

    applyDistractors(questions, results);
    results.forEach(r => done.add(r.idx));
    saveProgress(done);
    processed += results.length;
    const total = targets.length + (done.size - results.length); // original skipped + new
    console.log(`✅ (${done.size} of ~${questions.filter(q=>q.answers.length===4&&typeof q.correct==='number').length} done)`);

    // Persist after every batch so a crash doesn't lose work
    saveQuestions(questions);

    // Mild rate-limit courtesy
    if (start + BATCH_SIZE < targets.length) {
      await new Promise(r => setTimeout(r, 300));
    }
  }

  // Clean up progress file on success
  if (fs.existsSync(PROGRESS_FILE)) fs.unlinkSync(PROGRESS_FILE);

  console.log('\n✨ Done! questions.js updated with AI-generated distractors.');
  console.log(`   Total 4-choice questions processed: ${done.size}`);
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
