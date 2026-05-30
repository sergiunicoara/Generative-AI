const fs = require('fs');
const raw = fs.readFileSync('.ocr/cards.json', 'utf8').replace(/^﻿/, '');
const ocr = JSON.parse(raw);
const byName = new Map(ocr.map(item => [item.name, item]));
const defaultCenters = [130, 225, 325, 425, 520, 600];

function clean(s) {
  return String(s || '')
    // OCR: ( misread as l — fix before anything else
    .replace(/([A-Za-zĂÂÎȘȚăâîșț])\(/g, '$1l') // letter( → letterl  e.g. Hristosu( → Hristosul
    .replace(/\(ui\b/g, 'lui')                   // (ui → lui
    .replace(/\(a\b/g, 'la')                     // (a → la
    .replace(/\(ângă\b/g, 'lângă')               // (ângă → lângă
    .replace(/\(e\b/g, 'le')                     // (e → le
    // Word-level OCR fixes
    .replace(/\bIui\b/g, 'lui')
    .replace(/\bIa\b/g, 'la')
    .replace(/\bluda\b/g, 'Iuda')
    .replace(/\btoan\b/g, 'Ioan')
    .replace(/\[oan\b/g, 'Ioan')
    .replace(/[\[(]sus\b/g, 'Isus')
    .replace(/\bf'\s*știu\b/g, 'știu')           // f'știu → știu
    .replace(/\bCet\s+ce?\b/g, 'Cel ce')          // Cet c / Cet ce → Cel ce
    .replace(/\bvetr\b/g, 'veți')               // vetr → veți
    .replace(/gasl\b|gasł\b/g, 'găsi')          // gasl / gasł → găsi
    .replace(/\b160b?\b/g, '')                   // stray "160" / "160b" OCR artifacts
    .replace(/\bžse\b/g, '')                     // žse → (remove)
    .replace(/\bEI\b/g, 'El')                    // EI → El (caps misread)
    // Special character substitutions
    .replace(/ť/g, '„')
    .replace(/^[Žž]\s*/g, '')
    .replace(/[Žž]/g, '')                        // remove stray Ž anywhere
    .replace(/\.v/g, ' v')
    .replace(/łg/g, '19')                        // łg → 19 (must come before ł→l)
    .replace(/ł/g, 'l')                          // remaining ł → l
    .replace(/([a-zăâîșț])Đ([lI])/g, '$1t I')    // letterĐl → letter"t I" (e.g. întâlniĐlsus → întâlnit Isus)
    .replace(/\s*Đ\s*/g, ' ')                     // remaining Đ → space
    // Romanian word fixes (no \b on non-ASCII words — boundary doesn't work)
    .replace(/\bOmUl\b/g, 'Omul')
    .replace(/\bboinav/g, 'bolnav')
    .replace(/šase|șase/g, 'Șase')              // šase/șase → Șase
    .replace(/mormânt/g, 'Mormânt')             // capitalize
    .replace(/gasł|gasl/g, 'găsi')             // gasl/gasł → găsi
    .replace(/ludei/g, 'Iudei')                // ludei → Iudei
    .replace(/stainilor/g, 'străinilor')        // stainilor → străinilor
    .replace(/vietii/g, 'vieții')
    .replace(/cezarutl\b/g, 'cezarul')         // cezarutl → cezarul
    .replace(/\(urnina\b/g, 'lumina')          // (urnina → lumina
    .replace(/\(urnea\b/g, 'lumea')            // (urnea → lumea
    .replace(/\bLate\s+(?=[A-ZĂÂÎȘȚa-zăâîșț])/g, '') // stray "Late " prefix
    .replace(/,\s*lui\s*$/g, '')               // trailing ", lui" garbage
    // Whitespace
    .replace(/\s+([,.;:?!])/g, '$1')
    .replace(/\s+/g, ' ')
    .replace(/_\s+_/g, '__')
    .trim();
}

// Strip OCR garbage prefixes that appear on card-back answer lines.
// These are misreads of section-header labels printed on the physical cards.
function cleanOCR(s) {
  if (!s) return s;
  const patterns = [
    // "Cotita, lui XXXX " — garbled section code
    /^Cotita,?\s*(?:lui\s+)?\S*\s*/i,
    // Slujitoare/Slujitorul garbled: "Slu}iłe", "SlugiłĐ", "S lu}ite Catia lui" etc.
    // Extended charset to include Đ and underscore-suffixed chars
    /^[Ss]?\s*[Ll]u[}gj][a-z.,łŁĐ_]{0,6}\s*(?:[Cc][a-z.,_]{1,5}[bBqQ]\s*)?(?:lui\s+)?/i,
    // CatbQ / Ca,t.bo / COŁbQ / CO'łbQ / Cał,bQ etc.
    // 3rd char is now REQUIRED (not optional) so normal words like "Cana", "Caiafa" are not stripped
    /^[Cc][oOaA0']?[łŁtT.,_'][.,_]?[bBoO]?[qQ]?[.,]?\s*(?:lui\s+)?/i,
    // "Catia lui" — another garbled category prefix (must come after CatbQ to catch leftover "ia lui")
    /^[Cc]atia[,.]?\s*(?:lui\s+)?/i,
    // Residual "ia lui" left after stripping "Cat" but not "ia"
    /^ia\s+lui\s+/i,
    // 'catv / 'cav / 'cat / catw variants
    /^'?[Cc]a(?:t[vwW]?|v),?\s*/i,
    // "lui 'catv" / "lui 'cav" — lui followed by catv/cav garbage
    /^lui\s+'?[Cc]a[tv],?\s*/i,
    // "lui 16010 " — number garbage after lui
    /^lui\s+\d[\d\w]*\s*/i,
    // "lui Iosif / lui Ioan" — standalone "lui" header line merged into answer
    /^lui\s+(?=[A-ZĂÂÎȘȚ])/i,
    // Leading long number: "16010 ", "16QĐ "
    /^\d{3,}\w*\s+/i,
    // Scripture ref prefix: "8 Moise", "14:3g Capernaum", "14:1 vg Femeia", "12:14-6 2 Din"
    // Optional short lowercase word (like "vg") between ref and answer.
    // Only strips when the answer word is capitalized (so "46 de ani" is preserved).
    /^\d{1,2}(?:[:\-]\d+[\w\-]*)*(?:\s+[\da-z]{1,4})?\s+(?=[A-ZĂÂÎȘȚ])/,
    // "15t,tie" / "15tt.ie"
    /^15[.,]?\w*[.,]?\w+\s*/i,
    // "łbQ lui" / "lbQ lui" — CatbQ variant (ł already converted to l by clean())
    /^[łl][bB][Qq]\s*(?:lui\s+)?/i,
    // "G'łbQ / G'lbQ NNNN" — G-starting OCR garbage code (ł may already be l via clean())
    /^[Gg]'?[łŁl][.,]?[bB][Qq]\s*\d*\s*/i,
    // "Lt.ie lui" / "hăt.ie c-aŁbQ lui" — other garbled label variants
    /^[Lh][a-zăâ]?t[.,]?ie\s*(?:[a-z,.\-]*\s*)?(?:[Cc][a-z.,_]{0,5}[bBqQ]\s*)?(?:lui\s+)?(?:\d+\s+)?/i,
    // "loaĐ / loaÎ / loa Itu,'" — loaĐ-starting OCR garbage (Đ may already be a space from clean())
    /^loa[ĐDarÎ\s]\s*\S*,?\s*/i,
    // "o lui NNNNN" or "o, lui han" — 'o lui' fragment
    /^o,?\s+lui\s+(?:\S+\s+)?/i,
    // "IcaĐ / Ica»v" — garbled proper-noun prefix (IcaĐ, Ica»v, IcaD)
    /^[Ii]ca[ĐD»][vV]?\s*/i,
    // "han," / "hau" / "hau," — truncated OCR prefix (from Ioan/header)
    /^ha[nu][,.]?\s*/i,
    // "v Name" — lone letter v as OCR prefix before a capital word
    /^v\s+(?=[A-ZĂÂÎȘȚ])/,
    // "cui Name" — cui as OCR prefix before a capital word
    /^cui\s+(?=[A-ZĂÂÎȘȚ])/i,
    // "ia Name" — ia as OCR prefix before a capital word
    /^ia\s+(?=[A-ZĂÂÎȘȚ])/,
    // "luô / luo" — luô-starting OCR prefix
    /^lu[ôo]\s+/i,
    // Leading single special char remnants: Đ, _, etc.
    /^[Đ_]\s*/,
    // "(Ioan Name" — reference that bled into answer, strip "(Ioan " together (requires paren)
    /^[({\[]\s*Ioan\s+(?=[A-ZĂÂÎȘȚ])/i,
    // Leading dot or comma artifact (e.g. ". Maria și Marta")
    /^[.,]\s+(?=[A-ZĂÂÎȘȚ])/,
    // Other leading open paren/bracket
    /^[({\[]\s*/,
    // Residual "_e " prefix (e.g. from "ł_e Ioan Botezătorul")
    /^_?e\s+/i,
    // "-WORD lui" — dash-prefixed garbage (e.g. "-a,tbo lui")
    /^-\S*\s*(?:lui\s+)?/i,
  ];
  let prev;
  do {
    prev = s;
    for (const p of patterns) s = s.replace(p, '').trim();
  } while (s !== prev);
  // Strip trailing OCR reference fragments: "- loał", "- loar", "- loaĐ" (misread "- Ioan")
  return clean(
    s.replace(/\s*-\s*loa[łĐDar]\s*$/i, '')
     .replace(/\s*-\s*Ioan\s*$/i, '')
     .replace(/\s*[-–—,.']\s*$/, '')
  );
}

function centers(item, answerSide) {
  const found = {};
  for (const line of item.lines) {
    const t = line.text.trim();
    const m = answerSide ? t.match(/^(?:([1-6])|14|q|S)\./i) : t.match(/^([1-6]|S)$/);
    if (!m) continue;
    const n = t.startsWith('14') || /^q\./i.test(t) ? 4 : /^S/i.test(t) ? 5 : Number(m[1]);
    if (n >= 1 && n <= 6) found[n] = line.y;
  }
  const arr = [];
  for (let i = 1; i <= 6; i++) arr[i - 1] = found[i] ?? null;
  for (let i = 0; i < 6; i++) {
    if (arr[i] != null) continue;
    let lo = i - 1;
    while (lo >= 0 && arr[lo] == null) lo--;
    let hi = i + 1;
    while (hi < 6 && arr[hi] == null) hi++;
    if (lo >= 0 && hi < 6) {
      arr[i] = arr[lo] + (arr[hi] - arr[lo]) * (i - lo) / (hi - lo);
    } else if (hi < 6) {
      arr[i] = arr[hi] - (defaultCenters[hi] - defaultCenters[i]);
    } else if (lo >= 0) {
      arr[i] = arr[lo] + (defaultCenters[i] - defaultCenters[lo]);
    } else {
      arr[i] = defaultCenters[i];
    }
  }
  return arr;
}

function splitCard(item, answerSide = false) {
  const c = centers(item, answerSide);
  const mids = [];
  for (let i = 0; i < 5; i++) mids.push((c[i] + c[i + 1]) / 2);
  const groups = Array.from({ length: 6 }, () => []);

  for (const line of item.lines) {
    let t = line.text.trim();
    if (!t || /www\./i.test(t) || line.y < 25) continue;
    if (line.y < 115 && /^(S|Ș|SȘ|Sl|Sluj|Ist|Int|Înv|C[aă]rtea|Ca|Cat|Ž|Law)/i.test(t)) continue;
    if (!answerSide && /^[1-6S]$/.test(t)) continue;
    if (answerSide) t = t.replace(/^(?:[1-6]|14|q)\.\s*/i, '').trim();
    if (!t) continue;
    let idx = 0;
    while (idx < mids.length && line.y > mids[idx]) idx++;
    if (idx < 6) groups[idx].push({ ...line, text: t });
  }

  return groups.map(group => clean(
    group.sort((a, b) => a.y - b.y || a.x - b.x).map(line => line.text).join(' ')
  ));
}

function splitAnswer(rawAnswer) {
  const s = clean(rawAnswer).replace(/łg/g, '19').replace(/ł(?=\d)/g, '1').replace(/\bg:/g, '9:').replace(/\bl:(\d)/g, '1:$1');
  // Only treat "Ioan chapter[:verse]" as a reference (requires a number after Ioan)
  const m = s.match(/^(.*?)\s*-\s*(Ioan\s+\d[\d:.]*)/i);
  if (m) return { answer: clean(m[1]), ref: clean(m[2]) };
  const joan = s.match(/^(.*?)\s*(Ioan\s+\d[\d:.]*)/i);
  if (joan) return { answer: clean(joan[1]), ref: clean(joan[2]) };
  return { answer: s || 'Răspunsul indicat pe card', ref: '' };
}

function q3Choices(question) {
  const found = {};
  const re = /([abc])\.\s*(.*?)(?=\s+[abc]\.\s|$)/gi;
  for (const m of question.matchAll(re)) found[m[1].toLowerCase()] = clean(m[2]);
  return ['a', 'b', 'c'].map(letter => found[letter] || `Varianta ${letter}`);
}

function removeChoices(question) {
  return clean(question.replace(/\s+[abc]\.\s*.*?(?=\s+[abc]\.\s|$)/gi, ''));
}

function correctLetters(answer) {
  const m = answer.toLowerCase().match(/[abc](?:\s*,\s*[abc])*/);
  if (!m) return [0];
  return [...new Set(m[0].split(',').map(x => x.trim()).filter(Boolean))].map(letter => letter.charCodeAt(0) - 97);
}

function cardLabel(page, cardIndex) {
  return `Cartea lui Ioan · Card ${page}.${String(cardIndex).padStart(2, '0')}`;
}

function hasRealQuestionContent(parts) {
  return parts.filter(part => part.replace(/[^A-Za-zĂÂÎȘȚăâîșț0-9]/g, '').length >= 12).length >= 5;
}

function hasRealAnswerContent(parts) {
  return parts.join(' ').replace(/[^A-Za-zĂÂÎȘȚăâîșț0-9]/g, '').length >= 24;
}

// ── Pass 1: generate questions, record correct answer only ──────────────────
const generated = [];
const pages = [...new Set(
  ocr
    .map(item => item.name.match(/^fata(\d+)_\d{2}$/))
    .filter(Boolean)
    .map(match => Number(match[1]))
)].sort((a, b) => a - b);

for (const page of pages) {
  const cardIds = ocr
    .map(item => item.name.match(new RegExp(`^fata${page}_(\\d{2})$`)))
    .filter(Boolean)
    .map(match => Number(match[1]))
    .sort((a, b) => a - b);

  for (const card of cardIds) {
    const id = String(card).padStart(2, '0');
    const front = byName.get(`fata${page}_${id}`);
    const back = byName.get(`verso${page}_${id}`);
    if (!front || !back) continue;
    const questions = splitCard(front, false);
    const answers = splitCard(back, true);
    if (!hasRealQuestionContent(questions) || !hasRealAnswerContent(answers)) continue;

    for (let i = 0; i < 6; i++) {
      const qNumber = i + 1;
      const answerInfo = splitAnswer(answers[i]);
      // Apply OCR garbage cleanup to the extracted answer
      answerInfo.answer = cleanOCR(answerInfo.answer);
      const label = cardLabel(page, Number(id));
      const reference = answerInfo.ref ? ` (${answerInfo.ref}).` : '.';

      if (qNumber === 3) {
        const letters = correctLetters(answerInfo.answer);
        generated.push({
          card: label,
          question: removeChoices(questions[i]) || `Selectează varianta corectă pentru întrebarea ${qNumber}.`,
          answers: q3Choices(questions[i]),
          correct: letters.length === 1 ? letters[0] : letters,
          feedback: `Răspunsul corect este ${answerInfo.answer || letters.map(x => String.fromCharCode(97 + x)).join(', ')}${reference}`
        });
      } else if (qNumber === 6) {
        const isTrue = /^a\b/i.test(answerInfo.answer);
        generated.push({
          card: label,
          question: questions[i] || `Afirmația de pe card este adevărată.`,
          answers: ['Adevărat', 'Fals'],
          correct: isTrue ? 0 : 1,
          feedback: `Afirmația este ${isTrue ? 'adevărată' : 'falsă'}${reference}`
        });
      } else {
        const ans = answerInfo.answer || 'Răspunsul indicat pe card';
        // Store only the correct answer for now; distractors added in pass 2
        generated.push({
          card: label,
          question: questions[i] || `Care este răspunsul corect pentru întrebarea ${qNumber}?`,
          answers: [ans],   // [correct] only — filled out below
          correct: 0,
          feedback: `Răspunsul corect este: ${ans}${reference}`,
          _seed: page * 100 + Number(id) * 10 + qNumber,
        });
      }
    }
  }
}

// ── Pass 2: smart distractors ───────────────────────────────────────────────
function looksClean(s) {
  if (!s || s.length < 3) return false;
  if (s === 'Răspunsul indicat pe card') return false;
  // Must start with uppercase Romanian letter (not a quote or digit — quotes handled separately)
  if (!/^[A-ZĂÂÎȘȚ]/.test(s)) return false;
  // Answers starting with "Ioan chapter:verse" are scripture refs, not answers
  if (/^Ioan\s+\d/.test(s)) return false;
  // No non-Romanian OCR special chars
  if (/[łŁ{}|ĐđŁ»«žŽ]/.test(s)) return false;
  // No scripture ref pattern
  if (/\d{2,}:\d/.test(s)) return false;
  // No standalone number in the middle (e.g. "Maria 45 grădinarul")
  if (/\s\d+\s/.test(s)) return false;
  // Unbalanced parens indicate an OCR split fragment
  const opens = (s.match(/[({\[]/g) || []).length;
  const closes = (s.match(/[)}\]]/g) || []).length;
  if (opens !== closes) return false;
  // Known OCR code fragments
  if (/[Gg]'?ł|łb[Qq]|tbo\s|'catv|'cav\b/.test(s)) return false;
  if (/loa[ĐDar]\b|[Ii]ca[ĐD]|[Ll]t\.\s*ie/.test(s)) return false;
  if (/^Law\s/.test(s)) return false;
  return true;
}

// ── Answer-structure classifier ─────────────────────────────────────────────
// Biblical person names that appear in the Gospel of John (used to detect true person answers)
const PERSON_RE = /\b(Simon|Petru|Ioan|Nicodim|Pilat|Fariseii|Fariseilor|Ucenicii|Ucenicilor|Maria|Marta|Lazăr|Duhul|Caiafa|Iudeii|Isus|Moise|Filip|Toma|Andrei|Iuda|Baraba|Natanael|Iosif|Orbul|Samariteanca|Preoții|Leviților|Ana|Malhu|Botezătorul|Magdalena|Arimateea|Dumnezeu|Satana|Diavolul|Iosif|Nicodim|Centurion|Ostașii|Aprozii)\b/i;

function answerSubtype(a) {
  if (!a) return 'general';
  // Direct-speech quotes
  if (/^[„"]/.test(a)) return 'quote';
  const firstWord = a.split(/\s+/)[0];
  // Dative/genitive plural (recipients): Vânzătorilor, Fariseilor, Ucenicilor
  if (/ilor$|lor$/.test(firstWord)) return 'recipient';
  // Locative phrases starting with a preposition
  if (/^(La\s|În\s|Lângă\s|De\s+la\s|Prin\s|Sub\s|Pe\s|Spre\s|Până\s|Dinspre\s)/.test(a)) return 'locative';
  // Speaker + recipient pairs separated by comma: "Isus, lui Nicodim"
  if (/,/.test(a) && a.split(/\s+/).length <= 9) return 'pair';
  // True person name: contains a known biblical name
  if (PERSON_RE.test(a)) return 'true-person';
  // Other capital-starting answers (things, concepts, places — not people)
  return 'general';
}

function looksCleanQuote(s) {
  if (!s || s.length < 5 || !/^[„"]/.test(s)) return false;
  return !/[łŁĐžŽ»«]/.test(s);
}

// Build subtype pools from all clean correct answers (4-choice questions)
const subtypePools = {};
generated
  .filter(q => q.answers.length === 1)
  .forEach(q => {
    const ans = q.answers[0];
    const ok = /^[„"]/.test(ans) ? looksCleanQuote(ans) : looksClean(ans);
    if (!ok) return;
    const st = answerSubtype(ans);
    if (!subtypePools[st]) subtypePools[st] = new Set();
    subtypePools[st].add(ans);
  });
// Convert to sorted arrays; 'general' is the universal fallback
const allClean = [...new Set(
  Object.values(subtypePools).flatMap(s => [...s])
)];
const spools = {};
Object.entries(subtypePools).forEach(([k, s]) => spools[k] = [...s]);
spools.general = allClean;

console.log('Subtype pool sizes:', Object.entries(spools).map(([k,v])=>`${k}:${v.length}`).join(' '));

generated.forEach((q, qi) => {
  if (q.answers.length !== 1) return;   // skip Q3 and Q6

  const correct = q.answers[0];
  const seed = q._seed ?? qi;
  delete q._seed;

  const correctWords = correct.split(/\s+/).length;
  const st = answerSubtype(correct);

  // Pick from same-subtype pool, prefer ±1 word count, fall back progressively
  function tryPool(pool, maxDelta) {
    return (pool || []).filter(a => a !== correct && Math.abs(a.split(/\s+/).length - correctWords) <= maxDelta);
  }
  const pool = spools[st] || spools.general;
  const candidatePool =
    tryPool(pool, 1).length >= 4 ? tryPool(pool, 1) :
    tryPool(pool, 2).length >= 4 ? tryPool(pool, 2) :
    tryPool(pool, 5).length >= 4 ? tryPool(pool, 5) :
    spools.general.filter(a => a !== correct);

  // Deterministic Fisher-Yates shuffle seeded by question seed
  let rngState = seed;
  const rng = () => { rngState = (rngState * 1664525 + 1013904223) >>> 0; return rngState; };
  const shuffled = candidatePool.slice();
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = rng() % (i + 1);
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  const distractors = shuffled.slice(0, 3);
  const choices = [correct, ...distractors];

  // Rotate so the correct answer isn't always first
  const shift = seed % choices.length;
  const rotated = choices.slice(shift).concat(choices.slice(0, shift));

  q.answers = rotated;
  q.correct = rotated.indexOf(correct);
});

// ── Manual overrides for OCR-unreadable answers ─────────────────────────────
// Keyed by [card label, question substring] → correct answer + bible ref
const MANUAL = [
  {
    card: 'Cartea lui Ioan · Card 3.06',
    qSnippet: 'preoți și leviți',
    answer: 'Ioan Botezătorul',
    ref: 'Ioan 1:21',
  },
  {
    card: 'Cartea lui Ioan · Card 9.02',
    qSnippet: 'va învăța toate lucrurile',
    answer: 'Duhul Sfânt',
    ref: 'Ioan 14:26',
  },
  {
    card: 'Cartea lui Ioan · Card 9.10',
    qSnippet: 'scop a venit Domnul Isus',
    answer: 'Ca să ne mântuiască',
    ref: 'Ioan 3:17',
  },
  {
    card: 'Cartea lui Ioan · Card 2.10',
    qSnippet: 'viață veșnică',
    garbagePattern: /invafrc|%/,
    answer: 'Isus, Fiul lui Dumnezeu',
    ref: 'Ioan 3:36',
  },
  {
    card: 'Cartea lui Ioan · Card 7.04',
    qSnippet: 'nu merg oile după un străin',
    garbagePattern: /han,\s*Nu/i,
    answer: 'Nu cunosc glasul străinilor',
    ref: 'Ioan 10:5',
  },
  {
    card: 'Cartea lui Ioan · Card 3.08',
    qSnippet: 'dacă L-ai luat',
    garbagePattern: /că\s+e\s+\d+/,
    answer: 'Maria, lui Isus (crezând că e grădinarul)',
    ref: 'Ioan 20:15',
  },
  {
    card: 'Cartea lui Ioan · Card 7.09',
    qSnippet: 'Toma, după ce Isus l-a invitat',
    garbagePattern: /Dumnezeul\s*$/,
    answer: '„Domnul meu și Dumnezeul meu!"',
    ref: 'Ioan 20:28',
  },
  {
    card: 'Cartea lui Ioan · Card 8.01',
    qSnippet: 'ce fel de moarte',
    garbagePattern: /altul te va\s*[""]?\s*$/,
    answer: '„când vei îmbătrâni, altul te va încinge și te va duce unde nu vei vrea"',
    ref: 'Ioan 21:18',
  },
  {
    card: 'Cartea lui Ioan · Card 9.09',
    qSnippet: 'De unde mă cunoști',
    garbagePattern: /Ioana\s+1:/,
    answer: 'Natanael',
    ref: 'Ioan 1:48',
  },
];

generated.forEach(q => {
  if (q.answers.length !== 4) return;
  const correct = q.answers[q.correct];
  const isPlaceholder = correct === 'Răspunsul indicat pe card';
  const fix = MANUAL.find(m =>
    q.card === m.card &&
    q.question.includes(m.qSnippet) &&
    (isPlaceholder || (m.garbagePattern && m.garbagePattern.test(correct)))
  );
  if (!fix) return;
  // Replace placeholder with known correct answer and rebuild distractors
  const fixSt = answerSubtype(fix.answer);
  const fixPool = spools[fixSt] || spools.general;
  const distractors = fixPool
    .filter(a => a !== fix.answer)
    .sort(() => 0.5 - Math.random())
    .slice(0, 3);
  const choices = [fix.answer, ...distractors];
  const shift = choices.length - 1;
  const rotated = choices.slice(shift).concat(choices.slice(0, shift));
  q.answers = rotated;
  q.correct = rotated.indexOf(fix.answer);
  q.feedback = `Răspunsul corect este: ${fix.answer} (${fix.ref}).`;
  console.log('Manual fix applied:', q.card, '->', fix.answer);
});

// ── Write output ────────────────────────────────────────────────────────────
const pageRange = pages.length
  ? `fata${pages[0]}..${pages[pages.length - 1]} / verso${pages[0]}..${pages[pages.length - 1]}`
  : 'no source pages';
const output = `// ════════════════════════════════════════════════════════════\n// QUESTIONS — Cartea lui Ioan\n// Generated from ${pageRange}.\n// Rules:\n//   Q3 → only the a/b/c options shown on the card (multi-select allowed)\n//   Q6 → always Adevărat / Fals only (2 choices)\n//   Q1,Q2,Q4,Q5 → distractors sampled from same-word-count correct answers\n// ════════════════════════════════════════════════════════════\n\nconst QUESTIONS = ${JSON.stringify(generated, null, 2)};\n`;
fs.writeFileSync('questions.js', output, 'utf8');
console.log(`pages=${pages.join(',')} generated=${generated.length}`);
