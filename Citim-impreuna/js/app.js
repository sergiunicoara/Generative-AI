/* Citim împreună — logică joc: completează cuvântul lipsă din fiecare verset, 5 pe pagină. */

const PAGE_SIZE = 5;
const POINTS_PER_VERSE = 10;
const PAGE_CLEAN_BONUS = 20;
const CHEERS = ["Bravo!", "Excelent!", "Minunat!", "Felicitări!", "Super!"];
const STORAGE_SCORE = "ci_score";
const STORAGE_PROGRESS = "ci_progress";
const STORAGE_USER = "ci_user";
const STORAGE_CYCLE = "ci_cycle";

const totalPages = Math.ceil(VERSES.length / PAGE_SIZE);

// Verset → indexul paginii pe care se află (pentru sincronizarea progresului
// între dispozitive, calculată din evenimentele stocate în Supabase).
const REF_PAGE = new Map(VERSES.map((v, i) => [v.ref, Math.floor(i / PAGE_SIZE)]));

const el = {
  score: document.getElementById("score"),
  scoreChip: document.getElementById("score-chip"),
  progress: document.getElementById("progress"),
  container: document.getElementById("verses-container"),
  cheer: document.getElementById("cheer"),
  checkBtn: document.getElementById("check-btn"),
  nextBtn: document.getElementById("next-btn"),
  userChip: document.getElementById("user-chip"),
  userName: document.getElementById("user-name"),
  logoutBtn: document.getElementById("logout-btn"),
  statsBtn: document.getElementById("stats-btn"),
  bookTitle: document.getElementById("book-title"),
  // auth modal
  authModal: document.getElementById("auth-modal"),
  authLogged: document.getElementById("auth-logged"),
  authUsernameDisplay: document.getElementById("auth-username-display"),
  loginUsername: document.getElementById("login-username"),
  loginPassword: document.getElementById("login-password"),
  loginError: document.getElementById("login-error"),
  loginBtn: document.getElementById("login-btn"),
  authLogoutBtn: document.getElementById("auth-logout-btn"),
};

let score = parseInt(localStorage.getItem(STORAGE_SCORE), 10) || 0;
let page = parseInt(localStorage.getItem(STORAGE_PROGRESS), 10) || 0;
// Ciclul curent = de câte ori a parcurs toată cartea (1+2 Samuel). Crește doar
// la terminarea completă. Fiecare verset aduce puncte o dată PER CICLU, deci
// reluarea unei pagini în același ciclu nu mai adaugă puncte.
let cycle = parseInt(localStorage.getItem(STORAGE_CYCLE), 10) || 0;
let userName = "";
let hadMistake = false;

function save() {
  localStorage.setItem(STORAGE_SCORE, String(score));
  localStorage.setItem(STORAGE_PROGRESS, String(page));
  localStorage.setItem(STORAGE_CYCLE, String(cycle));
}

// Pagini deja terminate complet (corect) în ciclul curent — interzice reluarea
// lor (din reload de browser sau din „← Înapoi la citire"), ca să nu se poată
// da aceeași pagină de mai multe ori consecutiv fără să înceapă un ciclu nou.
const STORAGE_SOLVED = "ci_solved_pages";
function loadSolvedThisCycle() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_SOLVED) || "null");
    if (raw && raw.cycle === cycle) return new Set(raw.pages);
  } catch {
    // ignorăm date corupte, pornim de la un set gol
  }
  return new Set();
}
let solvedThisCycle = loadSolvedThisCycle();
function saveSolvedThisCycle() {
  localStorage.setItem(STORAGE_SOLVED, JSON.stringify({ cycle, pages: [...solvedThisCycle] }));
}

function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function updateScore(points) {
  score += points;
  el.score.textContent = score;
  el.scoreChip.classList.add("bump");
  setTimeout(() => el.scoreChip.classList.remove("bump"), 200);
  save();
}

// Publică scorul curent în tabelul agregat `scores` (un rând per utilizator),
// ca 📊 să nu mai descarce toate evenimentele. Fire-and-forget.
function pushScore() {
  if (!Tracker.enabled || !userName) return;
  Tracker.upsertScore(userName, score);
}

function buildVerseCard(v) {
  const card = document.createElement("section");
  card.className = "card";

  const ref = document.createElement("div");
  ref.className = "ref";
  ref.textContent = v.ref;
  card.appendChild(ref);

  const verse = document.createElement("p");
  verse.className = "verse";

  const parts = v.text.split(/(\{\d+\})/);
  for (const part of parts) {
    const m = part.match(/^\{(\d+)\}$/);
    if (!m) {
      verse.appendChild(document.createTextNode(part));
      continue;
    }
    const blank = v.blanks[Number(m[1])];
    const select = document.createElement("select");
    select.className = "blank";
    select.dataset.answer = blank.answer;
    select.dataset.ref = v.ref;

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "— alege —";
    placeholder.disabled = true;
    placeholder.selected = true;
    select.appendChild(placeholder);

    for (const word of shuffle(blank.options)) {
      const opt = document.createElement("option");
      opt.value = word;
      opt.textContent = word;
      select.appendChild(opt);
    }
    select.addEventListener("change", () => select.classList.remove("wrong"));
    verse.appendChild(select);
  }
  card.appendChild(verse);
  return card;
}

// Card de citire pentru o pagină deja terminată în acest ciclu — arată cuvintele
// corecte direct, fără select-uri, ca utilizatorul să nu poată răspunde din nou.
function buildSolvedVerseCard(v) {
  const card = document.createElement("section");
  card.className = "card";

  const ref = document.createElement("div");
  ref.className = "ref";
  ref.textContent = v.ref;
  card.appendChild(ref);

  const verse = document.createElement("p");
  verse.className = "verse";

  const parts = v.text.split(/(\{\d+\})/);
  for (const part of parts) {
    const m = part.match(/^\{(\d+)\}$/);
    if (!m) {
      verse.appendChild(document.createTextNode(part));
      continue;
    }
    const blank = v.blanks[Number(m[1])];
    const span = document.createElement("span");
    span.className = "locked";
    span.textContent = blank.answer;
    verse.appendChild(span);
  }
  card.appendChild(verse);
  return card;
}

function renderPage() {
  hadMistake = false;

  // Pagină deja terminată corect în acest ciclu, ultima din carte — se
  // consideră ciclul încheiat (poate fi cazul unui reload chiar înainte ca
  // temporizatorul de celebrare să apuce să declanșeze showFinal()).
  if (solvedThisCycle.has(page) && page + 1 >= totalPages) {
    showFinal();
    return;
  }

  const start = page * PAGE_SIZE;
  const pageVerses = VERSES.slice(start, start + PAGE_SIZE);

  el.progress.textContent = `Pagina ${page + 1} din ${totalPages}`;
  const firstRef = pageVerses[0]?.ref || "";
  el.bookTitle.textContent = firstRef.includes("2 Samuel") ? "Cartea 2 Samuel" : "Cartea 1 Samuel";
  el.cheer.hidden = true;
  el.cheer.innerHTML = "";
  el.container.innerHTML = "";

  if (solvedThisCycle.has(page)) {
    // Deja rezolvată — arătăm versetele complete, fără posibilitatea de a
    // răspunde din nou (blochează reluarea consecutivă a aceleiași pagini).
    el.checkBtn.hidden = true;
    el.nextBtn.hidden = false;
    for (const v of pageVerses) {
      el.container.appendChild(buildSolvedVerseCard(v));
    }
  } else {
    el.checkBtn.hidden = false;
    el.nextBtn.hidden = true;
    for (const v of pageVerses) {
      el.container.appendChild(buildVerseCard(v));
    }
  }

  updateSceneBackground(pageVerses);
}

/* --- Fundal animat pe canvas: scenă colorată aleasă după tema paginii --- */
let lastSceneId = null;
function updateSceneBackground(pageVerses) {
  if (typeof pickScene !== "function" || typeof SceneEngine === "undefined") return;
  const scene = pickScene(pageVerses, page, lastSceneId);
  lastSceneId = scene.id;
  // oglindire pe paginile impare, ca aceeași temă să nu arate identic
  SceneEngine.show(scene, page % 2 === 1);
}

function checkAnswers() {
  const selects = [...el.container.querySelectorAll("select.blank")];
  if (selects.some((s) => s.value === "")) {
    selects.filter((s) => s.value === "").forEach((s) => flashWrong(s, false));
    return;
  }

  let earned = 0;
  for (const s of selects) {
    Tracker.log({
      user_name: userName || "necunoscut",
      verse_ref: s.dataset.ref,
      answer: s.dataset.answer,
      chosen: s.value,
      correct: s.value === s.dataset.answer,
      cycle,
    });
    if (s.value === s.dataset.answer) {
      const span = document.createElement("span");
      span.className = "locked";
      span.textContent = s.dataset.answer;
      s.replaceWith(span);
      earned += POINTS_PER_VERSE;
    } else {
      hadMistake = true;
      flashWrong(s, true);
    }
  }
  Tracker.flush();
  if (earned > 0) updateScore(earned);

  if (!el.container.querySelector("select.blank")) {
    let bonus = 0;
    if (!hadMistake) {
      bonus = PAGE_CLEAN_BONUS;
      updateScore(bonus);
    }
    solvedThisCycle.add(page);
    saveSolvedThisCycle();
    pushScore();
    celebrate(bonus);
  }
}

function flashWrong(select, reset) {
  select.classList.remove("wrong");
  // forțează reluarea animației de scuturare
  void select.offsetWidth;
  select.classList.add("wrong");
  if (reset) select.value = "";
}

function celebrate(bonus) {
  const msg = CHEERS[Math.floor(Math.random() * CHEERS.length)];
  const bonusText = bonus > 0 ? ` (+${bonus} bonus fără greșeli!)` : "";
  el.cheer.innerHTML = `<div class="big">🎉 ${msg}</div><div class="points">Pagină completată${bonusText}</div>`;
  el.cheer.hidden = false;
  el.checkBtn.hidden = true;

  if (page + 1 < totalPages) {
    el.nextBtn.hidden = false;
  } else {
    el.nextBtn.hidden = true;
    setTimeout(showFinal, 1600);
  }
  launchConfetti();
}

function nextPage() {
  page += 1;
  save();
  renderPage();
}

function showFinal() {
  // A terminat toată cartea → ciclu nou. De acum versetele reluate aduc din nou
  // puncte (alt ciclu), dar în cadrul aceluiași ciclu reluarea nu adaugă nimic.
  cycle += 1;
  solvedThisCycle = new Set();
  page = 0;
  save();
  saveSolvedThisCycle();

  el.progress.textContent = "";
  el.cheer.hidden = true;
  el.checkBtn.hidden = true;
  el.nextBtn.hidden = true;
  el.container.innerHTML = "";

  const div = document.createElement("div");
  div.className = "final card";
  div.innerHTML = `
    <div class="trophy">🏆</div>
    <h2>Ai terminat toate versetele!</h2>
    <p>Punctaj total: <strong>⭐ ${score}</strong></p>
  `;
  const restart = document.createElement("button");
  restart.className = "btn secondary";
  restart.style.marginTop = "16px";
  restart.textContent = "Reia de la început";
  restart.addEventListener("click", () => {
    page = 0;
    save();
    renderPage();
  });
  div.appendChild(restart);
  el.container.appendChild(div);
}

/* --- Utilizator: autentificare prin Supabase Auth --- */
function updateUserChip() {
  const name = Auth.currentUser();
  el.userName.textContent = name || "Conectează-te";
  el.logoutBtn.hidden = !name;
}

function showAuthModal() {
  el.authModal.hidden = false;
  if (Auth.isLoggedIn()) {
    el.authLogged.hidden = false;
    document.getElementById("auth-login-panel").hidden = true;
    el.authUsernameDisplay.textContent = Auth.currentUser();
  } else {
    el.authLogged.hidden = true;
    document.getElementById("auth-login-panel").hidden = false;
    el.loginError.hidden = true;
    el.loginUsername.focus();
  }
}

function hideAuthModal() {
  if (!Auth.isLoggedIn()) return;
  el.authModal.hidden = true;
}

function setAuthError(el2, msg) {
  el2.textContent = msg;
  el2.hidden = false;
}

async function handleLogin() {
  const username = el.loginUsername.value.trim();
  const password = el.loginPassword.value;
  if (!username || !password) { setAuthError(el.loginError, "Completează ambele câmpuri."); return; }
  el.loginBtn.disabled = true;
  el.loginBtn.textContent = "Se verifică…";
  try {
    userName = await Auth.signIn(username, password);
    hideAuthModal();
    updateUserChip();
    el.loginPassword.value = "";
    progressSynced = false;
    syncProgressFromCloud();
  } catch (err) {
    setAuthError(el.loginError, err.message);
  } finally {
    el.loginBtn.disabled = false;
    el.loginBtn.textContent = "Intră";
  }
}


async function handleLogout() {
  await Auth.signOut();
  userName = "";
  updateUserChip();
  hideAuthModal();
  showAuthModal();
}

// Calculează unde a rămas utilizatorul, din evenimentele lui din Supabase, DOAR
// pentru ciclul curent (evenimentele din cicluri anterioare nu contează la
// progres): cea mai avansată pagină atinsă în ciclul curent; dacă acea pagină e
// complet rezolvată, continuă la următoarea. Întoarce -1 dacă nu are activitate.
function resumePageFromEvents(events, curCycle) {
  let far = -1;
  const correct = new Set();
  for (const e of events) {
    if ((e.cycle == null ? 0 : e.cycle) !== curCycle) continue;
    const p = REF_PAGE.get(e.verse_ref);
    if (p != null && p > far) far = p;
    if (e.correct) correct.add(e.verse_ref);
  }
  if (far < 0) return -1;
  const pageRefs = VERSES.slice(far * PAGE_SIZE, far * PAGE_SIZE + PAGE_SIZE).map((v) => v.ref);
  const pageDone = pageRefs.every((ref) => correct.has(ref));
  if (pageDone && far + 1 < totalPages) return far + 1;
  return far;
}

// Pagini complet rezolvate corect în ciclul dat, calculate din evenimentele
// cloud — sincronizează restricția „nu poți relua o pagină" și pe alt
// dispozitiv, nu doar pe cel pe care a rezolvat-o inițial.
function computeSolvedPagesForCycle(events, curCycle) {
  const correctByPage = new Map();
  for (const e of events) {
    if (!e.correct) continue;
    if ((e.cycle == null ? 0 : e.cycle) !== curCycle) continue;
    const p = REF_PAGE.get(e.verse_ref);
    if (p == null) continue;
    if (!correctByPage.has(p)) correctByPage.set(p, new Set());
    correctByPage.get(p).add(e.verse_ref);
  }
  const solved = new Set();
  for (const [p, refs] of correctByPage) {
    const pageRefs = VERSES.slice(p * PAGE_SIZE, p * PAGE_SIZE + PAGE_SIZE).map((v) => v.ref);
    if (pageRefs.every((ref) => refs.has(ref))) solved.add(p);
  }
  return solved;
}

// Rulează o singură dată la logare — aduce progresul de pe orice dispozitiv.
let progressSynced = false;
async function syncProgressFromCloud() {
  if (progressSynced || !Tracker.enabled || !userName) return;
  progressSynced = true;
  try {
    const events = await Tracker.fetchUserEvents(userName);

    // Sincronizează ciclul curent de pe orice dispozitiv: cel puțin cel mai mare
    // ciclu întâlnit în evenimente (și nu coborî sub ciclul local).
    let cloudCycle = 0;
    for (const e of events) {
      const c = e.cycle == null ? 0 : e.cycle;
      if (c > cloudCycle) cloudCycle = c;
    }
    cycle = Math.max(cycle, cloudCycle);

    // Scorul canonic din baza de date — aceeași valoare ca în clasament,
    // ca ⭐ din antet să nu mai difere de 📊.
    score = computePointsForUser(events);
    el.score.textContent = score;
    // Împrospătează rândul din clasament cu scorul canonic (și auto-populează
    // tabelul `scores` la prima logare a fiecărui utilizator după migrare).
    pushScore();

    // Paginile deja terminate în ciclul curent, calculate din cloud — astfel
    // restricția „nu poți relua o pagină" ține și dacă schimbă dispozitivul.
    solvedThisCycle = computeSolvedPagesForCycle(events, cycle);
    saveSolvedThisCycle();

    const resume = resumePageFromEvents(events, cycle);
    if (resume >= 0 && resume !== page) {
      page = resume;
    }
    save();
    if (resume >= 0) renderPage();
  } catch {
    // offline sau eroare — rămâne scorul/progresul local, se reîncearcă la login
    progressSynced = false;
  }
}

/* --- Ecran de statistici (agregate din evenimentele Supabase) --- */
async function renderStats() {
  el.progress.textContent = "Statistici";
  el.cheer.hidden = true;
  el.checkBtn.hidden = true;
  el.nextBtn.hidden = true;
  el.container.innerHTML = "";

  const wrap = document.createElement("section");
  wrap.className = "card stats";
  el.container.appendChild(wrap);

  const back = document.createElement("button");
  back.className = "btn secondary";
  back.textContent = "← Înapoi la citire";
  back.addEventListener("click", renderPage);
  el.container.appendChild(back);

  if (!Tracker.enabled) {
    wrap.innerHTML = "<p>Statisticile nu sunt configurate încă.</p>";
    return;
  }

  wrap.innerHTML = "<p>Se încarcă…</p>";
  try {
    // Clasamentul vine din tabelul agregat `scores` (un rând/utilizator),
    // iar panoul personal doar din evenimentele utilizatorului curent — nu
    // se mai descarcă tot istoricul la fiecare deschidere.
    const [scores, myEvents, config] = await Promise.all([
      Tracker.fetchScores(),
      Tracker.fetchUserEvents(userName),
      Tracker.fetchConfig(),
    ]);
    if (scores.length > 0) {
      renderStatsFromScores(wrap, scores, myEvents, config.leaderboard_size);
    } else {
      // Tabelul `scores` încă nu există sau nu e populat — recurge la calculul
      // clasic din toate evenimentele, ca statisticile să funcționeze oricum.
      const events = await Tracker.fetchAll();
      if (events.length === 0) {
        wrap.innerHTML = "<p>Nicio activitate înregistrată încă.</p>";
      } else {
        renderStatsContent(wrap, events, config.leaderboard_size);
      }
    }
  } catch {
    wrap.innerHTML = "<p>Statisticile au nevoie de internet. Încearcă din nou mai târziu.</p>";
  }
}

// Reface exact regulile de punctaj din joc (checkAnswers/celebrate), pornind
// doar din evenimentele brute — nu există un tabel separat de scor.
function computePointsForUser(userEvents) {
  const cycleOf = (e) => (e.cycle == null ? 0 : e.cycle);

  // Punct de bază: fiecare verset corect contează o dată PER CICLU. Reluarea
  // aceleiași pagini în același ciclu nu mai adaugă puncte; abia după ce termină
  // toată cartea (ciclu nou) versetele reluate aduc din nou puncte.
  const seen = new Set(); // "ciclu|verset"
  for (const e of userEvents) {
    if (e.correct) seen.add(cycleOf(e) + "|" + e.verse_ref);
  }
  let points = seen.size * POINTS_PER_VERSE;

  // Bonus de pagină curată — o dată PER (pagină, ciclu), pe baza primei terminări
  // curate din acel ciclu (fără nicio greșeală înainte de finalizare). Complet
  // monoton: o greșeală de mai târziu nu mai poate anula bonusul.
  //
  // O singură trecere prin evenimente ca să le grupăm pe (pagină → ciclu),
  // în loc să reparcurgem toate paginile cărții pentru fiecare utilizator
  // (altfel devine O(utilizatori × pagini × evenimente) la corpusul complet).
  const byPage = new Map(); // pagină → (ciclu → evenimente)
  for (const e of userEvents) {
    const p = REF_PAGE.get(e.verse_ref);
    if (p == null) continue;
    if (!byPage.has(p)) byPage.set(p, new Map());
    const byCycle = byPage.get(p);
    const c = cycleOf(e);
    if (!byCycle.has(c)) byCycle.set(c, []);
    byCycle.get(c).push(e);
  }

  for (const [p, byCycle] of byPage) {
    const pageRefs = VERSES.slice(p * PAGE_SIZE, p * PAGE_SIZE + PAGE_SIZE).map((v) => v.ref);

    for (const cycleEvents of byCycle.values()) {
      const sorted = [...cycleEvents].sort((a, b) =>
        (a.created_at || "").localeCompare(b.created_at || "")
      );
      const solved = new Set();
      let mistakeBeforeDone = false;
      let done = false;
      for (const e of sorted) {
        if (e.correct) solved.add(e.verse_ref);
        else if (!done) mistakeBeforeDone = true;
        if (pageRefs.every((ref) => solved.has(ref))) { done = true; break; }
      }
      if (done && !mistakeBeforeDone) points += PAGE_CLEAN_BONUS;
    }
  }
  return points;
}

function normName(n) {
  return (n || '').replace(/\b\w/g, c => c.toUpperCase());
}

function groupByUser(events) {
  const byUser = new Map();
  for (const e of events) {
    const name = normName(e.user_name);
    if (!byUser.has(name)) byUser.set(name, []);
    byUser.get(name).push(e);
  }
  return byUser;
}

function renderStatsContent(wrap, events, leaderboardSize) {
  const byUser = groupByUser(events);
  const ranking = [...byUser.entries()]
    .map(([name, evts]) => ({ name, points: computePointsForUser(evts) }))
    .sort((a, b) => b.points - a.points);

  wrap.innerHTML = "";

  const board = document.createElement("div");
  board.className = "leaderboard";
  board.innerHTML = "<h3>🏆 Clasament</h3>";
  ranking.slice(0, leaderboardSize).forEach((entry, i) => {
    const row = document.createElement("div");
    row.className = "leaderboard-row" + (entry.name === userName ? " me" : "");
    row.innerHTML = `<span class="rank">${i + 1}</span><span class="who">${entry.name}</span><span class="pts">${entry.points} pct</span>`;
    board.appendChild(row);
  });
  wrap.appendChild(board);
  wrap.appendChild(buildMyStatsPanel(byUser.get(userName)));
}

// Clasament din tabelul agregat `scores` (un rând/utilizator) + panoul personal
// din evenimentele proprii. Calea implicită; renderStatsContent rămâne fallback.
function renderStatsFromScores(wrap, scores, myEvents, leaderboardSize) {
  // Normalizează și fuzionează eventualele duplicate de nume (litere mari/mici).
  const byName = new Map();
  for (const s of scores) {
    const name = normName(s.user_name);
    byName.set(name, Math.max(byName.get(name) || 0, s.points || 0));
  }
  const ranking = [...byName.entries()]
    .map(([name, points]) => ({ name, points }))
    .sort((a, b) => b.points - a.points);

  wrap.innerHTML = "";

  const board = document.createElement("div");
  board.className = "leaderboard";
  board.innerHTML = "<h3>🏆 Clasament</h3>";
  ranking.slice(0, leaderboardSize).forEach((entry, i) => {
    const row = document.createElement("div");
    row.className = "leaderboard-row" + (entry.name === userName ? " me" : "");
    row.innerHTML = `<span class="rank">${i + 1}</span><span class="who">${entry.name}</span><span class="pts">${entry.points} pct</span>`;
    board.appendChild(row);
  });
  wrap.appendChild(board);
  wrap.appendChild(buildMyStatsPanel(myEvents && myEvents.length ? myEvents : null));
}

// Panoul „Statisticile mele" — doar evenimentele utilizatorului curent.
function buildMyStatsPanel(myEvents) {
  const mine = document.createElement("div");
  mine.className = "stat-user";
  if (!myEvents) {
    mine.innerHTML = `<h3>👤 Statisticile mele</h3><p>Nu ai încă activitate înregistrată.</p>`;
    return mine;
  }
  const total = myEvents.length;
  const correct = myEvents.filter((e) => e.correct).length;
  const accuracy = Math.round((correct / total) * 100);
  const distinctVerses = new Set(myEvents.filter((e) => e.correct).map((e) => e.verse_ref)).size;
  const lastDay = new Date(myEvents[0].created_at).toLocaleDateString("ro-RO", {
    day: "numeric",
    month: "short",
  });

  const mistakes = new Map();
  for (const e of myEvents) {
    if (e.correct) continue;
    if (!mistakes.has(e.verse_ref)) {
      mistakes.set(e.verse_ref, { count: 0, wrong: new Set(), answer: e.answer });
    }
    const m = mistakes.get(e.verse_ref);
    m.count += 1;
    m.wrong.add(e.chosen);
  }
  const topMistakes = [...mistakes.entries()].sort((a, b) => b[1].count - a[1].count).slice(0, 5);

  let html = `
    <h3>👤 Statisticile mele</h3>
    <p>Răspunsuri: <strong>${total}</strong> · Corecte: <strong>${correct}</strong> (${accuracy}%)</p>
    <p>Versete completate: <strong>${distinctVerses}</strong> · Ultima activitate: ${lastDay}</p>
  `;
  if (topMistakes.length > 0) {
    html += `<p class="stat-label">Unde am greșit:</p><ul>`;
    for (const [ref, m] of topMistakes) {
      const wrongList = [...m.wrong].map((w) => `„${w}”`).join(", ");
      const label = m.count === 1 ? "greșeală" : "greșeli";
      html += `<li><strong>${ref}</strong> — ${m.count} ${label}: ${wrongList} → corect: „${m.answer}”</li>`;
    }
    html += `</ul>`;
  } else {
    html += `<p class="stat-label">Nicio greșeală — felicitări! 🎉</p>`;
  }
  mine.innerHTML = html;
  return mine;
}

/* --- Confetti simplu pe canvas, fără dependențe --- */
function launchConfetti() {
  const canvas = document.getElementById("confetti");
  const ctx = canvas.getContext("2d");
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const colors = ["#f0b429", "#4c3aa3", "#2e9e5b", "#d64545", "#3e9be0"];
  const pieces = Array.from({ length: 90 }, () => ({
    x: Math.random() * canvas.width,
    y: -20 - Math.random() * canvas.height * 0.4,
    w: 6 + Math.random() * 6,
    h: 8 + Math.random() * 8,
    color: colors[Math.floor(Math.random() * colors.length)],
    vy: 2.5 + Math.random() * 3,
    vx: -1.5 + Math.random() * 3,
    rot: Math.random() * Math.PI,
    vrot: -0.1 + Math.random() * 0.2,
  }));

  const start = performance.now();
  function frame(now) {
    const t = now - start;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const p of pieces) {
      p.x += p.vx;
      p.y += p.vy;
      p.rot += p.vrot;
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
      ctx.restore();
    }
    if (t < 2500) {
      requestAnimationFrame(frame);
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }
  requestAnimationFrame(frame);
}

/* --- Inițializare --- */
el.checkBtn.addEventListener("click", checkAnswers);
el.nextBtn.addEventListener("click", nextPage);
el.statsBtn.addEventListener("click", renderStats);

// auth modal events (optional chaining = rezistență la SW care servește HTML vechi)
el.userChip.addEventListener("click", showAuthModal);
el.logoutBtn?.addEventListener("click", handleLogout);
el.loginBtn?.addEventListener("click", handleLogin);
el.authLogoutBtn?.addEventListener("click", handleLogout);
el.loginPassword?.addEventListener("keydown", (e) => { if (e.key === "Enter") handleLogin(); });
el.authModal?.addEventListener("click", (e) => { if (e.target === el.authModal && Auth.isLoggedIn()) hideAuthModal(); });

el.score.textContent = score;
if (page >= totalPages) page = 0;
renderPage();
Tracker.flush();

// Arătăm modalul imediat — nu așteptăm Supabase (poate fi lent/offline).
// Dacă sesiunea se restaurează, onAuthStateChange ascunde modalul automat.
showAuthModal();

Auth.init((user) => {
  userName = user || "";
  updateUserChip();
  if (user) hideAuthModal();
}).then((user) => {
  userName = user || "";
  updateUserChip();
  if (user) {
    hideAuthModal();
    syncProgressFromCloud();
  }
});

if ("serviceWorker" in navigator) {
  let refreshing = false;
  // Când un service worker NOU preia controlul (doar la o actualizare, nu la
  // prima instalare), reîncărcăm o singură dată — utilizatorul primește
  // automat ultima versiune, fără să șteargă manual cache-ul.
  if (navigator.serviceWorker.controller) {
    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (refreshing) return;
      refreshing = true;
      window.location.reload();
    });
  }
  navigator.serviceWorker
    .register("sw.js")
    .then((reg) => {
      reg.update();
      // verifică periodic dacă a apărut o versiune nouă (aplicația poate sta
      // deschisă zile întregi ca PWA instalat)
      setInterval(() => reg.update(), 30 * 60 * 1000);
    })
    .catch(() => {});
}
