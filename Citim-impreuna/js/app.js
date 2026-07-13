/* Citim împreună — logică joc: completează cuvântul lipsă din fiecare verset, 5 pe pagină. */

const PAGE_SIZE = 5;
const POINTS_PER_VERSE = 10;
const PAGE_CLEAN_BONUS = 20;
const CHEERS = ["Bravo!", "Excelent!", "Minunat!", "Felicitări!", "Super!"];
const STORAGE_SCORE = "ci_score";
const STORAGE_PROGRESS = "ci_progress";
const STORAGE_USER = "ci_user";

const totalPages = Math.ceil(VERSES.length / PAGE_SIZE);

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
let userName = "";
let hadMistake = false;

function save() {
  localStorage.setItem(STORAGE_SCORE, String(score));
  localStorage.setItem(STORAGE_PROGRESS, String(page));
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

function renderPage() {
  hadMistake = false;

  const start = page * PAGE_SIZE;
  const pageVerses = VERSES.slice(start, start + PAGE_SIZE);

  el.progress.textContent = `Pagina ${page + 1} din ${totalPages}`;
  const firstRef = pageVerses[0]?.ref || "";
  el.bookTitle.textContent = firstRef.includes("2 Samuel") ? "Cartea 2 Samuel" : "Cartea 1 Samuel";
  el.cheer.hidden = true;
  el.cheer.innerHTML = "";
  el.checkBtn.hidden = false;
  el.nextBtn.hidden = true;

  el.container.innerHTML = "";
  for (const v of pageVerses) {
    el.container.appendChild(buildVerseCard(v));
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
  page = 0;
  save();

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
    const [events, config] = await Promise.all([Tracker.fetchAll(), Tracker.fetchConfig()]);
    if (events.length === 0) {
      wrap.innerHTML = "<p>Nicio activitate înregistrată încă.</p>";
    } else {
      renderStatsContent(wrap, events, config.leaderboard_size);
    }
  } catch {
    wrap.innerHTML = "<p>Statisticile au nevoie de internet. Încearcă din nou mai târziu.</p>";
  }
}

// Reface exact regulile de punctaj din joc (checkAnswers/celebrate), pornind
// doar din evenimentele brute — nu există un tabel separat de scor.
function computePointsForUser(userEvents) {
  // Versete: per zi — același verset contează o dată pe zi
  const byDay = new Map();
  for (const e of userEvents) {
    const day = (e.created_at || '').slice(0, 10);
    if (!byDay.has(day)) byDay.set(day, []);
    byDay.get(day).push(e);
  }
  let points = 0;
  for (const dayEvents of byDay.values()) {
    const correct = new Set(dayEvents.filter((e) => e.correct).map((e) => e.verse_ref));
    points += correct.size * POINTS_PER_VERSE;
  }

  // Bonusuri pagină: all-time — toată pagina corectă, fără greșeli niciodată
  const allCorrect = new Set(userEvents.filter((e) => e.correct).map((e) => e.verse_ref));
  for (let i = 0; i < VERSES.length; i += PAGE_SIZE) {
    const pageRefs = VERSES.slice(i, i + PAGE_SIZE).map((v) => v.ref);
    const pageEvents = userEvents.filter((e) => pageRefs.includes(e.verse_ref));
    const allSolved = pageRefs.every((ref) => allCorrect.has(ref));
    const anyMistake = pageEvents.some((e) => !e.correct);
    if (allSolved && !anyMistake) points += PAGE_CLEAN_BONUS;
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

  // Statisticile proprii — doar greșelile utilizatorului curent, nu ale altora.
  const mine = document.createElement("div");
  mine.className = "stat-user";
  const myEvents = byUser.get(userName);
  if (!myEvents) {
    mine.innerHTML = `<h3>👤 Statisticile mele</h3><p>Nu ai încă activitate înregistrată.</p>`;
  } else {
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
  }
  wrap.appendChild(mine);
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
  if (user) hideAuthModal();
});

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}
