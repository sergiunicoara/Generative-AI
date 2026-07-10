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
  statsBtn: document.getElementById("stats-btn"),
  nameModal: document.getElementById("name-modal"),
  nameInput: document.getElementById("name-input"),
  nameSave: document.getElementById("name-save"),
  nameWarning: document.getElementById("name-warning"),
  bookTitle: document.getElementById("book-title"),
};

let score = parseInt(localStorage.getItem(STORAGE_SCORE), 10) || 0;
let page = parseInt(localStorage.getItem(STORAGE_PROGRESS), 10) || 0;
let userName = localStorage.getItem(STORAGE_USER) || "";
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

/* --- Utilizator: nume salvat local, schimbabil prin tap pe chip --- */
function updateUserChip() {
  el.userName.textContent = userName || "…";
}

function showNameModal() {
  el.nameInput.value = userName;
  el.nameModal.hidden = false;
  el.nameWarning.hidden = true;
  el.nameInput.focus();
}

async function nameTakenByOther(name) {
  if (!Tracker.enabled) return false;
  try {
    const events = await Tracker.fetchAll();
    const lower = name.toLowerCase();
    return events.some((e) => (e.user_name || "").toLowerCase() === lower) && lower !== userName.toLowerCase();
  } catch {
    return false; // fără internet — nu blocăm jocul pentru o verificare opțională
  }
}

async function saveName() {
  const name = el.nameInput.value.trim();
  if (!name) return;

  const warningAlreadyShown = !el.nameWarning.hidden;
  if (!warningAlreadyShown && (await nameTakenByOther(name))) {
    el.nameWarning.hidden = false;
    return;
  }

  userName = name;
  localStorage.setItem(STORAGE_USER, userName);
  el.nameModal.hidden = true;
  el.nameWarning.hidden = true;
  updateUserChip();
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
  const correctVerses = new Set(userEvents.filter((e) => e.correct).map((e) => e.verse_ref));
  let points = correctVerses.size * POINTS_PER_VERSE;

  for (let i = 0; i < VERSES.length; i += PAGE_SIZE) {
    const pageRefs = VERSES.slice(i, i + PAGE_SIZE).map((v) => v.ref);
    const pageEvents = userEvents.filter((e) => pageRefs.includes(e.verse_ref));
    const allSolved = pageRefs.every((ref) => correctVerses.has(ref));
    const anyMistake = pageEvents.some((e) => !e.correct);
    if (allSolved && !anyMistake) points += PAGE_CLEAN_BONUS;
  }
  return points;
}

function groupByUser(events) {
  const byUser = new Map();
  for (const e of events) {
    if (!byUser.has(e.user_name)) byUser.set(e.user_name, []);
    byUser.get(e.user_name).push(e);
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
el.userChip.addEventListener("click", showNameModal);
el.nameSave.addEventListener("click", saveName);
el.nameInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") saveName();
});
el.statsBtn.addEventListener("click", renderStats);

el.score.textContent = score;
updateUserChip();
if (!userName) showNameModal();
if (page >= totalPages) page = 0;
renderPage();
Tracker.flush();

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}
