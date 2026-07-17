/* Citim împreună — logică joc: completează cuvântul lipsă din fiecare verset, 5 pe pagină. */

const PAGE_SIZE = 5;
const POINTS_PER_VERSE = 10;
const PAGE_CLEAN_BONUS = 20;
const CHEERS = ["Bravo!", "Excelent!", "Minunat!", "Felicitări!", "Super!"];
const STORAGE_SCORE = "ci_score";
const STORAGE_PROGRESS = "ci_progress";
const STORAGE_USER = "ci_user";
const STORAGE_CYCLE = "ci_cycle";

// La fiecare prag de 1000 de puncte, un verset despre Cuvânt apare pe ecran.
// Se arată în ordine (pragul N → versetul N), iar când lista se termină se
// reia de la început. Indexul vine din scor, deci e identic pe orice dispozitiv.
const MILESTONE_STEP = 1000;
const MILESTONE_VERSES = [
  { text: "Strâng Cuvântul Tău în inima mea, ca să nu păcătuiesc împotriva Ta!", ref: "Psalmul 119:11" },
  { text: "Cuvântul Tău este o candelă pentru picioarele mele și o lumină pe cărarea mea.", ref: "Psalmul 119:105" },
  { text: "Cartea aceasta a legii să nu se depărteze de gura ta; cugetă asupra ei zi și noapte, căutând să faci tot ce este scris în ea, căci atunci vei izbândi în toate lucrările tale și atunci vei lucra cu înțelepciune.", ref: "Iosua 1:8" },
  { text: "Fiți împlinitori ai Cuvântului, nu numai ascultători, înșelându-vă singuri.", ref: "Iacov 1:22" },
  { text: "Frica de Domnul este curată și ține pe vecie; judecățile Domnului sunt adevărate, toate sunt drepte. Ele sunt mai de preț decât aurul, decât mult aur curat.", ref: "Psalmul 19:9-10a" },
  { text: "Căci sfatul este o candelă, învățătura este o lumină, iar îndemnul și mustrarea sunt calea vieții.", ref: "Proverbele 6:23" },
  { text: "Ține sfaturile mele și vei trăi.", ref: "Proverbele 7:2" },
  { text: "Cerul și pământul vor trece, dar cuvintele Mele nu vor trece.", ref: "Matei 24:35" },
  { text: "Dacă rămâneți în Cuvântul Meu, atunci sunteți într-adevăr ucenicii Mei. Veți cunoaște adevărul, iar adevărul vă va face liberi.", ref: "Ioan 8:31b-32" },
  { text: "Tu ești adăpostul și scutul meu; îmi pun speranța în Cuvântul Tău.", ref: "Psalmul 119:114" },
  { text: "Căci Domnul dă înțelepciune; din gura Lui iese cunoștință și pricepere.", ref: "Proverbele 2:6" },
];

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
  const before = score;
  score += points;
  el.score.textContent = score;
  el.scoreChip.classList.add("bump");
  setTimeout(() => el.scoreChip.classList.remove("bump"), 200);
  save();
  // Doar creșterile din joc declanșează pragul (nu sincronizarea din cloud,
  // care setează scorul direct și ar putea sări peste mai multe praguri odată).
  const m = Math.floor(score / MILESTONE_STEP);
  if (m > Math.floor(before / MILESTONE_STEP)) {
    // mică pauză, ca mesajul de pagină completată să apuce să se vadă întâi
    setTimeout(() => showMilestone(m), 900);
  }
}

// Ecran special la pragul de m×1000 puncte — scenă cinematică: o Biblie
// deschisă cu foile în evantai, lumină care izbucnește în sus din mijlocul
// ei, scântei care plutesc, iar versetul apare deasupra, plutind în lumină.
// Cartea e SVG inline; restul e CSS (keyframes „m-*” în style.css).
function showMilestone(m) {
  const verse = MILESTONE_VERSES[(m - 1) % MILESTONE_VERSES.length];
  const overlay = document.createElement("div");
  overlay.className = "milestone-overlay";
  // scântei cu poziții/ritmuri aleatoare, urcând din carte
  const sparks = Array.from({ length: 20 }, () =>
    `<span class="m-spark" style="left:${(30 + Math.random() * 40).toFixed(1)}%;` +
    `animation-delay:${(0.9 + Math.random() * 3).toFixed(2)}s;` +
    `animation-duration:${(2.2 + Math.random() * 2).toFixed(2)}s;` +
    `font-size:${(8 + Math.random() * 12).toFixed(0)}px"></span>`
  ).join("");
  overlay.innerHTML = `
    <div class="milestone-scene">
      <div class="m-verse">
        <h2>⭐ ${m * MILESTONE_STEP} de puncte!</h2>
        <p class="milestone-verse">„${verse.text}”</p>
        <p class="milestone-ref">${verse.ref}</p>
        <button class="btn primary">Continuă →</button>
      </div>
      <div class="m-stage">
        <div class="m-beam"></div>
        <div class="m-glow"></div>
        ${sparks}
        <svg class="m-book" viewBox="0 0 340 195" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id="mb-light" cx="50%" cy="45%" r="55%">
              <stop offset="0%" stop-color="#fffbe8"/>
              <stop offset="35%" stop-color="#ffe9a8" stop-opacity="0.9"/>
              <stop offset="100%" stop-color="#ffce54" stop-opacity="0"/>
            </radialGradient>
          </defs>
          <path d="M20,168 Q170,148 320,168 L320,184 Q170,198 20,184 Z" fill="#4a2f1a"/>
          <path d="M30,160 Q170,142 310,160 L310,174 Q170,188 30,174 Z" fill="#e9d9b0"/>
          <path d="M168,158 Q120,120 38,66 Q96,140 168,162 Z" fill="#e9d9b6"/>
          <path d="M168,158 Q128,105 66,42 Q112,130 168,162 Z" fill="#f2e7ca"/>
          <path d="M168,158 Q140,95 102,26 Q130,122 168,162 Z" fill="#f9f1dc"/>
          <path d="M168,158 Q152,90 138,16 Q146,116 168,162 Z" fill="#fdf8ea"/>
          <path d="M172,158 Q220,120 302,66 Q244,140 172,162 Z" fill="#e9d9b6"/>
          <path d="M172,158 Q212,105 274,42 Q228,130 172,162 Z" fill="#f2e7ca"/>
          <path d="M172,158 Q200,95 238,26 Q210,122 172,162 Z" fill="#f9f1dc"/>
          <path d="M172,158 Q188,90 202,16 Q194,116 172,162 Z" fill="#fdf8ea"/>
          <ellipse cx="170" cy="115" rx="95" ry="85" fill="url(#mb-light)"/>
        </svg>
      </div>
    </div>`;
  overlay.querySelector("button").addEventListener("click", () => overlay.remove());
  document.body.appendChild(overlay);
  // artificiile explodează exact când lumina iese din Biblie
  setTimeout(() => launchCelebration("fireworks"), 1100);
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
  launchCelebration();
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

/* --- Sărbători pe canvas, fără dependențe: confetti, artificii, bule, stele --- */
const FX_COLORS = ["#f0b429", "#4c3aa3", "#2e9e5b", "#d64545", "#3e9be0", "#e07be0"];
const FX_KINDS = ["confetti", "fireworks", "bubbles", "stars", "balloons"];

function fxColor() {
  return FX_COLORS[Math.floor(Math.random() * FX_COLORS.length)];
}

function drawStar(ctx, r) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const rad = i % 2 === 0 ? r : r * 0.45;
    const a = (Math.PI / 5) * i - Math.PI / 2;
    ctx.lineTo(Math.cos(a) * rad, Math.sin(a) * rad);
  }
  ctx.closePath();
  ctx.fill();
}

// kind: unul din FX_KINDS sau omis = ales aleator, ca sărbătoarea să fie
// mereu o mică surpriză. Durata ~2.5s, apoi canvas-ul se curăță singur.
function launchCelebration(kind) {
  kind = kind || FX_KINDS[Math.floor(Math.random() * FX_KINDS.length)];
  const canvas = document.getElementById("confetti");
  const ctx = canvas.getContext("2d");
  const W = (canvas.width = window.innerWidth);
  const H = (canvas.height = window.innerHeight);
  let pieces = [];

  if (kind === "confetti") {
    // ploaie de sus + două „tunuri" din colțurile de jos, trase în arc
    pieces = Array.from({ length: 110 }, () => ({
      x: Math.random() * W,
      y: -20 - Math.random() * H * 0.4,
      w: 7 + Math.random() * 7,
      h: 9 + Math.random() * 9,
      color: fxColor(),
      vy: 2.5 + Math.random() * 3.5,
      vx: -1.5 + Math.random() * 3,
      rot: Math.random() * Math.PI,
      vrot: -0.12 + Math.random() * 0.24,
    }));
    for (const side of [-1, 1]) {
      for (let i = 0; i < 45; i++) {
        pieces.push({
          x: side < 0 ? -10 : W + 10,
          y: H * (0.75 + Math.random() * 0.2),
          w: 7 + Math.random() * 7,
          h: 9 + Math.random() * 9,
          color: fxColor(),
          // spre interior și în sus, cu gravitație → arc de tun de confetti
          vx: -side * (4 + Math.random() * 7),
          vy: -(7 + Math.random() * 6),
          rot: Math.random() * Math.PI,
          vrot: -0.2 + Math.random() * 0.4,
          gravity: 0.22,
        });
      }
    }
  } else if (kind === "fireworks") {
    // 6-7 explozii mari, decalate, cu dâre luminoase și licărire
    const bursts = 6 + Math.floor(Math.random() * 2);
    for (let b = 0; b < bursts; b++) {
      const cx = W * (0.12 + Math.random() * 0.76);
      const cy = H * (0.12 + Math.random() * 0.5);
      const color = fxColor();
      const delay = b * 340;
      const sparks = 48;
      for (let i = 0; i < sparks; i++) {
        const a = (Math.PI * 2 * i) / sparks + Math.random() * 0.12;
        const speed = 2.5 + Math.random() * 4.5;
        pieces.push({
          x: cx, y: cy, color, delay,
          vx: Math.cos(a) * speed,
          vy: Math.sin(a) * speed,
          r: 2.2 + Math.random() * 2.4,
          life: 1100 + Math.random() * 600,
          tw: Math.random() * Math.PI * 2,
        });
      }
    }
  } else if (kind === "bubbles") {
    pieces = Array.from({ length: 60 }, () => ({
      x: Math.random() * W,
      y: H + 20 + Math.random() * H * 0.5,
      r: 8 + Math.random() * 18,
      color: fxColor(),
      vy: -(1.5 + Math.random() * 2.5),
      wob: Math.random() * Math.PI * 2,
      wobSpeed: 0.03 + Math.random() * 0.05,
    }));
  } else if (kind === "balloons") {
    // baloane cu aer cald: urcă lin de jos, legănându-se, cu coș și sfori
    pieces = Array.from({ length: 12 }, () => ({
      x: Math.random() * W,
      y: H + 60 + Math.random() * H * 0.35,
      r: 22 + Math.random() * 18,
      color: fxColor(),
      stripe: fxColor(),
      vy: -(1.2 + Math.random() * 1.6),
      wob: Math.random() * Math.PI * 2,
      wobSpeed: 0.015 + Math.random() * 0.02,
    }));
  } else {
    // stars: stele aurii care cad rotindu-se, cu licărire
    pieces = Array.from({ length: 70 }, () => ({
      x: Math.random() * W,
      y: -20 - Math.random() * H * 0.5,
      r: 7 + Math.random() * 9,
      color: Math.random() < 0.7 ? "#f0b429" : fxColor(),
      vy: 1.5 + Math.random() * 2.5,
      vx: -0.8 + Math.random() * 1.6,
      rot: Math.random() * Math.PI,
      vrot: -0.06 + Math.random() * 0.12,
      tw: Math.random() * Math.PI * 2,
    }));
  }

  // Dâre luminoase: în loc de curățare completă, cadrul vechi doar pălește —
  // scânteile lasă urme, ca artificiile adevărate. Baloanele nu (s-ar mânji).
  const FADE = { confetti: 0.32, fireworks: 0.14, bubbles: 0.3, stars: 0.22, balloons: 1 }[kind];
  function fadeFrame() {
    if (FADE >= 1) {
      ctx.clearRect(0, 0, W, H);
    } else {
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = `rgba(0,0,0,${FADE})`;
      ctx.fillRect(0, 0, W, H);
      ctx.globalCompositeOperation = "source-over";
    }
  }

  const start = performance.now();
  function frame(now) {
    const t = now - start;
    fadeFrame();
    for (const p of pieces) {
      if (kind === "confetti") {
        p.x += p.vx; p.y += p.vy; p.rot += p.vrot;
        if (p.gravity) p.vy += p.gravity;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
        ctx.restore();
      } else if (kind === "fireworks") {
        const age = t - p.delay;
        if (age < 0 || age > p.life) continue;
        p.x += p.vx; p.y += p.vy; p.vy += 0.05;
        p.tw += 0.35;
        // licărire + halou luminos în jurul fiecărei scântei
        ctx.globalAlpha = (1 - age / p.life) * (0.7 + Math.sin(p.tw) * 0.3);
        ctx.shadowBlur = 14;
        ctx.shadowColor = p.color;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
      } else if (kind === "bubbles") {
        p.y += p.vy;
        p.wob += p.wobSpeed;
        const x = p.x + Math.sin(p.wob) * 12;
        ctx.globalAlpha = 0.65;
        ctx.shadowBlur = 10;
        ctx.shadowColor = p.color;
        ctx.strokeStyle = p.color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(x, p.y, p.r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.shadowBlur = 0;
        // mic reflex de lumină, ca o bulă adevărată
        ctx.beginPath();
        ctx.arc(x - p.r * 0.35, p.y - p.r * 0.35, p.r * 0.2, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.globalAlpha = 1;
      } else if (kind === "balloons") {
        p.y += p.vy;
        p.wob += p.wobSpeed;
        const x = p.x + Math.sin(p.wob) * 18;
        const r = p.r;
        // anvelopa (formă de picătură) cu o dungă verticală colorată
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(x, p.y, r, Math.PI * 0.85, Math.PI * 0.15, false);
        ctx.quadraticCurveTo(x + r * 0.5, p.y + r * 0.9, x + r * 0.28, p.y + r * 1.15);
        ctx.lineTo(x - r * 0.28, p.y + r * 1.15);
        ctx.quadraticCurveTo(x - r * 0.5, p.y + r * 0.9, x - Math.cos(Math.PI * 0.15) * r, p.y + Math.sin(Math.PI * 0.15) * r);
        ctx.closePath();
        ctx.fill();
        ctx.fillStyle = p.stripe;
        ctx.beginPath();
        ctx.ellipse(x, p.y, r * 0.3, r, 0, Math.PI, 0, false);
        ctx.quadraticCurveTo(x + r * 0.15, p.y + r * 1.05, x + r * 0.1, p.y + r * 1.15);
        ctx.lineTo(x - r * 0.1, p.y + r * 1.15);
        ctx.quadraticCurveTo(x - r * 0.15, p.y + r * 1.05, x - r * 0.3, p.y);
        ctx.closePath();
        ctx.fill();
        // sforile și coșul
        ctx.strokeStyle = "#8a6b3f";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x - r * 0.24, p.y + r * 1.15);
        ctx.lineTo(x - r * 0.16, p.y + r * 1.5);
        ctx.moveTo(x + r * 0.24, p.y + r * 1.15);
        ctx.lineTo(x + r * 0.16, p.y + r * 1.5);
        ctx.stroke();
        ctx.fillStyle = "#a9793f";
        ctx.fillRect(x - r * 0.2, p.y + r * 1.5, r * 0.4, r * 0.32);
      } else {
        p.x += p.vx; p.y += p.vy; p.rot += p.vrot; p.tw += 0.15;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.globalAlpha = 0.65 + Math.sin(p.tw) * 0.35;
        ctx.shadowBlur = 12;
        ctx.shadowColor = p.color;
        ctx.fillStyle = p.color;
        drawStar(ctx, p.r);
        ctx.restore();
        ctx.globalAlpha = 1;
      }
    }
    // baloanele urcă lin, au nevoie de mai mult timp să traverseze ecranul
    if (t < (kind === "balloons" ? 6000 : 3200)) {
      requestAnimationFrame(frame);
    } else {
      ctx.clearRect(0, 0, W, H);
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
