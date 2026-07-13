/* Urmărirea activității: coadă offline în localStorage + trimitere batch la Supabase.
   Dacă js/config.js nu are chei, totul devine no-op și aplicația merge ca înainte. */

const Tracker = (() => {
  const QUEUE_KEY = "ci_pending_events";
  const enabled =
    typeof SUPABASE_URL === "string" &&
    SUPABASE_URL.startsWith("https://") &&
    typeof SUPABASE_ANON_KEY === "string" &&
    SUPABASE_ANON_KEY.length > 0;
  let flushing = false;

  function readQueue() {
    try {
      return JSON.parse(localStorage.getItem(QUEUE_KEY)) || [];
    } catch {
      return [];
    }
  }

  function writeQueue(q) {
    localStorage.setItem(QUEUE_KEY, JSON.stringify(q));
  }

  function log(evt) {
    if (!enabled) return;
    const q = readQueue();
    // păstrează momentul real al răspunsului, chiar dacă trimiterea se face mai târziu
    q.push({ ...evt, created_at: new Date().toISOString() });
    writeQueue(q);
    // nu declanșează flush() aici — checkAnswers() apelează log() de mai multe ori
    // la rând (o dată per verset); un singur flush() după buclă evită atât rafala
    // de cereri, cât și cursa în care evenimente adăugate în timpul unui flush
    // în desfășurare rămâneau blocate în coadă (flush-ul următor era ignorat
    // din cauza gărzii "flushing").
  }

  async function flush() {
    if (!enabled || flushing) return;
    const batch = readQueue();
    if (batch.length === 0) return;
    flushing = true;
    let sent = false;
    try {
      const res = await fetch(`${SUPABASE_URL}/rest/v1/events`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          apikey: SUPABASE_ANON_KEY,
          Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
          Prefer: "return=minimal",
        },
        body: JSON.stringify(batch),
      });
      if (res.ok) {
        // golește doar ce s-a trimis; evenimente sosite între timp rămân în coadă
        writeQueue(readQueue().slice(batch.length));
        sent = true;
      }
    } catch {
      // offline sau eroare de rețea — coada rămâne pentru următoarea încercare
    } finally {
      flushing = false;
    }
    // dacă au mai sosit evenimente cât timp acest batch era în tranzit, continuă
    if (sent && readQueue().length > 0) flush();
  }

  async function fetchAll() {
    if (!enabled) return [];
    // Supabase plafonează serverul la 1000 de rânduri per cerere, indiferent de
    // `limit`. Paginăm cu offset până se golește, ca să prindem TOȚI utilizatorii
    // (altfel cei cu evenimente mai vechi dispar din clasament).
    const PAGE = 1000;
    let all = [];
    for (let offset = 0; offset < 100000; offset += PAGE) {
      const url =
        `${SUPABASE_URL}/rest/v1/events` +
        `?select=user_name,verse_ref,answer,chosen,correct,created_at` +
        `&order=created_at.desc&limit=${PAGE}&offset=${offset}`;
      const res = await fetch(url, {
        headers: {
          apikey: SUPABASE_ANON_KEY,
          Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
        },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const rows = await res.json();
      all = all.concat(rows);
      if (rows.length < PAGE) break;
    }
    return all;
  }

  async function fetchUserEvents(userName) {
    if (!enabled || !userName) return [];
    // ilike fără wildcards = egalitate case-insensitive (prinde și numele
    // vechi salvate cu literă mică, ex. "sergiu" vs "Sergiu").
    // Paginat (cap Supabase = 1000/cerere) ca scorul să fie corect chiar și
    // pentru cine depășește 1000 de evenimente.
    const PAGE = 1000;
    let all = [];
    for (let offset = 0; offset < 100000; offset += PAGE) {
      const url =
        `${SUPABASE_URL}/rest/v1/events` +
        `?select=verse_ref,correct,created_at` +
        `&user_name=ilike.${encodeURIComponent(userName)}` +
        `&order=created_at.desc&limit=${PAGE}&offset=${offset}`;
      const res = await fetch(url, {
        headers: {
          apikey: SUPABASE_ANON_KEY,
          Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
        },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const rows = await res.json();
      all = all.concat(rows);
      if (rows.length < PAGE) break;
    }
    return all;
  }

  const DEFAULT_LEADERBOARD_SIZE = 5;

  async function fetchConfig() {
    if (!enabled) return { leaderboard_size: DEFAULT_LEADERBOARD_SIZE };
    try {
      const res = await fetch(
        `${SUPABASE_URL}/rest/v1/app_config?select=leaderboard_size&limit=1`,
        {
          headers: {
            apikey: SUPABASE_ANON_KEY,
            Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
          },
        }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const rows = await res.json();
      return rows[0] || { leaderboard_size: DEFAULT_LEADERBOARD_SIZE };
    } catch {
      // tabel lipsă sau offline — clasamentul tot funcționează, cu valoarea implicită
      return { leaderboard_size: DEFAULT_LEADERBOARD_SIZE };
    }
  }

  window.addEventListener("online", flush);

  return { enabled, log, flush, fetchAll, fetchUserEvents, fetchConfig };
})();
