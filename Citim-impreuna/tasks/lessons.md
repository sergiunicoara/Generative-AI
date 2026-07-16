# Lessons — Citim împreună

## L01 — PWA service workers need their own file to change, or updates never reach installed clients (2026-07-04)

Deployed two rounds of content/logic changes to `sergiunicoara.github.io/citim-impreuna` without touching `sw.js`. Browsers only re-check a service worker when the service worker *script's own bytes* change — untouched `sw.js` means the browser never re-installs it, so a cache-first `fetch` handler keeps serving whatever was cached on first visit forever, no matter how many times the underlying app files are pushed. User reported "still the same version on my phone" after two live deploys.

I had actually already written this exact warning into `tasks/todo.md` after the first cache issue (during local dev) — but the note only said "remember to bump `CACHE` in `sw.js`," and I didn't apply it when redeploying to the live site twice.

**Fix applied:** switched `sw.js`'s fetch handler from cache-first to network-first (try network, fall back to cache only when offline) — this makes the staleness bug structurally impossible going forward, instead of relying on remembering to bump a version string every deploy.

**Rule:** Any time an app has a service worker with a cache-first strategy, deploying new app content is *not* enough — either bump the cache-version string in the SW file itself with every deploy, or (preferred) use network-first/stale-while-revalidate so it self-heals. Prefer the structural fix over a "remember to do X" note — notes get missed under time pressure; a self-healing design doesn't.

**How to apply:** Before considering any static-asset deploy "done," check whether the project has a service worker and whether its fetch strategy could mask the very update just shipped.

## L02 — `display` set on the same selector as `[hidden]` silently breaks the `hidden` attribute (2026-07-04)

Wrote `.modal-backdrop { display: flex; ... }` for the name-entry modal, toggled via `element.hidden = true/false` in JS. This never actually hid the modal: CSS cascade origin ordering means *any* author-stylesheet rule beats the user-agent stylesheet's default `[hidden] { display: none }`, regardless of selector specificity. So the modal (full-viewport, `position: fixed`, `z-index: 20`) stayed visually on top and kept intercepting every tap, even though the JS `hidden` property correctly read `true`. Symptom from the user: "it got stuck on the first page after answering the name" — the modal was invisible-in-my-testing-assumption but very much still there and blocking.

Caught only by checking `getComputedStyle(el).display` directly — checking `el.hidden` (the JS/IDL property) or the accessibility snapshot was not enough, both looked "correct" while the element was still fully rendered and click-blocking.

**Rule:** Any time a component's CSS sets `display` unconditionally on an element that JS also toggles via the `hidden` attribute, add an explicit `.the-class[hidden] { display: none; }` override. Don't rely on the bare `hidden` attribute once the class already declares `display`.

**How to apply:** When building any show/hide overlay (modal, drawer, toast) styled with `display: flex/grid/block` in its base class, immediately add the `[hidden]` override in the same edit — don't wait to discover it by symptom. When diagnosing a "toggle doesn't seem to do anything" bug, check `getComputedStyle` first, not just the JS property.

## L03 — Enabling Supabase RLS with no policies silently blocks the anon key: reads return `[]`, not an error (2026-07-16)

Hit twice now. First on `events` (RLS auto-enabled when Supabase Auth was added — the leaderboard went empty with no error anywhere). Then again on `scores`: the table was created and RLS enabled, but without policies the anon key got `200 []` on SELECT and `401 / 42501 "new row violates row-level security policy"` on INSERT.

The nasty part is the asymmetry: **writes fail loudly, reads fail silently.** A blocked SELECT is not an error — RLS filters every row, so the client receives a perfectly valid empty array and cannot distinguish "table is empty" from "I am not allowed to see anything." Any `catch`-based error handling sails right past it.

Also note: `Prefer: resolution=merge-duplicates` (upsert) needs **both** an `insert` and an `update` policy — an insert policy alone makes the second write of the same key fail.

**Rule:** RLS is deny-by-default. Enabling it without policies does not "secure" a table — it disconnects it. After enabling RLS on any table this app reads with the anon key, immediately probe both verbs (`SELECT` and an upsert) with the anon key before assuming it works. Never infer "the table is empty" from `[]` on an RLS-enabled table.

**How to apply:** When adding any new Supabase table the client touches, ship the policies in the same SQL as the `create table`, then verify with a live read+write probe. Design the client to degrade safely (see `renderStats` — it falls back to the full-scan path when `scores` returns empty), so an RLS misconfiguration costs performance, never correctness.

**Related caveat:** `tracker.js` always authenticates with the *public* anon key, never the logged-in user's session JWT — so every request hits Postgres as the `anon` role. That forces `using (true)` policies, which for read/write are effectively equivalent to RLS-off (they do still block `DELETE`). Real per-user enforcement would require sending the user's access token as the Bearer and matching on the JWT claim.
