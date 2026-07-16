# Graph Report - .  (2026-07-16)

## Corpus Check
- 22 files · ~137,600 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 124 nodes · 178 edges · 11 communities (10 shown, 1 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 11 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Game Logic & Scoring|Game Logic & Scoring]]
- [[_COMMUNITY_Scene Animations|Scene Animations]]
- [[_COMMUNITY_UI Shell & Auth UI|UI Shell & Auth UI]]
- [[_COMMUNITY_Project Config & Lessons|Project Config & Lessons]]
- [[_COMMUNITY_Auth Flow|Auth Flow]]
- [[_COMMUNITY_Weekly Window Builder|Weekly Window Builder]]
- [[_COMMUNITY_Dev Server|Dev Server]]
- [[_COMMUNITY_1 Samuel Verse Data|1 Samuel Verse Data]]
- [[_COMMUNITY_2 Samuel Verse Data|2 Samuel Verse Data]]
- [[_COMMUNITY_Tracker & Supabase|Tracker & Supabase]]
- [[_COMMUNITY_Service Worker & Cache|Service Worker & Cache]]

## God Nodes (most connected - your core abstractions)
1. `renderPage()` - 9 edges
2. `syncProgressFromCloud()` - 9 edges
3. `checkAnswers()` - 6 edges
4. `handleLogin()` - 6 edges
5. `Auth` - 6 edges
6. `save()` - 5 edges
7. `showFinal()` - 5 edges
8. `handleLogout()` - 5 edges
9. `VERSES` - 5 edges
10. `saveSolvedThisCycle()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `renderPage()` --references--> `VERSES`  [EXTRACTED]
  js/app.js → js/verses.js
- `updateSceneBackground()` --references--> `SceneEngine`  [EXTRACTED]
  js/app.js → js/scenes.js
- `checkAnswers()` --references--> `Tracker`  [EXTRACTED]
  js/app.js → js/tracker.js
- `updateUserChip()` --references--> `Auth`  [EXTRACTED]
  js/app.js → js/auth.js
- `showAuthModal()` --references--> `Auth`  [EXTRACTED]
  js/app.js → js/auth.js

## Import Cycles
- None detected.

## Communities (11 total, 1 thin omitted)

### Community 0 - "Game Logic & Scoring"
Cohesion: 0.14
Nodes (28): buildSolvedVerseCard(), buildVerseCard(), celebrate(), checkAnswers(), CHEERS, computePointsForUser(), computeSolvedPagesForCycle(), el (+20 more)

### Community 1 - "Scene Animations"
Cohesion: 0.08
Nodes (5): updateSceneBackground(), NEUTRAL_ROTATION, pickScene(), SceneEngine, SCENES

### Community 2 - "UI Shell & Auth UI"
Cohesion: 0.13
Nodes (13): scripts/build-window.js (Window Generator), css/style.css (Styles), icons/icon.svg (App Icon), SVG Purple Rounded Rectangle Background, SVG Open Book Shape (white stroke), SVG Text Lines (gold stroke), index.html (App Shell), VERSES_1SAMUEL (+5 more)

### Community 3 - "Project Config & Lessons"
Cohesion: 0.13
Nodes (15): CLAUDE.md (Project Config), Cornilescu Bible 2023 Revised Edition, graphify Knowledge Graph (graphify-out/), L02: CSS display overrides hidden attribute, L01: Service Worker Cache-First Bug, manifest.webmanifest (PWA Manifest), Minimal Impact Principle, Simplicity First Principle (+7 more)

### Community 4 - "Auth Flow"
Cohesion: 0.52
Nodes (7): handleLogin(), handleLogout(), hideAuthModal(), setAuthError(), showAuthModal(), updateUserChip(), Auth

### Community 5 - "Weekly Window Builder"
Cohesion: 0.43
Nodes (6): fs, loadMaster(), main(), path, serializeVerse(), writeFile()

### Community 6 - "Dev Server"
Cohesion: 0.33
Nodes (3): NoCacheHandler, Static file server for local dev with caching fully disabled.  Plain `python -m, SimpleHTTPRequestHandler

### Community 7 - "1 Samuel Verse Data"
Cohesion: 0.50
Nodes (3): VERSES_1SAMUEL_MASTER, errors, { VERSES_1SAMUEL_MASTER }

### Community 8 - "2 Samuel Verse Data"
Cohesion: 0.50
Nodes (3): VERSES_2SAMUEL_MASTER, errors, { VERSES_2SAMUEL_MASTER }

### Community 9 - "Tracker & Supabase"
Cohesion: 0.50
Nodes (3): localStorage (ci_score, ci_progress, ci_user), Supabase app_config table, Supabase events table

## Knowledge Gaps
- **14 isolated node(s):** `CHEERS`, `totalPages`, `el`, `solvedThisCycle`, `NEUTRAL_ROTATION` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `updateSceneBackground()` connect `Scene Animations` to `Game Logic & Scoring`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **Why does `renderPage()` connect `Game Logic & Scoring` to `Scene Animations`?**
  _High betweenness centrality (0.004) - this node is a cross-community bridge._
- **What connects `CHEERS`, `totalPages`, `el` to the rest of the system?**
  _15 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Game Logic & Scoring` be split into smaller, more focused modules?**
  _Cohesion score 0.1425287356321839 - nodes in this community are weakly interconnected._
- **Should `Scene Animations` be split into smaller, more focused modules?**
  _Cohesion score 0.08333333333333333 - nodes in this community are weakly interconnected._
- **Should `UI Shell & Auth UI` be split into smaller, more focused modules?**
  _Cohesion score 0.13071895424836602 - nodes in this community are weakly interconnected._
- **Should `Project Config & Lessons` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._