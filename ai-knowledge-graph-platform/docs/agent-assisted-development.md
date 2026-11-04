# Agent-assisted development

This repository is written mainly through a coding agent (Claude Code), not
as an experiment beside hand-written code. This page states how that works in
practice: what the agent is given, which guards check its output, and the
failure modes those guards exist for, each tied to a recorded case. The
corrections themselves are logged in `tasks/lessons.md` (about 200 entries,
`L01`–`L16` for early lessons, `A01`–`A182` afterwards); the entries cited
below can be read there in full.

## What the agent is given

* **Standing instructions.** Workspace-level instructions (kept outside this
  repository) set the working rules: plan before multi-step changes, verify
  before calling anything done, find root causes instead of patching
  symptoms, delegate broad code search to a read-only subagent, run targeted
  tests while iterating and the full suite once at the end, and never start
  anything that costs money (LLM calls, evaluations, cloud) unless asked.
* **A lesson log.** `tasks/lessons.md` records each correction as what
  happened, the root cause and the rule that now prevents it. It is reviewed
  at the start of related work, so a mistake is meant to happen once.
* **Project tooling in `.claude/`.**
  * `hooks/run_tests_on_edit.py`: after any edit to `graphrag/**.py`, runs the
    matching `tests/unit/test_<module>.py` and shows the result, so a
    regression appears at the edit that caused it.
  * `agents/eval-reporter.md`: a read-only subagent (Read and Grep only, small
    model) that answers one question, what happened to a golden-eval case, from
    the recorded files; it cannot edit or re-run anything.
  * `skills/golden-eval` and `commands/golden-eval.md`: the evaluation runbook,
    which refuses to run against a half-started stack and must report a drop
    against the baseline explicitly.
* **Graph-assisted navigation.** A code knowledge graph of the repository
  (`graphify-out/`) answers "how does X connect to Y" before files are read
  cold, which keeps exploration out of the main context.

## Guards on agent output

| Guard | What it catches |
| --- | --- |
| Per-edit targeted tests (hook) | A regression in the module just edited |
| One full unit suite before every push | Interactions the targeted tests miss, stale assertions elsewhere |
| CI e2e tier on a real Neo4j (Testcontainers), failing when tests are skipped | Cypher that only works against mocks; a "green" run that ran nothing |
| Manual mutation-testing workflow (`mutmut`) | Tests that execute code without checking it |
| Evidence levels on every roadmap item | Claims stronger than the evidence (unit-tested, wired, live-validated) |
| Human review of every diff before commit | Everything else, including the cases below |

## Failure modes, each with a recorded case

**Describing a capability without wiring it.** A first pass added an ontology
YAML, an RDF export script and a rule method as evidence for a role; an audit
found the YAML was never loaded, the method did not exist and the RDF was built
by string concatenation (A52). Rule: a reviewer will grep for the call; every
claimed capability needs a test that executes it.

**Reporting gaps that are not gaps.** A gap analysis listed seven missing
items; re-exploration found five already built in earlier sessions (A56).
Rule: reconcile any gap list against the committed code before acting on it.
The same check is applied to job-description reviews in `docs/roadmap.md`.

**Confident names for code that does not exist.** A search subagent reported
class names (`MultimodalStore`, `SharePointConnector`) that are not in the code;
the real ones are `MultiModalEntityService` and `SharePointSyncConnector`. The
failing import was caught before commit. Rule: names returned by a subagent are
leads, verified by reading the file before they are used.

**Fixing against a false premise.** Two evaluation cases were scheduled for an
extraction fix on the grounds that the corpus contained a contradiction;
reading the source documents end to end showed it did not, so the cases were
wrong, not the extractor (A133). Rule: read the evidence itself, not a grep for
the expected token.

**Unit tests that cannot see the defect.** A new method silently replaced an
existing one with the same name (Python has no overloading) and broke every
retrieval query; unit tests mocked the client and passed, live verification did
not (A141). A route test overrode authentication and passed while every real
browser write failed on a missing CSRF cookie (A173). Rule: name what a test
bypasses, and verify the path it skips.

**Tests that look stronger than they are.** Round-trip tests compared triple
counts, which a corrupting serializer can preserve (A164); mutation testing
raised killed mutants from 49.5% to 91.1% and found a real serialisation bug
(A181). Rule: assert content, not counts, and measure test strength.

**Overstated status.** A passing mock test does not prove a live Neo4j or a
deployment path; the roadmap records the strongest evidence actually reached
for each item, and documents say plainly what was not run (A154).

**Adding untrusted code because it fits.** A new SHACL-to-Cypher package
looked like an exact match; it was evaluated from its published metadata and
not installed (single release, single maintainer, native code), and the needed
subset was written and tested here instead (see
`docs/interoperability-and-curation.md`).

## Working pattern

1. Classify the task and its risk before starting; plan multi-step changes.
2. Delegate broad searches to a read-only subagent; read only the narrow ranges
   that matter.
3. Write the change with its tests; let the hook run the targeted tests.
4. Review the diff, checking claims against code: names, calls, tenant scoping,
   what each test mocks.
5. Run the full unit suite once; commit only when it is green.
6. Record the evidence level reached, and log any correction in
   `tasks/lessons.md` with the rule that prevents it next time.

## Limits

The guards reduce, not remove, agent error: the per-edit hook only runs a test
file whose name matches the module, mutation testing is manual and covers
selected modules, and the e2e tier does not run in this local environment
(no Docker), so its results come from CI only. Review remains the final check.
