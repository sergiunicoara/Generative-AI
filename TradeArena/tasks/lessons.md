# Lessons Learned

## Session: Strategy F6+L8 Implementation (2026-05-16)

### L1 — Be explicit when listing counts upfront
**Mistake:** Said "let me make all three updates" without immediately naming the three things.
**User confusion:** "why 3?" — had to ask what the 3 referred to.
**Rule:** When saying "I will update N things", list them inline in the same sentence.
> Good: "I will update three files — tools.py (threshold), templates.py (prompt), signal_notify.py (notifier)"
> Bad: "I will make three updates" → then explain them separately.

### L2 — Keep tasks/todo.md and tasks/lessons.md current from session start
**Mistake:** CLAUDE.md requires plan-first workflow with todo.md and lessons.md, but these files were never created.
**Rule:** At the start of any session with non-trivial work, check if tasks/ files exist. Create them if missing before starting implementation.

### L3 — Confirm scope before multi-file changes
**Mistake:** Made changes across 3 files (tools.py, templates.py, signal_notify.py) in one message without a plan checkpoint.
**Rule:** For changes spanning 3+ files, write a brief plan first (even inline), get confirmation, then execute.

