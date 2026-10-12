"""Check that every configured model identifier is covered by config/licenses.yml (offline, no network).

This is a *register* check: it proves each model has a recorded licence that is on the allow-list, and surfaces
entries nobody has reviewed yet. It does not interpret licence text or replace legal review.
"""
from __future__ import annotations

import re
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "config" / "licenses.yml"
SCAN_DIRS = ("config", "graphrag")
# Identifier families this project can call. A new family needs a regex here as well as a register row.
_MODEL_ID = re.compile(
    r"(?<![\w./-])(gpt-oss-[\w.]+|deepseek-[a-z][\w.-]*|text-embedding-3-[a-z]+|ms-marco-MiniLM-[\w-]+|claude-[a-z][\w.-]*)")
_SUFFIXES = {".py", ".yml", ".yaml"}


def load_register(path: Path = REGISTER) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def find_model_ids(root: Path = ROOT, scan_dirs: tuple[str, ...] = SCAN_DIRS) -> dict[str, list[str]]:
    """Model identifiers mentioned in source/config, each with the files naming it."""
    found: dict[str, set[str]] = {}
    for sub in scan_dirs:
        for path in sorted((root / sub).rglob("*")):
            if path.suffix not in _SUFFIXES or "__pycache__" in path.parts or path.name == "licenses.yml":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for match in _MODEL_ID.findall(text):
                found.setdefault(match.rstrip(".-"), set()).add(path.relative_to(root).as_posix())
    return {k: sorted(v) for k, v in sorted(found.items())}


def check(register: dict[str, Any], model_ids: dict[str, list[str]]) -> dict[str, Any]:
    allowed = set(register.get("allowed_licences") or [])
    entries = register.get("models") or []
    unregistered, disallowed, needs_review, used = [], [], [], set()
    for model, files in model_ids.items():
        entry = next((e for e in entries if fnmatchcase(model, e["pattern"])), None)
        if entry is None:
            unregistered.append({"model": model, "files": files})
            continue
        used.add(entry["pattern"])
        if entry.get("licence") not in allowed:
            disallowed.append({"model": model, "licence": entry.get("licence")})
        if entry.get("review_status") != "reviewed":
            needs_review.append({"model": model, "pattern": entry["pattern"]})
    unused = [e["pattern"] for e in entries if e["pattern"] not in used]
    return {"ok": not unregistered and not disallowed, "unregistered": unregistered, "disallowed": disallowed,
            "needs_review": needs_review, "unused_entries": unused}


__all__ = ["check", "find_model_ids", "load_register"]
