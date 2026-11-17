"""Self-describing metadata and per-question helpers for evaluation result files.

A result file that cannot say which commit, models and golden set produced it
is not comparable to another one. ``collect_run_metadata`` records all of that;
the other helpers extract the per-question signals (agentic fallback, retrieval
latency) that ``QueryResult`` exposes only indirectly.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import date, datetime, time, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "ablation-run/1"

BUCHAREST = ZoneInfo("Europe/Bucharest")  # EET (UTC+2) in winter, EEST (UTC+3) in summer
# DeepSeek bills 2x during these UTC windows, Monday to Friday (same rule as
# graphrag/observability/pricing.py: PEAK_MULTIPLIER). Hours are [start, end).
DEEPSEEK_PEAK_UTC = ((1, 4), (6, 10))
DEEPSEEK_PEAK_MULTIPLIER = 2.0


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_commit() -> dict[str, Any]:
    """Commit, branch and whether tracked files differ from the commit."""
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "sha": sha,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def ragas_version() -> str | None:
    try:
        return importlib_metadata.version("ragas")
    except importlib_metadata.PackageNotFoundError:
        return None


def golden_set_info(path: Path, payload: dict, n_questions: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "version": payload.get("version"),
        "tenant": payload.get("tenant"),
        "n_questions": n_questions,
        "n_questions_in_file": len(payload.get("questions", [])),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _utc(now: datetime | None) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc)


def deepseek_peak(now: datetime | None = None) -> bool:
    """True when DeepSeek bills at the peak (2x) rate at ``now`` (default: current time)."""
    moment = _utc(now)
    return moment.weekday() < 5 and any(start <= moment.hour < end for start, end in DEEPSEEK_PEAK_UTC)


def deepseek_peak_windows_bucharest(day: date | datetime | None = None) -> list[str]:
    """The peak windows of ``day`` as ``HH:MM-HH:MM`` in Bucharest time (DST-aware)."""
    on = day.date() if isinstance(day, datetime) else (day or _utc(None).date())
    return [
        f"{datetime.combine(on, time(start), tzinfo=timezone.utc).astimezone(BUCHAREST):%H:%M}-"
        f"{datetime.combine(on, time(end), tzinfo=timezone.utc).astimezone(BUCHAREST):%H:%M}"
        for start, end in DEEPSEEK_PEAK_UTC
    ]


def deepseek_pricing_window(now: datetime | None = None) -> dict[str, Any]:
    moment = _utc(now)
    return {
        "peak_now": deepseek_peak(moment),
        "now_bucharest": moment.astimezone(BUCHAREST).strftime("%Y-%m-%d %H:%M %Z"),
        "peak_windows_bucharest": deepseek_peak_windows_bucharest(moment),
        "peak_days": "Mon-Fri",
        "peak_multiplier": DEEPSEEK_PEAK_MULTIPLIER,
    }


def collect_run_metadata(
    *, tenant: str, profile: str, profile_overrides: dict, mode: str,
    config_overrides: dict, models: dict, golden: dict, pinned: bool,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": now.isoformat(timespec="seconds"),
        "timestamp_bucharest": now.astimezone(BUCHAREST).isoformat(timespec="seconds"),
        "deepseek_peak_at_start": deepseek_peak(now),
        "git": git_commit(),
        "tenant": tenant,
        "retrieval_profile": profile,
        "profile_overrides": profile_overrides,
        "config_overrides": config_overrides,
        "mode": mode,
        "pinned_models": pinned,
        "models": models,
        "temperature": 0.0,
        "ragas_version": ragas_version(),
        "golden_set": golden,
        "python": sys.version.split()[0],
        "cache": "disabled (no query ID)",
    }


def agentic_fallback(result: Any) -> dict[str, Any]:
    """Whether the agentic (IRCoT) fallback fired for this answer, and why.

    ``QueryResult`` has no boolean for it; the signal is ``explanation["fallback"]``
    (set on both paths) with the trajectory as a cross-check.
    """
    explanation = getattr(result, "explanation", None) or {}
    fallback = explanation.get("fallback") if isinstance(explanation, dict) else None
    trajectory = getattr(result, "retrieval_trajectory", None)
    completed_by = getattr(trajectory, "completed_by", None)
    planned_mode = getattr(trajectory, "planned_mode", None)
    if isinstance(fallback, dict) and "triggered" in fallback:
        fired = bool(fallback["triggered"])
        reason = fallback.get("reason")
    else:  # no explanation: fall back to the trajectory
        fired = planned_mode == "agentic" or completed_by in ("reasoning_answer", "final_synthesis")
        reason = None
    return {"fired": fired, "reason": reason, "completed_by": completed_by}


def retrieval_latency_ms(result: Any) -> float | None:
    """Search-stage time from the first trajectory step (None if not captured).

    Excludes query rewriting, context building and generation. Under the agentic
    fallback this is still the primary retrieval step, not the fallback's.
    """
    trajectory = getattr(result, "retrieval_trajectory", None)
    steps = getattr(trajectory, "steps", None) or []
    value = getattr(steps[0], "latency_ms", None) if steps else None
    return round(float(value), 2) if value is not None else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return round(ordered[index], 2)
