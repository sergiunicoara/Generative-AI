"""Subgroup breakdown of pass/fail results (offline; no LLM, no new runs).

An overall pass rate hides uneven quality: a 90% average can be 100% on one kind of question and 50% on another.
``subgroup_report`` groups per-item outcomes by a key, gives each group a rate with a Wilson 95% interval (groups are
small, so a bare percentage overstates certainty) and flags a group only when it is both big enough to judge and
clearly below the overall rate. A flag is a prompt to look, not a verdict.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

_Z = 1.96


def wilson_interval(passed: int, total: int, z: float = _Z) -> tuple[float, float] | None:
    if total <= 0:
        return None
    p = passed / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def subgroup_report(
    items: Iterable[dict[str, Any]], *, group_key: str, passed_key: str = "passed",
    min_n: int = 3, min_gap: float = 0.15,
) -> dict[str, Any]:
    """Group ``items`` by ``item[group_key]`` and report the rate of truthy ``item[passed_key]`` per group.

    A group is flagged ``below_overall`` when it has at least ``min_n`` items and its rate is at least ``min_gap``
    under the overall rate. ``insufficient_n`` marks groups too small to judge. Missing keys group as ``"(none)"``.
    """
    groups: dict[str, list[bool]] = defaultdict(list)
    for item in items:
        groups[str(item.get(group_key, "(none)") or "(none)")].append(bool(item.get(passed_key)))
    total = sum(len(v) for v in groups.values())
    passed = sum(sum(v) for v in groups.values())
    overall = round(passed / total, 4) if total else None
    rows = []
    for name in sorted(groups):
        flags = groups[name]
        n, ok = len(flags), sum(flags)
        rate = round(ok / n, 4)
        enough = n >= min_n
        rows.append({
            "group": name, "n": n, "passed": ok, "rate": rate, "ci95": wilson_interval(ok, n),
            "insufficient_n": not enough,
            "below_overall": bool(enough and overall is not None and round(overall - rate, 6) >= min_gap),
        })
    rates = [r["rate"] for r in rows if not r["insufficient_n"]]
    return {
        "group_key": group_key, "n": total, "overall_rate": overall, "groups": rows,
        "max_gap": round(max(rates) - min(rates), 4) if len(rates) >= 2 else None,
        "flagged": [r["group"] for r in rows if r["below_overall"]],
        "thresholds": {"min_n": min_n, "min_gap": min_gap},
    }


__all__ = ["subgroup_report", "wilson_interval"]
