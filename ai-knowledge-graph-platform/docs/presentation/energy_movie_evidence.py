"""Small, fail-closed presentation contract for the recorded Energy workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path

DISCLAIMER = "Synthetic data · Advisory POC · No equipment control"
TRACE_PATH = Path(__file__).with_name("energy_demo_real_run.json")


def mapping_counts(trace_path: Path = TRACE_PATH) -> tuple[int, int]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    command = trace["commands"]["validate_r2rml"]
    if not trace["all_succeeded"] or command["returncode"] != 0:
        raise ValueError("Cannot present a failed mapping capture as successful")
    counts = []
    for field in ("entity_rows", "relation_rows"):
        match = re.search(rf"\b{field}=(\d+)\b", command["stdout"])
        if match is None:
            raise ValueError(f"Captured mapping output is missing {field}")
        counts.append(int(match.group(1)))
    return counts[0], counts[1]


def trace_excerpt(image):
    """Crop the technical details from the recorded 1440-wide UI, without rewriting it."""
    scale = image.width / 1440
    box = tuple(round(value * scale) for value in (555, 735, 1320, 1375))
    if image.height < box[3]:
        raise ValueError("Technical UI capture is too short for the documented excerpt")
    return image.crop(box)


def why_trace_excerpt(image):
    """Show the expanded ``Why am I seeing this?`` panel and the trace it reveals."""
    scale = image.width / 1440
    box = tuple(round(value * scale) for value in (535, 675, 1320, 1185))
    if image.height < box[3]:
        raise ValueError("Technical UI capture is too short for the Why-am-I-seeing-this excerpt")
    return image.crop(box)
