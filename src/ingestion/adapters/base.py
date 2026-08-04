"""Shared adapter types. Every source-shaped adapter (Salesforce, Showpad, ...)
parses one raw external record into a (domain entity, external_id, object_type,
content_hash) tuple — content_hash feeds src/ingestion/reconciliation.py's
identical/changed detection directly off the raw record, before any of this
repo's own field mapping/normalization is applied.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


def compute_content_hash(raw: dict) -> str:
    """Canonical (sorted-keys, no-whitespace) JSON hash of a raw source record —
    detects any field-level change, not just the fields this repo's adapters
    currently map to domain fields."""
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParsedRecord:
    entity: BaseModel
    external_id: str
    object_type: str
    content_hash: str
    extra: dict[str, Any] | None = None  # e.g. {"is_deleted": True} — adapter-specific signals
