"""Shared Energy vocabulary: RDF namespaces and the dataset's tenant identity.

Extracted from ``graphrag/domains/energy/demo.py`` so the answer-derivation
layer (``graphrag/domains/energy/answers.py``) can use the same namespaces
without importing the service that imports it. ``demo.py`` re-exports every
name here, so existing importers (``workflow.py``, ``lpg_projection.py``, and
the tests that do ``from graphrag.domains.energy.demo import ENERGY, REC,
TENANT``) keep working unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rdflib import Namespace

ENERGY = Namespace("https://example.energy.demo/ontology#")
ASSET = Namespace("https://example.energy.demo/asset/")
DOC = Namespace("https://example.energy.demo/document/")
REC = Namespace("https://example.energy.demo/record/")
PROV = Namespace("http://www.w3.org/ns/prov#")

# The published Energy dataset's own tenant. `answer()` compares the caller's
# requested tenant against the dataset's identity rather than against a
# literal written inline at the comparison site.
TENANT = "energy-demo"

UTC = timezone.utc


def parse_instant(value: str) -> datetime:
    """Parse an ISO-8601 instant (accepting a trailing ``Z``) as UTC."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def format_instant(value: datetime) -> str:
    """Render a UTC datetime in the ``...Z`` form the demo's data uses."""
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "ASSET", "DOC", "ENERGY", "PROV", "REC", "TENANT", "UTC",
    "format_instant", "parse_instant",
]
