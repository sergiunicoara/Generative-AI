"""Context composition metrics — where the prompt's tokens actually go.

The platform already records what a request *cost*
(`graphrag/observability/cost_attribution.py`) and how many tokens the
provider reported (`graphrag/observability/genai_telemetry.py`). Neither can
answer the question that decides whether a context is too big: which *part* of
the assembled context consumed the tokens.

`ContextBuilder.build()` assembles up to seven kinds of section — retrieved
chunks, linked chunks, document links, entity context, graph relationships,
unresolved conflicts, and a community-synthesized answer. Six are bounded by a
count (top_k, `[:5]`, `[:10]`), which is not a bound on tokens: five long
entity descriptions cost more than ten short edges. The community answer is
bounded by nothing at all. Emitting per-section token counts turns "the
context is too large" from an opinion into a number attributable to a specific
section.

Section labels are a small closed set defined by the builder, so they are safe
as a Prometheus label — unlike tenant, which `cost_attribution` deliberately
keeps out of labels for cardinality reasons.
"""

from __future__ import annotations

import structlog

from graphrag.observability.correlation import current_correlation_id

log = structlog.get_logger(__name__)

try:
    from prometheus_client import Histogram
except ImportError:  # pragma: no cover - optional local dependency
    Histogram = None


# Buckets span a single short edge line (~16 tokens) up to a context far larger
# than any current model window, so an outlier is visible rather than clipped
# into the top bucket alongside merely-large contexts.
_TOKEN_BUCKETS = (16, 64, 256, 1_024, 4_096, 16_384, 65_536)

_section_tokens = Histogram(
    "graphrag_context_section_tokens",
    "Estimated tokens contributed to the assembled context by one section kind",
    ["section"],
    buckets=_TOKEN_BUCKETS,
) if Histogram else None

_total_tokens = Histogram(
    "graphrag_context_total_tokens",
    "Estimated total tokens in the assembled context",
    buckets=_TOKEN_BUCKETS,
) if Histogram else None


def record_context_composition(composition: dict[str, int]) -> None:
    """Publish per-section and total context token counts.

    `composition` maps section label -> estimated tokens. Never raises: this is
    measurement, and a metrics backend problem must not fail a query that was
    otherwise answerable.
    """
    if not composition:
        return
    total = sum(composition.values())
    try:
        if _section_tokens:
            for section, tokens in composition.items():
                _section_tokens.labels(section).observe(tokens)
        if _total_tokens:
            _total_tokens.observe(total)
    except Exception as exc:  # noqa: BLE001 - metrics must never break retrieval
        log.warning("observability.context_composition_failed", error=str(exc)[:200])
    log.info(
        "observability.context_composition",
        total_tokens=total,
        sections=composition,
        correlation_id=current_correlation_id(),
    )


__all__ = ["record_context_composition"]
