"""Versioned per-model pricing for real cost computation.

Every LLM call already reports real (provider, model, input_tokens,
output_tokens) through `genai_telemetry.llm_call_span` -> `_finish()` -- but
`cost_usd` was hardcoded to 0.0 at every call site that emits a `CostEvent`
(`graphrag/observability/cost_attribution.py`), even for genuinely paid
calls. This table turns those real token counts into a real dollar figure.

Prices are estimates, not live provider quotes -- providers can change
pricing at any time. Each entry below is dated to when it was last verified
against the provider's own pricing page (or a cross-checked third-party
aggregator where the provider's page could not be reached), so a stale entry
is visible rather than silently wrong. `estimated_cost_usd()` returns `None`
for a (provider, model) pair not in this table rather than guessing --
consistent with this package's existing policy (see `genai_telemetry.py`'s
module docstring) that a fabricated number is worse than a missing one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float
    verified_on: str
    source: str


# Keyed by (provider, model) exactly as `response.model` / this codebase's
# internal provider name report them. Provider names match
# `genai_telemetry.system_for()`'s vocabulary.
_PRICE_TABLE: dict[tuple[str, str], ModelPrice] = {
    ("groq", "openai/gpt-oss-120b"): ModelPrice(
        input_per_1m=0.15, output_per_1m=0.60,
        verified_on="2026-09-19",
        source="third-party aggregator cross-check (aipricing.guru, markaicode.com) "
               "-- groq.com/pricing did not render at audit time",
    ),
    ("deepseek", "deepseek-flash"): ModelPrice(
        input_per_1m=0.15, output_per_1m=0.60,
        verified_on="2026-09-19",
        source="api-docs.deepseek.com/quick_start/pricing (off-peak cache-miss input, "
               "output); peak is 2x -- see PEAK_MULTIPLIER below",
    ),
    ("deepseek", "deepseek-v4-flash"): ModelPrice(
        # Legacy name -- DeepSeek retired this id and now serves it as
        # deepseek-flash, billed at the flash price (confirmed live 2026-09-19).
        input_per_1m=0.15, output_per_1m=0.60,
        verified_on="2026-09-19",
        source="api-docs.deepseek.com/quick_start/pricing -- legacy alias of deepseek-flash",
    ),
    ("cerebras", "gpt-oss-120b"): ModelPrice(
        input_per_1m=0.35, output_per_1m=0.75,
        verified_on="2026-09-19",
        source="cerebras.ai/pricing (Developer tier)",
    ),
}

# DeepSeek peak hours (01:00-04:00 and 06:00-10:00 UTC, Mon-Fri) bill at 2x
# off-peak. Cost is computed off-peak by default since the caller doesn't
# have access to the request timestamp at this layer -- this makes an
# estimate, not a guarantee; see the module docstring.
PEAK_MULTIPLIER = 2.0

# Providers whose free-tier/no-cost usage should report a real, confirmed
# $0.0 rather than "unknown" -- distinct from a model simply missing from
# the table above.
_FREE_PROVIDERS = {"openrouter"}


def estimated_cost_usd(
    provider: str, model: str | None, input_tokens: int | None, output_tokens: int | None,
) -> float | None:
    """Return an estimated dollar cost, or None if this (provider, model)
    pair has no known price or token counts are missing.

    None must be treated as "unknown", never coerced to 0.0 by the caller --
    see `cost_attribution.py` / `genai_telemetry.py` for why a fabricated
    zero is worse than an absent value.
    """
    if input_tokens is None or output_tokens is None:
        return None
    provider = (provider or "").lower()
    if provider in _FREE_PROVIDERS:
        return 0.0
    price = _PRICE_TABLE.get((provider, model or ""))
    if price is None:
        return None
    return (
        (max(0, input_tokens) / 1_000_000) * price.input_per_1m
        + (max(0, output_tokens) / 1_000_000) * price.output_per_1m
    )
