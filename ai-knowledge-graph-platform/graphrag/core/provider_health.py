"""In-memory circuit-breaker state for LLM providers.

Problem solved
--------------
DeepSeek deprecated the "deepseek-chat" model id mid-session (2026-07-24) and
every call to it started failing with a 400. Because the primary generation
LLM (`get_llm()`) had no fallback and no health tracking, this broke answer
synthesis for ~40+ minutes, undetected, with each broken call still paying the
full 3x10s retry cost. This module gives `DeepSeekLLM`/`GroqLLM` a way to
notice "the last several calls to this provider all failed" and skip most of
that wasted retry time — see `FallbackLLM` in llm_client.py for what happens
once a provider is marked unhealthy.

Design
------
Pure in-memory, dependency-free, no locking: all mutation happens
synchronously after an `await` resolves (single-threaded asyncio event loop),
never concurrently. Process restarts clear state naturally.

Two trip conditions, either one marks a provider unhealthy:
- 3 consecutive failures — the incident's failure mode was deterministic (a
  bad model id fails identically on every call), so a short consecutive
  streak is enough signal without needing a large sample.
- 80%+ failure rate over the last 20 recorded results (min 3 samples before
  judging) — catches a provider that's mostly-but-not-cleanly broken (e.g.
  flaky infra alternating pass/fail) without a clean streak forming.

Recording happens per retry *attempt*, not per logical call — see callers in
llm_client.py — so a single request against a fully-broken provider can trip
the breaker by itself.
"""

from __future__ import annotations

from collections import deque

_WINDOW_SIZE = 20
_MIN_SAMPLES = 3
_FAILURE_RATE_THRESHOLD = 0.8
_CONSECUTIVE_FAILURE_THRESHOLD = 3


class _ProviderState:
    __slots__ = ("results", "consecutive_failures")

    def __init__(self) -> None:
        self.results: deque[bool] = deque(maxlen=_WINDOW_SIZE)
        self.consecutive_failures: int = 0


_states: dict[str, _ProviderState] = {}


def _state(provider: str) -> _ProviderState:
    if provider not in _states:
        _states[provider] = _ProviderState()
    return _states[provider]


def record_result(provider: str, success: bool) -> None:
    """Record the outcome of one call attempt against `provider`."""
    s = _state(provider)
    s.results.append(success)
    s.consecutive_failures = 0 if success else s.consecutive_failures + 1


def is_healthy(provider: str) -> bool:
    """True if `provider` looks usable based on recent call history.

    No history yet (or fewer than _MIN_SAMPLES results) -> assume healthy;
    there's no evidence of trouble yet and treating an unseen provider as
    unhealthy would make every process's first call to it fail-fast for no
    reason.
    """
    s = _states.get(provider)
    if s is None or len(s.results) < _MIN_SAMPLES:
        return True
    if s.consecutive_failures >= _CONSECUTIVE_FAILURE_THRESHOLD:
        return False
    failure_rate = 1 - (sum(s.results) / len(s.results))
    return failure_rate < _FAILURE_RATE_THRESHOLD


def reset(provider: str | None = None) -> None:
    """Test-only helper: clear tracked state for one provider, or all."""
    if provider is None:
        _states.clear()
    else:
        _states.pop(provider, None)
