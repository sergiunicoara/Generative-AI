"""Unit tests for ClaimVerifier."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from graphrag.retrieval.claim_verifier import ClaimVerifier, _FALLBACK


CONTEXT = "FAA AD 2024-01-02 supersedes FAA AD 2022-03-07 and requires software update."


@pytest.fixture
def verifier():
    return ClaimVerifier()


# ── _check_claim ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_claim_yes(verifier):
    with patch("graphrag.retrieval.claim_verifier.get_fast_llm") as mock_llm:
        mock_llm.return_value.generate = AsyncMock(return_value="YES")
        result = await verifier._check_claim("FAA issued AD 2024-01-02.", CONTEXT)
    assert result is True


@pytest.mark.asyncio
async def test_check_claim_no(verifier):
    with patch("graphrag.retrieval.claim_verifier.get_fast_llm") as mock_llm:
        mock_llm.return_value.generate = AsyncMock(return_value="NO")
        result = await verifier._check_claim("The moon is made of cheese.", CONTEXT)
    assert result is False


@pytest.mark.asyncio
async def test_check_claim_yes_with_punctuation(verifier):
    """YES. and YES! should both be accepted."""
    with patch("graphrag.retrieval.claim_verifier.get_fast_llm") as mock_llm:
        mock_llm.return_value.generate = AsyncMock(return_value="YES.")
        result = await verifier._check_claim("FAA issued AD 2024-01-02.", CONTEXT)
    assert result is True


# ── verify ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_all_claims_grounded_returns_unchanged(verifier):
    answer = "FAA AD 2024-01-02 supersedes AD 2022-03-07. A software update is required."
    with patch("graphrag.retrieval.claim_verifier.get_fast_llm") as mock_llm:
        mock_llm.return_value.generate = AsyncMock(return_value="YES")
        clean, n_removed = await verifier.verify(answer, CONTEXT)
    assert n_removed == 0
    assert "FAA AD 2024-01-02" in clean
    assert "software update" in clean


@pytest.mark.asyncio
async def test_one_ungrounded_claim_removed(verifier):
    answer = "FAA AD 2024-01-02 supersedes AD 2022-03-07. The pilot was tired."
    responses = ["YES", "NO"]
    call_count = 0

    async def side_effect(prompt):
        nonlocal call_count
        r = responses[call_count]
        call_count += 1
        return r

    with patch("graphrag.retrieval.claim_verifier.get_fast_llm") as mock_llm:
        mock_llm.return_value.generate = AsyncMock(side_effect=side_effect)
        clean, n_removed = await verifier.verify(answer, CONTEXT)

    assert n_removed == 1
    assert "FAA AD 2024-01-02" in clean
    assert "pilot was tired" not in clean


@pytest.mark.asyncio
async def test_all_ungrounded_returns_fallback(verifier):
    answer = "The sky is green. Penguins fly."
    with patch("graphrag.retrieval.claim_verifier.get_fast_llm") as mock_llm:
        mock_llm.return_value.generate = AsyncMock(return_value="NO")
        clean, n_removed = await verifier.verify(answer, CONTEXT)
    assert clean == _FALLBACK
    assert n_removed == 2


@pytest.mark.asyncio
async def test_empty_answer_passthrough(verifier):
    clean, n_removed = await verifier.verify("", CONTEXT)
    assert clean == ""
    assert n_removed == 0


@pytest.mark.asyncio
async def test_fallback_message_passthrough(verifier):
    """Fallback message should not be re-verified."""
    with patch("graphrag.retrieval.claim_verifier.get_fast_llm") as mock_llm:
        mock_llm.return_value.generate = AsyncMock(return_value="NO")
        clean, n_removed = await verifier.verify(_FALLBACK, CONTEXT)
    # generate should never be called — the fallback is passed through unchanged
    mock_llm.return_value.generate.assert_not_called()
    assert clean == _FALLBACK
    assert n_removed == 0


@pytest.mark.asyncio
async def test_context_truncation_does_not_error(verifier):
    """Long context should be silently truncated, not raise."""
    long_context = "A " * 5000  # 10 000 chars > _MAX_CONTEXT_CHARS
    with patch("graphrag.retrieval.claim_verifier.get_fast_llm") as mock_llm:
        mock_llm.return_value.generate = AsyncMock(return_value="YES")
        clean, n_removed = await verifier.verify("Some claim.", long_context)
    assert n_removed == 0
