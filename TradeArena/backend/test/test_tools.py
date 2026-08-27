"""Tests for backend.traders.tools — get_state and trade logic."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from backend.environment.accounts import INITIAL_BALANCE, Accounts
from backend.traders.tools import TraderContext, get_state_impl, trade_impl


class FakePrices:
    def __init__(self, prices: dict[str, float]):
        self._prices = {k.upper(): v for k, v in prices.items()}

    async def aget_price(self, ticker: str) -> float:
        return self._prices[ticker.upper()]

    async def aget_prices(self, tickers: list[str]) -> dict[str, float]:
        return {t: self._prices[t.upper()] for t in tickers}


@pytest.fixture
def accounts():
    a = Accounts(":memory:")
    for tid in ("claude", "gpt", "kimi"):
        a.create_trader(tid)
    yield a
    a.close()


def make_ctx(
    accounts: Accounts,
    prices: dict[str, float],
    *,
    trader_id: str = "claude",
    rivals: list[str] | None = None,
) -> TraderContext:
    return TraderContext(
        trader_id=trader_id,
        accounts=accounts,
        prices=FakePrices(prices),
        started_at=datetime.now(timezone.utc) - timedelta(seconds=120),
        duration_seconds=3600.0,
        rival_ids=rivals if rivals is not None else ["gpt", "kimi"],
    )


# ── get_state tests ──────────────────────────────────────────────────────────

async def test_initial_state_is_all_cash(accounts):
    ctx = make_ctx(accounts, {})
    state = await get_state_impl(ctx)

    assert state["trader_id"] == "claude"
    assert state["cash"] == INITIAL_BALANCE
    assert state["holdings"] == {}
    assert state["total_portfolio_value"] == INITIAL_BALANCE
    assert state["total_pnl"] == 0.0


async def test_state_timing_fields(accounts):
    ctx = make_ctx(accounts, {})
    state = await get_state_impl(ctx)

    assert 100 <= state["time_elapsed_seconds"] <= 140
    assert 3460 <= state["time_remaining_seconds"] <= 3500


async def test_state_includes_rivals_portfolio_values(accounts):
    accounts.execute_trade("gpt", "AAPL", 10, 200.0)
    ctx = make_ctx(accounts, {"AAPL": 210.0})
    state = await get_state_impl(ctx)

    assert "gpt" in state["rivals_total_portfolio_value"]
    assert "kimi" in state["rivals_total_portfolio_value"]
    expected_gpt = INITIAL_BALANCE - 10 * 200.0 + 10 * 210.0
    assert abs(state["rivals_total_portfolio_value"]["gpt"] - expected_gpt) < 0.01
    assert state["rivals_total_portfolio_value"]["kimi"] == INITIAL_BALANCE


async def test_state_holdings_include_per_position_detail(accounts):
    accounts.execute_trade("claude", "NVDA", 100, 190.0)
    ctx = make_ctx(accounts, {"NVDA": 200.0})
    state = await get_state_impl(ctx)

    pos = state["holdings"]["NVDA"]
    assert pos["quantity"] == 100.0
    assert pos["avg_cost"] == 190.0
    assert pos["current_price"] == 200.0
    assert abs(pos["unrealized_pnl"] - 100 * (200.0 - 190.0)) < 0.001
    assert "stop_price" in pos
    assert "stop_triggered" in pos
    assert pos["stop_triggered"] is False  # price 200 > stop ~180.5


async def test_state_stop_triggered_when_price_below_stop(accounts):
    accounts.execute_trade("claude", "NVDA", 100, 200.0)
    # Stop = 200 * 0.95 = 190; price 188 triggers it
    ctx = make_ctx(accounts, {"NVDA": 188.0})
    state = await get_state_impl(ctx)

    pos = state["holdings"]["NVDA"]
    assert pos["stop_triggered"] is True
    assert "NVDA" in state["stops_triggered"]


async def test_state_total_pnl(accounts):
    accounts.execute_trade("claude", "NVDA", 100, 190.0)
    accounts.execute_trade("claude", "GOOGL", 50, 160.0)
    ctx = make_ctx(accounts, {"NVDA": 200.0, "GOOGL": 155.0})
    state = await get_state_impl(ctx)

    expected_pnl = 100 * (200.0 - 190.0) + 50 * (155.0 - 160.0)
    assert abs(state["total_pnl"] - expected_pnl) < 0.01


async def test_state_has_pre_earnings_signals_key(accounts):
    ctx = make_ctx(accounts, {})
    state = await get_state_impl(ctx)
    assert "pre_earnings_signals" in state
    assert isinstance(state["pre_earnings_signals"], list)


# ── trade tests ──────────────────────────────────────────────────────────────

async def test_trade_buy_fills_at_current_price(accounts):
    ctx = make_ctx(accounts, {"NVDA": 200.0})
    result = await trade_impl(ctx, "NVDA", 10)

    assert result["success"] is True
    assert result["ticker"] == "NVDA"
    assert result["price"] == 200.0
    assert result["side"] == "buy"
    assert abs(accounts.cash("claude") - (INITIAL_BALANCE - 10 * 200.0)) < 0.001


async def test_trade_sell_reduces_position(accounts):
    accounts.execute_trade("claude", "NVDA", 100, 190.0)
    ctx = make_ctx(accounts, {"NVDA": 200.0})
    result = await trade_impl(ctx, "NVDA", -40)

    assert result["success"] is True
    assert result["side"] == "sell"
    assert accounts.holdings("claude")["NVDA"]["quantity"] == 60.0


async def test_trade_fractional(accounts):
    ctx = make_ctx(accounts, {"NVDA": 200.0})
    result = await trade_impl(ctx, "NVDA", 2.5)

    assert result["success"] is True
    assert accounts.holdings("claude")["NVDA"]["quantity"] == 2.5


async def test_trade_ticker_normalized(accounts):
    ctx = make_ctx(accounts, {"NVDA": 200.0})
    result = await trade_impl(ctx, "nvda", 5)

    assert result["ticker"] == "NVDA"
    assert "NVDA" in accounts.holdings("claude")


async def test_trade_insufficient_cash_returns_error(accounts):
    ctx = make_ctx(accounts, {"NVDA": 200.0})
    # 1M / 200 = 5000 shares max; request 6000
    result = await trade_impl(ctx, "NVDA", 6_000)

    assert result["success"] is False
    assert "Insufficient cash" in result["error"]
    assert accounts.cash("claude") == INITIAL_BALANCE
    assert accounts.holdings("claude") == {}


async def test_trade_cannot_short(accounts):
    ctx = make_ctx(accounts, {"NVDA": 200.0})
    result = await trade_impl(ctx, "NVDA", -1)

    assert result["success"] is False
    assert "Cannot sell" in result["error"]


async def test_trade_rejects_zero_quantity(accounts):
    ctx = make_ctx(accounts, {"NVDA": 200.0})
    result = await trade_impl(ctx, "NVDA", 0)

    assert result["success"] is False
    assert "non-zero" in result["error"]


async def test_trade_stopped_by_stop_event(accounts):
    import asyncio
    stop = asyncio.Event()
    stop.set()
    ctx = make_ctx(accounts, {"NVDA": 200.0})
    ctx.stop_event = stop
    result = await trade_impl(ctx, "NVDA", 10)

    assert result["success"] is False
    assert "stopped" in result["error"].lower()

