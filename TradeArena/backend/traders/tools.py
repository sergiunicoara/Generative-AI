"""Game tools exposed to each trader agent.

`get_state` — full portfolio state + live pre-earnings signals with regime filter
              and dynamic scores (combined_strategy.md S2-Safer implementation).
`trade`     — buy / sell fractional shares at live price.

Strategy implemented: combined_strategy.md Strategy Two Safer
  Universe:    GOOGL, NVDA, AMZN, MSFT, META, AMD
  Regime:      QQQ > 150-day MA  (hold cash if not)
  Entry:       D-20 window (10-25 trading days before earnings)
  Score:       base + 1.20*mom20 + 0.80*mom60 + 1.50*rs20  (bounded ±20%)
  Threshold:   score >= 1.05  (safer version)
  Sizing:      35% of portfolio per position, max 4 positions
  Stop:        -5% from avg_cost
  Exit:        D-1 before earnings (earnings_exit_due=True) or stop hit
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from agents import RunContextWrapper, function_tool

from backend.environment.accounts import Accounts
from backend.environment.prices import Prices

# ---------------------------------------------------------------------------
# Strategy parameters (combined_strategy.md S2-Safer)
# ---------------------------------------------------------------------------

_WATCH_TICKERS   = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
_SCORE_THRESHOLD = 1.05       # S2 Safer minimum dynamic score
_STOP_PCT        = -5.0       # validated stop
_MAX_POSITIONS   = 4
_POSITION_PCT    = 0.35       # 35% of portfolio per position
_REGIME_MA       = 150        # QQQ simple moving average days

# Base quality scores per ticker (reflects historical win-rate × avg-return)
_BASE_QUALITY: dict[str, float] = {
    "GOOGL": 1.40,
    "NVDA":  1.50,
    "AMZN":  1.20,
    "MSFT":  1.10,
    "META":  1.10,
    "AMD":   1.00,
}

# ---------------------------------------------------------------------------
# TTL caches (avoid re-fetching on every cycle)
# ---------------------------------------------------------------------------

_regime_cache:   dict[str, Any] = {}   # {ts, ok, qqq, ma150}
_scores_cache:   dict[str, Any] = {}   # {ts, scores: {ticker: score}}
_earnings_cache: dict[str, list[str]] = {}

_REGIME_TTL = 3600   # 1 hour — 150dma moves slowly
_SCORES_TTL = 300    # 5 minutes — momentum changes intraday


# ---------------------------------------------------------------------------
# Regime filter
# ---------------------------------------------------------------------------

def _check_regime() -> dict:
    now = time.time()
    c   = _regime_cache
    if c.get("ts") and now - c["ts"] < _REGIME_TTL:
        return c
    try:
        hist = yf.Ticker("QQQ").history(period="1y", interval="1d")
        close = hist["Close"]
        ma    = float(close.rolling(_REGIME_MA).mean().iloc[-1])
        cur   = float(close.iloc[-1])
        c.update({"ts": now, "ok": cur > ma, "qqq": round(cur, 2), "ma150": round(ma, 2)})
    except Exception as e:
        c.update({"ts": now, "ok": True, "qqq": None, "ma150": None, "error": str(e)})
    return c


# ---------------------------------------------------------------------------
# Dynamic score
# ---------------------------------------------------------------------------

def _bounded(x: float) -> float:
    return max(-20.0, min(20.0, x)) / 100.0   # cap ±20%, convert to decimal


def _compute_scores() -> dict[str, float]:
    now = time.time()
    c   = _scores_cache
    if c.get("ts") and now - c["ts"] < _SCORES_TTL:
        return c.get("scores", {})
    try:
        all_tickers = _WATCH_TICKERS + ["QQQ"]
        raw = yf.download(all_tickers, period="90d", interval="1d",
                          progress=False, auto_adjust=True)
        close = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
        if isinstance(close, pd.Series):
            close = close.to_frame()

        qqq_20d = (close["QQQ"].iloc[-1] / close["QQQ"].iloc[-21] - 1) * 100

        scores: dict[str, float] = {}
        for t in _WATCH_TICKERS:
            if t not in close.columns:
                continue
            col = close[t].dropna()
            if len(col) < 62:
                continue
            mom20 = (col.iloc[-1] / col.iloc[-21] - 1) * 100
            mom60 = (col.iloc[-1] / col.iloc[-62] - 1) * 100
            rs20  = mom20 - float(qqq_20d)
            base  = _BASE_QUALITY.get(t, 1.0)
            score = (base
                     + 1.20 * _bounded(mom20)
                     + 0.80 * _bounded(mom60)
                     + 1.50 * _bounded(rs20))
            scores[t] = round(score, 3)

        c.update({"ts": now, "scores": scores})
        return scores
    except Exception:
        return c.get("scores", {})


# ---------------------------------------------------------------------------
# Earnings date helpers
# ---------------------------------------------------------------------------

def _fetch_earnings_dates(ticker: str) -> list[str]:
    if ticker in _earnings_cache:
        return _earnings_cache[ticker]
    dates: list[str] = []
    today = date.today().isoformat()
    try:
        obj = yf.Ticker(ticker)
        ed  = obj.earnings_dates
        if ed is not None and not ed.empty:
            for d in ed.index:
                ds = str(d.date()) if hasattr(d, "date") else str(d)[:10]
                if ds >= today:
                    dates.append(ds)
        try:
            cal = obj.calendar
            if cal:
                el = cal.get("Earnings Date", [])
                if hasattr(el, "tolist"):
                    el = el.tolist()
                for dv in (el if isinstance(el, list) else [el]):
                    ds = str(pd.Timestamp(dv).date())
                    if ds >= today and ds not in dates:
                        dates.append(ds)
        except Exception:
            pass
    except Exception:
        pass
    _earnings_cache[ticker] = sorted(set(dates))
    return _earnings_cache[ticker]


def _trading_days_until(ann_date: str) -> int:
    today  = date.today()
    target = date.fromisoformat(ann_date)
    if target <= today:
        return 0
    count, cur = 0, today
    while cur < target:
        cur = date.fromordinal(cur.toordinal() + 1)
        if cur.weekday() < 5:
            count += 1
    return count


def _next_earnings(ticker: str) -> tuple[str, int] | None:
    dates = _fetch_earnings_dates(ticker)
    if not dates:
        return None
    ann = dates[0]
    return (ann, _trading_days_until(ann))


# ---------------------------------------------------------------------------
# Signal scanner
# ---------------------------------------------------------------------------

def _scan_signals(regime_ok: bool) -> list[dict]:
    """
    Return scored pre-earnings signals in the D-20 window.
    Filtered by regime (QQQ > 150dma) and dynamic score >= 1.05.
    Sorted by score descending (best first).
    """
    if not regime_ok:
        return []

    scores = _compute_scores()
    signals = []

    for ticker in _WATCH_TICKERS:
        result = _next_earnings(ticker)
        if not result:
            continue
        ann, td = result
        if not (10 <= td <= 25):
            continue

        score = scores.get(ticker, _BASE_QUALITY.get(ticker, 1.0))
        passes = score >= _SCORE_THRESHOLD

        signals.append({
            "ticker":              ticker,
            "earnings_date":       ann,
            "trading_days_until":  td,
            "dynamic_score":       score,
            "score_passes":        passes,
            "action": "BUY" if passes else "SKIP (score below 1.05)",
            "note": (
                f"Score {score:.3f} ({'✓' if passes else '✗ below 1.05'}). "
                f"Earnings {ann} ({td}d). "
                f"{'Enter at 35% if slot available.' if passes else 'Skip — low momentum.'}"
            ),
        })

    # Sort by score desc; blocked signals still shown for transparency
    return sorted(signals, key=lambda s: -s["dynamic_score"])


# ---------------------------------------------------------------------------
# TraderContext
# ---------------------------------------------------------------------------

@dataclass
class TraderContext:
    trader_id:        str
    accounts:         Accounts
    prices:           Prices
    started_at:       datetime
    duration_seconds: float
    rival_ids:        list[str] = field(default_factory=list)
    stop_event:       asyncio.Event | None = None


# ---------------------------------------------------------------------------
# Holding detail (per position)
# ---------------------------------------------------------------------------

def holding_detail(ticker: str, pos: dict[str, float], price: float) -> dict[str, Any]:
    qty     = pos["quantity"]
    avg     = pos["avg_cost"]
    mv      = qty * price
    pnl     = mv - qty * avg
    pnl_pct = (price - avg) / avg * 100 if avg else 0
    stop_px = round(avg * (1 + _STOP_PCT / 100), 4)

    er = _next_earnings(ticker.upper())
    if er:
        earnings_date, td = er
        exit_due = td <= 1
    else:
        earnings_date, td, exit_due = None, None, False

    return {
        "quantity":                   qty,
        "avg_cost":                   avg,
        "current_price":              price,
        "market_value":               round(mv, 2),
        "unrealized_pnl":             round(pnl, 2),
        "unrealized_pnl_pct":         round(pnl_pct, 2),
        "stop_price":                 stop_px,
        "stop_triggered":             price <= stop_px,
        "earnings_date":              earnings_date,
        "trading_days_until_earnings":td,
        "earnings_exit_due":          exit_due,
    }


# ---------------------------------------------------------------------------
# get_state
# ---------------------------------------------------------------------------

async def get_state_impl(tc: TraderContext) -> dict[str, Any]:
    now     = datetime.now(timezone.utc)
    elapsed = (now - tc.started_at).total_seconds()

    own_holdings = tc.accounts.holdings(tc.trader_id)
    tickers      = set(own_holdings)
    for rid in tc.rival_ids:
        tickers.update(tc.accounts.holdings(rid))
    current_prices = await tc.prices.aget_prices(sorted(tickers)) if tickers else {}

    own_value = tc.accounts.portfolio_value(tc.trader_id, current_prices)
    cash      = tc.accounts.cash(tc.trader_id)

    regime    = _check_regime()
    signals   = _scan_signals(regime["ok"])

    holdings_detail = {
        t: holding_detail(t, p, current_prices[t])
        for t, p in own_holdings.items()
    }

    stops_triggered   = [t for t, h in holdings_detail.items() if h["stop_triggered"]]
    earnings_exit_due = [t for t, h in holdings_detail.items() if h["earnings_exit_due"]]
    sell_now          = sorted(set(stops_triggered + earnings_exit_due))

    # Best actionable buy right now
    buyable = [s for s in signals if s["score_passes"] and s["ticker"] not in own_holdings]
    next_buy = buyable[0]["ticker"] if buyable else None

    return {
        "trader_id":              tc.trader_id,
        "time_elapsed_seconds":   round(elapsed, 1),
        "time_remaining_seconds": round(max(0.0, tc.duration_seconds - elapsed), 1),
        # Regime
        "regime": {
            "ok":     regime["ok"],
            "qqq":    regime.get("qqq"),
            "ma150":  regime.get("ma150"),
            "note":   "TRADE — QQQ above 150dma" if regime["ok"] else "HOLD CASH — QQQ below 150dma, no new entries",
        },
        # Portfolio
        "cash":                   round(cash, 2),
        "open_positions":         len(own_holdings),
        "slots_available":        max(0, _MAX_POSITIONS - len(own_holdings)),
        "holdings":               holdings_detail,
        "total_portfolio_value":  round(own_value, 2),
        "total_pnl":              round(tc.accounts.pnl(tc.trader_id, own_value), 2),
        # Immediate actions
        "sell_now":               sell_now,
        "sell_now_reason": {
            t: ("stop -5% hit" if h["stop_triggered"] else "D-1 exit — earnings tomorrow")
            for t, h in holdings_detail.items()
            if h["stop_triggered"] or h["earnings_exit_due"]
        },
        "next_buy":               next_buy,
        # Signals
        "pre_earnings_signals":   signals,
        "rivals_total_portfolio_value": {
            rid: round(tc.accounts.portfolio_value(rid, current_prices), 2)
            for rid in tc.rival_ids
        },
    }


# ---------------------------------------------------------------------------
# trade
# ---------------------------------------------------------------------------

async def trade_impl(tc: TraderContext, ticker: str, quantity: float) -> dict[str, Any]:
    ticker = ticker.upper()
    if tc.stop_event and tc.stop_event.is_set():
        return {"success": False, "error": "Arena has stopped — no new trades accepted."}
    price = await tc.prices.aget_price(ticker)
    try:
        tc.accounts.execute_trade(tc.trader_id, ticker, quantity, price)
    except ValueError as e:
        return {"success": False, "error": str(e), "ticker": ticker,
                "requested_quantity": quantity, "price": price}
    return {
        "success":    True,
        "ticker":     ticker,
        "quantity":   quantity,
        "price":      price,
        "side":       "buy" if quantity > 0 else "sell",
        "cash_after": round(tc.accounts.cash(tc.trader_id), 2),
    }


# ---------------------------------------------------------------------------
# SDK tools
# ---------------------------------------------------------------------------

@function_tool
async def get_state(ctx: RunContextWrapper[TraderContext]) -> dict[str, Any]:
    """Current game state: regime filter, portfolio, sell_now list, scored pre-earnings signals."""
    return await get_state_impl(ctx.context)


@function_tool
async def trade(
    ctx: RunContextWrapper[TraderContext], ticker: str, quantity: float
) -> dict[str, Any]:
    """Buy or sell shares at live price.

    Args:
        ticker: US stock ticker e.g. 'NVDA'.
        quantity: Shares. Positive=buy, negative=sell. Fractional ok. No shorting.
    """
    return await trade_impl(ctx.context, ticker, quantity)

