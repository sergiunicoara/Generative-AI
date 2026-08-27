"""
Real Option Chain — yfinance integration
==========================================
Replaces Black-Scholes modelled prices with real market bid/ask from yfinance.

Key functions:
    get_option_entry(ticker, entry_price, target_dte)
        → finds best 10% ITM call with ~60 DTE, returns real ask price + metadata
        → liquidity check: OI > 100, spread < 15%, bid > 0

    get_option_exit(ticker, strike, expiry_date)
        → returns current bid for a held call (sell at bid)

    check_option_liquidity(ticker, strike, expiry_date)
        → returns True if option is tradeable

Falls back to Black-Scholes (with 25% haircut) if:
    - yfinance returns no chain data
    - Option not found at target strike
    - Liquidity check fails
    - Market is closed / outside hours

Usage:
    from backend.environment.options import get_option_entry, get_option_exit
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Liquidity thresholds
# ---------------------------------------------------------------------------

MIN_OPEN_INTEREST = 100      # contracts — below this the option is illiquid
MAX_SPREAD_PCT    = 15.0     # bid/ask spread as % of mid — above this too wide
MIN_BID           = 0.05     # minimum bid to consider (filters out near-zero options)

# ---------------------------------------------------------------------------
# Black-Scholes fallback
# ---------------------------------------------------------------------------

RISK_FREE   = 0.05
EXPIRY_DAYS = 60

def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def _hist_vol(ticker: str, entry_date: date) -> float:
    """30-day historical volatility as IV proxy for BS fallback."""
    try:
        df = yf.download(ticker, period="60d", interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        rets = df["Close"].pct_change().dropna()
        return float(rets.tail(30).std() * math.sqrt(252))
    except Exception:
        return 0.35


def bs_option_entry(S_entry: float, ticker: str, entry_date: date) -> dict:
    """Black-Scholes fallback for option entry — 10% ITM, 60 DTE, 25% haircut."""
    iv  = _hist_vol(ticker, entry_date)
    K   = S_entry * 0.90
    T   = EXPIRY_DAYS / 365
    px  = _bs_call(S_entry, K, T, RISK_FREE, iv) * 0.75   # 25% haircut
    return {
        "strike":        round(K, 2),
        "expiry_date":   str(entry_date + timedelta(days=EXPIRY_DAYS)),
        "entry_px":      round(px, 4),
        "iv":            round(iv, 4),
        "open_interest": 0,
        "spread_pct":    0.0,
        "dte":           EXPIRY_DAYS,
        "source":        "black_scholes",
        "liquid":        False,
    }


def bs_option_exit(S_exit: float, S_entry: float,
                   strike: float, entry_date: date,
                   exit_date: date, iv: float) -> float:
    """Black-Scholes fallback for option exit price."""
    days_held = (exit_date - entry_date).days
    T_rem     = max((EXPIRY_DAYS - days_held * 1.4) / 365, 0)
    return _bs_call(S_exit, strike, T_rem, RISK_FREE, iv) * 0.75


# ---------------------------------------------------------------------------
# yfinance option chain helpers
# ---------------------------------------------------------------------------

def _find_expiry(expiries: tuple[str, ...], target_dte: int) -> str | None:
    """Return expiry string closest to target_dte calendar days from today."""
    if not expiries:
        return None
    today       = date.today()
    target_date = today + timedelta(days=target_dte)
    return min(expiries, key=lambda e: abs((date.fromisoformat(e) - target_date).days))


def _find_strike(calls: pd.DataFrame, target_strike: float) -> pd.Series | None:
    """Return the row in calls closest to target_strike."""
    if calls.empty:
        return None
    calls = calls.copy()
    calls["_diff"] = (calls["strike"] - target_strike).abs()
    row = calls.loc[calls["_diff"].idxmin()]
    return row


def _liquidity_ok(bid: float, ask: float, oi: int) -> tuple[bool, str]:
    """Return (ok, reason) for a given option."""
    if bid < MIN_BID:
        return False, f"bid {bid:.2f} < min {MIN_BID}"
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid * 100 if mid > 0 else 999
    if spread_pct > MAX_SPREAD_PCT:
        return False, f"spread {spread_pct:.1f}% > max {MAX_SPREAD_PCT}%"
    if oi < MIN_OPEN_INTEREST:
        return False, f"OI {oi} < min {MIN_OPEN_INTEREST}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_option_entry(
    ticker:      str,
    entry_price: float,
    target_dte:  int = 60,
) -> dict[str, Any]:
    """
    Find and return the best 10% ITM call with ~target_dte DTE.

    Returns a dict with:
        strike, expiry_date, entry_px, iv, open_interest,
        spread_pct, dte, source ('yfinance' or 'black_scholes'), liquid

    Always returns something — falls back to Black-Scholes if chain unavailable
    or option is illiquid.
    """
    today = date.today()
    try:
        tk       = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            raise ValueError("no expiries")

        expiry = _find_expiry(expiries, target_dte)
        if expiry is None:
            raise ValueError("no expiry found")

        chain    = tk.option_chain(expiry)
        calls    = chain.calls
        if calls.empty:
            raise ValueError("empty chain")

        target_strike = entry_price * 0.90
        row           = _find_strike(calls, target_strike)
        if row is None:
            raise ValueError("no strike found")

        bid = float(row["bid"])   if not pd.isna(row["bid"])   else 0.0
        ask = float(row["ask"])   if not pd.isna(row["ask"])   else 0.0
        iv  = float(row["impliedVolatility"]) if not pd.isna(row["impliedVolatility"]) else 0.35
        oi  = int(row["openInterest"])        if not pd.isna(row["openInterest"])        else 0
        mid = (bid + ask) / 2 if (bid + ask) > 0 else 0.0

        liquid, reason = _liquidity_ok(bid, ask, oi)
        spread_pct = (ask - bid) / mid * 100 if mid > 0 else 0.0
        dte = (date.fromisoformat(expiry) - today).days

        if not liquid:
            print(f"  [options] {ticker} chain found but illiquid: {reason} — using BS fallback")
            fb = bs_option_entry(entry_price, ticker, today)
            fb["chain_note"] = reason
            return fb

        return {
            "strike":        round(float(row["strike"]), 2),
            "expiry_date":   expiry,
            "entry_px":      round(ask, 4),    # buy at ask
            "iv":            round(iv, 4),
            "open_interest": oi,
            "spread_pct":    round(spread_pct, 1),
            "dte":           dte,
            "source":        "yfinance",
            "liquid":        True,
        }

    except Exception as e:
        print(f"  [options] {ticker} chain unavailable ({e}) — using BS fallback")
        return bs_option_entry(entry_price, ticker, today)


def get_option_exit(
    ticker:      str,
    strike:      float,
    expiry_date: str,
) -> float | None:
    """
    Return current bid price for a held call (sell at bid).
    Returns None if unavailable — caller should use BS fallback.
    """
    try:
        tk    = yf.Ticker(ticker)
        chain = tk.option_chain(expiry_date)
        calls = chain.calls
        if calls.empty:
            return None
        row = _find_strike(calls, strike)
        if row is None:
            return None
        bid = float(row["bid"]) if not pd.isna(row["bid"]) else 0.0
        return bid if bid >= MIN_BID else None
    except Exception:
        return None


def check_option_liquidity(
    ticker:      str,
    strike:      float,
    expiry_date: str,
) -> tuple[bool, str]:
    """
    Returns (liquid, reason) for a specific option contract.
    Use on entry day to gate whether to deploy the options sleeve.
    """
    try:
        tk    = yf.Ticker(ticker)
        chain = tk.option_chain(expiry_date)
        calls = chain.calls
        if calls.empty:
            return False, "empty chain"
        row   = _find_strike(calls, strike)
        if row is None:
            return False, "strike not found"
        bid = float(row["bid"]) if not pd.isna(row["bid"]) else 0.0
        ask = float(row["ask"]) if not pd.isna(row["ask"]) else 0.0
        oi  = int(row["openInterest"]) if not pd.isna(row["openInterest"]) else 0
        return _liquidity_ok(bid, ask, oi)
    except Exception as e:
        return False, str(e)

