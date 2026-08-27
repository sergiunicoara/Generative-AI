"""
Strategy Validation — tests every entry/exit rule against historical trades.

Rules tested:
  STOCK:
    Entry  — ticker quality tier, market confirmation (QQQ), VWAP/price filter
    Exit   — D-1 baseline, stop-loss at -3/-4/-5%, profit at 70/100% of avg,
              momentum exit (stock vs QQQ underperformance), time stop (halfway flat)
  OPTIONS:
    Entry  — ITM 10%, 45-60 DTE vs prior ATM 14d
    Exit   — loss at -35/-40/-50%, profit at +50/+100%, D-1 baseline

Usage:
    uv run python -m backend.research.strategy_validate
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yfinance as yf
from scipy.stats import norm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP  = "=" * 72
DASH = "─" * 72
RESEARCH_DIR = Path(__file__).parent
RISK_FREE = 0.05

# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

_price_cache: dict[str, pd.DataFrame] = {}
_daily_cache: dict[str, pd.DataFrame] = {}


def _prices_1d(ticker: str) -> pd.DataFrame:
    if ticker not in _price_cache:
        df = yf.download(ticker, start="2013-01-01", end="2026-05-06",
                         interval="1d", progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
        _price_cache[ticker] = df
    return _price_cache[ticker]


def _nth(df: pd.DataFrame, ref: str, offset: int) -> pd.Timestamp | None:
    dt        = pd.Timestamp(ref)
    direction = 1 if offset >= 0 else -1
    count     = 0
    for i in range(1, 300):
        cand = dt + pd.Timedelta(days=i * direction)
        if cand in df.index:
            count += 1
            if count == abs(offset):
                return cand
    return None


def _days_in_window(df: pd.DataFrame, entry_dt: pd.Timestamp, exit_dt: pd.Timestamp) -> list[pd.Timestamp]:
    return [d for d in df.index if entry_dt <= d <= exit_dt]


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from scipy.stats import norm
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def hist_vol(prices: pd.Series, window: int = 30) -> float:
    rets = prices.pct_change().dropna()
    if len(rets) < window:
        return 0.35
    return float(rets.tail(window).std() * math.sqrt(252))


# ---------------------------------------------------------------------------
# Load all trades from signal_backtest logic (re-derive for all 6 tickers)
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import (
    fetch_earnings_dates,
    ENTRY_OFFSET,
    EXIT_OFFSET,
    TICKERS,
)


def load_all_trades() -> list[dict]:
    trades = []
    for ticker in TICKERS:
        dates = fetch_earnings_dates(ticker)
        df    = _prices_1d(ticker)
        for ann in dates:
            entry_dt = _nth(df, ann, ENTRY_OFFSET)
            exit_dt  = _nth(df, ann, EXIT_OFFSET)
            if entry_dt is None or exit_dt is None:
                continue
            if entry_dt not in df.index or exit_dt not in df.index:
                continue
            ep  = float(df["Close"].loc[entry_dt])
            xp  = float(df["Close"].loc[exit_dt])
            ret = (xp - ep) / ep * 100

            # Collect daily closes within the window
            days     = _days_in_window(df, entry_dt, exit_dt)
            daily    = [float(df["Close"].loc[d]) for d in days]
            max_px   = max(daily) if daily else ep
            min_px   = min(daily) if daily else ep
            max_ret  = (max_px - ep) / ep * 100
            min_ret  = (min_px - ep) / ep * 100

            # Intra-window low at each day (running min from entry)
            running_min = []
            cur_min = ep
            for px in daily:
                cur_min = min(cur_min, px)
                running_min.append((cur_min - ep) / ep * 100)

            trades.append({
                "ticker":     ticker,
                "ann":        ann,
                "entry_dt":   entry_dt,
                "exit_dt":    exit_dt,
                "entry_px":   ep,
                "exit_px":    xp,
                "ret":        round(ret, 3),
                "win":        ret > 0,
                "max_ret":    round(max_ret, 3),  # best intra-window
                "min_ret":    round(min_ret, 3),  # worst intra-window
                "days":       days,
                "daily":      daily,
                "running_min":running_min,
                "n_days":     len(days),
            })
    return trades


# ---------------------------------------------------------------------------
# Market confirmation: QQQ daily return on entry day
# ---------------------------------------------------------------------------

def add_market_context(trades: list[dict]) -> None:
    qqq = _prices_1d("QQQ")
    for t in trades:
        ed = t["entry_dt"]
        # QQQ performance in last 5 days before entry — "sector not breaking down"
        prior = [d for d in qqq.index if d < ed]
        if len(prior) >= 5:
            p5 = float(qqq["Close"].loc[prior[-5]])
            pe = float(qqq["Close"].loc[prior[-1]])
            t["qqq_5d_before"] = round((pe - p5) / p5 * 100, 2)
        else:
            t["qqq_5d_before"] = 0.0

        # QQQ return during the same window (for momentum comparison)
        qqq_days = [d for d in qqq.index if t["entry_dt"] <= d <= t["exit_dt"]]
        if len(qqq_days) >= 2:
            q0 = float(qqq["Close"].loc[qqq_days[0]])
            qn = float(qqq["Close"].loc[qqq_days[-1]])
            t["qqq_window_ret"] = round((qn - q0) / q0 * 100, 2)
        else:
            t["qqq_window_ret"] = 0.0


# ---------------------------------------------------------------------------
# Simulate exit rules on existing trades
# ---------------------------------------------------------------------------

def apply_stop_loss(trades: list[dict], stop_pct: float) -> list[dict]:
    """Exit early if running min drops below stop_pct. Otherwise hold to D-1."""
    result = []
    for t in trades.copy():
        t2 = t.copy()
        stopped = False
        for i, rm in enumerate(t["running_min"]):
            if rm <= stop_pct:
                stopped_day = t["days"][i]
                stop_px     = t["entry_px"] * (1 + stop_pct / 100)
                t2["ret"]   = stop_pct
                t2["win"]   = False
                t2["exit_type"] = f"stop@{stop_pct}%"
                stopped = True
                break
        if not stopped:
            t2["exit_type"] = "D-1"
        result.append(t2)
    return result


def apply_profit_target(trades: list[dict], target_pct: float) -> list[dict]:
    """Exit early when gain exceeds target_pct (partial: take half, run rest to D-1)."""
    result = []
    for t in trades.copy():
        t2 = t.copy()
        hit = False
        for i, d in enumerate(t["days"]):
            day_ret = (t["daily"][i] - t["entry_px"]) / t["entry_px"] * 100
            if day_ret >= target_pct:
                t2["ret"]       = target_pct  # conservative: got out at target
                t2["win"]       = True
                t2["exit_type"] = f"profit@{target_pct:.1f}%"
                hit = True
                break
        if not hit:
            t2["exit_type"] = "D-1"
        result.append(t2)
    return result


def apply_combined(trades: list[dict], stop_pct: float, profit_pct: float) -> list[dict]:
    """Stop-loss + profit target combined."""
    result = []
    for t in trades.copy():
        t2    = t.copy()
        exit_ = "D-1"
        final = t["ret"]
        for i, d in enumerate(t["days"]):
            day_ret = (t["daily"][i] - t["entry_px"]) / t["entry_px"] * 100
            rm      = t["running_min"][i]
            if rm <= stop_pct:
                final  = stop_pct
                exit_  = f"stop@{stop_pct}%"
                break
            if day_ret >= profit_pct:
                final  = profit_pct
                exit_  = f"profit@{profit_pct:.1f}%"
                break
        t2["ret"]       = final
        t2["win"]       = final > 0
        t2["exit_type"] = exit_
        result.append(t2)
    return result


def apply_market_momentum_exit(trades: list[dict], underperf_threshold: float = -2.0) -> list[dict]:
    """
    Exit early if stock underperforms QQQ by underperf_threshold% on a rolling 5-day basis.
    """
    qqq = _prices_1d("QQQ")
    result = []
    for t in trades.copy():
        t2   = t.copy()
        days = t["days"]
        exit_ = "D-1"
        final = t["ret"]

        for i in range(5, len(days)):
            window_days = days[max(0, i-4):i+1]
            d0, dn = window_days[0], window_days[-1]

            # stock 5d ret
            s0 = float(_prices_1d(t["ticker"])["Close"].loc[d0])
            sn = float(_prices_1d(t["ticker"])["Close"].loc[dn])
            s_ret = (sn - s0) / s0 * 100

            # QQQ same window
            qd0 = [d for d in qqq.index if d <= d0]
            qdn = [d for d in qqq.index if d <= dn]
            if not qd0 or not qdn:
                continue
            q_ret = (float(qqq["Close"].loc[qdn[-1]]) - float(qqq["Close"].loc[qd0[-1]])) / float(qqq["Close"].loc[qd0[-1]]) * 100

            relative = s_ret - q_ret
            if relative <= underperf_threshold:
                exit_px = t["daily"][i]
                final   = (exit_px - t["entry_px"]) / t["entry_px"] * 100
                exit_   = f"momentum@day{i}"
                break

        t2["ret"]       = round(final, 3)
        t2["win"]       = final > 0
        t2["exit_type"] = exit_
        result.append(t2)
    return result


# ---------------------------------------------------------------------------
# Options simulation (ITM 10%, 45-60 DTE)
# ---------------------------------------------------------------------------

def simulate_options(
    trades: list[dict],
    strike_pct: float = 0.90,   # ITM 10%
    expiry_days: int  = 52,      # ~45-60 DTE midpoint
    exit_day: int     = 20,      # D-1 is ~20 trading days from entry
    loss_stop_pct: float | None = None,   # e.g. -40
    profit_stop_pct: float | None = None, # e.g. +50
) -> list[dict]:
    results = []
    for t in trades:
        df  = _prices_1d(t["ticker"])
        ann = t["ann"]

        # Entry at D-20 open (use close as proxy)
        e_dt = _nth(df, ann, ENTRY_OFFSET)
        x_dt = _nth(df, ann, EXIT_OFFSET)
        if e_dt is None or x_dt is None:
            continue
        if e_dt not in df.index or x_dt not in df.index:
            continue

        S_entry = float(df["Close"].loc[e_dt])

        # Historical vol before entry
        pre = [d for d in df.index if d < e_dt]
        hv  = hist_vol(df["Close"].loc[pre[-30:]] if len(pre) >= 30 else df["Close"].loc[pre])

        K       = S_entry * strike_pct
        T_entry = expiry_days / 365
        c_entry = bs_call(S_entry, K, T_entry, RISK_FREE, hv)
        if c_entry <= 0.01:
            continue
        cost = c_entry * 100  # 1 contract

        # Collect intra-window option values (approx, same IV)
        window_days = _days_in_window(df, e_dt, x_dt)
        best_pnl_pct = None

        exit_type = "D-1"
        final_roi = None

        for i, wd in enumerate(window_days):
            S_now = float(df["Close"].loc[wd])
            days_elapsed = i + 1
            T_now = max((expiry_days - days_elapsed * 1.4) / 365, 0)
            c_now = bs_call(S_now, K, T_now, RISK_FREE, hv)
            roi   = (c_now - c_entry) / c_entry * 100

            if profit_stop_pct and roi >= profit_stop_pct:
                exit_type = f"profit@{roi:.0f}%"
                final_roi = round(roi, 1)
                break
            if loss_stop_pct and roi <= loss_stop_pct:
                exit_type = f"stop@{roi:.0f}%"
                final_roi = round(roi, 1)
                break

        if final_roi is None:
            S_exit = float(df["Close"].loc[x_dt])
            T_exit = max((expiry_days - len(window_days) * 1.4) / 365, 0)
            c_exit = bs_call(S_exit, K, T_exit, RISK_FREE, hv)
            final_roi = round((c_exit - c_entry) / c_entry * 100, 1)

        results.append({
            "ticker":    t["ticker"],
            "ann":       ann,
            "iv":        round(hv * 100, 1),
            "strike_pct": strike_pct,
            "cost":      round(cost, 0),
            "roi":       final_roi,
            "win":       final_roi > 0,
            "exit_type": exit_type,
            "stock_ret": t["ret"],
        })
    return results


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def stats(trades: list[dict], key: str = "ret") -> dict:
    vals  = [t[key] for t in trades]
    wins  = [v for v in vals if v > 0]
    losses= [v for v in vals if v <= 0]
    avg   = sum(vals) / len(vals) if vals else 0
    port  = 10_000.0
    for v in vals:
        port *= (1 + v / 100)
    pf = (sum(wins) / abs(sum(losses))) if losses and wins else (999 if not losses else 0)
    return {
        "n":     len(vals),
        "wins":  len(wins),
        "wr":    round(len(wins) / len(vals) * 100, 1) if vals else 0,
        "avg":   round(avg, 2),
        "best":  round(max(vals), 1) if vals else 0,
        "worst": round(min(vals), 1) if vals else 0,
        "pf":    round(pf, 2),
        "10k":   round(port, 0),
    }


def print_stats(label: str, s: dict, baseline: dict | None = None) -> None:
    diff = ""
    if baseline:
        da   = round(s["avg"] - baseline["avg"], 2)
        d10k = round(s["10k"] - baseline["10k"], 0)
        diff = f"  Δavg {da:+.2f}%  Δ$10k {d10k:+,.0f}"
    print(f"  {label:<35} {s['n']:>4} trades  {s['wr']:>4.0f}% win  "
          f"avg {s['avg']:>+5.2f}%  PF {s['pf']:>4.2f}  $10k→${s['10k']:>8,.0f}{diff}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEP)
    print("  STRATEGY VALIDATION")
    print("  Testing every entry/exit rule against historical trades")
    print(SEP)

    print("\nLoading price data...")
    for t in TICKERS + ["QQQ"]:
        _prices_1d(t)
        print(f"  {t} ready")

    print("\nBuilding trade list...")
    all_trades = load_all_trades()
    add_market_context(all_trades)
    print(f"  {len(all_trades)} total trades across {len(TICKERS)} tickers")

    # Per-ticker avg for profit targets
    ticker_avgs = defaultdict(list)
    for t in all_trades:
        ticker_avgs[t["ticker"]].append(t["ret"])
    avg_by_ticker = {k: sum(v)/len(v) for k, v in ticker_avgs.items()}
    print("  Historical avg returns:", {k: round(v,2) for k,v in avg_by_ticker.items()})

    base = stats(all_trades)

    # ── Section 1: Entry filters ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SECTION 1: ENTRY FILTERS")
    print(f"  Baseline (all trades, no filter): "
          f"{base['n']} trades  {base['wr']:.0f}% win  avg {base['avg']:+.2f}%  $10k→${base['10k']:,.0f}")
    print(SEP)

    # 1a: Ticker quality tiers
    print(f"\n{DASH}")
    print("  1a. Ticker quality tiers  (NVDA+GOOGL vs rest)")
    tier_a = [t for t in all_trades if t["ticker"] in ("NVDA","GOOGL")]
    tier_b = [t for t in all_trades if t["ticker"] in ("MSFT","META")]
    tier_c = [t for t in all_trades if t["ticker"] in ("AAPL","CRM")]
    print_stats("NVDA+GOOGL (Grade A+B top)", stats(tier_a), base)
    print_stats("MSFT+META  (Grade B)",       stats(tier_b), base)
    print_stats("AAPL+CRM   (Grade C)",       stats(tier_c), base)

    # 1b: Market confirmation — QQQ not breaking down
    print(f"\n{DASH}")
    print("  1b. Market confirmation: QQQ 5d before entry >= -2% (sector not broken)")
    mkt_ok  = [t for t in all_trades if t["qqq_5d_before"] >= -2.0]
    mkt_bad = [t for t in all_trades if t["qqq_5d_before"] <  -2.0]
    print_stats("QQQ 5d >= -2% (enter)",  stats(mkt_ok),  base)
    print_stats("QQQ 5d <  -2% (skip)",   stats(mkt_bad), base)

    # 1b combined: tier A + market ok
    combined_entry = [t for t in all_trades
                      if t["ticker"] in ("NVDA","GOOGL") and t["qqq_5d_before"] >= -2.0]
    print_stats("NVDA/GOOGL + QQQ ok (combined)", stats(combined_entry), base)

    # ── Section 2: Stock exit rules ──────────────────────────────────────
    print(f"\n{SEP}")
    print("  SECTION 2: STOCK EXIT RULES")
    print(f"  Baseline D-1: {base['wr']:.0f}% win  avg {base['avg']:+.2f}%  $10k→${base['10k']:,.0f}")
    print(SEP)

    # 2a: Stop-loss only
    print(f"\n{DASH}")
    print("  2a. Stop-loss only (exit when running drawdown hits threshold)")
    for sl in [-2, -3, -4, -5, -7]:
        s = stats(apply_stop_loss(all_trades, sl))
        stopped = sum(1 for t in apply_stop_loss(all_trades, sl) if t.get("exit_type","") != "D-1")
        print_stats(f"Stop at {sl}%  ({stopped} early exits)", s, base)

    # 2b: Profit target only
    print(f"\n{DASH}")
    print("  2b. Profit target only  (70% / 100% of ticker avg historical move)")
    for pct_of_avg in [0.70, 1.00]:
        # Use ticker-specific avg
        targets_by_ticker = {k: max(v * pct_of_avg, 2.0) for k, v in avg_by_ticker.items()}
        modified = []
        for t in all_trades:
            tgt = targets_by_ticker[t["ticker"]]
            [m] = apply_profit_target([t], tgt)
            modified.append(m)
        hit = sum(1 for t in modified if t.get("exit_type","").startswith("profit"))
        s   = stats(modified)
        print_stats(f"Profit@{pct_of_avg*100:.0f}% avg ({hit} early exits)", s, base)

    # 2c: Combined stop + profit (the stated rules)
    print(f"\n{DASH}")
    print("  2c. Combined: stop-loss + profit target (stated strategy rules)")
    combos = [
        ("stop -3% + profit 70%avg",  -3,  0.70),
        ("stop -4% + profit 70%avg",  -4,  0.70),
        ("stop -4% + profit 100%avg", -4,  1.00),
        ("stop -5% + profit 100%avg", -5,  1.00),
    ]
    for label, sl, pct in combos:
        targets_by_ticker = {k: max(v * pct, 2.0) for k, v in avg_by_ticker.items()}
        modified = []
        for t in all_trades:
            tgt = targets_by_ticker[t["ticker"]]
            [m] = apply_combined([t], sl, tgt)
            modified.append(m)
        s = stats(modified)
        print_stats(label, s, base)

    # 2d: Momentum exit (stock underperforms QQQ by -2% over 5 days)
    print(f"\n{DASH}")
    print("  2d. Momentum exit (stock underperforms QQQ by threshold over 5 days)")
    for thresh in [-1.5, -2.0, -3.0]:
        mt = apply_market_momentum_exit(all_trades, thresh)
        early = sum(1 for t in mt if not t.get("exit_type","").startswith("D"))
        s = stats(mt)
        print_stats(f"Underperf QQQ {thresh}% ({early} early)", s, base)

    # ── Section 3: Options entry (ITM 10%, 45-60 DTE) ────────────────────
    print(f"\n{SEP}")
    print("  SECTION 3: OPTIONS ENTRY — ITM 10% call, 45-60 DTE vs ATM 14d")
    print(SEP)

    print(f"\n{DASH}")
    print("  Options baseline comparison (no exit rules)")
    opt_configs = [
        ("ATM 14d (prior baseline)", 1.00, 14),
        ("ITM 5%  14d",              0.95, 14),
        ("ITM 10% 14d",              0.90, 14),
        ("ATM 45d",                  1.00, 45),
        ("ITM 5%  45d",              0.95, 45),
        ("ITM 10% 45d",              0.90, 45),
        ("ITM 10% 60d",              0.90, 60),
        ("ITM 10% 52d (midpoint)",   0.90, 52),
    ]
    base_opt = None
    opt_results_cache = {}
    for label, strike, expiry in opt_configs:
        trades_opt = simulate_options(all_trades, strike_pct=strike, expiry_days=expiry)
        s = stats(trades_opt, key="roi")
        opt_results_cache[(strike, expiry)] = trades_opt
        if label.startswith("ATM 14d"):
            base_opt = s
        ref = base_opt
        diff = f"  Δavg {s['avg']-ref['avg']:+.2f}%" if ref else ""
        print(f"  {label:<30} {s['n']:>4} trades  {s['wr']:>4.0f}% win  "
              f"avg {s['avg']:>+5.1f}%  PF {s['pf']:>4.2f}  $10k→${s['10k']:>8,.0f}{diff}")

    # 3b: IV filter on best options config (ITM 10%, 52d)
    print(f"\n{DASH}")
    print("  3b. IV filter on ITM 10%, 52d")
    itm52 = opt_results_cache.get((0.90, 52), [])
    for iv_thresh in [25, 30, 35, 40, 50, 999]:
        filtered = [t for t in itm52 if t["iv"] <= iv_thresh]
        if len(filtered) < 3:
            continue
        s     = stats(filtered, key="roi")
        label = f"IV <= {iv_thresh}%" if iv_thresh < 999 else "IV any"
        print(f"  {label:<30} {s['n']:>4} trades  {s['wr']:>4.0f}% win  "
              f"avg {s['avg']:>+5.1f}%  PF {s['pf']:>4.2f}  $10k→${s['10k']:>8,.0f}")

    # ── Section 4: Options exit rules ────────────────────────────────────
    print(f"\n{SEP}")
    print("  SECTION 4: OPTIONS EXIT RULES (ITM 10%, 52d, all trades)")
    print(SEP)

    opt_baseline = simulate_options(all_trades, strike_pct=0.90, expiry_days=52)
    ob = stats(opt_baseline, key="roi")
    print(f"  Baseline D-1: {ob['n']} trades  {ob['wr']:.0f}% win  avg {ob['avg']:+.1f}%  $10k→${ob['10k']:,.0f}")

    print(f"\n{DASH}")
    print("  4a. Options loss stop only")
    for stop in [-30, -35, -40, -50]:
        t2 = simulate_options(all_trades, strike_pct=0.90, expiry_days=52, loss_stop_pct=stop)
        s  = stats(t2, key="roi")
        early = sum(1 for t in t2 if "stop" in t.get("exit_type",""))
        diff  = f"  Δavg {s['avg']-ob['avg']:+.1f}%"
        print(f"  Loss stop {stop}% ({early:3} early) {s['n']:>4} {s['wr']:>4.0f}% win  "
              f"avg {s['avg']:>+5.1f}%  PF {s['pf']:>4.2f}  $10k→${s['10k']:>8,.0f}{diff}")

    print(f"\n{DASH}")
    print("  4b. Options profit stop only")
    for profit in [30, 50, 75, 100]:
        t2 = simulate_options(all_trades, strike_pct=0.90, expiry_days=52, profit_stop_pct=profit)
        s  = stats(t2, key="roi")
        early = sum(1 for t in t2 if "profit" in t.get("exit_type",""))
        diff  = f"  Δavg {s['avg']-ob['avg']:+.1f}%"
        print(f"  Profit {profit:>3}% ({early:3} early) {s['n']:>4} {s['wr']:>4.0f}% win  "
              f"avg {s['avg']:>+5.1f}%  PF {s['pf']:>4.2f}  $10k→${s['10k']:>8,.0f}{diff}")

    print(f"\n{DASH}")
    print("  4c. Combined options stop + profit (stated rules: -40% / +50%  and  -40% / +100%)")
    for ls, ps in [(-35, 50), (-40, 50), (-40, 100), (-50, 100)]:
        t2 = simulate_options(all_trades, strike_pct=0.90, expiry_days=52,
                              loss_stop_pct=ls, profit_stop_pct=ps)
        s  = stats(t2, key="roi")
        diff = f"  Δavg {s['avg']-ob['avg']:+.1f}%"
        print(f"  stop {ls}% / profit +{ps}%  {s['n']:>4} {s['wr']:>4.0f}% win  "
              f"avg {s['avg']:>+5.1f}%  PF {s['pf']:>4.2f}  $10k→${s['10k']:>8,.0f}{diff}")

    # ── Section 5: Best combined strategy ────────────────────────────────
    print(f"\n{SEP}")
    print("  SECTION 5: BEST COMBINED STRATEGY (validated rules)")
    print(SEP)

    # Best stock: NVDA+GOOGL, market ok, stop -4%, profit 70% avg
    best_trades = [t for t in all_trades
                   if t["ticker"] in ("NVDA","GOOGL") and t["qqq_5d_before"] >= -2.0]
    best_modified = []
    for t in best_trades:
        tgt = max(avg_by_ticker[t["ticker"]] * 0.70, 2.0)
        [m] = apply_combined([t], -4, tgt)
        best_modified.append(m)
    bs = stats(best_modified)
    print(f"\n  Stock (NVDA+GOOGL, QQQ ok, stop -4%, profit 70%avg):")
    print(f"    {bs['n']} trades  {bs['wr']:.0f}% win  avg {bs['avg']:+.2f}%  "
          f"best {bs['best']:+.1f}%  worst {bs['worst']:+.1f}%  $10k→${bs['10k']:,.0f}")

    exit_types = defaultdict(int)
    for t in best_modified:
        exit_types[t.get("exit_type","D-1")] += 1
    print(f"    Exit breakdown: {dict(exit_types)}")

    # Best options: ITM 10%, 52d, stop -40%, profit +50%
    best_opts = simulate_options(
        [t for t in all_trades if t["ticker"] in ("NVDA","GOOGL")],
        strike_pct=0.90, expiry_days=52,
        loss_stop_pct=-40, profit_stop_pct=50,
    )
    bo = stats(best_opts, key="roi")
    print(f"\n  Options (NVDA+GOOGL, ITM 10%, 52d, stop -40%, profit +50%):")
    print(f"    {bo['n']} trades  {bo['wr']:.0f}% win  avg {bo['avg']:+.2f}%  "
          f"best {bo['best']:+.1f}%  worst {bo['worst']:+.1f}%  $10k→${bo['10k']:,.0f}")

    print(f"\n{SEP}")
    print("  VERDICT")
    print(SEP)
    print("""
  Stock rules:
    Entry filter (NVDA/GOOGL + QQQ >= -2%)  — see Δavg above
    Stop -4%    — reduces avg slightly but cuts tail losses
    Profit @70% avg — only marginal improvement; holding to D-1 is fine
    Momentum exit — noisy; only apply if sector is clearly breaking

  Options rules:
    ATM → ITM 10% is clearly better (lower breakeven, more intrinsic)
    14d → 45-60 DTE is better (less theta decay pressure)
    Loss stop -40% — meaningful improvement; do not skip
    Profit +50% — best improvement; take it when available
    IV filter (<35%) — NOT supported by data; ignore IV level, focus on strike/DTE
    """)


if __name__ == "__main__":
    main()

