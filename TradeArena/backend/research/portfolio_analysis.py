"""
Portfolio Analysis — aligns arena config with latest research.

Answers:
  1. Recent-year (2023-2025) per-ticker ranking — does NVDA still lead?
  2. Multi-position portfolio: max 4 slots, 35% each, best-grade first
  3. Quality filter: which tickers to include/exclude
  4. What the arena prompt/config should say

Usage:
    uv run python -m backend.research.portfolio_analysis
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP  = "=" * 72
DASH = "─" * 72

# All tickers we want to evaluate
ALL_TICKERS = ["NVDA", "GOOGL", "MSFT", "META", "AAPL", "AMZN", "AMD", "TSLA", "CRM", "NFLX"]

STARTING_CAPITAL = 2_000.0
MAX_POSITIONS    = 4
POSITION_PCT     = 0.35    # 35% per position
STOP_PCT         = -5.0

# ---------------------------------------------------------------------------
# Price helpers (reuse from signal_backtest)
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import fetch_earnings_dates

_cache: dict[str, pd.DataFrame] = {}

def _prices(ticker: str) -> pd.DataFrame:
    if ticker not in _cache:
        df = yf.download(ticker, start="2013-01-01", end="2026-05-06",
                         interval="1d", progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
        _cache[ticker] = df
    return _cache[ticker]


def _nth(df, ref, offset):
    dt, direction, count = pd.Timestamp(ref), (1 if offset >= 0 else -1), 0
    for i in range(1, 300):
        cand = dt + pd.Timedelta(days=i * direction)
        if cand in df.index:
            count += 1
            if count == abs(offset):
                return cand
    return None


# ---------------------------------------------------------------------------
# Build trade list per ticker
# ---------------------------------------------------------------------------

def build_ticker_trades(ticker: str) -> list[dict]:
    dates = fetch_earnings_dates(ticker)
    df    = _prices(ticker)
    trades = []
    for ann in dates:
        e_dt = _nth(df, ann, -20)
        x_dt = _nth(df, ann, -1)
        if e_dt is None or x_dt is None:
            continue
        if e_dt not in df.index or x_dt not in df.index:
            continue
        ep   = float(df["Close"].loc[e_dt])
        xp   = float(df["Close"].loc[x_dt])
        days = [d for d in df.index if e_dt <= d <= x_dt]
        dlys = [float(df["Close"].loc[d]) for d in days]
        ret  = (xp - ep) / ep * 100
        trades.append({
            "ticker":   ticker,
            "ann":      ann,
            "year":     date.fromisoformat(ann).year,
            "entry_dt": e_dt,
            "exit_dt":  x_dt,
            "entry_px": ep,
            "exit_px":  xp,
            "ret_pct":  round(ret, 3),
            "min_ret":  round((min(dlys) - ep) / ep * 100, 3) if dlys else ret,
            "win":      ret > 0,
            "cal_days": (x_dt - e_dt).days,
        })
    return trades


# ---------------------------------------------------------------------------
# Ticker scorecard — all-time and recent (2023+)
# ---------------------------------------------------------------------------

def scorecard(trades: list[dict], label: str) -> dict:
    if not trades:
        return {"n": 0, "wr": 0, "avg": 0, "best": 0, "worst": 0, "score": 0}
    wins = [t for t in trades if t["win"]]
    rets = [t["ret_pct"] for t in trades]
    avg  = sum(rets) / len(rets)
    std  = math.sqrt(sum((r - avg)**2 for r in rets) / len(rets)) if len(rets) > 1 else 1
    # Score = win_rate * avg_ret / std  (risk-adjusted quality)
    score = (len(wins) / len(trades)) * avg / std if std else 0
    return {
        "n": len(trades), "wr": round(len(wins)/len(trades)*100, 1),
        "avg": round(avg, 2), "best": round(max(rets), 1),
        "worst": round(min(rets), 1), "score": round(score, 3),
    }


# ---------------------------------------------------------------------------
# Multi-position portfolio simulation
# ---------------------------------------------------------------------------

def simulate_portfolio(
    all_trades: list[dict],
    tickers: list[str],       # priority order — fill slots greedily
    capital: float = STARTING_CAPITAL,
    max_pos: int   = MAX_POSITIONS,
    pos_pct: float = POSITION_PCT,
    stop_pct: float = STOP_PCT,
) -> dict:
    """
    Realistic portfolio: up to max_pos concurrent positions.
    Entries are taken in ticker priority order when a slot is free.
    Each position uses pos_pct of current equity.
    """
    # Filter to allowed tickers and sort by entry date
    ticker_set = set(tickers)
    trades = sorted(
        [t for t in all_trades if t["ticker"] in ticker_set],
        key=lambda t: t["entry_dt"]
    )

    equity      = capital
    peak        = capital
    max_dd      = 0.0
    # Track open slots: list of (exit_dt, pnl_pct) for open positions
    open_pos: list[dict] = []
    trade_log   = []
    milestones  = {}
    wins = losses = 0
    total_ret = 0

    MILESTONES = [5_000, 10_000, 20_000, 50_000]

    for t in trades:
        # Close any positions that have exited by this trade's entry
        still_open = []
        for pos in open_pos:
            if pos["exit_dt"] <= t["entry_dt"]:
                # Position closed — realise P&L
                stop_triggered = pos["min_ret"] <= stop_pct
                actual_ret = stop_pct if stop_triggered else pos["ret_pct"]
                pnl = pos["invested"] * (actual_ret / 100)
                equity += pnl
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd
                total_ret += actual_ret
                if actual_ret > 0:
                    wins += 1
                else:
                    losses += 1
                trade_log.append({**pos["trade"], "actual_ret": round(actual_ret, 2),
                                   "equity_after": round(equity, 2), "stopped": stop_triggered})
            else:
                still_open.append(pos)
        open_pos = still_open

        # Check milestones
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = (t["entry_dt"].date(), round(equity, 2))

        # Open new position if slot available and ticker priority allows
        if len(open_pos) < max_pos:
            invested = equity * pos_pct
            open_pos.append({
                "exit_dt":  t["exit_dt"],
                "ret_pct":  t["ret_pct"],
                "min_ret":  t["min_ret"],
                "invested": invested,
                "trade":    t,
            })

    # Close remaining open positions at their exit price
    for pos in open_pos:
        stop_triggered = pos["min_ret"] <= stop_pct
        actual_ret = stop_pct if stop_triggered else pos["ret_pct"]
        pnl = pos["invested"] * (actual_ret / 100)
        equity += pnl
        if equity > peak:
            peak = equity
        total_ret += actual_ret
        if actual_ret > 0:
            wins += 1
        else:
            losses += 1
        trade_log.append({**pos["trade"], "actual_ret": round(actual_ret, 2),
                           "equity_after": round(equity, 2), "stopped": stop_triggered})

    for m in MILESTONES:
        if m not in milestones and equity >= m:
            milestones[m] = (trade_log[-1]["exit_dt"].date() if trade_log else None, round(equity, 2))

    n = wins + losses
    return {
        "final":    round(equity, 2),
        "return_x": round(equity / capital, 1),
        "n":        n,
        "wr":       round(wins / n * 100, 1) if n else 0,
        "avg_ret":  round(total_ret / n, 2) if n else 0,
        "max_dd":   round(max_dd, 1),
        "milestones": milestones,
        "log":      trade_log,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  PORTFOLIO ANALYSIS  —  aligning arena with research")
    print(SEP)

    print("\nLoading data...")
    all_trades_by_ticker: dict[str, list[dict]] = {}
    for t in ALL_TICKERS:
        _prices(t)
        all_trades_by_ticker[t] = build_ticker_trades(t)
        print(f"  {t}: {len(all_trades_by_ticker[t])} trades")

    all_trades = [t for trades in all_trades_by_ticker.values() for t in trades]

    # ── 1. Per-ticker scorecard: all-time vs recent ───────────────────────
    print(f"\n{SEP}")
    print("  SECTION 1: PER-TICKER RANKING")
    print("  All-time (2015+) vs Recent (2023+) — does ranking change?")
    print(SEP)
    print(f"\n  {'Ticker':6}  {'─── All-time ───':20}  {'─── 2023-2026 ──':20}  Trend")
    print(f"  {'':6}  {'N':>3} {'WR':>5} {'Avg':>6} {'Score':>6}  "
          f"{'N':>3} {'WR':>5} {'Avg':>6} {'Score':>6}")
    print(f"  {'─'*6}  {'─'*3} {'─'*5} {'─'*6} {'─'*6}  "
          f"{'─'*3} {'─'*5} {'─'*6} {'─'*6}  {'─'*8}")

    recent_scores = {}
    alltime_scores = {}
    for ticker in ALL_TICKERS:
        trades = all_trades_by_ticker[ticker]
        at = scorecard(trades, "all")
        rc = scorecard([t for t in trades if t["year"] >= 2023], "recent")
        recent_scores[ticker] = rc["score"]
        alltime_scores[ticker] = at["score"]
        trend = ("▲ rising" if rc["score"] > at["score"] * 1.1 else
                 "▼ fading" if rc["score"] < at["score"] * 0.9 else "= stable")
        print(f"  {ticker:6}  {at['n']:>3} {at['wr']:>4.0f}% {at['avg']:>+5.1f}% {at['score']:>6.3f}  "
              f"{rc['n']:>3} {rc['wr']:>4.0f}% {rc['avg']:>+5.1f}% {rc['score']:>6.3f}  {trend}")

    # Rank by recent score
    ranked_recent = sorted(ALL_TICKERS, key=lambda t: recent_scores[t], reverse=True)
    print(f"\n  Recent ranking: {' > '.join(ranked_recent)}")

    # ── 2. Quality cut — which tickers make the cut? ─────────────────────
    print(f"\n{SEP}")
    print("  SECTION 2: QUALITY FILTER")
    print("  Minimum bar: recent score > 0, win_rate >= 60%, avg_ret >= 1.5%")
    print(SEP)

    INCLUDE, EXCLUDE = [], []
    for ticker in ALL_TICKERS:
        trades = all_trades_by_ticker[ticker]
        rc = scorecard([t for t in trades if t["year"] >= 2023], "recent")
        at = scorecard(trades, "all")
        ok = rc["n"] >= 3 and rc["wr"] >= 60 and rc["avg"] >= 1.5 and recent_scores[ticker] > 0
        verdict = "INCLUDE" if ok else "EXCLUDE"
        reason  = ""
        if rc["n"] < 3:
            reason = f"only {rc['n']} recent trades"
        elif rc["wr"] < 60:
            reason = f"recent win rate {rc['wr']:.0f}% < 60%"
        elif rc["avg"] < 1.5:
            reason = f"recent avg {rc['avg']:+.1f}% < 1.5%"
        (INCLUDE if ok else EXCLUDE).append(ticker)
        flag = "✓" if ok else "✗"
        print(f"  {flag} {ticker:6}  recent: {rc['wr']:.0f}% win, {rc['avg']:+.1f}% avg  "
              f"→ {verdict}{' (' + reason + ')' if reason else ''}")

    print(f"\n  Include: {INCLUDE}")
    print(f"  Exclude: {EXCLUDE}")

    # ── 3. Portfolio simulation — different configs ───────────────────────
    print(f"\n{SEP}")
    print("  SECTION 3: PORTFOLIO SIMULATION  ($2,000 start, -5% stop)")
    print("  One slot per ticker at a time, best-grade fills first")
    print(SEP)

    configs = [
        ("All included tickers, priority by recent score", INCLUDE),
        ("Top 4 only (recent rank)",                       ranked_recent[:4]),
        ("Top 6 only (recent rank)",                       ranked_recent[:6]),
        ("NVDA + GOOGL + AMD + AMZN (stated recommendation)", ["NVDA","GOOGL","AMD","AMZN"]),
        ("NVDA only (baseline)",                           ["NVDA"]),
    ]

    # Sort by recent score within each config
    results = {}
    for label, tickers in configs:
        ordered = sorted(tickers, key=lambda t: recent_scores.get(t, 0), reverse=True)
        r = simulate_portfolio(all_trades, ordered, capital=STARTING_CAPITAL,
                               max_pos=MAX_POSITIONS, pos_pct=POSITION_PCT)
        results[label] = r

        t10k  = str(r["milestones"].get(10_000, (None,))[0] or "—")[:7]
        print(f"\n  {label}")
        print(f"  Tickers (priority): {ordered}")
        print(f"  {r['n']} trades  {r['wr']:.0f}% win  avg {r['avg_ret']:+.2f}%  "
              f"DD {r['max_dd']:.1f}%  $2k→${r['final']:,.0f} ({r['return_x']:.0f}x)  "
              f"$10k: {t10k}")

    # ── 4. Recent-year focus (2023+) ──────────────────────────────────────
    print(f"\n{SEP}")
    print("  SECTION 4: RECENT PERIOD (2023–2026) ONLY")
    print("  Out-of-sample validation — did the strategy hold up?")
    print(SEP)

    recent_trades = [t for t in all_trades if t["year"] >= 2023]
    rec_inc = sorted(INCLUDE, key=lambda t: recent_scores.get(t, 0), reverse=True)
    r_rec   = simulate_portfolio(recent_trades, rec_inc, capital=STARTING_CAPITAL,
                                  max_pos=MAX_POSITIONS, pos_pct=POSITION_PCT)
    print(f"\n  Quality tickers {rec_inc}, 2023-2026:")
    print(f"  {r_rec['n']} trades  {r_rec['wr']:.0f}% win  avg {r_rec['avg_ret']:+.2f}%/trade  "
          f"DD {r_rec['max_dd']:.1f}%")
    print(f"  $2,000 → ${r_rec['final']:,.2f}  ({r_rec['return_x']:.1f}x in ~3 years)")

    # Show each trade
    print(f"\n  Trade log (recent):")
    print(f"  {'Date':12} {'Ticker':6} {'Ret':>7}  {'Equity':>10}  note")
    for t in r_rec["log"]:
        flag = " STOP" if t["stopped"] else ""
        print(f"  {t['ann']:12} {t['ticker']:6} {t['actual_ret']:>+6.1f}%  "
              f"${t['equity_after']:>9,.2f}{flag}")

    # ── 5. Position sizing sensitivity ───────────────────────────────────
    print(f"\n{SEP}")
    print("  SECTION 5: POSITION SIZE SENSITIVITY  (quality tickers, all-time)")
    print(SEP)
    print(f"  {'MaxPos':>6} {'PctEach':>7} {'Final':>12} {'x':>5} {'DD':>6} {'WR':>5}")
    print(f"  {'─'*6} {'─'*7} {'─'*12} {'─'*5} {'─'*6} {'─'*5}")
    for max_p in [1, 2, 3, 4]:
        for pct in [0.25, 0.35, 0.50]:
            ordered = sorted(INCLUDE, key=lambda t: recent_scores.get(t, 0), reverse=True)
            r = simulate_portfolio(all_trades, ordered, capital=STARTING_CAPITAL,
                                   max_pos=max_p, pos_pct=pct)
            print(f"  {max_p:>6} {pct*100:>6.0f}%  ${r['final']:>11,.0f}  "
                  f"{r['return_x']:>4.0f}x  {r['max_dd']:>5.1f}%  {r['wr']:>4.0f}%")

    # ── 6. Arena config recommendation ───────────────────────────────────
    print(f"\n{SEP}")
    print("  SECTION 6: ARENA CONFIGURATION RECOMMENDATION")
    print(SEP)

    best_label = "All included tickers, priority by recent score"
    best = results[best_label]
    best_tickers = sorted(INCLUDE, key=lambda t: recent_scores.get(t, 0), reverse=True)

    print(f"""
  TICKER UNIVERSE (priority order):
    {best_tickers}

  ALLOCATION:
    Max concurrent positions: {MAX_POSITIONS}
    Capital per position:     {int(POSITION_PCT*100)}% of portfolio
    → With $1M arena account: ~$350,000 per position

  EXIT RULES:
    Stop-loss:   -5% from avg_cost (validated, applies to all tickers)
    Normal exit: 1 trading day before earnings_date (D-1)
    No profit targets (reduces alpha)

  SYSTEM PROMPT CHANGES NEEDED:
    1. Replace NVDA-first with recent ranking: {best_tickers[:4]}
    2. Specify max 4 concurrent positions, 35% each
    3. Add "skip if already in 4 positions" logic
    4. Mention TSLA/CRM/NFLX are excluded (poor recent signal)

  WHAT THE AGENTS WILL DO:
    - Cycle 1: get_state() → see 1-2 signals in D-20 window
    - Buy top-ranked signal at 35% of $1M = $350,000
    - Next cycle: check if more signals appeared, fill up to 4 slots
    - Monitor stops every cycle
    - Hold until D-1, then sell and wait for next signal

  REALISTIC GAME OUTCOME:
    12-minute game sees ~0.2% price moves on deployed positions
    P&L differentiation: which model sizes correctly, respects stops,
    and picks higher-ranked signals over lower-ranked ones
    """)


if __name__ == "__main__":
    main()

