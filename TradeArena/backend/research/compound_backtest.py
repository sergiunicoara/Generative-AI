"""
Compounding Backtest — $2,000 starting account
================================================
Tests every viable strategy from a $2,000 account and shows:
  - Compounded equity curve year by year
  - Time to reach $5k, $10k, $20k, $50k, $100k, $500k
  - Max drawdown and recovery
  - Expected calendar time per trade

Strategies tested:
  A. NVDA pre-earnings only (Grade A, best signal)
  B. NVDA + GOOGL (Grade A + B, diversified)
  C. All 6 tickers (NVDA/GOOGL/MSFT/META/AAPL/CRM)
  D. Grade A+B only (NVDA/GOOGL/MSFT/META)
  E. With -5% stop (validated)
  F. Without stop (baseline)

Usage:
    uv run python -m backend.research.compound_backtest
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP  = "=" * 72
DASH = "─" * 72
RESEARCH_DIR = Path(__file__).parent

STARTING_CAPITAL = 2_000.0
ENTRY_OFFSET     = -20
EXIT_OFFSET      = -1
STOP_PCT         = -5.0      # validated stop
MILESTONES       = [5_000, 10_000, 20_000, 50_000, 100_000, 500_000, 1_000_000]

# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

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


def _nth(df: pd.DataFrame, ref: str, offset: int) -> pd.Timestamp | None:
    dt, direction, count = pd.Timestamp(ref), (1 if offset >= 0 else -1), 0
    for i in range(1, 300):
        cand = dt + pd.Timedelta(days=i * direction)
        if cand in df.index:
            count += 1
            if count == abs(offset):
                return cand
    return None


# ---------------------------------------------------------------------------
# Build trade list
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import fetch_earnings_dates, TICKERS

TICKER_GRADES = {
    "NVDA": "A", "GOOGL": "B", "MSFT": "B", "META": "B",
    "AAPL": "C", "CRM": "C",
}


def build_trades(tickers: list[str]) -> list[dict]:
    """All historical pre-earnings trades for given tickers, sorted by entry date."""
    trades = []
    for ticker in tickers:
        dates = fetch_earnings_dates(ticker)
        df    = _prices(ticker)
        for ann in dates:
            entry_dt = _nth(df, ann, ENTRY_OFFSET)
            exit_dt  = _nth(df, ann, EXIT_OFFSET)
            if entry_dt is None or exit_dt is None:
                continue
            if entry_dt not in df.index or exit_dt not in df.index:
                continue
            ep = float(df["Close"].loc[entry_dt])
            xp = float(df["Close"].loc[exit_dt])

            # Intra-window daily closes for stop simulation
            days   = [d for d in df.index if entry_dt <= d <= exit_dt]
            daily  = [float(df["Close"].loc[d]) for d in days]
            min_px = min(daily) if daily else ep

            trades.append({
                "ticker":    ticker,
                "grade":     TICKER_GRADES.get(ticker, "C"),
                "ann":       ann,
                "entry_dt":  entry_dt,
                "exit_dt":   exit_dt,
                "entry_px":  ep,
                "exit_px":   xp,
                "ret_pct":   (xp - ep) / ep * 100,
                "min_px":    min_px,
                "min_ret":   (min_px - ep) / ep * 100,
                "hold_days": len(days),
                "calendar_days": (exit_dt - entry_dt).days,
            })

    return sorted(trades, key=lambda t: t["entry_dt"])


# ---------------------------------------------------------------------------
# Simulate compounding
# ---------------------------------------------------------------------------

def simulate(
    trades: list[dict],
    capital:    float = STARTING_CAPITAL,
    deploy_pct: float = 0.90,   # fraction of capital per trade
    stop_pct:   float | None = STOP_PCT,
    max_concurrent: int = 1,     # simultaneous positions (1 = sequential)
    grade_filter: set[str] | None = None,
) -> dict:
    """
    Simulate compounding from `capital` through every trade in order.
    Returns equity curve, milestones hit, drawdown stats.
    """
    if grade_filter:
        trades = [t for t in trades if t["grade"] in grade_filter]

    equity      = capital
    peak        = capital
    max_dd      = 0.0
    dd_start    = None
    dd_end      = None
    all_equity  = [(None, capital)]   # (date, equity)
    milestones_hit: dict[int, tuple] = {}
    wins = losses = 0
    total_ret    = 0.0
    total_cal_days = 0
    trade_log    = []

    for t in trades:
        if equity <= 0:
            break

        # Check milestones before trade
        for m in MILESTONES:
            if m not in milestones_hit and equity >= m:
                milestones_hit[m] = (t["entry_dt"].date(), equity)

        position_size = equity * deploy_pct
        shares        = position_size / t["entry_px"]

        # Apply stop: did price hit stop intra-window?
        stop_price = t["entry_px"] * (1 + (stop_pct or -999) / 100)
        if stop_pct and t["min_px"] <= stop_price:
            actual_ret = stop_pct
        else:
            actual_ret = t["ret_pct"]

        pnl    = shares * t["entry_px"] * (actual_ret / 100)
        equity = equity + pnl  # uninvested portion unchanged, invested portion gains/loses
        # More precisely: equity = (1-deploy_pct)*equity_before + deploy_pct*equity_before*(1+ret/100)
        equity_before = (equity - pnl)
        equity = equity_before * (1 - deploy_pct) + equity_before * deploy_pct * (1 + actual_ret / 100)

        # Track drawdown
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd   = dd
            dd_start = t["entry_dt"].date()

        total_ret     += actual_ret
        total_cal_days += t["calendar_days"]
        if actual_ret > 0:
            wins += 1
        else:
            losses += 1

        all_equity.append((t["exit_dt"].date(), equity))
        trade_log.append({
            **t,
            "actual_ret": round(actual_ret, 2),
            "equity_after": round(equity, 2),
            "stopped": stop_pct is not None and t["min_px"] <= stop_price,
        })

    # Check milestones at end
    for m in MILESTONES:
        if m not in milestones_hit and equity >= m:
            milestones_hit[m] = (trade_log[-1]["exit_dt"].date() if trade_log else None, equity)

    n = wins + losses
    avg_cal_days = total_cal_days / n if n else 0

    return {
        "final_equity":    round(equity, 2),
        "total_return_x":  round(equity / capital, 1),
        "n_trades":        n,
        "win_rate":        round(wins / n * 100, 1) if n else 0,
        "avg_ret":         round(total_ret / n, 2) if n else 0,
        "max_drawdown":    round(max_dd, 1),
        "avg_hold_days":   round(avg_cal_days, 0),
        "milestones":      milestones_hit,
        "equity_curve":    all_equity,
        "trade_log":       trade_log,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def milestone_str(milestones: dict, capital: float) -> str:
    parts = []
    for m in MILESTONES:
        if m in milestones:
            dt, eq = milestones[m]
            parts.append(f"${m//1000}k → {dt}")
        else:
            parts.append(f"${m//1000}k → not reached")
    return "  " + "\n  ".join(parts)


def years_from_start(start_date: date, end_date: date) -> float:
    return (end_date - start_date).days / 365.25


def print_result(label: str, r: dict, start_date: date) -> None:
    print(f"\n  {label}")
    print(f"  {'─'*60}")
    print(f"  Trades: {r['n_trades']}  Win: {r['win_rate']:.0f}%  "
          f"Avg ret/trade: {r['avg_ret']:+.2f}%  Avg hold: {r['avg_hold_days']:.0f} days")
    print(f"  Final equity:  ${r['final_equity']:>12,.2f}  ({r['total_return_x']:.0f}x)")
    print(f"  Max drawdown:  {r['max_drawdown']:.1f}%")
    print(f"  Time to milestones:")
    for m in MILESTONES:
        if m in r["milestones"]:
            dt, eq = r["milestones"][m]
            yrs = years_from_start(start_date, dt)
            print(f"    ${m:>9,}  on {dt}  ({yrs:.1f} yrs from start)")
        else:
            print(f"    ${m:>9,}  not reached")


def print_equity_by_year(r: dict) -> None:
    by_year: dict[int, float] = {}
    for dt, eq in r["equity_curve"]:
        if dt:
            by_year[dt.year] = eq
    print(f"  Year-end equity:")
    prev = STARTING_CAPITAL
    for y in sorted(by_year):
        eq   = by_year[y]
        gain = (eq - prev) / prev * 100
        bar  = ("+" if gain >= 0 else "-") + "█" * min(int(abs(gain) / 5), 30)
        print(f"    {y}: ${eq:>12,.0f}  ({gain:>+6.1f}% that year)  {bar}")
        prev = eq


def print_worst_trades(r: dict, n: int = 5) -> None:
    worst = sorted(r["trade_log"], key=lambda t: t["actual_ret"])[:n]
    print(f"  Worst {n} trades:")
    for t in worst:
        flag = " STOP" if t["stopped"] else ""
        print(f"    {t['ticker']} {t['ann']}  {t['actual_ret']:>+6.1f}%{flag}  "
              f"equity after: ${t['equity_after']:,.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEP)
    print(f"  COMPOUNDING BACKTEST  —  starting capital ${STARTING_CAPITAL:,.0f}")
    print(SEP)

    print("\nLoading price data...")
    for t in TICKERS:
        _prices(t)
        print(f"  {t} ready")

    print("\nBuilding trade lists...")
    all_trades   = build_trades(TICKERS)
    nvda_trades  = build_trades(["NVDA"])
    ab_trades    = build_trades(["NVDA", "GOOGL", "MSFT", "META"])
    nvda_g_trades= build_trades(["NVDA", "GOOGL"])

    # Find first trade date
    first_date = min(t["entry_dt"].date() for t in all_trades)
    print(f"  {len(all_trades)} total trades from {first_date}")

    # ── Strategy comparison ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  STRATEGY COMPARISON  (90% capital deployed per trade, -5% stop)")
    print(SEP)

    configs = [
        ("NVDA only (Grade A)",              nvda_trades,   {"stop_pct": STOP_PCT}),
        ("NVDA only (no stop)",              nvda_trades,   {"stop_pct": None}),
        ("NVDA + GOOGL (Grade A+B top-2)",   nvda_g_trades, {"stop_pct": STOP_PCT}),
        ("Grade A+B  (NVDA/GOOGL/MSFT/META)",ab_trades,     {"stop_pct": STOP_PCT}),
        ("All 6 tickers",                    all_trades,    {"stop_pct": STOP_PCT}),
    ]

    results = {}
    for label, trades, kwargs in configs:
        r = simulate(trades, capital=STARTING_CAPITAL, deploy_pct=0.90, **kwargs)
        results[label] = r
        print_result(label, r, first_date)

    # ── Deep dive: best strategy ──────────────────────────────────────────
    best_label = "NVDA only (Grade A)"
    best = results[best_label]

    print(f"\n{SEP}")
    print(f"  DEEP DIVE: {best_label}")
    print(SEP)
    print_equity_by_year(best)
    print()
    print_worst_trades(best, n=8)

    # ── Deploy % sensitivity ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  DEPLOY % SENSITIVITY  (NVDA only, -5% stop)")
    print("  How aggressive should the position sizing be?")
    print(SEP)
    print(f"  {'Deploy':>8} {'Final $':>14} {'Return':>8} {'MaxDD':>7} {'Trades':>7}  Time to $10k")
    print(f"  {'─'*8} {'─'*14} {'─'*8} {'─'*7} {'─'*7}  {'─'*12}")
    for pct in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]:
        r  = simulate(nvda_trades, capital=STARTING_CAPITAL, deploy_pct=pct, stop_pct=STOP_PCT)
        t10 = r["milestones"].get(10_000, (None,))[0]
        t10_str = str(t10) if t10 else "not reached"
        print(f"  {pct*100:>7.0f}%  ${r['final_equity']:>13,.0f}  "
              f"{r['total_return_x']:>6.0f}x  {r['max_drawdown']:>5.1f}%  "
              f"{r['n_trades']:>6}  {t10_str}")

    # ── Realistic calendar time ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("  REALISTIC CALENDAR TIME  (NVDA only, 90% deploy, -5% stop)")
    print("  How long between trades?")
    print(SEP)

    gaps = []
    for i in range(1, len(nvda_trades)):
        gap = (nvda_trades[i]["entry_dt"] - nvda_trades[i-1]["exit_dt"]).days
        gaps.append(gap)

    avg_gap  = sum(gaps) / len(gaps) if gaps else 0
    avg_hold = sum(t["calendar_days"] for t in nvda_trades) / len(nvda_trades)
    trades_yr= 365 / (avg_hold + avg_gap) if (avg_hold + avg_gap) else 0

    print(f"  Average hold period:   {avg_hold:.0f} calendar days (~{avg_hold/5:.0f} trading weeks)")
    print(f"  Average gap between:   {avg_gap:.0f} days")
    print(f"  Trades per year:       ~{trades_yr:.1f}")
    print(f"  Total NVDA trades:     {len(nvda_trades)} over {years_from_start(nvda_trades[0]['entry_dt'].date(), nvda_trades[-1]['exit_dt'].date()):.0f} years")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SUMMARY  —  from ${STARTING_CAPITAL:,.0f} starting capital")
    print(SEP)
    print(f"  {'Strategy':<35} {'Final':>12} {'x':>6} {'DD':>6}  10k       100k")
    print(f"  {'─'*35} {'─'*12} {'─'*6} {'─'*6}  {'─'*10}  {'─'*10}")
    for label, _, kwargs in configs:
        r    = results[label]
        t10  = str(r["milestones"].get(10_000,  (None,))[0] or "—")[:7]
        t100 = str(r["milestones"].get(100_000, (None,))[0] or "—")[:7]
        print(f"  {label:<35} ${r['final_equity']:>11,.0f}  "
              f"{r['total_return_x']:>5.0f}x  {r['max_drawdown']:>4.1f}%  "
              f"{t10:<10}  {t100}")

    # Save results
    out = {
        "starting_capital": STARTING_CAPITAL,
        "stop_pct": STOP_PCT,
        "strategies": {
            label: {
                "final_equity": r["final_equity"],
                "total_return_x": r["total_return_x"],
                "n_trades": r["n_trades"],
                "win_rate": r["win_rate"],
                "max_drawdown": r["max_drawdown"],
                "milestones": {
                    str(m): {"date": str(v[0]), "equity": round(v[1], 2)}
                    for m, v in r["milestones"].items()
                },
            }
            for label, r in results.items()
        },
    }
    out_path = RESEARCH_DIR / "compound_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()

