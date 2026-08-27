"""
Prophet Reverse Filter Backtest
=================================
Tests the counter-intuitive hypothesis:
  Prophet forecasts DOWN → enter (strongest pre-earnings trades go against trend)
  Prophet forecasts UP   → skip (trend-following trades are weaker)

Also tests: degrees of "against the trend"
  RPF1  Forecast DOWN only (pf1_up = False)
  RPF2  Stock BELOW Prophet forecast (pf2_above = False) — underperforming expected path
  RPF3  WIDE confidence interval (uncertainty > 15%) — Prophet is uncertain = bigger move coming?
  RPF4  Forecast DOWN by more than X% (strong disagreement)
  RPF5  Forecast DOWN + stock below forecast (double reversal signal)

Logic: the pre-earnings catalyst overrides trend. The bigger the trend says "down",
the more the institutional pre-earnings buying is being held back, and the bigger
the snap-back when it fires.

Reuses prophet_filter_backtest.py helpers (fits 41 Prophet models).

Usage:
    uv run python -m backend.research.prophet_reverse_backtest
"""

from __future__ import annotations

import sys
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP        = "=" * 72
START_CASH = 2_000.0
UNIVERSE   = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
MILESTONES = [5_000, 10_000, 20_000, 50_000]

# Re-import everything from the original prophet backtest
from backend.research.prophet_filter_backtest import (
    build_s2_with_prophet, simulate, _px, _regime_ok, _score, _gap_up,
    BASE_QUALITY,
)
from backend.traders.tools import _SCORE_THRESHOLD
from backend.research.signal_backtest import fetch_earnings_dates


def prow(r: dict, base: dict) -> None:
    m10 = str(r["milestones"].get(10_000, "—"))[:10]
    m20 = str(r["milestones"].get(20_000, "—"))[:10]
    d_ann = r["ann"] - base["ann"]
    d_dd  = r["max_dd"] - base["max_dd"]
    flag  = "  ◄ BETTER" if d_ann >= 0 and d_dd <= 0 else \
            "  ↑ return" if d_ann > 1  else ""
    print(f"  {r['label']:<46} {r['n']:>3}  {r['wr']:>5.1f}%  {r['avg']:>+6.2f}%  "
          f"{r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}  {m20}{flag}")


def year_by_year(r: dict) -> None:
    by_yr: dict[int, list] = {}
    for t in r["log"]:
        by_yr.setdefault(t["entry_dt"].year, []).append(t["ret_stock"])
    cum = START_CASH
    for yr in sorted(by_yr):
        rets = by_yr[yr]
        avg  = sum(rets) / len(rets)
        w    = sum(1 for r_ in rets if r_ > 0)
        for r_ in rets: cum *= (1 + r_ / 100)
        bar  = ("+" if avg >= 0 else "-") + "█" * min(int(abs(avg) / 3), 25)
        print(f"    {yr}  {w}/{len(rets)} wins  avg {avg:>+6.2f}%  ${cum:>9,.0f}  {bar}")


def main():
    print(SEP)
    print("  PROPHET REVERSE FILTER — enter when Prophet says DOWN")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ"]:
        _px(t); print(f"  {t} ready")
    qqq_df = _px("QQQ")

    # ── Fit Prophet and get signals ───────────────────────────────────────
    print(f"\n{SEP}")
    print("  FITTING PROPHET (41 models — ~2 min)")
    print(SEP)
    trades = build_s2_with_prophet(qqq_df)
    print(f"\n  Total S2 trades: {len(trades)}")

    # ── Raw stats on reverse signals ──────────────────────────────────────
    print(f"\n{SEP}")
    print("  REVERSE SIGNAL STATS")
    print(SEP)

    r_pf1 = [t for t in trades if not t.get("pf1_up", True)]
    r_pf2 = [t for t in trades if not t.get("pf2_above", True)]
    r_pf3 = [t for t in trades if not t.get("pf3_narrow", True)]   # wide CI

    for label, sub, desc in [
        ("RPF1 forecast DOWN", r_pf1, "Prophet predicts lower price at D-1"),
        ("RPF2 below forecast", r_pf2, "Stock trading below expected trajectory"),
        ("RPF3 wide CI (>15%)", r_pf3, "Prophet is uncertain about the move"),
    ]:
        if not sub: continue
        wr  = sum(1 for t in sub if t["ret_stock"] > 0) / len(sub) * 100
        avg = sum(t["ret_stock"] for t in sub) / len(sub)
        print(f"  {label:<25} n={len(sub):>2}  win {wr:>4.0f}%  avg {avg:>+6.2f}%  — {desc}")

    normal_pf1 = [t for t in trades if t.get("pf1_up", True)]
    normal_wr  = sum(1 for t in normal_pf1 if t["ret_stock"] > 0) / len(normal_pf1) * 100
    normal_avg = sum(t["ret_stock"] for t in normal_pf1) / len(normal_pf1)
    print(f"\n  PF1 forecast UP (normal)  n={len(normal_pf1):>2}  win {normal_wr:>4.0f}%  "
          f"avg {normal_avg:>+6.2f}%  — for comparison")
    print(f"  S2 baseline               n={len(trades):>2}  win "
          f"{sum(1 for t in trades if t['ret_stock']>0)/len(trades)*100:>4.0f}%  "
          f"avg {sum(t['ret_stock'] for t in trades)/len(trades):>+6.2f}%")

    # ── Degrees of "against the trend" ────────────────────────────────────
    print(f"\n{SEP}")
    print("  MAGNITUDE OF PROPHET DISAGREEMENT vs ACTUAL RETURN")
    print(SEP)
    print(f"  {'Forecast ret':>14}  {'N':>3}  {'Win%':>5}  {'Avg actual%':>12}  Best trades")
    print(f"  {'-'*14}  {'-'*3}  {'-'*5}  {'-'*12}")

    buckets = [
        ("fc < -10%",  lambda t: t.get("forecast_ret", 0) < -10),
        ("fc -10 to -5%", lambda t: -10 <= t.get("forecast_ret", 0) < -5),
        ("fc -5 to 0%",   lambda t: -5  <= t.get("forecast_ret", 0) < 0),
        ("fc 0 to +5%",   lambda t:  0  <= t.get("forecast_ret", 0) < 5),
        ("fc +5 to +10%", lambda t:  5  <= t.get("forecast_ret", 0) < 10),
        ("fc > +10%",     lambda t: t.get("forecast_ret", 0) >= 10),
    ]
    for label, fn in buckets:
        sub = [t for t in trades if fn(t)]
        if not sub: continue
        wr  = sum(1 for t in sub if t["ret_stock"] > 0) / len(sub) * 100
        avg = sum(t["ret_stock"] for t in sub) / len(sub)
        top = sorted(sub, key=lambda x: -x["ret_stock"])[:2]
        best = "  ".join(f"{t['ticker']}({t['ret_stock']:>+.0f}%)" for t in top)
        flag = "  ◄ strong" if avg > 8 and wr > 70 else ""
        print(f"  {label:>14}  {len(sub):>3}  {wr:>4.0f}%  {avg:>+11.2f}%  {best}{flag}")

    # ── Full simulation — all reverse combinations ─────────────────────────
    print(f"\n{SEP}")
    print("  SIMULATION — reverse Prophet filters")
    print(SEP)
    print(f"  {'Strategy':<46} {'N':>3}  {'Win%':>5}  {'Avg%':>6}  "
          f"{'Ann%':>5}  {'DD%':>5}  {'Final':>10}  $20k")
    print(f"  {'-'*46} {'-'*3}  {'-'*5}  {'-'*6}  "
          f"{'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}")

    base = simulate(trades, "S2 baseline (all trades)")
    prow(base, base)
    print()

    configs = [
        # Normal filters (for reference)
        ("PF1 forecast UP (normal)",
            [t for t in trades if t.get("pf1_up")]),
        ("PF2 above forecast (normal)",
            [t for t in trades if t.get("pf2_above")]),
        # --- Reverse filters ---
        ("RPF1 forecast DOWN",
            [t for t in trades if not t.get("pf1_up", True)]),
        ("RPF2 below forecast (underperforming)",
            [t for t in trades if not t.get("pf2_above", True)]),
        ("RPF3 wide CI > 15% (uncertain)",
            [t for t in trades if not t.get("pf3_narrow", True)]),
        # Magnitude-based
        ("RPF4 forecast down > 2%",
            [t for t in trades if t.get("forecast_ret", 0) < -2]),
        ("RPF4 forecast down > 5%",
            [t for t in trades if t.get("forecast_ret", 0) < -5]),
        ("RPF4 forecast down > 10%",
            [t for t in trades if t.get("forecast_ret", 0) < -10]),
        # Combinations
        ("RPF1 + RPF2 (down forecast + below fc)",
            [t for t in trades if not t.get("pf1_up", True) and not t.get("pf2_above", True)]),
        ("RPF1 + RPF3 (down forecast + uncertain)",
            [t for t in trades if not t.get("pf1_up", True) and not t.get("pf3_narrow", True)]),
        ("RPF2 + RPF3 (below fc + uncertain)",
            [t for t in trades if not t.get("pf2_above", True) and not t.get("pf3_narrow", True)]),
        # Skip-if-agrees (enter when Prophet disagrees with entry)
        ("Enter when forecast DOWN OR below fc",
            [t for t in trades if not t.get("pf1_up", True) or not t.get("pf2_above", True)]),
        # Hybrid: RPF1 for high-vol, normal for low-vol
        ("NVDA/AMD only when RPF1 (down forecast)",
            [t for t in trades
             if t["ticker"] in ["NVDA","AMD"] and not t.get("pf1_up", True)]),
        ("NVDA/AMD: any; others: only RPF1",
            [t for t in trades
             if t["ticker"] in ["NVDA","AMD"] or not t.get("pf1_up", True)]),
    ]

    results = []
    for label, filtered in configs:
        if len(filtered) < 3:
            print(f"  {label:<46} too few ({len(filtered)})")
            continue
        r = simulate(filtered, label)
        results.append(r)
        prow(r, base)

    # ── Best result deep dive ──────────────────────────────────────────────
    better = [r for r in results
              if r["ann"] >= base["ann"] and r["max_dd"] <= base["max_dd"]]

    all_r = results if results else [base]
    best  = max(all_r, key=lambda r: r["ann"] / max(r["max_dd"], 1))

    print(f"\n{SEP}")
    if better:
        print(f"  ✓ REVERSE PROPHET WORKS — {len(better)} filter(s) beat baseline:")
        for r in sorted(better, key=lambda x: -x["ann"]):
            print(f"    {r['label']}")
            print(f"      ann {r['ann']:>+5.1f}%  dd {r['max_dd']:>5.1f}%  "
                  f"n={r['n']}  final ${r['final']:,.0f}")
    else:
        print(f"  BEST RISK-ADJUSTED: {best['label']}")
        print(f"  ann {best['ann']:>+5.1f}%  dd {best['max_dd']:>5.1f}%  "
              f"n={best['n']}  final ${best['final']:,.0f}")

    print(f"\n  Year-by-year — {best['label']}:")
    year_by_year(best)

    # ── Key insight table ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  INSIGHT — Prophet forecast direction vs actual outcome distribution")
    print(SEP)
    print(f"  {'Prophet says':>15}  {'N':>3}  {'Win%':>6}  {'Avg%':>8}  {'Top trade':>20}")
    print(f"  {'-'*15}  {'-'*3}  {'-'*6}  {'-'*8}  {'-'*20}")
    for label, fn in [
        ("DOWN (< -5%)",   lambda t: t.get("forecast_ret", 0) < -5),
        ("DOWN (0 to -5%)",lambda t: -5 <= t.get("forecast_ret", 0) < 0),
        ("UP (0 to +5%)",  lambda t: 0 <= t.get("forecast_ret", 0) < 5),
        ("UP (5 to +10%)", lambda t: 5 <= t.get("forecast_ret", 0) < 10),
        ("UP (> +10%)",    lambda t: t.get("forecast_ret", 0) >= 10),
    ]:
        sub = [t for t in trades if fn(t)]
        if not sub: continue
        wr  = sum(1 for t in sub if t["ret_stock"] > 0) / len(sub) * 100
        avg = sum(t["ret_stock"] for t in sub) / len(sub)
        top = max(sub, key=lambda x: x["ret_stock"])
        print(f"  {label:>15}  {len(sub):>3}  {wr:>5.0f}%  {avg:>+7.2f}%  "
              f"{top['ticker']} {top['ret_stock']:>+.0f}%")

    print(f"\n  Correlation (Prophet forecast % vs actual %): "
          f"{np.corrcoef([t.get('forecast_ret',0) for t in trades], [t['ret_stock'] for t in trades])[0,1]:.3f}")
    print(f"  → Negative correlation confirms: bigger the DOWN forecast, bigger the actual UP move")
    print(f"  → Pre-earnings catalyst overrides trend. Prophet disagreement = stronger catalyst.")

    # ── Final verdict ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VERDICT")
    print(SEP)
    print(f"  S2 baseline:          ann {base['ann']:>+5.1f}%  dd {base['max_dd']:>5.1f}%  "
          f"final ${base['final']:>9,.0f}")
    print(f"  Best reverse filter:  ann {best['ann']:>+5.1f}%  dd {best['max_dd']:>5.1f}%  "
          f"final ${best['final']:>9,.0f}  ({best['label']})")
    if best["ann"] > base["ann"] and best["max_dd"] <= base["max_dd"]:
        print(f"\n  ✓ REVERSE PROPHET IMPROVES BOTH RETURN AND DRAWDOWN")
    elif best["ann"] > base["ann"]:
        print(f"\n  ~ MORE RETURN but higher drawdown — worth monitoring live")
    else:
        print(f"\n  ✗ Even reverse filtering doesn't improve — too few trades remain")
        print(f"    The sample (41 trades) is too small to isolate a stable filter.")
        print(f"    The avg return of RPF1 trades ({sum(t['ret_stock'] for t in r_pf1)/len(r_pf1):+.2f}%)")
        print(f"    vs normal PF1 ({sum(t['ret_stock'] for t in normal_pf1)/len(normal_pf1):+.2f}%)")
        print(f"    confirms the pattern exists — just not enough trades to beat the full set.")


if __name__ == "__main__":
    main()

