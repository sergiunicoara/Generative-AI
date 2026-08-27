"""
Filter Backtest — tests additional entry filters on top of S2-Safer.

Filters tested (individually + combined):
  F1  Entry day green      close > prev_close on entry day
  F2  Above 20dma          stock above its 20-day MA at entry
  F3  Not overbought       RSI(14) < 70 at entry
  F4  Sector strong        XLK above its 150dma at entry
  F5  Relative strength    stock 20d return > QQQ 20d return (already in score, explicit here)
  F6  Gap up               entry day open > prev day close (stock gapping up)

Baseline: S2-Safer rules (QQQ>150dma, score>=1.05, D-20 window, 100% deploy, -5% stop, D-1 exit)

Usage:
    uv run python -m backend.research.filter_backtest
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP  = "=" * 72
DASH = "─" * 68
RESEARCH_DIR = Path(__file__).parent

TICKERS    = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
STOP_PCT   = -5.0
START_CASH = 2_000.0

BASE_QUALITY = {
    "GOOGL": 1.40, "NVDA": 1.50, "AMZN": 1.20,
    "MSFT":  1.10, "META": 1.10, "AMD":  1.00,
}
SCORE_THRESHOLD = 1.05

# ---------------------------------------------------------------------------
# Price cache
# ---------------------------------------------------------------------------

_cache: dict[str, pd.DataFrame] = {}

def _prices(ticker: str) -> pd.DataFrame:
    if ticker not in _cache:
        df = yf.download(ticker, start="2013-01-01", end="2026-05-17",
                         interval="1d", progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
        _cache[ticker] = df
    return _cache[ticker]

def _nth(df: pd.DataFrame, ref: str, offset: int) -> pd.Timestamp | None:
    dt, d, c = pd.Timestamp(ref), (1 if offset >= 0 else -1), 0
    for i in range(1, 300):
        cand = dt + pd.Timedelta(days=i*d)
        if cand in df.index:
            c += 1
            if c == abs(offset): return cand
    return None

# ---------------------------------------------------------------------------
# Technical indicators at a given date
# ---------------------------------------------------------------------------

def _rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff().dropna()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi   = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1]
    return float(v) if not math.isnan(v) else 50.0

def _bounded(x: float) -> float:
    return max(-20.0, min(20.0, x)) / 100.0

def _dynamic_score(ticker: str, entry_dt: pd.Timestamp,
                   qqq_df: pd.DataFrame) -> float:
    df  = _prices(ticker)
    col = df["Close"].loc[df.index <= entry_dt]
    if len(col) < 62: return BASE_QUALITY.get(ticker, 1.0)
    m20 = (col.iloc[-1] / col.iloc[-21] - 1) * 100
    m60 = (col.iloc[-1] / col.iloc[-62] - 1) * 100
    qcol = qqq_df["Close"].loc[qqq_df.index <= entry_dt]
    qqq20 = (qcol.iloc[-1] / qcol.iloc[-21] - 1) * 100 if len(qcol) >= 21 else 0
    rs20 = m20 - float(qqq20)
    base = BASE_QUALITY.get(ticker, 1.0)
    return base + 1.20*_bounded(m20) + 0.80*_bounded(m60) + 1.50*_bounded(rs20)

# ---------------------------------------------------------------------------
# Build baseline trade list (S2-Safer without extra filters)
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import fetch_earnings_dates

def build_base_trades(qqq_df: pd.DataFrame, xlk_df: pd.DataFrame) -> list[dict]:
    """Build all S2-Safer trades with full indicator data attached."""
    raw = []
    for ticker in TICKERS:
        df    = _prices(ticker)
        dates = fetch_earnings_dates(ticker)
        for ann in dates:
            e_dt = _nth(df, ann, -20)
            x_dt = _nth(df, ann, -1)
            if e_dt is None or x_dt is None: continue
            if e_dt not in df.index or x_dt not in df.index: continue

            ep = float(df["Close"].loc[e_dt])
            xp = float(df["Close"].loc[x_dt])

            # Stop check
            window  = [d for d in df.index if e_dt <= d <= x_dt]
            min_px  = min(float(df["Close"].loc[d]) for d in window)
            stopped = min_px <= ep * 0.95
            ret     = STOP_PCT if stopped else (xp - ep) / ep * 100

            # ── Regime (QQQ > 150dma) ──
            qcol = qqq_df["Close"].loc[qqq_df.index <= e_dt]
            if len(qcol) < 150: continue
            qqq_ma150 = float(qcol.rolling(150).mean().iloc[-1])
            qqq_cur   = float(qcol.iloc[-1])
            regime_ok = qqq_cur > qqq_ma150

            # ── Dynamic score ──
            score = _dynamic_score(ticker, e_dt, qqq_df)
            score_ok = score >= SCORE_THRESHOLD

            # ── F1: entry day green ──
            prev_idx = [d for d in df.index if d < e_dt]
            if not prev_idx: continue
            prev_close = float(df["Close"].loc[prev_idx[-1]])
            entry_close = float(df["Close"].loc[e_dt])
            f1_green = entry_close > prev_close

            # ── F2: above 20dma ──
            col20 = df["Close"].loc[df.index <= e_dt]
            ma20  = float(col20.tail(20).mean()) if len(col20) >= 20 else ep
            f2_above_ma20 = ep > ma20

            # ── F3: RSI < 70 ──
            col_rsi = df["Close"].loc[df.index <= e_dt].tail(30)
            rsi_val = _rsi(col_rsi)
            f3_rsi_ok = rsi_val < 70

            # ── F4: XLK > 150dma ──
            xcol = xlk_df["Close"].loc[xlk_df.index <= e_dt]
            if len(xcol) >= 150:
                xlk_ma150 = float(xcol.rolling(150).mean().iloc[-1])
                xlk_cur   = float(xcol.iloc[-1])
                f4_xlk_ok = xlk_cur > xlk_ma150
            else:
                f4_xlk_ok = True

            # ── F5: relative strength (stock > QQQ over 20d) ──
            qqq20d  = (qcol.iloc[-1] / qcol.iloc[-21] - 1)*100 if len(qcol) >= 21 else 0
            col_rs  = df["Close"].loc[df.index <= e_dt]
            stk20d  = (col_rs.iloc[-1] / col_rs.iloc[-21] - 1)*100 if len(col_rs) >= 21 else 0
            f5_rs_ok = stk20d > float(qqq20d)

            # ── F6: gap up (open > prev close) ──
            if "Open" in df.columns:
                entry_open = float(df["Open"].loc[e_dt])
                f6_gap_up = entry_open > prev_close
            else:
                f6_gap_up = True

            raw.append({
                "ticker":    ticker,
                "ann":       ann,
                "year":      int(ann[:4]),
                "entry_dt":  e_dt,
                "exit_dt":   x_dt,
                "entry_px":  ep,
                "exit_px":   xp,
                "ret":       round(ret, 3),
                "win":       ret > 0,
                "stopped":   stopped,
                "regime_ok": regime_ok,
                "score":     round(score, 3),
                "score_ok":  score_ok,
                "rsi":       round(rsi_val, 1),
                # filters
                "f1_green":       f1_green,
                "f2_above_ma20":  f2_above_ma20,
                "f3_rsi_ok":      f3_rsi_ok,
                "f4_xlk_ok":      f4_xlk_ok,
                "f5_rs_ok":       f5_rs_ok,
                "f6_gap_up":      f6_gap_up,
            })

    return sorted(raw, key=lambda t: t["entry_dt"])

# ---------------------------------------------------------------------------
# Simulate sequential (1 position, 100% deploy, -5% stop)
# ---------------------------------------------------------------------------

def simulate_sequential(trades: list[dict]) -> dict:
    """1 trade at a time — skip if dates overlap with previous trade."""
    filtered, busy_until = [], None
    for t in trades:
        if busy_until and t["entry_dt"] <= busy_until:
            continue
        filtered.append(t)
        busy_until = t["exit_dt"]

    equity, peak, max_dd = START_CASH, START_CASH, 0.0
    milestones = {}
    wins = losses = 0
    equity_log = []

    for t in filtered:
        for m in [5_000, 10_000, 20_000]:
            if m not in milestones and equity >= m:
                milestones[m] = t["entry_dt"].date()

        equity = equity * (1 + t["ret"] / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if t["ret"] > 0: wins += 1
        else: losses += 1
        equity_log.append((t["exit_dt"].date(), t["ticker"], t["ret"], equity))

    n   = wins + losses
    rets = [t["ret"] for t in filtered]
    return {
        "n": n, "wins": wins,
        "wr": round(wins/n*100, 1) if n else 0,
        "avg": round(sum(rets)/n, 2) if n else 0,
        "final": round(equity, 2),
        "x": round(equity/START_CASH, 1),
        "max_dd": round(max_dd, 1),
        "milestones": milestones,
        "log": equity_log,
    }

# ---------------------------------------------------------------------------
# Print one result row
# ---------------------------------------------------------------------------

def prow(label: str, r: dict, base: dict) -> None:
    d_wr  = r["wr"]  - base["wr"]
    d_avg = r["avg"] - base["avg"]
    d_fin = r["final"] - base["final"]
    ms10  = str(r["milestones"].get(10_000, "—"))[:7]
    flag  = " ◄ BETTER" if r["final"] > base["final"] and r["max_dd"] <= base["max_dd"] else ""
    print(f"  {label:<35} {r['n']:>4}  {r['wr']:>5.1f}% ({d_wr:>+4.1f})  "
          f"{r['avg']:>+5.2f}% ({d_avg:>+4.2f})  "
          f"DD {r['max_dd']:>4.1f}%  ${r['final']:>8,.0f}  {ms10}{flag}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEP)
    print("  FILTER BACKTEST — does VWAP / news-proxy / RSI improve S2-Safer?")
    print(SEP)

    print("\nLoading prices...")
    for t in TICKERS + ["QQQ", "XLK"]:
        _prices(t)
        print(f"  {t} ready")

    qqq_df = _prices("QQQ")
    xlk_df = _prices("XLK")

    print("\nBuilding trade list with indicators...")
    all_trades = build_base_trades(qqq_df, xlk_df)
    print(f"  {len(all_trades)} raw trades (all tickers, all years)")

    # Baseline: regime + score only (current implementation)
    base_trades = [t for t in all_trades if t["regime_ok"] and t["score_ok"]]
    base = simulate_sequential(base_trades)

    print(f"\n{SEP}")
    print(f"  BASELINE S2-Safer (regime + score ≥ 1.05)")
    print(f"  {base['n']} trades | {base['wr']:.1f}% win | avg {base['avg']:+.2f}% | "
          f"DD {base['max_dd']:.1f}% | ${base['final']:,.0f} ({base['x']:.1f}x)")
    print(SEP)

    print(f"\n  {'Filter':<35} {'N':>4}  {'Win%':>8}  {'Avg%':>10}  {'MaxDD':>7}  {'Final':>9}  '$10k'")
    print(f"  {'-'*35} {'-'*4}  {'-'*8}  {'-'*10}  {'-'*7}  {'-'*9}  {'-'*7}")

    # ── Individual filters ────────────────────────────────────────────────
    print(f"\n  — Individual filters (added to baseline) —")
    filters = {
        "F1 Entry day green":       lambda t: t["f1_green"],
        "F2 Stock above 20dma":     lambda t: t["f2_above_ma20"],
        "F3 RSI < 70":              lambda t: t["f3_rsi_ok"],
        "F4 XLK above 150dma":      lambda t: t["f4_xlk_ok"],
        "F5 Stock outperforms QQQ": lambda t: t["f5_rs_ok"],
        "F6 Gap up on entry day":   lambda t: t["f6_gap_up"],
    }
    individual_results = {}
    for name, fn in filters.items():
        trades_f = [t for t in base_trades if fn(t)]
        r = simulate_sequential(trades_f)
        individual_results[name] = r
        prow(name, r, base)

    # ── Combined pairs ────────────────────────────────────────────────────
    print(f"\n  — Combinations —")
    combos = [
        ("F1 + F2  (green + above MA20)",       lambda t: t["f1_green"] and t["f2_above_ma20"]),
        ("F1 + F3  (green + RSI<70)",            lambda t: t["f1_green"] and t["f3_rsi_ok"]),
        ("F2 + F3  (MA20 + RSI<70)",             lambda t: t["f2_above_ma20"] and t["f3_rsi_ok"]),
        ("F2 + F5  (MA20 + outperforms QQQ)",    lambda t: t["f2_above_ma20"] and t["f5_rs_ok"]),
        ("F1 + F2 + F3",                         lambda t: t["f1_green"] and t["f2_above_ma20"] and t["f3_rsi_ok"]),
        ("F1 + F2 + F5",                         lambda t: t["f1_green"] and t["f2_above_ma20"] and t["f5_rs_ok"]),
        ("F2 + F3 + F5",                         lambda t: t["f2_above_ma20"] and t["f3_rsi_ok"] and t["f5_rs_ok"]),
        ("F1 + F2 + F3 + F5  (all momentum)",   lambda t: t["f1_green"] and t["f2_above_ma20"] and t["f3_rsi_ok"] and t["f5_rs_ok"]),
        ("F4 + F2 + F3  (sector + MA + RSI)",    lambda t: t["f4_xlk_ok"] and t["f2_above_ma20"] and t["f3_rsi_ok"]),
        ("All filters",                           lambda t: all([t["f1_green"],t["f2_above_ma20"],t["f3_rsi_ok"],t["f4_xlk_ok"],t["f5_rs_ok"],t["f6_gap_up"]])),
    ]
    combo_results = {}
    for name, fn in combos:
        trades_f = [t for t in base_trades if fn(t)]
        if len(trades_f) < 5:
            print(f"  {name:<35} too few trades ({len(trades_f)})")
            continue
        r = simulate_sequential(trades_f)
        combo_results[name] = r
        prow(name, r, base)

    # ── SKIP analysis (what we miss by filtering) ─────────────────────────
    print(f"\n{SEP}")
    print("  TRADES FILTERED OUT (missed opportunity cost)")
    print(SEP)
    for name, fn in list(filters.items())[:3]:
        skipped  = [t for t in base_trades if not fn(t)]
        if not skipped: continue
        sk_wins  = sum(1 for t in skipped if t["win"])
        sk_avg   = sum(t["ret"] for t in skipped) / len(skipped)
        print(f"  {name:<35} skips {len(skipped):>3}  "
              f"({sk_wins/len(skipped)*100:.0f}% would have won  avg {sk_avg:+.2f}%)")

    # ── Best combination ──────────────────────────────────────────────────
    all_results = {**individual_results, **combo_results}
    if all_results:
        best_name = max(all_results,
                        key=lambda k: all_results[k]["final"] / max(all_results[k]["max_dd"], 1))
        best = all_results[best_name]
        print(f"\n{SEP}")
        print(f"  BEST RISK-ADJUSTED: {best_name}")
        print(SEP)
        print(f"  Trades: {best['n']}  Win: {best['wr']:.1f}%  Avg: {best['avg']:+.2f}%  "
              f"MaxDD: {best['max_dd']:.1f}%  Final: ${best['final']:,.0f}")
        print(f"  vs baseline: "
              f"Δfinal ${best['final']-base['final']:+,.0f}  "
              f"Δwin {best['wr']-base['wr']:+.1f}%  "
              f"Δdd {best['max_dd']-base['max_dd']:+.1f}%")
        print(f"\n  Year-by-year (best filter):")
        by_yr: dict[int, list] = {}
        for dt, t, r, eq in best["log"]:
            by_yr.setdefault(dt.year, []).append(r)
        for yr in sorted(by_yr):
            rets = by_yr[yr]
            avg  = sum(rets)/len(rets)
            w    = sum(1 for r in rets if r > 0)
            bar  = ("+" if avg>=0 else "-") + "█"*min(int(abs(avg)/3),20)
            print(f"    {yr}  {w}/{len(rets)} wins  avg {avg:+.1f}%  {bar}")

        print(f"\n  Milestones from $2,000:")
        for m in [5_000, 10_000, 20_000]:
            bm = best["milestones"].get(m)
            om = base["milestones"].get(m)
            print(f"    ${m:>6,}: best {str(bm)[:10] if bm else '—':>12}  "
                  f"baseline {str(om)[:10] if om else '—':>12}")


if __name__ == "__main__":
    main()

