"""
Gap Opening Pattern Analysis
==============================
Identifies and classifies all gap opens on our universe (2015-2026).
Finds which patterns are tradeable and what the edge looks like.

A gap open = today's open differs from yesterday's close by >= threshold.

Patterns tested:
  G1  Gap up + continue      buy open, hold to close
  G2  Gap up + fade          sell open (short), cover at close
  G3  Gap fill trade         buy open if gap up, exit when price touches prev close
  G4  Gap up in trend        gap up AND stock above 50dma
  G5  Gap up on volume       gap up AND entry volume > 20d avg
  G6  Gap up pre-earnings    gap up in D-20 pre-earnings window (our F6 filter)
  G7  Large gap (>3%)        gap up >3% — does it continue or reverse?
  G8  Small gap (0.5–1.5%)   small gap up — more likely to fill?
  G9  Monday gap             weekend gaps behave differently?
  G10 Gap in QQQ regime      gap up when QQQ > 150dma only

Uses daily OHLC (no intraday tick data needed):
  - Gap size  = (open - prev_close) / prev_close
  - Same-day return  = (close - open) / open   [continuation]
  - Gap fill flag    = close < prev_close for gap up  [full fill same day]
  - Overnight return = (next_open - close) / close

Usage:
    uv run python -m backend.research.gap_pattern_backtest
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP = "=" * 72
START_CASH  = 2_000.0
MIN_GAP_PCT = 0.50    # minimum gap to consider (%)

UNIVERSE = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]

# ---------------------------------------------------------------------------
# Price cache
# ---------------------------------------------------------------------------

_cache: dict[str, pd.DataFrame] = {}

def _px(ticker: str) -> pd.DataFrame:
    if ticker not in _cache:
        df = yf.download(ticker, start="2013-01-01", end="2026-05-17",
                         interval="1d", progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
        _cache[ticker] = df
    return _cache[ticker]

# ---------------------------------------------------------------------------
# Build gap event table
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import fetch_earnings_dates

def _nth(df, ref, offset):
    dt, d, c = pd.Timestamp(ref), (1 if offset >= 0 else -1), 0
    for i in range(1, 300):
        cand = dt + pd.Timedelta(days=i * d)
        if cand in df.index:
            c += 1
            if c == abs(offset): return cand
    return None

def _regime_ok(dt, qqq_df):
    qc = qqq_df["Close"].loc[qqq_df.index <= dt]
    return len(qc) >= 150 and float(qc.iloc[-1]) > float(qc.rolling(150).mean().iloc[-1])

def build_gap_table(qqq_df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per gap-open event across all tickers."""
    rows = []
    for ticker in UNIVERSE:
        df      = _px(ticker)
        edates  = fetch_earnings_dates(ticker)
        # Build earnings window set: all dates within D-20 to D-1
        earn_win: set[pd.Timestamp] = set()
        for ann in edates:
            e20 = _nth(df, ann, -20)
            e1  = _nth(df, ann, -1)
            if e20 and e1:
                for d in df.index:
                    if e20 <= d <= e1:
                        earn_win.add(d)
        # Earnings announcement days (day of or day after)
        earn_days: set[pd.Timestamp] = set()
        for ann in edates:
            ts = pd.Timestamp(ann)
            earn_days.add(ts)
            nxt = _nth(df, ann, 1)
            if nxt: earn_days.add(nxt)

        idx = df.index.tolist()
        for i in range(1, len(idx)):
            today = idx[i]
            prev  = idx[i - 1]
            if today.year < 2015: continue

            o  = float(df["Open"].loc[today])   if "Open"   in df.columns else None
            c  = float(df["Close"].loc[today])
            pc = float(df["Close"].loc[prev])
            h  = float(df["High"].loc[today])   if "High"   in df.columns else c
            lo = float(df["Low"].loc[today])    if "Low"    in df.columns else c
            v  = float(df["Volume"].loc[today]) if "Volume" in df.columns else 0

            if o is None or pc == 0: continue
            gap_pct = (o - pc) / pc * 100

            if abs(gap_pct) < MIN_GAP_PCT: continue   # ignore tiny gaps

            # Same-day metrics
            intra_ret   = (c - o)  / o  * 100          # open → close
            gap_filled  = (gap_pct > 0 and c <= pc)    # full fill (gap up, closes below prev)
            gap_partial = (gap_pct > 0 and c < o) or (gap_pct < 0 and c > o)  # partial

            # Volume context
            vol_series = df["Volume"].loc[df.index <= today].tail(21).iloc[:-1] if "Volume" in df.columns else None
            vol_ratio  = v / float(vol_series.mean()) if vol_series is not None and len(vol_series) >= 10 and float(vol_series.mean()) > 0 else 1.0

            # Trend context
            close_series = df["Close"].loc[df.index <= today]
            ma50  = float(close_series.tail(50).mean()) if len(close_series) >= 50 else o
            above_ma50 = o > ma50

            # Regime
            reg_ok = _regime_ok(today, qqq_df)

            # Pre-earnings window
            in_earn_win = today in earn_win
            is_earn_day = today in earn_days

            # Gap classification
            if   gap_pct >  5.0: size = "huge_up"
            elif gap_pct >  3.0: size = "large_up"
            elif gap_pct >  1.5: size = "medium_up"
            elif gap_pct >  0.5: size = "small_up"
            elif gap_pct < -5.0: size = "huge_dn"
            elif gap_pct < -3.0: size = "large_dn"
            elif gap_pct < -1.5: size = "medium_dn"
            else:                size = "small_dn"

            rows.append({
                "ticker":       ticker,
                "date":         today,
                "weekday":      today.day_name(),
                "prev_close":   round(pc, 4),
                "open":         round(o, 4),
                "high":         round(h, 4),
                "low":          round(lo, 4),
                "close":        round(c, 4),
                "gap_pct":      round(gap_pct, 3),
                "intra_ret":    round(intra_ret, 3),   # open→close same day
                "gap_filled":   gap_filled,
                "vol_ratio":    round(vol_ratio, 2),
                "above_ma50":   above_ma50,
                "regime_ok":    reg_ok,
                "in_earn_win":  in_earn_win,
                "is_earn_day":  is_earn_day,
                "size":         size,
            })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("date").reset_index(drop=True)
    return df_out


# ---------------------------------------------------------------------------
# Pattern analysis
# ---------------------------------------------------------------------------

def analyse_pattern(df: pd.DataFrame, mask: pd.Series, label: str,
                    base_n: int, direction: str = "up") -> dict:
    """Compute stats for a filtered set of gap events."""
    sub = df[mask]
    if len(sub) < 5:
        return {"label": label, "n": len(sub), "note": "too few"}

    # Trade: buy open, sell close (gap up) OR sell open, cover close (gap down)
    if direction == "up":
        trade_ret = sub["intra_ret"]   # long open→close
    else:
        trade_ret = -sub["intra_ret"]  # short open→close

    wr  = (trade_ret > 0).mean() * 100
    avg = trade_ret.mean()
    std = trade_ret.std()
    fill_rate = sub["gap_filled"].mean() * 100 if direction == "up" else float("nan")

    return {
        "label":      label,
        "n":          len(sub),
        "pct_of_all": round(len(sub) / base_n * 100, 1),
        "wr":         round(wr, 1),
        "avg_ret":    round(avg, 2),
        "std":        round(std, 2),
        "fill_rate":  round(fill_rate, 1),
        "sharpe":     round(avg / std, 3) if std > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Sequential simulation (buy open, sell close, one trade at a time per ticker)
# ---------------------------------------------------------------------------

def simulate_pattern(df: pd.DataFrame, mask: pd.Series,
                     direction: str = "up") -> dict:
    """Simulate trading pattern: $2,000 start, one trade per event,
    buy open sell close. No overlapping same-ticker positions."""
    sub = df[mask].sort_values("date")
    if len(sub) < 5:
        return {"n": 0, "final": START_CASH, "wr": 0, "avg": 0, "max_dd": 0}

    equity = START_CASH
    peak   = START_CASH
    max_dd = 0.0
    wins   = 0
    rets   = []

    for _, row in sub.iterrows():
        ret = row["intra_ret"] if direction == "up" else -row["intra_ret"]
        equity *= (1 + ret / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if ret > 0: wins += 1
        rets.append(ret)

    n = len(rets)
    return {
        "n":      n,
        "final":  round(equity, 2),
        "x":      round(equity / START_CASH, 1),
        "wr":     round(wins / n * 100, 1),
        "avg":    round(sum(rets) / n, 2),
        "max_dd": round(max_dd, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  GAP OPENING PATTERN ANALYSIS — 2015–2026")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ"]:
        _px(t)
        print(f"  {t} ready")
    qqq_df = _px("QQQ")

    print("\nBuilding gap event table...")
    gaps = build_gap_table(qqq_df)
    gap_up = gaps[gaps["gap_pct"] > 0]
    gap_dn = gaps[gaps["gap_pct"] < 0]
    print(f"  Total gap events: {len(gaps)}")
    print(f"  Gap ups:  {len(gap_up)}  ({len(gap_up)/len(gaps)*100:.0f}%)")
    print(f"  Gap downs:{len(gap_dn)}  ({len(gap_dn)/len(gaps)*100:.0f}%)")

    # ── Distribution by size ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  GAP SIZE DISTRIBUTION")
    print(SEP)
    for sz in ["small_up","medium_up","large_up","huge_up",
               "small_dn","medium_dn","large_dn","huge_dn"]:
        sub = gaps[gaps["size"] == sz]
        if len(sub) == 0: continue
        avg_intra = sub["intra_ret"].mean() if "up" in sz else -sub["intra_ret"].mean()
        fill      = sub["gap_filled"].mean()*100 if "up" in sz else float("nan")
        fill_s    = f"  fill {fill:.0f}%" if not math.isnan(fill) else ""
        print(f"  {sz:<12} {len(sub):>5} events  avg intraday {avg_intra:>+5.2f}%{fill_s}")

    # ── Fill rate analysis ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  GAP FILL ANALYSIS (gap ups — does price return to prev close same day?)")
    print(SEP)
    for label, mask in [
        ("All gap ups",              gaps["gap_pct"] > 0),
        ("Small gap up (0.5–1.5%)",  (gaps["gap_pct"] > 0.5)  & (gaps["gap_pct"] <= 1.5)),
        ("Medium gap up (1.5–3%)",   (gaps["gap_pct"] > 1.5)  & (gaps["gap_pct"] <= 3.0)),
        ("Large gap up (>3%)",       gaps["gap_pct"] > 3.0),
        ("Huge gap up (>5%)",        gaps["gap_pct"] > 5.0),
        ("Earnings day gap",         (gaps["gap_pct"] > 0) & gaps["is_earn_day"]),
        ("Non-earnings gap",         (gaps["gap_pct"] > 0) & ~gaps["is_earn_day"]),
        ("Pre-earnings window gap",  (gaps["gap_pct"] > 0) & gaps["in_earn_win"] & ~gaps["is_earn_day"]),
        ("Monday gap up",            (gaps["gap_pct"] > 0) & (gaps["weekday"] == "Monday")),
        ("Gap up above 50dma",       (gaps["gap_pct"] > 0) & gaps["above_ma50"]),
        ("Gap up below 50dma",       (gaps["gap_pct"] > 0) & ~gaps["above_ma50"]),
        ("Gap up in QQQ regime",     (gaps["gap_pct"] > 0) & gaps["regime_ok"]),
        ("High volume gap up",       (gaps["gap_pct"] > 0) & (gaps["vol_ratio"] > 1.5)),
    ]:
        sub = gaps[mask]
        if len(sub) < 5: continue
        fill = sub["gap_filled"].mean() * 100
        avg_intra = sub["intra_ret"].mean()
        wr_cont = (sub["intra_ret"] > 0).mean() * 100  # % that close above open
        print(f"  {label:<35} n={len(sub):>4}  fill {fill:>4.0f}%  "
              f"intraday {avg_intra:>+5.2f}%  cont {wr_cont:>4.0f}%")

    # ── Pattern win rates ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TRADEABLE PATTERNS — buy open sell close (gap up)")
    print(f"  {'Pattern':<40} {'N':>5}  {'Win%':>6}  {'Avg%':>7}  {'Sharpe':>7}  {'Fill%':>6}")
    print(f"  {'-'*40} {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*6}")
    print(SEP)

    patterns_up = [
        ("All gap ups",
            gaps["gap_pct"] > 0),
        ("Gap up in QQQ regime",
            (gaps["gap_pct"] > 0) & gaps["regime_ok"]),
        ("Gap up above 50dma",
            (gaps["gap_pct"] > 0) & gaps["above_ma50"]),
        ("Gap up in regime + above 50dma",
            (gaps["gap_pct"] > 0) & gaps["regime_ok"] & gaps["above_ma50"]),
        ("Small gap up (0.5–1.5%)",
            (gaps["gap_pct"].between(0.5, 1.5))),
        ("Medium gap up (1.5–3%)",
            (gaps["gap_pct"].between(1.5, 3.0))),
        ("Large gap up (>3%)",
            gaps["gap_pct"] > 3.0),
        ("Pre-earnings gap up",
            (gaps["gap_pct"] > 0) & gaps["in_earn_win"] & ~gaps["is_earn_day"]),
        ("Pre-earnings + regime",
            (gaps["gap_pct"] > 0) & gaps["in_earn_win"] & ~gaps["is_earn_day"] & gaps["regime_ok"]),
        ("Pre-earnings + regime + above 50dma",
            (gaps["gap_pct"] > 0) & gaps["in_earn_win"] & ~gaps["is_earn_day"] & gaps["regime_ok"] & gaps["above_ma50"]),
        ("Earnings day gap up",
            (gaps["gap_pct"] > 0) & gaps["is_earn_day"]),
        ("Monday gap up",
            (gaps["gap_pct"] > 0) & (gaps["weekday"] == "Monday")),
        ("High vol gap up (>1.5x avg)",
            (gaps["gap_pct"] > 0) & (gaps["vol_ratio"] > 1.5)),
        ("High vol + regime + above 50dma",
            (gaps["gap_pct"] > 0) & (gaps["vol_ratio"] > 1.5) & gaps["regime_ok"] & gaps["above_ma50"]),
        ("Large gap + regime + above 50dma",
            (gaps["gap_pct"] > 3.0) & gaps["regime_ok"] & gaps["above_ma50"]),
        ("Medium gap + regime + above 50dma",
            (gaps["gap_pct"].between(1.5, 3.0)) & gaps["regime_ok"] & gaps["above_ma50"]),
    ]

    base_n = len(gaps[gaps["gap_pct"] > 0])
    results = []
    for label, mask in patterns_up:
        r = analyse_pattern(gaps, mask, label, base_n, "up")
        results.append(r)
        if r.get("note"): continue
        fill_s = f"{r['fill_rate']:>5.0f}%" if not math.isnan(r["fill_rate"]) else "   —  "
        flag = "  ◄" if r["wr"] >= 55 and r["avg_ret"] >= 0.3 and r["n"] >= 30 else ""
        print(f"  {label:<40} {r['n']:>5}  {r['wr']:>5.1f}%  {r['avg_ret']:>+6.2f}%  "
              f"{r['sharpe']:>7.3f}  {fill_s}{flag}")

    # ── Fade patterns (sell open, cover close) ────────────────────────────
    print(f"\n{SEP}")
    print("  FADE PATTERNS — sell open cover close (gap up reversal)")
    print(f"  {'Pattern':<40} {'N':>5}  {'Win%':>6}  {'Avg%':>7}  {'Sharpe':>7}")
    print(f"  {'-'*40} {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}")
    print(SEP)

    fade_patterns = [
        ("Fade all gap ups",
            gaps["gap_pct"] > 0),
        ("Fade small gap up (0.5–1.5%)",
            gaps["gap_pct"].between(0.5, 1.5)),
        ("Fade medium gap up (1.5–3%)",
            gaps["gap_pct"].between(1.5, 3.0)),
        ("Fade large gap up (>3%)",
            gaps["gap_pct"] > 3.0),
        ("Fade earnings gap up",
            (gaps["gap_pct"] > 0) & gaps["is_earn_day"]),
        ("Fade gap up below 50dma",
            (gaps["gap_pct"] > 0) & ~gaps["above_ma50"]),
        ("Fade gap up regime off",
            (gaps["gap_pct"] > 0) & ~gaps["regime_ok"]),
    ]
    for label, mask in fade_patterns:
        r = analyse_pattern(gaps, mask, label, base_n, "down")
        if r.get("note"): continue
        flag = "  ◄" if r["wr"] >= 55 and r["avg_ret"] >= 0.3 and r["n"] >= 30 else ""
        print(f"  {label:<40} {r['n']:>5}  {r['wr']:>5.1f}%  {r['avg_ret']:>+6.2f}%  "
              f"{r['sharpe']:>7.3f}{flag}")

    # ── Gap down patterns ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  GAP DOWN — buy open (mean reversion)")
    print(f"  {'Pattern':<40} {'N':>5}  {'Win%':>6}  {'Avg%':>7}  {'Sharpe':>7}")
    print(f"  {'-'*40} {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}")
    print(SEP)

    gap_dn_patterns = [
        ("Buy all gap downs",
            gaps["gap_pct"] < 0),
        ("Buy small gap down (-1.5 to -0.5%)",
            gaps["gap_pct"].between(-1.5, -0.5)),
        ("Buy medium gap down (-3 to -1.5%)",
            gaps["gap_pct"].between(-3.0, -1.5)),
        ("Buy large gap down (<-3%)",
            gaps["gap_pct"] < -3.0),
        ("Buy gap down in regime",
            (gaps["gap_pct"] < 0) & gaps["regime_ok"]),
        ("Buy gap down above 50dma",
            (gaps["gap_pct"] < 0) & gaps["above_ma50"]),
        ("Buy gap dn regime + above 50dma",
            (gaps["gap_pct"] < 0) & gaps["regime_ok"] & gaps["above_ma50"]),
    ]
    base_dn = len(gaps[gaps["gap_pct"] < 0])
    for label, mask in gap_dn_patterns:
        r = analyse_pattern(gaps, mask, label, base_dn, "up")  # buy = long
        if r.get("note"): continue
        flag = "  ◄" if r["wr"] >= 55 and r["avg_ret"] >= 0.3 and r["n"] >= 30 else ""
        print(f"  {label:<40} {r['n']:>5}  {r['wr']:>5.1f}%  {r['avg_ret']:>+6.2f}%  "
              f"{r['sharpe']:>7.3f}{flag}")

    # ── Per-ticker gap stats ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PER-TICKER GAP UP STATS")
    print(SEP)
    print(f"  {'Ticker':<8} {'N gaps':>7}  {'Fill%':>6}  {'Cont%':>6}  {'Avg intra':>10}  {'Pre-earn gaps':>14}")
    print(f"  {'-'*8} {'-'*7}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*14}")
    for ticker in UNIVERSE:
        sub = gaps[(gaps["ticker"] == ticker) & (gaps["gap_pct"] > 0)]
        if len(sub) == 0: continue
        fill  = sub["gap_filled"].mean() * 100
        cont  = (sub["intra_ret"] > 0).mean() * 100
        avg_i = sub["intra_ret"].mean()
        pre_e = len(sub[sub["in_earn_win"] & ~sub["is_earn_day"]])
        print(f"  {ticker:<8} {len(sub):>7}  {fill:>5.0f}%  {cont:>5.0f}%  {avg_i:>+9.2f}%  {pre_e:>14}")

    # ── Best equity simulation ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  BEST PATTERNS — EQUITY SIMULATION ($2,000 start)")
    print(SEP)

    best_candidates = [
        ("G1 Buy all gap ups",
            gaps["gap_pct"] > 0, "up"),
        ("G2 Pre-earnings gap up (regime)",
            (gaps["gap_pct"] > 0) & gaps["in_earn_win"] & ~gaps["is_earn_day"] & gaps["regime_ok"], "up"),
        ("G3 Pre-earnings + regime + above 50dma",
            (gaps["gap_pct"] > 0) & gaps["in_earn_win"] & ~gaps["is_earn_day"] & gaps["regime_ok"] & gaps["above_ma50"], "up"),
        ("G4 Medium gap up + regime + above 50dma",
            (gaps["gap_pct"].between(1.5, 3.0)) & gaps["regime_ok"] & gaps["above_ma50"], "up"),
        ("G5 Large gap up + regime + above 50dma",
            (gaps["gap_pct"] > 3.0) & gaps["regime_ok"] & gaps["above_ma50"], "up"),
        ("G6 Fade gap up below 50dma",
            (gaps["gap_pct"] > 0) & ~gaps["above_ma50"], "down"),
        ("G7 Buy gap down regime + above 50dma",
            (gaps["gap_pct"] < 0) & gaps["regime_ok"] & gaps["above_ma50"], "up"),
    ]
    print(f"  {'Pattern':<42} {'N':>5}  {'Win%':>6}  {'Avg%':>7}  {'MaxDD':>6}  {'Final':>10}")
    print(f"  {'-'*42} {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*10}")
    for label, mask, direction in best_candidates:
        r = simulate_pattern(gaps, mask, direction)
        if r["n"] < 5: continue
        flag = "  ◄" if r["final"] > START_CASH * 2 and r["max_dd"] < 20 else ""
        print(f"  {label:<42} {r['n']:>5}  {r['wr']:>5.1f}%  {r['avg']:>+6.2f}%  "
              f"{r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}{flag}")

    # ── Summary findings ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  KEY FINDINGS")
    print(SEP)
    total_gap_up = len(gaps[gaps["gap_pct"] > 0])
    fill_all     = gaps[gaps["gap_pct"] > 0]["gap_filled"].mean() * 100
    fill_small   = gaps[gaps["gap_pct"].between(0.5, 1.5)]["gap_filled"].mean() * 100
    fill_large   = gaps[gaps["gap_pct"] > 3.0]["gap_filled"].mean() * 100
    print(f"  Total gap-up events (2015-2026): {total_gap_up}")
    print(f"  Overall gap fill rate:           {fill_all:.0f}%")
    print(f"  Small gap (0.5-1.5%) fill rate:  {fill_small:.0f}%")
    print(f"  Large gap (>3%) fill rate:       {fill_large:.0f}%")
    avg_freq = total_gap_up / (len(UNIVERSE) * 11)
    print(f"  Avg gap-up freq per ticker/yr:   {avg_freq:.0f} events")
    print(f"  Pre-earnings gap-ups (our F6):   {len(gaps[gaps['in_earn_win'] & (gaps['gap_pct'] > 0) & ~gaps['is_earn_day']])}")


if __name__ == "__main__":
    main()

