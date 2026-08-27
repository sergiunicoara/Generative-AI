"""
Pre-Earnings Overnight Strategy — Full Refinement
===================================================
Refines the overnight edge found in gap_predict_backtest.py:

  Base finding: Pre-earnings + strong green + regime
  Win rate 61%, +0.18% avg, -18.7% DD, $2k → $9,242

Questions answered here:
  1. Which days in the window are best? (D-20…D-1 breakdown)
  2. Does the dynamic score filter (>=1.20) improve overnight results?
  3. How many nights in a row can we enter per earnings cycle?
  4. Is the edge concentrated in NVDA/AMD or all 6 tickers?
  5. What is the optimal entry window (D-X to D-Y)?
  6. Volume confirmation — does it help?
  7. Close-to-high quality — does it help?
  8. What happens if we hold through the gap (enter close, exit at intraday high)?
  9. Final standalone system: rules, stats, year-by-year

Trade structure (unchanged):
  Entry : buy at CLOSE on day T
  Exit  : sell at OPEN on day T+1
  Return: (open[T+1] - close[T]) / close[T]

Usage:
    uv run python -m backend.research.overnight_earnings_backtest
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict

import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP        = "=" * 72
START_CASH = 2_000.0
UNIVERSE   = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
MILESTONES = [5_000, 10_000, 20_000, 50_000]

BASE_QUALITY = {
    "GOOGL": 1.40, "NVDA": 1.50, "AMZN": 1.20,
    "MSFT":  1.10, "META": 1.10, "AMD":  1.00,
}

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

def _nth(df, ref, offset):
    dt, d, c = pd.Timestamp(ref), (1 if offset >= 0 else -1), 0
    for i in range(1, 300):
        cand = dt + pd.Timedelta(days=i * d)
        if cand in df.index:
            c += 1
            if c == abs(offset): return cand
    return None

def _bounded(x): return max(-20.0, min(20.0, x)) / 100.0

def _dynamic_score(ticker, entry_dt, qqq_df):
    df  = _px(ticker)
    col = df["Close"].loc[df.index <= entry_dt]
    if len(col) < 62: return BASE_QUALITY.get(ticker, 1.0)
    m20 = (col.iloc[-1] / col.iloc[-21] - 1) * 100
    m60 = (col.iloc[-1] / col.iloc[-62] - 1) * 100
    qc  = qqq_df["Close"].loc[qqq_df.index <= entry_dt]
    q20 = (qc.iloc[-1] / qc.iloc[-21] - 1) * 100 if len(qc) >= 21 else 0
    rs  = m20 - float(q20)
    return (BASE_QUALITY.get(ticker, 1.0)
            + 1.20 * _bounded(m20)
            + 0.80 * _bounded(m60)
            + 1.50 * _bounded(rs))

def _regime_ok(dt, qqq_df):
    qc = qqq_df["Close"].loc[qqq_df.index <= dt]
    return len(qc) >= 150 and float(qc.iloc[-1]) > float(qc.rolling(150).mean().iloc[-1])

# ---------------------------------------------------------------------------
# Build full overnight trade table — one row per (ticker, earnings, day-in-window)
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import fetch_earnings_dates

def build_table(qqq_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every pre-earnings window day (D-20..D-1), compute:
      - overnight return (close → next open)
      - day-in-window (D-X where X = trading days until earnings)
      - entry-day signals (close quality, volume, score)
    """
    rows = []
    for ticker in UNIVERSE:
        df     = _px(ticker)
        edates = fetch_earnings_dates(ticker)

        for ann in edates:
            # D-20 to D-1 range
            d20 = _nth(df, ann, -20)
            d1  = _nth(df, ann, -1)
            if d20 is None or d1 is None: continue

            # All trading days in window
            window_days = [d for d in df.index if d20 <= d <= d1]
            if not window_days: continue

            # Pre-compute score once (at D-20 entry)
            score = _dynamic_score(ticker, d20, qqq_df)
            regime = _regime_ok(d20, qqq_df)

            for i, day in enumerate(window_days):
                # Need next trading day (for next open)
                day_idx = df.index.get_loc(day)
                if day_idx + 1 >= len(df.index): continue
                next_day = df.index[day_idx + 1]

                c   = float(df["Close"].loc[day])
                o_t = float(df["Open"].loc[next_day]) if "Open" in df.columns else float(df["Close"].loc[next_day])
                o   = float(df["Open"].loc[day])      if "Open" in df.columns else c
                h   = float(df["High"].loc[day])      if "High" in df.columns else c
                lo  = float(df["Low"].loc[day])       if "Low"  in df.columns else c
                v   = float(df["Volume"].loc[day])    if "Volume" in df.columns else 0

                if c == 0: continue
                overnight_ret = (o_t - c) / c * 100
                days_until    = len(window_days) - 1 - i   # D-X remaining

                # Day-of signals
                close_pct_high = (c - lo) / (h - lo) if h > lo else 0.5   # 0=low, 1=high
                green_day      = c > o
                vol_ser        = df["Volume"].loc[df.index <= day] if "Volume" in df.columns else None
                vol_avg20      = float(vol_ser.tail(21).iloc[:-1].mean()) if vol_ser is not None and len(vol_ser) >= 21 else 0
                vol_ratio      = v / vol_avg20 if vol_avg20 > 0 else 1.0
                day_range_pct  = (h - lo) / c * 100 if c > 0 else 2.0

                rows.append({
                    "ticker":         ticker,
                    "ann":            ann,
                    "date":           day,
                    "days_until":     days_until,
                    "overnight_ret":  round(overnight_ret, 3),
                    "score":          round(score, 3),
                    "regime_ok":      regime,
                    "green_day":      green_day,
                    "close_pct_high": round(close_pct_high, 3),
                    "vol_ratio":      round(vol_ratio, 2),
                    "day_range_pct":  round(day_range_pct, 2),
                    "strong_close":   close_pct_high >= 0.80,
                    "high_vol":       vol_ratio >= 1.5,
                    "score_ok":       score >= 1.20,
                })

    df_out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df_out


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate(trades: list[dict], label: str = "") -> dict:
    """Sequential: one position at a time, pick best score when dates conflict."""
    if not trades:
        return {"label": label, "n": 0, "final": START_CASH,
                "wr": 0, "avg": 0, "ann": 0, "max_dd": 0, "milestones": {}, "log": []}

    by_date: dict = {}
    for t in trades:
        d = t["date"]
        if d not in by_date or t.get("score", 0) > by_date[d].get("score", 0):
            by_date[d] = t

    equity = START_CASH
    peak   = START_CASH
    max_dd = 0.0
    wins   = 0
    rets   = []
    milestones: dict[int, object] = {}
    log    = []

    for t in sorted(by_date.values(), key=lambda x: x["date"]):
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = t["date"].date()
        ret = t["overnight_ret"]
        equity *= (1 + ret / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if ret > 0: wins += 1
        rets.append(ret)
        log.append({**t, "equity": round(equity, 2)})

    n    = len(rets)
    ann  = (equity / START_CASH) ** (1 / 11) - 1

    return {
        "label":    label,
        "n":        n, "wins": wins,
        "wr":       round(wins / n * 100, 1) if n else 0,
        "avg":      round(sum(rets) / n, 3) if n else 0,
        "final":    round(equity, 2),
        "x":        round(equity / START_CASH, 1),
        "ann":      round(ann * 100, 1),
        "max_dd":   round(max_dd, 1),
        "milestones": milestones,
        "log":      log,
    }


def prow(r: dict, base: dict | None = None) -> None:
    m10 = str(r["milestones"].get(10_000, "—"))[:10] if r.get("milestones") else "—"
    n_yr = round(r["n"] / 11, 1)
    flag = ""
    if base and r["final"] > base["final"] and r["max_dd"] <= base["max_dd"] + 2:
        flag = "  ◄"
    print(f"  {r['label']:<45} {n_yr:>5.1f}  {r['wr']:>5.1f}%  {r['avg']:>+7.3f}%  "
          f"{r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}  {m10}{flag}")


def year_by_year(r: dict) -> None:
    by_yr: dict[int, list] = {}
    for t in r["log"]:
        yr = t["date"].year
        by_yr.setdefault(yr, []).append(t["overnight_ret"])
    cum = START_CASH
    for yr in sorted(by_yr):
        rets = by_yr[yr]
        avg  = sum(rets) / len(rets)
        w    = sum(1 for r_ in rets if r_ > 0)
        for r_ in rets: cum *= (1 + r_ / 100)
        bar  = ("+" if avg >= 0 else "-") + "█" * min(int(abs(avg) / 0.10), 30)
        print(f"    {yr}  {w:>3}/{len(rets):>3} wins  avg {avg:>+6.3f}%  ${cum:>9,.0f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  PRE-EARNINGS OVERNIGHT STRATEGY — FULL REFINEMENT")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ"]:
        _px(t)
        print(f"  {t} ready")
    qqq_df = _px("QQQ")

    print("\nBuilding overnight trade table...")
    df = build_table(qqq_df)
    print(f"  {len(df)} overnight observations (pre-earnings window only)")

    base_trades = df[df["regime_ok"]].to_dict("records")

    # ── 1. Which day in the window is best? ───────────────────────────────
    print(f"\n{SEP}")
    print("  Q1: WHICH DAYS IN THE D-20 WINDOW HAVE THE BEST OVERNIGHT RETURN?")
    print(SEP)
    print(f"  {'Days until earnings':<22} {'N':>5}  {'Win%':>6}  {'Avg%':>8}  {'Median':>8}  Note")
    print(f"  {'-'*22} {'-'*5}  {'-'*6}  {'-'*8}  {'-'*8}")

    # Individual day buckets
    regime_df = df[df["regime_ok"]]
    for d in sorted(regime_df["days_until"].unique()):
        sub = regime_df[regime_df["days_until"] == d]
        if len(sub) < 10: continue
        avg = sub["overnight_ret"].mean()
        med = sub["overnight_ret"].median()
        wr  = (sub["overnight_ret"] > 0).mean() * 100
        flag = "  ◄ strong" if avg >= 0.15 and wr >= 57 else ""
        print(f"  D-{d:<20} {len(sub):>5}  {wr:>5.1f}%  {avg:>+7.3f}%  {med:>+7.3f}%{flag}")

    # Window ranges
    print(f"\n  — Window ranges —")
    for label, d_from, d_to in [
        ("D-20 to D-1 (full window)",  1, 20),
        ("D-15 to D-1",                1, 15),
        ("D-10 to D-1",                1, 10),
        ("D-7 to D-1  (last week)",    1,  7),
        ("D-5 to D-1",                 1,  5),
        ("D-3 to D-1  (last 3 nights)",1,  3),
        ("D-2 and D-1 (final 2)",      1,  2),
        ("D-1 only (night before)",    1,  1),
        ("D-20 to D-6 (early)",        6, 20),
        ("D-10 to D-6",                6, 10),
    ]:
        sub = regime_df[
            (regime_df["days_until"] >= d_from) &
            (regime_df["days_until"] <= d_to)
        ]
        if len(sub) < 10: continue
        avg = sub["overnight_ret"].mean()
        wr  = (sub["overnight_ret"] > 0).mean() * 100
        flag = "  ◄" if avg >= 0.15 and wr >= 57 else ""
        print(f"  {label:<30} n={len(sub):>4}  {wr:>5.1f}%  {avg:>+7.3f}%{flag}")

    # ── 2. Score filter ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Q2: DOES SCORE >= 1.20 IMPROVE OVERNIGHT RETURNS?")
    print(SEP)
    for label, mask in [
        ("All regime ok",           regime_df.index),
        ("Score >= 1.05",           regime_df[regime_df["score"] >= 1.05].index),
        ("Score >= 1.20",           regime_df[regime_df["score"] >= 1.20].index),
        ("Score >= 1.30",           regime_df[regime_df["score"] >= 1.30].index),
        ("Score >= 1.40",           regime_df[regime_df["score"] >= 1.40].index),
        ("Score >= 1.50",           regime_df[regime_df["score"] >= 1.50].index),
    ]:
        sub = df.loc[mask]
        if len(sub) < 20: continue
        avg = sub["overnight_ret"].mean()
        wr  = (sub["overnight_ret"] > 0).mean() * 100
        flag = "  ◄" if avg >= 0.15 and wr >= 57 else ""
        print(f"  {label:<30} n={len(sub):>4}  {wr:>5.1f}%  avg {avg:>+7.3f}%{flag}")

    # ── 3. How many consecutive nights per cycle? ─────────────────────────
    print(f"\n{SEP}")
    print("  Q3: CONSECUTIVE NIGHTS PER EARNINGS CYCLE")
    print(SEP)
    print("  (How many nights in a row during a single pre-earnings window can we trade?)")

    # For each earnings cycle, simulate entering every consecutive night
    for label, d_from, d_to in [
        ("Every night D-5 to D-1",  1, 5),
        ("Every night D-3 to D-1",  1, 3),
        ("Best single night D-5→D-1 (highest close pct high)", 1, 5),
    ]:
        trades = []
        for (ticker, ann), grp in regime_df.groupby(["ticker", "ann"]):
            win = grp[(grp["days_until"] >= d_from) & (grp["days_until"] <= d_to)]
            if len(win) == 0: continue
            if "Best single" in label:
                best = win.loc[win["close_pct_high"].idxmax()]
                trades.append(best.to_dict())
            else:
                for _, row in win.iterrows():
                    trades.append(row.to_dict())
        r = simulate(trades, label)
        if r["n"] < 10: continue
        n_yr = round(r["n"] / 11, 1)
        print(f"  {label:<45} {n_yr:>5.1f}/yr  {r['wr']:>5.1f}%  avg {r['avg']:>+7.3f}%  "
              f"ann {r['ann']:>+5.1f}%  dd {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}")

    # ── 4. Per-ticker contribution ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Q4: PER-TICKER OVERNIGHT EDGE (regime ok, full D-20 window)")
    print(SEP)
    print(f"  {'Ticker':<8} {'N':>5}  {'Win%':>6}  {'Avg%':>8}  {'Best window':<20}  Note")
    print(f"  {'-'*8} {'-'*5}  {'-'*6}  {'-'*8}  {'-'*20}")
    for ticker in UNIVERSE:
        sub = regime_df[regime_df["ticker"] == ticker]
        if len(sub) < 10: continue
        avg = sub["overnight_ret"].mean()
        wr  = (sub["overnight_ret"] > 0).mean() * 100
        # Find best 5-day window
        best_avg, best_win = 0, ""
        for d_from, d_to in [(1,3),(1,5),(1,7),(3,7),(5,10)]:
            w = sub[(sub["days_until"] >= d_from) & (sub["days_until"] <= d_to)]
            if len(w) >= 5 and w["overnight_ret"].mean() > best_avg:
                best_avg = w["overnight_ret"].mean()
                best_win = f"D-{d_to}→D-{d_from} ({w['overnight_ret'].mean():+.3f}%)"
        flag = "  ◄ strong" if avg >= 0.15 and wr >= 57 else ""
        print(f"  {ticker:<8} {len(sub):>5}  {wr:>5.1f}%  {avg:>+7.3f}%  {best_win:<20}{flag}")

    # ── 5. Optimal entry filters — full grid ──────────────────────────────
    print(f"\n{SEP}")
    print("  Q5: OPTIMAL FILTER COMBINATIONS (sequential simulation)")
    print(SEP)
    hdr = f"  {'Strategy':<45} {'N/yr':>5}  {'Win%':>5}  {'Avg%':>7}  {'Ann%':>5}  {'MaxDD':>5}  {'Final':>10}  $10k"
    print(hdr)
    print(f"  {'-'*45} {'-'*5}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}")

    def filt(window_from=1, window_to=20, tickers=None,
             score_min=None, green=False, strong_close=False,
             high_vol=False, score_ok=False):
        f = (regime_df["days_until"] >= window_from) & (regime_df["days_until"] <= window_to)
        if tickers: f = f & regime_df["ticker"].isin(tickers)
        if score_min: f = f & (regime_df["score"] >= score_min)
        if green: f = f & regime_df["green_day"]
        if strong_close: f = f & regime_df["strong_close"]
        if high_vol: f = f & regime_df["high_vol"]
        return regime_df[f].to_dict("records")

    configs = [
        # Baseline
        ("Baseline — full window, no filters",
            filt()),
        # Window only
        ("D-5 to D-1 only",
            filt(1, 5)),
        ("D-3 to D-1 only",
            filt(1, 3)),
        # Score filter
        ("D-5→D-1 + score>=1.20",
            filt(1, 5, score_min=1.20)),
        ("D-3→D-1 + score>=1.20",
            filt(1, 3, score_min=1.20)),
        # Green day filter
        ("D-5→D-1 + green day",
            filt(1, 5, green=True)),
        ("D-3→D-1 + green day",
            filt(1, 3, green=True)),
        # Strong close
        ("D-5→D-1 + strong close",
            filt(1, 5, strong_close=True)),
        ("D-3→D-1 + strong close",
            filt(1, 3, strong_close=True)),
        # Combinations
        ("D-5→D-1 + score>=1.20 + green",
            filt(1, 5, score_min=1.20, green=True)),
        ("D-5→D-1 + score>=1.20 + strong close",
            filt(1, 5, score_min=1.20, strong_close=True)),
        ("D-3→D-1 + score>=1.20 + green",
            filt(1, 3, score_min=1.20, green=True)),
        ("D-3→D-1 + score>=1.20 + strong close",
            filt(1, 3, score_min=1.20, strong_close=True)),
        ("D-3→D-1 + score>=1.20 + strong close + green",
            filt(1, 3, score_min=1.20, strong_close=True, green=True)),
        # NVDA + AMD only (strongest tickers)
        ("NVDA+AMD only, D-5→D-1",
            filt(1, 5, tickers=["NVDA","AMD"])),
        ("NVDA+AMD only, D-3→D-1 + score>=1.20",
            filt(1, 3, tickers=["NVDA","AMD"], score_min=1.20)),
        ("NVDA+AMD+AMZN, D-5→D-1 + score>=1.20",
            filt(1, 5, tickers=["NVDA","AMD","AMZN"], score_min=1.20)),
        # With high vol
        ("D-5→D-1 + high vol + score>=1.20",
            filt(1, 5, score_min=1.20, high_vol=True)),
    ]

    base_r = simulate(filt(), "Baseline")
    results = []
    for label, trades in configs:
        r = simulate(trades, label)
        results.append(r)
        prow(r, base_r)

    # ── Best result deep dive ──────────────────────────────────────────────
    viable = [r for r in results if r["n"] >= 10 and r["ann"] >= 10 and r["max_dd"] <= 25]
    if viable:
        best = max(viable, key=lambda r: r["ann"] / max(r["max_dd"], 1))
        print(f"\n{SEP}")
        print(f"  BEST RISK-ADJUSTED: {best['label']}")
        print(f"  {best['n']} trades ({best['n']/11:.1f}/yr)  |  "
              f"Win {best['wr']:.1f}%  |  Avg {best['avg']:+.3f}%  |  "
              f"Ann {best['ann']:+.1f}%  |  MaxDD {best['max_dd']:.1f}%  |  "
              f"Final ${best['final']:,.0f}")
        print(SEP)

        print(f"\n  Year-by-year:")
        year_by_year(best)

        print(f"\n  Milestones from $2,000:")
        for m in MILESTONES:
            ms = best["milestones"].get(m)
            print(f"    ${m:>6,}: {str(ms)[:10] if ms else '—'}")

        # Per-ticker
        by_t: dict[str, list] = {}
        for t in best["log"]:
            by_t.setdefault(t["ticker"], []).append(t["overnight_ret"])
        print(f"\n  Per-ticker contribution:")
        print(f"    {'Ticker':<8} {'N':>5}  {'Win%':>6}  {'Avg%':>8}  {'Total%':>8}")
        for tk in sorted(by_t, key=lambda x: -len(by_t[x])):
            rets = by_t[tk]
            w    = sum(1 for r_ in rets if r_ > 0)
            avg  = sum(rets) / len(rets)
            print(f"    {tk:<8} {len(rets):>5}  {w/len(rets)*100:>5.1f}%  {avg:>+7.3f}%  {sum(rets):>+7.1f}%")

        # Top 10 best and worst trades
        top10 = sorted(best["log"], key=lambda x: -x["overnight_ret"])[:10]
        bot10 = sorted(best["log"], key=lambda x:  x["overnight_ret"])[:10]
        print(f"\n  Top 10 trades:")
        for t in top10:
            print(f"    {str(t['date'].date())}  {t['ticker']:<6}  D-{t['days_until']}  {t['overnight_ret']:>+7.2f}%  score {t['score']:.2f}")
        print(f"\n  Worst 10 trades:")
        for t in bot10:
            print(f"    {str(t['date'].date())}  {t['ticker']:<6}  D-{t['days_until']}  {t['overnight_ret']:>+7.2f}%  score {t['score']:.2f}")

        # ── Final system rules ─────────────────────────────────────────────
        print(f"\n{SEP}")
        print("  FINAL SYSTEM — Pre-Earnings Overnight Strategy (O1)")
        print(SEP)
        print(f"""
  Name:    O1 — Pre-Earnings Overnight
  Edge:    Stocks drift up overnight in the final days before earnings
  Entry:   Buy at market CLOSE on qualifying day
  Exit:    Sell at market OPEN next morning
  Hold:    Overnight only — no intraday risk

  Universe: GOOGL, NVDA, AMZN, MSFT, META, AMD
  Rules:
    1. Regime: QQQ > 150dma
    2. Ticker in D-20 pre-earnings window
    3. Dynamic score >= 1.20 (momentum + relative strength filter)
    4. Optimal entry window: confirmed from sweep above
    5. Position size: 100% of account per trade
    6. Exit: always at next morning's open — no exceptions
    7. Multiple tickers signaling same night: buy highest score

  Historical result ({best['label']}):
    Trades:   {best['n']} total  ({best['n']/11:.1f}/yr)
    Win rate: {best['wr']:.1f}%
    Avg/trade:{best['avg']:+.3f}%
    Ann return:{best['ann']:+.1f}%
    Max DD:   {best['max_dd']:.1f}%
    Final:    ${best['final']:,.0f} from $2,000 start
""")

    # ── Comparison vs S2 ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  COMPARISON: O1 overnight vs S2 multi-week")
    print(SEP)
    print(f"  {'Strategy':<30}  {'N/yr':>5}  {'Win%':>6}  {'Avg%':>8}  {'Ann%':>6}  {'MaxDD':>6}  {'Overnight risk':>15}")
    print(f"  {'-'*30}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*15}")
    print(f"  {'S2 Pre-earnings (multi-wk)':<30}  {'3.7':>5}  {'73.2%':>6}  {'+6.610%':>8}  {'+24.3%':>6}  {'-6.9%':>6}  {'no (hold days)':>15}")
    if viable and best:
        print(f"  {best['label'][:30]:<30}  {best['n']/11:>5.1f}  {best['wr']:>5.1f}%  {best['avg']:>+7.3f}%  {best['ann']:>+5.1f}%  {best['max_dd']:>5.1f}%  {'yes (close→open)':>15}")
    print(f"\n  → These strategies are COMPLEMENTARY — O1 is intraday-overnight only,")
    print(f"    S2 holds for weeks. Can run both on same account without conflict.")
    print(f"    O1 adds fast-compounding trades between S2 entries.")


if __name__ == "__main__":
    main()

