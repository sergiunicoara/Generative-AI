"""
F6 + Strategies Backtest
=========================
Runs all 3 strategies (S1, S2, S3) with and without the F6 filter
(gap up on entry day: open > prev_close) and compares outcomes.

Usage:
    uv run python -m backend.research.f6_strategies_backtest
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from scipy.stats import norm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP  = "=" * 72
DASH = "─" * 72
START_CASH   = 2_000.0
STOP_PCT     = -5.0
RISK_FREE    = 0.05
EXPIRY_DAYS  = 60
SCORE_THRESH = 1.05

S1_UNIVERSE = ["GOOGL", "NVDA", "AMZN"]
S2_UNIVERSE = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]

BASE_QUALITY = {
    "GOOGL": 1.40, "NVDA": 1.50, "AMZN": 1.20,
    "MSFT":  1.10, "META": 1.10, "AMD":  1.00,
}

MILESTONES = [5_000, 10_000, 20_000, 50_000]

# ---------------------------------------------------------------------------
# Price helpers
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

def _nth(df, ref, offset):
    dt, d, c = pd.Timestamp(ref), (1 if offset >= 0 else -1), 0
    for i in range(1, 300):
        cand = dt + pd.Timedelta(days=i*d)
        if cand in df.index:
            c += 1
            if c == abs(offset): return cand
    return None

def _bounded(x): return max(-20.0, min(20.0, x)) / 100.0

def _score(ticker, entry_dt, qqq_df):
    df  = _prices(ticker)
    col = df["Close"].loc[df.index <= entry_dt]
    if len(col) < 62: return BASE_QUALITY.get(ticker, 1.0)
    m20 = (col.iloc[-1] / col.iloc[-21] - 1) * 100
    m60 = (col.iloc[-1] / col.iloc[-62] - 1) * 100
    qc  = qqq_df["Close"].loc[qqq_df.index <= entry_dt]
    q20 = (qc.iloc[-1] / qc.iloc[-21] - 1) * 100 if len(qc) >= 21 else 0
    rs  = m20 - float(q20)
    return BASE_QUALITY.get(ticker, 1.0) + 1.20*_bounded(m20) + 0.80*_bounded(m60) + 1.50*_bounded(rs)

def _regime_ok(entry_dt, qqq_df):
    qc = qqq_df["Close"].loc[qqq_df.index <= entry_dt]
    if len(qc) < 150: return False
    return float(qc.iloc[-1]) > float(qc.rolling(150).mean().iloc[-1])

def _gap_up(ticker, entry_dt):
    """F6: entry day open > previous close."""
    df = _prices(ticker)
    if "Open" not in df.columns: return True
    prev = [d for d in df.index if d < entry_dt]
    if not prev: return True
    return float(df["Open"].loc[entry_dt]) > float(df["Close"].loc[prev[-1]])

# ---------------------------------------------------------------------------
# Black-Scholes (S3)
# ---------------------------------------------------------------------------

def _hist_vol(ticker, entry_dt):
    df  = _prices(ticker)
    col = df["Close"].loc[df.index < entry_dt].tail(30)
    if len(col) < 10: return 0.35
    rets = col.pct_change().dropna()
    return float(rets.std() * math.sqrt(252))

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(S - K, 0.0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)

def _opt_roi(ticker, entry_dt, exit_dt, S_entry, S_exit):
    """Return option ROI (with 25% real-world haircut)."""
    iv   = _hist_vol(ticker, entry_dt)
    K    = S_entry * 0.90
    hold = len([d for d in _prices(ticker).index if entry_dt <= d <= exit_dt])
    T0   = EXPIRY_DAYS / 365
    T1   = max((EXPIRY_DAYS - hold * 1.4) / 365, 0)
    c0   = bs_call(S_entry, K, T0, RISK_FREE, iv) * 0.75
    c1   = bs_call(S_exit,  K, T1, RISK_FREE, iv) * 0.75
    return (c1 - c0) / c0 * 100 if c0 > 0.01 else 0.0

# ---------------------------------------------------------------------------
# Build signal events per strategy
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import fetch_earnings_dates

def build_events(universe: list[str], qqq_df: pd.DataFrame,
                 fixed_rank: bool, use_f6: bool) -> list[dict]:
    """All valid entry events for this strategy, sorted by entry date."""
    events = []
    for ticker in universe:
        df    = _prices(ticker)
        dates = fetch_earnings_dates(ticker)
        for ann in dates:
            e_dt = _nth(df, ann, -20)
            x_dt = _nth(df, ann, -1)
            if e_dt is None or x_dt is None: continue
            if e_dt not in df.index or x_dt not in df.index: continue
            if not _regime_ok(e_dt, qqq_df): continue

            sc = _score(ticker, e_dt, qqq_df)
            if not fixed_rank and sc < SCORE_THRESH: continue
            if use_f6 and not _gap_up(ticker, e_dt): continue

            ep = float(df["Close"].loc[e_dt])
            xp = float(df["Close"].loc[x_dt])
            window = [d for d in df.index if e_dt <= d <= x_dt]
            min_px = min(float(df["Close"].loc[d]) for d in window)
            stopped = min_px <= ep * 0.95
            ret_stock = STOP_PCT if stopped else (xp - ep) / ep * 100

            events.append({
                "ticker":   ticker,
                "ann":      ann,
                "entry_dt": e_dt,
                "exit_dt":  x_dt,
                "entry_px": ep,
                "exit_px":  xp,
                "ret":      round(ret_stock, 3),
                "score":    round(sc, 3),
            })
    return sorted(events, key=lambda e: e["entry_dt"])


def pick_best(events_in_window: list[dict], fixed_rank: bool,
              universe: list[str]) -> dict | None:
    if not events_in_window: return None
    if fixed_rank:
        for t in universe:
            for ev in events_in_window:
                if ev["ticker"] == t: return ev
    return max(events_in_window, key=lambda e: e["score"])


def simulate(events: list[dict], fixed_rank: bool, universe: list[str],
             options_pct: float) -> dict:
    """Sequential simulation: 1 position at a time, 100% deployed."""
    equity, peak, max_dd = START_CASH, START_CASH, 0.0
    milestones: dict[int, object] = {}
    wins = losses = 0
    trade_log = []
    busy_until = None

    # Group events by entry date and pick best available each day
    from itertools import groupby
    by_date = {}
    for ev in events:
        by_date.setdefault(ev["entry_dt"], []).append(ev)

    taken_events = []
    busy_until   = None
    for entry_dt in sorted(by_date):
        if busy_until and entry_dt <= busy_until:
            continue
        candidates = by_date[entry_dt]
        chosen = pick_best(candidates, fixed_rank, universe)
        if chosen:
            taken_events.append(chosen)
            busy_until = chosen["exit_dt"]

    for ev in taken_events:
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = ev["entry_dt"].date()

        # Blend: stock + optional calls
        stock_ret = ev["ret"]
        if options_pct > 0:
            opt_roi = _opt_roi(ev["ticker"], ev["entry_dt"], ev["exit_dt"],
                               ev["entry_px"], ev["exit_px"])
            blended = (1.0 - options_pct) * stock_ret + options_pct * opt_roi
        else:
            blended = stock_ret

        equity = equity * (1 + blended / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if blended > 0: wins += 1
        else: losses += 1
        trade_log.append({
            "ticker": ev["ticker"], "ann": ev["ann"],
            "ret": round(blended, 2), "equity": round(equity, 2),
        })

    n    = wins + losses
    rets = [t["ret"] for t in trade_log]
    return {
        "n": n, "wins": wins,
        "wr":      round(wins/n*100, 1) if n else 0,
        "avg":     round(sum(rets)/n, 2) if n else 0,
        "final":   round(equity, 2),
        "x":       round(equity/START_CASH, 1),
        "max_dd":  round(max_dd, 1),
        "milestones": milestones,
        "log":     trade_log,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  F6 FILTER + ALL 3 STRATEGIES — $2,000 starting capital")
    print(SEP)

    print("\nLoading prices...")
    for t in S2_UNIVERSE + ["QQQ"]:
        _prices(t)
        print(f"  {t} ready")
    qqq_df = _prices("QQQ")

    configs = [
        ("S1  Fixed-rank",       S1_UNIVERSE, True,  0.00),
        ("S2  Dynamic safer",    S2_UNIVERSE, False, 0.00),
        ("S3  Stock+10% calls",  S2_UNIVERSE, False, 0.10),
    ]

    print(f"\n{'Building trade events...':}")
    results = {}
    for label, universe, fixed, opt_pct in configs:
        for use_f6 in [False, True]:
            key   = f"{label}{'  + F6' if use_f6 else ''}"
            evs   = build_events(universe, qqq_df, fixed_rank=fixed, use_f6=use_f6)
            r     = simulate(evs, fixed_rank=fixed, universe=universe, options_pct=opt_pct)
            results[key] = r
            print(f"  {key:<40} {len(evs):>4} events → {r['n']} trades")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  RESULTS COMPARISON")
    print(SEP)
    print(f"  {'Strategy':<36} {'N':>4}  {'Win%':>6}  {'Avg%':>7}  {'MaxDD':>6}  {'Final':>10}  {'x':>5}  $10k")
    print(f"  {'-'*36} {'-'*4}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*10}  {'-'*5}  {'-'*10}")

    for label, universe, fixed, opt_pct in configs:
        for use_f6 in [False, True]:
            key = f"{label}{'  + F6' if use_f6 else ''}"
            r   = results[key]
            ms  = str(r["milestones"].get(10_000, "—"))[:10]

            # Compare f6 vs no-f6 for same strategy
            base_key = label
            base_r   = results.get(base_key)
            flag = ""
            if use_f6 and base_r:
                better = r["final"] > base_r["final"] and r["max_dd"] <= base_r["max_dd"]
                flag   = "  ◄ BETTER" if better else ("  ↑ final only" if r["final"] > base_r["final"] else "")

            marker = "  + F6" if use_f6 else "      "
            print(f"  {label}{marker:<6}  {r['n']:>4}  {r['wr']:>5.1f}%  "
                  f"{r['avg']:>+6.2f}%  {r['max_dd']:>5.1f}%  "
                  f"${r['final']:>9,.0f}  {r['x']:>4.1f}x  {ms}{flag}")
        print()

    # ── Year-by-year for best version of each strategy ─────────────────────
    print(f"\n{SEP}")
    print(f"  YEAR-BY-YEAR — best version (with F6 if it helps)")
    print(SEP)

    for label, universe, fixed, opt_pct in configs:
        key_base = label
        key_f6   = f"{label}  + F6"
        r_base   = results[key_base]
        r_f6     = results[key_f6]
        best_key = key_f6 if r_f6["final"] > r_base["final"] else key_base
        r        = results[best_key]

        print(f"\n  {best_key}")
        by_yr: dict[int, list] = {}
        for t in r["log"]:
            yr = int(t["ann"][:4])
            by_yr.setdefault(yr, []).append(t["ret"])

        cum = START_CASH
        for yr in sorted(by_yr):
            rets = by_yr[yr]
            avg  = sum(rets)/len(rets)
            w    = sum(1 for r_ in rets if r_ > 0)
            for r_ in rets:
                cum *= (1 + r_/100)
            bar  = ("+" if avg>=0 else "-") + "█"*min(int(abs(avg)/3), 20)
            print(f"    {yr}  {w}/{len(rets):>2} wins  avg {avg:>+5.1f}%  "
                  f"equity ${cum:>9,.0f}  {bar}")

    # ── Milestone comparison ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  MILESTONES — $2,000 start")
    print(SEP)
    print(f"  {'Strategy':<42}  {'$5k':>12}  {'$10k':>12}  {'$20k':>12}")
    print(f"  {'-'*42}  {'-'*12}  {'-'*12}  {'-'*12}")
    for label, universe, fixed, opt_pct in configs:
        for use_f6 in [False, True]:
            key = f"{label}{'  + F6' if use_f6 else ''}"
            r   = results[key]
            m5  = str(r["milestones"].get(5_000,  "—"))[:10]
            m10 = str(r["milestones"].get(10_000, "—"))[:10]
            m20 = str(r["milestones"].get(20_000, "—"))[:10]
            print(f"  {key:<42}  {m5:>12}  {m10:>12}  {m20:>12}")
        print()

    # ── Final verdict ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VERDICT: Does F6 improve each strategy?")
    print(SEP)
    for label, universe, fixed, opt_pct in configs:
        r0 = results[label]
        r1 = results[f"{label}  + F6"]
        d_final = r1["final"] - r0["final"]
        d_wr    = r1["wr"]    - r0["wr"]
        d_dd    = r1["max_dd"]- r0["max_dd"]
        d_n     = r1["n"]     - r0["n"]
        verdict = "YES ◄" if r1["final"] > r0["final"] and r1["max_dd"] <= r0["max_dd"] else \
                  "MIXED" if r1["final"] > r0["final"] else "NO"
        print(f"  {label:<28}  {verdict:<6}  "
              f"Δfinal {d_final:>+8,.0f}  Δwin {d_wr:>+5.1f}%  "
              f"Δdd {d_dd:>+4.1f}%  Δtrades {d_n:>+3}")


if __name__ == "__main__":
    main()

