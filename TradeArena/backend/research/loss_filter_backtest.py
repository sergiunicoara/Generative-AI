"""
Loss Filter Backtest
====================
1. Analyzes what losing trades have in common (S2+F6 baseline).
2. Tests targeted filters that specifically attack those patterns.
3. Finds the combination with the best final equity / drawdown ratio.

New filters tested:
  L1  Not in recent downtrend   stock > 50dma at entry
  L2  No prior earnings gap-dn  previous earnings didn't cause >5% drop
  L3  Positive 5d momentum      stock up over last 5 trading days
  L4  IV not extreme            30d realized vol < 60% (stock not in panic)
  L5  Volume confirmation       entry day volume > 20d avg volume
  L6  Not far from 52w high     stock within 25% of 52-week high
  L7  Sector uptrend            XLK above its 20dma (shorter-term filter)
  L8  Min score 1.20            raise score bar from 1.05 to 1.20

All tested on S2+F6 and S3+F6 (the two improved strategies).

Usage:
    uv run python -m backend.research.loss_filter_backtest
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP  = "=" * 72
START_CASH   = 2_000.0
STOP_PCT     = -5.0
RISK_FREE    = 0.05
EXPIRY_DAYS  = 60
SCORE_THRESH = 1.05

S2_UNIVERSE = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
BASE_QUALITY = {
    "GOOGL": 1.40, "NVDA": 1.50, "AMZN": 1.20,
    "MSFT":  1.10, "META": 1.10, "AMD":  1.00,
}

MILESTONES = [5_000, 10_000, 20_000, 50_000]

# ---------------------------------------------------------------------------
# Price / indicator helpers
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
        cand = dt + pd.Timedelta(days=i*d)
        if cand in df.index:
            c += 1
            if c == abs(offset): return cand
    return None

def _bounded(x): return max(-20.0, min(20.0, x)) / 100.0

def _score(ticker, entry_dt, qqq_df):
    df  = _px(ticker)
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
    return len(qc) >= 150 and float(qc.iloc[-1]) > float(qc.rolling(150).mean().iloc[-1])

def _gap_up(ticker, entry_dt):
    df   = _px(ticker)
    if "Open" not in df.columns: return True
    prev = [d for d in df.index if d < entry_dt]
    if not prev: return True
    return float(df["Open"].loc[entry_dt]) > float(df["Close"].loc[prev[-1]])

# Black-Scholes helpers
def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(S - K, 0.0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)

def _opt_roi(ticker, entry_dt, exit_dt, S_entry, S_exit):
    df   = _px(ticker)
    col  = df["Close"].loc[df.index < entry_dt].tail(30)
    iv   = float(col.pct_change().dropna().std() * math.sqrt(252)) if len(col) >= 10 else 0.35
    K    = S_entry * 0.90
    hold = len([d for d in df.index if entry_dt <= d <= exit_dt])
    c0   = bs_call(S_entry, K, EXPIRY_DAYS/365, RISK_FREE, iv) * 0.75
    c1   = bs_call(S_exit, K, max((EXPIRY_DAYS - hold*1.4)/365, 0), RISK_FREE, iv) * 0.75
    return (c1 - c0) / c0 * 100 if c0 > 0.01 else 0.0

# ---------------------------------------------------------------------------
# Build enriched trade list (S2 + F6 baseline)
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import fetch_earnings_dates

def build_enriched(qqq_df: pd.DataFrame, xlk_df: pd.DataFrame) -> list[dict]:
    raw = []
    for ticker in S2_UNIVERSE:
        df    = _px(ticker)
        dates = fetch_earnings_dates(ticker)
        for ann in dates:
            e_dt = _nth(df, ann, -20)
            x_dt = _nth(df, ann, -1)
            if e_dt is None or x_dt is None: continue
            if e_dt not in df.index or x_dt not in df.index: continue
            if not _regime_ok(e_dt, qqq_df): continue
            sc = _score(ticker, e_dt, qqq_df)
            if sc < SCORE_THRESH: continue
            if not _gap_up(ticker, e_dt): continue   # F6 already applied

            ep = float(df["Close"].loc[e_dt])
            xp = float(df["Close"].loc[x_dt])
            window = [d for d in df.index if e_dt <= d <= x_dt]
            min_px = min(float(df["Close"].loc[d]) for d in window)
            stopped = min_px <= ep * 0.95
            ret_s = STOP_PCT if stopped else (xp - ep) / ep * 100
            ret_o = _opt_roi(ticker, e_dt, x_dt, ep, xp)

            col = df["Close"].loc[df.index <= e_dt]

            # L1: above 50dma
            ma50 = float(col.tail(50).mean()) if len(col) >= 50 else ep
            l1   = ep > ma50

            # L2: prior earnings gap not a disaster
            # Find previous earnings date
            prev_anns = [a for a in dates if a < ann]
            l2 = True
            if prev_anns:
                prev_ann  = prev_anns[-1]
                p_dt      = _nth(df, prev_ann, 1)   # day after prev earnings
                pp_dt     = _nth(df, prev_ann, 0)   # day of prev earnings (approx prev close)
                if p_dt and pp_dt and p_dt in df.index and pp_dt in df.index:
                    prev_earn_idx = [d for d in df.index if d < pp_dt]
                    if prev_earn_idx:
                        pre_close  = float(df["Close"].loc[prev_earn_idx[-1]])
                        post_open  = float(df["Open"].loc[p_dt]) if "Open" in df.columns else float(df["Close"].loc[p_dt])
                        earn_gap   = (post_open - pre_close) / pre_close * 100
                        l2 = earn_gap > -5.0   # previous earnings didn't gap down >5%

            # L3: positive 5d momentum
            mom5 = (col.iloc[-1] / col.iloc[-6] - 1) * 100 if len(col) >= 6 else 0
            l3   = mom5 > 0

            # L4: realized vol < 60%
            rv30 = float(col.tail(30).pct_change().dropna().std() * math.sqrt(252)) if len(col) >= 30 else 0.35
            l4   = rv30 < 0.60

            # L5: entry volume > 20d avg
            if "Volume" in df.columns:
                vol_entry = float(df["Volume"].loc[e_dt])
                vol_avg   = float(df["Volume"].loc[df.index <= e_dt].tail(20).mean())
                l5 = vol_entry > vol_avg * 0.8
            else:
                l5 = True

            # L6: within 25% of 52-week high
            hi52 = float(col.tail(252).max()) if len(col) >= 50 else ep
            l6   = ep >= hi52 * 0.75

            # L7: XLK above 20dma
            xc   = xlk_df["Close"].loc[xlk_df.index <= e_dt]
            ma20x= float(xc.tail(20).mean()) if len(xc) >= 20 else float(xc.iloc[-1])
            l7   = float(xc.iloc[-1]) > ma20x if len(xc) >= 20 else True

            # L8: score >= 1.20
            l8   = sc >= 1.20

            raw.append({
                "ticker":  ticker, "ann": ann,
                "entry_dt":e_dt,  "exit_dt": x_dt,
                "entry_px":ep,    "exit_px":  xp,
                "ret_stock": round(ret_s, 3),
                "ret_blend": round(0.9*ret_s + 0.1*ret_o, 3),  # S3 blend
                "stopped":  stopped,
                "score":    round(sc, 3),
                "rv30":     round(rv30*100, 1),
                "mom5":     round(mom5, 2),
                "l1":l1,"l2":l2,"l3":l3,"l4":l4,"l5":l5,"l6":l6,"l7":l7,"l8":l8,
            })
    return sorted(raw, key=lambda t: t["entry_dt"])

# ---------------------------------------------------------------------------
# Sequential simulation
# ---------------------------------------------------------------------------

def simulate(trades: list[dict], ret_key: str = "ret_stock") -> dict:
    filtered, busy = [], None
    for t in trades:
        if busy and t["entry_dt"] <= busy: continue
        filtered.append(t)
        busy = t["exit_dt"]

    equity, peak, max_dd = START_CASH, START_CASH, 0.0
    milestones: dict[int, object] = {}
    wins = losses = 0
    log  = []

    for t in filtered:
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = t["entry_dt"].date()
        equity = equity * (1 + t[ret_key] / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if t[ret_key] > 0: wins += 1
        else: losses += 1
        log.append(t)

    n    = wins + losses
    rets = [t[ret_key] for t in log]
    return {
        "n": n, "wins": wins,
        "wr":    round(wins/n*100, 1) if n else 0,
        "avg":   round(sum(rets)/n, 2) if n else 0,
        "final": round(equity, 2),
        "x":     round(equity/START_CASH, 1),
        "max_dd":round(max_dd, 1),
        "milestones": milestones,
        "log":   log,
    }

# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def prow(label, r, base, ret_key):
    d_wr  = r["wr"]    - base["wr"]
    d_avg = r["avg"]   - base["avg"]
    d_fin = r["final"] - base["final"]
    m10   = str(r["milestones"].get(10_000, "—"))[:10]
    better = r["final"] > base["final"] and r["max_dd"] <= base["max_dd"] + 1
    flag   = "  ◄" if better else ""
    label_s = label[:38]
    print(f"  {label_s:<38} {r['n']:>4}  {r['wr']:>5.1f}% ({d_wr:>+4.1f})  "
          f"{r['avg']:>+5.2f}% ({d_avg:>+4.2f})  "
          f"DD {r['max_dd']:>4.1f}%  ${r['final']:>9,.0f}  {m10}{flag}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  LOSS FILTER BACKTEST — targeting bad trades")
    print(SEP)

    print("\nLoading prices...")
    for t in S2_UNIVERSE + ["QQQ", "XLK"]:
        _px(t)
        print(f"  {t} ready")

    qqq_df = _px("QQQ")
    xlk_df = _px("XLK")

    print("\nBuilding enriched trade list (S2 + F6 base)...")
    all_t = build_enriched(qqq_df, xlk_df)
    print(f"  {len(all_t)} trades in S2+F6 baseline")

    # Baselines
    base_s2 = simulate(all_t, "ret_stock")
    base_s3 = simulate(all_t, "ret_blend")

    # ── Losing trade analysis ─────────────────────────────────────────────
    losses_s2 = [t for t in base_s2["log"] if t["ret_stock"] <= 0]
    print(f"\n{SEP}")
    print(f"  LOSING TRADES ANALYSIS  ({len(losses_s2)} losses out of {base_s2['n']})")
    print(SEP)
    print(f"  {'Ticker':<6} {'Date':12} {'Ret%':>7}  "
          f"{'L1':>3} {'L2':>3} {'L3':>3} {'L4':>3} {'L5':>3} {'L6':>3} {'L7':>3} {'L8':>3}  "
          f"{'Score':>6}  {'RV30':>5}  {'Mom5':>6}")
    print(f"  {'-'*6} {'-'*12} {'-'*7}  " + "  ".join(["-"*3]*8) + f"  {'-'*6}  {'-'*5}  {'-'*6}")
    for t in losses_s2:
        flags = "".join([
            "✓" if t["l1"] else "✗",
            "✓" if t["l2"] else "✗",
            "✓" if t["l3"] else "✗",
            "✓" if t["l4"] else "✗",
            "✓" if t["l5"] else "✗",
            "✓" if t["l6"] else "✗",
            "✓" if t["l7"] else "✗",
            "✓" if t["l8"] else "✗",
        ])
        print(f"  {t['ticker']:<6} {t['ann']:12} {t['ret_stock']:>+6.1f}%  "
              f"{'  '.join(flags)}  {t['score']:>6.3f}  {t['rv30']:>4.0f}%  {t['mom5']:>+5.1f}%")

    # Filter hit rates on losing trades
    print(f"\n  How often each filter would have blocked a losing trade:")
    for lx, name in [("l1","above 50dma"),("l2","no prior gap-dn"),("l3","5d mom>0"),
                      ("l4","RV<60%"),("l5","vol>avg"),("l6","<25% from 52wk hi"),
                      ("l7","XLK>20dma"),("l8","score>=1.20")]:
        blocked = sum(1 for t in losses_s2 if not t[lx])
        blocked_winners = sum(1 for t in all_t if t["ret_stock"] > 0 and not t[lx])
        print(f"    {name:<22} blocks {blocked:>2}/{len(losses_s2)} losses  "
              f"({blocked/len(losses_s2)*100:.0f}%)  "
              f"but also removes {blocked_winners} winning trades")

    # ── S2 filter results ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  S2 + F6 — adding loss filters")
    print(f"  Baseline: {base_s2['n']} trades | {base_s2['wr']:.1f}% win | "
          f"avg {base_s2['avg']:+.2f}% | DD {base_s2['max_dd']:.1f}% | ${base_s2['final']:,.0f}")
    print(SEP)
    print(f"  {'Filter':<38} {'N':>4}  {'Win%':>9}  {'Avg%':>9}  {'MaxDD':>7}  {'Final':>10}  $10k")
    print(f"  {'-'*38} {'-'*4}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*10}  {'-'*10}")

    single_filters = [
        ("L1  Above 50dma",              lambda t: t["l1"]),
        ("L2  No prior earnings gap-dn", lambda t: t["l2"]),
        ("L3  5-day momentum > 0",       lambda t: t["l3"]),
        ("L4  RV < 60%",                 lambda t: t["l4"]),
        ("L5  Volume > 80% avg",         lambda t: t["l5"]),
        ("L6  Within 25% of 52wk high",  lambda t: t["l6"]),
        ("L7  XLK above 20dma",          lambda t: t["l7"]),
        ("L8  Score >= 1.20",            lambda t: t["l8"]),
    ]

    indiv_results_s2 = {}
    for name, fn in single_filters:
        r = simulate([t for t in all_t if fn(t)], "ret_stock")
        indiv_results_s2[name] = r
        prow(name, r, base_s2, "ret_stock")

    print(f"\n  — Combinations —")
    combos = [
        ("L1+L2  (50dma + no prior gap)",    lambda t: t["l1"] and t["l2"]),
        ("L1+L3  (50dma + 5d mom)",          lambda t: t["l1"] and t["l3"]),
        ("L2+L3  (no prior gap + 5d mom)",   lambda t: t["l2"] and t["l3"]),
        ("L1+L2+L3",                         lambda t: t["l1"] and t["l2"] and t["l3"]),
        ("L2+L6  (no prior gap + near hi)",  lambda t: t["l2"] and t["l6"]),
        ("L1+L6  (50dma + near hi)",         lambda t: t["l1"] and t["l6"]),
        ("L2+L8  (no prior gap + score1.2)", lambda t: t["l2"] and t["l8"]),
        ("L1+L2+L6",                         lambda t: t["l1"] and t["l2"] and t["l6"]),
        ("L1+L2+L3+L6",                      lambda t: t["l1"] and t["l2"] and t["l3"] and t["l6"]),
        ("L2+L3+L6+L7",                      lambda t: t["l2"] and t["l3"] and t["l6"] and t["l7"]),
        ("L1+L2+L3+L6+L8",                   lambda t: t["l1"] and t["l2"] and t["l3"] and t["l6"] and t["l8"]),
    ]

    combo_results_s2 = {}
    for name, fn in combos:
        sub = [t for t in all_t if fn(t)]
        if len(sub) < 8:
            print(f"  {name:<38} too few ({len(sub)})")
            continue
        r = simulate(sub, "ret_stock")
        combo_results_s2[name] = r
        prow(name, r, base_s2, "ret_stock")

    # ── S3 filter results ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  S3 + F6 — adding loss filters (stock + 10% calls)")
    print(f"  Baseline: {base_s3['n']} trades | {base_s3['wr']:.1f}% win | "
          f"avg {base_s3['avg']:+.2f}% | DD {base_s3['max_dd']:.1f}% | ${base_s3['final']:,.0f}")
    print(SEP)
    print(f"  {'Filter':<38} {'N':>4}  {'Win%':>9}  {'Avg%':>9}  {'MaxDD':>7}  {'Final':>10}  $10k")
    print(f"  {'-'*38} {'-'*4}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*10}  {'-'*10}")

    all_filters_s3 = single_filters + combos
    best_s3 = None
    best_s3_name = ""
    for name, fn in all_filters_s3:
        sub = [t for t in all_t if fn(t)]
        if len(sub) < 8: continue
        r = simulate(sub, "ret_blend")
        if best_s3 is None or (r["final"] > best_s3["final"] and r["max_dd"] <= base_s3["max_dd"] + 2):
            best_s3 = r; best_s3_name = name
        prow(name, r, base_s3, "ret_blend")

    # ── Best combination summary ─────────────────────────────────────────
    all_s2 = {**indiv_results_s2, **combo_results_s2}
    best_s2 = max(all_s2.items(), key=lambda kv: kv[1]["final"] / max(kv[1]["max_dd"], 1))

    print(f"\n{SEP}")
    print(f"  BEST RESULTS")
    print(SEP)
    for label, base, best_name, best_r, ret_key in [
        ("S2+F6 baseline", base_s2, None, None, "ret_stock"),
        ("S2+F6 best filter: " + best_s2[0][:28], base_s2, best_s2[0], best_s2[1], "ret_stock"),
        ("S3+F6 baseline", base_s3, None, None, "ret_blend"),
        ("S3+F6 best filter: " + (best_s3_name[:28] if best_s3 else "none"), base_s3, best_s3_name, best_s3, "ret_blend"),
    ]:
        r = best_r if best_r else base
        m5  = str(r["milestones"].get(5_000,  "—"))[:10]
        m10 = str(r["milestones"].get(10_000, "—"))[:10]
        m20 = str(r["milestones"].get(20_000, "—"))[:10]
        print(f"\n  {label}")
        print(f"    {r['n']} trades | {r['wr']:.1f}% win | avg {r['avg']:+.2f}% | "
              f"DD {r['max_dd']:.1f}% | ${r['final']:,.0f} ({r['x']:.1f}x)")
        print(f"    $5k: {m5}  $10k: {m10}  $20k: {m20}")
        if best_r and best_r != base:
            print(f"    vs baseline: Δfinal ${best_r['final']-base['final']:+,.0f}  "
                  f"Δwin {best_r['wr']-base['wr']:+.1f}%  Δdd {best_r['max_dd']-base['max_dd']:+.1f}%")
            print(f"    Year-by-year:")
            by_yr: dict[int,list] = {}
            for t in best_r["log"]:
                by_yr.setdefault(int(t["ann"][:4]), []).append(t[ret_key])
            cum = START_CASH
            for yr in sorted(by_yr):
                rets = by_yr[yr]
                avg  = sum(rets)/len(rets)
                w    = sum(1 for r_ in rets if r_ > 0)
                for r_ in rets: cum *= (1+r_/100)
                bar  = ("+" if avg>=0 else "-") + "█"*min(int(abs(avg)/2.5),22)
                print(f"      {yr}  {w}/{len(rets)} wins  avg {avg:>+5.1f}%  ${cum:>9,.0f}  {bar}")


if __name__ == "__main__":
    main()

