"""
D-5 Concentrated Pre-Earnings Entry Backtest
==============================================
Hypothesis: the final 5 trading days before earnings (D-5 → D-1) carry
most of the pre-earnings drift. Entering later means:
  + Higher per-trade return % (concentrated window)
  + More trades/year (each cycle is 5d not 20d → more cycles fit in a year)
  + Less exposure (5d in market vs 20d)
  - Fewer setups qualify (less time for score/gap filters to align)

Tests:
  1. Day-by-day contribution: which entry day gives the best return?
     (D-20, D-15, D-10, D-7, D-5, D-3, D-2)
  2. D-5 strategy simulation vs S2 (D-20)
  3. Combined: D-20 for main entry + D-5 for missed cycles
  4. D-5 without any filters vs with regime/score

All exit at D-1. Stop-loss -5% (same as S2).
Universe + regime filter same as S2.

Usage:
    uv run python -m backend.research.d5_entry_backtest
"""

from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP        = "=" * 72
START_CASH = 2_000.0
UNIVERSE   = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
MILESTONES = [5_000, 10_000, 20_000, 50_000]
BASE_QUALITY = {"GOOGL":1.40,"NVDA":1.50,"AMZN":1.20,"MSFT":1.10,"META":1.10,"AMD":1.00}
SCORE_THRESH = 1.20

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
    dt = pd.Timestamp(ref); d = 1 if offset >= 0 else -1; c = 0
    for i in range(1, 300):
        cand = dt + pd.Timedelta(days=i*d)
        if cand in df.index:
            c += 1
            if c == abs(offset): return cand
    return None

def _bounded(x): return max(-20., min(20., x)) / 100.

def _score(ticker, entry_dt, qqq_df):
    df  = _px(ticker)
    col = df["Close"].loc[df.index <= entry_dt]
    if len(col) < 62: return BASE_QUALITY.get(ticker, 1.0)
    m20 = (col.iloc[-1]/col.iloc[-21]-1)*100
    m60 = (col.iloc[-1]/col.iloc[-62]-1)*100
    qc  = qqq_df["Close"].loc[qqq_df.index <= entry_dt]
    q20 = (qc.iloc[-1]/qc.iloc[-21]-1)*100 if len(qc) >= 21 else 0
    rs  = m20 - float(q20)
    return BASE_QUALITY.get(ticker,1.0)+1.20*_bounded(m20)+0.80*_bounded(m60)+1.50*_bounded(rs)

def _regime_ok(dt, qqq_df):
    qc = qqq_df["Close"].loc[qqq_df.index <= dt]
    return len(qc) >= 150 and float(qc.iloc[-1]) > float(qc.rolling(150).mean().iloc[-1])

def _gap_up(ticker, entry_dt):
    df = _px(ticker)
    if "Open" not in df.columns: return True
    prev = [d for d in df.index if d < entry_dt]
    if not prev: return True
    return float(df["Open"].loc[entry_dt]) > float(df["Close"].loc[prev[-1]])


# ---------------------------------------------------------------------------
# Build all pre-earnings trades with multiple entry points
# ---------------------------------------------------------------------------

def build_all_entries(qqq_df: pd.DataFrame) -> list[dict]:
    """
    For each earnings event, build one trade record per entry day
    (D-20, D-15, D-10, D-7, D-5, D-3, D-2).
    All exit at D-1. Stop -5%.
    """
    from backend.research.signal_backtest import fetch_earnings_dates

    entry_offsets = [20, 15, 10, 7, 5, 3, 2]
    trades = []

    for ticker in UNIVERSE:
        df    = _px(ticker)
        dates = fetch_earnings_dates(ticker)

        for ann in dates:
            x_dt = _nth(df, ann, -1)
            if x_dt is None or x_dt not in df.index: continue

            xp = float(df["Close"].loc[x_dt])

            for offset in entry_offsets:
                e_dt = _nth(df, ann, -offset)
                if e_dt is None or e_dt not in df.index: continue
                if e_dt.year < 2015: continue

                if not _regime_ok(e_dt, qqq_df): continue

                ep  = float(df["Close"].loc[e_dt])
                sc  = _score(ticker, e_dt, qqq_df)
                gap = _gap_up(ticker, e_dt)

                # Compute return with stop
                window  = [d for d in df.index if e_dt <= d <= x_dt]
                min_px  = min(float(df["Close"].loc[d]) for d in window) if window else ep
                stopped = min_px <= ep * 0.95
                ret     = -5.0 if stopped else (xp - ep) / ep * 100

                trades.append({
                    "ticker":  ticker,
                    "ann":     ann,
                    "entry_d": -offset,    # D-20, D-15, etc.
                    "entry_dt": e_dt,
                    "exit_dt":  x_dt,
                    "ret":     round(ret, 3),
                    "score":   round(sc, 3),
                    "gap_up":  gap,
                    "stopped": stopped,
                })

    return sorted(trades, key=lambda t: t["entry_dt"])


# ---------------------------------------------------------------------------
# Simulate sequential equity for a filtered trade list
# ---------------------------------------------------------------------------

def simulate(trades: list[dict], label: str) -> dict:
    # Sequential: respect exit dates, pick best score on same entry date
    by_date: dict = {}
    for t in trades:
        d = t["entry_dt"]
        if d not in by_date or t["score"] > by_date[d]["score"]:
            by_date[d] = t

    taken, busy = [], None
    for t in sorted(by_date.values(), key=lambda x: x["entry_dt"]):
        if busy and t["entry_dt"] <= busy: continue
        taken.append(t)
        busy = t["exit_dt"]

    equity = START_CASH; peak = START_CASH; max_dd = 0.0
    wins = 0; rets = []; milestones = {}; log = []

    for t in taken:
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = t["entry_dt"].date()
        equity *= (1 + t["ret"] / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if t["ret"] > 0: wins += 1
        rets.append(t["ret"])
        log.append({"ticker": t["ticker"], "date": t["entry_dt"],
                    "ret": t["ret"], "equity": round(equity, 2)})

    n = len(rets); years = 11
    ann = (equity / START_CASH) ** (1/years) - 1 if n > 0 else 0
    return {
        "label": label, "n": n, "wins": wins,
        "wr":  round(wins/n*100, 1) if n else 0,
        "avg": round(sum(rets)/n, 2) if n else 0,
        "final": round(equity, 2),
        "ann":  round(ann*100, 1),
        "max_dd": round(max_dd, 1),
        "milestones": milestones, "log": log,
    }


def prow(r: dict, base: dict) -> None:
    m10 = str(r["milestones"].get(10_000,"—"))[:10]
    m20 = str(r["milestones"].get(20_000,"—"))[:10]
    d_ann = r["ann"] - base["ann"]
    d_dd  = r["max_dd"] - base["max_dd"]
    flag  = "  ◄ BETTER"      if d_ann >= 0 and d_dd <= 0 else \
            "  ◄ SWEET SPOT"  if d_ann >= 2 and d_dd <= 3  else \
            "  ↑ more return" if d_ann > 1  else \
            "  ↓ less return" if d_ann < -1 else ""
    print(f"  {r['label']:<44} {r['n']:>4}  {r['wr']:>5.1f}%  {r['avg']:>+6.2f}%  "
          f"{r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}  "
          f"{m10}  {m20}{flag}")


def year_by_year(r: dict) -> None:
    by_yr: dict = {}
    for t in r["log"]:
        by_yr.setdefault(t["date"].year, []).append(t["ret"])
    cum = START_CASH
    for yr in sorted(by_yr):
        rets = by_yr[yr]
        avg  = sum(rets)/len(rets)
        w    = sum(1 for r_ in rets if r_ > 0)
        for r_ in rets: cum *= (1+r_/100)
        bar = ("+" if avg >= 0 else "-") + "█" * min(int(abs(avg)/2), 20)
        print(f"    {yr}  {w:>2}/{len(rets):>2} wins  avg {avg:>+6.2f}%  ${cum:>9,.0f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  D-5 CONCENTRATED PRE-EARNINGS ENTRY BACKTEST (2015–2026, $2,000)")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ"]:
        _px(t); print(f"  {t} ready")
    qqq_df = _px("QQQ")

    print("\nBuilding all entry-day trades (this takes ~1 min)...")
    all_trades = build_all_entries(qqq_df)
    print(f"  Total raw records: {len(all_trades)}")

    # ── 1. Day-by-day contribution ────────────────────────────────────────
    print(f"\n{SEP}")
    print("  1. RETURN BY ENTRY DAY — raw stats (no sequential filter, regime only)")
    print(SEP)
    print(f"  {'Entry':>8}  {'N':>4}  {'Win%':>5}  {'Avg%':>7}  "
          f"{'Hold days':>10}  {'Best ticker':>12}  Notes")
    print(f"  {'-'*8}  {'-'*4}  {'-'*5}  {'-'*7}  {'-'*10}  {'-'*12}")

    for offset in [20, 15, 10, 7, 5, 3, 2]:
        sub = [t for t in all_trades if t["entry_d"] == -offset]
        if not sub: continue
        wr  = sum(1 for t in sub if t["ret"] > 0) / len(sub) * 100
        avg = sum(t["ret"] for t in sub) / len(sub)
        # avg/day
        apd = avg / offset
        top = max(sub, key=lambda t: t["ret"])
        flag = "  ◄ 3% target" if avg >= 3.0 else \
               "  ◄ close"     if avg >= 2.0 else ""
        print(f"  {'D-'+str(offset):>8}  {len(sub):>4}  {wr:>4.0f}%  {avg:>+6.2f}%  "
              f"{offset:>10}d  {top['ticker']}({top['ret']:>+.0f}%){flag}")

    # ── 2. Filter impact on D-5 ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("  2. D-5 FILTER COMPARISON — regime / score / gap impact")
    print(SEP)
    print(f"  {'Filters':<30}  {'N':>4}  {'Win%':>5}  {'Avg%':>7}  Notes")
    print(f"  {'-'*30}  {'-'*4}  {'-'*5}  {'-'*7}")

    d5_all    = [t for t in all_trades if t["entry_d"] == -5]
    d5_score  = [t for t in d5_all if t["score"] >= SCORE_THRESH]
    d5_gap    = [t for t in d5_all if t["gap_up"]]
    d5_both   = [t for t in d5_all if t["score"] >= SCORE_THRESH and t["gap_up"]]

    for label, sub in [
        ("No extra filters (regime only)", d5_all),
        (f"Score ≥ {SCORE_THRESH}",        d5_score),
        ("Gap-up only",                     d5_gap),
        (f"Score + Gap-up",                 d5_both),
    ]:
        if not sub: continue
        wr  = sum(1 for t in sub if t["ret"] > 0) / len(sub) * 100
        avg = sum(t["ret"] for t in sub) / len(sub)
        flag = "  ◄ best" if avg >= 3.0 and wr >= 65 else ""
        print(f"  {label:<30}  {len(sub):>4}  {wr:>4.0f}%  {avg:>+6.2f}%{flag}")

    # ── 3. Per-ticker D-5 breakdown ───────────────────────────────────────
    print(f"\n{SEP}")
    print("  3. D-5 PER-TICKER — which tickers drive the 5-day edge?")
    print(SEP)
    print(f"  {'Ticker':<8}  {'N':>4}  {'Win%':>5}  {'Avg%':>7}  "
          f"{'Avg/day':>8}  {'Best':>7}  {'Worst':>7}")
    print(f"  {'-'*8}  {'-'*4}  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}")

    for ticker in UNIVERSE:
        sub = [t for t in d5_all if t["ticker"] == ticker]
        if not sub: continue
        wr   = sum(1 for t in sub if t["ret"] > 0) / len(sub) * 100
        avg  = sum(t["ret"] for t in sub) / len(sub)
        best = max(t["ret"] for t in sub)
        worst= min(t["ret"] for t in sub)
        flag = "  ◄" if avg >= 3.0 else ""
        print(f"  {ticker:<8}  {len(sub):>4}  {wr:>4.0f}%  {avg:>+6.2f}%  "
              f"{avg/5:>+7.2f}%  {best:>+6.1f}%  {worst:>+6.1f}%{flag}")

    # ── 4. Full simulation comparison ────────────────────────────────────
    print(f"\n{SEP}")
    print("  4. SIMULATION COMPARISON (2015–2026, $2,000, sequential)")
    print(SEP)
    print(f"  {'Strategy':<44} {'N':>4}  {'Win%':>5}  {'Avg%':>6}  "
          f"{'Ann%':>5}  {'DD%':>5}  {'Final':>10}  $10k          $20k")
    print(f"  {'-'*44} {'-'*4}  {'-'*5}  {'-'*6}  "
          f"{'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")

    # S2 baseline (D-20, score + gap)
    s2_trades = [t for t in all_trades
                 if t["entry_d"] == -20
                 and t["score"] >= SCORE_THRESH
                 and t["gap_up"]]
    r_s2 = simulate(s2_trades, "S2 baseline (D-20, score+gap)")

    # D-5 variants
    configs = [
        ("D-5 no filters (regime only)",
            [t for t in all_trades if t["entry_d"] == -5]),
        ("D-5 score ≥ 1.20",
            [t for t in all_trades if t["entry_d"] == -5 and t["score"] >= SCORE_THRESH]),
        ("D-5 gap-up only",
            [t for t in all_trades if t["entry_d"] == -5 and t["gap_up"]]),
        ("D-5 score + gap-up",
            [t for t in all_trades if t["entry_d"] == -5
             and t["score"] >= SCORE_THRESH and t["gap_up"]]),
        ("D-7 score + gap-up",
            [t for t in all_trades if t["entry_d"] == -7
             and t["score"] >= SCORE_THRESH and t["gap_up"]]),
        ("D-10 score + gap-up",
            [t for t in all_trades if t["entry_d"] == -10
             and t["score"] >= SCORE_THRESH and t["gap_up"]]),
        # NVDA + AMD only at D-5 (high-vol tickers)
        ("D-5 NVDA+AMD only (score+gap)",
            [t for t in all_trades if t["entry_d"] == -5
             and t["ticker"] in ["NVDA","AMD"]
             and t["score"] >= SCORE_THRESH and t["gap_up"]]),
    ]

    base = r_s2
    prow(r_s2, base)
    print()

    results = []
    for label, filtered in configs:
        if len(filtered) < 3:
            print(f"  {label:<44} too few ({len(filtered)})")
            continue
        r = simulate(filtered, label)
        results.append(r)
        prow(r, base)

    # ── 5. Combined: D-20 primary + D-5 catch missed cycles ──────────────
    print(f"\n{SEP}")
    print("  5. COMBINED STRATEGY — D-20 primary, D-5 for missed earnings cycles")
    print(SEP)
    print("  Logic: enter at D-20 when possible; if missed (was in another trade),")
    print("  enter at D-5 on the same ticker as a second chance.")
    print()

    # Build combined: D-20 first, then D-5 fills gaps
    d20_trades = [t for t in all_trades
                  if t["entry_d"] == -20
                  and t["score"] >= SCORE_THRESH and t["gap_up"]]
    d5_trades  = [t for t in all_trades
                  if t["entry_d"] == -5
                  and t["score"] >= SCORE_THRESH]

    # Combine: try D-20 first, D-5 if capital was busy
    combined_raw = sorted(d20_trades + d5_trades, key=lambda t: t["entry_dt"])
    # Deduplicate: if both D-20 and D-5 exist for same (ticker, ann), prefer D-20
    seen_ann: set = set()
    combined_dedup = []
    for t in combined_raw:
        key = (t["ticker"], str(t["ann"]))
        if t["entry_d"] == -20:
            seen_ann.add(key)
            combined_dedup.append(t)
        elif key not in seen_ann:   # D-5 only if D-20 wasn't taken
            combined_dedup.append(t)

    r_combined = simulate(combined_dedup, "D-20 primary + D-5 catch-up")
    prow(r_combined, base)

    # ── 6. Best result deep dive ──────────────────────────────────────────
    all_results = results + [r_combined]
    better = [r for r in all_results
              if r["ann"] >= base["ann"] and r["max_dd"] <= base["max_dd"]]
    best   = max(all_results, key=lambda r: r["ann"] / max(r["max_dd"], 1))

    print(f"\n{SEP}")
    if better:
        print(f"  ✓ D-5 STRATEGY WORKS — {len(better)} variant(s) beat S2 baseline:")
        for r in sorted(better, key=lambda x: -x["ann"]):
            print(f"    {r['label']}")
            print(f"      ann {r['ann']:>+5.1f}%  dd {r['max_dd']:>5.1f}%  "
                  f"n={r['n']}  avg {r['avg']:>+.2f}%/trade")
    else:
        print(f"  Best risk-adjusted D-5 variant: {best['label']}")
        print(f"  ann {best['ann']:>+5.1f}%  dd {best['max_dd']:>5.1f}%  avg {best['avg']:>+.2f}%/trade")

    print(f"\n  Year-by-year — {best['label']}:")
    year_by_year(best)

    # ── 7. Return distribution analysis ──────────────────────────────────
    print(f"\n{SEP}")
    print("  6. D-5 RETURN DISTRIBUTION — how often does 3% happen?")
    print(SEP)

    d5_filtered = [t for t in all_trades
                   if t["entry_d"] == -5 and t["score"] >= SCORE_THRESH]
    if d5_filtered:
        rets = [t["ret"] for t in d5_filtered]
        buckets = [
            ("> +10%",   lambda r: r > 10),
            ("+5 to 10%",lambda r: 5 < r <= 10),
            ("+3 to 5%", lambda r: 3 < r <= 5),
            ("+0 to 3%", lambda r: 0 < r <= 3),
            ("Stop (-5%)",lambda r: r == -5.0),
            ("-5 to 0%", lambda r: -5 < r <= 0),
        ]
        print(f"  {'Return bucket':<14}  {'N':>4}  {'%':>5}  Bar")
        for label, fn in buckets:
            sub = [r for r in rets if fn(r)]
            pct = len(sub)/len(rets)*100
            bar = "█" * int(pct/3)
            print(f"  {label:<14}  {len(sub):>4}  {pct:>4.0f}%  {bar}")

        above3 = sum(1 for r in rets if r >= 3.0)
        print(f"\n  Trades reaching 3%+ target: {above3}/{len(rets)} = "
              f"{above3/len(rets)*100:.0f}%")
        print(f"  Avg return: {sum(rets)/len(rets):>+.2f}%  "
              f"Median: {sorted(rets)[len(rets)//2]:>+.2f}%")

    # ── 8. Verdict ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VERDICT — is D-5 worth pursuing?")
    print(SEP)
    d5_r = simulate([t for t in all_trades if t["entry_d"] == -5
                     and t["score"] >= SCORE_THRESH], "D-5 score≥1.20")
    print(f"\n  S2 (D-20):     avg {r_s2['avg']:>+.2f}%/trade  "
          f"ann {r_s2['ann']:>+.1f}%  n={r_s2['n']}")
    print(f"  D-5:           avg {d5_r['avg']:>+.2f}%/trade  "
          f"ann {d5_r['ann']:>+.1f}%  n={d5_r['n']}")
    print(f"  D-20+D-5:      avg {r_combined['avg']:>+.2f}%/trade  "
          f"ann {r_combined['ann']:>+.1f}%  n={r_combined['n']}")

    avg_d5 = d5_r["avg"]
    if avg_d5 >= 3.0:
        print(f"\n  ✓ D-5 REACHES 3% TARGET — avg {avg_d5:>+.2f}%/trade in 5 days")
        print(f"    = {avg_d5/5:.2f}%/day  = {avg_d5/5*252:.0f}%/yr if repeated daily (theoretical)")
    elif avg_d5 >= 2.0:
        print(f"\n  ~ D-5 CLOSE but not quite — avg {avg_d5:>+.2f}% in 5 days")
        print(f"    Try NVDA+AMD only (highest momentum) for the 3% target")
    else:
        print(f"\n  ✗ D-5 doesn't hit 3% avg — {avg_d5:>+.2f}%/trade")
        print(f"    The edge is front-loaded at D-20 entry, not D-5")
        print(f"    Recommendation: keep S2 (D-20) as primary strategy")


if __name__ == "__main__":
    main()

