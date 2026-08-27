"""
S2 + MG Dynamic Combined Backtest
===================================
Single $2,000 account. S2 (pre-earnings drift) is the primary strategy.
MG (Monday Gap-Up) fills the idle capital during the ~70-day quiet periods.

Rules:
  1. S2 has absolute priority — if S2 signals, take it regardless of day.
  2. MG only fires when S2 holds NO open position (capital is idle).
  3. If both signal on the same date, S2 wins.
  4. MG is intraday (open → close). S2 is multi-week (D-20 → D-1).
  5. 100% capital deployment per trade, sequential.

Strategies compared:
  S2 alone         Pre-earnings drift only (baseline)
  MG alone         Monday gap-up only
  S2+MG dynamic   Single account, MG fills idle gaps
  S2+MG dynamic+  Same but with MG gap threshold ≥ 1.0% (higher quality)

Usage:
    uv run python -m backend.research.s2_mg_dynamic_backtest
"""

from __future__ import annotations

import sys
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)

import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP        = "=" * 72
START_CASH = 2_000.0
UNIVERSE   = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
MILESTONES = [5_000, 10_000, 20_000, 50_000]

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
# S2 pre-earnings trades (F6 + score≥1.20)
# ---------------------------------------------------------------------------

def build_s2_trades(qqq_df: pd.DataFrame) -> list[dict]:
    """
    Returns list of S2 trades with entry_date, exit_date, ret, ticker, score.
    Sequential simulation respects exit dates.
    """
    from backend.research.signal_backtest import fetch_earnings_dates

    BASE_QUALITY = {"GOOGL":1.40,"NVDA":1.50,"AMZN":1.20,"MSFT":1.10,"META":1.10,"AMD":1.00}
    SCORE_THRESH = 1.20

    def _nth(df, ref, offset):
        dt = pd.Timestamp(ref)
        d  = 1 if offset >= 0 else -1
        c  = 0
        for i in range(1, 300):
            cand = dt + pd.Timedelta(days=i * d)
            if cand in df.index:
                c += 1
                if c == abs(offset): return cand
        return None

    def _bounded(x): return max(-20., min(20., x)) / 100.

    def _score(ticker, entry_dt):
        df  = _px(ticker)
        col = df["Close"].loc[df.index <= entry_dt]
        if len(col) < 62: return BASE_QUALITY.get(ticker, 1.0)
        m20 = (col.iloc[-1] / col.iloc[-21] - 1) * 100
        m60 = (col.iloc[-1] / col.iloc[-62] - 1) * 100
        qc  = qqq_df["Close"].loc[qqq_df.index <= entry_dt]
        q20 = (qc.iloc[-1] / qc.iloc[-21] - 1) * 100 if len(qc) >= 21 else 0
        rs  = m20 - float(q20)
        return BASE_QUALITY.get(ticker, 1.0) + 1.20*_bounded(m20) + 0.80*_bounded(m60) + 1.50*_bounded(rs)

    def _regime_ok(dt):
        qc = qqq_df["Close"].loc[qqq_df.index <= dt]
        return len(qc) >= 150 and float(qc.iloc[-1]) > float(qc.rolling(150).mean().iloc[-1])

    def _gap_up(ticker, entry_dt):
        df = _px(ticker)
        if "Open" not in df.columns: return True
        prev = [d for d in df.index if d < entry_dt]
        if not prev: return True
        return float(df["Open"].loc[entry_dt]) > float(df["Close"].loc[prev[-1]])

    # Collect all candidates
    raw = []
    for ticker in UNIVERSE:
        df    = _px(ticker)
        dates = fetch_earnings_dates(ticker)
        for ann in dates:
            e_dt = _nth(df, ann, -20)
            x_dt = _nth(df, ann, -1)
            if e_dt is None or x_dt is None: continue
            if e_dt not in df.index or x_dt not in df.index: continue
            if not _regime_ok(e_dt): continue
            sc = _score(ticker, e_dt)
            if sc < SCORE_THRESH: continue
            if not _gap_up(ticker, e_dt): continue
            ep = float(df["Close"].loc[e_dt])
            xp = float(df["Close"].loc[x_dt])
            window  = [d for d in df.index if e_dt <= d <= x_dt]
            min_px  = min(float(df["Close"].loc[d]) for d in window)
            stopped = min_px <= ep * 0.95
            ret     = -5.0 if stopped else (xp - ep) / ep * 100
            raw.append({
                "ticker":     ticker,
                "entry_date": e_dt,
                "exit_date":  x_dt,
                "ret":        round(ret, 3),
                "score":      round(sc, 3),
                "ann":        ann,
                "strategy":   "S2",
            })

    # Pick best score per entry date
    raw.sort(key=lambda t: t["entry_date"])
    by_date: dict = {}
    for t in raw:
        d = t["entry_date"]
        if d not in by_date or t["score"] > by_date[d]["score"]:
            by_date[d] = t

    # Sequential — respect exit dates
    taken, busy = [], None
    for t in sorted(by_date.values(), key=lambda x: x["entry_date"]):
        if busy and t["entry_date"] <= busy:
            continue
        taken.append(t)
        busy = t["exit_date"]

    return taken


# ---------------------------------------------------------------------------
# MG Monday Gap-Up signals
# ---------------------------------------------------------------------------

def build_mg_signals(qqq_df: pd.DataFrame, min_gap: float = 0.50) -> list[dict]:
    """
    Returns list of all MG signal days (not yet filtered for S2 conflicts).
    Each row: date, ticker, trade_ret (open→close), gap_pct, score.
    """
    rows = []
    MA50_WINDOW = 50
    REGIME_MA   = 150

    qqq_close = qqq_df["Close"]

    for ticker in UNIVERSE:
        df  = _px(ticker)
        idx = df.index.tolist()

        for i in range(max(MA50_WINDOW, REGIME_MA), len(idx)):
            today = idx[i]
            prev  = idx[i - 1]
            if today.year < 2015: continue
            if today.day_name() != "Monday": continue

            o  = float(df["Open"].loc[today])  if "Open"  in df.columns else None
            c  = float(df["Close"].loc[today])
            pc = float(df["Close"].loc[prev])
            if o is None or pc == 0: continue

            gap_pct = (o - pc) / pc * 100
            if gap_pct < min_gap: continue

            intra_ret = (c - o) / o * 100

            # 50dma filter
            close_ser  = df["Close"].loc[df.index <= today]
            ma50       = float(close_ser.tail(MA50_WINDOW).mean())
            if o <= ma50: continue

            # Regime filter
            qc = qqq_close.loc[qqq_close.index <= today]
            if not (len(qc) >= REGIME_MA and
                    float(qc.iloc[-1]) > float(qc.rolling(REGIME_MA).mean().iloc[-1])):
                continue

            # Volume ratio for scoring
            vol_ratio = 1.0
            if "Volume" in df.columns:
                v       = float(df["Volume"].loc[today])
                vol_avg = float(df["Volume"].loc[df.index <= today].tail(21).iloc[:-1].mean())
                vol_ratio = v / vol_avg if vol_avg > 0 else 1.0

            rows.append({
                "ticker":     ticker,
                "entry_date": today,
                "exit_date":  today,   # intraday — same day
                "ret":        round(intra_ret, 3),
                "gap_pct":    round(gap_pct, 3),
                "score":      round(gap_pct * vol_ratio, 4),
                "strategy":   "MG",
            })

    # Per-day: keep highest-score ticker
    rows.sort(key=lambda r: r["entry_date"])
    by_date: dict = {}
    for r in rows:
        d = r["entry_date"]
        if d not in by_date or r["score"] > by_date[d]["score"]:
            by_date[d] = r

    return sorted(by_date.values(), key=lambda r: r["entry_date"])


# ---------------------------------------------------------------------------
# Dynamic combination — MG only when S2 is idle
# ---------------------------------------------------------------------------

def combine_dynamic(s2_trades: list[dict], mg_signals: list[dict]) -> list[dict]:
    """
    Merge S2 and MG into one trade list.
    MG fires only on dates when S2 holds no open position.
    S2 always wins on date conflicts.
    """
    # Build set of dates occupied by S2
    s2_busy_dates: set = set()
    for t in s2_trades:
        # Mark every calendar date from entry to exit as busy
        d = t["entry_date"]
        while d <= t["exit_date"]:
            s2_busy_dates.add(d)
            d += pd.Timedelta(days=1)

    # Filter MG: only take if not in any S2 window
    mg_filtered = [t for t in mg_signals if t["entry_date"] not in s2_busy_dates]

    # Merge and sort chronologically
    combined = sorted(s2_trades + mg_filtered, key=lambda t: t["entry_date"])
    return combined


# ---------------------------------------------------------------------------
# Simulation — sequential equity from trade list
# ---------------------------------------------------------------------------

def simulate(trades: list[dict], label: str) -> dict:
    equity = START_CASH
    peak   = START_CASH
    max_dd = 0.0
    wins   = 0
    rets   = []
    milestones: dict[int, object] = {}
    log    = []

    for t in trades:
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = t["entry_date"].date()
        ret = t["ret"]
        equity *= (1 + ret / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if ret > 0: wins += 1
        rets.append(ret)
        log.append({
            "ticker":   t["ticker"],
            "date":     t["entry_date"],
            "strategy": t.get("strategy", "?"),
            "ret":      ret,
            "equity":   round(equity, 2),
        })

    n    = len(rets)
    years = 11
    ann   = (equity / START_CASH) ** (1 / years) - 1

    return {
        "label":      label,
        "n":          n,
        "wins":       wins,
        "wr":         round(wins / n * 100, 1) if n else 0,
        "avg":        round(sum(rets) / n, 2) if n else 0,
        "final":      round(equity, 2),
        "ann":        round(ann * 100, 1),
        "max_dd":     round(max_dd, 1),
        "milestones": milestones,
        "log":        log,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prow(r: dict, base: dict) -> None:
    m10 = str(r["milestones"].get(10_000, "—"))[:10]
    m20 = str(r["milestones"].get(20_000, "—"))[:10]
    d_ann = r["ann"] - base["ann"]
    d_dd  = r["max_dd"] - base["max_dd"]
    flag  = "  ◄ BETTER" if d_ann >= 0 and d_dd <= 0 else \
            "  ↑ more return" if d_ann > 1 else \
            "  ↓ less return" if d_ann < -1 else ""
    print(f"  {r['label']:<42} {r['n']:>4}  {r['wr']:>5.1f}%  {r['avg']:>+6.2f}%  "
          f"{r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}  "
          f"{m10}  {m20}{flag}")


def year_by_year(r: dict) -> None:
    s2_yr: dict[int, list]  = {}
    mg_yr: dict[int, list]  = {}
    cum = START_CASH
    for t in r["log"]:
        yr = t["date"].year
        if t["strategy"] == "S2":
            s2_yr.setdefault(yr, []).append(t["ret"])
        else:
            mg_yr.setdefault(yr, []).append(t["ret"])
    all_yrs = sorted(set(list(s2_yr.keys()) + list(mg_yr.keys())))
    for yr in all_yrs:
        s2  = s2_yr.get(yr, [])
        mg  = mg_yr.get(yr, [])
        all_t = s2 + mg
        if not all_t: continue
        avg  = sum(all_t) / len(all_t)
        for r_ in all_t: cum *= (1 + r_ / 100)
        bar  = ("+" if avg >= 0 else "-") + "█" * min(int(abs(avg) / 2), 20)
        s2_str = f"S2:{len(s2)}" if s2 else "S2:0"
        mg_str = f"MG:{len(mg)}" if mg else "MG:0"
        print(f"    {yr}  {s2_str:>5}  {mg_str:>5}  avg {avg:>+6.2f}%  ${cum:>9,.0f}  {bar}")


def idle_analysis(s2_trades: list[dict], mg_filtered: list[dict]) -> None:
    """Show how much of the year S2 is idle and how many MG slots are available."""
    all_busy: set = set()
    for t in s2_trades:
        d = t["entry_date"]
        while d <= t["exit_date"]:
            all_busy.add(d)
            d += pd.Timedelta(days=1)

    from collections import Counter
    mg_by_yr: Counter = Counter()
    mg_taken_by_yr: Counter = Counter()
    all_mondays = [t for t in mg_filtered]

    for t in all_mondays:
        yr = t["entry_date"].year
        mg_taken_by_yr[yr] += 1

    print(f"  {'Year':>4}  {'S2 busy days':>13}  {'MG trades taken':>16}  {'Avg MG ret%':>12}")
    print(f"  {'-'*4}  {'-'*13}  {'-'*16}  {'-'*12}")
    for yr in range(2015, 2027):
        busy_yr = sum(1 for d in all_busy if d.year == yr)
        mg_yr   = mg_taken_by_yr.get(yr, 0)
        mg_rets = [t["ret"] for t in all_mondays if t["entry_date"].year == yr]
        avg_mg  = sum(mg_rets) / len(mg_rets) if mg_rets else 0
        print(f"  {yr:>4}  {busy_yr:>13}  {mg_yr:>16}  {avg_mg:>+11.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  S2 + MG DYNAMIC BACKTEST — single account, MG fills idle gaps")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ"]:
        _px(t)
        print(f"  {t} ready")
    qqq_df = _px("QQQ")

    # ── Build trade lists ─────────────────────────────────────────────────
    print("\nBuilding S2 trade list...")
    s2_trades = build_s2_trades(qqq_df)
    print(f"  S2 trades: {len(s2_trades)}")

    print("Building MG signal list (gap ≥ 0.5%)...")
    mg_05 = build_mg_signals(qqq_df, min_gap=0.50)
    print(f"  MG signals (≥0.5%): {len(mg_05)}")

    print("Building MG signal list (gap ≥ 1.0%)...")
    mg_10 = build_mg_signals(qqq_df, min_gap=1.00)
    print(f"  MG signals (≥1.0%): {len(mg_10)}")

    # ── Dynamic combinations ──────────────────────────────────────────────
    s2_mg_05 = combine_dynamic(s2_trades, mg_05)
    s2_mg_10 = combine_dynamic(s2_trades, mg_10)

    mg_only_05 = [t for t in s2_mg_05 if t["strategy"] == "MG"]
    mg_only_10 = [t for t in s2_mg_10 if t["strategy"] == "MG"]

    print(f"\n  MG trades taken after S2 filter (≥0.5%): {len(mg_only_05)}")
    print(f"  MG trades taken after S2 filter (≥1.0%): {len(mg_only_10)}")

    # ── Simulations ───────────────────────────────────────────────────────
    r_s2      = simulate(s2_trades, "S2 alone (baseline)")
    r_mg_05   = simulate(mg_05,     "MG alone (≥0.5% gap)")
    r_mg_10   = simulate(mg_10,     "MG alone (≥1.0% gap)")
    r_dyn_05  = simulate(s2_mg_05,  "S2 + MG dynamic (≥0.5%)")
    r_dyn_10  = simulate(s2_mg_10,  "S2 + MG dynamic (≥1.0%)")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  STRATEGY COMPARISON (2015–2026, $2,000 start)")
    print(SEP)
    print(f"  {'Strategy':<42} {'N':>4}  {'Win%':>5}  {'Avg%':>6}  "
          f"{'Ann%':>5}  {'DD%':>5}  {'Final':>10}  $10k          $20k")
    print(f"  {'-'*42} {'-'*4}  {'-'*5}  {'-'*6}  "
          f"{'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")

    base = r_s2
    for r in [r_s2, r_mg_05, r_mg_10, r_dyn_05, r_dyn_10]:
        prow(r, base)

    # ── Breakdown: how many S2 vs MG trades in dynamic ────────────────────
    print(f"\n{SEP}")
    print("  S2 + MG DYNAMIC — trade breakdown by strategy")
    print(SEP)
    for r, label in [(r_dyn_05, "≥0.5%"), (r_dyn_10, "≥1.0%")]:
        s2_t  = [t for t in r["log"] if t["strategy"] == "S2"]
        mg_t  = [t for t in r["log"] if t["strategy"] == "MG"]
        s2_wr = sum(1 for t in s2_t if t["ret"] > 0) / len(s2_t) * 100 if s2_t else 0
        mg_wr = sum(1 for t in mg_t if t["ret"] > 0) / len(mg_t) * 100 if mg_t else 0
        s2_avg = sum(t["ret"] for t in s2_t) / len(s2_t) if s2_t else 0
        mg_avg = sum(t["ret"] for t in mg_t) / len(mg_t) if mg_t else 0
        print(f"\n  Dynamic {label}:")
        print(f"    S2 trades:  n={len(s2_t):>3}  win {s2_wr:>4.0f}%  avg {s2_avg:>+6.2f}%")
        print(f"    MG trades:  n={len(mg_t):>3}  win {mg_wr:>4.0f}%  avg {mg_avg:>+6.2f}%")
        print(f"    MG adds {len(mg_t)} extra trades/yr avg ({len(mg_t)/11:.1f}/yr)")

    # ── Year-by-year ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  YEAR-BY-YEAR — S2 + MG Dynamic (≥0.5%)")
    print(SEP)
    print(f"  {'Year':>4}  {'S2':>5}  {'MG':>5}  {'Avg ret':>8}  {'Equity':>10}  Bar")
    year_by_year(r_dyn_05)

    print(f"\n{SEP}")
    print("  YEAR-BY-YEAR — S2 + MG Dynamic (≥1.0%)")
    print(SEP)
    print(f"  {'Year':>4}  {'S2':>5}  {'MG':>5}  {'Avg ret':>8}  {'Equity':>10}  Bar")
    year_by_year(r_dyn_10)

    # ── Idle analysis ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  IDLE CAPITAL ANALYSIS — how MG fills the gaps")
    print(SEP)
    idle_analysis(s2_trades, mg_only_05)

    # ── Worst drawdown analysis ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("  WORST 10 TRADES — S2 + MG Dynamic (≥0.5%)")
    print(SEP)
    worst = sorted(r_dyn_05["log"], key=lambda t: t["ret"])[:10]
    print(f"  {'Date':<12} {'Ticker':<6} {'Strat':<5} {'Return':>8}")
    for t in worst:
        print(f"  {str(t['date'].date()):<12} {t['ticker']:<6} {t['strategy']:<5} {t['ret']:>+7.2f}%")

    # ── MG contribution on S2-idle vs S2-active days ─────────────────────
    print(f"\n{SEP}")
    print("  MG QUALITY CHECK — does MG hurt or help in quiet periods?")
    print(SEP)
    mg_taken_05 = [t for t in s2_mg_05 if t["strategy"] == "MG"]
    if mg_taken_05:
        wr  = sum(1 for t in mg_taken_05 if t["ret"] > 0) / len(mg_taken_05) * 100
        avg = sum(t["ret"] for t in mg_taken_05) / len(mg_taken_05)
        best  = sorted(mg_taken_05, key=lambda t: -t["ret"])[:3]
        worst_t = sorted(mg_taken_05, key=lambda t: t["ret"])[:3]
        print(f"  MG trades during S2-idle:  n={len(mg_taken_05)}  win {wr:.0f}%  avg {avg:>+.3f}%/trade")
        print(f"  Best  MG:  " + "  ".join(f"{t['ticker']}({t['ret']:>+.1f}%)" for t in best))
        print(f"  Worst MG:  " + "  ".join(f"{t['ticker']}({t['ret']:>+.1f}%)" for t in worst_t))

    # ── Verdict ───────────────────────────────────────────────────────────
    best_dyn = r_dyn_05 if r_dyn_05["ann"] >= r_dyn_10["ann"] else r_dyn_10
    delta_ann = best_dyn["ann"] - r_s2["ann"]
    delta_dd  = best_dyn["max_dd"] - r_s2["max_dd"]

    print(f"\n{SEP}")
    print("  VERDICT")
    print(SEP)
    print(f"  S2 alone:          ann {r_s2['ann']:>+5.1f}%  dd {r_s2['max_dd']:>5.1f}%  "
          f"final ${r_s2['final']:>9,.0f}")
    print(f"  Best dynamic:      ann {best_dyn['ann']:>+5.1f}%  dd {best_dyn['max_dd']:>5.1f}%  "
          f"final ${best_dyn['final']:>9,.0f}  ({best_dyn['label']})")
    print(f"\n  Delta: return {delta_ann:>+.1f}%/yr  drawdown {delta_dd:>+.1f}%")

    if delta_ann >= 2 and delta_dd <= 3:
        print(f"\n  ✓ MG IMPROVES S2 — adds return without significant DD increase")
        print(f"    Recommendation: run MG on Mondays when S2 has no open position")
    elif delta_ann >= 2 and delta_dd > 3:
        print(f"\n  ~ MG ADDS RETURN but raises drawdown")
        print(f"    Consider only ≥1.0% gap filter to reduce MG trade frequency")
    elif delta_ann < 0:
        print(f"\n  ✗ MG HURTS — intraday noise drags down S2's cleaner returns")
        print(f"    Stick with S2 alone. The idle capital is fine in cash.")
    else:
        print(f"\n  ~ MARGINAL — MG adds {delta_ann:>+.1f}%/yr with {delta_dd:>+.1f}% DD change")
        print(f"    Not worth the complexity unless MG quality improves with higher gap threshold")


if __name__ == "__main__":
    main()

