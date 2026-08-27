"""
Signal Backtest — validates today's active signals against full history.

Auto-fetches earnings dates from yfinance (24 quarters back) and extends
with hardcoded history where available.  Validates against earnings_dataset.json.

Signals checked:
  1. Pre-earnings D-20 window for every ticker we have data on
  2. XLK + XLE sector rotation (current top-2 by 3m momentum)

Usage:
    uv run python -m backend.research.signal_backtest
"""

from __future__ import annotations

import json
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
RESEARCH_DIR = Path(__file__).parent

ENTRY_OFFSET = -20   # buy 20 trading days before earnings
EXIT_OFFSET  = -1    # sell 1 trading day before earnings

# Extended hardcoded history for tickers where yfinance only goes back 24 quarters
_EXTENDED_DATES: dict[str, list[str]] = {
    "NVDA": [
        "2015-08-20","2015-11-05","2016-02-17","2016-05-11","2016-08-11","2016-11-10",
        "2017-02-09","2017-05-09","2017-08-10","2017-11-09","2018-02-08","2018-05-10",
        "2018-08-16","2018-11-15","2019-02-14","2019-05-16","2019-08-15","2019-11-14",
    ],
    "CRM": [
        "2015-02-25","2015-05-20","2015-08-26","2015-11-18",
        "2016-02-24","2016-05-18","2016-08-24","2016-11-16",
        "2017-02-27","2017-05-17","2017-08-23","2017-11-21",
        "2018-02-28","2018-05-29","2018-08-28","2018-11-19",
        "2019-02-26","2019-05-29","2019-08-22","2019-11-19",
    ],
    "AAPL": [
        "2015-01-27","2015-04-27","2015-07-21","2015-10-27",
        "2016-01-26","2016-04-26","2016-07-26","2016-10-25",
        "2017-02-01","2017-05-02","2017-08-01","2017-11-02",
        "2018-02-01","2018-05-01","2018-07-31","2018-11-01",
        "2019-01-29","2019-04-30","2019-07-30","2019-10-30",
    ],
    "MSFT": [
        "2015-01-26","2015-04-23","2015-07-21","2015-10-22",
        "2016-01-28","2016-04-21","2016-07-19","2016-10-20",
        "2017-01-26","2017-04-27","2017-07-20","2017-10-26",
        "2018-01-31","2018-04-26","2018-07-19","2018-10-24",
        "2019-01-30","2019-04-24","2019-07-18","2019-10-23",
    ],
    "META": [
        "2015-01-28","2015-04-22","2015-07-29","2015-11-04",
        "2016-01-27","2016-04-27","2016-07-27","2016-11-02",
        "2017-02-01","2017-05-03","2017-07-26","2017-11-01",
        "2018-01-31","2018-04-25","2018-07-25","2018-10-31",
        "2019-01-30","2019-04-24","2019-07-24","2019-10-30",
    ],
    "GOOGL": [
        "2015-01-29","2015-04-23","2015-07-16","2015-10-22",
        "2016-01-28","2016-04-21","2016-07-28","2016-10-27",
        "2017-02-01","2017-04-27","2017-07-24","2017-10-26",
        "2018-02-01","2018-04-23","2018-07-23","2018-10-25",
        "2019-02-04","2019-04-29","2019-07-25","2019-10-28",
    ],
}

# ---------------------------------------------------------------------------
# Earnings date fetching
# ---------------------------------------------------------------------------

def fetch_earnings_dates(ticker: str) -> list[str]:
    """Auto-fetch from yfinance + merge extended hardcoded history, deduped & sorted."""
    yf_dates: list[str] = []
    try:
        ed = yf.Ticker(ticker).earnings_dates
        if ed is not None and not ed.empty:
            yf_dates = sorted(
                str(d.date()) for d in ed.index
                if str(d.date()) <= date.today().isoformat()
            )
    except Exception:
        pass

    extended = _EXTENDED_DATES.get(ticker.upper(), [])
    all_dates = sorted(set(yf_dates + extended))
    return all_dates


def validate_against_dataset(
    ticker: str,
    fetched: list[str],
    known: list[dict],
) -> tuple[int, int, list[str]]:
    """
    Compare fetched dates vs earnings_dataset.json for this ticker.
    Returns (matches, mismatches, discrepancy_notes).
    Allows ±1 calendar day tolerance (some reports land after-hours vs pre-market).
    """
    known_dates = sorted(ev["ann_date"] for ev in known if ev["ticker"] == ticker)
    matches, mismatches, notes = 0, 0, []

    for kd in known_dates:
        # Check exact or ±1 day
        kts = pd.Timestamp(kd)
        close = [f for f in fetched
                 if abs((pd.Timestamp(f) - kts).days) <= 1]
        if close:
            if close[0] != kd:
                notes.append(f"  {kd} → fetched as {close[0]} (±1d ok)")
            matches += 1
        else:
            notes.append(f"  {kd} MISSING from fetched dates")
            mismatches += 1

    return matches, mismatches, notes


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

_price_cache: dict[str, pd.DataFrame] = {}


def _prices(ticker: str) -> pd.DataFrame:
    if ticker not in _price_cache:
        df = yf.download(ticker, start="2013-01-01", end="2026-05-06",
                         interval="1d", progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
        _price_cache[ticker] = df
    return _price_cache[ticker]


def _nth_trading_day(df: pd.DataFrame, ref: str, offset: int) -> pd.Timestamp | None:
    dt        = pd.Timestamp(ref)
    direction = 1 if offset >= 0 else -1
    steps     = abs(offset)
    count     = 0
    for i in range(1, 300):
        cand = dt + pd.Timedelta(days=i * direction)
        if cand in df.index:
            count += 1
            if count == steps:
                return cand
    return None


# ---------------------------------------------------------------------------
# Pre-earnings backtest (single ticker)
# ---------------------------------------------------------------------------

def run_pre_earnings(ticker: str, known_evs: list[dict]) -> dict:
    """Run D-20 backtest and return summary dict."""
    dates = fetch_earnings_dates(ticker)
    df    = _prices(ticker)

    matches, mismatches, disc_notes = validate_against_dataset(ticker, dates, known_evs)

    trades = []
    for ann in dates:
        entry_dt = _nth_trading_day(df, ann, ENTRY_OFFSET)
        exit_dt  = _nth_trading_day(df, ann, EXIT_OFFSET)
        if entry_dt is None or exit_dt is None:
            continue
        if entry_dt not in df.index or exit_dt not in df.index:
            continue
        ep = float(df["Close"].loc[entry_dt])
        xp = float(df["Close"].loc[exit_dt])
        ret = (xp - ep) / ep * 100
        trades.append({
            "ann":    ann,
            "entry":  entry_dt.strftime("%Y-%m-%d"),
            "exit":   exit_dt.strftime("%Y-%m-%d"),
            "ret":    round(ret, 2),
            "win":    ret > 0,
        })

    if not trades:
        return {"ticker": ticker, "trades": 0}

    rets   = [t["ret"] for t in trades]
    wins   = [t for t in trades if t["win"]]
    avg    = sum(rets) / len(rets)
    std    = math.sqrt(sum((r - avg)**2 for r in rets) / len(rets)) if len(rets) > 1 else 0
    rr     = round(avg / std, 2) if std > 0 else 0

    # Compound $10k
    port = 10_000.0
    for t in trades:
        port *= (1 + t["ret"] / 100)

    return {
        "ticker":    ticker,
        "trades":    len(trades),
        "first_yr":  date.fromisoformat(trades[0]["ann"]).year,
        "last_yr":   date.fromisoformat(trades[-1]["ann"]).year,
        "win_rate":  round(len(wins) / len(trades) * 100, 1),
        "avg_ret":   round(avg, 2),
        "std":       round(std, 2),
        "rr":        rr,
        "best":      round(max(rets), 1),
        "worst":     round(min(rets), 1),
        "median":    round(sorted(rets)[len(rets) // 2], 1),
        "final_10k": round(port, 0),
        "matches":   matches,
        "mismatches":mismatches,
        "disc_notes":disc_notes,
        "recent":    trades[-8:],
        "all_trades":trades,
    }


def print_pre_earnings(r: dict) -> None:
    print(f"\n{DASH}")
    print(f"  {r['ticker']}  ({r['first_yr']}–{r['last_yr']}, {r['trades']} trades)")
    print(f"  Validation vs dataset: {r['matches']} match, {r['mismatches']} missing")
    for n in r.get("disc_notes", []):
        print(f"    {n}")
    if r["trades"] == 0:
        print("  No tradeable events")
        return

    grade = ("A" if r["win_rate"] >= 75 and r["avg_ret"] >= 4 else
             "B" if r["win_rate"] >= 65 and r["avg_ret"] >= 2 else
             "C" if r["win_rate"] >= 55 else "D")

    print(f"  Win rate: {r['win_rate']:.0f}%  Avg: {r['avg_ret']:+.2f}%  "
          f"Std: {r['std']:.2f}%  R/R: {r['rr']:.2f}  Grade: {grade}")
    print(f"  Best: {r['best']:+.1f}%  Worst: {r['worst']:+.1f}%  Median: {r['median']:+.1f}%")
    print(f"  $10k → ${r['final_10k']:>10,.0f}")

    # Year-by-year
    years: dict[int, list] = defaultdict(list)
    for t in r["all_trades"]:
        years[date.fromisoformat(t["ann"]).year].append(t["ret"])
    print(f"  Year breakdown:")
    for y in sorted(years):
        yr  = years[y]
        avg = sum(yr) / len(yr)
        w   = sum(1 for v in yr if v > 0)
        bar = ("+" if avg >= 0 else "-") + "█" * min(int(abs(avg) / 2), 20)
        print(f"    {y}: {w}/{len(yr)} wins  avg {avg:+.1f}%  {bar}")

    # Recent trades
    print(f"  Last 8 trades:")
    print(f"  {'Earnings':12} {'Entry':12} {'Exit':12} {'Ret%':>7}")
    for t in reversed(r["recent"]):
        flag = "✓" if t["win"] else "✗"
        print(f"  {t['ann']:12} {t['entry']:12} {t['exit']:12} {t['ret']:>+6.1f}%  {flag}")


# ---------------------------------------------------------------------------
# Sector rotation backtest
# ---------------------------------------------------------------------------

SECTOR_ETFS = ["XLK","XLF","XLE","XLV","XLY","XLU","XLI","XLB","XLC","XLRE"]


def run_sector_backtest(focus: list[str]) -> None:
    print(f"\n{DASH}")
    print(f"  SECTOR ROTATION — Top-2 by 3m momentum, monthly rebalance")
    print(f"  Current signal: {'+'.join(focus)}")
    print(DASH)

    data = {etf: _prices(etf) for etf in SECTOR_ETFS}
    spy  = _prices("SPY")

    months = pd.date_range("2018-01-01", "2026-04-01", freq="MS")

    def _first(month: pd.Timestamp) -> pd.Timestamp | None:
        end  = month + pd.DateOffset(months=1)
        cands = [d for d in spy.index if month <= d < end]
        return cands[0] if cands else None

    results, port, spy_val = [], 10_000.0, 10_000.0
    focus_months = []

    for i in range(len(months) - 1):
        e_dt = _first(months[i])
        x_dt = _first(months[i + 1])
        if e_dt is None or x_dt is None:
            continue

        lb = e_dt - pd.DateOffset(months=3)
        scores = {}
        for etf, df in data.items():
            p0 = [d for d in df.index if d <= lb]
            p1 = [d for d in df.index if d <= e_dt]
            if p0 and p1:
                scores[etf] = (float(df["Close"].loc[p1[-1]]) -
                               float(df["Close"].loc[p0[-1]])) / float(df["Close"].loc[p0[-1]]) * 100

        if len(scores) < 2:
            continue
        top = sorted(scores, key=lambda e: scores[e], reverse=True)[:2]

        rets = []
        for etf in top:
            df = data[etf]
            ep = [d for d in df.index if d <= e_dt]
            xp = [d for d in df.index if d <= x_dt]
            if ep and xp:
                rets.append((float(df["Close"].loc[xp[-1]]) -
                             float(df["Close"].loc[ep[-1]])) / float(df["Close"].loc[ep[-1]]) * 100)

        if not rets:
            continue
        mr = sum(rets) / len(rets)
        port *= (1 + mr / 100)

        spy_ep = [d for d in spy.index if d <= e_dt]
        spy_xp = [d for d in spy.index if d <= x_dt]
        sr = 0.0
        if spy_ep and spy_xp:
            sr = (float(spy["Close"].loc[spy_xp[-1]]) -
                  float(spy["Close"].loc[spy_ep[-1]])) / float(spy["Close"].loc[spy_ep[-1]]) * 100
        spy_val *= (1 + sr / 100)

        is_focus = set(top) == set(focus)
        if is_focus:
            focus_months.append({"month": months[i].strftime("%Y-%m"), "ret": mr, "spy": sr})

        results.append({
            "month": months[i].strftime("%Y-%m"),
            "top": top, "ret": round(mr, 2), "spy": round(sr, 2),
            "win": mr > 0, "beat": mr > sr, "focus": is_focus,
        })

    if not results:
        print("  No results")
        return

    wins  = sum(1 for r in results if r["win"])
    beats = sum(1 for r in results if r["beat"])
    avg   = sum(r["ret"] for r in results) / len(results)
    avg_s = sum(r["spy"] for r in results) / len(results)

    print(f"\n  {results[0]['month']} – {results[-1]['month']}  ({len(results)} months)")
    print(f"  Strategy: {wins/len(results)*100:.0f}% win months  "
          f"avg {avg:+.2f}%/mo  ann {(1+avg/100)**12-1:+.1%}")
    print(f"  SPY:      {sum(1 for r in results if r['spy']>0)/len(results)*100:.0f}% win months  "
          f"avg {avg_s:+.2f}%/mo  ann {(1+avg_s/100)**12-1:+.1%}")
    print(f"  Beat SPY: {beats/len(results)*100:.0f}% of months")
    print(f"  $10k → ${port:>10,.0f}  (strategy)  vs  ${spy_val:>10,.0f}  (SPY)")

    if focus_months:
        fa  = sum(m["ret"] for m in focus_months) / len(focus_months)
        fw  = sum(1 for m in focus_months if m["ret"] > 0)
        print(f"\n  {'+'.join(focus)} held together: {len(focus_months)} month(s)")
        print(f"    Win: {fw}/{len(focus_months)}  Avg: {fa:+.2f}%/mo")
        for m in focus_months:
            print(f"    {m['month']}: {m['ret']:+.1f}% vs SPY {m['spy']:+.1f}%")
    else:
        print(f"\n  {'+'.join(focus)} have never been top-2 simultaneously before.")

    from collections import Counter
    counts = Counter(e for r in results for e in r["top"])
    print(f"\n  ETF frequency in top-2 ({len(results)} months):")
    for etf, cnt in counts.most_common():
        bar  = "█" * int(cnt / len(results) * 30)
        mark = " ◄ current signal" if etf in focus else ""
        print(f"    {etf:5} {cnt:3}×  ({cnt/len(results)*100:4.0f}%)  {bar}{mark}")

    print(f"\n  Last 12 months:")
    print(f"  {'Month':8} {'ETFs':20} {'Strat':>6} {'SPY':>6} {'vs SPY':>7}")
    for r in results[-12:]:
        vs   = r["ret"] - r["spy"]
        mark = " ◄" if r["focus"] else ""
        print(f"  {r['month']:8} {'+'.join(r['top']):20} "
              f"{r['ret']:>+5.1f}% {r['spy']:>+5.1f}% {vs:>+6.1f}%{mark}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TICKERS = ["NVDA", "AAPL", "MSFT", "META", "GOOGL", "CRM"]


def main() -> None:
    print(SEP)
    print("  SIGNAL BACKTEST  —  pre-earnings D-20 + sector rotation")
    print("  Validating against earnings_dataset.json")
    print(SEP)

    # Load known dataset for validation
    ds_path = RESEARCH_DIR / "earnings_dataset.json"
    known_evs: list[dict] = []
    if ds_path.exists():
        known_evs = json.loads(ds_path.read_text())
    print(f"\nLoaded {len(known_evs)} events from earnings_dataset.json "
          f"({', '.join(sorted({e['ticker'] for e in known_evs}))})")

    print("\nDownloading prices + earnings dates...")
    for t in TICKERS + SECTOR_ETFS + ["SPY"]:
        _prices(t)
        print(f"  {t} ready")

    # ── Pre-earnings: all tickers ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PRE-EARNINGS SIGNAL  (buy D-20, sell D-1 before announcement)")
    print(SEP)

    summaries = []
    for ticker in TICKERS:
        r = run_pre_earnings(ticker, known_evs)
        summaries.append(r)
        print_pre_earnings(r)

    # ── Ranking table ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RANKING — pre-earnings signal quality")
    print(SEP)
    print(f"  {'Ticker':6} {'Trades':>7} {'Win%':>6} {'Avg%':>7} {'R/R':>5} "
          f"{'$10k→':>12} {'Validate':>10}  Grade")
    print(f"  {'─'*6} {'─'*7} {'─'*6} {'─'*7} {'─'*5} {'─'*12} {'─'*10}  {'─'*5}")

    ranked = sorted(
        [s for s in summaries if s["trades"] > 0],
        key=lambda s: (s["win_rate"] * s["avg_ret"]),
        reverse=True,
    )
    for s in ranked:
        grade = ("A" if s["win_rate"] >= 75 and s["avg_ret"] >= 4 else
                 "B" if s["win_rate"] >= 65 and s["avg_ret"] >= 2 else
                 "C" if s["win_rate"] >= 55 else "D")
        val   = f"{s['matches']}✓ {s['mismatches']}✗" if s["matches"] + s["mismatches"] else "no dataset"
        print(f"  {s['ticker']:6} {s['trades']:>7} {s['win_rate']:>5.0f}% "
              f"{s['avg_ret']:>+6.2f}% {s['rr']:>5.2f} "
              f"${s['final_10k']:>10,.0f}  {val:>10}  {grade}")

    # ── Sector rotation ───────────────────────────────────────────────────
    run_sector_backtest(["XLK", "XLE"])

    # ── Today's active signals ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TODAY'S SIGNALS  (2026-05-05)")
    print(SEP)
    for s in summaries:
        upcoming = [d for d in fetch_earnings_dates(s["ticker"]) if d > "2026-05-05"]
        if not upcoming:
            continue
        next_ann = upcoming[0]
        days_away = (date.fromisoformat(next_ann) - date.today()).days
        if days_away > 30:
            continue
        grade = ("A" if s["win_rate"] >= 75 and s["avg_ret"] >= 4 else
                 "B" if s["win_rate"] >= 65 and s["avg_ret"] >= 2 else
                 "C" if s["win_rate"] >= 55 else "D")
        df     = _prices(s["ticker"])
        cur_px = float(df["Close"].iloc[-1]) if not df.empty else 0
        print(f"\n  {s['ticker']}  —  earnings {next_ann} ({days_away}d)  Grade {grade}")
        print(f"    Price: ${cur_px:.2f}  |  Signal win rate: {s['win_rate']:.0f}%  "
              f"avg: {s['avg_ret']:+.2f}%")
        print(f"    Action: BUY now, sell before {next_ann}")
        if grade in ("C", "D"):
            print(f"    ⚠ Low conviction — consider half-size or skip")


if __name__ == "__main__":
    main()

