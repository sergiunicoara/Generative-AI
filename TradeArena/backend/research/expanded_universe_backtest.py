"""
Expanded Universe Backtest
===========================
Tests the pre-earnings D-20 strategy (S2 rules: F6 + score>=1.20)
on a broad S&P 500 large-cap subset to find which tickers have the
same edge as our current GOOGL/NVDA/AMZN core.

Questions answered:
  1. Which tickers show a consistent pre-earnings drift?
  2. What win rate / avg return do they achieve vs our core 6?
  3. Does adding the best tickers improve final equity and trade frequency?
  4. What base_quality score should each ticker get?

Rules (identical to S2+F6+L8):
  - QQQ > 150dma regime filter
  - D-20 entry / D-1 exit
  - F6: entry day open > previous close
  - Score >= 1.20 (dynamic score with per-ticker base_quality)
  - -5% stop-loss
  - 100% deployment, 1 position at a time, sequential

Usage:
    uv run python -m backend.research.expanded_universe_backtest
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
START_CASH   = 2_000.0
STOP_PCT     = -5.0
RISK_FREE    = 0.05
EXPIRY_DAYS  = 60
SCORE_THRESH = 1.20  # L8 filter

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

# Current validated core
CURRENT_UNIVERSE = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]

CURRENT_BASE_QUALITY = {
    "GOOGL": 1.40, "NVDA": 1.50, "AMZN": 1.20,
    "MSFT":  1.10, "META": 1.10, "AMD":  1.00,
}

# Candidates to evaluate — large-cap, high-volume, analyst-covered
CANDIDATES = [
    # Mega-cap tech (previously excluded — re-test with data)
    "AAPL", "TSLA", "NFLX", "CRM",
    # Semiconductors
    "AVGO", "QCOM", "INTC", "MU", "TXN", "AMAT", "LRCX", "KLAC", "MRVL", "ARM",
    # Cloud / SaaS
    "ADBE", "NOW", "SNOW", "PLTR", "DDOG", "CRWD", "PANW", "NET", "ZS",
    # Consumer tech / platforms
    "UBER", "ABNB",
    # Payments
    "V", "MA", "PYPL",
    # Broader S&P 500 large-cap
    "JPM", "BAC", "GS",
    "UNH", "LLY",
    "AMGN", "GILD",
    "XOM", "CVX",
    "HD", "WMT", "COST",
    "DIS", "NFLX",
]
# Deduplicate
CANDIDATES = list(dict.fromkeys(CANDIDATES))

ALL_TICKERS = list(dict.fromkeys(CURRENT_UNIVERSE + CANDIDATES))

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
# Helpers
# ---------------------------------------------------------------------------

def _nth(df, ref, offset):
    dt, d, c = pd.Timestamp(ref), (1 if offset >= 0 else -1), 0
    for i in range(1, 300):
        cand = dt + pd.Timedelta(days=i * d)
        if cand in df.index:
            c += 1
            if c == abs(offset):
                return cand
    return None

def _bounded(x): return max(-20.0, min(20.0, x)) / 100.0

def _score(ticker, entry_dt, qqq_df, base_q):
    df  = _px(ticker)
    col = df["Close"].loc[df.index <= entry_dt]
    if len(col) < 62: return base_q
    m20 = (col.iloc[-1] / col.iloc[-21] - 1) * 100
    m60 = (col.iloc[-1] / col.iloc[-62] - 1) * 100
    qc  = qqq_df["Close"].loc[qqq_df.index <= entry_dt]
    q20 = (qc.iloc[-1] / qc.iloc[-21] - 1) * 100 if len(qc) >= 21 else 0
    rs  = m20 - float(q20)
    return base_q + 1.20 * _bounded(m20) + 0.80 * _bounded(m60) + 1.50 * _bounded(rs)

def _regime_ok(entry_dt, qqq_df):
    qc = qqq_df["Close"].loc[qqq_df.index <= entry_dt]
    return len(qc) >= 150 and float(qc.iloc[-1]) > float(qc.rolling(150).mean().iloc[-1])

def _gap_up(ticker, entry_dt):
    df   = _px(ticker)
    if "Open" not in df.columns: return True
    prev = [d for d in df.index if d < entry_dt]
    if not prev: return True
    return float(df["Open"].loc[entry_dt]) > float(df["Close"].loc[prev[-1]])

# ---------------------------------------------------------------------------
# Per-ticker raw stats (no score filter — just regime + D-20 window + stop)
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import fetch_earnings_dates

def ticker_raw_stats(ticker: str, qqq_df: pd.DataFrame) -> dict:
    """All D-20 trades for this ticker with regime filter but NO score/F6 filter.
    Used to compute historical win rate and decide base_quality."""
    df    = _px(ticker)
    dates = fetch_earnings_dates(ticker)
    trades = []
    for ann in dates:
        e_dt = _nth(df, ann, -20)
        x_dt = _nth(df, ann, -1)
        if e_dt is None or x_dt is None: continue
        if e_dt not in df.index or x_dt not in df.index: continue
        if not _regime_ok(e_dt, qqq_df): continue

        ep = float(df["Close"].loc[e_dt])
        xp = float(df["Close"].loc[x_dt])
        window  = [d for d in df.index if e_dt <= d <= x_dt]
        min_px  = min(float(df["Close"].loc[d]) for d in window)
        stopped = min_px <= ep * 0.95
        ret     = STOP_PCT if stopped else (xp - ep) / ep * 100

        trades.append({"ann": ann, "entry_dt": e_dt, "ret": ret, "stopped": stopped})

    if not trades:
        return {"ticker": ticker, "n": 0, "wr": 0, "avg": 0, "trades": []}

    wins = sum(1 for t in trades if t["ret"] > 0)
    avg  = sum(t["ret"] for t in trades) / len(trades)
    return {
        "ticker": ticker,
        "n":      len(trades),
        "wr":     round(wins / len(trades) * 100, 1),
        "avg":    round(avg, 2),
        "trades": trades,
    }


def suggest_base_quality(wr: float, avg: float) -> float:
    """Assign base_quality from historical win rate + avg return.
    Mirrors the logic used for our current 6 tickers."""
    if wr >= 75 and avg >= 4.0:  return 1.50   # NVDA tier
    if wr >= 70 and avg >= 3.0:  return 1.40   # GOOGL tier
    if wr >= 65 and avg >= 2.5:  return 1.30   # strong candidate
    if wr >= 60 and avg >= 2.0:  return 1.20   # solid candidate
    if wr >= 55 and avg >= 1.5:  return 1.10   # marginal
    if wr >= 50 and avg >= 1.0:  return 1.00   # borderline
    return 0.90                                  # below threshold — exclude

# ---------------------------------------------------------------------------
# Build filtered trade list for a given universe + base_quality map
# ---------------------------------------------------------------------------

def build_trades(universe: list[str], base_quality: dict,
                 qqq_df: pd.DataFrame) -> list[dict]:
    raw = []
    for ticker in universe:
        df    = _px(ticker)
        dates = fetch_earnings_dates(ticker)
        bq    = base_quality.get(ticker, 1.00)
        for ann in dates:
            e_dt = _nth(df, ann, -20)
            x_dt = _nth(df, ann, -1)
            if e_dt is None or x_dt is None: continue
            if e_dt not in df.index or x_dt not in df.index: continue
            if not _regime_ok(e_dt, qqq_df): continue

            sc = _score(ticker, e_dt, qqq_df, bq)
            if sc < SCORE_THRESH: continue           # L8
            if not _gap_up(ticker, e_dt): continue   # F6

            ep = float(df["Close"].loc[e_dt])
            xp = float(df["Close"].loc[x_dt])
            window  = [d for d in df.index if e_dt <= d <= x_dt]
            min_px  = min(float(df["Close"].loc[d]) for d in window)
            stopped = min_px <= ep * 0.95
            ret     = STOP_PCT if stopped else (xp - ep) / ep * 100

            raw.append({
                "ticker": ticker, "ann": ann,
                "entry_dt": e_dt, "exit_dt": x_dt,
                "ret": round(ret, 3), "score": round(sc, 3),
            })

    return sorted(raw, key=lambda t: t["entry_dt"])


def simulate(trades: list[dict]) -> dict:
    """Sequential: 1 position, 100% deploy, highest-score wins conflicts."""
    # Group by entry date, pick best score if overlap
    by_date: dict = {}
    for t in trades:
        by_date.setdefault(t["entry_dt"], []).append(t)

    taken, busy = [], None
    for dt in sorted(by_date):
        if busy and dt <= busy: continue
        best = max(by_date[dt], key=lambda t: t["score"])
        taken.append(best)
        busy = best["exit_dt"]

    equity, peak, max_dd = START_CASH, START_CASH, 0.0
    milestones: dict[int, object] = {}
    wins = losses = 0

    for t in taken:
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = t["entry_dt"].date()
        equity *= (1 + t["ret"] / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if t["ret"] > 0: wins += 1
        else: losses += 1

    n    = wins + losses
    rets = [t["ret"] for t in taken]
    years = 11  # 2015–2026
    ann  = (equity / START_CASH) ** (1 / years) - 1 if n else 0

    return {
        "n": n, "wins": wins,
        "wr":     round(wins / n * 100, 1) if n else 0,
        "avg":    round(sum(rets) / n, 2) if n else 0,
        "final":  round(equity, 2),
        "x":      round(equity / START_CASH, 1),
        "ann":    round(ann * 100, 1),
        "max_dd": round(max_dd, 1),
        "milestones": milestones,
        "log":    taken,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  EXPANDED UNIVERSE BACKTEST — S2 rules (F6 + score>=1.20)")
    print(SEP)

    print(f"\nLoading prices for {len(ALL_TICKERS)} tickers + QQQ...")
    failed = []
    for t in ALL_TICKERS + ["QQQ"]:
        df = _px(t)
        if df.empty:
            print(f"  {t:<8} NO DATA")
            failed.append(t)
        else:
            print(f"  {t:<8} {len(df)} days")
    qqq_df = _px("QQQ")

    valid_tickers = [t for t in ALL_TICKERS if t not in failed and not _px(t).empty]

    # ── Per-ticker raw analysis (regime only, no score/F6) ─────────────────
    print(f"\n{SEP}")
    print(f"  PER-TICKER RAW STATS (regime filter only, all D-20 trades 2015–2026)")
    print(SEP)
    print(f"  {'Ticker':<8} {'N':>4}  {'Win%':>6}  {'Avg%':>7}  {'Suggested BQ':>13}  Note")
    print(f"  {'-'*8} {'-'*4}  {'-'*6}  {'-'*7}  {'-'*13}  {'-'*20}")

    raw_stats: dict[str, dict] = {}
    strong_candidates: list[str] = []

    for ticker in valid_tickers:
        st = ticker_raw_stats(ticker, qqq_df)
        raw_stats[ticker] = st
        if st["n"] < 5:
            print(f"  {ticker:<8} {st['n']:>4}  {'—':>6}  {'—':>7}  {'—':>13}  too few trades")
            continue
        bq   = suggest_base_quality(st["wr"], st["avg"])
        flag = ""
        in_current = ticker in CURRENT_UNIVERSE
        if bq >= 1.20 and not in_current:
            strong_candidates.append(ticker)
            flag = "  ← NEW CANDIDATE"
        elif in_current:
            flag = "  (current universe)"
        print(f"  {ticker:<8} {st['n']:>4}  {st['wr']:>5.1f}%  {st['avg']:>+6.2f}%  {bq:>13.2f}{flag}")

    # ── Build expanded universe ────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  STRONG NEW CANDIDATES (win%>=55, avg>=1.5%): {strong_candidates}")
    print(SEP)

    # Assign base quality to all tickers
    expanded_bq: dict[str, float] = dict(CURRENT_BASE_QUALITY)
    for ticker in strong_candidates:
        st  = raw_stats[ticker]
        bq  = suggest_base_quality(st["wr"], st["avg"])
        expanded_bq[ticker] = bq

    expanded_universe = CURRENT_UNIVERSE + [t for t in strong_candidates if t not in CURRENT_UNIVERSE]

    # ── Simulations ────────────────────────────────────────────────────────
    print(f"\n  Building trade lists...")
    trades_current  = build_trades(CURRENT_UNIVERSE, CURRENT_BASE_QUALITY, qqq_df)
    trades_expanded = build_trades(expanded_universe, expanded_bq, qqq_df)
    print(f"  Current universe  ({len(CURRENT_UNIVERSE)} tickers): {len(trades_current)} filtered trades")
    print(f"  Expanded universe ({len(expanded_universe)} tickers): {len(trades_expanded)} filtered trades")

    r_current  = simulate(trades_current)
    r_expanded = simulate(trades_expanded)

    # ── Comparison table ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SIMULATION RESULTS (2015–2026, $2,000 start, S2 F6+L8 rules)")
    print(SEP)
    print(f"  {'Version':<28} {'N':>4}  {'Win%':>6}  {'Avg%':>7}  {'Ann%':>6}  {'MaxDD':>6}  {'Final':>10}  $10k")
    print(f"  {'-'*28} {'-'*4}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}")

    for label, r in [("S2 current (6 tickers)", r_current),
                     (f"S2 expanded ({len(expanded_universe)} tickers)", r_expanded)]:
        m10 = str(r["milestones"].get(10_000, "—"))[:10]
        print(f"  {label:<28} {r['n']:>4}  {r['wr']:>5.1f}%  {r['avg']:>+6.2f}%  "
              f"{r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}  {m10}")

    d_final = r_expanded["final"] - r_current["final"]
    d_n     = r_expanded["n"]     - r_current["n"]
    d_dd    = r_expanded["max_dd"]- r_current["max_dd"]
    d_ann   = r_expanded["ann"]   - r_current["ann"]
    print(f"\n  Delta: Δtrades {d_n:+d}  Δann {d_ann:+.1f}%  Δdd {d_dd:+.1f}%  Δfinal ${d_final:+,.0f}")

    # ── Year-by-year expanded ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  YEAR-BY-YEAR — expanded universe")
    print(SEP)
    by_yr: dict[int, list] = {}
    for t in r_expanded["log"]:
        by_yr.setdefault(int(t["ann"][:4]), []).append(t["ret"])
    cum = START_CASH
    for yr in sorted(by_yr):
        rets = by_yr[yr]
        avg  = sum(rets) / len(rets)
        w    = sum(1 for r_ in rets if r_ > 0)
        for r_ in rets: cum *= (1 + r_ / 100)
        bar  = ("+" if avg >= 0 else "-") + "█" * min(int(abs(avg) / 2.5), 22)
        print(f"  {yr}  {w}/{len(rets):>2} wins  avg {avg:>+5.1f}%  ${cum:>9,.0f}  {bar}")

    # ── Ticker contribution in expanded ───────────────────────────────────
    print(f"\n{SEP}")
    print(f"  TICKER CONTRIBUTION in expanded universe")
    print(SEP)
    ticker_trades: dict[str, list] = {}
    for t in r_expanded["log"]:
        ticker_trades.setdefault(t["ticker"], []).append(t["ret"])
    print(f"  {'Ticker':<8} {'Trades':>7}  {'Win%':>6}  {'Avg%':>7}  {'Total contrib%':>15}")
    print(f"  {'-'*8} {'-'*7}  {'-'*6}  {'-'*7}  {'-'*15}")
    for ticker in sorted(ticker_trades, key=lambda t: -len(ticker_trades[t])):
        rets = ticker_trades[ticker]
        w    = sum(1 for r_ in rets if r_ > 0)
        avg  = sum(rets) / len(rets)
        total= sum(rets)
        print(f"  {ticker:<8} {len(rets):>7}  {w/len(rets)*100:>5.1f}%  {avg:>+6.2f}%  {total:>+14.1f}%")

    # ── Verdict ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  VERDICT")
    print(SEP)
    if r_expanded["final"] > r_current["final"] and r_expanded["max_dd"] <= r_current["max_dd"] + 2:
        print(f"  ✓ EXPAND — expanded universe improves final equity by ${d_final:+,.0f}")
        print(f"    +{d_n} more trades/yr, drawdown stays controlled.")
        print(f"\n  Recommended additions:")
        for t in strong_candidates:
            st = raw_stats[t]
            bq = expanded_bq.get(t, 1.00)
            print(f"    {t:<8} BQ={bq:.2f}  ({st['wr']:.0f}% win  avg {st['avg']:+.2f}%)")
    elif r_expanded["final"] > r_current["final"]:
        print(f"  ~ MIXED — more return (${d_final:+,.0f}) but drawdown increases {d_dd:+.1f}%")
        print(f"    Review individual candidates carefully before adding.")
    else:
        print(f"  ✗ NO IMPROVEMENT — expanding universe does not help.")
        print(f"    Current 6 tickers are already well-selected.")
        print(f"    Pre-earnings drift edge may be concentrated in mega-cap tech only.")


if __name__ == "__main__":
    main()

