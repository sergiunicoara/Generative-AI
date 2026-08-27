"""
S3 Stock/Option Split Comparison
==================================
Tests the S2 pre-earnings strategy with different stock/call allocations.

Splits tested:
  100 /  0   S2 pure stock (reference)
   90 / 10   S3 current (combined_strategy.md)
   80 / 20
   75 / 25
   50 / 50
   25 / 75
    0 /100   Pure calls (theoretical)

Options model: 10% ITM / 60 DTE calls, 25% real-world haircut (Black-Scholes).
Same F6 + score>=1.20 + regime filter as S2/S3.

Usage:
    uv run python -m backend.research.s3_splits_backtest
"""

from __future__ import annotations

import math
import sys

import pandas as pd
import yfinance as yf
from scipy.stats import norm

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

RISK_FREE   = 0.05
EXPIRY_DAYS = 60
STOP_PCT    = -5.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_cache: dict[str, pd.DataFrame] = {}

def _px(ticker: str) -> pd.DataFrame:
    if ticker not in _cache:
        df = yf.download(ticker, start="2012-01-01", end="2026-05-17",
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

def _score(ticker, dt, qqq_df):
    df  = _px(ticker)
    col = df["Close"].loc[df.index <= dt]
    if len(col) < 62: return BASE_QUALITY.get(ticker, 1.0)
    m20 = (col.iloc[-1] / col.iloc[-21] - 1) * 100
    m60 = (col.iloc[-1] / col.iloc[-62] - 1) * 100
    qc  = qqq_df["Close"].loc[qqq_df.index <= dt]
    q20 = (qc.iloc[-1] / qc.iloc[-21] - 1) * 100 if len(qc) >= 21 else 0
    rs  = m20 - float(q20)
    return (BASE_QUALITY.get(ticker, 1.0)
            + 1.20 * _bounded(m20)
            + 0.80 * _bounded(m60)
            + 1.50 * _bounded(rs))

def _regime_ok(dt, qqq_df):
    qc = qqq_df["Close"].loc[qqq_df.index <= dt]
    return len(qc) >= 150 and float(qc.iloc[-1]) > float(qc.rolling(150).mean().iloc[-1])

def _gap_up(ticker, dt):
    df = _px(ticker)
    if "Open" not in df.columns: return True
    prev = [d for d in df.index if d < dt]
    if not prev: return True
    return float(df["Open"].loc[dt]) > float(df["Close"].loc[prev[-1]])

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
    c1   = bs_call(S_exit,  K, max((EXPIRY_DAYS - hold*1.4)/365, 0), RISK_FREE, iv) * 0.75
    return (c1 - c0) / c0 * 100 if c0 > 0.01 else 0.0

# ---------------------------------------------------------------------------
# Build enriched trade list
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import fetch_earnings_dates

def build_trades(qqq_df: pd.DataFrame) -> list[dict]:
    raw = []
    for ticker in UNIVERSE:
        df    = _px(ticker)
        dates = fetch_earnings_dates(ticker)
        for ann in dates:
            e_dt = _nth(df, ann, -20)
            x_dt = _nth(df, ann, -1)
            if e_dt is None or x_dt is None: continue
            if e_dt not in df.index or x_dt not in df.index: continue
            if not _regime_ok(e_dt, qqq_df): continue
            sc = _score(ticker, e_dt, qqq_df)
            if sc < 1.20: continue
            if not _gap_up(ticker, e_dt): continue

            ep = float(df["Close"].loc[e_dt])
            xp = float(df["Close"].loc[x_dt])
            window  = [d for d in df.index if e_dt <= d <= x_dt]
            min_px  = min(float(df["Close"].loc[d]) for d in window)
            stopped = min_px <= ep * 0.95
            ret_s   = STOP_PCT if stopped else (xp - ep) / ep * 100
            ret_o   = _opt_roi(ticker, e_dt, x_dt, ep, xp)

            raw.append({
                "ticker":   ticker, "ann": ann,
                "entry_dt": e_dt,   "exit_dt": x_dt,
                "ret_stock":round(ret_s, 3),
                "ret_opt":  round(ret_o, 3),
                "score":    round(sc, 3),
                "stopped":  stopped,
            })

    raw.sort(key=lambda t: t["entry_dt"])
    by_date: dict = {}
    for t in raw:
        d = t["entry_dt"]
        if d not in by_date or t["score"] > by_date[d]["score"]:
            by_date[d] = t

    taken, busy = [], None
    for t in sorted(by_date.values(), key=lambda x: x["entry_dt"]):
        if busy and t["entry_dt"] <= busy: continue
        taken.append(t)
        busy = t["exit_dt"]
    return taken

# ---------------------------------------------------------------------------
# Simulate at a given split
# ---------------------------------------------------------------------------

def simulate(trades: list[dict], stock_pct: float, opt_pct: float,
             label: str) -> dict:
    equity = START_CASH
    peak   = START_CASH
    max_dd = 0.0
    wins   = losses = 0
    milestones: dict = {}
    log    = []

    for t in trades:
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = t["entry_dt"].date()

        ret = stock_pct * t["ret_stock"] + opt_pct * t["ret_opt"]

        equity *= (1 + ret / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if ret > 0: wins += 1
        else: losses += 1
        log.append({**t, "ret_blend": round(ret, 3), "equity": round(equity, 2)})

    n    = wins + losses
    rets = [t["ret_blend"] for t in log]
    ann  = (equity / START_CASH) ** (1 / 11) - 1

    return {
        "label":     label,
        "stock_pct": stock_pct,
        "opt_pct":   opt_pct,
        "n":         n,
        "wins":      wins,
        "wr":        round(wins / n * 100, 1) if n else 0,
        "avg":       round(sum(rets) / n, 2) if n else 0,
        "final":     round(equity, 2),
        "x":         round(equity / START_CASH, 1),
        "ann":       round(ann * 100, 1),
        "max_dd":    round(max_dd, 1),
        "milestones":milestones,
        "log":       log,
    }

# ---------------------------------------------------------------------------
# Year-by-year
# ---------------------------------------------------------------------------

def year_by_year(r: dict) -> None:
    by_yr: dict[int, list] = {}
    for t in r["log"]:
        by_yr.setdefault(t["entry_dt"].year, []).append(t["ret_blend"])
    cum = START_CASH
    for yr in sorted(by_yr):
        rets = by_yr[yr]
        avg  = sum(rets) / len(rets)
        w    = sum(1 for r_ in rets if r_ > 0)
        for r_ in rets: cum *= (1 + r_ / 100)
        bar  = ("+" if avg >= 0 else "-") + "█" * min(int(abs(avg) / 2.0), 25)
        print(f"    {yr}  {w}/{len(rets)} wins  avg {avg:>+6.2f}%  ${cum:>9,.0f}  {bar}")

# ---------------------------------------------------------------------------
# Per-trade detail for a given split
# ---------------------------------------------------------------------------

def trade_detail(r: dict) -> None:
    print(f"  {'Date':<12} {'Ticker':<6} {'Score':>6}  "
          f"{'Stock%':>7}  {'Opt%':>7}  {'Blend%':>7}  {'Equity':>10}  Stop?")
    print(f"  {'-'*12} {'-'*6} {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}")
    for t in r["log"]:
        flag = " STOP" if t["stopped"] else ""
        print(f"  {str(t['entry_dt'].date()):<12} {t['ticker']:<6} {t['score']:>6.3f}  "
              f"{t['ret_stock']:>+6.2f}%  {t['ret_opt']:>+6.2f}%  "
              f"{t['ret_blend']:>+6.2f}%  ${t['equity']:>9,.0f}{flag}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  S3 STOCK/OPTION SPLIT COMPARISON (2015-2026, $2,000 start)")
    print("  Options: 10% ITM / 60 DTE calls, 25% real-world haircut")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ"]:
        _px(t); print(f"  {t} ready")
    qqq_df = _px("QQQ")

    print("\nBuilding trade list (F6 + score>=1.20)...")
    trades = build_trades(qqq_df)
    print(f"  {len(trades)} trades")

    # Show raw stock vs option return per trade
    print(f"\n{SEP}")
    print("  RAW TRADE RETURNS — stock vs option (10% ITM / 60 DTE / 25% haircut)")
    print(SEP)
    print(f"  {'Date':<12} {'Ticker':<6} {'Score':>6}  {'Stock%':>8}  "
          f"{'Opt ROI%':>9}  {'Opt 2x?':>7}  {'Opt lose%':>9}")
    print(f"  {'-'*12} {'-'*6} {'-'*6}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*9}")
    opt_beats = 0
    opt_2x    = 0
    opt_lose  = 0
    for t in trades:
        beats = t["ret_opt"] > t["ret_stock"]
        two_x = t["ret_opt"] > t["ret_stock"] * 2
        lose  = t["ret_opt"] < 0
        if beats: opt_beats += 1
        if two_x:  opt_2x   += 1
        if lose:   opt_lose  += 1
        flag = "  ◄2x" if two_x else ("  ◄beats" if beats else "")
        print(f"  {str(t['entry_dt'].date()):<12} {t['ticker']:<6} {t['score']:>6.3f}  "
              f"{t['ret_stock']:>+7.2f}%  {t['ret_opt']:>+8.2f}%  "
              f"{'yes' if two_x else '—':>7}  "
              f"{'LOSE' if lose else '—':>9}{flag}")

    n = len(trades)
    print(f"\n  Option beats stock:  {opt_beats}/{n} ({opt_beats/n*100:.0f}%)")
    print(f"  Option returns 2x+:  {opt_2x}/{n}  ({opt_2x/n*100:.0f}%)")
    print(f"  Option loses money:  {opt_lose}/{n} ({opt_lose/n*100:.0f}%)")

    # ── All splits ────────────────────────────────────────────────────────
    splits = [
        (1.00, 0.00, "100% stock / 0% calls  (S2 pure)"),
        (0.90, 0.10, " 90% stock / 10% calls (S3 current)"),
        (0.80, 0.20, " 80% stock / 20% calls"),
        (0.75, 0.25, " 75% stock / 25% calls"),
        (0.50, 0.50, " 50% stock / 50% calls"),
        (0.25, 0.75, " 25% stock / 75% calls"),
        (0.00, 1.00, "  0% stock /100% calls (pure options)"),
    ]

    results = [simulate(trades, sp, op, lbl) for sp, op, lbl in splits]

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SPLIT COMPARISON — all metrics")
    print(SEP)
    print(f"  {'Split':<36} {'N':>3}  {'Win%':>5}  {'Avg%':>6}  "
          f"{'Ann%':>6}  {'MaxDD':>6}  {'Final':>10}  {'$10k':>10}  {'$20k':>10}")
    print(f"  {'-'*36} {'-'*3}  {'-'*5}  {'-'*6}  "
          f"{'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

    base = results[0]
    for r in results:
        m10 = str(r["milestones"].get(10_000, "—"))[:10]
        m20 = str(r["milestones"].get(20_000, "—"))[:10]
        flag = "  ◄" if r["ann"] > base["ann"] and r["max_dd"] <= base["max_dd"] * 2.5 else ""
        print(f"  {r['label']:<36} {r['n']:>3}  {r['wr']:>4.1f}%  {r['avg']:>+5.2f}%  "
              f"{r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}  "
              f"{m10:>10}  {m20:>10}{flag}")

    # ── Risk/return chart ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RISK / RETURN CURVE")
    print(SEP)
    print(f"  {'Split':<36}  Ann%    DD%   Ratio (ann/dd)")
    print(f"  {'-'*36}  {'-'*6}  {'-'*6}  {'-'*14}")
    for r in results:
        ratio = r["ann"] / r["max_dd"] if r["max_dd"] > 0 else 0
        bar   = "█" * min(int(ratio * 3), 30)
        print(f"  {r['label']:<36}  {r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  {ratio:>6.2f}  {bar}")

    # ── Year-by-year for key splits ────────────────────────────────────────
    for r in results:
        if r["opt_pct"] not in [0.0, 0.10, 0.20, 0.50]: continue
        print(f"\n{SEP}")
        print(f"  YEAR-BY-YEAR — {r['label'].strip()}")
        print(SEP)
        year_by_year(r)

    # ── Worst drawdown analysis per split ──────────────────────────────────
    print(f"\n{SEP}")
    print("  WORST 5 TRADES — by split")
    print(SEP)
    for r in results:
        worst = sorted(r["log"], key=lambda t: t["ret_blend"])[:5]
        worst_str = "  ".join(
            f"{t['ticker']}({t['ret_blend']:>+.1f}%)" for t in worst
        )
        print(f"  {r['label'][:34]:<34}  {worst_str}")

    # ── Marginal benefit of adding more options ────────────────────────────
    print(f"\n{SEP}")
    print("  MARGINAL BENEFIT OF EACH 10% SHIFT FROM STOCK → OPTIONS")
    print(SEP)
    print(f"  {'Shift':<30}  {'Δ Ann%':>7}  {'Δ MaxDD%':>9}  {'Δ Final':>10}  Worth it?")
    print(f"  {'-'*30}  {'-'*7}  {'-'*9}  {'-'*10}")
    pairs = list(zip(results, results[1:]))
    for r0, r1 in pairs:
        d_ann = r1["ann"]    - r0["ann"]
        d_dd  = r1["max_dd"] - r0["max_dd"]
        d_fin = r1["final"]  - r0["final"]
        worth = "YES" if d_ann > 0 and (d_dd / r0["max_dd"] < 0.5 or r0["max_dd"] < 10) else "NO"
        print(f"  {r0['label'][:14].strip():<6} → {r1['label'][:14].strip():<20}  "
              f"{d_ann:>+6.1f}%  {d_dd:>+8.1f}%  ${d_fin:>+9,.0f}  {worth}")

    # ── Sweet spot summary ─────────────────────────────────────────────────
    best_ratio = max(results, key=lambda r: r["ann"] / max(r["max_dd"], 1))
    best_return = max(results, key=lambda r: r["ann"])

    print(f"\n{SEP}")
    print("  CONCLUSIONS")
    print(SEP)
    print(f"  Best risk-adjusted (ann/dd ratio): {best_ratio['label'].strip()}")
    print(f"    Ann {best_ratio['ann']:+.1f}%  DD {best_ratio['max_dd']:.1f}%  "
          f"Final ${best_ratio['final']:,.0f}")
    print(f"\n  Best absolute return: {best_return['label'].strip()}")
    print(f"    Ann {best_return['ann']:+.1f}%  DD {best_return['max_dd']:.1f}%  "
          f"Final ${best_return['final']:,.0f}")
    print(f"\n  Note: Options returns are MODELLED (Black-Scholes + 25% haircut).")
    print(f"  Real execution needs: live IV, bid/ask spreads, open interest check.")
    print(f"  Add real option data before deploying any split above 10% options.")


if __name__ == "__main__":
    main()

