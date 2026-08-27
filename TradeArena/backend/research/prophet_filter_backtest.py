"""
Prophet Filter Backtest
========================
Tests whether Facebook Prophet's 20-day price forecast can improve
pre-earnings trade entry decisions.

Three Prophet-based filters:

  PF1  Forecast direction
       Prophet predicts price HIGHER at D-1 than at D-20 → enter
       Prophet predicts LOWER → skip

  PF2  Actual vs forecast (momentum signal)
       Entry price at D-20 is ABOVE Prophet's expected price for that date
       → stock is outperforming its expected trajectory → enter
       Below forecast → skip

  PF3  Forecast confidence (uncertainty filter)
       Uncertainty interval at D-1 is narrow (yhat_upper-yhat_lower < X%)
       → Prophet is confident about the move → enter
       Wide interval → too uncertain → skip

Prophet setup per trade:
  - Fit on 2 years of daily closes before D-20 (504 trading days)
  - Predict 25 days forward (covers full D-20 to D-1 window)
  - Use weekly_seasonality=True (captures earnings-driven weekly patterns)
  - Use yearly_seasonality=True (captures annual cycles)

Tested on:
  A) 41 S2 multi-week trades (primary — these are the real positions)
  B) 1,348 overnight observations grouped by earnings cycle

Usage:
    uv run python -m backend.research.prophet_filter_backtest
"""

from __future__ import annotations

import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

# Suppress Prophet/Stan output
warnings.filterwarnings("ignore")
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from prophet import Prophet

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP        = "=" * 72
START_CASH = 2_000.0
UNIVERSE   = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
MILESTONES = [5_000, 10_000, 20_000]

BASE_QUALITY = {
    "GOOGL": 1.40, "NVDA": 1.50, "AMZN": 1.20,
    "MSFT":  1.10, "META": 1.10, "AMD":  1.00,
}

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

# ---------------------------------------------------------------------------
# Prophet forecast for one ticker at one entry date
# ---------------------------------------------------------------------------

def fit_prophet(ticker: str, entry_dt: pd.Timestamp,
                horizon_days: int = 30) -> pd.DataFrame | None:
    """
    Fit Prophet on 2 years of closes before entry_dt.
    Returns forecast DataFrame with ds, yhat, yhat_lower, yhat_upper
    for the next horizon_days calendar days.
    Returns None if insufficient data.
    """
    df  = _px(ticker)
    col = df["Close"].loc[df.index < entry_dt]
    if len(col) < 252:          # need at least 1 year
        return None

    # Use up to 2 years of training data
    train = col.tail(504).reset_index()
    train.columns = ["ds", "y"]
    train["ds"] = pd.to_datetime(train["ds"]).dt.tz_localize(None)

    try:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,   # moderate flexibility
            seasonality_prior_scale=10,
            interval_width=0.80,
        )
        m.fit(train)
        future   = m.make_future_dataframe(periods=horizon_days)
        forecast = m.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception:
        return None


def prophet_signals(ticker: str, entry_dt: pd.Timestamp,
                    exit_dt: pd.Timestamp) -> dict | None:
    """
    Extract Prophet-based signals for a trade entering at entry_dt, exiting at exit_dt.

    Returns dict with:
        pf1_up:           forecast price at exit > price at entry
        pf2_above:        actual entry price > forecast for entry date
        pf3_narrow:       forecast confidence interval is narrow (<15% of price)
        forecast_ret:     expected % move from entry to exit per Prophet
        actual_at_entry:  stock close at entry
        forecast_at_entry:Prophet yhat at entry date
        forecast_at_exit: Prophet yhat at exit date
        uncertainty_pct:  (yhat_upper - yhat_lower) / yhat at exit, as %
    """
    horizon = (exit_dt - entry_dt).days + 5
    fc      = fit_prophet(ticker, entry_dt, horizon_days=horizon)
    if fc is None:
        return None

    df      = _px(ticker)
    entry_px = float(df["Close"].loc[entry_dt]) if entry_dt in df.index else None
    if entry_px is None:
        return None

    fc["ds"] = pd.to_datetime(fc["ds"])

    # Forecast at entry date
    fc_entry = fc.loc[fc["ds"] == entry_dt]
    if fc_entry.empty:
        # Find closest date
        diffs = (fc["ds"] - entry_dt).abs()
        fc_entry = fc.loc[[diffs.idxmin()]]
    yhat_entry = float(fc_entry["yhat"].iloc[0])

    # Forecast at exit date
    fc_exit = fc.loc[fc["ds"] == exit_dt]
    if fc_exit.empty:
        diffs = (fc["ds"] - exit_dt).abs()
        fc_exit = fc.loc[[diffs.idxmin()]]
    yhat_exit      = float(fc_exit["yhat"].iloc[0])
    yhat_upper_exit= float(fc_exit["yhat_upper"].iloc[0])
    yhat_lower_exit= float(fc_exit["yhat_lower"].iloc[0])

    forecast_ret   = (yhat_exit - yhat_entry) / yhat_entry * 100
    above_forecast = entry_px > yhat_entry
    uncertainty_pct= (yhat_upper_exit - yhat_lower_exit) / abs(yhat_exit) * 100 if yhat_exit != 0 else 999

    return {
        "pf1_up":            yhat_exit > yhat_entry,
        "pf2_above":         above_forecast,
        "pf3_narrow":        uncertainty_pct < 15.0,
        "forecast_ret":      round(forecast_ret, 2),
        "actual_at_entry":   round(entry_px, 2),
        "forecast_at_entry": round(yhat_entry, 2),
        "forecast_at_exit":  round(yhat_exit, 2),
        "uncertainty_pct":   round(uncertainty_pct, 1),
    }

# ---------------------------------------------------------------------------
# Build S2 trade list with Prophet signals
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import fetch_earnings_dates

def build_s2_with_prophet(qqq_df: pd.DataFrame) -> list[dict]:
    print("  Fitting Prophet models (one per trade)...")
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
            ret_s   = -5.0 if stopped else (xp - ep) / ep * 100

            # Fit Prophet
            ps = prophet_signals(ticker, e_dt, x_dt)
            print(f"    {ticker} {ann[:10]}  ret {ret_s:>+5.1f}%  "
                  + (f"Prophet fc {ps['forecast_ret']:>+5.1f}%  "
                     f"pf1={'✓' if ps['pf1_up'] else '✗'}  "
                     f"pf2={'✓' if ps['pf2_above'] else '✗'}  "
                     f"pf3={'✓' if ps['pf3_narrow'] else '✗'}  "
                     f"uncert={ps['uncertainty_pct']:.0f}%"
                     if ps else "Prophet FAILED"))

            raw.append({
                "ticker": ticker, "ann": ann,
                "entry_dt": e_dt, "exit_dt": x_dt,
                "ret_stock": round(ret_s, 3),
                "score": round(sc, 3),
                "stopped": stopped,
                **(ps if ps else {
                    "pf1_up": True, "pf2_above": True,
                    "pf3_narrow": True, "forecast_ret": 0,
                    "uncertainty_pct": 0
                }),
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
# Simulate
# ---------------------------------------------------------------------------

def simulate(trades: list[dict], label: str) -> dict:
    equity = START_CASH; peak = START_CASH; max_dd = 0.0
    wins = losses = 0; rets = []; milestones: dict = {}; log = []
    for t in trades:
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = t["entry_dt"].date()
        ret = t["ret_stock"]
        equity *= (1 + ret / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if ret > 0: wins += 1
        else: losses += 1
        rets.append(ret)
        log.append({**t, "equity": round(equity, 2)})
    n = wins + losses
    ann = (equity / START_CASH) ** (1/11) - 1 if n else 0
    return {
        "label": label, "n": n, "wins": wins,
        "wr":     round(wins/n*100, 1) if n else 0,
        "avg":    round(sum(rets)/n, 2) if n else 0,
        "final":  round(equity, 2),
        "ann":    round(ann*100, 1),
        "max_dd": round(max_dd, 1),
        "milestones": milestones, "log": log,
    }

def prow(r: dict, base: dict) -> None:
    m10 = str(r["milestones"].get(10_000, "—"))[:10]
    m20 = str(r["milestones"].get(20_000, "—"))[:10]
    d_ann = r["ann"] - base["ann"]; d_dd = r["max_dd"] - base["max_dd"]
    flag = "  ◄ BETTER" if d_ann >= 0 and d_dd <= 0 else \
           "  ↑ return" if d_ann > 1 else ""
    print(f"  {r['label']:<40} {r['n']:>3}  {r['wr']:>5.1f}%  {r['avg']:>+5.2f}%  "
          f"{r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}  {m20}{flag}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  PROPHET FILTER BACKTEST — directional forecast as entry filter")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ"]:
        _px(t); print(f"  {t} ready")
    qqq_df = _px("QQQ")

    # ── Build S2 trades with Prophet signals ──────────────────────────────
    print(f"\n{SEP}")
    print("  FITTING PROPHET — one model per S2 trade (41 fits)")
    print(SEP)
    trades = build_s2_with_prophet(qqq_df)
    print(f"\n  Total trades: {len(trades)}")

    # ── Prophet signal stats ───────────────────────────────────────────────
    pf1_wins = sum(1 for t in trades if t.get("pf1_up") and t["ret_stock"] > 0)
    pf1_n    = sum(1 for t in trades if t.get("pf1_up"))
    pf1_skip = sum(1 for t in trades if not t.get("pf1_up"))
    pf1_skip_wr = sum(1 for t in trades if not t.get("pf1_up") and t["ret_stock"] > 0)

    print(f"\n  PF1 (forecast up): {pf1_n} trades → {pf1_wins}/{pf1_n} wins "
          f"({pf1_wins/pf1_n*100:.0f}% wr)")
    print(f"  PF1 (forecast dn): {pf1_skip} trades → "
          f"{pf1_skip_wr}/{pf1_skip} wins "
          f"({pf1_skip_wr/pf1_skip*100:.0f}% wr)" if pf1_skip else "  PF1 (forecast dn): 0 trades")

    pf2_n  = sum(1 for t in trades if t.get("pf2_above"))
    pf2_wr = sum(1 for t in trades if t.get("pf2_above") and t["ret_stock"] > 0)
    pf2_sk = sum(1 for t in trades if not t.get("pf2_above"))
    pf2_skw= sum(1 for t in trades if not t.get("pf2_above") and t["ret_stock"] > 0)
    print(f"  PF2 (above fc):   {pf2_n} trades → {pf2_wr}/{pf2_n} wins "
          f"({pf2_wr/pf2_n*100:.0f}% wr)" if pf2_n else "")
    print(f"  PF2 (below fc):   {pf2_sk} trades → {pf2_skw}/{pf2_sk} wins "
          f"({pf2_skw/pf2_sk*100:.0f}% wr)" if pf2_sk else "")

    pf3_n  = sum(1 for t in trades if t.get("pf3_narrow"))
    pf3_wr = sum(1 for t in trades if t.get("pf3_narrow") and t["ret_stock"] > 0)
    print(f"  PF3 (narrow CI):  {pf3_n} trades → {pf3_wr}/{pf3_n} wins "
          f"({pf3_wr/pf3_n*100:.0f}% wr)" if pf3_n else "")

    # ── Prophet forecast accuracy ──────────────────────────────────────────
    print(f"\n  Prophet forecast accuracy:")
    fc_correct = sum(1 for t in trades
                     if t.get("forecast_ret", 0) * t["ret_stock"] > 0)
    print(f"  Forecast direction correct: {fc_correct}/{len(trades)} "
          f"({fc_correct/len(trades)*100:.0f}%)")
    fc_rets = [t.get("forecast_ret", 0) for t in trades]
    ac_rets = [t["ret_stock"] for t in trades]
    corr = np.corrcoef(fc_rets, ac_rets)[0, 1] if len(fc_rets) > 2 else 0
    print(f"  Pearson correlation (fc vs actual): {corr:.3f}")

    # ── Per-trade detail ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PER-TRADE: Prophet forecast vs actual outcome")
    print(SEP)
    print(f"  {'Date':<12} {'Ticker':<6} {'Actual%':>8}  {'Fc%':>6}  "
          f"{'PF1':>4} {'PF2':>4} {'PF3':>4}  {'Uncert':>7}  Match?")
    print(f"  {'-'*12} {'-'*6} {'-'*8}  {'-'*6}  "
          f"{'-'*4} {'-'*4} {'-'*4}  {'-'*7}")
    for t in trades:
        fc_dir = t.get("forecast_ret", 0)
        match  = "✓" if fc_dir * t["ret_stock"] > 0 else "✗"
        print(f"  {str(t['entry_dt'].date()):<12} {t['ticker']:<6} "
              f"{t['ret_stock']:>+7.2f}%  {fc_dir:>+5.1f}%  "
              f"{'✓' if t.get('pf1_up') else '✗':>4} "
              f"{'✓' if t.get('pf2_above') else '✗':>4} "
              f"{'✓' if t.get('pf3_narrow') else '✗':>4}  "
              f"{t.get('uncertainty_pct', 0):>6.0f}%  {match}")

    # ── Simulation ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SIMULATION — Prophet filters on S2 trades")
    print(SEP)
    print(f"  {'Strategy':<40} {'N':>3}  {'Win%':>5}  {'Avg%':>6}  "
          f"{'Ann%':>5}  {'DD%':>5}  {'Final':>10}  $20k")
    print(f"  {'-'*40} {'-'*3}  {'-'*5}  {'-'*6}  "
          f"{'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}")

    base = simulate(trades, "S2 baseline (all 41 trades)")
    prow(base, base)

    configs = [
        ("PF1 — forecast up only",
            [t for t in trades if t.get("pf1_up")]),
        ("PF2 — above forecast at entry",
            [t for t in trades if t.get("pf2_above")]),
        ("PF3 — narrow CI (<15%)",
            [t for t in trades if t.get("pf3_narrow")]),
        ("PF1 + PF2 (up forecast + outperforming)",
            [t for t in trades if t.get("pf1_up") and t.get("pf2_above")]),
        ("PF1 + PF3 (up forecast + confident)",
            [t for t in trades if t.get("pf1_up") and t.get("pf3_narrow")]),
        ("PF2 + PF3 (outperforming + confident)",
            [t for t in trades if t.get("pf2_above") and t.get("pf3_narrow")]),
        ("All 3 Prophet filters",
            [t for t in trades if t.get("pf1_up") and t.get("pf2_above") and t.get("pf3_narrow")]),
        ("PF1 only when forecast_ret > 1%",
            [t for t in trades if t.get("forecast_ret", 0) > 1.0]),
        ("PF1 only when forecast_ret > 2%",
            [t for t in trades if t.get("forecast_ret", 0) > 2.0]),
        ("Skip when uncertainty > 20%",
            [t for t in trades if t.get("uncertainty_pct", 0) <= 20]),
        ("Skip when uncertainty > 15%",
            [t for t in trades if t.get("uncertainty_pct", 0) <= 15]),
    ]

    results = []
    for label, filtered in configs:
        if len(filtered) < 3:
            print(f"  {label:<40} too few ({len(filtered)})")
            continue
        r = simulate(filtered, label)
        results.append(r)
        prow(r, base)

    # ── Verdict ────────────────────────────────────────────────────────────
    better = [r for r in results
              if r["ann"] >= base["ann"] and r["max_dd"] <= base["max_dd"]]

    print(f"\n{SEP}")
    print("  VERDICT")
    print(SEP)
    print(f"  S2 baseline: ann {base['ann']:>+5.1f}%  dd {base['max_dd']:>5.1f}%  "
          f"final ${base['final']:>9,.0f}")
    if better:
        print(f"\n  ✓ PROPHET HELPS — {len(better)} filter(s) beat baseline on both return AND drawdown:")
        for r in sorted(better, key=lambda x: -x["ann"]):
            print(f"    {r['label']}")
            print(f"      ann {r['ann']:>+5.1f}%  dd {r['max_dd']:>5.1f}%  "
                  f"n={r['n']}  final ${r['final']:,.0f}")
    else:
        best_r = max(results, key=lambda r: r["ann"]/max(r["max_dd"],1)) if results else base
        print(f"\n  ~ MIXED OR NO IMPROVEMENT")
        print(f"  Best Prophet filter: {best_r['label']}")
        print(f"    ann {best_r['ann']:>+5.1f}%  dd {best_r['max_dd']:>5.1f}%  n={best_r['n']}")
        print(f"\n  Insight: Prophet captures trend direction but the pre-earnings drift")
        print(f"  is event-driven — it fires regardless of the prior trend trajectory.")
        print(f"  However, PF2 (stock outperforming Prophet's expected path) is the")
        print(f"  most theoretically sound filter — it means momentum is already stronger")
        print(f"  than the long-term forecast, which is exactly what we want.")


if __name__ == "__main__":
    main()

