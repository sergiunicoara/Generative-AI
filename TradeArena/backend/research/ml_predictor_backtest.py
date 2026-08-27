"""
ML Predictor Backtest — Gradient Boosting on Overnight Pre-Earnings Trades
===========================================================================
Trains a GradientBoostingClassifier to predict whether an overnight
pre-earnings trade will be profitable (overnight_ret > 0).

Uses strict walk-forward validation — model trained on years T..N,
tested on year N+1. No look-ahead bias.

Features:
    Ticker context:   score, days_until, ticker_encoded
    Momentum:         mom20, rsi14, macd_bull, bb_b, stoch (0–100)
    Candle quality:   close_pct_high, green_day, vol_ratio, atr_ratio
    Weekly:           green_week, wk_close_pct, above_ma10w
    Market context:   qqq_day_ret, qqq_green, vix_cur, vix_falling, xlk_green
    OBV:              obv_trend

Target: overnight_ret > 0  (binary classification)

Walk-forward:
    Train 2015–2019 → test 2020
    Train 2015–2020 → test 2021
    ...
    Train 2015–2025 → test 2026

Output:
    Per-threshold simulation vs baseline
    Feature importance
    Calibration (does P(win)=0.65 actually win 65% of the time?)
    Year-by-year comparison

Usage:
    uv run python -m backend.research.ml_predictor_backtest
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score

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

TICKER_MAP = {t: i for i, t in enumerate(UNIVERSE)}

# Minimum training years before we start predicting
MIN_TRAIN_YEARS = 4

# ---------------------------------------------------------------------------
# Price + indicator helpers (reuse from advanced_gap_predictor)
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

def rsi(series, period=14):
    delta = series.diff().dropna()
    g = delta.where(delta > 0, 0.0).rolling(period).mean()
    l = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = g.iloc[-1] / l.iloc[-1] if l.iloc[-1] != 0 else 100
    return float(100 - 100 / (1 + rs))

def macd_val(series):
    if len(series) < 26: return 0.0
    e12 = series.ewm(span=12, adjust=False).mean()
    e26 = series.ewm(span=26, adjust=False).mean()
    m   = e12 - e26
    s   = m.ewm(span=9, adjust=False).mean()
    return float(m.iloc[-1] - s.iloc[-1])

def bb_pct(series, period=20):
    if len(series) < period: return 0.5
    ma  = series.rolling(period).mean().iloc[-1]
    std = series.rolling(period).std().iloc[-1]
    if std == 0: return 0.5
    return float((series.iloc[-1] - (ma - 2*std)) / (4*std))

def stoch_k(hi, lo, cl, period=14):
    if len(cl) < period: return 50.0
    h = hi.tail(period).max(); l = lo.tail(period).min()
    return float((cl.iloc[-1] - l) / (h - l) * 100) if h != l else 50.0

def atr_r(hi, lo, cl, period=14):
    if len(cl) < period+1: return 1.0
    trs = [max(hi.iloc[-i]-lo.iloc[-i],
               abs(hi.iloc[-i]-cl.iloc[-i-1]),
               abs(lo.iloc[-i]-cl.iloc[-i-1]))
           for i in range(1, period+1)]
    a = sum(trs)/len(trs)
    return float((hi.iloc[-1]-lo.iloc[-1])/a) if a>0 else 1.0

def obv_tr(vol, cl, period=10):
    if len(cl) < period+1: return 0.0
    obv = 0.0; obvs = []
    cl_l = cl.tolist(); vl_l = vol.tolist()
    for i in range(1, len(cl_l)):
        obv += vl_l[i] if cl_l[i]>cl_l[i-1] else (-vl_l[i] if cl_l[i]<cl_l[i-1] else 0)
        obvs.append(obv)
    if len(obvs) < period: return 0.0
    slope = (obvs[-1] - obvs[-period]) / period
    avg_v = sum(vl_l[-period:])/period if period<=len(vl_l) else 1
    return float(slope/avg_v) if avg_v>0 else 0.0

# ---------------------------------------------------------------------------
# Build feature table
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import fetch_earnings_dates

FEATURE_COLS = [
    "ticker_enc", "score", "days_until",
    "mom20", "rsi14", "macd_v", "bb_b", "stoch",
    "close_pct_high", "green_day", "vol_ratio", "atr_ratio", "obv_trend",
    "green_week", "wk_close_pct", "above_ma10w",
    "qqq_day_ret", "qqq_green", "vix_cur", "vix_falling", "xlk_green",
]

def build_feature_table(qqq_df, xlk_df, vix_df) -> pd.DataFrame:
    rows = []
    for ticker in UNIVERSE:
        df     = _px(ticker)
        edates = fetch_earnings_dates(ticker)
        weekly = df.resample("W-FRI").agg(
            {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
        ).dropna()

        for ann in edates:
            d20 = _nth(df, ann, -20); d1 = _nth(df, ann, -1)
            if d20 is None or d1 is None: continue
            window_days = [d for d in df.index if d20 <= d <= d1]
            if not window_days: continue
            score  = _score(ticker, d20, qqq_df)
            regime = _regime_ok(d20, qqq_df)
            if not regime: continue

            for i, day in enumerate(window_days):
                day_idx = df.index.get_loc(day)
                if day_idx + 1 >= len(df.index): continue
                next_day = df.index[day_idx + 1]

                c   = float(df["Close"].loc[day])
                o   = float(df["Open"].loc[day])   if "Open"   in df.columns else c
                h   = float(df["High"].loc[day])   if "High"   in df.columns else c
                lo  = float(df["Low"].loc[day])    if "Low"    in df.columns else c
                v   = float(df["Volume"].loc[day]) if "Volume" in df.columns else 0
                o_t = float(df["Open"].loc[next_day]) if "Open" in df.columns else float(df["Close"].loc[next_day])
                if c == 0: continue

                overnight_ret = (o_t - c) / c * 100
                days_until    = len(window_days) - 1 - i

                close_ser = df["Close"].loc[df.index <= day]
                hi_ser    = df["High"].loc[df.index   <= day] if "High"   in df.columns else close_ser
                lo_ser    = df["Low"].loc[df.index    <= day] if "Low"    in df.columns else close_ser
                vol_ser   = df["Volume"].loc[df.index <= day] if "Volume" in df.columns else None

                # momentum
                mom20 = (c / close_ser.iloc[-21] - 1)*100 if len(close_ser)>=21 else 0

                # candle quality
                close_pct_high = (c-lo)/(h-lo) if h>lo else 0.5
                green_day      = float(c > o)
                vol_avg20 = float(vol_ser.tail(21).iloc[:-1].mean()) if vol_ser is not None and len(vol_ser)>=21 else 0
                vol_ratio = v/vol_avg20 if vol_avg20>0 else 1.0

                # technical
                rsi14   = rsi(close_ser.tail(30))  if len(close_ser)>=15 else 50.0
                macd_v  = macd_val(close_ser)       if len(close_ser)>=30 else 0.0
                bb_b    = bb_pct(close_ser)         if len(close_ser)>=20 else 0.5
                stoch   = stoch_k(hi_ser,lo_ser,close_ser) if len(close_ser)>=14 else 50.0
                atr_ratio_v = atr_r(hi_ser,lo_ser,close_ser) if len(close_ser)>=15 else 1.0
                obv_trend   = obv_tr(vol_ser,close_ser) if vol_ser is not None and len(close_ser)>=11 else 0.0

                # weekly
                wk = weekly.loc[weekly.index <= day]
                if len(wk) >= 2:
                    wk_c=float(wk["Close"].iloc[-1]); wk_o=float(wk["Open"].iloc[-1])
                    wk_h=float(wk["High"].iloc[-1]);  wk_l=float(wk["Low"].iloc[-1])
                    green_week   = float(wk_c > wk_o)
                    wk_close_pct = (wk_c-wk_l)/(wk_h-wk_l) if wk_h>wk_l else 0.5
                    ma10w = float(wk["Close"].tail(10).mean()) if len(wk)>=10 else wk_c
                    above_ma10w  = float(wk_c > ma10w)
                else:
                    green_week=1.0; wk_close_pct=0.5; above_ma10w=1.0

                # QQQ same day
                qqq_row = qqq_df.loc[qqq_df.index == day]
                if not qqq_row.empty and "Open" in qqq_row.columns:
                    qqq_o = float(qqq_row["Open"].iloc[0])
                    qqq_c = float(qqq_row["Close"].iloc[0])
                    qqq_prev = qqq_df["Close"].loc[qqq_df.index < day]
                    qqq_pc = float(qqq_prev.iloc[-1]) if len(qqq_prev)>0 else qqq_o
                    qqq_day_ret = (qqq_c-qqq_o)/qqq_o*100
                    qqq_green   = float(qqq_c > qqq_o)
                else:
                    qqq_day_ret=0.0; qqq_green=1.0

                # VIX
                vix_now = vix_df.loc[vix_df.index <= day]
                if len(vix_now) >= 10:
                    vix_cur     = float(vix_now["Close"].iloc[-1])
                    vix_ma10    = float(vix_now["Close"].tail(10).mean())
                    vix_falling = float(vix_cur < vix_ma10)
                else:
                    vix_cur=20.0; vix_falling=1.0

                # XLK
                xlk_row = xlk_df.loc[xlk_df.index == day]
                if not xlk_row.empty and "Open" in xlk_row.columns:
                    xlk_green = float(float(xlk_row["Close"].iloc[0]) > float(xlk_row["Open"].iloc[0]))
                else:
                    xlk_green = 1.0

                rows.append({
                    "ticker":         ticker,
                    "ticker_enc":     float(TICKER_MAP[ticker]),
                    "date":           day,
                    "year":           day.year,
                    "overnight_ret":  round(overnight_ret, 3),
                    "win":            int(overnight_ret > 0),
                    "score":          round(score, 3),
                    "days_until":     float(days_until),
                    "mom20":          round(mom20, 2),
                    "rsi14":          round(rsi14, 1),
                    "macd_v":         round(macd_v, 4),
                    "bb_b":           round(bb_b, 3),
                    "stoch":          round(stoch, 1),
                    "close_pct_high": round(close_pct_high, 3),
                    "green_day":      green_day,
                    "vol_ratio":      round(vol_ratio, 2),
                    "atr_ratio":      round(atr_ratio_v, 2),
                    "obv_trend":      round(obv_trend, 4),
                    "green_week":     green_week,
                    "wk_close_pct":   round(wk_close_pct, 3),
                    "above_ma10w":    above_ma10w,
                    "qqq_day_ret":    round(qqq_day_ret, 3),
                    "qqq_green":      qqq_green,
                    "vix_cur":        round(vix_cur, 1),
                    "vix_falling":    vix_falling,
                    "xlk_green":      xlk_green,
                })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Walk-forward simulation
# ---------------------------------------------------------------------------

def walk_forward(df: pd.DataFrame, threshold: float) -> dict:
    """
    For each year from first_test_year onwards:
      - Train on all data before that year
      - Predict probabilities for that year
      - Simulate trading using only observations with P(win) >= threshold
    """
    years      = sorted(df["year"].unique())
    first_test = years[MIN_TRAIN_YEARS]   # first year we can test

    equity = START_CASH
    peak   = START_CASH
    max_dd = 0.0
    wins   = losses = 0
    rets   = []
    milestones: dict = {}
    log    = []
    all_probs = []    # for calibration analysis

    for test_year in years:
        if test_year < first_test:
            continue

        train = df[df["year"] < test_year]
        test  = df[df["year"] == test_year]

        if len(train) < 50 or len(test) < 5:
            continue

        X_train = train[FEATURE_COLS].values
        y_train = train["win"].values
        X_test  = test[FEATURE_COLS].values

        # Train calibrated GBM
        base = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        model = CalibratedClassifierCV(base, cv=3, method="isotonic")
        try:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
        except Exception:
            continue

        test_with_probs = test.copy()
        test_with_probs["prob"] = probs

        # Collect calibration data
        for _, row in test_with_probs.iterrows():
            all_probs.append({"prob": row["prob"], "win": row["win"],
                               "ret": row["overnight_ret"]})

        # Simulate: pick best P(win) trade per date, filtered by threshold
        by_date = {}
        for _, row in test_with_probs.iterrows():
            d = row["date"]
            if row["prob"] < threshold:
                continue
            if d not in by_date or row["prob"] > by_date[d]["prob"]:
                by_date[d] = row

        for d in sorted(by_date):
            row = by_date[d]
            for m in MILESTONES:
                if m not in milestones and equity >= m:
                    milestones[m] = d.date()
            ret = row["overnight_ret"]
            equity *= (1 + ret / 100)
            if equity > peak: peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd
            if ret > 0: wins += 1
            else: losses += 1
            rets.append(ret)
            log.append({"date": d, "ticker": row["ticker"],
                        "ret": ret, "prob": row["prob"],
                        "equity": round(equity, 2)})

    n   = wins + losses
    ann = (equity / START_CASH) ** (1/11) - 1 if n > 0 else 0
    return {
        "threshold": threshold,
        "n": n, "wins": wins,
        "wr":     round(wins/n*100, 1) if n else 0,
        "avg":    round(sum(rets)/n, 3) if n else 0,
        "final":  round(equity, 2),
        "ann":    round(ann*100, 1),
        "max_dd": round(max_dd, 1),
        "milestones": milestones,
        "log":    log,
        "all_probs": all_probs,
    }


def baseline(df: pd.DataFrame) -> dict:
    """No ML filter — all regime-ok overnight trades from test years."""
    years      = sorted(df["year"].unique())
    first_test = years[MIN_TRAIN_YEARS]
    test_df    = df[df["year"] >= first_test]

    equity = START_CASH; peak = START_CASH; max_dd = 0.0
    wins = losses = 0; rets = []; log = []; milestones: dict = {}

    by_date = {}
    for _, row in test_df.iterrows():
        d = row["date"]
        if d not in by_date or row["score"] > by_date[d]["score"]:
            by_date[d] = row

    for d in sorted(by_date):
        row = by_date[d]
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = d.date()
        ret = row["overnight_ret"]
        equity *= (1 + ret / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if ret > 0: wins += 1
        else: losses += 1
        rets.append(ret)
        log.append({"date": d, "ticker": row["ticker"], "ret": ret, "equity": round(equity, 2)})

    n = wins + losses
    ann = (equity / START_CASH) ** (1/11) - 1 if n else 0
    return {"n":n,"wins":wins,"wr":round(wins/n*100,1) if n else 0,
            "avg":round(sum(rets)/n,3) if n else 0,
            "final":round(equity,2),"ann":round(ann*100,1),
            "max_dd":round(max_dd,1),"milestones":milestones,"log":log}


# ---------------------------------------------------------------------------
# Feature importance (train on full dataset)
# ---------------------------------------------------------------------------

def get_feature_importance(df: pd.DataFrame) -> list[tuple[str, float]]:
    X = df[FEATURE_COLS].values
    y = df["win"].values
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    clf.fit(X, y)
    imp = sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1])
    return imp


# ---------------------------------------------------------------------------
# Calibration check
# ---------------------------------------------------------------------------

def calibration_report(all_probs: list[dict]) -> None:
    print(f"\n  CALIBRATION — does P(win)=X actually win X% of the time?")
    print(f"  {'Prob bucket':<15} {'N':>5}  {'Actual win%':>11}  {'Avg ret%':>9}  {'Expected':>9}")
    print(f"  {'-'*15} {'-'*5}  {'-'*11}  {'-'*9}  {'-'*9}")
    buckets = [(0.40,0.50),(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,1.01)]
    for lo, hi in buckets:
        sub = [p for p in all_probs if lo <= p["prob"] < hi]
        if len(sub) < 5: continue
        wr  = sum(p["win"] for p in sub) / len(sub) * 100
        avg = sum(p["ret"] for p in sub) / len(sub)
        exp = (lo + hi) / 2 * 100
        flag = "  ✓" if abs(wr - exp) < 8 else "  ✗ miscal"
        print(f"  {lo:.2f}–{hi:.2f}         {len(sub):>5}  {wr:>10.1f}%  {avg:>+8.3f}%  {exp:>8.0f}%{flag}")


# ---------------------------------------------------------------------------
# Year-by-year
# ---------------------------------------------------------------------------

def year_by_year(r: dict) -> None:
    by_yr: dict[int, list] = {}
    for t in r["log"]:
        by_yr.setdefault(t["date"].year, []).append(t["ret"])
    cum = START_CASH
    for yr in sorted(by_yr):
        rets = by_yr[yr]
        avg  = sum(rets)/len(rets)
        w    = sum(1 for r_ in rets if r_>0)
        for r_ in rets: cum *= (1+r_/100)
        bar  = ("+" if avg>=0 else "-") + "█"*min(int(abs(avg)/0.10),25)
        print(f"    {yr}  {w:>3}/{len(rets):>3}  avg {avg:>+6.3f}%  ${cum:>9,.0f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  ML PREDICTOR — GradientBoosting on overnight pre-earnings trades")
    print("  Walk-forward validation (no look-ahead bias)")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ", "XLK"]:
        _px(t); print(f"  {t} ready")

    print("  ^VIX...", end=" ")
    try:
        vix = yf.download("^VIX", start="2012-01-01", end="2026-05-17",
                          interval="1d", progress=False, auto_adjust=True)
        if not vix.empty:
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix.index = pd.to_datetime(vix.index).tz_localize(None)
        print(f"{len(vix)} days")
    except Exception:
        vix = pd.DataFrame({"Close":[20.0]}, index=[pd.Timestamp("2020-01-01")])
        print("failed — using 20")

    qqq_df = _px("QQQ")
    xlk_df = _px("XLK")

    print("\nBuilding feature table (regime-ok observations only)...")
    df = build_feature_table(qqq_df, xlk_df, vix)
    print(f"  {len(df)} observations  |  {df['win'].mean()*100:.1f}% positive")
    print(f"  Years: {df['year'].min()}–{df['year'].max()}")
    print(f"  Test window: {sorted(df['year'].unique())[MIN_TRAIN_YEARS]}–{df['year'].max()}")

    # ── Feature importance (full dataset) ────────────────────────────────
    print(f"\n{SEP}")
    print("  FEATURE IMPORTANCE (trained on full dataset)")
    print(SEP)
    imp = get_feature_importance(df)
    for feat, score in imp:
        bar = "█" * min(int(score * 300), 30)
        print(f"  {feat:<20} {score:.4f}  {bar}")

    # ── Walk-forward at multiple thresholds ───────────────────────────────
    print(f"\n{SEP}")
    print("  WALK-FORWARD SIMULATION — multiple probability thresholds")
    print(f"  (test period starts {sorted(df['year'].unique())[MIN_TRAIN_YEARS]})")
    print(SEP)

    base = baseline(df)
    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
    results = []
    print("  Running walk-forward (this takes ~60s)...")
    for thresh in thresholds:
        r = walk_forward(df, thresh)
        results.append(r)
        print(f"  p>={thresh:.2f} done — {r['n']} trades")

    print(f"\n  {'Threshold':<12} {'N/yr':>5}  {'Win%':>6}  {'Avg%':>8}  "
          f"{'Ann%':>6}  {'DD%':>5}  {'Final':>10}  $10k")
    print(f"  {'-'*12} {'-'*5}  {'-'*6}  {'-'*8}  "
          f"{'-'*6}  {'-'*5}  {'-'*10}  {'-'*10}")

    test_years = 11 - MIN_TRAIN_YEARS
    b_nyr = round(base['n'] / test_years, 1)
    print(f"  {'Baseline':<12} {b_nyr:>5.1f}  {base['wr']:>5.1f}%  {base['avg']:>+7.3f}%  "
          f"{base['ann']:>+5.1f}%  {base['max_dd']:>4.1f}%  ${base['final']:>9,.0f}  "
          f"{str(base['milestones'].get(10_000,'—'))[:10]}")

    for r in results:
        n_yr = round(r['n'] / test_years, 1)
        m10  = str(r["milestones"].get(10_000,"—"))[:10]
        flag = "  ◄ BETTER" if r["ann"] > base["ann"] and r["max_dd"] <= base["max_dd"] else \
               "  ↑ return"  if r["ann"] > base["ann"] else ""
        print(f"  p>={r['threshold']:.2f}      {n_yr:>5.1f}  {r['wr']:>5.1f}%  {r['avg']:>+7.3f}%  "
              f"{r['ann']:>+5.1f}%  {r['max_dd']:>4.1f}%  ${r['final']:>9,.0f}  {m10}{flag}")

    # ── Calibration ────────────────────────────────────────────────────────
    best_r = max(results, key=lambda r: r["ann"] / max(r["max_dd"], 1))
    calibration_report(best_r["all_probs"])

    # ── Best result deep dive ──────────────────────────────────────────────
    viable = [r for r in results
              if r["n"] >= 30 and r["ann"] >= base["ann"] and r["max_dd"] <= base["max_dd"]]
    if not viable:
        viable = [r for r in results if r["n"] >= 20]

    best = max(viable, key=lambda r: r["ann"] / max(r["max_dd"], 1)) if viable else best_r

    print(f"\n{SEP}")
    print(f"  BEST: p>={best['threshold']:.2f}")
    print(f"  {best['n']} trades ({best['n']/test_years:.1f}/yr)  "
          f"win {best['wr']:.1f}%  avg {best['avg']:+.3f}%  "
          f"ann {best['ann']:+.1f}%  dd {best['max_dd']:.1f}%  "
          f"final ${best['final']:,.0f}")
    print(SEP)
    print("\n  Year-by-year:")
    year_by_year(best)

    # ── Top 10 highest-confidence correct calls ───────────────────────────
    correct = sorted([t for t in best["log"] if t["ret"] > 0],
                     key=lambda x: -x["prob"])[:10]
    wrong   = sorted([t for t in best["log"] if t["ret"] <= 0],
                     key=lambda x: -x["prob"])[:5]

    print(f"\n  Top 10 highest-confidence WINS:")
    for t in correct:
        print(f"    {str(t['date'].date())}  {t['ticker']:<6}  "
              f"P={t['prob']:.3f}  ret {t['ret']:>+6.2f}%")

    print(f"\n  Top 5 highest-confidence LOSSES (model failures):")
    for t in wrong:
        print(f"    {str(t['date'].date())}  {t['ticker']:<6}  "
              f"P={t['prob']:.3f}  ret {t['ret']:>+6.2f}%")

    # ── vs baseline summary ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VERDICT — does ML improve overnight O1 strategy?")
    print(SEP)
    print(f"  Baseline (no filter):   ann {base['ann']:>+5.1f}%  dd {base['max_dd']:>5.1f}%  "
          f"final ${base['final']:>9,.0f}  ({b_nyr:.0f} trades/yr)")
    print(f"  Best ML filter p>={best['threshold']:.2f}:  "
          f"ann {best['ann']:>+5.1f}%  dd {best['max_dd']:>5.1f}%  "
          f"final ${best['final']:>9,.0f}  ({best['n']/test_years:.0f} trades/yr)")
    d_ann = best['ann'] - base['ann']
    d_dd  = best['max_dd'] - base['max_dd']
    if d_ann > 0 and d_dd <= 0:
        print(f"\n  ✓ ML IMPROVES: +{d_ann:.1f}% ann return, {d_dd:+.1f}% drawdown")
    elif d_ann > 0:
        print(f"\n  ~ MIXED: +{d_ann:.1f}% return but +{d_dd:.1f}% drawdown")
    else:
        print(f"\n  ✗ NO IMPROVEMENT: ML filter does not beat the baseline.")
        print(f"    The pre-earnings overnight edge is uniformly distributed —")
        print(f"    no combination of indicators reliably predicts the best nights.")


if __name__ == "__main__":
    main()

