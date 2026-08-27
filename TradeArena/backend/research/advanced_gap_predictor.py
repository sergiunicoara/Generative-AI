"""
Advanced Gap Predictor — Extended Indicator Set
=================================================
Uses a much wider set of indicators, timeframes, and market context
to find the best predictors of overnight gap-ups in pre-earnings window.

New indicators vs previous tests:

  TECHNICAL (daily):
    RSI(14)           momentum oscillator
    MACD signal       trend/momentum crossover
    Bollinger %B      where price sits in BB bands
    ATR ratio         today's range vs 14d ATR (volatility expansion)
    OBV trend         on-balance volume direction
    Stochastic %K     fast stochastic (price vs recent range)

  TIMEFRAME (weekly):
    Weekly close > weekly open     green week
    Weekly close pct of weekly high
    Above 10-week MA               weekly trend

  MARKET CONTEXT (same day):
    QQQ intraday return    how tech market performed today
    QQQ gap               QQQ itself gapped up today
    VIX level             fear/greed — low VIX = cleaner moves
    VIX trend             VIX falling = bullish
    XLK vs QQQ            sector outperformance

  EARNINGS-SPECIFIC:
    D-X bucket            which day in window (D-9 was strongest)
    Prev earnings surprise did last quarter beat consensus?

  COMPOSITE SCORES:
    Momentum score        combines RSI + MACD + stochastic
    Setup quality         combines close quality + volume + weekly trend
    Market alignment      QQQ + VIX + XLK all agreeing

Goal: find features with lift >= +3% on gap-up rate OR avg >= +0.20%
Then build a composite entry score and simulate with it.

Usage:
    uv run python -m backend.research.advanced_gap_predictor
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache

import numpy as np
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
        df = yf.download(ticker, start="2012-01-01", end="2026-05-17",
                         interval="1d", progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
        _cache[ticker] = df
    return _cache[ticker]

# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
    return float(100 - 100 / (1 + rs))

def macd_signal(series: pd.Series) -> float:
    """Returns MACD line - signal line. Positive = bullish."""
    if len(series) < 26: return 0.0
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    return float(macd.iloc[-1] - sig.iloc[-1])

def bollinger_pct_b(series: pd.Series, period: int = 20) -> float:
    """0 = at lower band, 1 = at upper band, >1 = overbought."""
    if len(series) < period: return 0.5
    ma  = series.rolling(period).mean().iloc[-1]
    std = series.rolling(period).std().iloc[-1]
    if std == 0: return 0.5
    upper = ma + 2 * std
    lower = ma - 2 * std
    return float((series.iloc[-1] - lower) / (upper - lower))

def stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> float:
    """Fast stochastic %K."""
    if len(close) < period: return 50.0
    h14 = high.tail(period).max()
    l14 = low.tail(period).min()
    if h14 == l14: return 50.0
    return float((close.iloc[-1] - l14) / (h14 - l14) * 100)

def atr_ratio(high: pd.Series, low: pd.Series, close: pd.Series,
              period: int = 14) -> float:
    """Today's range / 14d ATR. >1 = expanding volatility."""
    if len(close) < period + 1: return 1.0
    tr_list = []
    cl = close.tolist(); hi = high.tolist(); lo = low.tolist()
    for i in range(1, min(len(cl), period + 1)):
        tr = max(hi[-i] - lo[-i],
                 abs(hi[-i] - cl[-i-1]),
                 abs(lo[-i] - cl[-i-1]))
        tr_list.append(tr)
    atr = sum(tr_list) / len(tr_list)
    today_range = hi[-1] - lo[-1]
    return float(today_range / atr) if atr > 0 else 1.0

def obv_trend(volume: pd.Series, close: pd.Series, period: int = 10) -> float:
    """OBV 10-day slope normalised. Positive = accumulation."""
    if len(close) < period + 1: return 0.0
    obv = 0.0
    obvs = []
    cl = close.tolist(); vl = volume.tolist()
    for i in range(1, len(cl)):
        obv += vl[i] if cl[i] > cl[i-1] else (-vl[i] if cl[i] < cl[i-1] else 0)
        obvs.append(obv)
    if len(obvs) < period: return 0.0
    slope = (obvs[-1] - obvs[-period]) / period
    avg_v = sum(vl[-period:]) / period if period <= len(vl) else 1
    return float(slope / avg_v) if avg_v > 0 else 0.0

# ---------------------------------------------------------------------------
# Build enriched overnight table
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import fetch_earnings_dates

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
            + 1.20 * _bounded(m20) + 0.80 * _bounded(m60) + 1.50 * _bounded(rs))

def build_table(qqq_df: pd.DataFrame, xlk_df: pd.DataFrame,
                vix_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in UNIVERSE:
        df     = _px(ticker)
        edates = fetch_earnings_dates(ticker)

        # Weekly resample
        weekly = df.resample("W-FRI").agg(
            {"Open": "first", "High": "max", "Low": "min",
             "Close": "last", "Volume": "sum"}
        ).dropna()

        for ann in edates:
            d20 = _nth(df, ann, -20)
            d1  = _nth(df, ann, -1)
            if d20 is None or d1 is None: continue
            window_days = [d for d in df.index if d20 <= d <= d1]
            if not window_days: continue

            score   = _score(ticker, d20, qqq_df)
            # Regime at D-20
            qc      = qqq_df["Close"].loc[qqq_df.index <= d20]
            regime  = len(qc) >= 150 and float(qc.iloc[-1]) > float(qc.rolling(150).mean().iloc[-1])

            for i, day in enumerate(window_days):
                day_idx = df.index.get_loc(day)
                if day_idx + 1 >= len(df.index): continue
                next_day = df.index[day_idx + 1]

                # ── Raw OHLCV ─────────────────────────────────────────────
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
                high_ser  = df["High"].loc[df.index  <= day] if "High"   in df.columns else close_ser
                low_ser   = df["Low"].loc[df.index   <= day] if "Low"    in df.columns else close_ser
                vol_ser   = df["Volume"].loc[df.index <= day] if "Volume" in df.columns else None

                # ── Existing features ─────────────────────────────────────
                vol_avg20 = float(vol_ser.tail(21).iloc[:-1].mean()) if vol_ser is not None and len(vol_ser) >= 21 else 0
                vol_ratio = v / vol_avg20 if vol_avg20 > 0 else 1.0
                close_pct_high = (c - lo) / (h - lo) if h > lo else 0.5
                green_day = c > o
                ma50 = float(close_ser.tail(50).mean()) if len(close_ser) >= 50 else c

                # ── Technical indicators ──────────────────────────────────
                rsi14  = rsi(close_ser.tail(30))   if len(close_ser) >= 15 else 50.0
                macd_s = macd_signal(close_ser)    if len(close_ser) >= 30 else 0.0
                bb_b   = bollinger_pct_b(close_ser) if len(close_ser) >= 20 else 0.5
                stoch  = stochastic_k(high_ser, low_ser, close_ser) if len(close_ser) >= 14 else 50.0
                atr_r  = atr_ratio(high_ser, low_ser, close_ser)    if len(close_ser) >= 15 else 1.0
                obv_t  = obv_trend(vol_ser, close_ser) if vol_ser is not None and len(close_ser) >= 11 else 0.0

                # ── Weekly context ────────────────────────────────────────
                wk = weekly.loc[weekly.index <= day]
                if len(wk) >= 2:
                    wk_c = float(wk["Close"].iloc[-1])
                    wk_o = float(wk["Open"].iloc[-1])
                    wk_h = float(wk["High"].iloc[-1])
                    wk_l = float(wk["Low"].iloc[-1])
                    green_week   = wk_c > wk_o
                    wk_close_pct = (wk_c - wk_l) / (wk_h - wk_l) if wk_h > wk_l else 0.5
                    ma10w = float(wk["Close"].tail(10).mean()) if len(wk) >= 10 else wk_c
                    above_ma10w = wk_c > ma10w
                else:
                    green_week = True; wk_close_pct = 0.5; above_ma10w = True

                # ── Market context (QQQ same day) ─────────────────────────
                qqq_today = qqq_df.loc[qqq_df.index == day]
                if not qqq_today.empty and "Open" in qqq_today.columns:
                    qqq_o = float(qqq_today["Open"].iloc[0])
                    qqq_c = float(qqq_today["Close"].iloc[0])
                    qqq_pc = float(qqq_df["Close"].loc[qqq_df.index < day].iloc[-1]) if len(qqq_df.loc[qqq_df.index < day]) > 0 else qqq_o
                    qqq_day_ret  = (qqq_c - qqq_o) / qqq_o * 100   # intraday
                    qqq_gap_pct  = (qqq_o - qqq_pc) / qqq_pc * 100  # gap
                    qqq_green    = qqq_c > qqq_o
                    qqq_strong   = qqq_day_ret > 0.3
                else:
                    qqq_day_ret = 0.0; qqq_gap_pct = 0.0
                    qqq_green = True; qqq_strong = False

                # Stock intraday vs QQQ intraday
                stk_day_ret = (c - o) / o * 100 if o > 0 else 0.0
                outperf_qqq_today = stk_day_ret > qqq_day_ret

                # ── VIX context ───────────────────────────────────────────
                vix_today = vix_df.loc[vix_df.index <= day]
                if len(vix_today) >= 10:
                    vix_cur   = float(vix_today["Close"].iloc[-1])
                    vix_ma10  = float(vix_today["Close"].tail(10).mean())
                    vix_low   = vix_cur < 20
                    vix_very_low = vix_cur < 15
                    vix_falling  = vix_cur < vix_ma10
                else:
                    vix_cur = 20; vix_low = True; vix_very_low = False; vix_falling = True
                    vix_ma10 = 20

                # ── XLK sector context ────────────────────────────────────
                xlk_today = xlk_df.loc[xlk_df.index == day]
                if not xlk_today.empty and "Open" in xlk_today.columns:
                    xlk_o = float(xlk_today["Open"].iloc[0])
                    xlk_c = float(xlk_today["Close"].iloc[0])
                    xlk_green = xlk_c > xlk_o
                    xlk_day_ret = (xlk_c - xlk_o) / xlk_o * 100
                else:
                    xlk_green = True; xlk_day_ret = 0.0

                # Both QQQ and XLK green today
                mkt_aligned = qqq_green and xlk_green

                # ── Composite scores ──────────────────────────────────────
                # Momentum score: RSI 40-70 + MACD positive + stoch 30-70
                momentum_score = (
                    (1 if 40 <= rsi14 <= 70 else 0) +
                    (1 if macd_s > 0 else 0) +
                    (1 if 30 <= stoch <= 75 else 0)
                )
                # Setup quality: close near high + green + above MA
                setup_score = (
                    (1 if close_pct_high >= 0.70 else 0) +
                    (1 if green_day else 0) +
                    (1 if c > ma50 else 0) +
                    (1 if above_ma10w else 0) +
                    (1 if green_week else 0)
                )
                # Market alignment: QQQ strong + VIX low + XLK green
                mkt_score = (
                    (1 if qqq_strong else 0) +
                    (1 if vix_low else 0) +
                    (1 if vix_falling else 0) +
                    (1 if xlk_green else 0)
                )
                # D-bucket quality
                d_prime = days_until in [2, 3, 4, 7, 8, 9, 13]   # historically strong days

                rows.append({
                    "ticker":        ticker,
                    "ann":           ann,
                    "date":          day,
                    "days_until":    days_until,
                    "overnight_ret": round(overnight_ret, 3),
                    "score":         round(score, 3),
                    "regime_ok":     regime,
                    # raw
                    "green_day":     green_day,
                    "close_pct_high":round(close_pct_high, 3),
                    "vol_ratio":     round(vol_ratio, 2),
                    "above_ma50":    c > ma50,
                    "strong_close":  close_pct_high >= 0.80,
                    # technical
                    "rsi14":         round(rsi14, 1),
                    "macd_bull":     macd_s > 0,
                    "bb_b":          round(bb_b, 3),
                    "stoch_mid":     30 <= stoch <= 75,
                    "atr_normal":    atr_r < 1.5,
                    "obv_positive":  obv_t > 0,
                    "rsi_range":     40 <= rsi14 <= 70,
                    # weekly
                    "green_week":    green_week,
                    "wk_close_pct":  round(wk_close_pct, 3),
                    "above_ma10w":   above_ma10w,
                    "wk_strong":     wk_close_pct >= 0.70 and green_week,
                    # market context
                    "qqq_green":     qqq_green,
                    "qqq_strong":    qqq_strong,
                    "qqq_day_ret":   round(qqq_day_ret, 3),
                    "vix_low":       vix_low,
                    "vix_very_low":  vix_very_low,
                    "vix_falling":   vix_falling,
                    "vix_cur":       round(vix_cur, 1),
                    "xlk_green":     xlk_green,
                    "mkt_aligned":   mkt_aligned,
                    "outperf_today": outperf_qqq_today,
                    # composite scores
                    "momentum_score":momentum_score,
                    "setup_score":   setup_score,
                    "mkt_score":     mkt_score,
                    "d_prime":       d_prime,
                    # total entry score
                    "entry_score":   momentum_score + setup_score + mkt_score + (1 if d_prime else 0),
                })

    df_out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df_out


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------

def simulate(trades: list[dict], label: str = "") -> dict:
    if not trades:
        return {"label": label, "n": 0, "final": START_CASH,
                "wr": 0, "avg": 0, "ann": 0, "max_dd": 0,
                "milestones": {}, "log": []}

    by_date: dict = {}
    for t in trades:
        d = t["date"]
        sc = t.get("entry_score", t.get("score", 0))
        if d not in by_date or sc > by_date[d].get("entry_score", by_date[d].get("score", 0)):
            by_date[d] = t

    equity, peak, max_dd = START_CASH, START_CASH, 0.0
    wins, rets, log = 0, [], []
    milestones: dict = {}

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

    n   = len(rets)
    ann = (equity / START_CASH) ** (1 / 11) - 1
    return {
        "label": label, "n": n, "wins": wins,
        "wr":    round(wins / n * 100, 1) if n else 0,
        "avg":   round(sum(rets) / n, 3) if n else 0,
        "final": round(equity, 2),
        "x":     round(equity / START_CASH, 1),
        "ann":   round(ann * 100, 1),
        "max_dd":round(max_dd, 1),
        "milestones": milestones,
        "log":   log,
    }


def prow(r: dict, base: dict | None = None) -> None:
    n_yr = round(r["n"] / 11, 1)
    m10  = str(r["milestones"].get(10_000, "—"))[:10] if r.get("milestones") else "—"
    flag = ""
    if base:
        better_ret = r["ann"] > base["ann"]
        better_dd  = r["max_dd"] <= base["max_dd"] + 2
        if better_ret and better_dd: flag = "  ◄ BETTER"
        elif better_ret:             flag = "  ↑ return"
    print(f"  {r['label']:<48} {n_yr:>5.1f}  {r['wr']:>5.1f}%  {r['avg']:>+7.3f}%"
          f"  {r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}  {m10}{flag}")


def year_by_year(r: dict) -> None:
    by_yr: dict[int, list] = {}
    for t in r["log"]:
        by_yr.setdefault(t["date"].year, []).append(t["overnight_ret"])
    cum = START_CASH
    for yr in sorted(by_yr):
        rets = by_yr[yr]
        avg  = sum(rets) / len(rets)
        w    = sum(1 for r_ in rets if r_ > 0)
        for r_ in rets: cum *= (1 + r_ / 100)
        bar  = ("+" if avg >= 0 else "-") + "█" * min(int(abs(avg) / 0.10), 30)
        print(f"    {yr}  {w:>3}/{len(rets):>3}  avg {avg:>+6.3f}%  ${cum:>9,.0f}  {bar}")


# ---------------------------------------------------------------------------
# Feature importance table
# ---------------------------------------------------------------------------

def feature_importance(df: pd.DataFrame) -> None:
    base_avg = df["overnight_ret"].mean()
    base_gu  = (df["overnight_ret"] > 0).mean() * 100

    features = [
        # Technical
        ("RSI 40-70 (not extreme)",    "rsi_range"),
        ("MACD bullish",               "macd_bull"),
        ("Stoch 30-75 (mid-range)",    "stoch_mid"),
        ("ATR normal (<1.5x)",         "atr_normal"),
        ("OBV positive trend",         "obv_positive"),
        ("BB %B < 0.8 (not OB)",       None),   # computed inline
        # Candle / daily
        ("Green day",                  "green_day"),
        ("Strong close (top 20%)",     "strong_close"),
        ("Close pct high >= 0.70",     None),
        ("Above 50dma",                "above_ma50"),
        ("High volume (>1.5x)",        None),
        # Weekly
        ("Green week",                 "green_week"),
        ("Weekly strong close",        "wk_strong"),
        ("Above 10-week MA",           "above_ma10w"),
        # Market context
        ("QQQ green today",            "qqq_green"),
        ("QQQ strong (>+0.3%)",        "qqq_strong"),
        ("VIX < 20",                   "vix_low"),
        ("VIX < 15 (very low)",        "vix_very_low"),
        ("VIX falling",                "vix_falling"),
        ("XLK green today",            "xlk_green"),
        ("Market aligned (QQQ+XLK)",   "mkt_aligned"),
        ("Stock outperf QQQ today",    "outperf_today"),
        # D-window
        ("Prime D-day (9,8,7,4,3,2)", "d_prime"),
        ("D-3 to D-1 only",            None),
        ("D-10 to D-1 only",           None),
    ]

    print(f"\n{SEP}")
    print(f"  FEATURE IMPORTANCE — lift vs base avg {base_avg:+.3f}%  ({base_gu:.1f}% positive)")
    print(SEP)
    print(f"  {'Feature':<35} {'N':>5}  {'Win%':>6}  {'Avg%':>8}  {'Lift avg':>9}  {'Lift win%':>9}")
    print(f"  {'-'*35} {'-'*5}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*9}")

    for label, col in features:
        if col == "BB %B < 0.8 (not OB)":
            mask = df["bb_b"] < 0.8
        elif col == "Close pct high >= 0.70":
            mask = df["close_pct_high"] >= 0.70
        elif col == "High volume (>1.5x)":
            mask = df["vol_ratio"] >= 1.5
        elif col == "D-3 to D-1 only":
            mask = df["days_until"] <= 3
        elif col == "D-10 to D-1 only":
            mask = df["days_until"] <= 10
        elif col is None:
            continue
        else:
            mask = df[col].astype(bool)

        sub = df[mask]
        if len(sub) < 30: continue
        avg  = sub["overnight_ret"].mean()
        wr   = (sub["overnight_ret"] > 0).mean() * 100
        flag = "  ◄" if (avg - base_avg) >= 0.05 or (wr - base_gu) >= 3 else ""
        print(f"  {label:<35} {len(sub):>5}  {wr:>5.1f}%  {avg:>+7.3f}%  "
              f"{avg-base_avg:>+8.3f}%  {wr-base_gu:>+8.1f}%{flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  ADVANCED GAP PREDICTOR — expanded indicators + timeframes")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ", "XLK"]:
        _px(t); print(f"  {t} ready")

    print("  ^VIX (fear index)...", end=" ")
    try:
        vix = yf.download("^VIX", start="2012-01-01", end="2026-05-17",
                          interval="1d", progress=False, auto_adjust=True)
        if not vix.empty:
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix.index = pd.to_datetime(vix.index).tz_localize(None)
        print(f"{len(vix)} days")
    except Exception as e:
        print(f"failed ({e}) — using dummy VIX=20")
        vix = pd.DataFrame({"Close": [20.0]}, index=[pd.Timestamp("2020-01-01")])

    qqq_df = _px("QQQ")
    xlk_df = _px("XLK")

    print("\nBuilding enriched table (this takes ~60s)...")
    df = build_table(qqq_df, xlk_df, vix)
    regime_df = df[df["regime_ok"]]
    print(f"  {len(df)} observations  |  {len(regime_df)} regime-ok")

    # ── Feature importance ─────────────────────────────────────────────────
    feature_importance(regime_df)

    # ── Composite score distribution ───────────────────────────────────────
    print(f"\n{SEP}")
    print("  COMPOSITE ENTRY SCORE DISTRIBUTION (momentum + setup + market + d-prime)")
    print(SEP)
    print(f"  {'Score':<8} {'N':>5}  {'Win%':>6}  {'Avg%':>8}  {'Lift':>8}")
    base_avg = regime_df["overnight_ret"].mean()
    for sc in sorted(regime_df["entry_score"].unique()):
        sub  = regime_df[regime_df["entry_score"] == sc]
        avg  = sub["overnight_ret"].mean()
        wr   = (sub["overnight_ret"] > 0).mean() * 100
        flag = "  ◄" if avg >= base_avg + 0.05 and wr >= 61 else ""
        print(f"  {sc:<8} {len(sub):>5}  {wr:>5.1f}%  {avg:>+7.3f}%  {avg-base_avg:>+7.3f}%{flag}")

    # ── Individual indicators ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VIX BUCKETS — overnight return by VIX level")
    print(SEP)
    for label, lo, hi in [
        ("<12 (very calm)", 0, 12),
        ("12-15",          12, 15),
        ("15-20",          15, 20),
        ("20-25",          20, 25),
        ("25-30",          25, 30),
        (">30 (fear)",     30, 999),
    ]:
        sub = regime_df[(regime_df["vix_cur"] >= lo) & (regime_df["vix_cur"] < hi)]
        if len(sub) < 20: continue
        avg  = sub["overnight_ret"].mean()
        wr   = (sub["overnight_ret"] > 0).mean() * 100
        flag = "  ◄" if avg >= 0.20 and wr >= 62 else ""
        print(f"  VIX {label:<20} n={len(sub):>4}  {wr:>5.1f}%  {avg:>+7.3f}%{flag}")

    print(f"\n{SEP}")
    print("  QQQ DAY RETURN BUCKETS — overnight return by QQQ's intraday move")
    print(SEP)
    for label, lo, hi in [
        ("QQQ down >1%",    -99, -1.0),
        ("QQQ -1% to -0.3%",-1.0, -0.3),
        ("QQQ flat ±0.3%",  -0.3, 0.3),
        ("QQQ +0.3% to +1%", 0.3, 1.0),
        ("QQQ up >1%",       1.0, 99),
    ]:
        sub = regime_df[(regime_df["qqq_day_ret"] >= lo) & (regime_df["qqq_day_ret"] < hi)]
        if len(sub) < 20: continue
        avg  = sub["overnight_ret"].mean()
        wr   = (sub["overnight_ret"] > 0).mean() * 100
        flag = "  ◄" if avg >= 0.20 and wr >= 62 else ""
        print(f"  {label:<28} n={len(sub):>4}  {wr:>5.1f}%  {avg:>+7.3f}%{flag}")

    # ── Best filter combinations simulation ───────────────────────────────
    print(f"\n{SEP}")
    print("  SIMULATION — enter close, exit next open ($2,000 start)")
    print(SEP)
    hdr = (f"  {'Strategy':<48} {'N/yr':>5}  {'Win%':>5}  {'Avg%':>7}"
           f"  {'Ann%':>5}  {'DD%':>5}  {'Final':>10}  $10k")
    print(hdr)
    print(f"  {'-'*48} {'-'*5}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}")

    def filt(**kwargs) -> list[dict]:
        mask = regime_df["regime_ok"].copy()  # all regime-ok
        for k, v in kwargs.items():
            if k == "score_min":   mask = mask & (regime_df["score"] >= v)
            elif k == "d_max":     mask = mask & (regime_df["days_until"] <= v)
            elif k == "d_min":     mask = mask & (regime_df["days_until"] >= v)
            elif k == "entry_min": mask = mask & (regime_df["entry_score"] >= v)
            elif k == "vix_max":   mask = mask & (regime_df["vix_cur"] <= v)
            elif k == "qqq_ret_min": mask = mask & (regime_df["qqq_day_ret"] >= v)
            else:                  mask = mask & regime_df[k].astype(bool)
        return regime_df[mask].to_dict("records")

    baseline = simulate(filt(), "Baseline (regime only)")

    configs = [
        # Baseline
        ("Baseline (regime only)",              filt()),
        # Technical
        ("+ MACD bullish",                      filt(macd_bull=True)),
        ("+ RSI 40-70",                         filt(rsi_range=True)),
        ("+ MACD + RSI 40-70",                  filt(macd_bull=True, rsi_range=True)),
        ("+ MACD + RSI + OBV positive",         filt(macd_bull=True, rsi_range=True, obv_positive=True)),
        # Market context
        ("+ QQQ green today",                   filt(qqq_green=True)),
        ("+ QQQ strong (>+0.3%)",               filt(qqq_strong=True)),
        ("+ VIX < 20",                          filt(vix_low=True)),
        ("+ VIX < 15",                          filt(vix_very_low=True)),
        ("+ VIX falling",                       filt(vix_falling=True)),
        ("+ Market aligned (QQQ+XLK green)",    filt(mkt_aligned=True)),
        ("+ QQQ green + VIX < 20",              filt(qqq_green=True, vix_low=True)),
        ("+ QQQ green + VIX falling",           filt(qqq_green=True, vix_falling=True)),
        # Weekly
        ("+ Green week",                        filt(green_week=True)),
        ("+ Above 10-week MA",                  filt(above_ma10w=True)),
        ("+ Weekly strong close",               filt(wk_strong=True)),
        # D-window
        ("+ Prime D-day only",                  filt(d_prime=True)),
        ("+ D-10 to D-1",                       filt(d_max=10)),
        ("+ D-5 to D-1",                        filt(d_max=5)),
        ("+ D-3 to D-1",                        filt(d_max=3)),
        # Entry score threshold
        ("+ Entry score >= 5",                  filt(entry_min=5)),
        ("+ Entry score >= 6",                  filt(entry_min=6)),
        ("+ Entry score >= 7",                  filt(entry_min=7)),
        ("+ Entry score >= 8",                  filt(entry_min=8)),
        # Best combos
        ("MACD + RSI + QQQ green + VIX<20",
            filt(macd_bull=True, rsi_range=True, qqq_green=True, vix_low=True)),
        ("MACD + RSI + QQQ green + VIX fall + wk strong",
            filt(macd_bull=True, rsi_range=True, qqq_green=True, vix_falling=True, wk_strong=True)),
        ("Entry score>=6 + QQQ green + VIX<20",
            filt(entry_min=6, qqq_green=True, vix_low=True)),
        ("Entry score>=6 + D-10→D-1 + VIX<20",
            filt(entry_min=6, d_max=10, vix_low=True)),
        ("Entry score>=7 + D-10→D-1",
            filt(entry_min=7, d_max=10)),
        ("Entry score>=7 + D-10→D-1 + VIX<20",
            filt(entry_min=7, d_max=10, vix_low=True)),
        ("Entry score>=8 + D-10→D-1",
            filt(entry_min=8, d_max=10)),
        ("MACD + RSI + green_day + QQQ green + VIX<20 + D-10",
            filt(macd_bull=True, rsi_range=True, green_day=True,
                 qqq_green=True, vix_low=True, d_max=10)),
        ("score>=1.30 + MACD + QQQ green + VIX<20",
            filt(score_min=1.30, macd_bull=True, qqq_green=True, vix_low=True)),
        ("score>=1.30 + entry>=6 + VIX<20 + D-10",
            filt(score_min=1.30, entry_min=6, vix_low=True, d_max=10)),
    ]

    results = []
    for label, trades in configs:
        r = simulate(trades, label)
        results.append(r)
        prow(r, baseline)

    # ── Best result deep dive ──────────────────────────────────────────────
    viable = [r for r in results
              if r["n"] >= 50 and r["ann"] >= baseline["ann"] and r["max_dd"] <= baseline["max_dd"]]
    if viable:
        best = max(viable, key=lambda r: r["ann"] / max(r["max_dd"], 1))
        print(f"\n{SEP}")
        print(f"  BEST: {best['label']}")
        print(f"  {best['n']} trades ({best['n']/11:.1f}/yr)  win {best['wr']:.1f}%  "
              f"avg {best['avg']:+.3f}%  ann {best['ann']:+.1f}%  "
              f"dd {best['max_dd']:.1f}%  final ${best['final']:,.0f}")
        print(SEP)
        print(f"\n  Year-by-year:")
        year_by_year(best)

        print(f"\n  Milestones:")
        for m in MILESTONES:
            ms = best["milestones"].get(m)
            print(f"    ${m:>6,}: {str(ms)[:10] if ms else '—'}")

        # Worst 5
        worst5 = sorted(best["log"], key=lambda x: x["overnight_ret"])[:5]
        print(f"\n  5 worst trades:")
        for t in worst5:
            print(f"    {t['date'].date()}  {t['ticker']:<6}  D-{t['days_until']}  "
                  f"{t['overnight_ret']:>+6.2f}%  VIX={t['vix_cur']}  "
                  f"QQQ={t['qqq_day_ret']:+.2f}%  score={t['score']:.2f}")

    print(f"\n{SEP}")
    print(f"  BASELINE  ann {baseline['ann']:+.1f}%  dd {baseline['max_dd']:.1f}%  "
          f"${baseline['final']:,.0f} from $2,000")
    print(f"  All viable filters vs baseline shown above.")


if __name__ == "__main__":
    main()

