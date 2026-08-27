"""
Gap Prediction Backtest
========================
Instead of trading AFTER a gap appears, find patterns in the days
BEFORE a gap occurs — enter at close, exit at next open (capturing
the overnight / pre-market move).

Trade structure:
    Entry : buy at CLOSE on day T  (based on T's signals)
    Exit  : sell at OPEN on day T+1
    Return: (open[T+1] - close[T]) / close[T]

Two gap types predicted:
    GU  Gap Up  (next open > today close by >= threshold)
    GD  Gap Down (next open < today close by >= threshold)

Predictors tested (all measured at end of day T):
    P1  Strong close      close >= 98% of day's high  (closed near top)
    P2  Up day            close > open (green candle)
    P3  High volume       volume > 1.5x 20d avg
    P4  Above 20dma       close > 20-day MA
    P5  Above 50dma       close > 50-day MA
    P6  5-day uptrend     5 consecutive higher closes
    P7  3-day uptrend     3 consecutive higher closes
    P8  Relative strength close 5d return > QQQ 5d return
    P9  Pre-earnings win  in D-20 pre-earnings window
    P10 QQQ regime        QQQ > 150dma
    P11 Friday setup      today is Friday (predict Monday gap)
    P12 Narrow range      (high-low)/close < 1.5% (quiet day = compression)
    P13 Vol expansion     today's volume > yesterday's volume

Usage:
    uv run python -m backend.research.gap_predict_backtest
"""

from __future__ import annotations

import math
import sys

import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP        = "=" * 72
START_CASH = 2_000.0
UNIVERSE   = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
MILESTONES = [5_000, 10_000, 20_000]

# Minimum gap size to label as a "gap event"
GAP_THRESH = 0.50   # %

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
# Build overnight return table
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

def build_overnight_table(qqq_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (ticker, day T).
    overnight_ret = (open[T+1] - close[T]) / close[T]
    All predictors measured at END of day T.
    """
    rows = []
    for ticker in UNIVERSE:
        df      = _px(ticker)
        edates  = fetch_earnings_dates(ticker)
        idx     = df.index.tolist()

        # Pre-earnings window set
        earn_win: set[pd.Timestamp] = set()
        for ann in edates:
            e20 = _nth(df, ann, -20)
            e1  = _nth(df, ann, -1)
            if e20 and e1:
                for d in df.index:
                    if e20 <= d <= e1:
                        earn_win.add(d)

        qqq_close = qqq_df["Close"]

        for i in range(50, len(idx) - 1):
            today = idx[i]
            tmrw  = idx[i + 1]
            if today.year < 2015: continue

            c   = float(df["Close"].loc[today])
            o   = float(df["Open"].loc[today])   if "Open"  in df.columns else c
            h   = float(df["High"].loc[today])   if "High"  in df.columns else c
            lo  = float(df["Low"].loc[today])    if "Low"   in df.columns else c
            v   = float(df["Volume"].loc[today]) if "Volume" in df.columns else 0
            o_t = float(df["Open"].loc[tmrw])    if "Open"  in df.columns else float(df["Close"].loc[tmrw])

            overnight_ret = (o_t - c) / c * 100
            next_gap_up   = overnight_ret >= GAP_THRESH
            next_gap_dn   = overnight_ret <= -GAP_THRESH

            # ── Predictors at end of day T ──────────────────────────────

            close_ser = df["Close"].loc[df.index <= today]

            # P1: closed near high  (close/high >= 0.98)
            p1_strong_close = (c / h >= 0.98) if h > 0 else False

            # P2: green candle
            p2_up_day = c > o

            # P3: volume expansion
            vol_ser   = df["Volume"].loc[df.index <= today] if "Volume" in df.columns else None
            vol_avg20 = float(vol_ser.tail(21).iloc[:-1].mean()) if vol_ser is not None and len(vol_ser) >= 21 else 0
            p3_high_vol = (v > vol_avg20 * 1.5) if vol_avg20 > 0 else False

            # P4: above 20dma
            ma20 = float(close_ser.tail(20).mean()) if len(close_ser) >= 20 else c
            p4_above_ma20 = c > ma20

            # P5: above 50dma
            ma50 = float(close_ser.tail(50).mean()) if len(close_ser) >= 50 else c
            p5_above_ma50 = c > ma50

            # P6: 5-day uptrend (5 consecutive higher closes)
            if len(close_ser) >= 6:
                last5 = close_ser.tail(6).tolist()
                p6_5day = all(last5[j+1] > last5[j] for j in range(5))
            else:
                p6_5day = False

            # P7: 3-day uptrend
            if len(close_ser) >= 4:
                last3 = close_ser.tail(4).tolist()
                p7_3day = all(last3[j+1] > last3[j] for j in range(3))
            else:
                p7_3day = False

            # P8: 5d relative strength vs QQQ
            qc = qqq_close.loc[qqq_close.index <= today]
            stk5 = (close_ser.iloc[-1] / close_ser.iloc[-6] - 1) * 100 if len(close_ser) >= 6 else 0
            qqq5 = (qc.iloc[-1] / qc.iloc[-6] - 1) * 100 if len(qc) >= 6 else 0
            p8_rs = stk5 > float(qqq5)

            # P9: pre-earnings window
            p9_earn_win = today in earn_win

            # P10: QQQ regime
            regime_ok = len(qc) >= 150 and float(qc.iloc[-1]) > float(qc.rolling(150).mean().iloc[-1])

            # P11: Friday (predicts Monday gap)
            p11_friday = today.day_name() == "Friday"

            # P12: narrow range (compression)
            day_range_pct = (h - lo) / c * 100 if c > 0 else 2.0
            p12_narrow = day_range_pct < 1.5

            # P13: volume expansion (today > yesterday)
            if vol_ser is not None and len(vol_ser) >= 2:
                p13_vol_expand = v > float(vol_ser.iloc[-2])
            else:
                p13_vol_expand = False

            rows.append({
                "ticker":         ticker,
                "date":           today,
                "next_date":      tmrw,
                "close":          round(c, 4),
                "overnight_ret":  round(overnight_ret, 3),
                "next_gap_up":    next_gap_up,
                "next_gap_dn":    next_gap_dn,
                "weekday":        today.day_name(),
                "regime_ok":      regime_ok,
                # predictors
                "p1_strong_close":  p1_strong_close,
                "p2_up_day":        p2_up_day,
                "p3_high_vol":      p3_high_vol,
                "p4_above_ma20":    p4_above_ma20,
                "p5_above_ma50":    p5_above_ma50,
                "p6_5day_up":       p6_5day,
                "p7_3day_up":       p7_3day,
                "p8_rs":            p8_rs,
                "p9_earn_win":      p9_earn_win,
                "p11_friday":       p11_friday,
                "p12_narrow":       p12_narrow,
                "p13_vol_expand":   p13_vol_expand,
            })

    df_out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df_out


# ---------------------------------------------------------------------------
# Predictor accuracy analysis
# ---------------------------------------------------------------------------

def predictor_accuracy(df: pd.DataFrame) -> None:
    """For each predictor: what % of days where predictor=True result in a gap up next day?"""
    print(f"\n{SEP}")
    print("  PREDICTOR ACCURACY — % of days that result in gap-up next open")
    print(f"  Base rate: {df['next_gap_up'].mean()*100:.1f}% of all days gap up next open")
    print(SEP)
    print(f"  {'Predictor':<35} {'N_true':>7}  {'Gap-up%':>8}  {'vs base':>8}  {'Gap-dn%':>8}")
    print(f"  {'-'*35} {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")

    base_gu = df["next_gap_up"].mean() * 100
    predictors = [
        ("P1  Close near high (>=98%)",    "p1_strong_close"),
        ("P2  Green candle (close>open)",   "p2_up_day"),
        ("P3  High volume (>1.5x avg)",     "p3_high_vol"),
        ("P4  Above 20dma",                 "p4_above_ma20"),
        ("P5  Above 50dma",                 "p5_above_ma50"),
        ("P6  5 consecutive up closes",     "p6_5day_up"),
        ("P7  3 consecutive up closes",     "p7_3day_up"),
        ("P8  5d rel strength > QQQ",       "p8_rs"),
        ("P9  Pre-earnings window",         "p9_earn_win"),
        ("P11 Friday",                      "p11_friday"),
        ("P12 Narrow range (<1.5%)",        "p12_narrow"),
        ("P13 Volume expanding",            "p13_vol_expand"),
        ("Regime ok (QQQ>150dma)",          "regime_ok"),
    ]
    for label, col in predictors:
        sub = df[df[col]]
        if len(sub) < 20: continue
        gu_rate = sub["next_gap_up"].mean() * 100
        gd_rate = sub["next_gap_dn"].mean() * 100
        lift    = gu_rate - base_gu
        flag    = "  ◄" if lift >= 2.0 else ""
        print(f"  {label:<35} {len(sub):>7}  {gu_rate:>7.1f}%  {lift:>+7.1f}%  {gd_rate:>7.1f}%{flag}")


# ---------------------------------------------------------------------------
# Simulate overnight strategy
# ---------------------------------------------------------------------------

def simulate_overnight(df: pd.DataFrame, mask: pd.Series,
                       direction: str = "up", label: str = "") -> dict:
    """
    Buy close on days where mask=True, sell next open.
    direction: 'up'=long, 'down'=short
    Pick highest overnight_ret magnitude if same date for multiple tickers.
    """
    sub = df[mask].copy()
    if len(sub) < 10:
        return {"label": label, "n": len(sub), "note": "too few"}

    # Score: for gap-up prediction use overnight_ret as outcome,
    # but we pick by score (strong close + volume) at entry time
    sub["score"] = (
        sub["p1_strong_close"].astype(int) * 2 +
        sub["p2_up_day"].astype(int) +
        sub["p3_high_vol"].astype(int) +
        sub["p8_rs"].astype(int)
    )

    # Pick best ticker per date
    by_date = {}
    for _, row in sub.iterrows():
        d = row["date"]
        if d not in by_date or row["score"] > by_date[d]["score"]:
            by_date[d] = row

    trades = sorted(by_date.values(), key=lambda r: r["date"])

    equity = START_CASH
    peak   = START_CASH
    max_dd = 0.0
    wins   = 0
    rets   = []
    milestones: dict[int, object] = {}
    log    = []

    for row in trades:
        ret = row["overnight_ret"] if direction == "up" else -row["overnight_ret"]
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = row["date"].date()
        equity *= (1 + ret / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if ret > 0: wins += 1
        rets.append(ret)
        log.append({"ticker": row["ticker"], "date": row["date"],
                    "ret": ret, "equity": round(equity, 2),
                    "gap": row["next_gap_up"] if direction == "up" else row["next_gap_dn"]})

    n = len(rets)
    years = 11
    ann   = (equity / START_CASH) ** (1 / years) - 1

    return {
        "label":    label,
        "n":        n,
        "wins":     wins,
        "wr":       round(wins / n * 100, 1),
        "avg":      round(sum(rets) / n, 2),
        "final":    round(equity, 2),
        "x":        round(equity / START_CASH, 1),
        "ann":      round(ann * 100, 1),
        "max_dd":   round(max_dd, 1),
        "milestones": milestones,
        "log":      log,
        "hit_rate": round(sum(1 for t in log if t["gap"]) / n * 100, 1),
    }


def year_by_year(r: dict) -> None:
    by_yr: dict[int, list] = {}
    for t in r["log"]:
        yr = t["date"].year
        by_yr.setdefault(yr, []).append(t["ret"])
    cum = START_CASH
    for yr in sorted(by_yr):
        rets = by_yr[yr]
        avg  = sum(rets) / len(rets)
        w    = sum(1 for r_ in rets if r_ > 0)
        for r_ in rets: cum *= (1 + r_ / 100)
        bar  = ("+" if avg >= 0 else "-") + "█" * min(int(abs(avg) / 0.10), 30)
        print(f"    {yr}  {w:>3}/{len(rets):>3} wins  avg {avg:>+6.3f}%  ${cum:>9,.0f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  GAP PREDICTION BACKTEST — enter close, exit next open")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ"]:
        _px(t)
        print(f"  {t} ready")
    qqq_df = _px("QQQ")

    print("\nBuilding overnight return table...")
    df = build_overnight_table(qqq_df)
    n_gu = df["next_gap_up"].sum()
    n_gd = df["next_gap_dn"].sum()
    print(f"  {len(df)} day-ticker observations")
    print(f"  Next-day gap-ups   (>={GAP_THRESH}%): {n_gu} ({n_gu/len(df)*100:.1f}%)")
    print(f"  Next-day gap-downs (<=-{GAP_THRESH}%): {n_gd} ({n_gd/len(df)*100:.1f}%)")

    # ── Predictor accuracy ─────────────────────────────────────────────────
    predictor_accuracy(df)

    # ── Overnight return distribution by condition ─────────────────────────
    print(f"\n{SEP}")
    print("  OVERNIGHT RETURN DISTRIBUTION — by day-before condition")
    print(SEP)
    print(f"  {'Condition':<40} {'N':>6}  {'Avg ovn%':>9}  {'Med ovn%':>9}  {'Pos%':>6}")
    print(f"  {'-'*40} {'-'*6}  {'-'*9}  {'-'*9}  {'-'*6}")

    conditions = [
        ("Baseline (all days)",
            pd.Series([True]*len(df), index=df.index)),
        ("P1 Close near high",
            df["p1_strong_close"]),
        ("P2 Green candle",
            df["p2_up_day"]),
        ("P1 + P2 (strong green close)",
            df["p1_strong_close"] & df["p2_up_day"]),
        ("P1 + P2 + P5 (+ above 50dma)",
            df["p1_strong_close"] & df["p2_up_day"] & df["p5_above_ma50"]),
        ("P1 + P2 + P5 + regime",
            df["p1_strong_close"] & df["p2_up_day"] & df["p5_above_ma50"] & df["regime_ok"]),
        ("P1 + P2 + P8 (+ rel strength)",
            df["p1_strong_close"] & df["p2_up_day"] & df["p8_rs"]),
        ("P3 High volume",
            df["p3_high_vol"]),
        ("P2 + P3 (green + high vol)",
            df["p2_up_day"] & df["p3_high_vol"]),
        ("P2 + P3 + P5 + regime",
            df["p2_up_day"] & df["p3_high_vol"] & df["p5_above_ma50"] & df["regime_ok"]),
        ("P7 3-day uptrend",
            df["p7_3day_up"]),
        ("P6 5-day uptrend",
            df["p6_5day_up"]),
        ("P9 Pre-earnings window",
            df["p9_earn_win"]),
        ("P9 + regime + P1 + P2",
            df["p9_earn_win"] & df["regime_ok"] & df["p1_strong_close"] & df["p2_up_day"]),
        ("P11 Friday (weekend gap predict)",
            df["p11_friday"]),
        ("P11 + P1 + P2 (strong Friday)",
            df["p11_friday"] & df["p1_strong_close"] & df["p2_up_day"]),
        ("P11 + P1 + P2 + P5 + regime",
            df["p11_friday"] & df["p1_strong_close"] & df["p2_up_day"] & df["p5_above_ma50"] & df["regime_ok"]),
        ("P12 Narrow range (compression)",
            df["p12_narrow"]),
        ("P12 + P5 + regime (tight + trend)",
            df["p12_narrow"] & df["p5_above_ma50"] & df["regime_ok"]),
        ("All strong (P1+P2+P3+P5+P8+regime)",
            df["p1_strong_close"] & df["p2_up_day"] & df["p3_high_vol"] &
            df["p5_above_ma50"] & df["p8_rs"] & df["regime_ok"]),
    ]

    for label, mask in conditions:
        sub = df[mask]
        if len(sub) < 20: continue
        avg = sub["overnight_ret"].mean()
        med = sub["overnight_ret"].median()
        pos = (sub["overnight_ret"] > 0).mean() * 100
        flag = "  ◄" if avg >= 0.08 and pos >= 54 and len(sub) >= 50 else ""
        print(f"  {label:<40} {len(sub):>6}  {avg:>+8.3f}%  {med:>+8.3f}%  {pos:>5.1f}%{flag}")

    # ── Simulate best setups ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SIMULATION — enter close, exit next open ($2,000 start)")
    print(SEP)
    print(f"  {'Strategy':<45} {'N':>5}  {'Hit%':>6}  {'Win%':>6}  {'Avg%':>8}  {'Ann%':>6}  {'MaxDD':>6}  {'Final':>10}")
    print(f"  {'-'*45} {'-'*5}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*10}")

    setups = [
        ("All days (baseline)",
            pd.Series([True]*len(df), index=df.index), "up"),
        ("Strong green close (P1+P2)",
            df["p1_strong_close"] & df["p2_up_day"], "up"),
        ("Strong green + above 50dma + regime",
            df["p1_strong_close"] & df["p2_up_day"] & df["p5_above_ma50"] & df["regime_ok"], "up"),
        ("Strong green + high vol + regime",
            df["p1_strong_close"] & df["p2_up_day"] & df["p3_high_vol"] & df["regime_ok"], "up"),
        ("Strong green + rel str + above 50dma + regime",
            df["p1_strong_close"] & df["p2_up_day"] & df["p8_rs"] & df["p5_above_ma50"] & df["regime_ok"], "up"),
        ("Pre-earnings + strong green + regime",
            df["p9_earn_win"] & df["p1_strong_close"] & df["p2_up_day"] & df["regime_ok"], "up"),
        ("Pre-earnings + regime (no close filter)",
            df["p9_earn_win"] & df["regime_ok"], "up"),
        ("Strong Friday (P11+P1+P2+P5+regime)",
            df["p11_friday"] & df["p1_strong_close"] & df["p2_up_day"] & df["p5_above_ma50"] & df["regime_ok"], "up"),
        ("Friday + regime + P5 (any close)",
            df["p11_friday"] & df["p5_above_ma50"] & df["regime_ok"], "up"),
        ("3-day uptrend + above 50dma + regime",
            df["p7_3day_up"] & df["p5_above_ma50"] & df["regime_ok"], "up"),
        ("Compression + above 50dma + regime",
            df["p12_narrow"] & df["p5_above_ma50"] & df["regime_ok"], "up"),
        ("All strong signals",
            df["p1_strong_close"] & df["p2_up_day"] & df["p3_high_vol"] &
            df["p5_above_ma50"] & df["p8_rs"] & df["regime_ok"], "up"),
    ]

    best = None
    for label, mask, direction in setups:
        r = simulate_overnight(df, mask, direction, label)
        if r.get("note"): continue
        flag = "  ◄" if r["ann"] >= 8 and r["max_dd"] <= 20 else ""
        print(f"  {label:<45} {r['n']:>5}  {r['hit_rate']:>5.1f}%  {r['wr']:>5.1f}%  "
              f"{r['avg']:>+7.3f}%  {r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  "
              f"${r['final']:>9,.0f}{flag}")
        if flag and (best is None or r["ann"] > best["ann"]):
            best = r

    # ── Year-by-year for best ──────────────────────────────────────────────
    if best:
        print(f"\n{SEP}")
        print(f"  YEAR-BY-YEAR — {best['label']}")
        print(SEP)
        year_by_year(best)

        print(f"\n{SEP}")
        print(f"  BEST OVERNIGHT TRADES — {best['label']}")
        print(SEP)
        top = sorted(best["log"], key=lambda x: -x["ret"])[:15]
        print(f"  {'Date':<12} {'Ticker':<8} {'Ret%':>8}  Gap?")
        for t in top:
            print(f"  {str(t['date'].date()):<12} {t['ticker']:<8} {t['ret']:>+7.2f}%  {'✓' if t['gap'] else '—'}")

    # ── Friday premium analysis ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  FRIDAY → MONDAY GAP PREMIUM")
    print(SEP)
    fri = df[df["p11_friday"]]
    for label, mask in [
        ("All Fridays",         fri.index),
        ("Strong Fri (P1+P2)",  fri[fri["p1_strong_close"] & fri["p2_up_day"]].index),
        ("Weak Fri (not P2)",   fri[~fri["p2_up_day"]].index),
        ("Fri + P5 + regime",   fri[fri["p5_above_ma50"] & fri["regime_ok"]].index),
    ]:
        sub = df.loc[mask]
        if len(sub) < 10: continue
        avg = sub["overnight_ret"].mean()
        pos = (sub["overnight_ret"] > 0).mean() * 100
        gap_rate = sub["next_gap_up"].mean() * 100
        print(f"  {label:<30} n={len(sub):>4}  avg {avg:>+6.3f}%  pos {pos:>4.1f}%  gap-up rate {gap_rate:>4.1f}%")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  OVERNIGHT RETURN — per ticker summary")
    print(SEP)
    print(f"  {'Ticker':<8} {'N':>6}  {'Avg ovn%':>9}  {'Pos%':>6}  {'Gap-up rate':>12}")
    for ticker in UNIVERSE:
        sub = df[df["ticker"] == ticker]
        avg = sub["overnight_ret"].mean()
        pos = (sub["overnight_ret"] > 0).mean() * 100
        gu  = sub["next_gap_up"].mean() * 100
        print(f"  {ticker:<8} {len(sub):>6}  {avg:>+8.3f}%  {pos:>5.1f}%  {gu:>11.1f}%")


if __name__ == "__main__":
    main()

