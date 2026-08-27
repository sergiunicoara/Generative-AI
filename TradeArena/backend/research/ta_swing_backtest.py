"""
Technical Analysis Swing Trading Backtest
==========================================
Tests pattern-based entries targeting 3% in 3-5 days.

Patterns tested:
  P1  3 consecutive green candles (close > open × 3 days)
  P2  MACD line crosses above signal (bullish cross)
  P3  MACD histogram turning positive (momentum shift)
  P4  P1 + P2 combined
  P5  P1 + P2 + above 20dma
  P6  P1 + P2 + above 20dma + volume > avg
  P7  RSI oversold bounce (RSI crosses above 40 from below)
  P8  Bollinger Band lower bounce (close crosses back above lower band)

Exit strategies (tested for each pattern):
  E1  3% profit target OR -1.5% stop OR 5-day time limit
  E2  3% profit target OR -2.0% stop OR 5-day time limit
  E3  Time only — hold 5 days, no target/stop

Universe: our 6 tickers + optional broader set
Regime filter: QQQ > 150dma (same as S2)

Usage:
    uv run python -m backend.research.ta_swing_backtest
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
UNIVERSE_6 = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
# Broader universe — more liquid tech/growth stocks for more trade opportunities
UNIVERSE_BROAD = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD",
                  "TSLA", "AAPL", "NFLX", "CRM", "AVGO", "QCOM"]
MILESTONES = [5_000, 10_000, 20_000, 50_000]

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
# Technical indicators
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _macd(close: pd.Series):
    ema12   = _ema(close, 12)
    ema26   = _ema(close, 26)
    line    = ema12 - ema26
    signal  = _ema(line, 9)
    hist    = line - signal
    return line, signal, hist

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _bb_lower(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    ma  = close.rolling(period).mean()
    sd  = close.rolling(period).std()
    return ma - std * sd


# ---------------------------------------------------------------------------
# Build all signals for a universe
# ---------------------------------------------------------------------------

def build_signals(universe: list[str], qqq_df: pd.DataFrame) -> list[dict]:
    """
    Scan daily for all TA pattern signals.
    Returns list of potential trade setups (before exit simulation).
    """
    qqq_close = qqq_df["Close"]
    signals   = []

    for ticker in universe:
        df = _px(ticker)
        if df.empty or len(df) < 60:
            continue

        c = df["Close"]
        o = df["Open"]   if "Open"   in df.columns else None
        v = df["Volume"] if "Volume" in df.columns else None
        h = df["High"]   if "High"   in df.columns else None
        l = df["Low"]    if "Low"    in df.columns else None

        macd_line, macd_sig, macd_hist = _macd(c)
        rsi_vals  = _rsi(c)
        bb_low    = _bb_lower(c)
        ma20      = c.rolling(20).mean()
        ma50      = c.rolling(50).mean()
        vol_avg   = v.rolling(20).mean() if v is not None else None

        idx = df.index.tolist()

        for i in range(60, len(idx) - 6):
            today = idx[i]
            if today.year < 2015:
                continue

            # Regime filter
            qc = qqq_close.loc[qqq_close.index <= today]
            if len(qc) < 150:
                continue
            regime_ok = float(qc.iloc[-1]) > float(qc.rolling(150).mean().iloc[-1])
            if not regime_ok:
                continue

            # Current values
            c0   = float(c.iloc[i])
            c1   = float(c.iloc[i-1])
            c2   = float(c.iloc[i-2])
            c3   = float(c.iloc[i-3])
            o0   = float(o.iloc[i])   if o is not None else c1
            o1   = float(o.iloc[i-1]) if o is not None else c2
            o2   = float(o.iloc[i-2]) if o is not None else c3

            ml0  = float(macd_line.iloc[i]);   ml1 = float(macd_line.iloc[i-1])
            ms0  = float(macd_sig.iloc[i]);    ms1 = float(macd_sig.iloc[i-1])
            mh0  = float(macd_hist.iloc[i]);   mh1 = float(macd_hist.iloc[i-1])
            rsi0 = float(rsi_vals.iloc[i]);    rsi1= float(rsi_vals.iloc[i-1])
            bbl0 = float(bb_low.iloc[i])
            ma20_v = float(ma20.iloc[i]) if not pd.isna(ma20.iloc[i]) else 0
            vol0 = float(v.iloc[i]) if v is not None else 0
            vavg = float(vol_avg.iloc[i]) if vol_avg is not None and not pd.isna(vol_avg.iloc[i]) else 1

            # Entry price = next day open
            if i + 1 >= len(idx):
                continue
            entry_date = idx[i + 1]
            entry_px   = float(o.iloc[i+1]) if o is not None else float(c.iloc[i+1])

            # ── Pattern detection ──────────────────────────────────────
            # P1: 3 green candles
            p1 = (c0 > o0) and (c1 > o1) and (c2 > o2)

            # P2: MACD bullish cross (line crosses above signal)
            p2 = (ml0 > ms0) and (ml1 <= ms1)

            # P3: MACD histogram turns positive
            p3 = (mh0 > 0) and (mh1 <= 0)

            # P4: P1 + P2
            p4 = p1 and p2

            # P5: P1 + P2 + above 20dma
            p5 = p1 and p2 and (c0 > ma20_v)

            # P6: P1 + P2 + above 20dma + volume surge
            p6 = p5 and (vol0 > vavg * 1.2)

            # P7: RSI bounce — was below 40, now above 40
            p7 = (rsi0 >= 40) and (rsi1 < 40)

            # P8: Bollinger Band lower bounce
            p8 = (c1 <= bbl0) and (c0 > bbl0)

            active_patterns = {
                "P1_3green":       p1,
                "P2_macd_cross":   p2,
                "P3_macd_hist":    p3,
                "P4_3g+macd":      p4,
                "P5_3g+macd+ma20": p5,
                "P6_3g+macd+vol":  p6,
                "P7_rsi_bounce":   p7,
                "P8_bb_bounce":    p8,
            }

            fired = [k for k, v in active_patterns.items() if v]
            if not fired:
                continue

            signals.append({
                "ticker":     ticker,
                "signal_date":today,
                "entry_date": entry_date,
                "entry_px":   entry_px,
                "patterns":   fired,
                "rsi":        round(rsi0, 1),
                "macd_hist":  round(mh0, 4),
                "above_ma20": c0 > ma20_v,
                "vol_ratio":  round(vol0/vavg, 2) if vavg > 0 else 1.0,
                # Forward prices for exit simulation (up to 10 days)
                "_future_idx": i + 1,
                "_df_idx":     idx,
                "_close":      c,
                "_open":       o,
            })

    return sorted(signals, key=lambda s: s["entry_date"])


# ---------------------------------------------------------------------------
# Simulate exit for a signal
# ---------------------------------------------------------------------------

def simulate_exit(sig: dict, tp_pct: float, sl_pct: float,
                  max_days: int) -> dict:
    """
    Given an entry, simulate exit with TP/SL/time limit.
    Returns ret%, exit_date, exit_reason.
    """
    i0    = sig["_future_idx"]
    idx   = sig["_df_idx"]
    close = sig["_close"]
    o_ser = sig["_open"]
    ep    = sig["entry_px"]

    tp = ep * (1 + tp_pct / 100)
    sl = ep * (1 + sl_pct / 100)   # sl_pct is negative

    for day_num in range(1, max_days + 1):
        ii = i0 + day_num
        if ii >= len(idx):
            break
        day = idx[ii]

        hi = float(close.iloc[ii])   # use close as proxy (no intraday sim)
        lo = float(close.iloc[ii])

        # Check stop first (conservative)
        if lo <= sl:
            return {"ret": round((sl - ep) / ep * 100, 3),
                    "exit_date": day, "reason": "stop",
                    "days_held": day_num}
        # Check target
        if hi >= tp:
            return {"ret": round(tp_pct, 3),
                    "exit_date": day, "reason": "target",
                    "days_held": day_num}

    # Time exit — use last available close
    last_ii = min(i0 + max_days, len(idx) - 1)
    last_px = float(close.iloc[last_ii])
    return {"ret": round((last_px - ep) / ep * 100, 3),
            "exit_date": idx[last_ii], "reason": "time",
            "days_held": max_days}


# ---------------------------------------------------------------------------
# Build full trade list for a pattern + exit config
# ---------------------------------------------------------------------------

def build_trades(signals: list[dict], pattern: str,
                 tp_pct: float, sl_pct: float,
                 max_days: int) -> list[dict]:
    """Filter signals by pattern, simulate exit, return trade list."""
    trades = []
    for sig in signals:
        if pattern not in sig["patterns"]:
            continue
        exit_info = simulate_exit(sig, tp_pct, sl_pct, max_days)
        trades.append({
            "ticker":     sig["ticker"],
            "entry_date": sig["entry_date"],
            "exit_date":  exit_info["exit_date"],
            "entry_px":   sig["entry_px"],
            "ret":        exit_info["ret"],
            "reason":     exit_info["reason"],
            "days_held":  exit_info["days_held"],
            "rsi":        sig["rsi"],
            "vol_ratio":  sig["vol_ratio"],
        })
    return trades


# ---------------------------------------------------------------------------
# Simulate sequential equity
# ---------------------------------------------------------------------------

def simulate(trades: list[dict], label: str) -> dict:
    # Per day: pick one trade (avoid overlaps — sequential)
    by_date: dict = {}
    for t in trades:
        d = t["entry_date"]
        if d not in by_date:
            by_date[d] = t

    taken, busy = [], None
    for t in sorted(by_date.values(), key=lambda x: x["entry_date"]):
        if busy and t["entry_date"] <= busy:
            continue
        taken.append(t)
        busy = t["exit_date"]

    if not taken:
        return {"label": label, "n": 0, "wr": 0, "avg": 0,
                "ann": 0, "max_dd": 0, "final": START_CASH,
                "milestones": {}, "log": []}

    equity = START_CASH; peak = START_CASH; max_dd = 0.0
    wins = 0; rets = []; milestones = {}; log = []

    for t in taken:
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = t["entry_date"].date()
        equity *= (1 + t["ret"] / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if t["ret"] > 0: wins += 1
        rets.append(t["ret"])
        log.append({"ticker": t["ticker"], "date": t["entry_date"],
                    "ret": t["ret"], "reason": t["reason"],
                    "days": t["days_held"], "equity": round(equity, 2)})

    n = len(rets); years = 11
    ann = (equity / START_CASH) ** (1/years) - 1

    target_hits = sum(1 for t in taken if t["reason"] == "target")
    stops       = sum(1 for t in taken if t["reason"] == "stop")

    return {
        "label": label, "n": n, "wins": wins,
        "wr":  round(wins/n*100, 1),
        "avg": round(sum(rets)/n, 2),
        "final": round(equity, 2),
        "ann":  round(ann*100, 1),
        "max_dd": round(max_dd, 1),
        "target_hits": target_hits, "stops": stops,
        "milestones": milestones, "log": log,
    }


def prow(r: dict, base: dict) -> None:
    if r["n"] == 0:
        print(f"  {r['label']:<46} no trades")
        return
    d_ann = r["ann"] - base["ann"]
    d_dd  = r["max_dd"] - base["max_dd"]
    flag  = "  ◄ BEATS S2"    if d_ann >= 0 and d_dd <= 0 else \
            "  ◄ BEST TA"     if r["ann"] >= 15 and r["max_dd"] <= 20 else \
            "  ↑ viable"      if r["ann"] >= 10 and r["max_dd"] <= 25 else ""
    tgt   = f"tgt:{r.get('target_hits',0)}"
    stp   = f"stp:{r.get('stops',0)}"
    print(f"  {r['label']:<46} {r['n']:>4}  {r['wr']:>5.1f}%  {r['avg']:>+6.2f}%  "
          f"{r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>8,.0f}  "
          f"{tgt} {stp}{flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  TA SWING BACKTEST — 3 green candles + MACD (2015–2026, $2,000)")
    print(SEP)

    print("\nLoading prices (6 core tickers + 6 broad)...")
    all_tickers = list(dict.fromkeys(UNIVERSE_6 + UNIVERSE_BROAD + ["QQQ"]))
    for t in all_tickers:
        _px(t); print(f"  {t} ready")
    qqq_df = _px("QQQ")

    # ── Build signals ────────────────────────────────────────────────────
    print(f"\nBuilding signals — core 6 tickers...")
    sigs_6 = build_signals(UNIVERSE_6, qqq_df)
    print(f"  {len(sigs_6)} signal events on core 6")

    print(f"Building signals — broad 12 tickers...")
    sigs_broad = build_signals(UNIVERSE_BROAD, qqq_df)
    print(f"  {len(sigs_broad)} signal events on broad 12")

    # ── Pattern frequency ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SIGNAL FREQUENCY — how often each pattern fires (core 6, per year)")
    print(SEP)
    all_patterns = ["P1_3green","P2_macd_cross","P3_macd_hist",
                    "P4_3g+macd","P5_3g+macd+ma20","P6_3g+macd+vol",
                    "P7_rsi_bounce","P8_bb_bounce"]
    for p in all_patterns:
        n = sum(1 for s in sigs_6 if p in s["patterns"])
        print(f"  {p:<22}  {n:>4} signals  ({n/11:.1f}/yr  {n/6/11:.1f}/ticker/yr)")

    # ── Main simulation table ────────────────────────────────────────────
    s2_ref = {"label": "S2 baseline (D-20)", "n": 41, "wr": 73.2,
              "avg": 6.61, "ann": 24.3, "max_dd": 6.9,
              "final": 21_966, "target_hits": 0, "stops": 0, "milestones": {}}

    exit_configs = [
        ("TP+3% / SL-1.5% / 5d", 3.0, -1.5, 5),
        ("TP+3% / SL-2.0% / 5d", 3.0, -2.0, 5),
        ("TP+3% / SL-1.5% / 3d", 3.0, -1.5, 3),
        ("Time only 5d",          99.0, -99.0, 5),
    ]

    for universe_label, sigs in [("CORE 6", sigs_6), ("BROAD 12", sigs_broad)]:
        print(f"\n{SEP}")
        print(f"  SIMULATION — {universe_label} tickers")
        print(SEP)
        print(f"  {'Strategy':<46} {'N':>4}  {'Win%':>5}  {'Avg%':>6}  "
              f"{'Ann%':>5}  {'DD%':>5}  {'Final':>9}  tgt/stp")
        print(f"  {'-'*46} {'-'*4}  {'-'*5}  {'-'*6}  "
              f"{'-'*5}  {'-'*5}  {'-'*9}")
        prow(s2_ref, s2_ref)
        print()

        best_r = None
        for pattern in all_patterns:
            for exit_label, tp, sl, days in exit_configs:
                trades = build_trades(sigs, pattern, tp, sl, days)
                if len(trades) < 10:
                    continue
                label = f"{pattern} | {exit_label}"
                r = simulate(trades, label)
                if r["n"] < 10:
                    continue
                # Only print if ann > 8% to reduce noise
                if r["ann"] >= 8 or (r["ann"] >= 5 and r["max_dd"] <= 15):
                    prow(r, s2_ref)
                if best_r is None or r["ann"] / max(r["max_dd"],1) > best_r["ann"] / max(best_r["max_dd"],1):
                    best_r = r

        if best_r:
            print(f"\n  Best risk-adjusted on {universe_label}: {best_r['label']}")
            print(f"  ann {best_r['ann']:>+.1f}%  dd {best_r['max_dd']:.1f}%"
                  f"  avg {best_r['avg']:>+.2f}%/trade  n={best_r['n']}")

    # ── Deep dive on P4 (3 green + MACD) ─────────────────────────────────
    print(f"\n{SEP}")
    print("  DEEP DIVE — P4 (3 green candles + MACD cross) per ticker")
    print(SEP)
    print(f"  {'Ticker':<8} {'N':>4}  {'Win%':>5}  {'Avg%':>7}  "
          f"{'Tgt%':>5}  {'Stp%':>5}  Notes")
    print(f"  {'-'*8} {'-'*4}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*5}")

    for ticker in UNIVERSE_6:
        sub = [s for s in sigs_6 if "P4_3g+macd" in s["patterns"]
               and s["ticker"] == ticker]
        if not sub: continue
        trades = []
        for sig in sub:
            e = simulate_exit(sig, 3.0, -1.5, 5)
            trades.append({"ret": e["ret"], "reason": e["reason"]})
        wr   = sum(1 for t in trades if t["ret"] > 0) / len(trades) * 100
        avg  = sum(t["ret"] for t in trades) / len(trades)
        tgt  = sum(1 for t in trades if t["reason"] == "target") / len(trades) * 100
        stp  = sum(1 for t in trades if t["reason"] == "stop")   / len(trades) * 100
        flag = "  ◄ viable" if avg >= 1.5 and wr >= 55 else ""
        print(f"  {ticker:<8} {len(trades):>4}  {wr:>4.0f}%  {avg:>+6.2f}%  "
              f"{tgt:>4.0f}%  {stp:>4.0f}%{flag}")

    # ── Combined S2 + TA swing ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  COMBINED — S2 (D-20) + best TA pattern when S2 is idle")
    print(SEP)

    from backend.research.signal_backtest import fetch_earnings_dates

    def _nth(df, ref, offset):
        dt = pd.Timestamp(ref); d = 1 if offset >= 0 else -1; c = 0
        for i in range(1, 300):
            cand = dt + pd.Timedelta(days=i*d)
            if cand in df.index:
                c += 1
                if c == abs(offset): return cand
        return None

    def _bounded(x): return max(-20., min(20., x)) / 100.
    def _score(ticker, entry_dt):
        df  = _px(ticker)
        col = df["Close"].loc[df.index <= entry_dt]
        if len(col) < 62: return 1.0
        m20 = (col.iloc[-1]/col.iloc[-21]-1)*100
        m60 = (col.iloc[-1]/col.iloc[-62]-1)*100
        qc  = qqq_df["Close"].loc[qqq_df.index <= entry_dt]
        q20 = (qc.iloc[-1]/qc.iloc[-21]-1)*100 if len(qc) >= 21 else 0
        from ta_swing_backtest import BASE_QUALITY
        bq  = {"GOOGL":1.40,"NVDA":1.50,"AMZN":1.20,"MSFT":1.10,"META":1.10,"AMD":1.00}
        rs  = m20 - float(q20)
        return bq.get(ticker,1.0)+1.20*_bounded(m20)+0.80*_bounded(m60)+1.50*_bounded(rs)

    def _gap_up(ticker, entry_dt):
        df = _px(ticker)
        if "Open" not in df.columns: return True
        prev = [d for d in df.index if d < entry_dt]
        return float(df["Open"].loc[entry_dt]) > float(df["Close"].loc[prev[-1]]) if prev else True

    def _regime_ok(dt):
        qc = qqq_df["Close"].loc[qqq_df.index <= dt]
        return len(qc) >= 150 and float(qc.iloc[-1]) > float(qc.rolling(150).mean().iloc[-1])

    # S2 trades
    s2_list = []
    bq = {"GOOGL":1.40,"NVDA":1.50,"AMZN":1.20,"MSFT":1.10,"META":1.10,"AMD":1.00}
    for ticker in UNIVERSE_6:
        df = _px(ticker)
        for ann in fetch_earnings_dates(ticker):
            e_dt = _nth(df, ann, -20)
            x_dt = _nth(df, ann, -1)
            if not e_dt or not x_dt: continue
            if e_dt not in df.index or x_dt not in df.index: continue
            if not _regime_ok(e_dt): continue
            sc = _score(ticker, e_dt)
            if sc < 1.20: continue
            if not _gap_up(ticker, e_dt): continue
            ep = float(df["Close"].loc[e_dt])
            xp = float(df["Close"].loc[x_dt])
            window = [d for d in df.index if e_dt <= d <= x_dt]
            stopped = min(float(df["Close"].loc[d]) for d in window) <= ep * 0.95
            ret = -5.0 if stopped else (xp - ep) / ep * 100
            s2_list.append({"ticker": ticker, "entry_date": e_dt,
                            "exit_date": x_dt, "ret": round(ret,3),
                            "score": sc, "strategy": "S2"})

    # Best TA trades (P4, TP3%/SL-1.5%/5d, broad universe)
    ta_best = build_trades(sigs_broad, "P4_3g+macd", 3.0, -1.5, 5)

    # Combine: S2 priority, TA fills idle periods
    from s2_mg_dynamic_backtest import combine_dynamic
    for t in s2_list: t["strategy"] = "S2"
    for t in ta_best: t["strategy"] = "TA"

    combined = combine_dynamic(s2_list, ta_best)
    r_s2_only   = simulate(s2_list,  "S2 alone")
    r_ta_only   = simulate(ta_best,  "TA alone (P4, broad 12)")
    r_combined  = simulate(combined, "S2 + TA swing (P4)")

    print(f"  {'Strategy':<46} {'N':>4}  {'Win%':>5}  {'Avg%':>6}  "
          f"{'Ann%':>5}  {'DD%':>5}  {'Final':>9}")
    print(f"  {'-'*46} {'-'*4}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*9}")
    for r in [r_s2_only, r_ta_only, r_combined]:
        prow(r, r_s2_only)

    # ── Verdict ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VERDICT")
    print(SEP)

    best_ta = None
    for pattern in all_patterns:
        for tp, sl, days in [(3.0,-1.5,5),(3.0,-2.0,5)]:
            trades = build_trades(sigs_broad, pattern, tp, sl, days)
            if len(trades) < 10: continue
            r = simulate(trades, pattern)
            if r["n"] < 10: continue
            if best_ta is None or r["ann"]/max(r["max_dd"],1) > best_ta["ann"]/max(best_ta["max_dd"],1):
                best_ta = r

    if best_ta and best_ta["ann"] >= 10:
        print(f"\n  ✓ TA SWING HAS EDGE — best pattern: {best_ta['label']}")
        print(f"    ann {best_ta['ann']:>+.1f}%  dd {best_ta['max_dd']:.1f}%"
              f"  avg {best_ta['avg']:>+.2f}%/trade")
        if r_combined["ann"] > r_s2_only["ann"]:
            d = r_combined["ann"] - r_s2_only["ann"]
            print(f"\n  ✓ S2 + TA COMBINED better: +{d:.1f}%/yr vs S2 alone")
        else:
            print(f"\n  ~ S2 + TA combined doesn't improve — TA trades disrupt S2 compounding")
    else:
        print(f"\n  ✗ TA PATTERNS DO NOT PRODUCE RELIABLE EDGE on these tickers")
        print(f"    3 candles + MACD is well-known → arbitraged away on liquid large-caps")
        print(f"    The patterns fire too frequently (~{sum(1 for s in sigs_6 if 'P4_3g+macd' in s['patterns'])//11}/yr)")
        print(f"    with too low average return to compound meaningfully")
        print()
        print(f"  Where TA swing DOES work:")
        print(f"    → Small/mid-cap stocks (less institutional coverage)")
        print(f"    → Crypto (MACD works better on 24/7 markets)")
        print(f"    → With additional context filters (sector rotation, macro regime)")


if __name__ == "__main__":
    main()

