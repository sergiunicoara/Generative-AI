"""
Gap Strategies Full Backtest
==============================
Two intraday strategies identified from gap pattern analysis:

  GD  Gap-Down Reversion
      When a strong stock (above 50dma, QQQ regime ok) gaps DOWN,
      mean reversion kicks in. Buy the open, sell the close.
      Edge: 55.4% win, +0.16% avg per trade.

  MG  Monday Gap-Up Continuation
      Weekend gaps in tech tend to follow through.
      Buy Monday open if stock gapped up, sell Monday close.
      Edge: 57.7% win, +0.33% avg per trade.

Both are intraday (open → close), no overnight risk.
Compared against S2 pre-earnings strategy.
Also tested: running GD + MG together, and all three combined.

Capital model:
  - Each strategy runs its own $2,000 account
  - 100% deployment per trade, open→close
  - No stop-loss (intraday, exit at close regardless)
  - One trade at a time per strategy (sequential within same ticker)
  - When multiple tickers signal same day: pick highest score

Usage:
    uv run python -m backend.research.gap_strategies_backtest
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP        = "=" * 72
START_CASH = 2_000.0
UNIVERSE   = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
MILESTONES = [5_000, 10_000, 20_000, 50_000]

# Filter parameters
MIN_GAP_DN  = -0.50   # minimum downward gap % to consider
MIN_GAP_UP  = 0.50    # minimum upward gap % for Monday signal
MA50_WINDOW = 50
REGIME_MA   = 150

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
# Build daily event tables
# ---------------------------------------------------------------------------

def build_events(qqq_df: pd.DataFrame) -> pd.DataFrame:
    """One row per trading day per ticker with all context needed."""
    rows = []
    for ticker in UNIVERSE:
        df  = _px(ticker)
        idx = df.index.tolist()

        # Pre-compute QQQ regime rolling mean
        qqq_close = qqq_df["Close"]

        for i in range(max(MA50_WINDOW, REGIME_MA), len(idx)):
            today = idx[i]
            prev  = idx[i - 1]
            if today.year < 2015: continue

            o  = float(df["Open"].loc[today])  if "Open"  in df.columns else None
            c  = float(df["Close"].loc[today])
            pc = float(df["Close"].loc[prev])
            if o is None or pc == 0: continue

            gap_pct   = (o - pc) / pc * 100
            intra_ret = (c - o)  / o  * 100   # open → close

            # 50dma
            close_ser  = df["Close"].loc[df.index <= today]
            ma50       = float(close_ser.tail(MA50_WINDOW).mean())
            above_ma50 = o > ma50

            # Regime (QQQ > 150dma)
            qc = qqq_close.loc[qqq_close.index <= today]
            regime_ok = len(qc) >= REGIME_MA and float(qc.iloc[-1]) > float(qc.rolling(REGIME_MA).mean().iloc[-1])

            # Dynamic score proxy (20d momentum relative to QQQ)
            mom20 = (close_ser.iloc[-1] / close_ser.iloc[-21] - 1) * 100 if len(close_ser) >= 21 else 0
            q20   = (qc.iloc[-1] / qc.iloc[-21] - 1) * 100 if len(qc) >= 21 else 0
            rs20  = mom20 - float(q20)

            # Volume ratio
            if "Volume" in df.columns:
                v         = float(df["Volume"].loc[today])
                vol_avg   = float(df["Volume"].loc[df.index <= today].tail(21).iloc[:-1].mean())
                vol_ratio = v / vol_avg if vol_avg > 0 else 1.0
            else:
                vol_ratio = 1.0

            rows.append({
                "ticker":     ticker,
                "date":       today,
                "weekday":    today.day_name(),
                "gap_pct":    round(gap_pct, 3),
                "intra_ret":  round(intra_ret, 3),
                "above_ma50": above_ma50,
                "regime_ok":  regime_ok,
                "mom20":      round(mom20, 2),
                "rs20":       round(rs20, 2),
                "vol_ratio":  round(vol_ratio, 2),
            })

    df_out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df_out


# ---------------------------------------------------------------------------
# Strategy: Gap-Down Reversion (GD)
# ---------------------------------------------------------------------------

def gd_signals(events: pd.DataFrame) -> pd.DataFrame:
    """
    Entry: stock gaps DOWN >= 0.5% AND above 50dma AND QQQ regime ok
    Trade: buy open, sell close (intraday)
    Score for conflict resolution: |gap_pct| * (1 + rs20/100) — bigger gap + stronger stock wins
    """
    mask = (
        (events["gap_pct"] <= MIN_GAP_DN) &
        events["above_ma50"] &
        events["regime_ok"]
    )
    sigs = events[mask].copy()
    sigs["score"] = sigs["gap_pct"].abs() * (1 + sigs["rs20"] / 100)
    sigs["trade_ret"] = sigs["intra_ret"]   # long: buy open, sell close
    return sigs


# ---------------------------------------------------------------------------
# Strategy: Monday Gap-Up Continuation (MG)
# ---------------------------------------------------------------------------

def mg_signals(events: pd.DataFrame) -> pd.DataFrame:
    """
    Entry: Monday, stock gaps UP >= 0.5% AND above 50dma AND QQQ regime ok
    Trade: buy open, sell close (intraday)
    Score: gap_pct * vol_ratio — bigger gap + high volume wins
    """
    mask = (
        (events["weekday"] == "Monday") &
        (events["gap_pct"] >= MIN_GAP_UP) &
        events["above_ma50"] &
        events["regime_ok"]
    )
    sigs = events[mask].copy()
    sigs["score"] = sigs["gap_pct"] * sigs["vol_ratio"]
    sigs["trade_ret"] = sigs["intra_ret"]   # long
    return sigs


# ---------------------------------------------------------------------------
# Simulate — sequential, 1 position at a time, pick best score on conflicts
# ---------------------------------------------------------------------------

def simulate(signals: pd.DataFrame, label: str) -> dict:
    if signals.empty:
        return {"label": label, "n": 0, "final": START_CASH}

    # Group by date, pick highest-score ticker per day
    by_date = {}
    for _, row in signals.iterrows():
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
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = row["date"].date()
        ret = row["trade_ret"]
        equity *= (1 + ret / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if ret > 0: wins += 1
        rets.append(ret)
        log.append({"ticker": row["ticker"], "date": row["date"],
                    "ret": ret, "equity": round(equity, 2)})

    n    = len(rets)
    years = 11
    ann  = (equity / START_CASH) ** (1 / years) - 1

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
    }


# ---------------------------------------------------------------------------
# Combined simulation — GD + MG signals merged, pick 1 trade/day
# ---------------------------------------------------------------------------

def simulate_combined(sigs_list: list[pd.DataFrame], label: str) -> dict:
    combined = pd.concat(sigs_list, ignore_index=True)
    return simulate(combined, label)


# ---------------------------------------------------------------------------
# Year-by-year breakdown
# ---------------------------------------------------------------------------

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
# Per-ticker contribution
# ---------------------------------------------------------------------------

def ticker_contrib(r: dict) -> None:
    by_t: dict[str, list] = {}
    for t in r["log"]:
        by_t.setdefault(t["ticker"], []).append(t["ret"])
    print(f"    {'Ticker':<8} {'Trades':>7}  {'Win%':>6}  {'Avg%':>8}  {'Total%':>8}")
    print(f"    {'-'*8} {'-'*7}  {'-'*6}  {'-'*8}  {'-'*8}")
    for tk in sorted(by_t, key=lambda x: -len(by_t[x])):
        rets = by_t[tk]
        w    = sum(1 for r_ in rets if r_ > 0)
        avg  = sum(rets) / len(rets)
        tot  = sum(rets)
        print(f"    {tk:<8} {len(rets):>7}  {w/len(rets)*100:>5.1f}%  {avg:>+7.3f}%  {tot:>+7.1f}%")


# ---------------------------------------------------------------------------
# Parameter sweep — find best gap threshold for GD
# ---------------------------------------------------------------------------

def sweep_gd_threshold(events: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  GD PARAMETER SWEEP — gap threshold & 50dma filter")
    print(SEP)
    print(f"  {'Gap threshold':<20} {'50dma':>6}  {'N':>5}  {'Win%':>6}  {'Avg%':>7}  {'Ann%':>6}  {'MaxDD':>6}  {'Final':>10}")
    print(f"  {'-'*20} {'-'*6}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*10}")

    for thresh in [-0.25, -0.50, -0.75, -1.00, -1.50, -2.00]:
        for ma_filter in [False, True]:
            mask = (events["gap_pct"] <= thresh) & events["regime_ok"]
            if ma_filter:
                mask = mask & events["above_ma50"]
            sigs = events[mask].copy()
            sigs["score"]     = sigs["gap_pct"].abs()
            sigs["trade_ret"] = sigs["intra_ret"]
            r = simulate(sigs, "")
            if r["n"] < 10: continue
            flag = "  ◄" if r["ann"] > 5 and r["max_dd"] < 25 else ""
            print(f"  gap<={thresh:>5.2f}%  {'yes' if ma_filter else 'no':>6}  "
                  f"{r['n']:>5}  {r['wr']:>5.1f}%  {r['avg']:>+6.3f}%  "
                  f"{r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}{flag}")


def sweep_mg_threshold(events: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  MG PARAMETER SWEEP — gap threshold & volume filter")
    print(SEP)
    print(f"  {'Gap threshold':<20} {'Vol>avg':>7}  {'N':>5}  {'Win%':>6}  {'Avg%':>7}  {'Ann%':>6}  {'MaxDD':>6}  {'Final':>10}")
    print(f"  {'-'*20} {'-'*7}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*10}")

    for thresh in [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]:
        for vol_filter in [False, True]:
            mask = ((events["weekday"] == "Monday") &
                    (events["gap_pct"] >= thresh) &
                    events["above_ma50"] &
                    events["regime_ok"])
            if vol_filter:
                mask = mask & (events["vol_ratio"] > 1.0)
            sigs = events[mask].copy()
            sigs["score"]     = sigs["gap_pct"]
            sigs["trade_ret"] = sigs["intra_ret"]
            r = simulate(sigs, "")
            if r["n"] < 10: continue
            flag = "  ◄" if r["ann"] > 5 and r["max_dd"] < 25 else ""
            print(f"  gap>={thresh:>4.2f}%   {'yes' if vol_filter else 'no':>7}  "
                  f"{r['n']:>5}  {r['wr']:>5.1f}%  {r['avg']:>+6.3f}%  "
                  f"{r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}{flag}")


# ---------------------------------------------------------------------------
# S2 reference simulation (pre-earnings, for comparison)
# ---------------------------------------------------------------------------

def load_s2_reference(qqq_df: pd.DataFrame) -> dict:
    """Quick re-run of S2+F6+L8 for the comparison table."""
    from backend.research.signal_backtest import fetch_earnings_dates

    BASE_QUALITY = {"GOOGL":1.40,"NVDA":1.50,"AMZN":1.20,"MSFT":1.10,"META":1.10,"AMD":1.00}
    SCORE_THRESH = 1.20

    def _nth(df, ref, offset):
        dt, d, c = pd.Timestamp(ref), (1 if offset >= 0 else -1), 0
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
            raw.append({"ticker": ticker, "date": e_dt, "ret": round(ret,3),
                        "score": round(sc,3), "ann": ann})

    raw.sort(key=lambda t: t["date"])
    by_date = {}
    for t in raw:
        d = t["date"]
        if d not in by_date or t["score"] > by_date[d]["score"]:
            by_date[d] = t

    # Sequential (respect exit dates)
    from backend.research.signal_backtest import fetch_earnings_dates as _fed
    taken, busy = [], None
    for t in sorted(by_date.values(), key=lambda x: x["date"]):
        df   = _px(t["ticker"])
        dates = _fed(t["ticker"])
        ann  = t["ann"]
        x_dt = _nth(df, ann, -1)
        if busy and t["date"] <= busy: continue
        taken.append(t)
        busy = x_dt

    equity, peak, max_dd = START_CASH, START_CASH, 0.0
    wins, rets = 0, []
    milestones = {}
    log = []
    for t in taken:
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = t["date"].date()
        equity *= (1 + t["ret"] / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if t["ret"] > 0: wins += 1
        rets.append(t["ret"])
        log.append({"ticker": t["ticker"], "date": t["date"], "ret": t["ret"], "equity": round(equity,2)})

    n = len(rets)
    ann = (equity / START_CASH) ** (1/11) - 1
    return {
        "label": "S2 Pre-earnings (F6+L8)",
        "n": n, "wins": wins,
        "wr": round(wins/n*100,1) if n else 0,
        "avg": round(sum(rets)/n,2) if n else 0,
        "final": round(equity,2),
        "x": round(equity/START_CASH,1),
        "ann": round(ann*100,1),
        "max_dd": round(max_dd,1),
        "milestones": milestones,
        "log": log,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  GAP STRATEGIES FULL BACKTEST — GD + MG (2015–2026, $2,000 start)")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ"]:
        _px(t)
        print(f"  {t} ready")
    qqq_df = _px("QQQ")

    print("\nBuilding event table...")
    events = build_events(qqq_df)
    print(f"  {len(events)} trading day-ticker events")

    # ── Parameter sweeps ──────────────────────────────────────────────────
    sweep_gd_threshold(events)
    sweep_mg_threshold(events)

    # ── Build signals with best parameters ────────────────────────────────
    gd_sigs = gd_signals(events)
    mg_sigs = mg_signals(events)
    print(f"\n  GD signals: {len(gd_sigs)}")
    print(f"  MG signals: {len(mg_sigs)}")

    # ── Simulate each strategy ────────────────────────────────────────────
    print("\nRunning simulations...")
    r_gd  = simulate(gd_sigs,  "GD Gap-Down Reversion")
    r_mg  = simulate(mg_sigs,  "MG Monday Gap-Up")
    r_comb= simulate_combined([gd_sigs, mg_sigs], "GD + MG Combined")
    r_s2  = load_s2_reference(qqq_df)

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  STRATEGY COMPARISON (2015–2026, $2,000 start)")
    print(SEP)
    print(f"  {'Strategy':<30} {'N/yr':>5}  {'Win%':>6}  {'Avg%':>8}  "
          f"{'Ann%':>6}  {'MaxDD':>6}  {'Final':>10}  $10k")
    print(f"  {'-'*30} {'-'*5}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}")

    for r in [r_s2, r_gd, r_mg, r_comb]:
        m10 = str(r["milestones"].get(10_000, "—"))[:10] if r.get("milestones") else "—"
        n_yr = round(r["n"] / 11, 1)
        print(f"  {r['label']:<30} {n_yr:>5.1f}  {r['wr']:>5.1f}%  {r['avg']:>+7.3f}%  "
              f"{r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}  {m10}")

    # ── Year-by-year ───────────────────────────────────────────────────────
    for r in [r_gd, r_mg, r_comb]:
        print(f"\n{SEP}")
        print(f"  YEAR-BY-YEAR — {r['label']}")
        print(SEP)
        year_by_year(r)

    # ── Per-ticker breakdown ───────────────────────────────────────────────
    for r in [r_gd, r_mg]:
        print(f"\n{SEP}")
        print(f"  TICKER BREAKDOWN — {r['label']}")
        print(SEP)
        ticker_contrib(r)

    # ── Worst drawdown periods ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  WORST PERIODS — GD Gap-Down Reversion")
    print(SEP)
    log = sorted(r_gd["log"], key=lambda x: x["ret"])[:10]
    print(f"  {'Date':<12} {'Ticker':<8} {'Return':>8}")
    for t in log:
        print(f"  {str(t['date'].date()):<12} {t['ticker']:<8} {t['ret']:>+7.2f}%")

    print(f"\n{SEP}")
    print("  WORST PERIODS — MG Monday Gap-Up")
    print(SEP)
    log = sorted(r_mg["log"], key=lambda x: x["ret"])[:10]
    for t in log:
        print(f"  {str(t['date'].date()):<12} {t['ticker']:<8} {t['ret']:>+7.2f}%")

    # ── Verdict ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VERDICT")
    print(SEP)
    for r in [r_gd, r_mg, r_comb]:
        ann_s  = r["ann"]
        dd     = r["max_dd"]
        viable = ann_s >= 8 and dd <= 30
        print(f"  {r['label']:<30}  ann {ann_s:>+5.1f}%  dd {dd:>5.1f}%  "
              f"{'✓ VIABLE' if viable else '✗ NOT VIABLE'}")

    print(f"\n  S2 reference: ann {r_s2['ann']:>+5.1f}%  dd {r_s2['max_dd']:>5.1f}%")
    print(f"\n  Key question: can GD or MG run ALONGSIDE S2 on same account?")
    print(f"  → Both are intraday (open→close). S2 is multi-week hold.")
    print(f"  → No conflict if run on separate capital allocation.")
    print(f"  → Combined portfolio: split $2,000 into S2 + GD/MG sleeves.")


if __name__ == "__main__":
    main()

