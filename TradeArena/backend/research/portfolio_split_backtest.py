"""
Portfolio Split Backtest
=========================
Tests combining S2 (multi-week pre-earnings) with O1 (overnight pre-earnings)
at various capital splits, plus S3 (S2 + 10% options) as reference.

Models tested:
  S2  pure       100% in S2 multi-week holds
  S3  pure       90% stock + 10% ITM calls (same as combined_strategy.md)
  S2+O1  90/10   90% of capital in S2 sleeve, 10% in O1 overnight
  S2+O1  80/20
  S2+O1  75/25
  S2+O1  50/50

Also: Dynamic model — O1 runs on the IDLE capital when S2 is in cash.
      When S2 holds a position: full S2 allocation deployed.
      When S2 is in cash: that idle portion runs O1 overnight trades.

Both sleeves compound independently on their allocated capital.
Final portfolio = S2_sleeve_value + O1_sleeve_value.

Usage:
    uv run python -m backend.research.portfolio_split_backtest
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
# Price / helpers cache
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
# Black-Scholes for S3
# ---------------------------------------------------------------------------

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def _opt_roi(ticker, entry_dt, exit_dt, S_entry, S_exit):
    df   = _px(ticker)
    col  = df["Close"].loc[df.index < entry_dt].tail(30)
    iv   = float(col.pct_change().dropna().std() * math.sqrt(252)) if len(col) >= 10 else 0.35
    K    = S_entry * 0.90
    hold = len([d for d in df.index if entry_dt <= d <= exit_dt])
    c0   = bs_call(S_entry, K, EXPIRY_DAYS / 365, RISK_FREE, iv) * 0.75
    c1   = bs_call(S_exit,  K, max((EXPIRY_DAYS - hold * 1.4) / 365, 0), RISK_FREE, iv) * 0.75
    return (c1 - c0) / c0 * 100 if c0 > 0.01 else 0.0

# ---------------------------------------------------------------------------
# Build S2 trade list (multi-week, F6+L8)
# ---------------------------------------------------------------------------

from backend.research.signal_backtest import fetch_earnings_dates

def build_s2_trades(qqq_df: pd.DataFrame) -> list[dict]:
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
                "entry_px": ep,     "exit_px": xp,
                "ret_stock":round(ret_s, 3),
                "ret_blend_s3": round(0.90 * ret_s + 0.10 * ret_o, 3),
                "score":    round(sc, 3),
            })
    raw.sort(key=lambda t: t["entry_dt"])

    # Sequential — pick best score per date, respect exit dates
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
# Build O1 trade list (overnight pre-earnings, regime only)
# ---------------------------------------------------------------------------

def build_o1_trades(qqq_df: pd.DataFrame) -> list[dict]:
    rows = []
    for ticker in UNIVERSE:
        df     = _px(ticker)
        edates = fetch_earnings_dates(ticker)
        for ann in edates:
            d20 = _nth(df, ann, -20)
            d1  = _nth(df, ann, -1)
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
                o_t = float(df["Open"].loc[next_day]) if "Open" in df.columns else float(df["Close"].loc[next_day])
                if c == 0: continue
                overnight_ret = (o_t - c) / c * 100
                days_until    = len(window_days) - 1 - i
                rows.append({
                    "ticker":        ticker,
                    "date":          day,
                    "days_until":    days_until,
                    "overnight_ret": round(overnight_ret, 3),
                    "score":         round(score, 3),
                })

    # Pick best score per date
    by_date: dict = {}
    for t in rows:
        d = t["date"]
        if d not in by_date or t["score"] > by_date[d]["score"]:
            by_date[d] = t
    return sorted(by_date.values(), key=lambda x: x["date"])


# ---------------------------------------------------------------------------
# Simulate a single sleeve
# ---------------------------------------------------------------------------

def sim_sleeve(trades: list[dict], ret_key: str,
               start: float) -> tuple[float, float, list]:
    """Returns (final_equity, max_dd, log)."""
    equity = start
    peak   = start
    max_dd = 0.0
    log    = []
    for t in trades:
        ret = t[ret_key]
        equity *= (1 + ret / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        log.append({**t, "equity": round(equity, 2)})
    return round(equity, 2), round(max_dd, 1), log


# ---------------------------------------------------------------------------
# Combined portfolio simulation
# ---------------------------------------------------------------------------

def simulate_combined(
    s2_trades: list[dict],
    o1_trades: list[dict],
    s2_pct: float,       # fraction of START_CASH allocated to S2
    o1_pct: float,       # fraction of START_CASH allocated to O1
    label: str,
    s3_blend: bool = False,   # if True, S2 sleeve uses ret_blend_s3
) -> dict:
    """
    Fixed-split model: S2 sleeve and O1 sleeve start with their
    allocated capital and compound independently.
    """
    s2_start = START_CASH * s2_pct
    o1_start = START_CASH * o1_pct

    s2_key = "ret_blend_s3" if s3_blend else "ret_stock"
    s2_fin, s2_dd, s2_log = sim_sleeve(s2_trades, s2_key, s2_start)
    o1_fin, o1_dd, o1_log = sim_sleeve(o1_trades, "overnight_ret", o1_start)

    # Combined equity curve (merge logs by date for DD calc)
    all_events = []
    for t in s2_log:
        all_events.append(("s2", t["entry_dt"] if "entry_dt" in t else t["date"],
                           t["equity"], 0.0))
    for t in o1_log:
        all_events.append(("o1", t["date"], 0.0, t["equity"]))

    # Just use final values + rough DD estimate
    final     = round(s2_fin + o1_fin, 2)
    # Combined DD approximation: worst combined drawdown
    # Compute properly via combined equity timeline
    combined_dd = _combined_dd(s2_log, o1_log, s2_start, o1_start)

    years = 11
    ann   = (final / START_CASH) ** (1 / years) - 1

    # Milestones from combined equity
    milestones = _combined_milestones(s2_log, o1_log, s2_start, o1_start)

    n_s2 = len(s2_trades)
    n_o1 = len(o1_trades)
    return {
        "label":    label,
        "final":    final,
        "ann":      round(ann * 100, 1),
        "max_dd":   combined_dd,
        "s2_final": s2_fin,
        "o1_final": o1_fin,
        "n_s2":     n_s2,
        "n_o1":     n_o1,
        "milestones": milestones,
    }


def _combined_dd(s2_log, o1_log, s2_start, o1_start) -> float:
    """Compute true combined max drawdown from merged equity curves."""
    # Build date → combined equity snapshots
    events: dict[pd.Timestamp, tuple[float, float]] = {}

    s2_eq = s2_start
    for t in s2_log:
        d = t.get("entry_dt", t.get("date"))
        events[d] = (t["equity"], events.get(d, (s2_eq, o1_start))[1])
        s2_eq = t["equity"]

    o1_eq = o1_start
    for t in o1_log:
        d = t["date"]
        cur_s2 = events.get(d, (s2_start, o1_eq))[0]
        events[d] = (cur_s2, t["equity"])
        o1_eq = t["equity"]

    # Walk through combined equity
    combined = sorted((d, s + o) for d, (s, o) in events.items())
    if not combined: return 0.0

    peak   = START_CASH
    max_dd = 0.0
    for _, eq in combined:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd
    return round(max_dd, 1)


def _combined_milestones(s2_log, o1_log, s2_start, o1_start) -> dict:
    events: dict[pd.Timestamp, tuple[float, float]] = {}
    s2_eq = s2_start
    for t in s2_log:
        d = t.get("entry_dt", t.get("date"))
        events[d] = (t["equity"], events.get(d, (s2_eq, o1_start))[1])
        s2_eq = t["equity"]
    o1_eq = o1_start
    for t in o1_log:
        d = t["date"]
        cur_s2 = events.get(d, (s2_start, o1_eq))[0]
        events[d] = (cur_s2, t["equity"])
        o1_eq = t["equity"]

    milestones = {}
    for d, (s, o) in sorted(events.items()):
        eq = s + o
        for m in MILESTONES:
            if m not in milestones and eq >= m:
                milestones[m] = d.date()
    return milestones


# ---------------------------------------------------------------------------
# Dynamic model — O1 runs on idle S2 capital
# ---------------------------------------------------------------------------

def simulate_dynamic(s2_trades: list[dict], o1_trades: list[dict],
                     label: str) -> dict:
    """
    When S2 is in a position: full account in S2.
    When S2 is in cash: run O1 overnight trades on full account.
    No fixed split — capital flows between strategies.
    """
    # Build S2 busy periods
    s2_periods = [(t["entry_dt"], t["exit_dt"]) for t in s2_trades]

    def in_s2(d: pd.Timestamp) -> bool:
        return any(entry <= d <= exit for entry, exit in s2_periods)

    # Merge all events sorted by date
    s2_by_dt = {t["entry_dt"]: t for t in s2_trades}
    o1_by_dt = {t["date"]: t for t in o1_trades}

    all_dates = sorted(set(list(s2_by_dt.keys()) + list(o1_by_dt.keys())))

    equity = START_CASH
    peak   = START_CASH
    max_dd = 0.0
    wins   = 0
    rets   = []
    milestones: dict = {}
    log    = []
    s2_active_exit = None

    # Track which S2 trade is active
    s2_idx  = 0
    s2_done = set()

    for dt in all_dates:
        # Check S2 entry
        if dt in s2_by_dt and dt not in s2_done:
            t = s2_by_dt[dt]
            # S2 trade — runs from entry_dt to exit_dt
            # Apply S2 return at exit date (we process as a lump)
            ret = t["ret_stock"]
            for m in MILESTONES:
                if m not in milestones and equity >= m:
                    milestones[m] = dt.date()
            equity *= (1 + ret / 100)
            if equity > peak: peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd
            if ret > 0: wins += 1
            rets.append(ret)
            s2_done.add(dt)
            log.append({"date": dt, "type": "S2", "ret": ret, "equity": round(equity, 2)})
            continue

        # Check O1 (only if not in S2 window)
        if dt in o1_by_dt and not in_s2(dt):
            t = o1_by_dt[dt]
            ret = t["overnight_ret"]
            for m in MILESTONES:
                if m not in milestones and equity >= m:
                    milestones[m] = dt.date()
            equity *= (1 + ret / 100)
            if equity > peak: peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd
            if ret > 0: wins += 1
            rets.append(ret)
            log.append({"date": dt, "type": "O1", "ret": ret, "equity": round(equity, 2)})

    n   = len(rets)
    ann = (equity / START_CASH) ** (1 / 11) - 1
    wr  = wins / n * 100 if n else 0
    avg = sum(rets) / n if n else 0

    n_s2 = sum(1 for t in log if t["type"] == "S2")
    n_o1 = sum(1 for t in log if t["type"] == "O1")
    return {
        "label":    label,
        "final":    round(equity, 2),
        "ann":      round(ann * 100, 1),
        "max_dd":   round(max_dd, 1),
        "wr":       round(wr, 1),
        "avg":      round(avg, 3),
        "n_s2":     n_s2,
        "n_o1":     n_o1,
        "n_total":  n,
        "milestones": milestones,
        "log":      log,
    }


# ---------------------------------------------------------------------------
# Year-by-year from log
# ---------------------------------------------------------------------------

def year_by_year(log: list[dict]) -> None:
    by_yr: dict[int, list] = {}
    for t in log:
        yr = t["date"].year if hasattr(t["date"], "year") else pd.Timestamp(t["date"]).year
        by_yr.setdefault(yr, []).append(t["ret"])
    cum = START_CASH
    for yr in sorted(by_yr):
        rets = by_yr[yr]
        avg  = sum(rets) / len(rets)
        w    = sum(1 for r in rets if r > 0)
        for r in rets: cum *= (1 + r / 100)
        bar  = ("+" if avg >= 0 else "-") + "█" * min(int(abs(avg) / 0.20), 25)
        print(f"    {yr}  {w:>3}/{len(rets):>3}  avg {avg:>+6.3f}%  ${cum:>9,.0f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  PORTFOLIO SPLIT BACKTEST — S2, S3, S2+O1 splits")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ"]:
        _px(t); print(f"  {t} ready")
    qqq_df = _px("QQQ")

    print("\nBuilding S2 trade list (F6 + score>=1.20)...")
    s2_trades = build_s2_trades(qqq_df)
    print(f"  {len(s2_trades)} S2 trades")

    print("Building O1 trade list (overnight pre-earnings, regime only)...")
    o1_trades = build_o1_trades(qqq_df)
    print(f"  {len(o1_trades)} O1 overnight observations")

    # ── Pure S2 and S3 baselines ──────────────────────────────────────────
    print("\nRunning pure S2 / S3 simulations...")
    s2_fin, s2_dd, s2_log = sim_sleeve(s2_trades, "ret_stock",    START_CASH)
    s3_fin, s3_dd, s3_log = sim_sleeve(s2_trades, "ret_blend_s3", START_CASH)
    o1_fin, o1_dd, o1_log = sim_sleeve(o1_trades, "overnight_ret", START_CASH)

    s2_ann = (s2_fin / START_CASH) ** (1/11) - 1
    s3_ann = (s3_fin / START_CASH) ** (1/11) - 1
    o1_ann = (o1_fin / START_CASH) ** (1/11) - 1

    def s2_milestones():
        ms = {}
        eq = START_CASH
        for t in s2_trades:
            for m in MILESTONES:
                if m not in ms and eq >= m: ms[m] = t["entry_dt"].date()
            eq *= (1 + t["ret_stock"] / 100)
        return ms

    # ── Split combinations ────────────────────────────────────────────────
    splits = [
        (1.00, 0.00),   # S2 pure
        (0.90, 0.10),
        (0.80, 0.20),
        (0.75, 0.25),
        (0.50, 0.50),
        (0.00, 1.00),   # O1 pure
    ]

    print("\nRunning split simulations...")
    split_results = []
    for s2p, o1p in splits:
        label = (f"S2 {int(s2p*100)}% + O1 {int(o1p*100)}%"
                 if 0 < s2p < 1 else
                 ("S2 100% (pure)" if s2p == 1 else "O1 100% (pure)"))
        r = simulate_combined(s2_trades, o1_trades, s2p, o1p, label)
        split_results.append(r)

    # S3 pure
    r_s3 = simulate_combined(s2_trades, [], 1.00, 0.00, "S3 100% (90%stk+10%calls)", s3_blend=True)
    r_s3["o1_final"] = 0
    r_s3["n_o1"] = 0

    # S3 + O1 splits
    s3_splits = [
        ("S3 80% + O1 20%",  0.80, 0.20),
        ("S3 75% + O1 25%",  0.75, 0.25),
        ("S3 50% + O1 50%",  0.50, 0.50),
    ]
    s3_split_results = []
    for label, s3p, o1p in s3_splits:
        r = simulate_combined(s2_trades, o1_trades, s3p, o1p, label, s3_blend=True)
        s3_split_results.append(r)

    # Dynamic model
    r_dyn = simulate_dynamic(s2_trades, o1_trades,
                             "Dynamic: S2 active + O1 on idle capital")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RESULTS — $2,000 starting capital, 2015–2026")
    print(SEP)
    print(f"  {'Strategy':<40} {'Ann%':>6}  {'MaxDD':>6}  {'Final':>10}  {'$10k':>12}  {'$20k':>12}")
    print(f"  {'-'*40} {'-'*6}  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*12}")

    def prow(label, ann, dd, final, ms):
        m10 = str(ms.get(10_000, "—"))[:10]
        m20 = str(ms.get(20_000, "—"))[:10]
        flag = "  ◄" if ann > 24.3 and dd < 15 else ""
        print(f"  {label:<40} {ann:>+5.1f}%  {dd:>5.1f}%  ${final:>9,.0f}  {m10:>12}  {m20:>12}{flag}")

    # Pure strategies
    s2_ms = s2_milestones()
    prow("S2 pure (F6+L8, multi-week)",   round(s2_ann*100,1), s2_dd, s2_fin, s2_ms)
    prow("S3 pure (90%stk+10%calls)",     round(s3_ann*100,1), s3_dd, s3_fin, {})
    print()

    # S2+O1 splits
    for r in split_results[1:-1]:   # skip pure S2 and pure O1
        ms = r.get("milestones", {})
        prow(r["label"], r["ann"], r["max_dd"], r["final"], ms)
    print()

    # S3+O1 splits
    for r in s3_split_results:
        ms = r.get("milestones", {})
        prow(r["label"], r["ann"], r["max_dd"], r["final"], ms)
    print()

    # Dynamic
    dyn_ms = r_dyn.get("milestones", {})
    prow(r_dyn["label"], r_dyn["ann"], r_dyn["max_dd"], r_dyn["final"], dyn_ms)

    # ── Dynamic year-by-year ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  YEAR-BY-YEAR — Dynamic model (S2 active + O1 on idle capital)")
    print(SEP)
    year_by_year(r_dyn["log"])
    print(f"\n  S2 trades: {r_dyn['n_s2']}  O1 trades: {r_dyn['n_o1']}  "
          f"Total: {r_dyn['n_total']}")

    # ── S3 year-by-year ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  YEAR-BY-YEAR — S3 pure (90% stock + 10% calls)")
    print(SEP)
    year_by_year(s3_log)

    # ── Best split year-by-year ────────────────────────────────────────────
    all_results = split_results + s3_split_results
    best_split = max(all_results,
                     key=lambda r: r["ann"] / max(r["max_dd"], 1))
    print(f"\n{SEP}")
    print(f"  BEST RISK-ADJUSTED SPLIT: {best_split['label']}")
    print(f"  ann {best_split['ann']:+.1f}%  dd {best_split['max_dd']:.1f}%  "
          f"final ${best_split['final']:,.0f}")
    print(SEP)

    # ── Contribution breakdown ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SLEEVE CONTRIBUTION — what each part adds")
    print(SEP)
    print(f"  {'Strategy':<40} {'S2 final':>10}  {'O1 final':>10}  {'Combined':>10}")
    print(f"  {'-'*40} {'-'*10}  {'-'*10}  {'-'*10}")
    for r in split_results + s3_split_results:
        print(f"  {r['label']:<40} ${r['s2_final']:>9,.0f}  ${r['o1_final']:>9,.0f}  ${r['final']:>9,.0f}")

    # ── Final verdict ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VERDICT")
    print(SEP)
    print(f"  S2 pure:  ann {s2_ann*100:+.1f}%  dd {s2_dd:.1f}%  — reference")
    print(f"  S3 pure:  ann {s3_ann*100:+.1f}%  dd {s3_dd:.1f}%  — options add return but need real data")
    print(f"  Dynamic:  ann {r_dyn['ann']:+.1f}%  dd {r_dyn['max_dd']:.1f}%  — O1 fills S2 idle time")
    print()
    for r in split_results[1:-1]:
        vs_s2 = r["ann"] - round(s2_ann*100, 1)
        vs_dd = r["max_dd"] - s2_dd
        verdict = "BETTER" if vs_s2 > 0 and vs_dd <= 5 else "WORSE" if vs_s2 < 0 else "MIXED"
        print(f"  {r['label']:<38}  Δann {vs_s2:>+4.1f}%  Δdd {vs_dd:>+4.1f}%  → {verdict}")


if __name__ == "__main__":
    main()

