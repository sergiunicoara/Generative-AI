"""
PEAD — Post-Earnings Announcement Drift Backtest
==================================================
Academic finding (first documented 1968, still profitable):
  After a positive earnings surprise (stock gaps up), it continues to drift
  higher for ~20 trading days. The market systematically under-reacts to
  earnings news, creating a predictable multi-week drift.

Entry/Exit logic:
  Entry  : open of first trading day AFTER earnings announcement
  Signal : gap% = (open_after / close_before_earnings) - 1
           captures full earnings reaction (AH or pre-market reporters)
  Exit   : close of 20th trading day after entry
  Stop   : -8% from entry open (tighter than S2's -5%, earnings gaps can reverse)

Tested combinations:
  PEAD alone — gap thresholds 0%, 3%, 5%, 8%
  PEAD + hold period sweep (10, 15, 20, 30 days)
  PEAD vs S2 side-by-side
  S2 + PEAD dynamic (PEAD only when S2 is idle — same capital pool)
  S2 + MG + PEAD triple dynamic

Universe: GOOGL, NVDA, AMZN, MSFT, META, AMD (same as S2)
Period  : 2015–2026, $2,000 start

Usage:
    uv run python -m backend.research.pead_backtest
"""

from __future__ import annotations

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP        = "=" * 72
START_CASH = 2_000.0
UNIVERSE   = ["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
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


def _nth_trading_day(df: pd.DataFrame, ref: pd.Timestamp, offset: int) -> pd.Timestamp | None:
    """Return the Nth trading day after (+) or before (-) ref date."""
    idx = df.index.tolist()
    d   = 1 if offset >= 0 else -1
    c   = 0
    for i in range(1, 300):
        cand = ref + pd.Timedelta(days=i * d)
        if cand in df.index:
            c += 1
            if c == abs(offset):
                return cand
    return None


# ---------------------------------------------------------------------------
# Build PEAD signals
# ---------------------------------------------------------------------------

def build_pead_signals(qqq_df: pd.DataFrame,
                       min_gap: float = 3.0,
                       hold_days: int = 20,
                       stop_pct: float = -8.0) -> list[dict]:
    """
    For each earnings date per ticker:
      1. Measure gap = (open_after_earnings / close_before_earnings - 1) * 100
      2. If gap >= min_gap AND regime ok → PEAD entry
      3. Simulate: buy open_after, hold hold_days trading days, sell close
      4. Stop at -8% intraday if any day's close < entry * (1 + stop_pct/100)

    Returns list of trade dicts.
    """
    from backend.research.signal_backtest import fetch_earnings_dates

    def _regime_ok(dt):
        qc = qqq_df["Close"].loc[qqq_df.index <= dt]
        return len(qc) >= 150 and float(qc.iloc[-1]) > float(qc.rolling(150).mean().iloc[-1])

    trades = []
    for ticker in UNIVERSE:
        df    = _px(ticker)
        dates = fetch_earnings_dates(ticker)

        for ann in dates:
            ann_ts = pd.Timestamp(ann)
            if ann_ts.year < 2015: continue

            # Day before earnings (pre-earnings reference close)
            d_before = _nth_trading_day(df, ann_ts, -1)
            # First trading day after earnings (entry)
            d_after  = _nth_trading_day(df, ann_ts,  1)
            if d_before is None or d_after is None: continue
            if d_before not in df.index or d_after not in df.index: continue

            # Regime check at entry date
            if not _regime_ok(d_after): continue

            # Entry price = open after earnings
            if "Open" not in df.columns: continue
            entry_open  = float(df["Open"].loc[d_after])
            close_before = float(df["Close"].loc[d_before])
            if close_before <= 0 or entry_open <= 0: continue

            # Gap measurement
            gap_pct = (entry_open / close_before - 1) * 100
            if gap_pct < min_gap: continue

            # Simulate hold: check each day for stop, exit at close of hold_days
            idx_list = [d for d in df.index if d >= d_after]
            if len(idx_list) < hold_days: continue

            stop_price = entry_open * (1 + stop_pct / 100)
            exit_price = None
            stopped    = False
            exit_date  = None

            for i, day in enumerate(idx_list[:hold_days]):
                day_close = float(df["Close"].loc[day])
                day_low   = float(df["Low"].loc[day]) if "Low" in df.columns else day_close
                if day_low <= stop_price:
                    exit_price = stop_price
                    stopped    = True
                    exit_date  = day
                    break
                if i == hold_days - 1:
                    exit_price = day_close
                    exit_date  = day

            if exit_price is None: continue

            ret = (exit_price - entry_open) / entry_open * 100

            trades.append({
                "ticker":     ticker,
                "ann":        ann_ts,
                "entry_date": d_after,
                "exit_date":  exit_date,
                "entry_px":   round(entry_open, 2),
                "exit_px":    round(exit_price, 2),
                "gap_pct":    round(gap_pct, 2),
                "ret":        round(ret, 3),
                "stopped":    stopped,
                "strategy":   "PEAD",
            })

    trades.sort(key=lambda t: t["entry_date"])
    return trades


# ---------------------------------------------------------------------------
# Build S2 trades (reused from s2_mg_dynamic_backtest)
# ---------------------------------------------------------------------------

def build_s2_trades(qqq_df: pd.DataFrame) -> list[dict]:
    from backend.research.signal_backtest import fetch_earnings_dates

    BASE_QUALITY = {"GOOGL":1.40,"NVDA":1.50,"AMZN":1.20,"MSFT":1.10,"META":1.10,"AMD":1.00}
    SCORE_THRESH = 1.20

    def _nth(df, ref, offset):
        dt = pd.Timestamp(ref); d = 1 if offset >= 0 else -1; c = 0
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
            raw.append({
                "ticker": ticker, "entry_date": e_dt, "exit_date": x_dt,
                "ret": round(ret, 3), "score": round(sc, 3),
                "ann": ann, "strategy": "S2",
            })

    raw.sort(key=lambda t: t["entry_date"])
    by_date: dict = {}
    for t in raw:
        d = t["entry_date"]
        if d not in by_date or t["score"] > by_date[d]["score"]:
            by_date[d] = t

    taken, busy = [], None
    for t in sorted(by_date.values(), key=lambda x: x["entry_date"]):
        if busy and t["entry_date"] <= busy: continue
        taken.append(t)
        busy = t["exit_date"]
    return taken


# ---------------------------------------------------------------------------
# Build MG signals (reused)
# ---------------------------------------------------------------------------

def build_mg_signals(qqq_df: pd.DataFrame, min_gap: float = 1.00) -> list[dict]:
    rows = []
    MA50_WINDOW = 50; REGIME_MA = 150
    qqq_close = qqq_df["Close"]
    for ticker in UNIVERSE:
        df  = _px(ticker)
        idx = df.index.tolist()
        for i in range(max(MA50_WINDOW, REGIME_MA), len(idx)):
            today = idx[i]; prev = idx[i-1]
            if today.year < 2015: continue
            if today.day_name() != "Monday": continue
            o  = float(df["Open"].loc[today]) if "Open" in df.columns else None
            c  = float(df["Close"].loc[today])
            pc = float(df["Close"].loc[prev])
            if o is None or pc == 0: continue
            gap_pct = (o - pc) / pc * 100
            if gap_pct < min_gap: continue
            intra_ret = (c - o) / o * 100
            close_ser = df["Close"].loc[df.index <= today]
            ma50 = float(close_ser.tail(MA50_WINDOW).mean())
            if o <= ma50: continue
            qc = qqq_close.loc[qqq_close.index <= today]
            if not (len(qc) >= REGIME_MA and
                    float(qc.iloc[-1]) > float(qc.rolling(REGIME_MA).mean().iloc[-1])): continue
            vol_ratio = 1.0
            if "Volume" in df.columns:
                v = float(df["Volume"].loc[today])
                va = float(df["Volume"].loc[df.index <= today].tail(21).iloc[:-1].mean())
                vol_ratio = v / va if va > 0 else 1.0
            rows.append({
                "ticker": ticker, "entry_date": today, "exit_date": today,
                "ret": round(intra_ret, 3), "gap_pct": round(gap_pct, 3),
                "score": round(gap_pct * vol_ratio, 4), "strategy": "MG",
            })
    rows.sort(key=lambda r: r["entry_date"])
    by_date: dict = {}
    for r in rows:
        d = r["entry_date"]
        if d not in by_date or r["score"] > by_date[d]["score"]:
            by_date[d] = r
    return sorted(by_date.values(), key=lambda r: r["entry_date"])


# ---------------------------------------------------------------------------
# Dynamic combiner — strategy list in priority order
# ---------------------------------------------------------------------------

def combine_dynamic(strategy_lists: list[list[dict]],
                    priorities: list[str]) -> list[dict]:
    """
    Merge multiple strategy trade lists into one sequential account.
    Priority order = strategies listed first win on date conflicts.
    Later strategies only fire when no earlier strategy holds an open position.
    """
    # Build busy date ranges from higher-priority strategies progressively
    combined: list[dict] = []
    busy_dates: set = set()

    # Process in priority order
    all_trades: list[dict] = []
    for trades in strategy_lists:
        all_trades.extend(trades)
    all_trades.sort(key=lambda t: (t["entry_date"], priorities.index(t["strategy"])
                                   if t["strategy"] in priorities else 99))

    busy_ranges: list[tuple] = []   # (entry_date, exit_date) of taken trades

    def is_busy(entry: pd.Timestamp, exit_: pd.Timestamp) -> bool:
        for (be, bx) in busy_ranges:
            # overlap if entry <= bx and exit_ >= be
            if entry <= bx and exit_ >= be:
                return True
        return False

    # Group by date, pick highest-priority per date
    by_date: dict = {}
    for t in all_trades:
        d = t["entry_date"]
        if d not in by_date:
            by_date[d] = t
        else:
            existing_pri = priorities.index(by_date[d]["strategy"]) if by_date[d]["strategy"] in priorities else 99
            new_pri      = priorities.index(t["strategy"]) if t["strategy"] in priorities else 99
            if new_pri < existing_pri:
                by_date[d] = t

    for t in sorted(by_date.values(), key=lambda x: x["entry_date"]):
        if is_busy(t["entry_date"], t["exit_date"]):
            continue
        combined.append(t)
        busy_ranges.append((t["entry_date"], t["exit_date"]))

    return combined


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(trades: list[dict], label: str) -> dict:
    equity = START_CASH; peak = START_CASH; max_dd = 0.0
    wins = 0; rets = []; milestones: dict = {}; log = []

    for t in trades:
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = t["entry_date"].date()
        ret = t["ret"]
        equity *= (1 + ret / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if ret > 0: wins += 1
        rets.append(ret)
        log.append({
            "ticker": t["ticker"], "date": t["entry_date"],
            "strategy": t.get("strategy", "?"), "ret": ret,
            "equity": round(equity, 2),
        })

    n = len(rets); years = 11
    ann = (equity / START_CASH) ** (1 / years) - 1
    return {
        "label": label, "n": n, "wins": wins,
        "wr":    round(wins / n * 100, 1) if n else 0,
        "avg":   round(sum(rets) / n, 2) if n else 0,
        "final": round(equity, 2), "ann": round(ann * 100, 1),
        "max_dd": round(max_dd, 1), "milestones": milestones, "log": log,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def prow(r: dict, base: dict) -> None:
    m10  = str(r["milestones"].get(10_000, "—"))[:10]
    m20  = str(r["milestones"].get(20_000, "—"))[:10]
    d_ann = r["ann"] - base["ann"]
    d_dd  = r["max_dd"] - base["max_dd"]
    flag  = "  ◄ BETTER"       if d_ann >= 0 and d_dd <= 0 else \
            "  ◄ SWEET SPOT"   if d_ann >= 2 and d_dd <= 3  else \
            "  ↑ good tradeoff" if d_ann >= 2 and d_dd <= 5  else \
            "  ↑ more return"  if d_ann > 1  else \
            "  ↓ less return"  if d_ann < -1 else ""
    print(f"  {r['label']:<44} {r['n']:>4}  {r['wr']:>5.1f}%  {r['avg']:>+6.2f}%  "
          f"{r['ann']:>+5.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}  "
          f"{m10}  {m20}{flag}")


def year_by_year(r: dict) -> None:
    by_strat_yr: dict[str, dict[int, list]] = {}
    cum = START_CASH
    for t in r["log"]:
        s = t["strategy"]; yr = t["date"].year
        by_strat_yr.setdefault(s, {}).setdefault(yr, []).append(t["ret"])
    all_yrs = sorted({t["date"].year for t in r["log"]})
    for yr in all_yrs:
        all_t = [t["ret"] for t in r["log"] if t["date"].year == yr]
        if not all_t: continue
        avg = sum(all_t) / len(all_t)
        for v in all_t: cum *= (1 + v / 100)
        parts = []
        for s in ["S2", "MG", "PEAD"]:
            n = len(by_strat_yr.get(s, {}).get(yr, []))
            if n: parts.append(f"{s}:{n}")
        bar = ("+" if avg >= 0 else "-") + "█" * min(int(abs(avg) / 2), 20)
        print(f"    {yr}  {' '.join(parts):<18}  avg {avg:>+6.2f}%  ${cum:>9,.0f}  {bar}")


def ticker_breakdown(trades: list[dict], label: str) -> None:
    by_t: dict[str, list] = {}
    for t in trades:
        by_t.setdefault(t["ticker"], []).append(t)
    print(f"\n  {label}")
    print(f"  {'Ticker':<8} {'N':>4}  {'Win%':>5}  {'Avg%':>7}  "
          f"{'AvgGap%':>8}  {'Stopped':>8}  {'Best':>8}  {'Worst':>8}")
    print(f"  {'-'*8} {'-'*4}  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for tk in sorted(by_t, key=lambda x: -len(by_t[x])):
        ts  = by_t[tk]
        wr  = sum(1 for t in ts if t["ret"] > 0) / len(ts) * 100
        avg = sum(t["ret"] for t in ts) / len(ts)
        ag  = sum(t["gap_pct"] for t in ts) / len(ts)
        st  = sum(1 for t in ts if t.get("stopped"))
        best  = max(t["ret"] for t in ts)
        worst = min(t["ret"] for t in ts)
        print(f"  {tk:<8} {len(ts):>4}  {wr:>4.0f}%  {avg:>+6.2f}%  "
              f"{ag:>+7.1f}%  {st:>8}  {best:>+7.1f}%  {worst:>+7.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  PEAD BACKTEST — Post-Earnings Announcement Drift (2015–2026, $2,000)")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ"]:
        _px(t); print(f"  {t} ready")
    qqq_df = _px("QQQ")

    # ── PEAD gap threshold sweep ──────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PEAD PARAMETER SWEEP — gap threshold (hold=20d, stop=-8%)")
    print(SEP)
    print(f"  {'Min gap%':<10} {'N':>4}  {'Win%':>5}  {'Avg%':>7}  "
          f"{'Ann%':>5}  {'DD%':>5}  {'Final':>10}  $10k")
    print(f"  {'-'*10} {'-'*4}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}")

    sweep_results = {}
    for thresh in [0.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        trades = build_pead_signals(qqq_df, min_gap=thresh, hold_days=20)
        r = simulate(trades, f"PEAD gap≥{thresh:.0f}%")
        sweep_results[thresh] = (trades, r)
        m10 = str(r["milestones"].get(10_000, "—"))[:10]
        flag = "  ◄" if r["ann"] > 15 and r["max_dd"] < 20 else ""
        print(f"  {thresh:>8.0f}%   {r['n']:>4}  {r['wr']:>4.0f}%  {r['avg']:>+6.2f}%  "
              f"{r['ann']:>+4.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}  {m10}{flag}")

    # ── Hold period sweep (best gap threshold) ────────────────────────────
    print(f"\n{SEP}")
    print("  PEAD HOLD PERIOD SWEEP — gap≥3%, stop=-8%")
    print(SEP)
    print(f"  {'Hold days':<12} {'N':>4}  {'Win%':>5}  {'Avg%':>7}  "
          f"{'Ann%':>5}  {'DD%':>5}  {'Final':>10}")
    print(f"  {'-'*12} {'-'*4}  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*10}")
    best_hold_r = None
    for hold in [5, 10, 15, 20, 30]:
        trades = build_pead_signals(qqq_df, min_gap=3.0, hold_days=hold)
        r = simulate(trades, f"PEAD {hold}d hold")
        flag = "  ◄" if r["ann"] > 15 and r["max_dd"] < 20 else ""
        if best_hold_r is None or r["ann"] > best_hold_r["ann"]:
            best_hold_r = r
            best_hold_trades = trades
            best_hold = hold
        print(f"  {hold:>10}d   {r['n']:>4}  {r['wr']:>4.0f}%  {r['avg']:>+6.2f}%  "
              f"{r['ann']:>+4.1f}%  {r['max_dd']:>5.1f}%  ${r['final']:>9,.0f}{flag}")

    # ── Best PEAD config ──────────────────────────────────────────────────
    # Pick best gap threshold by ann%/dd ratio
    best_thresh = max(sweep_results.items(),
                      key=lambda x: x[1][1]["ann"] / max(x[1][1]["max_dd"], 1))
    best_gap       = best_thresh[0]
    best_pead_all  = best_thresh[1][0]
    best_pead_r    = best_thresh[1][1]

    print(f"\n  → Best gap threshold: ≥{best_gap:.0f}%  ({best_pead_r['n']} trades, "
          f"ann {best_pead_r['ann']:>+.1f}%, dd {best_pead_r['max_dd']:.1f}%)")

    # Use gap≥3% as the deployment candidate (balance of n and quality)
    pead_3_trades, _ = sweep_results[3.0]
    pead_3_r         = simulate(pead_3_trades, "PEAD gap≥3% (20d hold)")

    # ── Per-ticker breakdown ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PEAD PER-TICKER BREAKDOWN (gap≥3%, 20d hold)")
    print(SEP)
    ticker_breakdown(pead_3_trades, "All PEAD gap≥3% trades")

    # ── Relationship: gap size vs drift magnitude ─────────────────────────
    print(f"\n{SEP}")
    print("  GAP SIZE vs SUBSEQUENT DRIFT (gap≥3%, 20d hold)")
    print(SEP)
    print(f"  {'Gap bucket':<18} {'N':>4}  {'Win%':>5}  {'Avg drift%':>11}  Best")
    print(f"  {'-'*18} {'-'*4}  {'-'*5}  {'-'*11}")
    buckets = [
        ("3–5%",   lambda t: 3.0 <= t["gap_pct"] < 5.0),
        ("5–8%",   lambda t: 5.0 <= t["gap_pct"] < 8.0),
        ("8–12%",  lambda t: 8.0 <= t["gap_pct"] < 12.0),
        ("12–20%", lambda t: 12.0 <= t["gap_pct"] < 20.0),
        (">20%",   lambda t: t["gap_pct"] >= 20.0),
    ]
    for label, fn in buckets:
        sub = [t for t in pead_3_trades if fn(t)]
        if not sub: continue
        wr  = sum(1 for t in sub if t["ret"] > 0) / len(sub) * 100
        avg = sum(t["ret"] for t in sub) / len(sub)
        top = max(sub, key=lambda t: t["ret"])
        flag = "  ◄ strong" if avg > 5 and wr > 60 else ""
        print(f"  {label:<18} {len(sub):>4}  {wr:>4.0f}%  {avg:>+10.2f}%  "
              f"{top['ticker']}+{top['ret']:.0f}%{flag}")

    # ── S2 reference ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  BUILDING S2 + MG FOR COMPARISON")
    print(SEP)
    s2_trades = build_s2_trades(qqq_df)
    mg_trades = build_mg_signals(qqq_df, min_gap=1.00)
    print(f"  S2: {len(s2_trades)} trades  |  MG: {len(mg_trades)} signals  |  "
          f"PEAD(≥3%): {len(pead_3_trades)} signals")

    # ── Dynamic combinations ──────────────────────────────────────────────
    s2_r           = simulate(s2_trades, "S2 alone")
    pead_3_r       = simulate(pead_3_trades, "PEAD alone (gap≥3%)")

    # S2 + PEAD (S2 priority)
    s2_pead        = combine_dynamic([s2_trades, pead_3_trades], ["S2", "PEAD"])
    s2_pead_r      = simulate(s2_pead, "S2 + PEAD dynamic")

    # S2 + MG (MG fills idle when S2 not running)
    s2_mg          = combine_dynamic([s2_trades, mg_trades], ["S2", "MG"])
    s2_mg_r        = simulate(s2_mg, "S2 + MG dynamic")

    # S2 + MG + PEAD (S2 → PEAD → MG priority)
    s2_mg_pead     = combine_dynamic([s2_trades, pead_3_trades, mg_trades], ["S2", "PEAD", "MG"])
    s2_mg_pead_r   = simulate(s2_mg_pead, "S2 + PEAD + MG dynamic")

    # S2 + PEAD higher gap threshold (≥5%) — fewer but higher quality
    pead_5_trades, _ = sweep_results[5.0]
    s2_pead_5      = combine_dynamic([s2_trades, pead_5_trades], ["S2", "PEAD"])
    s2_pead_5_r    = simulate(s2_pead_5, "S2 + PEAD dynamic (gap≥5%)")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  STRATEGY COMPARISON (2015–2026, $2,000 start)")
    print(SEP)
    print(f"  {'Strategy':<44} {'N':>4}  {'Win%':>5}  {'Avg%':>6}  "
          f"{'Ann%':>5}  {'DD%':>5}  {'Final':>10}  $10k          $20k")
    print(f"  {'-'*44} {'-'*4}  {'-'*5}  {'-'*6}  "
          f"{'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")

    base = s2_r
    for r in [s2_r, pead_3_r, s2_pead_r, s2_pead_5_r, s2_mg_r, s2_mg_pead_r]:
        prow(r, base)

    # ── Trade breakdown in combined strategies ────────────────────────────
    print(f"\n{SEP}")
    print("  TRADE COMPOSITION — S2 + PEAD + MG Dynamic")
    print(SEP)
    for strat in ["S2", "PEAD", "MG"]:
        sub = [t for t in s2_mg_pead_r["log"] if t["strategy"] == strat]
        if not sub: continue
        wr  = sum(1 for t in sub if t["ret"] > 0) / len(sub) * 100
        avg = sum(t["ret"] for t in sub) / len(sub)
        print(f"  {strat:<6}  n={len(sub):>3}  win {wr:>4.0f}%  avg {avg:>+6.2f}%  "
              f"({len(sub)/11:.1f}/yr)")

    # ── Year-by-year for best combined ────────────────────────────────────
    print(f"\n{SEP}")
    print("  YEAR-BY-YEAR — S2 + PEAD + MG Dynamic")
    print(SEP)
    year_by_year(s2_mg_pead_r)

    # ── PEAD vs S2 overlap analysis ───────────────────────────────────────
    print(f"\n{SEP}")
    print("  PEAD TIMING ANALYSIS — do PEAD and S2 conflict on same ticker?")
    print(SEP)
    # Check: how many PEAD signals were blocked by S2 being active
    s2_busy: list[tuple] = [(t["entry_date"], t["exit_date"]) for t in s2_trades]
    pead_blocked = 0; pead_same_ticker = 0
    for pt in pead_3_trades:
        for (be, bx) in s2_busy:
            if pt["entry_date"] <= bx and pt["exit_date"] >= be:
                pead_blocked += 1
                break
    pead_taken = [t for t in s2_pead if t["strategy"] == "PEAD"]
    print(f"  PEAD signals total:    {len(pead_3_trades)}")
    print(f"  PEAD blocked by S2:    {pead_blocked}  ({pead_blocked/len(pead_3_trades)*100:.0f}%)")
    print(f"  PEAD actually taken:   {len(pead_taken)}  ({len(pead_taken)/11:.1f}/yr)")
    print(f"  Avg gap on taken PEAD: {sum(t['gap_pct'] for t in pead_taken)/len(pead_taken):.1f}%"
          if pead_taken else "  No PEAD taken")

    # Best and worst PEAD trades
    if pead_taken:
        best_p  = sorted(pead_taken, key=lambda t: -t["ret"])[:3]
        worst_p = sorted(pead_taken, key=lambda t:  t["ret"])[:3]
        print(f"\n  Best PEAD:  " + "  ".join(
            f"{t['ticker']}(gap+{t['gap_pct']:.0f}%→ret{t['ret']:>+.0f}%)" for t in best_p))
        print(f"  Worst PEAD: " + "  ".join(
            f"{t['ticker']}(gap+{t['gap_pct']:.0f}%→ret{t['ret']:>+.0f}%)" for t in worst_p))

    # ── Verdict ───────────────────────────────────────────────────────────
    candidates = [s2_pead_r, s2_pead_5_r, s2_mg_r, s2_mg_pead_r]
    best = max(candidates, key=lambda r: r["ann"] / max(r["max_dd"], 1))

    print(f"\n{SEP}")
    print("  VERDICT")
    print(SEP)
    print(f"  S2 alone:              ann {s2_r['ann']:>+5.1f}%  dd {s2_r['max_dd']:>5.1f}%  "
          f"final ${s2_r['final']:>9,.0f}")
    print(f"  PEAD alone (gap≥3%):   ann {pead_3_r['ann']:>+5.1f}%  dd {pead_3_r['max_dd']:>5.1f}%  "
          f"final ${pead_3_r['final']:>9,.0f}")
    print(f"  S2+MG:                 ann {s2_mg_r['ann']:>+5.1f}%  dd {s2_mg_r['max_dd']:>5.1f}%  "
          f"final ${s2_mg_r['final']:>9,.0f}")
    print(f"  S2+PEAD:               ann {s2_pead_r['ann']:>+5.1f}%  dd {s2_pead_r['max_dd']:>5.1f}%  "
          f"final ${s2_pead_r['final']:>9,.0f}")
    print(f"  S2+PEAD+MG:            ann {s2_mg_pead_r['ann']:>+5.1f}%  dd {s2_mg_pead_r['max_dd']:>5.1f}%  "
          f"final ${s2_mg_pead_r['final']:>9,.0f}")
    print(f"\n  Best risk-adjusted:    {best['label']}")

    d_ann = best["ann"] - s2_r["ann"]
    d_dd  = best["max_dd"] - s2_r["max_dd"]
    print(f"  vs S2 alone:           return {d_ann:>+.1f}%/yr  drawdown {d_dd:>+.1f}%")

    if d_ann >= 3 and d_dd <= 5:
        print(f"\n  ✓ PEAD ADDS VALUE — complementary to S2 (different timing window)")
        print(f"    Pre-earnings: S2 captures the drift INTO earnings")
        print(f"    Post-earnings: PEAD captures the drift OUT OF earnings")
        print(f"    Both edges exist because the market under-reacts at different stages")
    elif d_ann >= 2:
        print(f"\n  ~ PEAD MARGINALLY USEFUL — some improvement but check DD carefully")
    else:
        print(f"\n  ✗ PEAD does not add significant value on this universe/period")
        print(f"    The pre-earnings edge (S2) and post-earnings edge (PEAD) may compete")
        print(f"    for the same capital when earnings are close together")


if __name__ == "__main__":
    main()

