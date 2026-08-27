"""
Filtered Options Backtest
==========================
Tests whether filtering WHICH trades get the options sleeve
improves risk-adjusted returns vs a flat 90/10 or 80/20 split.

Key question: the 8 losing option trades drag everything down.
Can we identify them in advance and skip the options on those?

Filters tested on OPTIONS sleeve only (stock leg always runs):
  T1  Ticker filter          — options only on NVDA / AMD / AMZN (best historical)
  T2  Exclude MSFT options   — MSFT caused 3 of the worst option losses
  S1  Score >= 1.40          — only add options on high-conviction trades
  S2  Score >= 1.50          — very high conviction only
  S3  Score >= 1.60
  M1  20d momentum > 10%     — stock already running hard
  M2  Stock above 20dma      — trend confirmed
  V1  VIX < 20               — calm markets only
  V2  VIX < 25
  D1  D-15 to D-1 only       — more time for drift = better option delta
  D2  D-10 to D-1 only

  Dynamic split: variable options % based on score
    score < 1.40  → 0% options (stock only)
    score 1.40–1.60 → 10% options
    score >= 1.60 → 20% options

Also tested: conditional split — options only when stock trend is very strong

Usage:
    uv run python -m backend.research.filtered_options_backtest
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

def build_trades(qqq_df: pd.DataFrame, vix_df: pd.DataFrame) -> list[dict]:
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

            # Extra context for filtering
            close_ser = df["Close"].loc[df.index <= e_dt]
            mom20 = (close_ser.iloc[-1] / close_ser.iloc[-21] - 1) * 100 if len(close_ser) >= 21 else 0
            ma20  = float(close_ser.tail(20).mean()) if len(close_ser) >= 20 else ep
            above_ma20 = ep > ma20

            # VIX at entry
            vix_now = vix_df.loc[vix_df.index <= e_dt]
            vix_val = float(vix_now["Close"].iloc[-1]) if len(vix_now) >= 1 else 20.0

            # D-window (using D-20 as baseline, compute days until earnings at entry)
            days_until = len([d for d in df.index if e_dt <= d <= x_dt]) - 1

            raw.append({
                "ticker":    ticker, "ann": ann,
                "entry_dt":  e_dt,   "exit_dt": x_dt,
                "entry_px":  ep,     "exit_px":  xp,
                "ret_stock": round(ret_s, 3),
                "ret_opt":   round(ret_o, 3),
                "score":     round(sc, 3),
                "stopped":   stopped,
                "mom20":     round(mom20, 2),
                "above_ma20":above_ma20,
                "vix":       round(vix_val, 1),
                "days_until":days_until,
                "opt_winner": ret_o > 0,
                "opt_beats_stock": ret_o > ret_s,
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
# Simulate — each trade gets a (stock_pct, opt_pct) that can vary per trade
# ---------------------------------------------------------------------------

def simulate(trades: list[dict],
             split_fn,          # callable(trade) -> (stock_pct, opt_pct)
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

        sp, op = split_fn(t)
        ret = sp * t["ret_stock"] + op * t["ret_opt"]

        equity *= (1 + ret / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if ret > 0: wins += 1
        else: losses += 1
        log.append({**t, "ret_blend": round(ret, 3),
                    "sp": sp, "op": op, "equity": round(equity, 2)})

    n    = wins + losses
    rets = [t["ret_blend"] for t in log]
    ann  = (equity / START_CASH) ** (1 / 11) - 1

    return {
        "label":     label,
        "n":         n, "wins": wins,
        "wr":        round(wins / n * 100, 1) if n else 0,
        "avg":       round(sum(rets) / n, 2) if n else 0,
        "final":     round(equity, 2),
        "ann":       round(ann * 100, 1),
        "max_dd":    round(max_dd, 1),
        "milestones":milestones,
        "log":       log,
    }

def prow(r: dict, ref: dict) -> None:
    m10  = str(r["milestones"].get(10_000, "—"))[:10]
    m20  = str(r["milestones"].get(20_000, "—"))[:10]
    d_ann = r["ann"]    - ref["ann"]
    d_dd  = r["max_dd"] - ref["max_dd"]
    d_fin = r["final"]  - ref["final"]
    ratio = r["ann"] / r["max_dd"] if r["max_dd"] > 0 else 0
    flag  = "  ◄ BETTER" if d_ann >= 0 and d_dd <= 0 else \
            "  ↑ more return" if d_ann > 2 and d_dd <= 5 else ""
    print(f"  {r['label']:<44} {r['wr']:>5.1f}%  {r['avg']:>+6.2f}%  "
          f"{r['ann']:>+6.1f}%  {r['max_dd']:>5.1f}%  "
          f"${r['final']:>10,.0f}  {ratio:>5.2f}  {m20}{flag}")

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
        bar  = ("+" if avg >= 0 else "-") + "█" * min(int(abs(avg) / 3), 25)
        print(f"    {yr}  {w}/{len(rets)} wins  avg {avg:>+6.2f}%  ${cum:>9,.0f}  {bar}")

# ---------------------------------------------------------------------------
# Per-ticker option performance analysis
# ---------------------------------------------------------------------------

def per_ticker_option_analysis(trades: list[dict]) -> None:
    print(f"\n{SEP}")
    print("  PER-TICKER OPTION PERFORMANCE")
    print(SEP)
    print(f"  {'Ticker':<8} {'N':>3}  {'Opt win%':>8}  {'Avg opt%':>9}  "
          f"{'Beats stk%':>10}  {'Worst opt%':>10}  {'Best opt%':>10}  Verdict")
    print(f"  {'-'*8} {'-'*3}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*10}")

    for ticker in UNIVERSE:
        sub = [t for t in trades if t["ticker"] == ticker]
        if not sub: continue
        opt_rets = [t["ret_opt"] for t in sub]
        stk_rets = [t["ret_stock"] for t in sub]
        wr_opt   = sum(1 for r in opt_rets if r > 0) / len(opt_rets) * 100
        avg_opt  = sum(opt_rets) / len(opt_rets)
        beats    = sum(1 for t in sub if t["ret_opt"] > t["ret_stock"]) / len(sub) * 100
        worst    = min(opt_rets)
        best     = max(opt_rets)
        verdict  = ("✓ STRONG" if wr_opt >= 75 and avg_opt >= 30 else
                    "✓ GOOD"   if wr_opt >= 60 and avg_opt >= 15 else
                    "⚠ MIXED"  if wr_opt >= 50 else "✗ WEAK")
        print(f"  {ticker:<8} {len(sub):>3}  {wr_opt:>7.1f}%  {avg_opt:>+8.1f}%  "
              f"{beats:>9.1f}%  {worst:>+9.1f}%  {best:>+9.1f}%  {verdict}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  FILTERED OPTIONS BACKTEST — which trades deserve the options sleeve?")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ"]:
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
        vix = pd.DataFrame({"Close": [20.0]}, index=[pd.Timestamp("2020-01-01")])
        print("failed — using 20")

    qqq_df = _px("QQQ")

    print("\nBuilding trades...")
    trades = build_trades(qqq_df, vix)
    print(f"  {len(trades)} trades")

    # ── Per-ticker option analysis ────────────────────────────────────────
    per_ticker_option_analysis(trades)

    # ── Option losers analysis ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  OPTION LOSERS — what do the 8 losing option trades have in common?")
    print(SEP)
    losers  = [t for t in trades if t["ret_opt"] < 0]
    winners = [t for t in trades if t["ret_opt"] >= 0]
    print(f"  {'':12} {'Ticker':<6} {'Score':>6}  {'Stock%':>7}  "
          f"{'Opt%':>8}  {'Mom20':>6}  {'VIX':>5}  {'Days':>5}  {'Stopped?'}")
    for t in losers:
        print(f"  {str(t['entry_dt'].date()):<12} {t['ticker']:<6} {t['score']:>6.3f}  "
              f"{t['ret_stock']:>+6.2f}%  {t['ret_opt']:>+7.2f}%  "
              f"{t['mom20']:>+5.1f}%  {t['vix']:>5.1f}  {t['days_until']:>5}  "
              f"{'✓' if t['stopped'] else '—'}")

    print(f"\n  Loser patterns:")
    print(f"    Avg score:    losers {sum(t['score'] for t in losers)/len(losers):.3f}  "
          f"winners {sum(t['score'] for t in winners)/len(winners):.3f}")
    print(f"    Avg VIX:      losers {sum(t['vix'] for t in losers)/len(losers):.1f}  "
          f"winners {sum(t['vix'] for t in winners)/len(winners):.1f}")
    print(f"    Avg mom20:    losers {sum(t['mom20'] for t in losers)/len(losers):.1f}%  "
          f"winners {sum(t['mom20'] for t in winners)/len(winners):.1f}%")
    print(f"    Stopped stk:  losers {sum(t['stopped'] for t in losers)}/{len(losers)}  "
          f"winners {sum(t['stopped'] for t in winners)}/{len(winners)}")
    print(f"    Tickers: {[t['ticker'] for t in losers]}")

    # ── Define split functions ─────────────────────────────────────────────
    # References
    s2_pure   = lambda t: (1.00, 0.00)
    s3_90_10  = lambda t: (0.90, 0.10)
    s3_80_20  = lambda t: (0.80, 0.20)
    s3_75_25  = lambda t: (0.75, 0.25)

    # Ticker filters — options only on specific tickers
    def opt_nvda_amd_only(t):
        return (0.80, 0.20) if t["ticker"] in ["NVDA","AMD"] else (1.00, 0.00)

    def opt_nvda_amd_amzn(t):
        return (0.80, 0.20) if t["ticker"] in ["NVDA","AMD","AMZN"] else (1.00, 0.00)

    def opt_exclude_msft(t):
        return (0.80, 0.20) if t["ticker"] != "MSFT" else (1.00, 0.00)

    def opt_exclude_msft_meta(t):
        return (0.80, 0.20) if t["ticker"] not in ["MSFT","META"] else (1.00, 0.00)

    def opt_90_exclude_msft(t):
        return (0.90, 0.10) if t["ticker"] != "MSFT" else (1.00, 0.00)

    # Score filters — options only on high-conviction trades
    def opt_score_140(t):
        return (0.80, 0.20) if t["score"] >= 1.40 else (1.00, 0.00)

    def opt_score_150(t):
        return (0.80, 0.20) if t["score"] >= 1.50 else (1.00, 0.00)

    def opt_score_160(t):
        return (0.80, 0.20) if t["score"] >= 1.60 else (1.00, 0.00)

    def opt_score_140_90(t):
        return (0.90, 0.10) if t["score"] >= 1.40 else (1.00, 0.00)

    # Momentum filters
    def opt_mom_10pct(t):
        return (0.80, 0.20) if t["mom20"] >= 10 else (1.00, 0.00)

    def opt_mom_5pct(t):
        return (0.80, 0.20) if t["mom20"] >= 5 else (1.00, 0.00)

    # VIX filter — options only in calm markets
    def opt_vix_20(t):
        return (0.80, 0.20) if t["vix"] <= 20 else (1.00, 0.00)

    def opt_vix_25(t):
        return (0.80, 0.20) if t["vix"] <= 25 else (1.00, 0.00)

    # Dynamic split based on score
    def opt_dynamic_score(t):
        if   t["score"] >= 1.60: return (0.75, 0.25)
        elif t["score"] >= 1.40: return (0.85, 0.15)
        else:                    return (1.00, 0.00)

    def opt_dynamic_score_aggressive(t):
        if   t["score"] >= 1.80: return (0.70, 0.30)
        elif t["score"] >= 1.60: return (0.80, 0.20)
        elif t["score"] >= 1.40: return (0.90, 0.10)
        else:                    return (1.00, 0.00)

    # Combination filters
    def opt_nvda_amd_score140(t):
        if t["ticker"] in ["NVDA","AMD"] and t["score"] >= 1.40:
            return (0.75, 0.25)
        return (1.00, 0.00)

    def opt_best_combo(t):
        # No options on MSFT, only options when score >= 1.40 and VIX <= 25
        if t["ticker"] == "MSFT": return (1.00, 0.00)
        if t["score"] >= 1.40 and t["vix"] <= 25: return (0.80, 0.20)
        return (1.00, 0.00)

    def opt_best_combo_v2(t):
        # No options on MSFT/META, only when score >= 1.40
        if t["ticker"] in ["MSFT","META"]: return (1.00, 0.00)
        if t["score"] >= 1.40: return (0.80, 0.20)
        return (1.00, 0.00)

    def opt_aggressive_filtered(t):
        # NVDA/AMD/AMZN/GOOGL with score >= 1.40 get 75/25
        if t["ticker"] in ["NVDA","AMD","AMZN","GOOGL"] and t["score"] >= 1.40:
            return (0.75, 0.25)
        elif t["ticker"] not in ["MSFT","META"] and t["score"] >= 1.50:
            return (0.80, 0.20)
        return (1.00, 0.00)

    def opt_tiered(t):
        # Tier by ticker + score
        strong  = t["ticker"] in ["NVDA","AMD"] and t["score"] >= 1.50
        medium  = t["ticker"] in ["GOOGL","AMZN"] and t["score"] >= 1.40
        weak    = t["ticker"] in ["MSFT","META"]
        if strong: return (0.70, 0.30)
        if medium: return (0.80, 0.20)
        if weak:   return (1.00, 0.00)
        return (0.90, 0.10)

    # ── Run all configurations ─────────────────────────────────────────────
    configs = [
        # References
        ("S2 pure (100% stock)",                    s2_pure),
        ("S3  90/10 flat",                          s3_90_10),
        ("S3  80/20 flat",                          s3_80_20),
        ("S3  75/25 flat",                          s3_75_25),
        # Ticker filters
        ("80/20 on NVDA+AMD only",                  opt_nvda_amd_only),
        ("80/20 on NVDA+AMD+AMZN only",             opt_nvda_amd_amzn),
        ("80/20 excl. MSFT",                        opt_exclude_msft),
        ("80/20 excl. MSFT+META",                   opt_exclude_msft_meta),
        ("90/10 excl. MSFT",                        opt_90_exclude_msft),
        # Score filters
        ("80/20 when score >= 1.40",                opt_score_140),
        ("80/20 when score >= 1.50",                opt_score_150),
        ("80/20 when score >= 1.60",                opt_score_160),
        ("90/10 when score >= 1.40",                opt_score_140_90),
        # Momentum filters
        ("80/20 when 20d mom >= 10%",               opt_mom_10pct),
        ("80/20 when 20d mom >= 5%",                opt_mom_5pct),
        # VIX filters
        ("80/20 when VIX <= 20",                    opt_vix_20),
        ("80/20 when VIX <= 25",                    opt_vix_25),
        # Dynamic splits
        ("Dynamic: 75/25 score>=1.60, 85/15 >=1.40", opt_dynamic_score),
        ("Dynamic aggressive (score-tiered)",       opt_dynamic_score_aggressive),
        # Combinations
        ("NVDA+AMD score>=1.40 → 75/25, else stock", opt_nvda_amd_score140),
        ("80/20 if not MSFT + score>=1.40 + VIX<=25", opt_best_combo),
        ("80/20 if not MSFT/META + score>=1.40",    opt_best_combo_v2),
        ("Aggressive filtered (NVDA/AMD/AMZN/GOOGL+score>=1.40)", opt_aggressive_filtered),
        ("Tiered: NVDA/AMD→70/30, GOOGL/AMZN→80/20, MSFT/META→stock", opt_tiered),
    ]

    ref = simulate(trades, s3_90_10, "S3 90/10 flat")

    print(f"\n{SEP}")
    print("  ALL CONFIGURATIONS vs S3 90/10 reference")
    print(SEP)
    print(f"  {'Strategy':<44} {'Win%':>5}  {'Avg%':>6}  {'Ann%':>6}  "
          f"{'DD%':>5}  {'Final':>10}  {'Ratio':>5}  $20k")
    print(f"  {'-'*44} {'-'*5}  {'-'*6}  {'-'*6}  "
          f"{'-'*5}  {'-'*10}  {'-'*5}  {'-'*10}")

    results = []
    for label, fn in configs:
        r = simulate(trades, fn, label)
        results.append(r)
        prow(r, ref)

    # ── Best result deep dive ──────────────────────────────────────────────
    # Best: higher ann than S3 90/10 with lower or equal DD
    better = [r for r in results
              if r["ann"] >= ref["ann"] and r["max_dd"] <= ref["max_dd"]]
    if not better:
        better = [r for r in results
                  if r["ann"] >= ref["ann"] and r["max_dd"] <= ref["max_dd"] + 3]

    if better:
        best = max(better, key=lambda r: r["ann"] / max(r["max_dd"], 1))
        print(f"\n{SEP}")
        print(f"  BEST: {best['label']}")
        print(f"  {best['n']} trades  win {best['wr']:.1f}%  avg {best['avg']:+.2f}%  "
              f"ann {best['ann']:+.1f}%  dd {best['max_dd']:.1f}%  "
              f"final ${best['final']:,.0f}  ratio {best['ann']/best['max_dd']:.2f}")
        print(SEP)
        print(f"\n  Year-by-year:")
        year_by_year(best)

        print(f"\n  Milestones:")
        for m in MILESTONES:
            ms = best["milestones"].get(m)
            print(f"    ${m:>6,}: {str(ms)[:10] if ms else '—'}")

        print(f"\n  Per-trade breakdown (showing options allocation per trade):")
        print(f"  {'Date':<12} {'Ticker':<6} {'Score':>6}  {'Split':>8}  "
              f"{'Stock%':>7}  {'Opt%':>8}  {'Blend%':>8}  {'Equity':>10}")
        for t in best["log"]:
            split = f"{int(t['sp']*100)}/{int(t['op']*100)}"
            print(f"  {str(t['entry_dt'].date()):<12} {t['ticker']:<6} {t['score']:>6.3f}  "
                  f"{split:>8}  {t['ret_stock']:>+6.2f}%  {t['ret_opt']:>+7.2f}%  "
                  f"{t['ret_blend']:>+7.2f}%  ${t['equity']:>9,.0f}")

    # ── Summary verdict ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VERDICT — can filtering beat flat S3 90/10?")
    print(SEP)
    s3_ref = next(r for r in results if "90/10 flat" in r["label"])
    beats_both = [r for r in results
                  if r["ann"] > s3_ref["ann"] and r["max_dd"] < s3_ref["max_dd"]
                  and "flat" not in r["label"] and "pure" not in r["label"]]
    if beats_both:
        print(f"  YES — {len(beats_both)} configuration(s) beat S3 90/10 on BOTH return AND drawdown:")
        for r in sorted(beats_both, key=lambda x: -x["ann"]):
            print(f"    {r['label']}")
            print(f"      ann {r['ann']:+.1f}% (vs {s3_ref['ann']:+.1f}%)  "
                  f"dd {r['max_dd']:.1f}% (vs {s3_ref['max_dd']:.1f}%)  "
                  f"final ${r['final']:,.0f}")
    else:
        beats_ann = [r for r in results
                     if r["ann"] > s3_ref["ann"] and "flat" not in r["label"]
                     and "pure" not in r["label"]]
        if beats_ann:
            print(f"  PARTIAL — {len(beats_ann)} config(s) beat return but with higher DD:")
            for r in sorted(beats_ann, key=lambda x: -x["ann"])[:5]:
                print(f"    {r['label']}")
                print(f"      ann {r['ann']:+.1f}%  dd {r['max_dd']:.1f}%  "
                      f"ratio {r['ann']/r['max_dd']:.2f}")
        else:
            print(f"  NO — flat S3 90/10 remains the best options configuration.")
            print(f"  S3 90/10: ann {s3_ref['ann']:+.1f}%  dd {s3_ref['max_dd']:.1f}%")
            best_filtered = max(
                [r for r in results if "flat" not in r["label"] and "pure" not in r["label"]],
                key=lambda r: r["ann"] / max(r["max_dd"], 1)
            )
            print(f"  Best filtered: {best_filtered['label']}")
            print(f"    ann {best_filtered['ann']:+.1f}%  dd {best_filtered['max_dd']:.1f}%  "
                  f"ratio {best_filtered['ann']/max(best_filtered['max_dd'],1):.2f}")


if __name__ == "__main__":
    main()

