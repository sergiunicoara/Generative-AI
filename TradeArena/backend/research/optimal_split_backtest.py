"""
Per-Ticker Optimal Option Split Backtest
==========================================
Tests every combination of stock/option splits independently for
NVDA, AMD and AMZN to find the true optimal allocation per ticker.

MSFT and META: always 100% stock (proven in filtered_options_backtest.py).
GOOGL: tested separately at the end to see if it belongs in the options tier.

Splits tested per ticker: 100/0, 90/10, 80/20, 75/25, 70/30, 60/40, 50/50
Combinations: 7 × 7 × 7 = 343 for (NVDA, AMD, AMZN)

Ranked by: ann/dd ratio (risk-adjusted), then absolute annual return.

Usage:
    uv run python -m backend.research.optimal_split_backtest
"""

from __future__ import annotations

import math
import sys
from itertools import product

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
# Build trades
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
                "ticker":    ticker, "ann": ann,
                "entry_dt":  e_dt,   "exit_dt": x_dt,
                "ret_stock": round(ret_s, 3),
                "ret_opt":   round(ret_o, 3),
                "score":     round(sc, 3),
                "stopped":   stopped,
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
# Simulate with a per-ticker split map
# ---------------------------------------------------------------------------

def simulate(trades: list[dict], split_map: dict[str, tuple[float,float]]) -> dict:
    """
    split_map: {ticker: (stock_pct, opt_pct)}
    Default for tickers not in map: (1.0, 0.0)
    """
    equity = START_CASH
    peak   = START_CASH
    max_dd = 0.0
    wins   = losses = 0
    milestones: dict = {}

    for t in trades:
        for m in MILESTONES:
            if m not in milestones and equity >= m:
                milestones[m] = t["entry_dt"].date()

        sp, op = split_map.get(t["ticker"], (1.0, 0.0))
        ret = sp * t["ret_stock"] + op * t["ret_opt"]

        equity *= (1 + ret / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if ret > 0: wins += 1
        else: losses += 1

    n   = wins + losses
    ann = (equity / START_CASH) ** (1 / 11) - 1
    return {
        "n": n, "wins": wins,
        "wr":     round(wins / n * 100, 1) if n else 0,
        "final":  round(equity, 2),
        "ann":    round(ann * 100, 1),
        "max_dd": round(max_dd, 1),
        "ratio":  round(ann * 100 / max_dd, 2) if max_dd > 0 else 0,
        "milestones": milestones,
    }

# ---------------------------------------------------------------------------
# Per-ticker isolated analysis
# ---------------------------------------------------------------------------

def per_ticker_sweep(trades: list[dict],
                     ticker: str,
                     fixed_others: dict[str, tuple]) -> None:
    """Sweep all splits for one ticker, holding others fixed."""
    splits = [1.00, 0.90, 0.80, 0.75, 0.70, 0.60, 0.50]
    t_trades = [t for t in trades if t["ticker"] == ticker]
    print(f"\n  {ticker} — {len(t_trades)} trades, "
          f"opt wins {sum(1 for t in t_trades if t['ret_opt']>0)}/{len(t_trades)}, "
          f"avg opt {sum(t['ret_opt'] for t in t_trades)/len(t_trades):+.1f}%")
    print(f"  {'Split':>8}  {'Win%':>5}  {'Avg%':>7}  {'Ann%':>6}  {'DD%':>5}  {'Final':>10}  {'Ratio':>6}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*10}  {'-'*6}")

    best = None
    for sp in splits:
        op = round(1.0 - sp, 2)
        sm = {**fixed_others, ticker: (sp, op)}
        r  = simulate(trades, sm)
        lbl = f"{int(sp*100)}/{int(op*100)}"
        flag = ""
        if best is None or r["ratio"] > best["ratio"]:
            best = {**r, "sp": sp, "op": op}
            flag = "  ◄"
        rets = [sp * t["ret_stock"] + op * t["ret_opt"] for t in t_trades]
        avg  = sum(rets) / len(rets) if rets else 0
        print(f"  {lbl:>8}  {r['wr']:>4.1f}%  {avg:>+6.2f}%  "
              f"{r['ann']:>+5.1f}%  {r['max_dd']:>4.1f}%  "
              f"${r['final']:>9,.0f}  {r['ratio']:>6.2f}{flag}")

    return best

# ---------------------------------------------------------------------------
# Full grid search
# ---------------------------------------------------------------------------

def full_grid(trades: list[dict],
              tickers: list[str],
              splits: list[float]) -> list[dict]:
    """Test every combination of splits for the given tickers."""
    results = []
    fixed_tickers = {t: (1.0, 0.0) for t in UNIVERSE if t not in tickers}

    for combo in product(splits, repeat=len(tickers)):
        sm = {**fixed_tickers}
        for tk, sp in zip(tickers, combo):
            sm[tk] = (sp, round(1.0 - sp, 2))
        r = simulate(trades, sm)
        r["combo"] = {tk: (sp, round(1-sp, 2)) for tk, sp in zip(tickers, combo)}
        results.append(r)

    return sorted(results, key=lambda r: -r["ratio"])

# ---------------------------------------------------------------------------
# Year-by-year
# ---------------------------------------------------------------------------

def year_by_year(trades: list[dict], split_map: dict) -> None:
    by_yr: dict[int, list] = {}
    for t in trades:
        sp, op = split_map.get(t["ticker"], (1.0, 0.0))
        ret = sp * t["ret_stock"] + op * t["ret_opt"]
        by_yr.setdefault(t["entry_dt"].year, []).append(ret)
    cum = START_CASH
    for yr in sorted(by_yr):
        rets = by_yr[yr]
        avg  = sum(rets) / len(rets)
        w    = sum(1 for r in rets if r > 0)
        for r in rets: cum *= (1 + r / 100)
        bar  = ("+" if avg >= 0 else "-") + "█" * min(int(abs(avg) / 3), 25)
        print(f"    {yr}  {w}/{len(rets)} wins  avg {avg:>+6.2f}%  ${cum:>9,.0f}  {bar}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  PER-TICKER OPTIMAL OPTION SPLIT — full grid search")
    print(SEP)

    print("\nLoading prices...")
    for t in UNIVERSE + ["QQQ"]:
        _px(t); print(f"  {t} ready")
    qqq_df = _px("QQQ")

    print("\nBuilding trades (F6 + score>=1.20)...")
    trades = build_trades(qqq_df)
    print(f"  {len(trades)} trades")

    # ── Raw per-trade option ROI by ticker ────────────────────────────────
    print(f"\n{SEP}")
    print("  PER-TRADE OPTION ROI — NVDA, AMD, AMZN")
    print(SEP)
    for ticker in ["NVDA", "AMD", "AMZN", "GOOGL"]:
        sub = [t for t in trades if t["ticker"] == ticker]
        if not sub: continue
        print(f"\n  {ticker} ({len(sub)} trades):")
        print(f"  {'Date':<12} {'Score':>6}  {'Stock%':>8}  {'Opt%':>10}  {'Stopped'}")
        for t in sub:
            print(f"  {str(t['entry_dt'].date()):<12} {t['score']:>6.3f}  "
                  f"{t['ret_stock']:>+7.2f}%  {t['ret_opt']:>+9.2f}%  "
                  f"{'✓ STOP' if t['stopped'] else '—'}")
        opt_rets = [t["ret_opt"] for t in sub]
        print(f"  → avg {sum(opt_rets)/len(opt_rets):+.1f}%  "
              f"min {min(opt_rets):+.1f}%  max {max(opt_rets):+.1f}%  "
              f"win {sum(1 for r in opt_rets if r>0)}/{len(sub)}")

    # ── Fixed baseline for comparison ────────────────────────────────────
    s2_pure   = simulate(trades, {})
    s4_ref    = simulate(trades, {"NVDA":(0.80,0.20),"AMD":(0.80,0.20),"AMZN":(0.80,0.20)})

    print(f"\n{SEP}")
    print("  REFERENCE POINTS")
    print(SEP)
    for label, r in [("S2 pure (100% stock)", s2_pure),
                     ("S4 flat 80/20 on NVDA+AMD+AMZN", s4_ref)]:
        print(f"  {label:<40}  ann {r['ann']:>+5.1f}%  dd {r['max_dd']:>5.1f}%  "
              f"final ${r['final']:>9,.0f}  ratio {r['ratio']:>5.2f}")

    # ── Per-ticker isolated sweep ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  STEP 1 — PER-TICKER SWEEP (others held at 80/20 or stock-only)")
    print(SEP)

    # Hold AMD+AMZN at 80/20 while sweeping NVDA
    fixed_amd_amzn = {"AMD":(0.80,0.20),"AMZN":(0.80,0.20),
                       "GOOGL":(1.0,0.0),"MSFT":(1.0,0.0),"META":(1.0,0.0)}
    best_nvda = per_ticker_sweep(trades, "NVDA", fixed_amd_amzn)

    # Hold NVDA at best + AMZN at 80/20 while sweeping AMD
    fixed_nvda_amzn = {"NVDA":(best_nvda["sp"],best_nvda["op"]),
                        "AMZN":(0.80,0.20),
                        "GOOGL":(1.0,0.0),"MSFT":(1.0,0.0),"META":(1.0,0.0)}
    best_amd = per_ticker_sweep(trades, "AMD", fixed_nvda_amzn)

    # Hold NVDA+AMD at best while sweeping AMZN
    fixed_nvda_amd = {"NVDA":(best_nvda["sp"],best_nvda["op"]),
                       "AMD":(best_amd["sp"],best_amd["op"]),
                       "GOOGL":(1.0,0.0),"MSFT":(1.0,0.0),"META":(1.0,0.0)}
    best_amzn = per_ticker_sweep(trades, "AMZN", fixed_nvda_amd)

    # Also sweep GOOGL
    fixed_core = {"NVDA":(best_nvda["sp"],best_nvda["op"]),
                  "AMD":(best_amd["sp"],best_amd["op"]),
                  "AMZN":(best_amzn["sp"],best_amzn["op"]),
                  "MSFT":(1.0,0.0),"META":(1.0,0.0)}
    print(f"\n  GOOGL (for comparison — should it get options?)")
    best_googl = per_ticker_sweep(trades, "GOOGL", fixed_core)

    # ── Full grid search (NVDA × AMD × AMZN) ─────────────────────────────
    print(f"\n{SEP}")
    print("  STEP 2 — FULL GRID SEARCH (NVDA × AMD × AMZN, 343 combinations)")
    print(SEP)

    splits_grid = [1.00, 0.90, 0.80, 0.75, 0.70, 0.60, 0.50]
    all_results = full_grid(trades, ["NVDA","AMD","AMZN"], splits_grid)

    print(f"\n  TOP 20 BY RISK-ADJUSTED RATIO (ann%/dd%):")
    print(f"  {'NVDA':>8}  {'AMD':>8}  {'AMZN':>8}  {'Ann%':>6}  "
          f"{'DD%':>5}  {'Final':>10}  {'Ratio':>6}  $20k")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  "
          f"{'-'*5}  {'-'*10}  {'-'*6}  {'-'*10}")
    for r in all_results[:20]:
        c  = r["combo"]
        n_sp = c["NVDA"][0]; a_sp = c["AMD"][0]; z_sp = c["AMZN"][0]
        m20  = str(r["milestones"].get(20_000,"—"))[:10]
        flag = "  ◄" if r["ratio"] > s4_ref["ratio"] else ""
        print(f"  {int(n_sp*100):>4}/{int((1-n_sp)*100):<3}  "
              f"{int(a_sp*100):>4}/{int((1-a_sp)*100):<3}  "
              f"{int(z_sp*100):>4}/{int((1-z_sp)*100):<3}  "
              f"{r['ann']:>+5.1f}%  {r['max_dd']:>4.1f}%  "
              f"${r['final']:>9,.0f}  {r['ratio']:>6.2f}  {m20}{flag}")

    print(f"\n  TOP 20 BY ABSOLUTE ANNUAL RETURN:")
    print(f"  {'NVDA':>8}  {'AMD':>8}  {'AMZN':>8}  {'Ann%':>6}  "
          f"{'DD%':>5}  {'Final':>10}  {'Ratio':>6}  $20k")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  "
          f"{'-'*5}  {'-'*10}  {'-'*6}  {'-'*10}")
    by_ann = sorted(all_results, key=lambda r: -r["ann"])
    for r in by_ann[:20]:
        c  = r["combo"]
        n_sp = c["NVDA"][0]; a_sp = c["AMD"][0]; z_sp = c["AMZN"][0]
        m20  = str(r["milestones"].get(20_000,"—"))[:10]
        print(f"  {int(n_sp*100):>4}/{int((1-n_sp)*100):<3}  "
              f"{int(a_sp*100):>4}/{int((1-a_sp)*100):<3}  "
              f"{int(z_sp*100):>4}/{int((1-z_sp)*100):<3}  "
              f"{r['ann']:>+5.1f}%  {r['max_dd']:>4.1f}%  "
              f"${r['final']:>9,.0f}  {r['ratio']:>6.2f}  {m20}")

    # ── Best overall ──────────────────────────────────────────────────────
    best_ratio  = all_results[0]
    best_ann    = by_ann[0]

    print(f"\n{SEP}")
    print("  OPTIMAL SPLITS")
    print(SEP)

    for label, best in [("Best risk-adjusted (ratio)", best_ratio),
                         ("Best absolute return",       best_ann)]:
        c = best["combo"]
        n_sp=c["NVDA"][0]; a_sp=c["AMD"][0]; z_sp=c["AMZN"][0]
        print(f"\n  {label}:")
        print(f"    NVDA  {int(n_sp*100)}% stock + {int((1-n_sp)*100)}% calls")
        print(f"    AMD   {int(a_sp*100)}% stock + {int((1-a_sp)*100)}% calls")
        print(f"    AMZN  {int(z_sp*100)}% stock + {int((1-z_sp)*100)}% calls")
        print(f"    MSFT  100% stock (no options)")
        print(f"    META  100% stock (no options)")
        print(f"    GOOGL 100% stock (no options)")
        print(f"    → ann {best['ann']:+.1f}%  dd {best['max_dd']:.1f}%  "
              f"final ${best['final']:,.0f}  ratio {best['ratio']:.2f}")

        print(f"\n  Year-by-year:")
        year_by_year(trades, {**c,
                               "GOOGL":(1.0,0.0),"MSFT":(1.0,0.0),"META":(1.0,0.0)})

    # ── Compare against S4 flat 80/20 ────────────────────────────────────
    print(f"\n{SEP}")
    print("  COMPARISON vs S4 flat 80/20")
    print(SEP)
    print(f"  {'Strategy':<45}  {'Ann%':>6}  {'DD%':>5}  {'Final':>10}  {'Ratio':>6}")
    print(f"  {'-'*45}  {'-'*6}  {'-'*5}  {'-'*10}  {'-'*6}")
    for label, r in [
        ("S2 pure (100% stock)",            s2_pure),
        ("S4 flat 80/20 (NVDA+AMD+AMZN)",   s4_ref),
        ("Optimal ratio split",             best_ratio),
        ("Optimal return split",            best_ann),
    ]:
        c = best_ratio["combo"] if label == "Optimal ratio split" else \
            best_ann["combo"]   if label == "Optimal return split" else None
        combo_str = ""
        if c:
            n=c["NVDA"][0]; a=c["AMD"][0]; z=c["AMZN"][0]
            combo_str = f" (NVDA {int(n*100)}/{int((1-n)*100)} AMD {int(a*100)}/{int((1-a)*100)} AMZN {int(z*100)}/{int((1-z)*100)})"
        print(f"  {label+combo_str:<45}  {r['ann']:>+5.1f}%  {r['max_dd']:>4.1f}%  "
              f"${r['final']:>9,.0f}  {r['ratio']:>6.2f}")


if __name__ == "__main__":
    main()

