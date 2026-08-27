"""
Paper Trader — 3 accounts, one per strategy from combined_strategy.md

  S1  Fixed-rank stock    GOOGL > NVDA > AMZN, 90% deployment, 1 position
  S2  Dynamic stock       Best score ≥ 1.05, all 6 tickers, 90%, 1 position
  S3  Stock + 10% calls   S2 rules + 10% ITM/60DTE call sleeve (modeled)

All three run daily at 09:35, execute trades, email a comparison summary.

Usage:
    uv run python -m backend.research.paper_trader          # normal daily run
    uv run python -m backend.research.paper_trader --reset  # reset all 3 accounts
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from scipy.stats import norm
from dotenv import load_dotenv

load_dotenv(override=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT  = Path(__file__).resolve().parents[2]
DB_PATH    = REPO_ROOT / "backend" / "environment" / "paper_trading.db"
START_CASH = 2_000.0
STOP_PCT   = -5.0
DEPLOY_PCT = 0.90     # 90% deployed, 10% cash buffer
RISK_FREE  = 0.05
EXPIRY_DAYS = 60      # S3 call option DTE at entry

from backend.environment.accounts import Accounts
from backend.environment.prices   import Prices
from backend.traders.tools import (
    _check_regime,
    _compute_scores,
    _next_earnings,
    _SCORE_THRESHOLD,
)
from backend.research.signal_notify import send_email

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    id:          str
    name:        str
    ticker_id:   str          # trader_id in DB
    universe:    list[str]
    fixed_rank:  bool         # True = S1 (rank by position in universe list)
    options_pct: float        # 0.0 for stock-only, 0.10 for S3

S1 = Strategy(
    id="s1", name="S1 Fixed-rank",
    ticker_id="paper_s1",
    universe=["GOOGL", "NVDA", "AMZN"],
    fixed_rank=True,
    options_pct=0.0,
)
S2 = Strategy(
    id="s2", name="S2 Dynamic (safer)",
    ticker_id="paper_s2",
    universe=["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"],
    fixed_rank=False,
    options_pct=0.0,
)
S3 = Strategy(
    id="s3", name="S3 Stock + 10% calls",
    ticker_id="paper_s3",
    universe=["GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"],
    fixed_rank=False,
    options_pct=0.10,
)

ALL_STRATEGIES = [S1, S2, S3]

# ---------------------------------------------------------------------------
# Virtual options ledger (JSON file — S3 only)
# ---------------------------------------------------------------------------

OPTIONS_FILE = REPO_ROOT / "backend" / "environment" / "paper_options.json"

def load_options() -> dict:
    if OPTIONS_FILE.exists():
        try: return json.loads(OPTIONS_FILE.read_text())
        except Exception: pass
    return {}   # {trader_id: {ticker: {strike, entry_px, qty, entry_date, expiry_days, iv}}}

def save_options(data: dict) -> None:
    OPTIONS_FILE.write_text(json.dumps(data, indent=2, default=str))

# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(S - K, 0.0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)

def hist_vol_for(ticker: str) -> float:
    try:
        h = yf.download(ticker, period="60d", interval="1d",
                        progress=False, auto_adjust=True)
        if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.get_level_values(0)
        rets = h["Close"].pct_change().dropna()
        return float(rets.tail(30).std() * math.sqrt(252))
    except Exception:
        return 0.35

def option_value(ticker: str, opt: dict, current_px: float) -> float:
    """Current Black-Scholes value of a held call, with 25% real-world haircut."""
    from datetime import date
    entry  = date.fromisoformat(opt["entry_date"])
    today  = date.today()
    days_held = (today - entry).days
    T_rem  = max((opt["expiry_days"] - days_held * 1.4) / 365, 0)
    raw    = bs_call(current_px, opt["strike"], T_rem, RISK_FREE, opt["iv"])
    return raw * 0.75   # 25% haircut for spread/real-world

# ---------------------------------------------------------------------------
# Account bootstrap
# ---------------------------------------------------------------------------

def get_accounts() -> Accounts:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    acct = Accounts(DB_PATH)
    for s in ALL_STRATEGIES:
        try:
            acct.cash(s.ticker_id)
        except KeyError:
            acct.create_trader(s.ticker_id, START_CASH)
            print(f"  Created {s.name} account ({s.ticker_id}) @ ${START_CASH:,.0f}")
    return acct

def reset_accounts() -> None:
    acct = Accounts(DB_PATH)
    acct.reset_working_state()
    for s in ALL_STRATEGIES:
        acct.create_trader(s.ticker_id, START_CASH)
        print(f"  Reset {s.name} → ${START_CASH:,.0f}")
    acct.close()
    save_options({})
    print("All 3 accounts reset.")

# ---------------------------------------------------------------------------
# Select entry for a strategy
# ---------------------------------------------------------------------------

def pick_entry(strategy: Strategy, scores: dict, held_tickers: set) -> str | None:
    """Return the best ticker to enter, or None."""
    candidates = []
    for ticker in strategy.universe:
        if ticker in held_tickers:
            continue
        er = _next_earnings(ticker)
        if not er:
            continue
        ann, td = er
        if not (10 <= td <= 25):
            continue
        score = scores.get(ticker, 0.0)
        if not strategy.fixed_rank and score < _SCORE_THRESHOLD:
            continue
        candidates.append((ticker, score, td))

    if not candidates:
        return None

    if strategy.fixed_rank:
        # Pick first in universe list that has a valid signal
        for ticker in strategy.universe:
            if any(t == ticker for t, _, _ in candidates):
                return ticker
        return None
    else:
        # Pick highest score
        return max(candidates, key=lambda x: x[1])[0]

# ---------------------------------------------------------------------------
# Run one strategy
# ---------------------------------------------------------------------------

async def run_strategy(
    strategy: Strategy,
    acct:     Accounts,
    prices:   Prices,
    scores:   dict,
    regime:   dict,
    opts:     dict,
) -> dict:
    """Execute one strategy cycle. Returns result dict."""
    tid      = strategy.ticker_id
    actions  = []
    errors   = []

    holdings = acct.holdings(tid)
    held     = set(holdings)

    # Fetch live prices for held positions
    cur_prices: dict[str, float] = {}
    if held:
        cur_prices = await prices.aget_prices(sorted(held))

    # Options value for S3
    opt_positions = opts.get(tid, {})
    opt_value_total = sum(
        option_value(t, o, cur_prices.get(t, holdings[t]["avg_cost"])) * o["qty"] * 100
        if t in holdings else 0
        for t, o in opt_positions.items()
    )

    # Portfolio value: stock + cash + option market value
    stock_value = acct.portfolio_value(tid, cur_prices) if cur_prices or not holdings else acct.cash(tid)
    portfolio   = stock_value + opt_value_total
    pnl         = portfolio - START_CASH
    ret_pct     = pnl / START_CASH * 100

    # ── EXITS ────────────────────────────────────────────────────────────
    to_sell: list[tuple[str, str]] = []
    for ticker, pos in holdings.items():
        px = cur_prices.get(ticker, pos["avg_cost"])
        # Stop
        if px <= pos["avg_cost"] * (1 + STOP_PCT / 100):
            to_sell.append((ticker, f"stop -5% (${px:.2f} ≤ ${pos['avg_cost']*0.95:.2f})"))
            continue
        # D-1
        er = _next_earnings(ticker)
        if er and er[1] <= 1:
            to_sell.append((ticker, f"D-1 exit (earnings {er[0]})"))

    for ticker, reason in to_sell:
        pos = holdings[ticker]
        px  = cur_prices.get(ticker, pos["avg_cost"])
        qty = pos["quantity"]
        try:
            acct.execute_trade(tid, ticker, -qty, px)
            stock_ret = (px - pos["avg_cost"]) / pos["avg_cost"] * 100
            stock_pnl = (px - pos["avg_cost"]) * qty

            # S3: close option too
            opt_pnl = 0.0
            if strategy.options_pct > 0 and ticker in opt_positions:
                o       = opt_positions[ticker]
                o_exit  = option_value(ticker, o, px) * o["qty"] * 100
                o_entry = o["entry_px"] * o["qty"] * 100
                opt_pnl = o_exit - o_entry
                del opt_positions[ticker]
                opts[tid] = opt_positions

            total_pnl = stock_pnl + opt_pnl
            actions.append({
                "action": "SELL", "ticker": ticker,
                "price": px, "qty": qty,
                "stock_ret_pct": round(stock_ret, 2),
                "stock_pnl": round(stock_pnl, 2),
                "opt_pnl": round(opt_pnl, 2),
                "total_pnl": round(total_pnl, 2),
                "reason": reason,
            })
        except Exception as e:
            errors.append(f"SELL {ticker}: {e}")

    # Refresh
    holdings = acct.holdings(tid)
    held     = set(holdings)

    # ── ENTRY ────────────────────────────────────────────────────────────
    if regime["ok"] and len(holdings) == 0:   # 1 position at a time
        ticker = pick_entry(strategy, scores, held)
        if ticker:
            er      = _next_earnings(ticker)
            ann, td = er if er else ("?", 0)
            score   = scores.get(ticker, 0.0)
            try:
                px = await prices.aget_price(ticker)
            except Exception as e:
                errors.append(f"Price {ticker}: {e}")
                ticker = None

            if ticker:
                cash     = acct.cash(tid)
                # 90% stock, or 80% stock + 10% calls for S3
                stock_alloc = cash * (DEPLOY_PCT - strategy.options_pct)
                qty         = stock_alloc / px
                try:
                    acct.execute_trade(tid, ticker, qty, px)
                    actions.append({
                        "action": "BUY", "ticker": ticker,
                        "price": px, "qty": round(qty, 6),
                        "cost": round(qty*px, 2),
                        "score": round(score, 3),
                        "td": td, "earnings_date": ann,
                    })

                    # S3: buy modeled call with remaining 10%
                    if strategy.options_pct > 0:
                        iv      = hist_vol_for(ticker)
                        K       = px * 0.90   # 10% ITM
                        T0      = EXPIRY_DAYS / 365
                        c0      = bs_call(px, K, T0, RISK_FREE, iv) * 0.75  # haircut
                        opt_cash= cash * strategy.options_pct
                        opt_contracts = opt_cash / (c0 * 100) if c0 > 0.01 else 0
                        if opt_contracts > 0:
                            opt_positions[ticker] = {
                                "strike": round(K, 4),
                                "entry_px": round(c0, 4),
                                "qty": round(opt_contracts, 6),
                                "entry_date": datetime.now().strftime("%Y-%m-%d"),
                                "expiry_days": EXPIRY_DAYS,
                                "iv": round(iv, 4),
                            }
                            opts[tid] = opt_positions
                            actions[-1]["option"] = {
                                "strike": round(K, 2),
                                "premium": round(c0, 2),
                                "contracts": round(opt_contracts, 4),
                                "cost": round(opt_cash, 2),
                                "iv_pct": round(iv*100, 1),
                            }
                except Exception as e:
                    errors.append(f"BUY {ticker}: {e}")

    # Final state
    holdings   = acct.holdings(tid)
    cur_prices2: dict[str, float] = {}
    if holdings:
        cur_prices2 = await prices.aget_prices(sorted(holdings))
    opt_value2 = sum(
        option_value(t, o, cur_prices2.get(t, holdings[t]["avg_cost"])) * o["qty"] * 100
        if t in holdings else 0
        for t, o in opts.get(tid, {}).items()
    )
    stock_val2 = acct.portfolio_value(tid, cur_prices2) if cur_prices2 or not holdings else acct.cash(tid)
    final_port  = stock_val2 + opt_value2
    final_pnl   = final_port - START_CASH
    final_ret   = final_pnl / START_CASH * 100

    return {
        "strategy":    strategy.name,
        "id":          strategy.id,
        "portfolio":   round(final_port, 2),
        "pnl":         round(final_pnl, 2),
        "ret_pct":     round(final_ret, 2),
        "cash":        round(acct.cash(tid), 2),
        "holdings":    {t: {**p, "price": cur_prices2.get(t, p["avg_cost"])}
                        for t, p in holdings.items()},
        "opt_positions": opts.get(tid, {}),
        "opt_value":   round(opt_value2, 2),
        "actions":     actions,
        "errors":      errors,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_all() -> None:
    ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\nPaper trader — {ts}")
    print("=" * 60)

    acct   = get_accounts()
    prices = Prices()
    opts   = load_options()

    # Shared: regime + scores (computed once for all strategies)
    print("  Computing regime + scores...")
    regime = _check_regime()
    scores = _compute_scores()
    print(f"  Regime: QQQ {regime.get('qqq')} vs 150dma {regime.get('ma150')} → "
          f"{'OK' if regime['ok'] else 'OFF'}")
    for t, s in sorted(scores.items(), key=lambda x: -x[1]):
        flag = " ✓" if s >= _SCORE_THRESHOLD else ""
        print(f"    {t:<6} {s:.3f}{flag}")

    # Run all 3
    results = []
    for s in ALL_STRATEGIES:
        print(f"\n  ── {s.name} ({s.ticker_id}) ──")
        try:
            r = await run_strategy(s, acct, prices, scores, regime, opts)
            results.append(r)
            print(f"    Portfolio: ${r['portfolio']:,.2f}  ({r['ret_pct']:+.2f}%)")
            for a in r["actions"]:
                if a["action"] == "BUY":
                    opt_str = f" + call K={a['option']['strike']:.2f}" if "option" in a else ""
                    print(f"    ✓ BUY  {a['ticker']} {a['qty']:.4f}sh @${a['price']:.2f}"
                          f"  score {a['score']:.3f}  D-{a['td']}{opt_str}")
                else:
                    print(f"    ✓ SELL {a['ticker']}  {a['stock_ret_pct']:+.1f}%"
                          f"  stock ${a['stock_pnl']:+,.2f}"
                          + (f"  opt ${a['opt_pnl']:+,.2f}" if a["opt_pnl"] else "")
                          + f"  — {a['reason']}")
            for h, pos in r["holdings"].items():
                px   = pos["price"]
                ret_ = (px - pos["avg_cost"]) / pos["avg_cost"] * 100
                er   = _next_earnings(h)
                days = f"D-{er[1]}" if er else "?"
                stop = pos["avg_cost"] * 0.95
                print(f"    → {h} {pos['quantity']:.4f}sh  entry ${pos['avg_cost']:.2f}"
                      f"  now ${px:.2f}  {ret_:+.1f}%  stop ${stop:.2f}  {days}")
                opt = r["opt_positions"].get(h)
                if opt:
                    oval = option_value(h, opt, px) * opt["qty"] * 100
                    print(f"       call K=${opt['strike']:.2f}  {opt['qty']:.4f}c"
                          f"  entry ${opt['entry_px']:.2f} → now ~${oval/opt['qty']/100:.2f}/sh"
                          f"  val ${oval:.2f}")
            if r["errors"]:
                for e in r["errors"]: print(f"    ✗ {e}")
        except Exception as ex:
            print(f"    ERROR: {ex}")
            results.append({"strategy": s.name, "id": s.id, "portfolio": START_CASH,
                             "pnl": 0, "ret_pct": 0, "cash": START_CASH,
                             "holdings": {}, "opt_positions": {}, "opt_value": 0,
                             "actions": [], "errors": [str(ex)]})

    save_options(opts)
    acct.close()

    # ── Comparison summary ────────────────────────────────────────────────
    print(f"\n  {'Strategy':<24} {'Portfolio':>12} {'P&L':>10} {'Ret%':>7} {'Position'}")
    print(f"  {'-'*24} {'-'*12} {'-'*10} {'-'*7} {'-'*20}")
    for r in results:
        held = ", ".join(r["holdings"].keys()) or "cash"
        print(f"  {r['strategy']:<24} ${r['portfolio']:>11,.2f} "
              f"${r['pnl']:>+9,.2f} {r['ret_pct']:>+6.2f}%  {held}")

    _send_email(results, regime, scores, ts)


def _send_email(results: list, regime: dict, scores: dict, ts: str) -> None:
    any_action = any(r["actions"] for r in results)
    total_actions = sum(len(r["actions"]) for r in results)

    if any_action:
        action_parts = []
        for r in results:
            for a in r["actions"]:
                if a["action"] == "BUY":
                    action_parts.append(f"{r['id'].upper()}: BUY {a['ticker']} D-{a['td']}")
                else:
                    action_parts.append(f"{r['id'].upper()}: SELL {a['ticker']} {a['stock_ret_pct']:+.1f}%")
        subject = f"[PaperTrader] {ts[:10]} — {' | '.join(action_parts)}"
    else:
        rets = " | ".join(f"{r['id'].upper()} {r['ret_pct']:+.1f}%" for r in results)
        subject = f"[PaperTrader] {ts[:10]} — No action | {rets}"

    lines = [
        f"PAPER TRADER DAILY — {ts}",
        "=" * 56,
        f"Regime: QQQ {regime.get('qqq')} vs 150dma {regime.get('ma150')}"
        f"  → {'TRADING' if regime['ok'] else 'HOLD CASH'}",
        "",
        f"{'Strategy':<24} {'Portfolio':>12} {'P&L':>10} {'Ret%':>8}",
        f"{'-'*24} {'-'*12} {'-'*10} {'-'*8}",
    ]
    for r in results:
        lines.append(f"{r['strategy']:<24} ${r['portfolio']:>11,.2f} "
                     f"${r['pnl']:>+9,.2f} {r['ret_pct']:>+7.2f}%")

    if any_action:
        lines += ["", "TRADES TODAY:"]
        for r in results:
            for a in r["actions"]:
                if a["action"] == "BUY":
                    opt = f" + call K={a['option']['strike']:.2f}" if "option" in a else ""
                    lines.append(f"  [{r['id'].upper()}] BUY {a['ticker']}  {a['qty']:.4f}sh"
                                 f" @${a['price']:.2f}  score {a['score']:.3f}  D-{a['td']}{opt}")
                else:
                    lines.append(f"  [{r['id'].upper()}] SELL {a['ticker']}"
                                 f"  stock {a['stock_ret_pct']:+.1f}% (${a['stock_pnl']:+,.2f})"
                                 + (f"  opt ${a['opt_pnl']:+,.2f}" if a["opt_pnl"] else "")
                                 + f"  — {a['reason']}")

    lines += ["", "OPEN POSITIONS:"]
    for r in results:
        if r["holdings"]:
            for t, pos in r["holdings"].items():
                px   = pos["price"]
                ret_ = (px - pos["avg_cost"]) / pos["avg_cost"] * 100
                er   = _next_earnings(t)
                days = f"D-{er[1]}" if er else "?"
                lines.append(f"  [{r['id'].upper()}] {t}  entry ${pos['avg_cost']:.2f}"
                             f"  now ${px:.2f}  {ret_:+.1f}%  {days}")
                opt = r["opt_positions"].get(t)
                if opt:
                    lines.append(f"         call K=${opt['strike']:.2f}  val ${r['opt_value']:.2f}")
        else:
            lines.append(f"  [{r['id'].upper()}] cash")

    lines += [
        "", "DYNAMIC SCORES:",
        *[f"  {t:<6} {s:.3f}{'  ✓' if s >= _SCORE_THRESHOLD else ''}"
          for t, s in sorted(scores.items(), key=lambda x: -x[1])],
        "", "S1=Fixed-rank  S2=Dynamic  S3=S2+10%calls(modeled)",
    ]

    send_email(subject, "\n".join(lines))
    print(f"\n  Email: {subject}")


def main() -> None:
    if "--reset" in sys.argv:
        reset_accounts()
        return
    asyncio.run(run_all())


if __name__ == "__main__":
    main()

