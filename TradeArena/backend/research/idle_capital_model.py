"""
Idle Capital & Minimum Account Analysis
=========================================
Answers two questions:
  1. Is it worth moving idle capital from IB → Romanian bank RON deposit
     during the ~250 idle days/year when S2 has no open position?

  2. What is the minimum account size to cover all running costs and
     generate meaningful net-of-tax returns?

Strategy context (from backtest 2015-2026):
  S2 alone       : +24.3%/yr,  -6.9% DD
  S2 + MG (≥1%)  : +30.2%/yr, -10.0% DD    ← deployment target

IB → Romanian bank flow (idle periods):
  IB (USD/EUR) → SEPA → Romanian bank → EUR→RON convert → 6%/yr deposit
  → RON→EUR convert → SEPA → IB

Usage:
    uv run python -m backend.research.idle_capital_model
"""

from __future__ import annotations
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP = "=" * 72

# ---------------------------------------------------------------------------
# Parameters — adjust to your situation
# ---------------------------------------------------------------------------

# Strategy returns (annualised, from backtest)
S2_ANNUAL_RETURN   = 0.243   # S2 alone
S2MG_ANNUAL_RETURN = 0.302   # S2 + MG (≥1% gap) — recommended

# Idle capital: how many days/year S2+MG has no open position
# From backtest busy-day analysis: S2 uses 86-171 days/year (avg ~115)
# MG is intraday so capital is idle outside those days
# Conservative: 200 idle days/year (some MG trades use capital briefly)
IDLE_DAYS_PER_YEAR = 200

# --- IB costs ---
IB_COMMISSION_PER_TRADE  = 1.00    # EUR — IB Pro per-trade minimum
S2MG_TRADES_PER_YEAR     = 13      # S2: ~4/yr, MG: ~9/yr
IB_INACTIVITY_MONTHS     = 2       # months with <$10 commissions → €10/mo fee
IB_INACTIVITY_FEE        = 10.0    # EUR/month

# --- Taxes (Romania) ---
RO_CAPITAL_GAINS_TAX = 0.10   # 10% on stock/option gains
RO_DEPOSIT_INTEREST_TAX = 0.10  # 10% withheld by bank on deposit interest

# --- RON deposit parameters ---
RON_DEPOSIT_RATE    = 0.06    # 6%/year gross
RON_FX_SPREAD_ONE_WAY = 0.008  # 0.8% bank EUR/RON spread each way
IB_FX_EURUSD        = 0.002   # 0.2% IB EUR↔USD conversion each way

# FX cost components per round trip:
#   EUR→RON (at Romanian bank): 0.8%
#   RON→EUR (at Romanian bank): 0.8%
#   EUR→USD (at IB, re-deploying): 0.2%
#   USD→EUR (at IB, withdrawing): 0.2%
FX_ROUNDTRIP_COST = 2 * RON_FX_SPREAD_ONE_WAY + 2 * IB_FX_EURUSD  # 2.0%

# RON depreciation vs EUR (historical ~0.5-1.5%/year average)
RON_DEPRECIATION_RISK   = 0.010   # 1%/year — conservative estimate
RON_TRANSFERS_PER_YEAR  = 1       # IB→bank round trips per year

# --- IB money market (alternative to RON deposits) ---
IB_MONEY_MARKET_RATE    = 0.045   # ~4.5% on EUR balances > €10k (current rates)
IB_MONEY_MARKET_FX_COST = 0.004   # minimal FX round trip at IB (USD↔EUR 0.2%×2)
IB_MM_MIN_BALANCE       = 10_000  # EUR — IB starts paying meaningful rate above this

# SEPA transfer cost: free for SEPA (within EU)
SEPA_TRANSFER_COST = 0.0

# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------

def ron_deposit_net_yield(idle_days: int, n_trips: int = 1) -> dict:
    """Net annual gain (as % of account) from RON deposit strategy."""
    T = idle_days / 365

    gross_deposit  = RON_DEPOSIT_RATE * T * n_trips
    tax_deduction  = gross_deposit * RO_DEPOSIT_INTEREST_TAX
    after_tax      = gross_deposit - tax_deduction

    depreciation   = RON_DEPRECIATION_RISK * T * n_trips
    fx_cost        = FX_ROUNDTRIP_COST * n_trips

    net = after_tax - depreciation - fx_cost
    return {
        "gross_deposit_pct":   round(gross_deposit * 100, 3),
        "tax_pct":             round(tax_deduction * 100, 3),
        "after_tax_pct":       round(after_tax * 100, 3),
        "depreciation_pct":    round(depreciation * 100, 3),
        "fx_cost_pct":         round(fx_cost * 100, 3),
        "net_pct":             round(net * 100, 3),
    }


def ib_mm_net_yield(idle_days: int, account: float) -> float:
    """Net annual gain from IB money market on idle balance (% of account)."""
    if account < IB_MM_MIN_BALANCE:
        return 0.0   # IB pays 0% on small balances
    T = idle_days / 365
    return (IB_MONEY_MARKET_RATE * T - IB_MONEY_MARKET_FX_COST) * 100


def ib_tbill_net_yield(idle_days: int) -> float:
    """Net yield from IB US T-Bills ETF (BIL/SHV) on idle USD balance."""
    # T-Bill rate ~5.3%, minimal friction (ETF spread ~0.1%, FX 0.4% round trip)
    T = idle_days / 365
    return (0.053 * T - 0.001 - 0.004) * 100  # ETF expense + FX


def annual_ib_cost(n_trades: int) -> float:
    """Total IB costs per year in EUR."""
    commissions = n_trades * IB_COMMISSION_PER_TRADE
    inactivity  = IB_INACTIVITY_MONTHS * IB_INACTIVITY_FEE
    return commissions + inactivity


def net_strategy_return(gross_annual_return: float, tax_rate: float,
                        ib_cost_eur: float, account: float) -> float:
    """Net annual return after tax and fixed IB costs."""
    gross_gain   = account * gross_annual_return
    tax          = gross_gain * tax_rate
    net_gain     = gross_gain - tax - ib_cost_eur
    return net_gain / account * 100


def min_account_for_target(target_eur: float, net_return_pct: float,
                            ib_cost_eur: float) -> float:
    """Account size needed to clear target EUR after IB costs."""
    # target = account × net_return_pct/100 - ib_cost
    return (target_eur + ib_cost_eur) / (net_return_pct / 100)


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def main():
    print(SEP)
    print("  IDLE CAPITAL & MINIMUM ACCOUNT ANALYSIS")
    print(SEP)

    ib_cost = annual_ib_cost(S2MG_TRADES_PER_YEAR)

    # ── 1. RON deposit cost breakdown ────────────────────────────────────
    print(f"\n{SEP}")
    print("  1. RON DEPOSIT — cost breakdown per round trip (1 transfer/year)")
    print(SEP)
    ron = ron_deposit_net_yield(IDLE_DAYS_PER_YEAR, n_trips=1)
    print(f"  Idle days assumed:        {IDLE_DAYS_PER_YEAR} days/year")
    print(f"  Gross RON deposit yield:  {ron['gross_deposit_pct']:>+6.3f}%   "
          f"(6% × {IDLE_DAYS_PER_YEAR}/365)")
    print(f"  RO interest tax (10%):   {-ron['tax_pct']:>+6.3f}%")
    print(f"  After-tax yield:          {ron['after_tax_pct']:>+6.3f}%")
    print(f"  RON depreciation risk:   {-ron['depreciation_pct']:>+6.3f}%   "
          f"(~1%/yr × {IDLE_DAYS_PER_YEAR}/365)")
    print(f"  FX round-trip cost:      {-ron['fx_cost_pct']:>+6.3f}%   "
          f"(EUR→RON 0.8% + RON→EUR 0.8% + IB EUR/USD 0.4%)")
    print(f"  ─────────────────────────────────────────")
    print(f"  NET gain on account:      {ron['net_pct']:>+6.3f}%/yr")

    # ── 2. Compare idle capital strategies ────────────────────────────────
    print(f"\n{SEP}")
    print("  2. IDLE CAPITAL OPTIONS — what else can you do with the money?")
    print(SEP)
    print(f"  {'Strategy':<35} {'Yield on idle':>14}  {'Annual % of acct':>17}  Notes")
    print(f"  {'-'*35} {'-'*14}  {'-'*17}  {'-'*20}")

    options = [
        ("IB cash (small acct <€10k)",      0.0,           "IB pays 0% on small EUR balances"),
        ("IB cash (large acct >€10k)",       ib_mm_net_yield(IDLE_DAYS_PER_YEAR, 50_000),  "~4.5% APY, no friction"),
        ("IB T-Bills ETF (BIL/SHV)",         ib_tbill_net_yield(IDLE_DAYS_PER_YEAR), "~5.3% APY, 0.4% FX cost"),
        ("RON bank deposit (1 trip/yr)",      ron["net_pct"],  "6% gross → ~" + f"{ron['net_pct']:.2f}% net"),
        ("RON deposit (2 trips/yr)",          ron_deposit_net_yield(IDLE_DAYS_PER_YEAR//2, n_trips=2)["net_pct"], "Split idle period"),
    ]
    for label, yield_pct, note in options:
        bar = ("+" if yield_pct >= 0 else "-") + "█" * min(int(abs(yield_pct) / 0.2), 20)
        print(f"  {label:<35} {yield_pct:>+13.2f}%  {bar:<18}  {note}")

    # ── 3. Break-even idle days ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  3. BREAK-EVEN — how many idle days needed for RON to make sense?")
    print(SEP)

    print(f"  vs IB cash (0%): RON break-even at ...")
    for days in range(30, 365, 10):
        r = ron_deposit_net_yield(days, 1)
        if r["net_pct"] > 0:
            print(f"    {days} idle days → RON net {r['net_pct']:>+.3f}%  ◄ break-even vs IB cash")
            breakeven_vs_cash = days
            break

    print(f"\n  vs IB T-Bills (5.3%): RON break-even at ...")
    for days in range(30, 730, 5):
        r = ron_deposit_net_yield(days, 1)
        tb = ib_tbill_net_yield(days)
        if r["net_pct"] > tb:
            print(f"    {days} idle days → RON {r['net_pct']:>+.3f}% vs T-Bills {tb:>+.3f}%  ◄ break-even")
            breakeven_vs_tbill = days
            break
    else:
        print(f"    ✗ RON NEVER beats IB T-Bills — T-Bill yield is too close to RON net yield")
        breakeven_vs_tbill = None

    # Show the curve
    print(f"\n  Idle days  RON net    IB T-Bills  IB MM(>10k)  vs cash  vs T-Bills")
    print(f"  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*11}  {'-'*7}  {'-'*10}")
    for days in [30, 60, 90, 120, 150, 200, 250, 300, 365]:
        r   = ron_deposit_net_yield(days, 1)
        tb  = ib_tbill_net_yield(days)
        mm  = ib_mm_net_yield(days, 50_000)
        vs_cash  = "✓" if r["net_pct"] > 0  else "✗"
        vs_tb    = "✓" if r["net_pct"] > tb else "✗"
        print(f"  {days:>9}  {r['net_pct']:>+8.2f}%  {tb:>+9.2f}%  {mm:>+10.2f}%  "
              f"  {vs_cash:>5}    {vs_tb:>8}")

    # ── 4. Annual $ contribution by account size ──────────────────────────
    print(f"\n{SEP}")
    print("  4. RON DEPOSIT — absolute EUR gain by account size")
    print(SEP)
    print(f"  {'Account':>10}  {'RON net gain':>13}  {'IB T-Bill gain':>14}  {'Difference':>12}  Worth it?")
    print(f"  {'-'*10}  {'-'*13}  {'-'*14}  {'-'*12}  {'-'*10}")
    for acct in [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]:
        ron_gain = acct * ron["net_pct"] / 100
        tb_gain  = acct * ib_tbill_net_yield(IDLE_DAYS_PER_YEAR) / 100
        diff     = ron_gain - tb_gain
        worth    = "Yes" if diff > 200 else ("Marginal" if diff > 50 else "No")
        print(f"  {acct:>10,.0f}  €{ron_gain:>11,.0f}  €{tb_gain:>12,.0f}  "
              f"€{diff:>+10,.0f}  {worth}")

    # ── 5. True cost of IB account (annual) ──────────────────────────────
    print(f"\n{SEP}")
    print("  5. IB RUNNING COSTS (S2 + MG, ~13 trades/year)")
    print(SEP)
    print(f"  Trade commissions: {S2MG_TRADES_PER_YEAR} × €{IB_COMMISSION_PER_TRADE:.0f} = "
          f"€{S2MG_TRADES_PER_YEAR * IB_COMMISSION_PER_TRADE:.0f}/yr")
    print(f"  Inactivity fee:    {IB_INACTIVITY_MONTHS} quiet months × €{IB_INACTIVITY_FEE:.0f} = "
          f"€{IB_INACTIVITY_MONTHS * IB_INACTIVITY_FEE:.0f}/yr")
    print(f"  FX conversions:    ~{S2MG_TRADES_PER_YEAR * 2} × 0.2% (per trade round-trip, at account level)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total fixed IB costs: ~€{ib_cost:.0f}/year")
    print(f"  FX costs depend on account currency vs stock currency (USD-denominated stocks)")

    # ── 6. Minimum account — net return targets ────────────────────────────
    print(f"\n{SEP}")
    print("  6. MINIMUM ACCOUNT — to clear target EUR profit per year")
    print(SEP)
    print(f"  Strategy: S2+MG → +{S2MG_ANNUAL_RETURN*100:.1f}%/yr gross")
    print(f"  After 10% RO tax: +{S2MG_ANNUAL_RETURN*(1-RO_CAPITAL_GAINS_TAX)*100:.1f}%/yr")
    print(f"  IB fixed costs: ~€{ib_cost:.0f}/yr")
    print()

    net_after_tax_pct = S2MG_ANNUAL_RETURN * (1 - RO_CAPITAL_GAINS_TAX) * 100

    print(f"  {'Target profit':>16}  {'Min account':>14}  Comment")
    print(f"  {'-'*16}  {'-'*14}  {'-'*35}")
    targets = [
        (0,      "Break-even (cover IB costs only)"),
        (100,    "€100/yr — token return"),
        (500,    "€500/yr — €40/month"),
        (1_000,  "€1,000/yr — €83/month"),
        (3_000,  "€3,000/yr — €250/month"),
        (6_000,  "€6,000/yr — min wage Romania"),
        (12_000, "€12,000/yr — avg wage Romania"),
        (24_000, "€24,000/yr — above avg wage"),
    ]
    for target, comment in targets:
        acct = min_account_for_target(target, net_after_tax_pct, ib_cost)
        print(f"  €{target:>14,.0f}  €{acct:>13,.0f}  {comment}")

    # ── 7. Combined S2+MG + RON deposit — full model ──────────────────────
    print(f"\n{SEP}")
    print("  7. COMBINED MODEL — S2+MG + RON deposit vs S2+MG + T-Bills")
    print(SEP)
    print(f"  Account S2+MG return is ALREADY +{S2MG_ANNUAL_RETURN*100:.1f}%/yr on full account.")
    print(f"  The idle capital bonus is ADDITIVE to this.")
    print()
    print(f"  {'Account':>10}  {'S2+MG net':>10}  {'+ RON dep':>10}  {'+ T-Bills':>10}  "
          f"{'S2+MG €':>9}  {'+ RON €':>8}  {'+ T-Bill €':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  "
          f"{'-'*9}  {'-'*8}  {'-'*10}")
    for acct in [2_000, 5_000, 10_000, 20_000, 50_000, 100_000]:
        s2mg_net   = S2MG_ANNUAL_RETURN * (1 - RO_CAPITAL_GAINS_TAX) * 100
        ron_addon  = ron["net_pct"]
        tb_addon   = ib_tbill_net_yield(IDLE_DAYS_PER_YEAR)
        total_ron  = s2mg_net + ron_addon
        total_tb   = s2mg_net + tb_addon
        eur_s2mg   = acct * s2mg_net / 100 - ib_cost
        eur_ron    = eur_s2mg + acct * ron_addon / 100
        eur_tb     = eur_s2mg + acct * tb_addon / 100
        print(f"  {acct:>10,.0f}  {s2mg_net:>+9.1f}%  {total_ron:>+9.1f}%  {total_tb:>+9.1f}%  "
              f"€{eur_s2mg:>8,.0f}  €{eur_ron:>7,.0f}  €{eur_tb:>9,.0f}")

    # ── 8. Verdict ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VERDICT")
    print(SEP)
    print(f"""
  RON deposit strategy (6%/yr) — is it worth it?

  ✗ NEVER beats IB T-Bills (BIL/SHV ETF, ~5.3%/yr):
      RON net after tax + FX + depreciation = ~{ron['net_pct']:.2f}%/{IDLE_DAYS_PER_YEAR}d
      IB T-Bills for same period             = ~{ib_tbill_net_yield(IDLE_DAYS_PER_YEAR):.2f}%/{IDLE_DAYS_PER_YEAR}d
      T-Bills win by ~{ib_tbill_net_yield(IDLE_DAYS_PER_YEAR) - ron['net_pct']:.2f}%/yr with ZERO friction.

  ✓ BEATS doing nothing (IB cash, 0%) only if:
      Idle period > ~{next(d for d in range(30,365,5) if ron_deposit_net_yield(d,1)['net_pct'] > 0)} consecutive days
      AND only 1 round trip per year (2+ trips never break even)
      AND RON doesn't depreciate more than {RON_DEPRECIATION_RISK*100:.0f}%/yr

  Best idle capital strategy by account size:
      < €10,000  : IB T-Bills ETF (BIL/SHV) — 5.3%/yr, zero friction, instant liquidity
      > €10,000  : IB money market (4.5%/yr) OR T-Bills, whichever is higher
      Any size   : RON deposits NOT recommended — FX friction kills the edge

  Minimum account to make S2+MG worthwhile:
      Break-even (cover IB costs):  €{min_account_for_target(0, net_after_tax_pct, ib_cost):>8,.0f}
      €500/year profit:             €{min_account_for_target(500, net_after_tax_pct, ib_cost):>8,.0f}
      €1,000/year profit:           €{min_account_for_target(1_000, net_after_tax_pct, ib_cost):>8,.0f}
      Romanian min wage equivalent: €{min_account_for_target(6_000, net_after_tax_pct, ib_cost):>8,.0f}
    """)

    print(f"  Note: +{S2MG_ANNUAL_RETURN*100:.1f}%/yr is the backtest average (2015-2026).")
    print(f"  Real forward returns may be lower. Never bet more than you can afford to lose.")


if __name__ == "__main__":
    main()

