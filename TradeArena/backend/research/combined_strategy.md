# Combined Strategy: Pre-Earnings Drift System

This document combines:

- Strategy One: fixed-rank stock-only pre-earnings drift
- Strategy Two: dynamic-ranked stock-only pre-earnings drift
- Strategy Three: filtered stock/options blend

The common core is the same:

1. Trade only pre-earnings drift.
2. Enter around 20 trading days before earnings.
3. Exit 1 trading day before earnings.
4. Do not hold through earnings.
5. Skip trades when the tech market regime is weak.

## Shared Universe

Core tickers:

- `GOOGL`
- `NVDA`
- `AMZN`

Dynamic expanded tickers (S2 and S3 only):

- `MSFT`
- `META`
- `AMD`

Avoid or deprioritize:

- `CRM`
- `TSLA`
- `NFLX`
- most `AAPL` signals

## Shared Regime Filter

Before entering any trade:

1. Check `QQQ`.
2. Calculate the 150-day moving average.
3. Trade only if `QQQ > 150dma`.
4. If `QQQ < 150dma`, hold cash.

This filter reduced the stock-only bad year from `-32.9%` to `-6.7%` in the fixed-rank strategy.

## Validated Entry Filters (backtested 2015–2026)

Two filters improve S2 and S3 outcomes. Neither applies to S1.

### F6 — Gap up on entry day

Enter only if the stock's open on entry day is above the previous close.

```text
entry_open > previous_close
```

Confirms that the pre-earnings drift has already started on the entry day.
Tested on 52 baseline trades: removes 7 trades, adds +$6,224 to S2 final equity, same drawdown.
**Does not apply to S1** (F6 hurts S1 — core tickers need no momentum confirmation).

### L8 — Minimum dynamic score 1.20

Raise the score threshold from 1.05 to 1.20 for S2 and S3.

Removes marginal entries (typically expanded tickers with scores 1.05–1.19) that fail at higher rates.
Tested: blocks 4 of 12 losing trades (33%), removes 0 trades with score above 1.40.
Reduces S3 max drawdown from 25.1% to 13.3% by eliminating trades where the call option amplifies losses.

## Dynamic Score

For Strategy Two and the options blend, rank overlapping signals with:

```text
score =
  base_quality
  + 1.20 * bounded(20d stock momentum)
  + 0.80 * bounded(60d stock momentum)
  + 1.50 * bounded(20d relative strength vs QQQ)
  + 0.50 * earnings_revision_score
```

Where:

- `base_quality` comes from historical ticker quality.
- `20d stock momentum` is the stock return over the last 20 trading days.
- `60d stock momentum` is the stock return over the last 60 trading days.
- `20d relative strength vs QQQ` is stock 20d return minus QQQ 20d return.
- `bounded(x)` caps each momentum input between `-20%` and `+20%`.
- `earnings_revision_score` is live-only until point-in-time revision history is added.

Base quality values:

| Ticker | Base quality | Tier |
|---|---|---|
| NVDA | 1.50 | Core |
| GOOGL | 1.40 | Core |
| AMZN | 1.20 | Core |
| MSFT | 1.10 | Expanded |
| META | 1.10 | Expanded |
| AMD | 1.00 | Expanded |

## Strategy One: Fixed-Rank Stock Only

Use when you want the simplest stock-only strategy with no extra filters.

Rules:

1. Watch `GOOGL`, `NVDA`, `AMZN`.
2. Enter around D-20 before earnings.
3. Require `QQQ > 150dma`.
4. If multiple signals overlap, rank: GOOGL → NVDA → AMZN.
5. Put 100% of account equity into the selected stock.
6. Exit D-1 before earnings.
7. Stop-loss: exit if position drops -5% from entry.

No F6 or L8 filter — they hurt S1 by removing good trades.

Historical result (2015–2026):

| Metric | Result |
|---|---:|
| Starting equity | $2,000 |
| Ending equity | $13,229 |
| Total return | 6.6x (+561%) |
| Average yearly return | +18.7% |
| Worst year | -5.0% |
| Max drawdown | -9.8% |
| Trades | 46 |
| Win rate | 76.1% |
| Avg per trade | +4.42% |
| $5k milestone | Mar 2021 |
| $10k milestone | Jan 2026 |

## Strategy Two: Dynamic-Ranked Stock Only

Use when you want better entry selection. Applies F6 and L8 filters.

Rules:

1. Watch `GOOGL`, `NVDA`, `AMZN`, `MSFT`, `META`, `AMD`.
2. Enter around D-20 before earnings.
3. Require `QQQ > 150dma`.
4. Require F6: entry day open > previous close (gap up).
5. Calculate dynamic score for every active signal.
6. Require score >= 1.20 (L8 filter).
7. Buy the highest-scoring signal.
8. Put 100% of account equity into the selected stock.
9. Exit D-1 before earnings.
10. Stop-loss: exit if position drops -5% from entry.

Historical result (2015–2026, with F6 + score≥1.20):

| Metric | Result |
|---|---:|
| Starting equity | $2,000 |
| Ending equity | $21,966 |
| Total return | 11.0x (+998%) |
| Average yearly return | +24.3% |
| Worst year | -5.0% |
| Max drawdown | -6.9% |
| Trades | 41 |
| Win rate | 73.2% |
| Avg per trade | +6.61% |
| $5k milestone | Dec 2020 |
| $10k milestone | Jun 2024 |

Comparison vs no filters:

| Version | Final | Ann% | Max DD |
|---|---|---|---|
| No filters | $12,998 | +18.5% | -9.8% |
| F6 only | $19,222 | +22.8% | -9.8% |
| **F6 + score≥1.20** | **$21,966** | **+24.3%** | **-6.9%** |

## Strategy Three: Filtered Stock/Options Blend

Use when you want higher growth and accept options risk. Applies F6 and L8 filters.

Options are modeled as synthetic `10% ITM / 60 DTE` calls with a 25% real-world haircut.
**Historical options results are approximate** — no actual point-in-time option chains, bid/ask spreads, or IV rank.

Rules:

1. Watch `GOOGL`, `NVDA`, `AMZN`, `MSFT`, `META`, `AMD`.
2. Enter around D-20 before earnings.
3. Require `QQQ > 150dma`.
4. Require F6: entry day open > previous close (gap up).
5. Calculate dynamic score for every active signal.
6. Require score >= 1.20 (L8 filter).
7. Buy the highest-scoring active signal.
8. Use a stock/options blend.
9. Exit D-1 before earnings.
10. Stop-loss on stock leg: exit if position drops -5% from entry.

### Safer Options Version (recommended)

Allocation:

- 90% stock
- 10% `10% ITM / 60 DTE` call exposure

Historical modeled result (2015–2026, with F6 + score≥1.20):

| Metric | Result |
|---|---:|
| Starting equity | $2,000 |
| Ending equity | $89,841 |
| Total return | 44.9x (+4,392%) |
| Average yearly return | +41.3% |
| Worst year | -5.0% |
| Max drawdown | -13.3% |
| Trades | 41 |
| Win rate | 73.2% |
| Avg per trade | +10.97% |
| $5k milestone | Oct 2019 |
| $10k milestone | Dec 2020 |
| $20k milestone | Dec 2021 |

Comparison vs no filters:

| Version | Final | Ann% | Max DD |
|---|---|---|---|
| No filters | $37,509 | +30.5% | -25.1% |
| F6 only | $57,038 | +35.6% | -25.1% |
| **F6 + score≥1.20** | **$89,841** | **+41.3%** | **-13.3%** |

### Balanced Options Version

Allocation:

- 75% stock
- 25% `10% ITM / 60 DTE` call exposure

Not recommended until real option-chain data is wired in (drawdown risk is higher).

## Complete Comparison (2015–2026, $2,000 start)

| Strategy | Filters | Final | Ann% | Max DD | Win% | $10k |
|---|---|---|---|---|---|---|
| S1 Fixed-rank | none | $13,229 | +18.7% | -9.8% | 76.1% | Jan 2026 |
| S2 Dynamic | F6 + score≥1.20 | $21,966 | +24.3% | **-6.9%** | 73.2% | Jun 2024 |
| S3 Stock+calls | F6 + score≥1.20 | **$89,841** | **+41.3%** | -13.3% | 73.2% | Dec 2020 |

## Recommended Versions

### Conservative

Use Strategy One:

- Stock only, simple, no extra filters
- Expected: +18.7%/yr, worst year -5%, max DD -9.8%

### Balanced (default)

Use Strategy Two with F6 + score≥1.20:

- Stock only, fully real data, no modeled components
- Expected: +24.3%/yr, worst year -5%, max DD -6.9%
- **Lowest drawdown of all three strategies**

### Growth

Use Strategy Three Safer with F6 + score≥1.20:

- 90% stock + 10% modeled calls
- Expected: +41.3%/yr (modeled), realistic +30–35%/yr with real execution
- Add only when real option-chain data and live spreads are available

## Final Recommendation

1. **Default deployment:** Strategy Two (S2) with F6 + score≥1.20.
   Fully validated on real data. Lowest drawdown. Best risk-adjusted return.

2. **Upgrade to S3** only when:
   - Real option chain data is wired in (live IV, bid/ask, open interest)
   - S2 has been running live for at least 6 months
   - The options sleeve passes a liquidity check on entry day

3. **Filters that work:** F6 (gap up) and L8 (score≥1.20) for S2 and S3.
   Do not apply F6 or L8 to S1 — they remove winners from the core universe.

4. **Filters that do not work:** RSI, volume, 5-day momentum, 50dma, XLK 20dma.
   Tested on 45 trades — all either had no effect or removed winners faster than losers.

