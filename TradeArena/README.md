# TradeArena

Recovered from the original Claude session history for the missing
`Agentic-AI/TradeArena` working tree. The `backend/research` folder now
contains the source snapshots that Claude wrote during that session, including
the strategy documents and historical backtests.

The engine is offline by default and does not place orders or download market
data implicitly. Provide a CSV with:

```text
date,symbol,close,earnings_date
```

Run:

```powershell
python -m backend.research.tradearena_cli data/prices.csv --market data/qqq.csv --dma 150
```

The recovered strategy family is the pre-earnings drift setup: enter around
D-20, exit around D-1, choose one overlapping signal, and optionally require
the benchmark to be above its long moving average. Several research scripts
also model option sleeves using yfinance option-chain data or Black-Scholes
fallbacks. They are research tools, not live-trading software or investment
advice.

The session did not preserve every application dependency (notably parts of
`backend/environment` such as account and price stores), so the research
scripts are the fully recovered component. All recovered Python source files
compile successfully.
