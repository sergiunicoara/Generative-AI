"""Command-line entry point for reproducible TradeArena research runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest_core import apply_regime_filter, build_signals, compound, select_positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline TradeArena pre-earnings backtest")
    parser.add_argument("prices", type=Path, help="CSV containing date,symbol,close,earnings_date")
    parser.add_argument("--market", type=Path, help="Optional benchmark CSV containing date,close")
    parser.add_argument("--dma", type=int, default=150)
    parser.add_argument("--option-weight", type=float, default=0.0)
    args = parser.parse_args()

    signals = build_signals(pd.read_csv(args.prices), symbols=["GOOGL", "NVDA", "AMZN"])
    if args.market:
        signals = apply_regime_filter(signals, pd.read_csv(args.market), window=args.dma)
    signals = select_positions(signals, ranking={"GOOGL": 3, "NVDA": 2, "AMZN": 1})
    result = compound(signals, option_weight=args.option_weight)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
