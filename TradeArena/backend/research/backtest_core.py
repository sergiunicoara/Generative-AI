"""Small, dependency-light backtest engine for TradeArena research.

Input data is intentionally explicit: a DataFrame with date, symbol, close,
and earnings_date columns. No market data is downloaded implicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class Position:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    stock_weight: float = 1.0
    option_weight: float = 0.0
    option_return: float = 0.0

    @property
    def return_pct(self) -> float:
        stock_return = self.exit_price / self.entry_price - 1.0
        return self.stock_weight * stock_return + self.option_weight * self.option_return


def normalize_events(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "close", "earnings_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["earnings_date"] = pd.to_datetime(out["earnings_date"]).dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="raise")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_signals(
    prices: pd.DataFrame,
    *,
    entry_days: int = 20,
    exit_days: int = 1,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build one signal per symbol/earnings event using trading-day offsets."""
    data = normalize_events(prices)
    if symbols is not None:
        allowed = set(symbols)
        data = data[data["symbol"].isin(allowed)]
    rows: list[dict] = []
    for (symbol, earnings), group in data.groupby(["symbol", "earnings_date"], sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        before = group[group["date"] < earnings]
        if len(before) <= entry_days:
            continue
        entry = before.iloc[-entry_days]
        exit_row = before.iloc[-exit_days]
        if entry["date"] >= exit_row["date"]:
            continue
        rows.append({
            "symbol": symbol,
            "earnings_date": earnings,
            "entry_date": entry["date"],
            "exit_date": exit_row["date"],
            "entry_price": float(entry["close"]),
            "exit_price": float(exit_row["close"]),
            "stock_return": float(exit_row["close"] / entry["close"] - 1.0),
        })
    return pd.DataFrame(rows)


def apply_regime_filter(signals: pd.DataFrame, market: pd.DataFrame, *, window: int = 150) -> pd.DataFrame:
    """Keep entries where the market benchmark is above its moving average."""
    if signals.empty:
        return signals.copy()
    required = {"date", "close"}
    if not required.issubset(market.columns):
        raise ValueError("market must contain date and close columns")
    benchmark = market.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    benchmark = benchmark.sort_values("date")
    benchmark["moving_average"] = benchmark["close"].rolling(window).mean()
    joined = pd.merge_asof(
        signals.sort_values("entry_date"),
        benchmark[["date", "close", "moving_average"]].rename(columns={"close": "market_close"}),
        left_on="entry_date", right_on="date", direction="backward",
    )
    return joined[joined["market_close"] > joined["moving_average"]].drop(columns="date")


def select_positions(signals: pd.DataFrame, *, ranking: dict[str, float] | None = None,
                     score_column: str | None = None, minimum_score: float | None = None) -> pd.DataFrame:
    """Select at most one overlapping position, using fixed or dynamic ranking."""
    if signals.empty:
        return signals.copy()
    work = signals.copy()
    if score_column and minimum_score is not None:
        work = work[work[score_column] >= minimum_score]
    if ranking is not None:
        work["rank"] = work["symbol"].map(ranking).fillna(float("-inf"))
    elif "rank" not in work:
        work["rank"] = 0.0
    chosen: list[pd.Series] = []
    occupied_until = pd.Timestamp.min
    for _, row in work.sort_values(["entry_date", "rank"], ascending=[True, False]).iterrows():
        if row["entry_date"] >= occupied_until:
            chosen.append(row)
            occupied_until = row["exit_date"]
    return pd.DataFrame(chosen).reset_index(drop=True)


def compound(signals: pd.DataFrame, *, option_weight: float = 0.0,
             option_return_column: str = "option_return") -> pd.DataFrame:
    """Compound a one-position-at-a-time strategy and return equity by year."""
    if signals.empty:
        return pd.DataFrame(columns=["year", "return", "equity"])
    equity = 1.0
    rows = []
    for _, signal in signals.sort_values("exit_date").iterrows():
        stock_weight = 1.0 - option_weight
        option_return = float(signal.get(option_return_column, 0.0))
        total_return = stock_weight * float(signal["stock_return"]) + option_weight * option_return
        equity *= 1.0 + total_return
        rows.append({"date": signal["exit_date"], "return": total_return, "equity": equity})
    curve = pd.DataFrame(rows)
    curve["year"] = pd.to_datetime(curve["date"]).dt.year
    yearly = curve.groupby("year", as_index=False).agg(equity=("equity", "last"))
    yearly["return"] = yearly["equity"].pct_change().fillna(yearly["equity"].iloc[0] - 1.0)
    return yearly[["year", "return", "equity"]]
