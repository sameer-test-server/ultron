"""Multi-strategy scenario engine for Ultron."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ScenarioResult:
    """Summary of a strategy backtest."""

    name: str
    total_return_pct: float
    trades_count: int
    win_rate: float
    max_drawdown_pct: float


def _drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak <= 0:
            continue
        dd = (peak - value) / peak
        max_dd = max(max_dd, dd)
    return max_dd * 100.0


def _run_strategy(close: pd.Series, entry: pd.Series, exit: pd.Series) -> ScenarioResult:
    capital = 1.0
    in_pos = False
    entry_price = 0.0
    trades = []
    equity = [capital]

    for idx in range(1, len(close)):
        price = float(close.iloc[idx])
        if not in_pos and bool(entry.iloc[idx]):
            in_pos = True
            entry_price = price
            continue
        if in_pos and bool(exit.iloc[idx]):
            if entry_price > 0:
                ret = (price / entry_price) - 1.0
                capital *= (1.0 + ret)
                trades.append(ret)
                equity.append(capital)
            in_pos = False
            entry_price = 0.0

    if in_pos and entry_price > 0:
        price = float(close.iloc[-1])
        ret = (price / entry_price) - 1.0
        capital *= (1.0 + ret)
        trades.append(ret)
        equity.append(capital)

    wins = sum(1 for r in trades if r > 0)
    win_rate = (wins / len(trades) * 100.0) if trades else 0.0
    total_return = (capital - 1.0) * 100.0
    return ScenarioResult(
        name="",
        total_return_pct=round(total_return, 2),
        trades_count=len(trades),
        win_rate=round(win_rate, 1),
        max_drawdown_pct=round(_drawdown(equity), 2),
    )


def run_scenarios(data: pd.DataFrame) -> list[ScenarioResult]:
    """Run multiple strategies on a dataset and return summaries."""
    if data.empty:
        return []

    df = data.copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    if len(df) < 50:
        return []

    close = df["Close"].reset_index(drop=True)
    sma50 = pd.to_numeric(df.get("SMA_50"), errors="coerce").fillna(method="ffill")
    rsi = pd.to_numeric(df.get("RSI_14"), errors="coerce").fillna(50)

    # Trend-following: close above/below SMA50
    entry_tf = close > sma50
    exit_tf = close < sma50
    tf = _run_strategy(close, entry_tf, exit_tf)
    tf = ScenarioResult("Trend Following", tf.total_return_pct, tf.trades_count, tf.win_rate, tf.max_drawdown_pct)

    # Mean-reversion: RSI oversold to neutral
    entry_mr = rsi < 30
    exit_mr = rsi > 50
    mr = _run_strategy(close, entry_mr, exit_mr)
    mr = ScenarioResult("Mean Reversion", mr.total_return_pct, mr.trades_count, mr.win_rate, mr.max_drawdown_pct)

    # Volatility breakout: close above 20-day high, exit below 20-day mid
    rolling_high = close.rolling(20, min_periods=20).max()
    rolling_mid = close.rolling(20, min_periods=20).mean()
    entry_vb = close > rolling_high
    exit_vb = close < rolling_mid
    vb = _run_strategy(close, entry_vb, exit_vb)
    vb = ScenarioResult("Volatility Breakout", vb.total_return_pct, vb.trades_count, vb.win_rate, vb.max_drawdown_pct)

    return [tf, mr, vb]
