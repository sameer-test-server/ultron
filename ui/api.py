"""Simulation/projection helpers for Ultron UI API routes."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.analyst import StockAnalysis
from core.paper_trader import PaperTradeResult


def parse_float(raw_value: str | None, default: float, min_value: float, max_value: float) -> float:
    """Parse bounded float from request args."""
    try:
        value = float(raw_value) if raw_value is not None else default
    except (TypeError, ValueError):
        value = default
    if pd.isna(value):
        value = default
    return max(min_value, min(max_value, value))


def parse_int(raw_value: str | None, default: int, min_value: int, max_value: int) -> int:
    """Parse bounded int from request args."""
    try:
        value = int(raw_value) if raw_value is not None else default
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def serialize_simulation(result: PaperTradeResult) -> dict[str, Any]:
    """Convert simulation dataclass to API payload."""
    return {
        "initial_capital": round(result.initial_capital, 2),
        "final_capital": round(result.final_capital, 2),
        "total_return_pct": round(result.total_return_pct, 2),
        "win_rate": round(result.win_rate, 2),
        "total_fees": round(result.total_fees, 2),
        "max_drawdown_pct": round(result.max_drawdown_pct, 2),
        "profit_factor": round(result.profit_factor, 2),
        "avg_trade_return_pct": round(result.avg_trade_return_pct, 2),
        "best_trade_pct": round(result.best_trade_pct, 2),
        "worst_trade_pct": round(result.worst_trade_pct, 2),
        "avg_holding_days": round(result.avg_holding_days, 1),
        "trades_count": len(result.trades),
        "trades": [
            {
                "entry_date": trade.entry_date,
                "entry_price": round(trade.entry_price, 2),
                "exit_date": trade.exit_date,
                "exit_price": round(trade.exit_price, 2),
                "return_pct": round(trade.return_pct, 2),
                "profit_loss": round(trade.profit_loss, 2),
                "fees_paid": round(trade.fees_paid, 2),
                "holding_days": trade.holding_days,
                "exit_reason": trade.exit_reason,
                "outcome": "WIN" if trade.profit_loss >= 0 else "LOSS",
            }
            for trade in result.trades
        ],
    }


def projection_payload(analysis: StockAnalysis, amount: float, horizon_days: int) -> dict[str, Any]:
    """
    Build forward estimate from historical daily returns.
    Uses adaptive drift (short + long window blend) and backtests this estimator.
    """
    data = analysis.data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    if len(data) < max(45, horizon_days + 5):
        raise ValueError("Insufficient data for projection.")

    close = data["Close"].reset_index(drop=True)
    dates = data["Date"].reset_index(drop=True)
    returns = close.pct_change().dropna()
    if len(returns) < 20:
        raise ValueError("Insufficient returns history for projection.")

    short_window = min(20, len(returns))
    long_window = min(120, len(returns))
    mu_short = float(returns.tail(short_window).mean())
    mu_long = float(returns.tail(long_window).mean())
    drift = (0.65 * mu_short) + (0.35 * mu_long)
    sigma = float(returns.tail(long_window).std(ddof=0)) if long_window > 1 else 0.0

    expected_return = (1.0 + drift) ** horizon_days - 1.0
    uncertainty = sigma * (horizon_days ** 0.5)
    lower_return = max(expected_return - uncertainty, -0.99)
    upper_return = max(expected_return + uncertainty, -0.99)

    projected_value = amount * (1.0 + expected_return)
    low_value = amount * (1.0 + lower_return)
    high_value = amount * (1.0 + upper_return)

    lookback = min(60, max(20, len(close) // 6))
    errors: list[float] = []
    signed_errors: list[float] = []
    direction_hits = 0

    for idx in range(lookback, len(close) - horizon_days):
        history_slice = close.iloc[idx - lookback : idx]
        history_returns = history_slice.pct_change().dropna()
        if len(history_returns) < 12:
            continue

        rolling_short = min(20, len(history_returns))
        rolling_long = min(60, len(history_returns))
        rolling_mu = (0.65 * float(history_returns.tail(rolling_short).mean())) + (
            0.35 * float(history_returns.tail(rolling_long).mean())
        )
        predicted_return = (1.0 + rolling_mu) ** horizon_days - 1.0
        actual_return = (float(close.iloc[idx + horizon_days]) / float(close.iloc[idx])) - 1.0

        errors.append(abs(predicted_return - actual_return))
        signed_errors.append(predicted_return - actual_return)
        if predicted_return == 0 and actual_return == 0:
            direction_hits += 1
        elif predicted_return * actual_return > 0:
            direction_hits += 1

    validation_count = len(errors)
    direction_hit_rate = (direction_hits / validation_count * 100.0) if validation_count else 0.0
    avg_abs_error_pct = (sum(errors) / validation_count * 100.0) if validation_count else 0.0
    mean_bias_pct = (sum(signed_errors) / validation_count * 100.0) if validation_count else 0.0

    last_close = float(close.iloc[-1])
    last_close_date = dates.iloc[-1].strftime("%Y-%m-%d")

    latest_completed_return_pct = None
    latest_completed_start_date = None
    if len(close) > horizon_days:
        actual_recent = (float(close.iloc[-1]) / float(close.iloc[-1 - horizon_days])) - 1.0
        latest_completed_return_pct = actual_recent * 100.0
        latest_completed_start_date = dates.iloc[-1 - horizon_days].strftime("%Y-%m-%d")

    return {
        "horizon_days": horizon_days,
        "expected_return_pct": round(expected_return * 100.0, 2),
        "projected_value": round(projected_value, 2),
        "projected_gain": round(projected_value - amount, 2),
        "range_low_value": round(low_value, 2),
        "range_high_value": round(high_value, 2),
        "reference_close": round(last_close, 2),
        "reference_date": last_close_date,
        "validation_samples": validation_count,
        "direction_hit_rate": round(direction_hit_rate, 2),
        "avg_abs_error_pct": round(avg_abs_error_pct, 2),
        "mean_bias_pct": round(mean_bias_pct, 2),
        "latest_completed_start_date": latest_completed_start_date,
        "latest_completed_return_pct": round(latest_completed_return_pct, 2)
        if latest_completed_return_pct is not None
        else None,
    }
