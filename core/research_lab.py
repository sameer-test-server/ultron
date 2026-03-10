"""Research lab: parameter grid runner."""

from __future__ import annotations

from dataclasses import dataclass

from core.paper_trader import simulate_paper_trades
from core.analyst import StockAnalysis


@dataclass(frozen=True)
class GridResult:
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    total_return_pct: float
    win_rate: float


def run_parameter_grid(analysis: StockAnalysis) -> list[GridResult]:
    if analysis is None:
        return []

    grid = []
    position_sizes = [0.5, 0.75, 1.0]
    stop_losses = [0.05, 0.08, 0.12]
    take_profits = [0.12, 0.18, 0.24]

    for ps in position_sizes:
        for sl in stop_losses:
            for tp in take_profits:
                result = simulate_paper_trades(
                    analysis.signals,
                    initial_capital=2000.0,
                    position_size_pct=ps,
                    stop_loss_pct=sl,
                    take_profit_pct=tp,
                )
                grid.append(
                    GridResult(
                        position_size_pct=ps * 100.0,
                        stop_loss_pct=sl * 100.0,
                        take_profit_pct=tp * 100.0,
                        total_return_pct=result.total_return_pct,
                        win_rate=result.win_rate,
                    )
                )

    grid.sort(key=lambda g: g.total_return_pct, reverse=True)
    return grid[:5]
