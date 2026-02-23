"""Paper trading simulator using analyst-generated hypothetical signals."""

from __future__ import annotations

from dataclasses import dataclass

from core.analyst import SignalPoint


@dataclass(frozen=True)
class PaperTrade:
    """One simulated position lifecycle."""

    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    profit_loss: float
    return_pct: float
    shares: float = 0.0
    fees_paid: float = 0.0
    holding_days: int = 0
    exit_reason: str = "Signal exit"


@dataclass(frozen=True)
class PaperTradeResult:
    """Aggregate paper trading performance metrics."""

    initial_capital: float
    final_capital: float
    trades: list[PaperTrade]
    total_return_pct: float
    win_rate: float
    total_fees: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    avg_holding_days: float = 0.0


def _drawdown_from_equity(equity_points: list[float]) -> float:
    """Calculate maximum drawdown percentage from equity curve."""
    if not equity_points:
        return 0.0

    peak = equity_points[0]
    max_drawdown = 0.0
    for equity in equity_points:
        peak = max(peak, equity)
        if peak <= 0:
            continue
        drawdown = (peak - equity) / peak
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown * 100.0


def simulate_paper_trades(
    signals: list[SignalPoint],
    initial_capital: float = 2000.0,
    position_size_pct: float = 1.0,
    brokerage_pct: float = 0.0008,
    slippage_pct: float = 0.0005,
    max_holding_days: int = 60,
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.18,
) -> PaperTradeResult:
    """Simulate a long-only strategy from hypothetical entry/exit markers with execution costs."""
    capital = float(initial_capital)
    initial_capital_value = float(initial_capital)
    in_position = False
    shares = 0.0
    entry: SignalPoint | None = None
    entry_fill_price = 0.0
    invested_amount = 0.0
    entry_fees = 0.0
    total_fees_paid = 0.0
    trades: list[PaperTrade] = []
    equity_points = [capital]

    if position_size_pct <= 0:
        position_size_pct = 1.0
    position_size_pct = max(0.05, min(position_size_pct, 1.0))

    for signal in signals:
        if signal.kind == "ENTRY" and not in_position:
            if signal.price <= 0:
                continue

            entry_fill_price = signal.price * (1.0 + slippage_pct)
            allocated_capital = capital * position_size_pct
            if allocated_capital <= 0:
                continue

            entry_fees = allocated_capital * brokerage_pct
            if allocated_capital + entry_fees > capital:
                allocated_capital = capital / (1.0 + brokerage_pct)
                entry_fees = allocated_capital * brokerage_pct
            if allocated_capital <= 0:
                continue

            shares = allocated_capital / entry_fill_price
            invested_amount = allocated_capital
            capital -= (allocated_capital + entry_fees)
            total_fees_paid += entry_fees
            entry = signal
            in_position = True
            continue

        if signal.kind == "EXIT" and in_position and entry is not None:
            exit_fill_price = signal.price * (1.0 - slippage_pct)
            gross_exit_value = shares * exit_fill_price
            exit_fees = gross_exit_value * brokerage_pct
            net_exit_value = gross_exit_value - exit_fees

            total_fees_paid += exit_fees
            capital += net_exit_value

            total_cost_basis = invested_amount + entry_fees
            pnl = net_exit_value - total_cost_basis
            ret = (pnl / total_cost_basis * 100.0) if total_cost_basis > 0 else 0.0

            holding_days = max((signal.date - entry.date).days, 0)

            exit_reason = signal.reason or "Signal exit"
            raw_return = ((exit_fill_price / entry_fill_price) - 1.0) if entry_fill_price > 0 else 0.0
            if raw_return <= -abs(stop_loss_pct):
                exit_reason = f"Stop-loss hit ({abs(stop_loss_pct) * 100:.1f}%)"
            elif raw_return >= abs(take_profit_pct):
                exit_reason = f"Take-profit hit ({abs(take_profit_pct) * 100:.1f}%)"
            elif holding_days >= max_holding_days:
                exit_reason = f"Max holding period reached ({max_holding_days} days)"

            trades.append(
                PaperTrade(
                    entry_date=entry.date.strftime("%Y-%m-%d"),
                    entry_price=entry_fill_price,
                    exit_date=signal.date.strftime("%Y-%m-%d"),
                    exit_price=exit_fill_price,
                    profit_loss=pnl,
                    return_pct=ret,
                    shares=shares,
                    fees_paid=(entry_fees + exit_fees),
                    holding_days=holding_days,
                    exit_reason=exit_reason,
                )
            )

            equity_points.append(capital)
            shares = 0.0
            entry = None
            in_position = False
            entry_fill_price = 0.0
            invested_amount = 0.0
            entry_fees = 0.0

    wins = sum(1 for trade in trades if trade.profit_loss > 0)
    gross_wins = sum(trade.profit_loss for trade in trades if trade.profit_loss > 0)
    gross_losses = sum(abs(trade.profit_loss) for trade in trades if trade.profit_loss < 0)
    win_rate = (wins / len(trades) * 100) if trades else 0.0
    total_return = ((capital / initial_capital_value) - 1.0) * 100 if initial_capital_value else 0.0
    avg_trade_return = (sum(trade.return_pct for trade in trades) / len(trades)) if trades else 0.0
    best_trade = max((trade.return_pct for trade in trades), default=0.0)
    worst_trade = min((trade.return_pct for trade in trades), default=0.0)
    avg_holding = (sum(trade.holding_days for trade in trades) / len(trades)) if trades else 0.0
    max_drawdown = _drawdown_from_equity(equity_points)
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0.0)
    if profit_factor == float("inf"):
        profit_factor = 999.0

    return PaperTradeResult(
        initial_capital=initial_capital_value,
        final_capital=capital,
        trades=trades,
        total_return_pct=total_return,
        win_rate=win_rate,
        total_fees=total_fees_paid,
        max_drawdown_pct=max_drawdown,
        profit_factor=profit_factor,
        avg_trade_return_pct=avg_trade_return,
        best_trade_pct=best_trade,
        worst_trade_pct=worst_trade,
        avg_holding_days=avg_holding,
    )
