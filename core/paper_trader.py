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


@dataclass(frozen=True)
class PaperTradeResult:
    """Aggregate paper trading performance metrics."""

    initial_capital: float
    final_capital: float
    trades: list[PaperTrade]
    total_return_pct: float
    win_rate: float


def simulate_paper_trades(signals: list[SignalPoint], initial_capital: float = 2000.0) -> PaperTradeResult:
    """Simulate a long-only strategy from hypothetical entry/exit markers."""
    capital = float(initial_capital)
    in_position = False
    shares = 0.0
    entry: SignalPoint | None = None
    trades: list[PaperTrade] = []

    for signal in signals:
        if signal.kind == "ENTRY" and not in_position:
            if signal.price <= 0:
                continue
            shares = capital / signal.price
            entry = signal
            in_position = True
            continue

        if signal.kind == "EXIT" and in_position and entry is not None:
            new_capital = shares * signal.price
            pnl = new_capital - capital
            ret = ((signal.price / entry.price) - 1.0) * 100

            trades.append(
                PaperTrade(
                    entry_date=entry.date.strftime("%Y-%m-%d"),
                    entry_price=entry.price,
                    exit_date=signal.date.strftime("%Y-%m-%d"),
                    exit_price=signal.price,
                    profit_loss=pnl,
                    return_pct=ret,
                )
            )

            capital = new_capital
            shares = 0.0
            entry = None
            in_position = False

    wins = sum(1 for trade in trades if trade.profit_loss > 0)
    win_rate = (wins / len(trades) * 100) if trades else 0.0
    total_return = ((capital / initial_capital) - 1.0) * 100 if initial_capital else 0.0

    return PaperTradeResult(
        initial_capital=initial_capital,
        final_capital=capital,
        trades=trades,
        total_return_pct=total_return,
        win_rate=win_rate,
    )
