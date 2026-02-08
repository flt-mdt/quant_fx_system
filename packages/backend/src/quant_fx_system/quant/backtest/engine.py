from __future__ import annotations

import pandas as pd

from .costs import compute_costs, compute_turnover
from .returns import compute_simple_returns
from .types import BacktestConfig, BacktestResult
from .validation import (
    align_price_position,
    validate_no_lookahead_alignment,
    validate_position_series,
    validate_price_series,
)


def _compute_equity(pnl: pd.Series, initial_equity: float) -> pd.Series:
    equity_values = []
    equity = initial_equity
    for value in pnl.fillna(0.0):
        if equity <= 0:
            equity = 0.0
        else:
            equity = equity * (1.0 + value)
            if equity < 0:
                equity = 0.0
        equity_values.append(equity)
    return pd.Series(equity_values, index=pnl.index)


def run_backtest(
    price: pd.Series,
    position: pd.Series,
    cfg: BacktestConfig | None = None,
) -> BacktestResult:
    cfg = cfg or BacktestConfig()

    if cfg.execution != "next_bar":
        raise ValueError("Only next_bar execution is supported")
    if cfg.return_type != "simple":
        raise ValueError("Only simple returns are supported")

    validate_price_series(price)
    validate_position_series(position, cfg.max_leverage)

    price_aligned, position_aligned = align_price_position(price, position)

    returns = compute_simple_returns(price_aligned)
    position_applied = position_aligned.shift(1).reindex(returns.index)
    validate_no_lookahead_alignment(position_aligned, position_applied, returns.index)
    position_applied = position_applied.fillna(0.0)

    turnover = compute_turnover(position_aligned).reindex(returns.index).fillna(0.0)
    costs = compute_costs(turnover, cfg.transaction_cost_bps, cfg.slippage_bps)

    pnl = position_applied * returns - costs
    equity = _compute_equity(pnl, cfg.initial_equity)

    metadata = {
        "execution": cfg.execution,
        "return_type": cfg.return_type,
        "transaction_cost_bps": cfg.transaction_cost_bps,
        "slippage_bps": cfg.slippage_bps,
        "max_leverage": cfg.max_leverage,
        "initial_equity": cfg.initial_equity,
        "start": returns.index.min(),
        "end": returns.index.max(),
    }

    return BacktestResult(
        returns=returns,
        position=position_applied,
        pnl=pnl,
        equity=equity,
        turnover=turnover,
        costs=costs,
        metadata=metadata,
    )
