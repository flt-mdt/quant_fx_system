"""Backtest orchestration service."""

from __future__ import annotations

import pandas as pd

from quant_fx_system.quant.backtest import BacktestConfig, run_backtest
from quant_fx_system.quant.evaluation import EvaluationConfig, evaluate_strategy


def run_accounting_and_evaluation(
    *,
    price: pd.Series,
    position_target: pd.Series,
    backtest_cfg: BacktestConfig | None = None,
    evaluation_cfg: EvaluationConfig | None = None,
) -> dict[str, object]:
    backtest_cfg = backtest_cfg or BacktestConfig(initial_equity=100_000.0)
    bt_result = run_backtest(price=price, position=position_target, cfg=backtest_cfg)

    evaluation_cfg = evaluation_cfg or EvaluationConfig()
    ev_result = evaluate_strategy(
        returns=bt_result.returns,
        equity=bt_result.equity,
        pnl=bt_result.pnl,
        position=bt_result.position,
        turnover=bt_result.turnover,
        costs=bt_result.costs,
        cfg=evaluation_cfg,
    )

    return {
        "backtest": bt_result,
        "evaluation": ev_result,
    }
