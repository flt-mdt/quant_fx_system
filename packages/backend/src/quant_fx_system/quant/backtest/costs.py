from __future__ import annotations

import pandas as pd


def compute_turnover(position: pd.Series) -> pd.Series:
    return position.diff().abs()


def compute_costs(
    turnover: pd.Series,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> pd.Series:
    total_bps = transaction_cost_bps + slippage_bps
    if total_bps == 0:
        return turnover * 0.0
    return (total_bps / 1e4) * turnover
