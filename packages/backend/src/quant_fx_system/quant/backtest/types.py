from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 1.0
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    max_leverage: float = 1.0
    execution: Literal["next_bar"] = "next_bar"
    return_type: Literal["simple"] = "simple"


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series
    position: pd.Series
    pnl: pd.Series
    equity: pd.Series
    turnover: pd.Series
    costs: pd.Series
    metadata: dict
