from __future__ import annotations

import numpy as np
import pandas as pd


def compute_equity(returns: pd.Series, equity: pd.Series | None = None) -> pd.Series:
    if equity is not None:
        return equity
    return (1.0 + returns).cumprod()


def compute_drawdowns(equity: pd.Series) -> tuple[pd.Series, float, int, float]:
    running_max = equity.cummax()
    underwater = 1.0 - equity / running_max
    max_drawdown = float(underwater.max()) if len(underwater) else float("nan")

    drawdown_active = underwater > 0
    max_tuw = 0
    current = 0
    for active in drawdown_active:
        if active:
            current += 1
            max_tuw = max(max_tuw, current)
        else:
            current = 0

    ulcer = float(np.sqrt(np.mean(np.square(underwater)))) if len(underwater) else float("nan")
    return underwater, max_drawdown, max_tuw, ulcer
