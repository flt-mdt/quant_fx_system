from __future__ import annotations

import numpy as np
import pandas as pd


def compute_turnover(position: pd.Series) -> pd.Series:
    turnover = position.diff().abs().fillna(0.0)
    turnover.name = "turnover"
    return turnover


def cap_position_delta(position: pd.Series, max_turnover_per_bar: float) -> pd.Series:
    if max_turnover_per_bar < 0:
        raise ValueError("max_turnover_per_bar must be non-negative")

    values = position.to_numpy(copy=True)
    if len(values) == 0:
        return position

    capped = np.empty_like(values)
    capped[0] = values[0]
    for i in range(1, len(values)):
        delta = values[i] - capped[i - 1]
        if abs(delta) > max_turnover_per_bar:
            capped[i] = capped[i - 1] + np.sign(delta) * max_turnover_per_bar
        else:
            capped[i] = values[i]
    capped_series = pd.Series(capped, index=position.index, name=position.name)
    return capped_series


def ewma_smooth_position(position: pd.Series, alpha: float) -> pd.Series:
    smoothed = position.ewm(alpha=alpha, adjust=False).mean()
    smoothed.name = position.name
    return smoothed
