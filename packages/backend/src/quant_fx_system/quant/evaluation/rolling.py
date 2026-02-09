from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_rolling_metrics(
    returns: pd.Series,
    *,
    windows: tuple[int, ...],
    periods_per_year: int | None,
    risk_free_rate: float,
) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(index=returns.index)

    rf_per_period = risk_free_rate / periods_per_year if periods_per_year else 0.0
    excess = returns - rf_per_period
    data: dict[str, pd.Series] = {}
    for window in windows:
        rolling_mean = excess.rolling(window=window).mean()
        rolling_vol = excess.rolling(window=window).std(ddof=0)
        rolling_vol = rolling_vol.replace(0.0, np.nan)
        rolling_sr = rolling_mean / rolling_vol
        if periods_per_year:
            rolling_sr = rolling_sr * math.sqrt(periods_per_year)

        data[f"mean_roll_{window}"] = rolling_mean
        data[f"vol_roll_{window}"] = rolling_vol
        data[f"sr_roll_{window}"] = rolling_sr

    return pd.DataFrame(data, index=returns.index)
