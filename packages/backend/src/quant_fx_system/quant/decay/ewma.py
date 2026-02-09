from __future__ import annotations

import math

import numpy as np
import pandas as pd


def half_life_to_alpha(half_life_bars: float) -> float:
    if half_life_bars <= 0:
        raise ValueError("half_life_bars must be > 0")
    return 1.0 - math.exp(-math.log(2.0) / half_life_bars)


def ewma_time_aware(
    series: pd.Series,
    half_life_time: pd.Timedelta,
    *,
    min_periods: int = 1,
    fillna_value: float | None = None,
    carry_forward_nan: bool = True,
) -> pd.Series:
    if half_life_time <= pd.Timedelta(0):
        raise ValueError("half_life_time must be > 0")

    values = series.to_numpy(dtype=float)
    if fillna_value is not None:
        values = np.where(np.isnan(values), fillna_value, values)

    index = series.index
    output = np.empty_like(values)

    hl_seconds = half_life_time.total_seconds()
    prev_value = np.nan
    valid_count = 0

    for i in range(len(values)):
        x_i = values[i]
        if i == 0:
            if np.isnan(x_i) and carry_forward_nan:
                prev_value = np.nan
            else:
                prev_value = x_i
        else:
            dt = (index[i] - index[i - 1]).total_seconds()
            if dt < 0:
                raise ValueError("index must be monotonic increasing")
            lambda_i = math.exp(-math.log(2.0) * dt / hl_seconds)
            if np.isnan(x_i):
                prev_value = prev_value if carry_forward_nan else np.nan
            else:
                if np.isnan(prev_value):
                    prev_value = x_i
                else:
                    prev_value = (1.0 - lambda_i) * x_i + lambda_i * prev_value

        if not np.isnan(prev_value):
            valid_count += 1

        if valid_count < min_periods:
            output[i] = np.nan
        else:
            output[i] = prev_value

    return pd.Series(output, index=index, name=series.name)
