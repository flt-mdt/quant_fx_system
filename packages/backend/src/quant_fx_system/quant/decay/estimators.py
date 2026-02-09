from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from quant_fx_system.quant.decay.validation import validate_utc_series


def estimate_ar1_phi(series: pd.Series, *, demean: bool = True) -> float:
    series = validate_utc_series(series, "series", allow_nans=True)
    values = series.to_numpy(dtype=float)
    x_prev = values[:-1]
    x_next = values[1:]
    mask = np.isfinite(x_prev) & np.isfinite(x_next)
    if not mask.any():
        return float("nan")
    x_prev = x_prev[mask]
    x_next = x_next[mask]
    if demean:
        x_prev = x_prev - np.mean(x_prev)
        x_next = x_next - np.mean(x_next)
    denom = np.dot(x_prev, x_prev)
    if denom == 0:
        return float("nan")
    return float(np.dot(x_next, x_prev) / denom)


def estimate_half_life_ar1(series: pd.Series) -> float:
    phi = estimate_ar1_phi(series)
    if not (0 < phi < 1):
        return float("nan")
    return -math.log(2.0) / math.log(phi)


def ic_decay_curve(
    signal: pd.Series, future_returns: pd.Series, horizons: Iterable[int]
) -> pd.Series:
    signal = validate_utc_series(signal, "signal", allow_nans=True)
    future_returns = validate_utc_series(future_returns, "future_returns", allow_nans=True)

    common_index = signal.index.intersection(future_returns.index)
    signal = signal.loc[common_index]
    future_returns = future_returns.loc[common_index]

    results: dict[int, float] = {}
    for horizon in horizons:
        horizon_int = int(horizon)
        if horizon_int < 1:
            raise ValueError("horizon must be >= 1")
        shifted = future_returns.shift(-horizon_int)
        results[horizon_int] = signal.corr(shifted)
    return pd.Series(results, name="ic_decay")
