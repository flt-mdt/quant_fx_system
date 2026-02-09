from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from quant_fx_system.quant.decay.validation import validate_utc_series


def estimate_ar1_phi(series: pd.Series) -> float:
    validate_utc_series(series, "series", allow_nans=True)
    values = series.to_numpy(dtype=float)
    x_prev = values[:-1]
    x_next = values[1:]
    mask = np.isfinite(x_prev) & np.isfinite(x_next)
    if not mask.any():
        return float("nan")
    x_prev = x_prev[mask]
    x_next = x_next[mask]
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
    validate_utc_series(signal, "signal", allow_nans=True)
    validate_utc_series(future_returns, "future_returns", allow_nans=True)

    common_index = signal.index.intersection(future_returns.index)
    signal = signal.loc[common_index]
    future_returns = future_returns.loc[common_index]

    results: dict[int, float] = {}
    for horizon in horizons:
        shifted = future_returns.shift(-int(horizon))
        results[int(horizon)] = signal.corr(shifted)
    return pd.Series(results, name="ic_decay")
