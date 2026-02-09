from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def validate_utc_series(series: pd.Series, name: str, allow_nans: bool = False) -> None:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{name} index must be a pandas DatetimeIndex")
    if series.index.tz is None or str(series.index.tz) != "UTC":
        raise ValueError(f"{name} index must be UTC timezone-aware")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonic increasing")
    if not series.index.is_unique:
        raise ValueError(f"{name} index must be unique")

    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy()
    if not allow_nans and np.isnan(values).any():
        raise ValueError(f"{name} contains NaN values")
    if np.isinf(values).any():
        raise ValueError(f"{name} contains non-finite values")


def validate_position_bounds(position: pd.Series, max_leverage: float) -> None:
    if max_leverage <= 0:
        raise ValueError("max_leverage must be positive")
    if (position.abs() > max_leverage).any():
        raise ValueError("position exceeds max_leverage bounds")


def align_series(*series: pd.Series | None) -> tuple[pd.Series | None, ...]:
    valid_series = [s for s in series if s is not None]
    if not valid_series:
        return tuple(series)

    common_index = valid_series[0].index
    for s in valid_series[1:]:
        common_index = common_index.intersection(s.index)

    aligned: list[pd.Series | None] = []
    for s in series:
        if s is None:
            aligned.append(None)
        else:
            aligned.append(s.loc[common_index])
    return tuple(aligned)


def validate_positive(series: pd.Series, name: str) -> None:
    if (series <= 0).any():
        raise ValueError(f"{name} must be strictly positive")


def validate_window(window: int, name: str) -> None:
    if window < 2:
        raise ValueError(f"{name} must be >= 2")


def validate_alpha(alpha: float, name: str) -> None:
    if alpha <= 0 or alpha > 1:
        raise ValueError(f"{name} must be in (0, 1]")
