from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import validate_utc_series


def cusum_flags(series: pd.Series, threshold: float, drift: float) -> pd.Series:
    if threshold <= 0:
        raise ValueError("CUSUM threshold must be > 0.")
    if drift < 0:
        raise ValueError("CUSUM drift must be >= 0.")
    series = validate_utc_series(series, "series", allow_nans=True)
    values = series.to_numpy()
    pos_sum = 0.0
    neg_sum = 0.0
    flags = np.zeros(len(values), dtype=int)
    for idx, value in enumerate(values):
        if np.isnan(value):
            flags[idx] = 0
            continue
        pos_sum = max(0.0, pos_sum + value - drift)
        neg_sum = min(0.0, neg_sum + value + drift)
        if pos_sum > threshold or neg_sum < -threshold:
            flags[idx] = 1
            pos_sum = 0.0
            neg_sum = 0.0
    return pd.Series(flags, index=series.index, name="cusum_flag")


__all__ = ["cusum_flags"]
