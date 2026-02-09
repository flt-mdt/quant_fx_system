from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd

from .periods import infer_periods_per_year


def validate_utc_series(
    series: pd.Series,
    name: str,
    *,
    allow_nans: bool = False,
) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise ValueError(f"{name} must be a pandas Series.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex.")
    if series.index.tz is None:
        raise ValueError(f"{name} index must be timezone-aware UTC.")
    if str(series.index.tz) != "UTC" and series.index.tz != timezone.utc:
        raise ValueError(f"{name} index must be UTC.")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonic increasing.")
    if not series.index.is_unique:
        raise ValueError(f"{name} index must be unique.")

    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy()
    if allow_nans:
        mask = ~np.isnan(values)
        if not np.isfinite(values[mask]).all():
            raise ValueError(f"{name} must contain finite values.")
    else:
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite values.")
    return numeric


def align_series(*series: pd.Series | None) -> tuple[pd.Series | None, ...]:
    non_null = [s for s in series if s is not None]
    if not non_null:
        return tuple(series)
    common_index = non_null[0].index
    for s in non_null[1:]:
        common_index = common_index.intersection(s.index)
    aligned = []
    for s in series:
        if s is None:
            aligned.append(None)
        else:
            aligned.append(s.reindex(common_index))
    return tuple(aligned)


__all__ = ["align_series", "infer_periods_per_year", "validate_utc_series"]
