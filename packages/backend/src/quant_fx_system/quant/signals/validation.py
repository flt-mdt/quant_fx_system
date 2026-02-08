"""Validation utilities for signals."""

from __future__ import annotations

import pandas as pd


def validate_series_index(series: pd.Series, *, name: str) -> None:
    """Validate a series index for signal outputs."""

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(f"{name} index must be a DatetimeIndex.")
    if series.index.tz is None:
        raise ValueError(f"{name} index must be timezone-aware (UTC).")
    if str(series.index.tz) != "UTC":
        raise ValueError(f"{name} index must be UTC.")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonic increasing.")
    if not series.index.is_unique:
        raise ValueError(f"{name} index must be unique.")


def validate_features_for_signals(features: pd.DataFrame, *, allow_nans: bool = False) -> None:
    """Validate input features for signals."""

    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features index must be a DatetimeIndex.")
    if features.index.tz is None:
        raise ValueError("features index must be timezone-aware (UTC).")
    if str(features.index.tz) != "UTC":
        raise ValueError("features index must be UTC.")
    if not features.index.is_monotonic_increasing:
        raise ValueError("features index must be monotonic increasing.")
    if not features.index.is_unique:
        raise ValueError("features index must be unique.")
    decision_shift = features.attrs.get("decision_shift")
    if decision_shift is None:
        raise ValueError("features.attrs['decision_shift'] is required for signals.")
    if decision_shift < 1:
        raise ValueError("features.attrs['decision_shift'] must be >= 1.")
    if not allow_nans and features.isna().any().any():
        raise ValueError("features must not contain NaNs when allow_nans=False.")
