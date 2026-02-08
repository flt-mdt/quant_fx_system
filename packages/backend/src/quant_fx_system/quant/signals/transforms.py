"""Reusable transforms for signals."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def clip(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Clip a series between bounds."""

    return series.clip(lower=lo, upper=hi)


def winsorize_by_quantile(series: pd.Series, q: float = 0.01) -> pd.Series:
    """Winsorize series by quantile limits."""

    lower = series.quantile(q)
    upper = series.quantile(1.0 - q)
    return series.clip(lower=lower, upper=upper)


def zscore_rolling(series: pd.Series, window: int, epsilon: float) -> pd.Series:
    """Compute rolling z-score for a series."""

    rolling = series.rolling(window=window, min_periods=window)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    return (series - mean) / (std + epsilon)


def scale_to_target_std(
    series: pd.Series,
    target: float = 1.0,
    window: int = 60,
    epsilon: float = 1e-12,
) -> pd.Series:
    """Scale a series to target rolling standard deviation."""

    rolling = series.rolling(window=window, min_periods=window)
    std = rolling.std(ddof=0)
    scale = target / (std + epsilon)
    return series * scale


def to_position_from_score(
    score: pd.Series,
    *,
    method: Literal["tanh", "clip"],
    max_leverage: float,
    k: float = 1.0,
) -> pd.Series:
    """Convert a score into a bounded position series."""

    if method == "tanh":
        return max_leverage * np.tanh(k * score)
    if method == "clip":
        max_abs = score.abs().max()
        if pd.isna(max_abs) or max_abs == 0:
            normalized = score * 0.0
        else:
            normalized = score / max_abs
        return normalized.clip(lower=-max_leverage, upper=max_leverage)
    raise ValueError(f"Unsupported method: {method}")
