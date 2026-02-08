"""Feature engineering utilities for FX time series."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature engineering.

    Attributes:
        momentum_windows: Lookback windows for momentum features.
        vol_windows: Lookback windows for rolling volatility.
        zscore_windows: Lookback windows for z-scored momentum.
        epsilon: Numerical stability term for standard deviation.
        decision_shift: Shift applied to decision-grade features.
    """

    momentum_windows: list[int] = field(default_factory=lambda: [5, 20, 60])
    vol_windows: list[int] = field(default_factory=lambda: [20])
    zscore_windows: list[int] = field(default_factory=lambda: [20])
    epsilon: float = 1e-12
    decision_shift: int = 1


def compute_log_returns(price: pd.Series) -> pd.Series:
    """Compute log returns from a price series."""

    if price.empty:
        raise ValueError("Price series is empty.")
    returns = np.log(price).diff()
    return returns.rename("ret_1")


def rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    """Compute rolling volatility (standard deviation) of returns."""

    if window < 1:
        raise ValueError("window must be >= 1 for rolling volatility.")
    return returns.rolling(window=window, min_periods=window).std(ddof=0)


def momentum(returns: pd.Series, window: int) -> pd.Series:
    """Compute momentum as sum of past returns."""

    if window < 1:
        raise ValueError("window must be >= 1 for momentum.")
    return returns.rolling(window=window, min_periods=window).sum()


def zscore(series: pd.Series, window: int, *, epsilon: float = 1e-12) -> pd.Series:
    """Compute rolling z-score with numerical stability."""

    if window < 1:
        raise ValueError("window must be >= 1 for z-score.")
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - rolling_mean) / (rolling_std + epsilon)


def build_features(price: pd.Series, *, cfg: FeatureConfig) -> pd.DataFrame:
    """Build a feature matrix with no-lookahead shift applied.

    Decision-grade features are shifted by `cfg.decision_shift` so that
    values available at time t use information through t-1.
    """

    if not isinstance(price.index, pd.DatetimeIndex):
        raise ValueError("Price series must have a DatetimeIndex.")
    if price.index.tz is None:
        raise ValueError("Price series index must be timezone-aware.")
    if not price.index.is_monotonic_increasing or not price.index.is_unique:
        raise ValueError("Price series index must be strictly increasing without duplicates.")

    features = pd.DataFrame(index=price.index)

    ret_1 = compute_log_returns(price)
    features["ret_1"] = ret_1

    for window in cfg.momentum_windows:
        features[f"mom_{window}"] = momentum(ret_1, window=window)

    for window in cfg.vol_windows:
        features[f"rv_{window}"] = rolling_vol(ret_1, window=window)

    for window in cfg.zscore_windows:
        base = momentum(ret_1, window=window)
        features[f"z_mom_{window}"] = zscore(base, window=window, epsilon=cfg.epsilon)

    if cfg.decision_shift < 0:
        raise ValueError("decision_shift must be non-negative.")

    if cfg.decision_shift > 0:
        features = features.shift(cfg.decision_shift)

    features = features.dropna()
    features.attrs["decision_shift"] = cfg.decision_shift
    return features


def validate_no_lookahead(features: pd.DataFrame) -> None:
    """Validate basic sanity checks for no-lookahead features."""

    if features.empty:
        raise ValueError("Features DataFrame is empty.")

    if not isinstance(features.index, pd.DatetimeIndex):
        raise ValueError("Features index must be a DatetimeIndex.")

    if features.index.tz is None:
        raise ValueError("Features index must be timezone-aware.")

    if not features.index.is_monotonic_increasing or not features.index.is_unique:
        raise ValueError("Features index must be strictly increasing without duplicates.")

    if features.isna().any().any():
        raise ValueError("Features contain NaN values; ensure sufficient history and shift.")

    decision_shift = features.attrs.get("decision_shift", None)
    if decision_shift is None or decision_shift < 1:
        raise ValueError(
            "Features are missing a decision_shift >= 1; ensure no-lookahead shift applied."
        )
