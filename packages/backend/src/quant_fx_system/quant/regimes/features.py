from __future__ import annotations

import numpy as np
import pandas as pd

from .types import FeatureConfig
from .validation import infer_periods


def _rolling_slope_r2(log_price: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = np.sum(x_centered**2)

    def apply_slope(values: np.ndarray) -> float:
        y = values
        y_mean = y.mean()
        cov = np.sum(x_centered * (y - y_mean))
        return cov / denom if denom != 0 else np.nan

    def apply_r2(values: np.ndarray) -> float:
        y = values
        y_mean = y.mean()
        slope = np.sum(x_centered * (y - y_mean)) / denom if denom != 0 else np.nan
        intercept = y_mean - slope * x_mean
        y_hat = slope * x + intercept
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - ss_res / ss_tot

    slope = log_price.rolling(window, min_periods=window).apply(apply_slope, raw=True)
    r2 = log_price.rolling(window, min_periods=window).apply(apply_r2, raw=True)
    return slope, r2


def compute_features(
    price: pd.Series | None,
    returns: pd.Series | None,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    if returns is None:
        if price is None:
            raise ValueError("Either price or returns must be provided.")
        returns = price.pct_change()
    if price is None:
        price = (1 + returns.fillna(0.0)).cumprod()

    data: dict[str, pd.Series] = {}
    data["ret_1"] = returns
    data["ret_abs"] = returns.abs()
    data["ret_sq"] = returns**2

    periods_per_year = infer_periods(cfg, returns.index)
    for window in cfg.windows:
        vol = returns.rolling(window, min_periods=window).std(ddof=0)
        if periods_per_year is not None:
            vol = vol * np.sqrt(periods_per_year)
        data[f"vol_roll_{window}"] = vol

        mom = price / price.shift(window) - 1
        data[f"mom_{window}"] = mom

        log_price = np.log(price)
        slope, r2 = _rolling_slope_r2(log_price, window)
        data[f"slope_{window}"] = slope
        data[f"r2_{window}"] = r2

        roll_mean = price.rolling(window, min_periods=window).mean()
        roll_std = price.rolling(window, min_periods=window).std(ddof=0)
        roll_std = roll_std.replace(0.0, np.nan)
        zscore = (price - roll_mean) / roll_std
        data[f"zscore_price_{window}"] = zscore

        vol_of_vol = vol.rolling(window, min_periods=window).std(ddof=0)
        data[f"vol_of_vol_{window}"] = vol_of_vol

    features = pd.DataFrame(data, index=returns.index)
    if cfg.feature_shift > 0:
        features = features.shift(cfg.feature_shift)
    return features


__all__ = ["compute_features"]
