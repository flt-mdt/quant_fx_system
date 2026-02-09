from __future__ import annotations

import math
import statistics

import numpy as np
import pandas as pd


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_ppf(value: float) -> float:
    return statistics.NormalDist().inv_cdf(value)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    sr_threshold: float = 0.0,
) -> float:
    if n_obs < 2:
        return float("nan")
    numerator = (sharpe - sr_threshold) * math.sqrt(n_obs - 1)
    denominator = math.sqrt(1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe)
    if denominator == 0.0 or math.isnan(denominator):
        return float("nan")
    return _normal_cdf(numerator / denominator)


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
) -> float:
    if n_trials <= 1:
        return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurtosis, sr_threshold=0.0)
    if n_obs < 2:
        return float("nan")
    sigma_sr = math.sqrt(
        max(1e-12, (1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe) / (n_obs - 1))
    )
    z = _normal_ppf(1.0 - 1.0 / n_trials)
    sr_threshold = z * sigma_sr
    return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurtosis, sr_threshold=sr_threshold)


def hac_adjusted_sharpe(returns: pd.Series, sharpe: float) -> float:
    if returns.size < 2:
        return float("nan")
    rho1 = float(returns.autocorr(lag=1))
    if math.isnan(rho1):
        return sharpe
    if rho1 <= 0:
        return sharpe
    n_eff = returns.size * (1.0 - rho1) / (1.0 + rho1)
    if n_eff <= 0:
        return float("nan")
    return sharpe * math.sqrt(n_eff / returns.size)


def stability_from_rolling(sr_series: pd.Series) -> dict[str, float]:
    if sr_series.empty:
        return {}
    clean = sr_series.dropna()
    if clean.empty:
        return {}
    return {
        "stability_sr_pos_ratio": float((clean > 0).mean()),
        "sr_roll_p10": float(clean.quantile(0.1)),
        "sr_roll_p50": float(clean.quantile(0.5)),
        "sr_roll_p90": float(clean.quantile(0.9)),
    }
