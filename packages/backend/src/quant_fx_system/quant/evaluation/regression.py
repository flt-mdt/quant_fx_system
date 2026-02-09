from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_benchmark_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: int | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if returns.empty or benchmark_returns.empty:
        return metrics

    active = returns - benchmark_returns
    tracking_error = float(active.std(ddof=0))
    if periods_per_year:
        tracking_error *= math.sqrt(periods_per_year)
    metrics["tracking_error"] = tracking_error

    mean_active = float(active.mean())
    if active.std(ddof=0) == 0.0:
        information_ratio = 0.0
    else:
        information_ratio = mean_active / float(active.std(ddof=0))
        if periods_per_year:
            information_ratio *= math.sqrt(periods_per_year)
    metrics["information_ratio"] = information_ratio

    var_b = float(np.var(benchmark_returns, ddof=0))
    cov_rb = float(np.cov(returns, benchmark_returns, ddof=0)[0, 1])
    beta = cov_rb / var_b if var_b != 0.0 else float("nan")
    alpha = float(returns.mean()) - beta * float(benchmark_returns.mean())
    if periods_per_year:
        alpha *= periods_per_year
    metrics["beta"] = beta
    metrics["alpha"] = alpha

    corr = float(returns.corr(benchmark_returns))
    metrics["r2"] = corr * corr if not math.isnan(corr) else float("nan")
    return metrics
