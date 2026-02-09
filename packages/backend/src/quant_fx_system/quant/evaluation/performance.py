from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or math.isnan(denominator):
        return float("nan")
    return numerator / denominator


def _annualize_ratio(value: float, periods_per_year: int | None) -> float:
    if periods_per_year is None:
        return value
    return value * math.sqrt(periods_per_year)


def _annualize_return(total_return: float, n_obs: int, periods_per_year: int | None) -> float:
    if periods_per_year is None or n_obs == 0:
        return float("nan")
    return (1.0 + total_return) ** (periods_per_year / n_obs) - 1.0


def compute_performance_metrics(
    returns: pd.Series,
    *,
    equity: pd.Series | None,
    position: pd.Series | None,
    periods_per_year: int | None,
    risk_free_rate: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    n_obs = len(returns)
    if n_obs == 0:
        return metrics

    rf_per_period = risk_free_rate / periods_per_year if periods_per_year else 0.0
    excess = returns - rf_per_period

    total_return = (1.0 + returns).prod() - 1.0
    metrics["total_return"] = float(total_return)
    metrics["cagr"] = _annualize_return(total_return, n_obs, periods_per_year)

    mean_excess = float(excess.mean())
    vol = float(returns.std(ddof=0))
    metrics["vol_annual"] = _annualize_ratio(vol, periods_per_year)

    sharpe = _safe_divide(mean_excess, vol)
    if vol == 0.0:
        sharpe = 0.0
    metrics["sharpe"] = _annualize_ratio(sharpe, periods_per_year)

    downside = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if n_obs else float("nan")
    sortino = _safe_divide(mean_excess, downside_dev)
    if downside_dev == 0.0:
        sortino = 0.0
    metrics["sortino"] = _annualize_ratio(sortino, periods_per_year)

    metrics["hit_rate"] = float((returns > 0).mean())
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    metrics["avg_win"] = float(wins.mean()) if not wins.empty else float("nan")
    metrics["avg_loss"] = float(losses.mean()) if not losses.empty else float("nan")
    metrics["profit_factor"] = _safe_divide(float(wins.sum()), abs(float(losses.sum())))

    if equity is not None and len(equity) == n_obs:
        if equity.iloc[0] > 0:
            eq_total = equity.iloc[-1] / equity.iloc[0] - 1.0
            metrics["cagr"] = _annualize_return(eq_total, n_obs, periods_per_year)

    if position is not None and len(position) == n_obs:
        abs_pos = position.abs()
        metrics["exposure_mean"] = float(abs_pos.mean())
        metrics["exposure_pct_nonzero"] = float((abs_pos > 0).mean())
        metrics["exposure_max_abs"] = float(abs_pos.max())

    return metrics
