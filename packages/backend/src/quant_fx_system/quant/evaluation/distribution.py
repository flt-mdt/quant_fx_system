from __future__ import annotations

import numpy as np
import pandas as pd


def compute_distribution_metrics(returns: pd.Series) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    summary: dict[str, float] = {}
    tables: dict[str, pd.DataFrame] = {}

    summary["skew"] = float(returns.skew())
    excess_kurtosis = float(returns.kurtosis())
    summary["excess_kurtosis"] = excess_kurtosis
    summary["kurtosis"] = excess_kurtosis + 3.0

    var_1 = returns.quantile(0.01)
    summary["var_1pct"] = float(var_1)
    tail_losses = returns[returns <= var_1]
    summary["es_1pct"] = float(tail_losses.mean()) if not tail_losses.empty else float("nan")

    q95 = returns.quantile(0.95)
    q5 = returns.quantile(0.05)
    summary["tail_ratio"] = float(abs(q95) / abs(q5)) if q5 != 0 else float("nan")

    summary["best_return"] = float(returns.max())
    summary["worst_return"] = float(returns.min())

    percentiles = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]
    tables["percentiles"] = returns.quantile(percentiles).to_frame("return")
    return summary, tables
