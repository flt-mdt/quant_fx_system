from __future__ import annotations

import numpy as np
import pandas as pd

from .types import RegimeConfig


def label_from_quantiles(
    vol: pd.Series,
    quantiles: tuple[float, ...],
) -> tuple[pd.Series, pd.DataFrame]:
    thresholds = [vol.quantile(q) for q in quantiles]
    bins = [-np.inf, *thresholds, np.inf]
    labels = range(len(bins) - 1)
    regime = pd.cut(vol, bins=bins, labels=labels, include_lowest=True)
    regime = regime.astype("float").astype("Int64")
    proba = pd.get_dummies(regime, prefix="p_state")
    for state in range(len(labels)):
        col = f"p_state_{state}"
        if col not in proba.columns:
            proba[col] = 0.0
    proba = proba.sort_index(axis=1).astype(float)
    proba = proba.reindex(vol.index)
    return regime, proba


def label_trend_range(
    features: pd.DataFrame,
    cfg: RegimeConfig,
) -> tuple[pd.Series, pd.DataFrame | None]:
    slope_cols = [c for c in features.columns if c.startswith("slope_")]
    r2_cols = [c for c in features.columns if c.startswith("r2_")]
    if not slope_cols or not r2_cols:
        raise ValueError("Trend/range labeling requires slope and r2 features.")
    slope = features[slope_cols[0]]
    r2 = features[r2_cols[0]]
    slope_threshold = slope.abs().quantile(cfg.trend_slope_quantile)
    is_trend = (slope.abs() >= slope_threshold) & (r2 >= cfg.trend_r2_threshold)
    regime = is_trend.astype(int)
    regime = regime.where(~features[slope_cols[0]].isna(), pd.NA)
    proba = pd.DataFrame(
        {
            "p_state_0": (~is_trend).astype(float),
            "p_state_1": is_trend.astype(float),
        },
        index=features.index,
    )
    proba = proba.where(~features[slope_cols[0]].isna())
    return regime, proba


__all__ = ["label_from_quantiles", "label_trend_range"]
