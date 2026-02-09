from __future__ import annotations

import numpy as np
import pandas as pd


def _bin_series(series: pd.Series, n_bins: int = 10) -> pd.Categorical:
    return pd.qcut(series, q=n_bins, duplicates="drop")


def joint_entropy(x: pd.Series, y: pd.Series, n_bins: int = 10) -> float:
    aligned = pd.concat([x, y], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    bx = _bin_series(aligned.iloc[:, 0], n_bins=n_bins)
    by = _bin_series(aligned.iloc[:, 1], n_bins=n_bins)
    joint = pd.crosstab(bx, by, normalize=True)
    probs = joint.to_numpy().ravel()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))
