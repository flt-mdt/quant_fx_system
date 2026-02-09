from __future__ import annotations

import numpy as np
import pandas as pd

_DEF_EPS = 1e-12


def _bin_series(series: pd.Series, bins: int, binning: str) -> np.ndarray:
    if binning == "quantile":
        binned = pd.qcut(series, q=bins, labels=False, duplicates="drop")
    elif binning == "equal_width":
        binned = pd.cut(series, bins=bins, labels=False)
    else:
        raise ValueError("binning must be 'quantile' or 'equal_width'")
    return binned.to_numpy(dtype=float)


def entropy(
    x: pd.Series,
    estimator: str = "hist",
    bins: int = 10,
    binning: str = "quantile",
) -> float:
    if estimator != "hist":
        raise ValueError("only hist estimator is supported")
    binned = _bin_series(x.dropna(), bins, binning)
    hist = np.histogram(binned[~np.isnan(binned)], bins=bins)[0].astype(float)
    hist = hist + _DEF_EPS
    p = hist / hist.sum()
    return float(-np.sum(p * np.log(p)))


def joint_entropy(
    x: pd.Series,
    y: pd.Series,
    bins: int = 10,
    binning: str = "quantile",
) -> float:
    aligned = pd.concat([x, y], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    bx = _bin_series(aligned.iloc[:, 0], bins, binning)
    by = _bin_series(aligned.iloc[:, 1], bins, binning)
    hist = np.histogram2d(bx, by, bins=bins)[0].astype(float) + _DEF_EPS
    p = hist / hist.sum()
    return float(-np.sum(p * np.log(p)))


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float) + _DEF_EPS
    q = np.asarray(q, dtype=float) + _DEF_EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(-np.sum(p * np.log(q)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float) + _DEF_EPS
    q = np.asarray(q, dtype=float) + _DEF_EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float) + _DEF_EPS
    q = np.asarray(q, dtype=float) + _DEF_EPS
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def surprisal(x: pd.Series, bins: int = 10, binning: str = "quantile") -> pd.Series:
    binned = _bin_series(x.dropna(), bins, binning)
    hist = np.histogram(binned[~np.isnan(binned)], bins=bins)[0].astype(float) + _DEF_EPS
    p = hist / hist.sum()
    indices = binned.astype(int)
    surprise = -np.log(p[indices])
    return pd.Series(surprise, index=x.dropna().index)
