from __future__ import annotations

import numpy as np
import pandas as pd


def fit_binning(series: pd.Series, n_bins: int = 10) -> np.ndarray:
    cleaned = series.dropna()
    if cleaned.empty:
        raise ValueError("Cannot fit binning on empty series.")
    _, edges = pd.qcut(cleaned, q=n_bins, retbins=True, duplicates="drop")
    edges = np.unique(edges)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_binning(series: pd.Series, edges: np.ndarray) -> pd.Series:
    return pd.cut(series, bins=edges, include_lowest=True)


def _hist_probs(series: pd.Series, edges: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    binned = apply_binning(series, edges)
    counts = binned.value_counts(sort=False, dropna=True)
    probs = counts / counts.sum()
    cats = binned.cat.categories
    probs = probs.reindex(cats, fill_value=0.0).to_numpy()
    probs = np.clip(probs, epsilon, 1.0)
    probs = probs / probs.sum()
    return probs


def psi(train: pd.Series, live: pd.Series, n_bins: int = 10, epsilon: float = 1e-12) -> float:
    edges = fit_binning(train, n_bins=n_bins)
    p = _hist_probs(train, edges, epsilon=epsilon)
    q = _hist_probs(live, edges, epsilon=epsilon)
    return float(np.sum((p - q) * np.log(p / q)))


def kl_divergence(
    train: pd.Series, live: pd.Series, n_bins: int = 10, epsilon: float = 1e-12
) -> float:
    edges = fit_binning(train, n_bins=n_bins)
    p = _hist_probs(train, edges, epsilon=epsilon)
    q = _hist_probs(live, edges, epsilon=epsilon)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(
    train: pd.Series, live: pd.Series, n_bins: int = 10, epsilon: float = 1e-12
) -> float:
    edges = fit_binning(train, n_bins=n_bins)
    p = _hist_probs(train, edges, epsilon=epsilon)
    q = _hist_probs(live, edges, epsilon=epsilon)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))
