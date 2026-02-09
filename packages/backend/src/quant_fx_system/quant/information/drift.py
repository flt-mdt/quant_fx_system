from __future__ import annotations

import numpy as np
import pandas as pd

from .entropy import js_divergence as _js_from_probs
from .types import InformationConfig

_DEF_EPS = 1e-12


def _bin_edges(series: pd.Series, bins: int, binning: str) -> np.ndarray:
    clean = series.dropna()
    if clean.empty:
        return np.array([0.0, 1.0])
    if binning == "quantile":
        _, edges = pd.qcut(clean, q=bins, retbins=True, duplicates="drop")
    elif binning == "equal_width":
        _, edges = pd.cut(clean, bins=bins, retbins=True)
    else:
        raise ValueError("binning must be 'quantile' or 'equal_width'")
    edges = np.unique(edges)
    if edges.size < 2:
        val = float(clean.iloc[0])
        edges = np.array([val - _DEF_EPS, val + _DEF_EPS])
    return edges


def _hist_probs(series: pd.Series, edges: np.ndarray) -> np.ndarray:
    clean = series.dropna()
    if clean.empty:
        hist = np.zeros(len(edges) - 1, dtype=float)
    else:
        bounded_edges = edges.copy()
        bounded_edges[0] = -np.inf
        bounded_edges[-1] = np.inf
        hist = np.histogram(clean.to_numpy(), bins=bounded_edges)[0].astype(float)
    hist = hist + _DEF_EPS
    return hist / hist.sum()


def psi(train: pd.Series, live: pd.Series, bins: int = 10, binning: str = "quantile") -> float:
    edges = _bin_edges(train, bins, binning)
    p = _hist_probs(train, edges)
    q = _hist_probs(live, edges)
    return float(np.sum((p - q) * np.log((p + _DEF_EPS) / (q + _DEF_EPS))))


def kl_divergence(train: pd.Series, live: pd.Series, bins: int = 10, binning: str = "quantile") -> float:
    edges = _bin_edges(train, bins, binning)
    p = _hist_probs(train, edges)
    q = _hist_probs(live, edges)
    return float(np.sum(p * np.log((p + _DEF_EPS) / (q + _DEF_EPS))))


def js_divergence(train: pd.Series, live: pd.Series, bins: int = 10, binning: str = "quantile") -> float:
    edges = _bin_edges(train, bins, binning)
    p = _hist_probs(train, edges)
    q = _hist_probs(live, edges)
    return float(_js_from_probs(p, q))


def hellinger(train: pd.Series, live: pd.Series, bins: int = 10, binning: str = "quantile") -> float:
    edges = _bin_edges(train, bins, binning)
    p = _hist_probs(train, edges)
    q = _hist_probs(live, edges)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def drift_report(
    train_df: pd.DataFrame,
    live_df: pd.DataFrame,
    cfg: InformationConfig,
) -> pd.DataFrame:
    rows = []
    common = [col for col in train_df.columns if col in live_df.columns]
    for col in common:
        rows.append(
            {
                "feature": col,
                "psi": psi(train_df[col], live_df[col], cfg.drift_bins, cfg.drift_binning),
                "kl": kl_divergence(
                    train_df[col], live_df[col], cfg.drift_bins, cfg.drift_binning
                ),
                "js": js_divergence(
                    train_df[col], live_df[col], cfg.drift_bins, cfg.drift_binning
                ),
                "hellinger": hellinger(
                    train_df[col], live_df[col], cfg.drift_bins, cfg.drift_binning
                ),
            }
        )
    return pd.DataFrame(rows).set_index("feature")
