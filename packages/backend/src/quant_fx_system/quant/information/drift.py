from __future__ import annotations

import numpy as np
import pandas as pd

from .entropy import js_divergence as _js_from_probs
from .types import InformationConfig

_DEF_EPS = 1e-12


def _bin_series(series: pd.Series, bins: int, binning: str) -> np.ndarray:
    if binning == "quantile":
        binned = pd.qcut(series, q=bins, labels=False, duplicates="drop")
    elif binning == "equal_width":
        binned = pd.cut(series, bins=bins, labels=False)
    else:
        raise ValueError("binning must be 'quantile' or 'equal_width'")
    return binned.to_numpy(dtype=float)


def _hist_probs(series: pd.Series, bins: int, binning: str) -> np.ndarray:
    binned = _bin_series(series.dropna(), bins, binning)
    hist = np.histogram(binned[~np.isnan(binned)], bins=bins)[0].astype(float) + _DEF_EPS
    return hist / hist.sum()


def psi(train: pd.Series, live: pd.Series, bins: int = 10, binning: str = "quantile") -> float:
    p = _hist_probs(train, bins, binning)
    q = _hist_probs(live, bins, binning)
    return float(np.sum((p - q) * np.log((p + _DEF_EPS) / (q + _DEF_EPS))))


def kl_divergence(train: pd.Series, live: pd.Series, bins: int = 10, binning: str = "quantile") -> float:
    p = _hist_probs(train, bins, binning)
    q = _hist_probs(live, bins, binning)
    return float(np.sum(p * np.log((p + _DEF_EPS) / (q + _DEF_EPS))))


def js_divergence(train: pd.Series, live: pd.Series, bins: int = 10, binning: str = "quantile") -> float:
    p = _hist_probs(train, bins, binning)
    q = _hist_probs(live, bins, binning)
    return float(_js_from_probs(p, q))


def hellinger(train: pd.Series, live: pd.Series, bins: int = 10, binning: str = "quantile") -> float:
    p = _hist_probs(train, bins, binning)
    q = _hist_probs(live, bins, binning)
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
